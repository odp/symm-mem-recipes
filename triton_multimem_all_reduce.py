import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from triton_barrier import blockwise_barrier
from triton_utils import get_flat_tid, sync_threads
from utils import log_triton_kernel


@triton.jit
def multimem_ld_reduce_128(multicast_ptrs, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[multicast_ptrs, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def multimem_all_reduce_kernel(
    multicast_ptr,
    signal_pad_ptrs,
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = get_flat_tid()

    # From this point on, we pretend each element is 128-bit
    numel = numel // NUMEL_PER_THREAD
    numel_per_rank = tl.cdiv(numel, WORLD_SIZE)
    block_start = pid * BLOCK_SIZE

    while block_start < numel_per_rank:
        offsets = block_start + tid
        mask = offsets < numel_per_rank

        # Each pointer points to a 128-bit bit pack
        ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + (RANK * numel_per_rank + offsets) * 2
        )
        (x, y, z, w) = multimem_ld_reduce_128(ptrs, mask=mask)
        multimem_st_128(ptrs, x, y, z, w, mask=mask)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="acq_rel")


def multimem_all_reduce(tensor: torch.Tensor):
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 4
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16

    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    numel_per_thread = BYTES_PER_THREAD // tensor.element_size()

    assert (
        tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    num_threads = triton.cdiv(
        tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )
    if num_threads < MAX_BLOCK_SIZE:
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    kernel = multimem_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.multicast_ptr,
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=tensor.numel(),
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    log_triton_kernel(kernel)
    return tensor


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 triton_multimem_all_reduce.py
    """
    from symm_mem_all_reduce import main

    main(["--impl", "triton_multimem_all_reduce"])
