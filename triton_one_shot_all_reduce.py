import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from triton_barrier import blockwise_barrier
from triton_utils import sync_threads
from utils import benchmark_with_profiler


@triton.jit
def load_128(addrs, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32             %p0, $3, 1;
            @%p0 ld.global.v2.u64   {$0, $1}, [$2];
        }
        """,
        "=l,=l,l,r",
        args=[addrs, mask.to(tl.int32)],
        dtype=(tl.uint64, tl.uint64),
        is_pure=True,
        pack=1,
    )


@triton.jit
def add_v8_bf16(a_hi, a_lo, b_hi, b_lo):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .v4 .b32 %acc, %tmp;
            mov.v4.b32  %acc, 0;
            mov.b64     {%acc.x, %acc.y}, $2;
            mov.b64     {%acc.z, %acc.w}, $3;
            mov.b64     {%tmp.x, %tmp.y}, $4;
            mov.b64     {%tmp.z, %tmp.w}, $5;
            add.bf16x2  %acc.x, %acc.x, %tmp.x;
            add.bf16x2  %acc.y, %acc.y, %tmp.y;
            add.bf16x2  %acc.z, %acc.z, %tmp.z;
            add.bf16x2  %acc.w, %acc.w, %tmp.w;
            mov.b64     $0, {%acc.x, %acc.y};
            mov.b64     $1, {%acc.z, %acc.w};
        }
        """,
        "=l,=l,l,l,l,l",
        args=[a_hi, a_lo, b_hi, b_lo],
        dtype=(tl.uint64, tl.uint64),
        is_pure=True,
        pack=1,
    )


@triton.jit
def one_shot_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    pid = tl.program_id(axis=0)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    while block_start < (numel // NUMEL_PER_THREAD):
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.
        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        acc_hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
        acc_lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.uint64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)

        tl.store(output_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(output_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")


def one_shot_all_reduce(tensor: torch.Tensor):
    MAX_NUM_BLOCKS = 24
    NUM_WARPS = 16
    BLOCK_SIZE = NUM_WARPS * 32
    NUMEL_PER_THREAD = 8

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        tensor.numel() % NUMEL_PER_THREAD == 0
    ), "The number of elements must be 128-bit aligned."
    num_blocks = min(
        triton.cdiv(triton.cdiv(tensor.numel(), NUMEL_PER_THREAD), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )

    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)
    output = torch.empty_like(tensor)

    one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem_hdl.rank,
        world_size=symm_mem_hdl.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        num_warps=NUM_WARPS,
    )
    return output


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 triton_one_shot_all_reduce.py
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    tensor = symm_mem.empty(8192, dtype=torch.bfloat16, device=device).fill_(1)

    output = one_shot_all_reduce(tensor)
    assert output.eq(world_size).all().item()

    lat_us = benchmark_with_profiler(
        lambda: one_shot_all_reduce(tensor),
        "one_shot_all_reduce_kernel",
        benchmark_iters=200,
    )
    if rank == 0:
        print(f"Median latency: {lat_us:.2f} us")

    dist.destroy_process_group()
