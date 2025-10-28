#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# File: triton_multimem_all_reduce_annotated.py
# Purpose: Line-by-line documented version of a Triton-based NVLS (NVLink SHARP)
#          all-reduce using CUDA/PTX "multimem" instructions on multicast
#          addresses via PyTorch Symmetric Memory.
#
# What this file does (high level):
#   • Creates a Triton kernel that performs an all-reduce(sum) by issuing:
#       - multimem.ld_reduce  : read-and-reduce across all replicas (GPUs)
#       - multimem.st         : broadcast/write the reduced result to all replicas
#     on a "multicast" (aka "multimem") address mapped identically on each GPU.
#   • Uses Symmetric Memory rendezvous to obtain:
#       - multicast_ptr       : the special pointer for multimem.* ops
#       - signal_pad_ptrs     : lightweight cross-GPU synchronization pads
#   • Performs block/rank partitioning so each GPU reduces its slice.
#
# Key constraints/assumptions:
#   • Requires Hopper-or-newer GPUs (SM90+) in an NVSwitch (v3+) domain for NVLS.
#   • Data type: bf16 (bfloat16). The PTX variant here uses 128-bit packets
#     (v4 of bf16x2 = 8 bf16 elements per 128-bit chunk).
#   • Tensor must be 128-bit aligned (numel multiple of 8 for bf16).
#
# Notes:
#   • All explanations are in comments; the original logic is preserved.
#   • External references/sources are listed after this code block.
# =============================================================================

import torch                                   # PyTorch tensors & device utilities
import torch.distributed as dist               # Process groups, world/rank info
import torch.distributed._symmetric_memory as symm_mem  # Symmetric Memory API

import triton                                  # Triton JIT compiler/runtime
import triton.language as tl                   # Triton language intrinsics

# Local utilities (assumed available in your environment):
# - blockwise_barrier: GPU-side distributed barrier using Symmetric Memory signal pads
# - get_flat_tid:     Helper to get a flattened thread index within a program (block)
# - sync_threads:     CTA-level barrier (threads synchronization inside a block)
from triton_barrier import blockwise_barrier
from triton_utils import get_flat_tid, sync_threads
from utils import log_triton_kernel


# -----------------------------------------------------------------------------
# Inline PTX helpers
# -----------------------------------------------------------------------------
# We create two small Triton @jit helpers that issue PTX "multimem.*" instructions
# to a multicast pointer. We guard each instruction with a predicate 'mask' so we
# can handle edge cases (tails) safely. The 'inline_asm_elementwise' maps each
# scalar lane to the corresponding asm snippet.


@triton.jit
def multimem_ld_reduce_128(multicast_ptrs, mask):
    """
    Issue a PTX 'multimem.ld_reduce' that:
      - Reads from ALL replicas behind a multicast ("multimem") address
      - Reduces (sum) those values in the fabric (NVLS) into registers
      - Returns a 128-bit vector as four 32-bit registers (v4 of bf16x2)
    Args:
      multicast_ptrs: pointer tensor to 128-bit chunks (rank-major addressing)
      mask          : per-element predicate (1=execute, 0=skip)
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;                           // predicate register
            setp.eq.s32 %p0, $5, 1;                   // %p0 = (mask == 1)
            @!%p0 bra end;                            // if not set, skip
            // NVLS reduce-load across all GPU replicas into 128 bits:
            multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        # Output/Input constraints mapping:
        #   "=r,=r,=r,=r" -> four 32-bit outputs (bf16x2 lanes)
        #   "l"           -> pointer (64-bit address) input
        #   "r"           -> scalar register input (mask)
        "=r,=r,=r,=r,l,r",
        args=[multicast_ptrs, mask.to(tl.int32)],     # pass pointer + mask
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),  # 4x 32-bit outputs
        is_pure=True,                                 # declares no side-effects
        pack=1,                                       # operate per element
    )


@triton.jit
def multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    """
    Issue a PTX 'multimem.st' that:
      - Stores 128 bits (v4.f32 here) to ALL replicas (broadcast in fabric)
    Args:
      multicast_ptrs: destination multicast pointer (128-bit chunk base)
      x,y,z,w       : four 32-bit lanes to write
      mask          : per-element predicate guard
    """
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;                           // predicate register
            setp.eq.s32 %p0, $6, 1;                   // %p0 = (mask == 1)
            @!%p0 bra end;                            // if not set, skip
            // NVLS broadcast-store to all GPU replicas:
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        # Output/Input constraints mapping:
        #   "=r"          -> dummy output (kept for symmetry; PTX st has no real out)
        #   "l"           -> pointer (64-bit address)
        #   "r,r,r,r"     -> 4x 32-bit register inputs (the data)
        #   "r"           -> mask
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],  # args -> asm placeholders
        dtype=(tl.uint32),                            # placeholder dst (unused)
        is_pure=False,                                # writes memory (has side-effects)
        pack=1,
    )


# -----------------------------------------------------------------------------
# Triton kernel: NVLS all-reduce via multimem ld_reduce + st
# -----------------------------------------------------------------------------
@triton.jit
def multimem_all_reduce_kernel(
    multicast_ptr,                # (ptr) multicast ("multimem") base pointer
    signal_pad_ptrs,              # (ptr) symmetric signal pads for x-GPU barrier
    numel,                        # (int) number of bf16 elements in the tensor
    BLOCK_SIZE: tl.constexpr,     # (int const) threads per program (block)
    NUMEL_PER_THREAD: tl.constexpr,  # (int const) elements each thread processes per 128b packet
    RANK: tl.constexpr,           # (int const) this GPU's rank
    WORLD_SIZE: tl.constexpr,     # (int const) number of GPUs
):
    # ---- Entry barrier (relaxed) across GPUs:
    # Ensures all peers have reached the starting line and data is visible enough
    # for the upcoming fabric operations. Uses Symmetric Memory signal pads.
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="relaxed")

    # Intra-CTA sync to ensure all threads locally are aligned before proceeding
    sync_threads()

    # Triton launch geometry:
    pid = tl.program_id(axis=0)   # program (block) id along the single grid dimension
    tid = get_flat_tid()          # flattened thread index within this program (block)

    # Convert "elements" to "128-bit packets" count:
    # bf16 element size = 2 bytes; BYTES_PER_THREAD=16 -> 8 elements per packet
    numel = numel // NUMEL_PER_THREAD

    # Split the work evenly across ranks (ceil-div to cover tails)
    numel_per_rank = tl.cdiv(numel, WORLD_SIZE)

    # Starting packet index for this program (grid-stride loop follows)
    block_start = pid * BLOCK_SIZE

    # Grid-stride loop over all 128-bit packets assigned to this rank
    while block_start < numel_per_rank:
        # Compute per-thread packet offset
        offsets = block_start + tid

        # Mask guards the last partial tile (if any)
        mask = offsets < numel_per_rank

        # Rank-major addressing:
        #  Each GPU owns [RANK * numel_per_rank, (RANK+1) * numel_per_rank)
        #  We treat each packet as two 64-bit words -> add "* 2" in u64 pointer arithmetic.
        ptrs = (
            multicast_ptr.to(tl.pointer_type(tl.uint64))
            + (RANK * numel_per_rank + offsets) * 2
        )

        # 1) Reduce-load 128 bits from ALL replicas (NVSwitch fabric does the sum)
        (x, y, z, w) = multimem_ld_reduce_128(ptrs, mask=mask)

        # 2) Broadcast (store) the reduced 128 bits to ALL replicas (NVSwitch fan-out)
        multimem_st_128(ptrs, x, y, z, w, mask=mask)

        # Advance by the number of programs in the grid (grid-stride)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    # Local sync to ensure all threads in CTA completed their stores
    sync_threads()

    # ---- Exit barrier (acq_rel) across GPUs:
    # Acquire-Release ensures visibility of the just-written results on all peers.
    blockwise_barrier(signal_pad_ptrs, None, RANK, WORLD_SIZE, sem="acq_rel")


# -----------------------------------------------------------------------------
# Python wrapper: chooses launch config and runs the Triton NVLS kernel
# -----------------------------------------------------------------------------
def multimem_all_reduce(tensor: torch.Tensor):
    """
    NVLS-backed all-reduce on a bf16 tensor using Symmetric Memory + Triton.

    Args:
      tensor (torch.Tensor, bf16, CUDA): symmetric tensor to all-reduce (in-place)

    Returns:
      tensor (same object): now containing the all-reduced result.
    """
    # Launch tuning knobs
    WARP_SIZE = 32
    MAX_NUM_BLOCKS = 4
    MAX_BLOCK_SIZE = 1024
    BYTES_PER_THREAD = 16            # 128 bits per thread (v4.* PTX vector width)

    # Establish (or fetch cached) Symmetric Memory handle for 'tensor';
    # This must be called in the SAME order across all ranks in the group.
    symm_mem_hdl = symm_mem.rendezvous(tensor, group=dist.group.WORLD)

    # Kernel supports bfloat16 only (bf16x2 packing in PTX)
    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."

    # Elements per 128-bit packet: 16B / 2B = 8 elements
    numel_per_thread = BYTES_PER_THREAD // tensor.element_size()

    # Enforce 128-bit alignment: number of elements must be divisible by 8
    assert (
        tensor.numel() % numel_per_thread == 0
    ), "The number of elements must be 128-bit aligned."

    # Packets per rank (ceil-div by world size)
    num_threads = triton.cdiv(
        tensor.numel() // numel_per_thread, symm_mem_hdl.world_size
    )

    # Choose a block size and grid size:
    if num_threads < MAX_BLOCK_SIZE:
        # Small problem: use a single block sized to next power-of-two
        block_size = 1
        while block_size < num_threads:
            block_size *= 2
        num_warps = block_size // WARP_SIZE
        num_blocks = 1
    else:
        # Larger problem: cap block to 1024 threads, use a few blocks
        block_size = MAX_BLOCK_SIZE
        num_warps = MAX_BLOCK_SIZE // WARP_SIZE
        num_blocks = min(
            triton.cdiv(num_threads, MAX_BLOCK_SIZE),
            MAX_NUM_BLOCKS,
        )

    # Launch the Triton kernel with Symmetric Memory pointers:
    #  - multicast_ptr:   NVLS "multimem" address for tensor's symmetric buffer
    #  - signal_pad_ptrs: per-rank pads for cross-GPU barriers
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

    # Optional: log compilation/launch info for debugging/profiling
    log_triton_kernel(kernel)

    # The tensor is reduced in-place (via the multimem stores).
    return tensor


# -----------------------------------------------------------------------------
# Script entry point: CLI invocation via torchrun (same as original)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example invocation (single node, 8 GPUs):

    torchrun \
      --nnodes 1 --nproc-per-node 8 \
      --rdzv-backend c10d --rdzv-endpoint localhost:0 \
      --no_python python3 triton_multimem_all_reduce.py
    """
    # Reuse a harness that wires up args/inputs and calls our implementation.
    # Expected to import main(["--impl", "triton_multimem_all_reduce"])
    # which will exercise multimem_all_reduce(...) above.
    from symm_mem_all_reduce import main

    main(["--impl", "triton_multimem_all_reduce"])

