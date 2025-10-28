import functools
import os

import click
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from utils import benchmark_with_profiler


def multimem_all_reduce(msg):
    torch.ops.symm_mem.multimem_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def one_shot_all_reduce(msg):
    torch.ops.symm_mem.one_shot_all_reduce(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def two_shot_all_reduce(msg):
    torch.ops.symm_mem.two_shot_all_reduce_(
        msg,
        "sum",
        dist.group.WORLD.group_name,
    )


def triton_multimem_all_reduce(msg):
    from triton_multimem_all_reduce import multimem_all_reduce

    multimem_all_reduce(msg)


def triton_one_shot_all_reduce(msg):
    from triton_one_shot_all_reduce import one_shot_all_reduce

    one_shot_all_reduce(msg)


def get_impl(impl: str):
    if impl == "multimem_all_reduce":
        return multimem_all_reduce
    elif impl == "one_shot_all_reduce":
        return one_shot_all_reduce
    elif impl == "two_shot_all_reduce":
        return two_shot_all_reduce
    elif impl == "triton_multimem_all_reduce":
        return triton_multimem_all_reduce
    elif impl == "triton_one_shot_all_reduce":
        return triton_one_shot_all_reduce
    else:
        raise NotImplementedError(impl)


def benchmark(device: torch.device, impl: str, msg_sz_bytes: int):
    msg_numel = msg_sz_bytes // torch.bfloat16.itemsize
    msg = symm_mem.empty(
        msg_numel,
        dtype=torch.bfloat16,
        device=device,
    )
    symm_mem.rendezvous(msg, dist.group.WORLD.group_name)

    target_fn = functools.partial(get_impl(impl), msg)
    baseline_fn = functools.partial(dist.all_reduce, msg)

    target_us = benchmark_with_profiler(
        target_fn, ".*all_reduce.*", benchmark_iters=200
    )
    baseline_us = benchmark_with_profiler(
        baseline_fn, ".*AllReduce.*", benchmark_iters=200
    )
    if dist.get_rank() == 0:
        print(
            f"msg_sz_bytes: {msg_sz_bytes}\t"
            f"nccl_ring: {baseline_us:.2f} us\t"
            f"{impl}: {target_us:.2f} us\t"
        )


@click.command()
@click.option(
    "--impl",
    help="Valid options: multimem_all_reduce, one_shot_all_reduce, two_shot_all_reduce, triton_multimem_all_reduce, triton_one_shot_all_reduce",
    default="multimem_all_reduce",
)
def main(impl: str):
    """
    Benchmark for the symmetric memory-based all-reduce variants.
    NVSwitch is required.

    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 symm_mem_all_reduce.py
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    torch.manual_seed(42 + local_rank)

    if dist.get_rank() == 0:
        print(f"Benchmarking {impl}...")

    msg_sizes = [2**exp for exp in range(12, 33)]
    for msg_sz_bytes in msg_sizes:
        benchmark(device, impl, msg_sz_bytes)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
