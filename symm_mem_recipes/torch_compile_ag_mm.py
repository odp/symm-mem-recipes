import functools
import os
from typing import List

import click
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import all_gather_tensor

from symm_mem_recipes.utils import benchmark_with_event


def parse_csv(ctx, param, value):
    return [int(num) for num in value.split(",")]


def all_gather_matmul(a_shard, bs, gather_dim, group_name):
    a = all_gather_tensor(a_shard.contiguous(), gather_dim=gather_dim, group=group_name)
    return [torch.matmul(a, b) for b in bs]


compiled_all_gather_matmul = torch.compile(
    options={
        "_micro_pipeline_tp": True,
        "keep_output_stride": False,
    },
    fullgraph=True,
)(all_gather_matmul)


def scaled_matmul(a, b, a_scale, b_scale, **kwargs):
    leading_dims = a.shape[:-1]
    c = torch._scaled_mm(a.flatten(0, -2), b, a_scale, b_scale, **kwargs)
    return c.view(*leading_dims, -1)


def all_gather_scaled_matmul(a_shard, bs, a_scale, b_scales, gather_dim, group_name):
    a = all_gather_tensor(a_shard.contiguous(), gather_dim=gather_dim, group=group_name)
    return [
        scaled_matmul(
            a, b, a_scale, b_scale, out_dtype=torch.bfloat16, use_fast_accum=True
        )
        for b, b_scale in zip(bs, b_scales)
    ]


compiled_all_gather_scaled_matmul = torch.compile(
    options={
        "_micro_pipeline_tp": True,
        "keep_output_stride": False,
    },
    fullgraph=True,
)(all_gather_scaled_matmul)


@click.command()
@click.option("--batch", default=1)
@click.option("--M", default=8192)
@click.option("--N", callback=parse_csv, default="3584")
@click.option("--K", default=8192)
@click.option("--dtype", default="bfloat16")
@click.option("--gather-dim", default=0)
@click.option("--scale-mode", default="tensor-wise")
@click.option("--cuda-graph", is_flag=True, default=False)
def main(
    batch: int,
    m: int,
    n: int,
    k: List[int],
    dtype: str,
    gather_dim: int,
    scale_mode: str,
    cuda_graph: bool,
):
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 -m symm_mem_recipes.torch_compile_ag_mm --cuda-graph
    """
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHE"] = "1"
    os.environ["TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP"] = "1"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"M={m}, N={n}, K={k}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)

    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)

    a_shard = torch.rand(batch, m // world_size, k, dtype=torch.bfloat16, device="cuda")
    bs = [torch.rand(N, k, dtype=torch.bfloat16, device="cuda").T for N in n]

    if dtype == "bfloat16":
        baseline = functools.partial(
            all_gather_matmul, a_shard, bs, gather_dim=gather_dim, group_name=group_name
        )
        compiled = functools.partial(
            compiled_all_gather_matmul,
            symm_mem.restride_A_shard_for_fused_all_gather_matmul(
                a_shard, gather_dim=gather_dim
            ),
            bs,
            gather_dim=gather_dim,
            group_name=group_name,
        )

    elif dtype == "float8":
        a_shard = a_shard.to(torch.float8_e4m3fn)
        bs = [B.to(torch.float8_e4m3fn) for B in bs]

        if scale_mode == "tensor-wise":
            A_scale = torch.tensor(0.1, device="cuda")
            B_scales = [torch.tensor(0.1, device="cuda") for _ in n]
        elif scale_mode == "row-wise":
            A_scale = torch.full((batch, m // world_size, 1), 0.1, device="cuda")
            B_scales = [torch.full((1, N), 0.1, device="cuda") for N in n]
        else:
            raise AssertionError(f"Invalid scale_mode: {scale_mode}")

        baseline = functools.partial(
            all_gather_scaled_matmul,
            a_shard,
            bs,
            A_scale,
            B_scales,
            gather_dim=gather_dim,
            group_name=group_name,
        )
        compiled = functools.partial(
            compiled_all_gather_scaled_matmul,
            symm_mem.restride_A_shard_for_fused_all_gather_matmul(
                a_shard, gather_dim=gather_dim
            ),
            bs,
            A_scale,
            B_scales,
            gather_dim=gather_dim,
            group_name=group_name,
        )

    else:
        raise AssertionError(f"Invalid dtype: {dtype}")

    torch.testing.assert_close(baseline(), compiled())
    baseline_us = benchmark_with_event(baseline, flush_l2=True, cuda_graph=cuda_graph)
    compiled_us = benchmark_with_event(compiled, flush_l2=True, cuda_graph=cuda_graph)
    print(f"baseline us: {baseline_us:.2f}; compiled us: {compiled_us:.2f}")


if __name__ == "__main__":
    main()
