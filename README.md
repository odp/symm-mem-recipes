# symm-mem-recipes

This repository includes:
- Usage and benchmarks of `SymmetricMemory`-based multi-GPU algorithms in PyTorch.
- Examples and benchmarks of multi-GPU algorithms built with `SymmetricMemory` + Triton.

---
## symm_mem_all_reduce.py

This script demonstrates the usage of `SymmetricMemory`-based NVLink all-reduce implementations and benchmarks their performance. The available variants are:
- `multimem_all_reduce` (PyTorch op available in nightly)
- `one_shot_all_reduce` (PyTorch op available in nightly)
- `two_shot_all_reduce` (PyTorch op available in nightly)
- `triton_multimem_all_reduce` (Triton kernel defined in this repo)
- `triton_one_shot_all_reduce` (Triton kernel defined in this repo)

Usage:
```bash
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 symm_mem_all_reduce.py --impl multimem_all_reduce
```

Some benchmarks on 8xH100 with NVSwitch:

<img src="https://github.com/user-attachments/assets/5de69841-7683-4b7a-9a38-f1aac3785060" width="60%">

<img src="https://github.com/user-attachments/assets/c666cd6c-3f70-4380-9fa1-0d8e953cb382" width="60%">

<img src="https://github.com/user-attachments/assets/597e12d8-37ed-4776-aca8-2b12bba58bff" width="60%">

<img src="https://github.com/user-attachments/assets/1cfa320d-589f-466f-a54f-7fa45e6f132e" width="60%">


---
## triton_all_gather_matmul.py

This is a fused all-gather matmul example using Triton + `SymmetricMemory`, based on the `tma_persistent` Triton tutorial with slight modifications.

This example requires PyTorch Nightly and Triton 3.0.0+ to run.

Usage:
```bash
torchrun \
--nnodes 1 --nproc-per-node 8 \
--rdzv-backend c10d --rdzv-endpoint localhost:0 \
--no_python python3 triton_all_gather_matmul.py \
--M 16384 --N 6656 --K 16384 --BLOCK_SIZE_M 128 --BLOCK_SIZE_N 256 --BLOCK_SIZE_K 64
```

Some benchmarks on 8xH100 (special version with HBM2e, at 500W) with NVSwitch:

#### Llama 3 8B (N=1792, K=4096)
| Problem Size<br>(M) | Block Size<br>(M, N, K) | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 128, 128 | 105 | 125 | 230 | 213 |
| 8192 | 128, 128, 128 | 194 | 236 | 416 | 318 |
| 16384 | 256, 128, 64 | 391 | 434 | 819 | 514 |

#### Llama 3 70B (N=3584, K=8192)
| Problem Size<br>(M) | Block Size<br>(M, N, K) | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 128, 128 | 403 | 483 | 652 | 543 |
| 8192 | 256, 128, 64 | 828 | 849 | 1291 | 948 |
| 16384 | 256, 128, 64 | 1672 | 1655 | 2541 | 1846 |

#### Llama 3 105B (N=6656, K=16384)
| Problem Size<br>(M) | Block Size<br>(M, N, K) | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) |
|------------|------------|------------|------------|------------|------------|
| 4096 | 128, 256, 64 | 1558 | 1595 | 2077 | 1776 |
| 8192 | 128, 256, 64 | 2879 | 2953 | 3847 | 3243 |
| 16384 | 128, 256, 64 | 5842 | 5948 | 7801 | 6538 |
