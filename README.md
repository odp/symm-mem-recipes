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

Some benchmarks on 8xH100 (special version with HBM2e, at 650W) with NVSwitch:

#### Llama 3 8B (N=1792, K=4096)
| Problem Size<br>(M) | Config<sup>1</sup>  | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) | Speedup |
|------------|------------|------------|------------|------------|------------|------------|
| 4096 | 64,128,128,4 | 100 | 142 | 223 | 211 | 1.05x<sup>2</sup> |
| 8192 | 128,128,64,6 | 186 | 198 | 393 | 293 | 1.34x |
| 16384 | 128,256,64,3 | 363 | 363 | 748 | 485 | 1.54x |

#### Llama 3 70B (N=3584, K=8192)
| Problem Size<br>(M) | Config<sup>1</sup>  | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) | Speedup |
|------------|------------|------------|------------|------------|------------|------------|
| 4096 | 128,128,64,6 | 376 | 392 | 587 | 453 | 1.29x |
| 8192 | 128,256,64,3 | 746 | 706 | 1168 | 821 | 1.42x |
| 16384 | 128,256,64,3 | 1502 | 1403  | 2306 | 1566 | 1.47x |

#### Llama 3 105B (N=6656, K=16384)
| Problem Size<br>(M) | Config<sup>1</sup>  | cuBLAS MM<br>Only (µs) | Triton MM<br>Only (µs) | cuBLAS +<br>NCCL (µs) | Triton<br>Fused (µs) | Speedup |
|------------|------------|------------|------------|------------|------------|------------|
| 4096 | 128,256,64,3 | 1358 | 1425 | 1858 | 1615 | 1.15x |
| 8192 | 128,256,64,3 | 2567 | 2656 | 3533 | 2907 | 1.22x |
| 16384 | 128,256,64,3 | 5249 | 5375 | 6982 | 5814 | 1.20x |

<sup>1</sup> Config refers to `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, and `num_stages`.

<sup>2</sup> For this problem size, using multicast all-gather would be a more suitable optimization.
