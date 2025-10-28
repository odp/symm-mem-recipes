#!/bin/bash
#SBATCH --job-name=symm-mem-allreduce
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # 1 launcher; torchrun will spawn 8 procs
#SBATCH --gres=gpu:8
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --begin=now+0minutes
#SBATCH --wait-all-nodes=1
#SBATCH --reservation=moe
#SBATCH --exclude=fs-mbz-gpu-[041,022,003,439,060,383]
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

echo "START: $(date)"
echo "JOB  : ${SLURM_JOB_NAME}  ID=${SLURM_JOB_ID}"

# ---- Communication/runtime knobs (cluster-specific; adjust as needed) ----
export CUDA_DEVICE_MAX_CONNECTIONS=1            # stream/queue mapping; often set to 1 for TP overlap
export NCCL_DEBUG=WARN                          # quieter logs
export NCCL_IBEXT_DISABLE=1                     # disable external IB plugin if undesired
export NCCL_NVLS_ENABLE=1                       # enable NVLink SHARP/NVLS when available
export NCCL_IB_HCA=mlx5                         # prefer mlx5-class HCAs
export UCX_NET_DEVICES=mlx5_0:1                 # pick the IB device/port (example; tune per node)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200    # extend watchdog heartbeat timeout

# Optional scratch/tmp
# export TMPDIR=/path/to/fast/tmp

# Optional: tokens/creds should be injected securely (don’t hardcode)
# export HF_TOKEN=...

cd "${SLURM_SUBMIT_DIR}"

source $HOME/symm-mem-recipes/.venv/bin/activate

# Single-node, 8 GPUs as requested

torchrun \
  --nnodes 1 --nproc-per-node 8 \
  --rdzv-backend c10d --rdzv-endpoint localhost:0 \
  --no_python python3 triton_all_gather_matmul.py \
  --M 16384 --N 6656 --K 16384 --BLOCK_SIZE_M 128 --BLOCK_SIZE_N 256 --BLOCK_SIZE_K 64

echo "END  : $(date)"
