#!/bin/bash

# Parameters
#SBATCH --exclusive
#SBATCH --job-name=sagemaker-llama-8b
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --output={$results_dir}/llama-8b/log-sagemaker-llama-8b_%j.out
#SBATCH --time=6-00:00:00

# setup
export NCCL_DEBUG=WARN
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IGNORE_DISABLED_P2P=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Prepare distributed files
srun -l bash -c "scontrol show hostnames | sort > {$results_dir}/llama-8b/hostname"

srun -l bash {$results_dir}/llama-8b/launch_docker_container.sh
srun -l bash {$results_dir}/llama-8b/docker_exec_script.sh