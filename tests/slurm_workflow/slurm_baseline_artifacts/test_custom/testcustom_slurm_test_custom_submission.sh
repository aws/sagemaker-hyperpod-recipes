#!/bin/bash

# Parameters
#SBATCH --error={$results_dir}/test_custom/log-testcustom_slurm_test_custom_%j.err
#SBATCH --exclusive
#SBATCH --job-name=testcustom_slurm_test_custom
#SBATCH --nodes=2
#SBATCH --output={$results_dir}/test_custom/log-testcustom_slurm_test_custom_%j.out

# setup
export NCCL_DEBUG=DEBUG
export FI_PROVIDER=efa
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IGNORE_DISABLED_P2P=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DIST_INIT_BARRIER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1


# Prepare distributed files
srun -l bash -c "scontrol show hostnames | sort > {$results_dir}/test_custom/hostname"

srun -l bash {$results_dir}/test_custom/launch_docker_container.sh
srun -l bash {$results_dir}/test_custom/docker_exec_script.sh