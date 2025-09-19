#!/bin/bash
set -e

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=fine-tuning/nova/nova_lite_g5_g6_12x_gpu_lora_sft \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results"
