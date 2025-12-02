#!/bin/bash
set -e

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=fine-tuning/nova/nova_2_0/nova_lite/SFT/nova_lite_2_0_p5_gpu_lora_sft \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results"
