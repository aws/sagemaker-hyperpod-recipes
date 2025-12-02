#!/bin/bash
set -e

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=fine-tuning/nova/nova_1_0/nova_pro/distill/nova_pro_1_0_r5_cpu_distill \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results"
