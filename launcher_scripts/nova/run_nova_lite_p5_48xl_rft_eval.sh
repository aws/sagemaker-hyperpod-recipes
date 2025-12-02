#!/bin/bash
set -e

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=evaluation/nova/nova_1_0/nova_lite/nova_lite_1_0_p5_48xl_gpu_rft_eval \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results"
