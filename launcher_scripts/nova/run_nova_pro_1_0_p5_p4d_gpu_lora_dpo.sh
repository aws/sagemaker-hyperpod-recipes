#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

# AUTO-GENERATED SCRIPT - DO NOT EDIT MANUALLY
# See scripts/launcher_scripts_generator/README.md for customization instructions.

# Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=fine-tuning/nova/nova_1_0/nova_pro/DPO/nova_pro_1_0_p5_p4d_gpu_lora_dpo \
    base_results_dir=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results
