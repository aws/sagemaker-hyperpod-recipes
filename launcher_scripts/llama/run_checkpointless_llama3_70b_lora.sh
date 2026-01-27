#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

# AUTO-GENERATED SCRIPT - DO NOT EDIT MANUALLY
# See scripts/launcher_scripts_generator/README.md for customization instructions.

# Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DIR="${TRAIN_DIR}"
VAL_DIR="${VAL_DIR}"
EXP_DIR="${EXP_DIR}"
LOG_DIR="${LOG_DIR}"
CONTAINER_MOUNT="/data"
CONTAINER="${CONTAINER}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=fine-tuning/llama/checkpointless_llama3_70b_lora \
    recipes.dataset.dataset_path="${TRAIN_DIR}" \
    recipes.exp_manager.exp_dir="${EXP_DIR}" \
    recipes.log_dir="${LOG_DIR}" \
    recipes.resume.restore_config.path="${MODEL_NAME_OR_PATH}" \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    git.use_default=false \
    cluster=k8s \
    cluster_type=k8s \
    container="${CONTAINER}" \
    +cluster.hostNetwork=true \
    +cluster.persistent_volume_claims.0.claimName=fsx-claim \
    +cluster.persistent_volume_claims.0.mountPath="${CONTAINER_MOUNT}" \
