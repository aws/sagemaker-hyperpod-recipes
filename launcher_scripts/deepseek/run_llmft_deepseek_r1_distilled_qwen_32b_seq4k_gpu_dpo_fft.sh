#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

# AUTO-GENERATED SCRIPT - DO NOT EDIT MANUALLY
# See scripts/launcher_scripts_generator/README.md for customization instructions.

# Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DATA_NAME="${TRAIN_DATA_NAME}"
TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DATA_NAME="${VAL_DATA_NAME}"
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, etc.

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"

MODEL_SAVE_NAME="DeepSeek-R1-Distill-Qwen-32B"

CONTAINER_MOUNT="${CONTAINER_MOUNT}"

CONTAINER="${CONTAINER}"

ENTRY_MODULE="amzn_awsllm_fine_tuning.train_hp"

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_dpo_fft \
    base_results_dir=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results \
    recipes.training_config.model_config.model_name_or_path=$MODEL_NAME_OR_PATH \
    recipes.training_config.model_config.model_save_name=$MODEL_SAVE_NAME \
    recipes.training_config.training_args.training_dir=$EXP_DIR \
    recipes.training_config.datasets.train_data.name=$TRAIN_DATA_NAME \
    recipes.training_config.datasets.train_data.file_path=$TRAIN_DIR \
    recipes.training_config.datasets.val_data.name=$VAL_DATA_NAME \
    recipes.training_config.datasets.val_data.file_path=$VAL_DIR \
    container=$CONTAINER \
    +cluster.container_mounts.0=$CONTAINER_MOUNT \
    git.entry_module=$ENTRY_MODULE \
    git.use_default=false \
