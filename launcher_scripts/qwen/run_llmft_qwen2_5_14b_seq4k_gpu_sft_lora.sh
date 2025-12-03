#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DATA_NAME="${TRAIN_DATA_NAME}"
TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DATA_NAME="${VAL_DATA_NAME}"
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, etc.

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"

MODEL_SAVE_NAME="Qwen2.5-14B-Instruct"

CONTAINER_MOUNT="${CONTAINER_MOUNT}"

CONTAINER="${CONTAINER}"

ENTRY_SCRIPT="/app/src/train_hp.py"

HF_ACCESS_TOKEN="" # Needs to be set for accessing models from gated huggingface repositories

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=fine-tuning/qwen/llmft_qwen2_5_14b_seq4k_gpu_sft_lora \
    base_results_dir=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results \
    recipes.run.hf_access_token=$HF_ACCESS_TOKEN \
    recipes.training_config.model_config.model_name_or_path=$MODEL_NAME_OR_PATH \
    recipes.training_config.model_config.model_save_name=$MODEL_SAVE_NAME \
    recipes.training_config.training_args.training_dir=$EXP_DIR \
    recipes.training_config.datasets.train_data.name=$TRAIN_DATA_NAME \
    recipes.training_config.datasets.train_data.file_path=$TRAIN_DIR \
    recipes.training_config.datasets.val_data.name=$VAL_DATA_NAME \
    recipes.training_config.datasets.val_data.file_path=$VAL_DIR \
    container=$CONTAINER \
    +cluster.container_mounts.0=$CONTAINER_MOUNT \
    git.entry_script=$ENTRY_SCRIPT \
    git.use_default=false \
