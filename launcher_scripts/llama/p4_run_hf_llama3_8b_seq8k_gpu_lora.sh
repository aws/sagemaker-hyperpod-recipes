#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH}" # HuggingFace pretrained model name or path
HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN}" # Optional HuggingFace access token

TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, etc.


HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=fine-tuning/llama/p4_hf_llama3_8b_seq8k_gpu_lora \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    recipes.run.name="hf-llama3-8b-lora" \
    recipes.exp_manager.exp_dir="$EXP_DIR" \
    recipes.trainer.num_nodes=1 \
    recipes.model.train_batch_size=1 \
    recipes.model.data.train_dir="$TRAIN_DIR" \
    recipes.model.data.val_dir="$VAL_DIR" \
    recipes.model.hf_model_name_or_path="$HF_MODEL_NAME_OR_PATH" \
    recipes.model.hf_access_token="$HF_ACCESS_TOKEN" \
