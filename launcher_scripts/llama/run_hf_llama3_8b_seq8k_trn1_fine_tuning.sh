#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

COMPILE="${COMPILE}"
COMPILER_CACHE_PATH="${COMPILER_CACHE_PATH}"
TOKENIZER_TYPE="${TOKENIZER_TYPE}"
TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DIR="${VAL_DIR}" # Location of validation dataset
RESUME_FROM_CHECKPOINT_DIR="${RESUME_FROM_CHECKPOINT_DIR}"
MODEL_CONFIG="${MODEL_CONFIG}" # Location of config.json for the model

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    instance_type="trn1.32xlarge" \
    recipes=training/llama/hf_llama3_8b_seq8k_trn1_fine_tuning \
    recipes.run.name="hf-llama3-8b-sft" \
    recipes.run.compile="$COMPILE" \
    recipes.trainer.max_steps=50 \
    recipes.compiler_cache_url="$COMPILER_CACHE_PATH" \
    recipes.data.tokenizer.type="$TOKENIZER_TYPE" \
    recipes.data.train_dir="$TRAIN_DIR" \
    recipes.data.val_dir="$VAL_DIR" \
    recipes.exp_manager.resume_from_checkpoint="$RESUME_FROM_CHECKPOINT_DIR" \
    recipes.model.model_config="$MODEL_CONFIG" \
