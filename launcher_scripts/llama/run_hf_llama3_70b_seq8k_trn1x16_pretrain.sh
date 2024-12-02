#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

COMPILE="${COMPILE}" # Set to 1 to compile the model, 0 to load a pre-compiled model
TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
MODEL_CONFIG="${MODEL_CONFIG}" # Location of config.json for the model

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    instance_type="trn1.32xlarge" \
    recipes=training/llama/hf_llama3_70b_seq8k_trn1x16_pretrain \
    recipes.run.name="hf-llama3-70b" \
    recipes.run.compile="$COMPILE" \
    recipes.trainer.max_steps=50 \
    recipes.data.train_dir="$TRAIN_DIR" \
    recipes.model.model_config="$MODEL_CONFIG" \
