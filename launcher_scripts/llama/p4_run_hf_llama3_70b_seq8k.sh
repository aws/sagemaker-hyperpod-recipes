#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, etc.


HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=training/llama/p4_hf_llama3_70b_seq8k_gpu \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    recipes.run.name="hf-llama3-70b" \
    recipes.exp_manager.exp_dir="$EXP_DIR" \
    recipes.trainer.num_nodes=32 \
    recipes.model.train_batch_size=1 \
    recipes.model.data.train_dir="$TRAIN_DIR" \
    recipes.model.data.val_dir="$VAL_DIR" \
