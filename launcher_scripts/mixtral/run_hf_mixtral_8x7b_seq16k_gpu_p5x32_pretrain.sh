#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DIR="${TRAIN_DIR}" # Location of training dataset
VAL_DIR="${VAL_DIR}" # Location of validation dataset

EXP_DIR="${EXP_DIR}" # Location to save experiment info including logging, checkpoints, etc.


HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=training/mixtral/hf_mixtral_8x7b_seq16k_gpu_p5x32_pretrain \
    base_results_dir=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results \
    recipes.run.name="hf-mixtral-8x7b" \
    recipes.exp_manager.exp_dir=$EXP_DIR \
    recipes.model.data.train_dir=$TRAIN_DIR \
    recipes.model.data.val_dir=$VAL_DIR \
