#!/bin/bash

# Original Copyright (c), NVIDIA CORPORATION. Modifications © Amazon.com

# AUTO-GENERATED SCRIPT - DO NOT EDIT MANUALLY
# See scripts/launcher_scripts_generator/README.md for customization instructions.

# Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

# Data paths
TRAIN_DATA=${TRAIN_DATA}
VAL_DATA=${VAL_DATA}

# Model paths
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH}

# Output and experiment directory
EXP_DIR=${EXP_DIR}

# Cluster config
CONTAINER_MOUNT=${CONTAINER_MOUNT}
CONTAINER=${CONTAINER}
NAMESPACE=${NAMESPACE}

# Set run name for training job
RUN_NAME="verl-grpo-llama-3-dot-2-3b-instruct-lora"

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    recipes=fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-2-3b-instruct-lora \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    run.name="${RUN_NAME}" \
    cluster=k8s \
    cluster_type=k8s \
    cluster.namespace=${NAMESPACE} \
    container="${CONTAINER}" \
    ++recipes.training_config.trainer.default_local_dir="${EXP_DIR}" \
    ++recipes.training_config.data.train_files="${TRAIN_DATA}" \
    ++recipes.training_config.data.val_files="${VAL_DATA}" \
    ++recipes.training_config.actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    ++recipes.training_config.critic.model.path="${CRITIC_MODEL_PATH}" \
    ++recipes.training_config.critic.model.tokenizer_path="${ACTOR_MODEL_PATH}" \
    ++recipes.training_config.reward_model.model.path="${REWARD_MODEL_PATH}" \
    ++recipes.training_config.reward_model.model.input_tokenizer="${ACTOR_MODEL_PATH}" \
    "$@"
