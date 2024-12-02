#!/bin/bash

#Users should setup their cluster type in /recipes_collection/config.yaml

SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}

TRAIN_DIR=${TRAIN_DIR} # Location of training dataset
VAL_DIR=${VAL_DIR} # Location of talidation dataset

EXP_DIR=${EXP_DIR} # Location to save experiment info including logging, checkpoints, ect


HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
--config-path=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/launcher_scripts/custom_script \
--config-name=config_slurm \
base_results_dir=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results \
training_cfg.entry_script=${SAGEMAKER_TRAINING_LAUNCHER_DIR}/laucher_scripts/custom_script/custom_allreduce.py \
container_mounts=[${SAGEMAKER_TRAINING_LAUNCHER_DIR}] \
container=<mycontainer>\
