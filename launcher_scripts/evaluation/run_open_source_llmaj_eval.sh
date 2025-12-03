#!/bin/bash

REGION="us-west-2"
IMAGE="708977205387.dkr.ecr.${REGION}.amazonaws.com/nova-fine-tune-repo:SM-TJ-SFT-latest"
SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}
EXP_DIR=${EXP_DIR:-"/tmp/llmaj_eval_experiment"}
EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:-"s3://your-bucket/eval-results"}
EVAL_TENSORBOARD_RESULTS_DIR=${EVAL_TENSORBOARD_RESULTS_DIR:-"s3://your-bucket/tensorboard"}
DATA_S3_PATH=${DATA_S3_PATH:-"s3://test-bucket/data"}
CLUSTER_TYPE=${CLUSTER_TYPE:-"k8s"}

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    hydra.job.chdir=True \
    base_results_dir=${EXP_DIR} \
    recipes=evaluation/open-source/open_source_llmaj_eval \
    recipes.run.data_s3_path=${DATA_S3_PATH} \
    recipes.output.eval_results_dir=${EVAL_RESULTS_DIR} \
    recipes.output.eval_tensorboard_results_dir=${EVAL_TENSORBOARD_RESULTS_DIR} \
    container=${IMAGE} \
    cluster_type=${CLUSTER_TYPE} \
    launch_json=true
