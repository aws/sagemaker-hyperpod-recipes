#!/bin/bash

REGION="us-west-2"
IMAGE="885173626742.dkr.ecr.${REGION}.amazonaws.com/cjz-eval-poc:s3-util-log"
SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}
EXP_DIR=${EXP_DIR:-"/tmp/deterministic_eval_experiment"}
EVAL_RESULTS_DIR=${EVAL_RESULTS_DIR:-"s3://your-bucket/eval-results"}
EVAL_TENSORBOARD_RESULTS_DIR=${EVAL_TENSORBOARD_RESULTS_DIR:-"s3://your-bucket/tensorboard"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"s3://my-model-path/test"}
BASE_MODEL_NAME=${BASE_MODEL_NAME:-"meta-textgeneration-llama-3-1-8b-instruct"}
DATA_S3_PATH=${DATA_S3_PATH:-"s3://test-bucket/data"}
TASK=${TASK:-"mmlu"}
STRATEGY=${STRATEGY:-"zs_cot"}
METRIC=${METRIC:-"accuracy"}
SUBTASK=${SUBTASK:-"abstract_algebra"}
CLUSTER_TYPE=${CLUSTER_TYPE:-"k8s"}

HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
    hydra.job.chdir=True \
    base_results_dir=${EXP_DIR} \
    recipes=evaluation/open-source/open_source_deterministic_eval \
    recipes.run.name="deterministic-eval-job" \
    recipes.run.model_name_or_path=${MODEL_NAME_OR_PATH} \
    recipes.run.base_model_name=${BASE_MODEL_NAME} \
    recipes.run.data_s3_path=${DATA_S3_PATH} \
    recipes.evaluation.task=${TASK} \
    recipes.evaluation.strategy=${STRATEGY} \
    recipes.evaluation.metric=${METRIC} \
    recipes.evaluation.subtask=${SUBTASK} \
    recipes.output.eval_results_dir=${EVAL_RESULTS_DIR} \
    recipes.output.eval_tensorboard_results_dir=${EVAL_TENSORBOARD_RESULTS_DIR} \
    container=${IMAGE} \
    cluster_type=${CLUSTER_TYPE} \
    launch_json=true
