#!/bin/bash
set -ex

function job_epilogue {
  docker ps -a --filter 'name=sm_training_launcher' --format '{{.ID}}' | xargs -I{} docker rm -f {} > /dev/null 2>&1 || true
}
trap job_epilogue EXIT SIGTERM SIGINT

docker exec sm_training_launcher bash {$results_dir}/llama-8b/train_script.sh

exit 0