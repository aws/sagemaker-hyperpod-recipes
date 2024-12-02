#!/bin/bash
set -ex
echo "image is test_container"
echo "Not an ECR image, skipping ECR login"
# Getting EFA devices
device=("--device=/dev/gdrdrv")
while IFS= read -r -d '' d; do
  device+=("--device=${d}")
done < <(find "/dev/infiniband" -name "uverbs*" -print0)

# Clean old containers
docker ps -a --filter 'name=sm_training_launcher' --format '{{.ID}}' | xargs -I{} docker rm -f {} > /dev/null 2>&1 || true
docker ps -a --filter 'name=sm_training_launcher' --format '{{.ID}}' | xargs -I{} docker wait {} || true

docker pull "test_container"
docker run --gpus 8 \
  --privileged --rm -d --name "sm_training_launcher" \
  --uts=host --ulimit stack=67108864 --ulimit memlock=-1 --ipc=host --net=host \
  --security-opt seccomp=unconfined  \
  "${device[@]}" \
  -v {$workspace_dir}/launcher/nemo/nemo_framework_launcher/launcher_scripts:{$workspace_dir}/launcher/nemo/nemo_framework_launcher/launcher_scripts \
  -v {$results_dir}:{$results_dir} \
  "test_container" sleep infinity

# Running post launching commands
docker exec -itd "sm_training_launcher" bash -c "printf \"Port 2022\n\" >> /etc/ssh/sshd_config"
docker exec -itd "sm_training_launcher" bash -c "printf \"  Port 2022\n\" >> /root/.ssh/config"
docker exec -itd "sm_training_launcher" bash -c "service ssh start"

exit 0