# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
# Portions taken from https://github.com/NVIDIA/NeMo-Framework-Launcher, Copyright Nvidia Corporation


import json
import logging
import re
import shutil
import subprocess
from ast import literal_eval
from pathlib import Path
from typing import Dict, List

import omegaconf
from hydra.core.hydra_config import HydraConfig
from nemo_launcher.core.stages import Training, _hydra_interpolation
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf, open_dict

from ..accelerator_devices import get_num_accelerator_devices
from ..efa import (
    efa_supported_instance,
    instanceWithMultipleEFAs,
    instanceWithRDMASupport,
)
from ..recipe_templatization.checkpointless.checkpointless_recipe_template_processor import (
    CheckpointlessRecipeTemplateProcessor,
)
from ..recipe_templatization.llmft.llmft_recipe_template_processor import (
    LLMFTRecipeTemplateProcessor,
)
from ..recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)
from ..telemetry import Telemetry
from .constants import ROOT_DIR
from .launchers import SMAutoLauncher

logger = logging.getLogger(__name__)

# Predefined distributed args for torchrun
PROCESSES_PER_NODE = "PROCESSES_PER_NODE"
NNODES = "NNODES"
NODEID = "NODEID"
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"
DISTRIBUTED_ARGS = "DISTRIBUTED_ARGS"
CONTAINER_NAME = "sm_training_launcher"
TRANSFORMERS_VERSION_FOR_MULTIMODAL = "4.45.2"


# Get the original recipe path
def get_recipe_file_path():
    hydra_cfg = HydraConfig.get()
    recipe_path = None

    # Check if recipes override was used
    if hydra_cfg.overrides.task and len(hydra_cfg.overrides.task) > 0 and "recipes" in hydra_cfg.overrides.task[0]:
        recipe_path = hydra_cfg.overrides.task[0].split("=")[1]
        return recipe_path

    # If no recipes override found, return None (for custom training)
    return None


def set_multinode_envs(env_vars, instance_type):
    # https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md
    if get_num_efa_devices(instance_type) > 0:
        env_vars["FI_PROVIDER"] = "efa"
    if "NCCL_SOCKET_IFNAME" not in env_vars:
        env_vars["NCCL_SOCKET_IFNAME"] = "^lo,docker0,veth_def_agent"

    env_vars["NCCL_IGNORE_DISABLED_P2P"] = "1"
    env_vars["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env_vars["TORCH_DIST_INIT_BARRIER"] = "1"
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    return env_vars


def allow_rdma(instance_type):
    return instance_type in instanceWithRDMASupport


def get_instance_type(cfg):
    instance_type = None

    if cfg.get("instance_type"):
        instance_type = cfg.instance_type
    else:
        # custom path
        instance_type = cfg.cluster.instance_type

    assert instance_type is not None, "instance type is required from config"

    if instance_type.startswith("ml."):
        instance_type = instance_type[3:]

    return instance_type.lower()


def get_device_type(cfg):
    instance_type = get_instance_type(cfg)

    # p and g instances are GPU instances
    if instance_type.startswith(("p", "g")):
        device_type = "gpu"
    elif instance_type.startswith("trn"):
        device_type = "trainium"
    else:
        device_type = "cpu"

    return device_type


def get_num_efa_devices(instance_type):
    # If not a EFA instance, return 0
    if instance_type not in efa_supported_instance:
        return 0
    # If multi-EFA, return from mapping
    if instance_type in instanceWithMultipleEFAs:
        return instanceWithMultipleEFAs[instance_type]
    # Only a single EFA device
    return 1


def get_ntasks_per_node(stage_cfg):
    """
    Get the number of processes per node used for training
    When running with custom script it will be stage_cfg.run.ntasks_per_node
    """
    ntasks = OmegaConf.select(stage_cfg, "run.ntasks_per_node")
    if ntasks is None:
        ntasks = stage_cfg.get("trainer").get("devices")
    return ntasks


def get_num_nodes(stage_cfg):
    """
    Get the number of nodes used for training
    When running with custom script it will be stage_cfg.run.nodes
    """
    run_cfg = stage_cfg.get("run")
    nodes = run_cfg.get("nodes")
    if nodes is None:
        trainer_cfg = stage_cfg.get("trainer", {})
        nodes = trainer_cfg.get("num_nodes") or trainer_cfg.get("nnodes")
    return nodes


def get_container_type(container):
    if container is None:
        return None
    if container.endswith(".sqsh"):
        return "enroot"
    return "docker"


def convert_dict_to_command_line_args(key_values):
    command = " ".joi
    for key, value in key_values:
        command += f"{key}=value "


class SMTraining(Training):
    """
    Base stage class for doing training on Sagemaker
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Use GPU device for default flow for NeMo runs
        self.device = "gpu"
        self.instance_type = get_instance_type(cfg)
        self.num_efa_devices = get_num_efa_devices(self.instance_type)
        self.telemetry = Telemetry()
        self._launch_json = OmegaConf.select(cfg, "launch_json", default=False)
        if self._launch_json and self.cluster != "k8s" and self.cluster != "sm_jobs":
            raise RuntimeError("`launch_json` is only supported for cluster type 'k8s' and 'sm_jobs'.")

    @property
    def _default_repo(self):
        # Default repo to mount script from
        return None

    @property
    def _default_branch(self):
        # Default repo branch to mount script from
        return None

    def _make_torchrun_string(self):
        """
        Create torchrun string based on single/multi-node job
        """
        ntasks_per_node = get_ntasks_per_node(self.stage_cfg)
        if int(get_num_nodes(self.stage_cfg)) > 1:
            return f"torchrun ${DISTRIBUTED_ARGS} "
        else:
            return f"torchrun --nproc_per_node {ntasks_per_node} "

    def _make_custom_call_string(self, stage_cfg_path=None) -> str:
        """
        Create the training command with torchrun, script and args
        """
        entry_point = ""
        if hasattr(self, "_entry_module") and self._entry_module:
            entry_point += f"-m {self._entry_module}"
        elif self._entry_script_path:
            entry_point += str(self._entry_script_path)
        else:
            raise ValueError("Must set entry_script or entry_module")

        torchrun_cmd = self._make_torchrun_string()
        script_args_str = self.get_script_args_str(stage_cfg_path)
        command = [torchrun_cmd, entry_point, script_args_str]
        command_string = " \\\n  ".join(command)
        return command_string

    def _get_hostfile_location(self):
        """
        Get the file location to store the hostnames
        """
        job_path = self.get_job_path()
        hostfile_location = Path(job_path.folder / "hostname")
        return hostfile_location

    def _use_local_repo(self) -> bool:
        repo_url_or_path = None
        if OmegaConf.select(self.cfg, "git.repo_url_or_path"):
            repo_url_or_path = self.cfg.git.repo_url_or_path
        return repo_url_or_path is not None and not (
            repo_url_or_path.startswith("http") or repo_url_or_path.startswith("codecommit::")
        )

    def _make_docker_exec_script_text(self, stage_cfg_path):
        docker_exec_script_text = ["#!/bin/bash", "set -ex"]

        docker_exec_script_text.append("")
        docker_exec_script_text.append("function job_epilogue {")
        docker_exec_script_text.append(
            "  docker ps -a --filter 'name="
            + CONTAINER_NAME
            + "' --format '{{.ID}}' | xargs -I{} docker rm -f {} > /dev/null 2>&1 || true"
        )
        docker_exec_script_text.append("}")
        docker_exec_script_text.append("trap job_epilogue EXIT SIGTERM SIGINT")

        docker_exec_script_text.append("")
        docker_exec_script_text.append(f"docker exec {CONTAINER_NAME} bash {stage_cfg_path.parents[0]}/train_script.sh")

        docker_exec_script_text.append("")
        docker_exec_script_text.append("exit 0")

        return "\n".join(docker_exec_script_text)

    def _make_launch_docker_container_text(self):
        """
        Creating a script to launch container on all nodes
        This will be called only when running docker container on Slurm cluster
        """
        launch_docker_container_text = ["#!/bin/bash", "set -ex"]
        image = self.cfg.container

        # Login ECR
        launch_docker_container_text.append(f'echo "image is {image}"')
        is_ecr_image = "amazonaws.com" in image
        if not is_ecr_image:
            launch_docker_container_text.append(f'echo "Not an ECR image, skipping ECR login"')
        else:
            # format will be account.dkr.ecr.region.amazonaws.com/repo:tag
            link = image.split("/")[0]
            region = link.split(".")[3]
            launch_docker_container_text.append(f"# Login ECR")
            launch_docker_container_text.append(
                f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {link}"
            )
            launch_docker_container_text.append("")

        # Handle EFA devices
        if get_num_efa_devices(self.instance_type) > 0:
            launch_docker_container_text.append(f"# Getting EFA devices")
            if allow_rdma(self.instance_type):
                launch_docker_container_text.append('device=("--device=/dev/gdrdrv")')
            else:
                launch_docker_container_text.append("device=()")
            launch_docker_container_text.extend(
                [
                    "while IFS= read -r -d '' d; do",
                    '  device+=("--device=${d}")',
                    'done < <(find "/dev/infiniband" -name "uverbs*" -print0)',
                ]
            )
            launch_docker_container_text.append("")

        # Clean old containers
        launch_docker_container_text.append(f"# Clean old containers")
        launch_docker_container_text.append(
            "docker ps -a --filter 'name="
            + CONTAINER_NAME
            + "' --format '{{.ID}}' | xargs -I{} docker rm -f {} > /dev/null 2>&1 || true"
        )
        launch_docker_container_text.append(
            "docker ps -a --filter 'name=" + CONTAINER_NAME + "' --format '{{.ID}}' | xargs -I{} docker wait {} || true"
        )
        launch_docker_container_text.append("")

        # Pull new container
        launch_docker_container_text.append(f'docker pull "{image}"')

        # Docker run command
        launch_docker_container_text.extend(
            [
                f"docker run --gpus {get_ntasks_per_node(self.stage_cfg)} \\",
                f'  --privileged --rm -d --name "{CONTAINER_NAME}" \\',
                "  --uts=host --ulimit stack=67108864 --ulimit memlock=-1 --ipc=host --net=host \\",
                "  --security-opt seccomp=unconfined  \\",
            ]
        )

        if get_num_efa_devices(self.instance_type) > 0:
            launch_docker_container_text.append('  "${device[@]}" \\')

        # Handle volume mounting
        mount_str = self._make_container_mounts_string()
        for mount in mount_str.split(","):
            launch_docker_container_text.append(f"  -v {mount} \\")

        # Handle user run args and post run commands
        post_launch_commands = []
        if OmegaConf.select(self.cfg, "cluster.slurm_docker_cfg", default=None) is not None:
            if self.cfg.cluster.slurm_docker_cfg.get("docker_args", None) is not None:
                user_arg = []
                for arg in self.cfg.cluster.slurm_docker_cfg.docker_args:
                    user_arg.append(arg)
                if len(user_arg) > 0:
                    user_arg = " ".join(user_arg)
                    launch_docker_container_text.append(f"  {user_arg} \\")
            if self.cfg.cluster.slurm_docker_cfg.get("post_launch_commands", None) is not None:
                for cmd in self.cfg.cluster.slurm_docker_cfg.post_launch_commands:
                    post_launch_commands.append(cmd)
            if OmegaConf.select(self.cfg, "recipes.model.multi_modal", default=False):
                transformers_upgrade_cmd = "pip install transformers==4.45.2"
                post_launch_commands.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "deepseek_r1":
                transformers_upgrade_cmd = "pip install transformers==4.48.2"
                post_launch_commands.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "llama_v4":
                transformers_upgrade_cmd = "pip install transformers==4.51.3"
                post_launch_commands.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "gpt_oss":
                transformers_upgrade_cmd = "pip install transformers==4.55.0"
                post_launch_commands.append(transformers_upgrade_cmd)

        launch_docker_container_text.append(f'  --entrypoint /bin/bash "{image}" -c "sleep infinity"')
        launch_docker_container_text.append("")

        # Allow containers to talk to each other
        launch_docker_container_text.append(f"# Running post launching commands")
        launch_docker_container_text.extend(
            [
                f'docker exec -itd "{CONTAINER_NAME}" bash -c "printf \\"Port 2022\\n\\" >> /etc/ssh/sshd_config"',
                f'docker exec -itd "{CONTAINER_NAME}" bash -c "printf \\"  Port 2022\\n\\" >> /root/.ssh/config"',
                f'docker exec -itd "{CONTAINER_NAME}" bash -c "service ssh start"',
            ]
        )
        for cmd in post_launch_commands:
            launch_docker_container_text.append(f'docker exec "{CONTAINER_NAME}" bash -c "{cmd}"')
        launch_docker_container_text.append("")

        # Exit
        launch_docker_container_text.append("exit 0")

        return "\n".join(launch_docker_container_text)

    def _make_train_script_text(self, stage_cfg_path=None, port=41000) -> str:
        """
        The custom train entry script, it will be responsible for following
        - Handle resolving hostname and create torch distribtued args
        - Pull from github if required
        - Launch torchrun command
        """
        nodes = get_num_nodes(self.stage_cfg)
        ntasks_per_node = get_ntasks_per_node(self.stage_cfg)
        script_text = ["#!/bin/bash", "set -ex"]

        # Also export env vars here so that they can be consumed by docker container
        env_vars = self.get_env_vars()
        if env_vars:
            script_text.extend([f"export {k}={v}" for k, v in env_vars.items()])

        # Prepare for the host information to create the torchrun command
        if nodes > 1:
            script_text.extend(
                [
                    f"{MASTER_ADDR}=$(head -n 1 {str(self._get_hostfile_location())})",
                    f'{NODEID}=$(($(grep -nx -o "\\b$(hostname)\\b" {str(self._get_hostfile_location())} | cut -d ":" -f 1) - 1))',
                    f"{NNODES}={nodes}",
                    f"{PROCESSES_PER_NODE}={ntasks_per_node}",
                    f"{MASTER_PORT}={port}",
                    "",
                ]
            )
            if self.device == "trainium":
                script_text.append(
                    f'{DISTRIBUTED_ARGS}="--nproc_per_node ${PROCESSES_PER_NODE} --nnodes ${NNODES} --node_rank ${NODEID} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"'
                )
            else:
                script_text.append(
                    f'{DISTRIBUTED_ARGS}="--nproc_per_node ${PROCESSES_PER_NODE} --nnodes ${NNODES} --rdzv_endpoint=${MASTER_ADDR} --rdzv_id=100 --rdzv_backend=c10d"'
                )
        else:
            script_text.append(f'{DISTRIBUTED_ARGS}="--nproc_per_node {ntasks_per_node}"')

        # Prepare github pull
        # Aligns with the train-script preparation in launcher/nemo/k8s_templates/training.yaml
        script_text.append("")
        if self.cfg.get("git", None) is not None or self._default_repo is not None:
            repo_url_or_path = self._default_repo
            branch = self._default_branch
            if self.cfg.get("git", None) is not None:
                if self.cfg.git.get("repo_url_or_path", None) is not None:
                    repo_url_or_path = str(self.cfg.git.get("repo_url_or_path"))
                if self.cfg.git.get("token", None) is not None:
                    repo_url_or_path = self.insert_git_token(repo_url_or_path, self.cfg.git.token)
                if self.cfg.git.get("branch", None) is not None:
                    branch = self.cfg.git.branch

            if repo_url_or_path and not self._use_local_repo():
                # Remote repo, clone the repo url
                script_text.extend(
                    [
                        "# For greater env stability, grab hostname from `hostname`",
                        "# https://sim.amazon.com/issues/P162624109",
                        'LAUNCHER_HOSTNAME="$(hostname)"',
                        "",
                        "mkdir -p $HOME/tmp",
                        'GIT_CLONE_DIR="$HOME/tmp/$LAUNCHER_HOSTNAME"',
                        "[[ -d $GIT_CLONE_DIR ]] && rm -rf $GIT_CLONE_DIR",
                        f"git clone {repo_url_or_path} $GIT_CLONE_DIR",
                        "GIT_CLONE_DIR=${GIT_CLONE_DIR}/",
                        "cd $GIT_CLONE_DIR",
                        # cache can lead to unexpected behavior when user clones
                        # the Adapter and modifies it
                        "rm -rf __pycache__",
                    ]
                )
            else:
                if repo_url_or_path:
                    # simply cd to the directory for local repo
                    script_text.append(f"cd {repo_url_or_path}")

            if branch is not None:
                script_text.append(f"git checkout {branch}")
            if self.cfg.get("git", None) is not None and self.cfg.git.get("commit", None) is not None:
                script_text.append(f"git fetch origin {self.cfg.git.commit}")
                script_text.append(f"git reset --hard {self.cfg.git.commit}")
            if OmegaConf.select(self.cfg, "git.update_adapter", default=False):
                script_text.append("\npip install . --force-reinstall --no-deps")
        else:
            script_text.append('GIT_CLONE_DIR=""')

        if not OmegaConf.select(self.cfg, "training.run.model_type", default="").startswith("neuron"):
            script_text.append("")
            script_text.append("unset SLURM_NTASKS")

        if get_container_type(self.cfg.get("container", None)) == "enroot" and self.cluster == "bcm":
            if OmegaConf.select(self.cfg, "recipes.model.multi_modal", default=False):
                transformers_upgrade_cmd = "pip install transformers==4.45.2"
                script_text.append("")
                script_text.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=False) == "deepseek_r1":
                transformers_upgrade_cmd = "pip install transformers==4.48.2"
                script_text.append("")
                script_text.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "llama_v4":
                transformers_upgrade_cmd = "pip install transformers==4.51.3"
                script_text.append("")
                script_text.append(transformers_upgrade_cmd)
            if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "gpt_oss":
                transformers_upgrade_cmd = "pip install transformers==4.55.0"
                script_text.append("")
                script_text.append(transformers_upgrade_cmd)

        script_text.append("")
        if not self._is_ray_job():
            script_text.append(self._make_custom_call_string(stage_cfg_path))
        else:
            script_text.append("# Ray job execution handled by Ray cluster templates")
        return "\n".join(script_text)

    @staticmethod
    def save_stage_hydra_config(stage_cfg: OmegaConf, job_path: JobPaths, cfg: OmegaConf) -> Path:
        """
        Overriding from Training.save_stage_hydra_config, remove the addition of extra keys in k8s case
        Interpolate and save hydra config file for current stage

        :param OmegaConf stage_cfg: current stage's hydra configuration
        :param JobPaths job_path: JobPaths object
        :param OmegaConf cfg: base config for job
        :return: path current stage's essential nemo scripts code
        :rtype: Path
        """

        _hydra_interpolation(stage_cfg)

        cfg_save_path = job_path.config_file
        omegaconf.OmegaConf.save(stage_cfg, cfg_save_path)
        return cfg_save_path

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """
        Custom run stage which will invoke the entry script only
        [TODO] Make this compatiable with NeMo flow as well
        """
        if get_container_type(self.cfg.get("container", None)) == "docker":
            logger.warning(
                f"![WARNING] You're using docker container directly for slurm workload, we'd highly recommend using enroot instead"
            )
            command_groups = [
                [
                    # Launch container first
                    f"srun -l bash {stage_cfg_path.parents[0]}/launch_docker_container.sh",
                    f"srun -l bash {stage_cfg_path.parents[0]}/docker_exec_script.sh",
                ]
            ]
        # There will be only a single command group
        # enroot or conda/venv, no need to launch docker container
        else:
            command_groups = [[f"bash {stage_cfg_path.parents[0]}/train_script.sh"]]

        return command_groups

    def create_sm_jobs_script(self, job_folder):
        full_recipe_path = Path(job_folder) / "recipe.yaml"
        OmegaConf.save(config=self.cfg.get("training"), f=full_recipe_path)
        sm_jobs_config_path = Path(job_folder) / "sm_jobs_config.yaml"
        sm_jobs_config = self.cfg.cluster.get("sm_jobs_config")

        if not sm_jobs_config:
            raise ValueError("sm_jobs_config is required for sm_jobs cluster type")

        OmegaConf.save(config=sm_jobs_config, f=sm_jobs_config_path)

        # Choose template based on pysdk_launch_executor (estimator or model_trainer)
        pysdk_launch_executor = sm_jobs_config.get("pysdk_launch_executor")
        if pysdk_launch_executor not in ("estimator", "model_trainer"):
            raise ValueError(
                f"Invalid pysdk_launch_executor '{pysdk_launch_executor}'. Valid values: 'estimator', 'model_trainer'"
            )

        if pysdk_launch_executor == "model_trainer":
            script_src = Path(ROOT_DIR) / "template" / "sm_jobs_model_trainer.py"
            logger.info("Using ModelTrainer API for SM Jobs")
        else:
            script_src = Path(ROOT_DIR) / "template" / "sm_jobs_estimator.py"
            logger.info("Using PyTorch Estimator API for SM Jobs")
        script_dst = Path(job_folder) / "launch.py"

        # Generate VERL config file if this is a Ray job
        # For SMTJ: Creates verl-config.yaml in job root (accessible at /opt/ml/code/verl-config.yaml)
        # Contains only VERL-specific configs, filters out platform-specific sections
        if self._is_ray_job():
            self._generate_verl_config_for_sagemaker(Path(job_folder))

        # Always copy the standard launch script
        shutil.copy(script_src, script_dst)

        # FIXME: Remove transformers requirement when container is updated to include the version
        # required to run multi-modal.
        if OmegaConf.select(self.cfg, "recipes.model.multi_modal", default=False):
            reqs_filename = Path(job_folder) / "requirements.txt"
            with open(reqs_filename, "w") as reqs_file:
                reqs_file.write(f"transformers=={TRANSFORMERS_VERSION_FOR_MULTIMODAL}")

    def make_sm_jobs_command(self):
        """
        Make submit command for sm_jobs cluster type.
        """
        instance_type = self.cfg.get("instance_type")
        if instance_type is None:
            raise ValueError("Expected instance_type to be set with sm_jobs cluster type")

        sm_jobs_config = self.cfg.cluster.get("sm_jobs_config")
        if sm_jobs_config is None:
            raise ValueError("Expected sm_jobs_config to be set with sm_jobs cluster type")
        if sm_jobs_config.get("output_path") is None:
            raise ValueError("Expected output_path to be set with sm_jobs cluster type")
        command = f"python launch.py --job_name {self.job_name} --instance_type {instance_type}"
        command_groups = [["pushd $(dirname -- $0)", command, "popd"]]
        return command_groups

    def make_smtj_launch_json(self, job_folder, recipe_template_processor, stage_cfg_path, recipe_file_path):
        """
        Create launch.json file containing the argument values for creating
        a SageMaker PyTorch estimator.
        """
        sm_jobs_config = OmegaConf.load(Path(job_folder) / "sm_jobs_config.yaml")

        # Convert overrides / additional kwargs
        additional_estimator_kwargs = sm_jobs_config.get("additional_estimator_kwargs", {})

        # Read the YAML file and convert it to a properly formatted YAML string
        with open(stage_cfg_path, "r") as f:
            recipe_yaml_content = f.read()

        launch_json = {
            "training_recipe.yaml": recipe_yaml_content,
            "output_path": sm_jobs_config.get("output_path"),
            "launch_overrides": sm_jobs_config.get("recipe_overrides"),
            **additional_estimator_kwargs,
        }
        additional_data = recipe_template_processor.get_additional_data(recipe_file_path)

        # Check if container is available for this platform
        if additional_data is None:
            logger.error("Recipe templatization failed because additional_data was returned as None")
            return

        (
            launch_json["metadata"],
            launch_json["recipe_override_parameters"],
            launch_json["regional_parameters"],
        ) = additional_data

        # Fields to drop for smtj
        override_params_to_be_deleted_in_smtj = ["instance_type", "namespace", "instance_count", "replicas"]
        for key in override_params_to_be_deleted_in_smtj:
            if key in launch_json["recipe_override_parameters"]:
                del launch_json["recipe_override_parameters"][key]

        tensorboard_config = sm_jobs_config.get("tensorboard_config")
        if tensorboard_config:
            launch_json["tensorboard_config"] = OmegaConf.to_container(tensorboard_config, resolve=True)

        launch_json_path = Path(job_folder) / "launch.json"
        with open(launch_json_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=True)
            f.write("\n")

    def run(self) -> str:
        """
        Run current stage
        """
        # Block VERL recipes on slurm - not currently supported
        model_type = OmegaConf.select(self.cfg, "recipes.run.model_type", default=None)
        cluster_type = self.get_cluster_type() if hasattr(self, "get_cluster_type") else self.cluster
        if model_type == "verl" and cluster_type in ("slurm", "bcm"):
            raise ValueError("VERL recipes are not supported on Slurm clusters.")

        # Setup folders and datasets
        self.setup_folder_and_data()
        # Save stage hydra config
        job_path = self.get_job_path()
        # Identify if launching a trainium job
        is_trainium = self.__class__.__name__ == "SMTrainingTrainiumRecipe"

        # Get recipe file path
        recipe_file_path = get_recipe_file_path()

        # Used for templatizing recipes when launch_json is enabled
        recipe_template_processor = None
        if recipe_file_path is not None:
            # Check if this is a Verl or LLMFT recipe
            model_type = OmegaConf.select(self.cfg, "recipes.run.model_type")
            match model_type:
                case "verl":
                    recipe_template_processor = VerlRecipeTemplateProcessor(self.cfg, platform=self.cfg.cluster_type)
                case "llm_finetuning_aws" | "hf":
                    recipe_template_processor = LLMFTRecipeTemplateProcessor(self.cfg, platform=self.cfg.cluster_type)
                case "hyperpod_checkpointless_nemo":
                    recipe_template_processor = CheckpointlessRecipeTemplateProcessor(
                        self.cfg, platform=self.cfg.cluster_type
                    )
                case _:
                    raise ValueError(f"Unsupported model type: {model_type}")

        is_custom = self.cfg.get("training_cfg") is not None
        templatized_recipe = None
        if not is_custom:
            if self._launch_json and recipe_template_processor is not None:
                recipe_file_path = get_recipe_file_path()
                # cfg.recipes and cfg.training are duplicates. For recipe templatization
                # removing cfg.training from cfg
                if self.cfg.get("training", None) != None:
                    training = self.cfg["training"]
                    with open_dict(self.cfg):
                        del self.cfg["training"]
                        # The process_recipe method needs to work on the actual recipe data
                        # We need to temporarily update the staging_cfg in the processor
                        recipe_template_processor.staging_cfg = self.cfg.recipes
                        templatized_recipe = recipe_template_processor.process_recipe(
                            recipe_file_path=recipe_file_path,
                        )
                        # Update self.cfg.recipes with the templatized result
                        self.cfg.recipes = templatized_recipe
                        self.cfg.training = training  # Restore the training reference
                stage_cfg_path = self.save_stage_hydra_config(self.cfg.recipes, job_path, self.cfg)
            else:
                stage_cfg_path = self.save_stage_hydra_config(self.stage_cfg, job_path, self.cfg)
        else:
            stage_cfg_path = job_path.config_file

        if self.cluster == "sm_jobs":
            if is_custom:
                raise RuntimeError("SM jobs launcher is not supported with custom training.")
            cluster_parameters = {"job_name": self.job_name}
            self.create_sm_jobs_script(job_path.folder)
            command_groups = self.make_sm_jobs_command()
            if self._launch_json:
                if recipe_template_processor is not None:
                    self.make_smtj_launch_json(
                        job_path.folder, recipe_template_processor, stage_cfg_path, recipe_file_path
                    )
                return
        else:
            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster)

            cluster_parameters["train_script_text"] = self._make_train_script_text(stage_cfg_path)
            if get_container_type(self.cfg.container) == "docker":
                cluster_parameters["launch_docker_container_text"] = self._make_launch_docker_container_text()
                cluster_parameters["docker_exec_script_text"] = self._make_docker_exec_script_text(stage_cfg_path)
            if get_container_type(self.cfg.container) != "enroot":
                cluster_parameters.pop("container_mounts", None)
            # if self.cfg.get("slurm_create_submission_file_only", None) is not None:
            #     cluster_parameters["slurm_create_submission_file_only"] = self.cfg.slurm_create_submission_file_only
            cluster_parameters["hostfile"] = self._get_hostfile_location()

            if is_trainium and self.get_cluster_type() == "bcm":
                # Save temp training config file with string interpolations resolved so it can be
                # copied into Neuron's package by the compute node(s) eventually selected by Slurm.
                # NOTE: This file can't be removed. Multiple nodes may run the job asynchronously
                # so there aren't any order guarantees nor an ideal moment to remove the file.
                OmegaConf.save(self.cfg.training, self._temp_training_conf_file, True)

            # Make k8s config file if necessary
            if self.cluster == "k8s":
                # The following two methods are overrides from the Training class. They require
                # `template_root` but in our implementation we re-define it inside those methods.
                # Therefore, `template_root` is just a sentinel so parent behavior is not broken.
                sentinel_template_root = ""
                self._make_k8s_spec_file(sentinel_template_root, cluster_parameters, job_path, stage_cfg_path)
                self._copy_k8s_helm_chart(sentinel_template_root, job_path)
                if self._launch_json:
                    if recipe_template_processor is not None:
                        self._create_launch_json(job_path, recipe_template_processor, recipe_file_path)
                    return

                # k8s does not need command groups
                command_groups = None
            else:
                command_groups = self.make_stage_command_groups(stage_cfg_path)

        launcher = SMAutoLauncher(
            folder=job_path.folder,
            cluster=self.cluster,
            **cluster_parameters,
        )
        job_id = launcher.launch(command_groups=command_groups)

        if self.cluster == "bcm":
            try:
                self.telemetry.start(
                    self.cluster,
                    self.instance_type,
                    get_num_nodes(self.stage_cfg),
                    job_id=job_id,
                    container=self.cfg.get("container", None),
                )
            except:
                pass

        return job_id

    def get_cluster_type(self) -> str:
        """
        Get cluster type depending on whether configuration is custom or recipe
        """
        # custom configurations have the `training_cfg` key
        is_custom = self.cfg.get("training_cfg") is not None

        cluster_type = None

        if is_custom:
            cluster_type = OmegaConf.select(self.cfg, "cluster.cluster_type")
        else:
            cluster_type = self.cfg.get("cluster_type")

        if cluster_type is None:
            raise AttributeError("`cluster_type` is not defined in the configuration file")

        return cluster_type

    def get_script_args_str(self, stage_cfg_path: Path) -> str:
        """
        Based on https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L608
        """
        if self.cluster == "k8s":
            return "--config-path=/config --config-name=config.yaml"
        return f"--config-path={stage_cfg_path.parents[0]} --config-name={stage_cfg_path.name}"

    def insert_git_token(self, repo_url_or_path: str, token: str) -> str:
        """
        Insert git token to git repo url. Currently only support github repo
        """
        if "github.com" in repo_url_or_path:
            splitted_url = repo_url_or_path.split("github.com", 1)
            repo_url_or_path = splitted_url[0] + self.cfg.git.token + "@github.com" + splitted_url[1]
        return repo_url_or_path

    def _make_nemo_path_command(self) -> List[str]:
        """Extend nemo path to python path"""
        # [TODO] clone the nemo/SFA/NxTT repo and handle point to the right path
        return super()._make_nemo_path_command()

    def _make_container_mounts_string(self) -> str:
        """
        Make container mounting string based on hydra configurations

        :return: container mounting string, e.g. "/path/to/A:/path/to/A,/path/to/B:/path/to/B,..."
        :rtype: str
        """

        def add_container_mounts(container_mounts):
            mounts_str = ""
            if container_mounts is not None:
                assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
                for mount in container_mounts:
                    if mount is not None and isinstance(mount, str):
                        mounts_str += f",{mount}" if ":" in mount else f",{mount}:{mount}"
            return mounts_str

        cfg = self.cfg
        base_results_dir = cfg.get("base_results_dir")
        mounts_string = (
            f"{self._launcher_scripts_path}:{self._launcher_scripts_path},{base_results_dir}:{base_results_dir}"
        )

        # mount volume only if inside a Hyperpod environment
        hp_logs_dir = "/var/log/aws/clusters"
        if Path(hp_logs_dir).is_dir():
            mounts_string += f",{hp_logs_dir}:{hp_logs_dir}"

        """Start of SM change"""
        container_mounts = cfg.cluster.get("container_mounts")
        """End of SM change"""

        mounts_string += add_container_mounts(container_mounts)

        # https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L264
        # We do not have data dir for custom launching
        mounts_string = mounts_string.replace(",None:None", "")
        if self._use_local_repo():
            mounts_string += f",{self.cfg.git.repo_url_or_path}:{self.cfg.git.repo_url_or_path}"
        return mounts_string

    def generate_default_k8s_value_template(self, template_root, cluster_parameters, stage_cfg_path=None):
        """
        Setting the general k8s configs that will be applicable for all device types and training methods
        """
        with open(template_root / "values.yaml") as value_file:
            values_template = OmegaConf.load(value_file)

        values_template.image.trainingImage = cluster_parameters["container_image"]
        values_template.trainingConfig.jobName = self.stage_cfg.run.name

        # Cluster configs
        values_template.trainingConfig.numEFADevices = self.num_efa_devices
        if "pullPolicy" in cluster_parameters:
            values_template.image.pullPolicy = cluster_parameters["pullPolicy"]
        if "env_vars" in cluster_parameters:
            values_template.trainingConfig.envVars = cluster_parameters["env_vars"]
        if "restartPolicy" in cluster_parameters:
            values_template.trainingConfig.restartPolicy = cluster_parameters["restartPolicy"]
        if "cleanPodPolicy" in cluster_parameters:
            values_template.trainingConfig.cleanPodPolicy = cluster_parameters["cleanPodPolicy"]
        if "persistent_volume_claims" in cluster_parameters:
            values_template.trainingConfig.persistentVolumeClaims = cluster_parameters["persistent_volume_claims"]
        if "volumes" in cluster_parameters:
            values_template.trainingConfig.volumes = cluster_parameters["volumes"]
        if cluster_parameters.get("namespace", None) is not None:
            values_template.trainingConfig.namespace = cluster_parameters["namespace"]
        if cluster_parameters.get("annotations", None) is not None:
            values_template.trainingConfig.annotations = literal_eval(cluster_parameters["annotations"])
        if cluster_parameters.get("priority_class_name", None) is not None:
            values_template.trainingConfig.priorityClassName = cluster_parameters["priority_class_name"]
        if cluster_parameters.get("service_account_name") is not None:
            values_template.trainingConfig.serviceAccountName = cluster_parameters["service_account_name"]
        if cluster_parameters.get("custom_labels", None) is not None:
            values_template.trainingConfig.customLabels = cluster_parameters["custom_labels"]
        if cluster_parameters.get("label_selector", None) is not None:
            values_template.trainingConfig.labelSelector = cluster_parameters["label_selector"]
        if cluster_parameters.get("queue_name", None) is not None:
            values_template.trainingConfig.queue_name = cluster_parameters["queue_name"]

        # Resource configs
        if OmegaConf.select(self.cfg, "recipes.resources.hugepages"):
            values_template.trainingConfig.hugepages = self.cfg.recipes.resources.hugepages
        if OmegaConf.select(self.cfg, "recipes.resources.memory"):
            values_template.trainingConfig.memory = self.cfg.recipes.resources.memory
        values_template.trainingConfig.compile = OmegaConf.select(self.cfg, "recipes.run.compile", default=0)
        if self._default_repo is not None:
            values_template.trainingConfig.git.repo_url_or_path = self._default_repo
        if self._default_branch is not None:
            values_template.trainingConfig.git.branch = self._default_branch

        # Git configs
        if self.cfg.get("git", None) is not None:
            if self.cfg.git.get("repo_url_or_path", None) is not None:
                repo_url_or_path = str(self.cfg.git.repo_url_or_path)
                # We only support to use local repo path for slurm, bcm is nemo launcher version of slurm cluster
                if not (repo_url_or_path.startswith("http") or repo_url_or_path.startswith("codecommit::")):
                    raise ValueError("local git repo path is only supported for slurm based cluster")
                if self.cfg.git.get("token", None) is not None:
                    repo_url_or_path = self.insert_git_token(repo_url_or_path, self.cfg.git.token)

                values_template.trainingConfig.git.repo_url_or_path = repo_url_or_path
            if self.cfg.git.get("branch", None) is not None:
                values_template.trainingConfig.git.branch = self.cfg.git.branch
            if self.cfg.git.get("commit", None) is not None:
                values_template.trainingConfig.git.commit = self.cfg.git.commit
            if self.cfg.git.get("update_adapter", None) is not None:
                values_template.trainingConfig.git.update_adapter = self.cfg.git.update_adapter

        values_template.trainingConfig.device = self.device
        values_template.trainingConfig.scriptArgs = self.get_script_args_str(stage_cfg_path)
        # Only add HyperPod fields if the flag is explicitly set to True
        use_hyperpod_pytorch_job = OmegaConf.select(self.cfg, "cluster.use_hyperpod_pytorch_job", default=None)
        if use_hyperpod_pytorch_job:
            values_template.trainingConfig.useHyperPodPytorchJob = True
            values_template.trainingConfig.instanceType = self.instance_type

        values_template.trainingConfig.pre_script = self.stage_cfg.get("pre_script", [])
        values_template.trainingConfig.post_script = self.stage_cfg.get("post_script", [])
        return values_template

    def _update_ray_specific_values(self, values_template):
        """
        Update Ray-specific k8s values from config files
        Maps values from config.yaml, k8s.yaml, and recipe files to Ray template
        """
        # Map trainer config (from recipe training_config)
        trainer_cfg = OmegaConf.select(self.cfg, "recipes.training_config.trainer")

        if trainer_cfg:
            if trainer_cfg.get("num_nodes"):
                # Ray uses head + workers, so worker replicas = total nodes - 1
                values_template.rayCluster.workerNodes.replicas = max(0, trainer_cfg.num_nodes - 1)
            elif trainer_cfg.get("nnodes"):
                values_template.rayCluster.workerNodes.replicas = max(0, trainer_cfg.nnodes - 1)

            if trainer_cfg.get("n_gpus_per_node"):
                values_template.rayCluster.workerNodes.gpu = str(trainer_cfg.n_gpus_per_node)

        # Map data directories (from recipe training_config)
        data_cfg = OmegaConf.select(self.cfg, "recipes.training_config.data")

        if data_cfg:
            if data_cfg.get("train_files"):
                values_template.trainingConfig.trainDir = data_cfg.train_files
            if data_cfg.get("val_files"):
                values_template.trainingConfig.valDir = data_cfg.val_files

        # Map training type from algorithm config (from recipe training_config)
        algorithm_cfg = OmegaConf.select(self.cfg, "recipes.training_config.algorithm")

        if algorithm_cfg and algorithm_cfg.get("adv_estimator"):
            values_template.rlConfig.trainingType = algorithm_cfg.adv_estimator

        # Map Ray cluster resource config (from recipe if available)
        if OmegaConf.select(self.cfg, "recipes.ray_cluster"):
            ray_cfg = self.cfg.recipes.ray_cluster
            if ray_cfg.get("head_node"):
                for key in ["cpu", "memory", "gpu"]:
                    if ray_cfg.head_node.get(key):
                        setattr(values_template.rayCluster.headNode, key, str(ray_cfg.head_node[key]))

            if ray_cfg.get("worker_nodes"):
                for key in ["replicas", "cpu", "memory", "gpu"]:
                    if ray_cfg.worker_nodes.get(key):
                        setattr(values_template.rayCluster.workerNodes, key, str(ray_cfg.worker_nodes[key]))

        # Handle instance types separately for job submitter vs head/workers
        # Get the instance type from recipe/config
        instance_type = self.cfg.get("instance_type")
        if instance_type is None:
            instance_type = self.cfg.cluster.instance_type

        if instance_type and not instance_type.startswith("ml."):
            instance_type = f"ml.{instance_type}"

        # Set worker instance type (GPU instances from recipe)
        values_template.trainingConfig.workerInstanceType = instance_type

        # Set CPU instance types for head and job submitter
        # Check if user has specified CPU instance types in cluster config
        cpu_instance_type = OmegaConf.select(self.cfg, "cluster.cpu_instance_type")
        if cpu_instance_type:
            # Ensure cpu_instance_type is always a list
            if isinstance(cpu_instance_type, str):
                values_template.trainingConfig.cpuInstanceTypes = [cpu_instance_type]
            else:
                values_template.trainingConfig.cpuInstanceTypes = cpu_instance_type
        else:
            # Use default CPU instance types for head/job submitter
            values_template.trainingConfig.cpuInstanceTypes = [
                # Compute optimized
                "ml.c5.large",
                "ml.c5.xlarge",
                "ml.c5.2xlarge",
                "ml.c5.4xlarge",
                # General purpose
                "ml.m5.large",
                "ml.m5.xlarge",
                "ml.m5.2xlarge",
                "ml.m5.4xlarge",
                # Memory optimized
                "ml.r5.large",
                "ml.r5.xlarge",
                "ml.r5.2xlarge",
            ]

        # Note: Verl Config file will be written in write_value_template method

        return values_template

    def _generate_verl_config_base(self):
        """
        Extract VERL config from the training_config section.

        This method processes the recipe configuration to generate VERL-specific config
        by extracting the training_config section and applying user overrides.

        VERL-specific configs under training_config:
        - actor_rollout_ref: Actor, reference model, and rollout configurations
        - critic: Critic model configuration for value estimation
        - reward_model: Reward model configuration for RLHF
        - algorithm: RL algorithm configuration (GRPO, PPO, etc.)
        - trainer: Training loop configuration
        - data: Dataset and data loading configuration
        - ray_init: Ray initialization parameters
        - custom_reward_function: Custom reward function configuration

        Platform configs (at recipe root level, excluded from VERL config):
        - run: Platform-specific run configuration
        - ray_cluster: K8s-specific Ray cluster specs
        - exp_dir: Platform-specific experiment directory paths
        - sagemaker: SMTJ-specific SageMaker configuration

        Returns:
            Dict: Processed VERL configuration ready for serialization
        """
        recipe_cfg = self.cfg.get("recipes", {})

        # Extract training_config section which contains all VERL configs
        verl_config = recipe_cfg.get("training_config", {})

        # Merge user overrides from rlConfig.verlArgs if present
        # This allows users to override any VERL config at runtime
        rl_config = self.cfg.get("rlConfig", {})
        verl_args_overrides = rl_config.get("verlArgs", {})
        if verl_args_overrides:
            verl_config = OmegaConf.merge(verl_config, verl_args_overrides)

        return verl_config

    def _generate_verl_config(self, job_path):
        """Generate VERL config file for K8s Ray jobs"""
        verl_config = self._generate_verl_config_base()

        # Write config file to k8s_template directory
        config_path = Path(job_path) / "k8s_template" / "verl_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            OmegaConf.save(verl_config, f)

    def write_value_template(self, values_template, job_path):
        """
        Write the value template into disk
        """
        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

        # Generate VERL config file for Ray jobs
        # For K8s: Creates verl_config.yaml in k8s_template/ (mounted via ConfigMap)
        # Contains only VERL-specific configs, filters out platform-specific sections
        if self._is_ray_job():
            self._generate_verl_config(k8s_template_path)

    def update_stage_specific_k8s_values(self, values_template):
        """
        Update the k8s configs that is related to the current stage
        """
        if self._is_ray_job():
            return self._update_ray_specific_values(values_template)

        values_template.trainingConfig.ntasksPerNode = self.stage_cfg.trainer.devices
        values_template.trainingConfig.nodes = self.stage_cfg.trainer.num_nodes
        choice_model_type, _ = self.get_stage_config_choice()
        if self.cfg.git.get("entry_script", None) is not None:
            # Override with entry script provided by the customer
            values_template.trainingConfig.scriptPath = self.cfg.git.entry_script
        else:
            if self._entry_module:
                values_template.trainingConfig.scriptPath = f"-m {self._entry_module}"
            elif self._entry_script_path:
                values_template.trainingConfig.scriptPath = str(self._entry_script_path)
            else:
                raise ValueError("Must set entry_script or entry_module")

        if OmegaConf.select(self.cfg, "recipes.model.multi_modal", default=False):
            transformers_upgrade_cmd = "pip install transformers==4.45.2"
            values_template.trainingConfig.pre_script.append(transformers_upgrade_cmd)
        if OmegaConf.select(self.cfg, "recipes.model.model_type", default=False) == "deepseek_r1":
            transformers_upgrade_cmd = "pip install transformers==4.48.2"
            values_template.trainingConfig.pre_script.append(transformers_upgrade_cmd)
        if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "llama_v4":
            transformers_upgrade_cmd = "pip install transformers==4.51.3"
            values_template.trainingConfig.pre_script.append(transformers_upgrade_cmd)
        if OmegaConf.select(self.cfg, "recipes.model.model_type", default=None) == "gpt_oss":
            transformers_upgrade_cmd = "pip install transformers==4.55.0"
            values_template.trainingConfig.pre_script.append(transformers_upgrade_cmd)

        return values_template

    def _create_launch_json(
        self, job_path: JobPaths, recipe_template_processor: LLMFTRecipeTemplateProcessor, recipe_file_path
    ):
        """
        Run `helm template [NAME] [CHART]` and dumps the output into a launch.json file
        containing the content of the launch artifacts.
        Example structure of launch.json:
        {
            "template_file_1.yaml": "<CONTENT OF template_file_1.yaml>",
            "template_file_2.yaml": "<CONTENT OF template_file_2.yaml>"
        }
        """
        # Render templates into a temp dir
        templates_path = Path(job_path.folder / "k8s_template")
        render_dir = Path(job_path.folder / "k8s_template" / "rendered_templates")
        render_dir.mkdir(parents=True, exist_ok=True)

        logger.info("chart_path: " + str(templates_path))
        # Run helm template and capture output
        cmd = [
            "helm",
            "template",
            self.stage_cfg.run.name,
            str(templates_path),
            "--output-dir",
            str(render_dir),
        ]
        subprocess.run(cmd, check=True)

        # Walk rendered files and dump into JSON
        launch_json = {}
        for path in sorted(render_dir.rglob("*.yaml"), key=lambda p: p.name):
            content = path.read_text()

            # Replace following references in content. String replace is followed here instead of templatization because
            # for these values the helm rendering process fails if we have templates
            job_name_with_hyphens = self.job_name.replace("_", "-")
            content = content.replace(f"name: {self.job_name}", "name: {{name}}")
            content = content.replace(f"name: {job_name_with_hyphens}", "name: {{name}}")
            content = content.replace(f"training-config-{self.job_name}", "training-config-{{name}}")
            content = content.replace(f"training-config-{job_name_with_hyphens}", "training-config-{{name}}")
            content = content.replace(f"app: {self.job_name}", "app: {{name}}")
            content = content.replace(f"app: {job_name_with_hyphens}", "app: {{name}}")
            content = content.replace(f"value: {self.job_name}", "value: {{name}}")
            content = content.replace(f"value: {job_name_with_hyphens}", "value: {{name}}")

            if self.cfg.cluster.get("namespace"):
                namespace_value = self.cfg.cluster["namespace"]
                content = content.replace(f"namespace: {namespace_value}", "namespace: {{namespace}}")

            # Replace container image references with parameterized values
            container_image = self.cfg.get("container", "test_container")
            if container_image:
                content = content.replace(f"image: {container_image}", "image: {{container_image}}")
                content = content.replace(f"{container_image}", "{{container_image}}")

            # Replace instance type references with template variable
            instance_type = self.cfg.get("instance_type", "")
            if instance_type:
                content = content.replace(f'value: "{instance_type}"', 'value: "{{instance_type}}"')
                content = content.replace(f'instance-type: "{instance_type}"', 'instance-type: "{{instance_type}}"')
                # Also replace instance type in node affinity values
                content = content.replace(f'- "{instance_type}"', '- "{{instance_type}}"')

            # Replace K8s node selector instance type pattern using regex
            # This replaces ANY value after beta.kubernetes.io/instance-type: with {{instance_type}}
            # The \S+ pattern matches non-whitespace characters and automatically stops at newline
            content = re.sub(
                r"beta\.kubernetes\.io/instance-type:\s*\S+",
                "beta.kubernetes.io/instance-type: {{instance_type}}",
                content,
            )

            # Replace placeholder values with template variables when launch_json is enabled
            if self._launch_json:
                content = content.replace('value: "PLACEHOLDER_INSTANCE_TYPE"', 'value: "{{instance_type}}"')
                content = content.replace('- "PLACEHOLDER_INSTANCE_TYPE"', '- "{{instance_type}}"')

            launch_json[path.name] = content
            launch_json["training-config.yaml"] = launch_json["training-config.yaml"].replace(
                f"training-config-{self.job_name}", "training-config-{{name}}"
            )

        # following train scripts are moved inside the container. Removing them from launch_json
        if "train-script-gpu.yaml" in launch_json:
            del launch_json["train-script-gpu.yaml"]
        if "train-script-trn.yaml" in launch_json:
            del launch_json["train-script-trn.yaml"]

        additional_data = recipe_template_processor.get_additional_data(recipe_file_path)

        # Check if container is available for this platform
        if additional_data is None:
            logger.error("Recipe templatization failed because additional_data was returned as None")
            return

        (
            launch_json["metadata"],
            launch_json["recipe_override_parameters"],
            launch_json["regional_parameters"],
        ) = additional_data

        # Exclude instance_type overridable for verl templates
        # ToDo: Cleanup conditionals like this
        if "verl-" not in launch_json["metadata"]["Name"]:
            launch_json["recipe_override_parameters"]["instance_type"] = {
                "type": "string",
                "required": False,
                "enum": launch_json["metadata"]["InstanceTypes"],
                "default": launch_json["metadata"]["InstanceTypes"][0],
            }

        launch_path = templates_path / "launch.json"
        with open(launch_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=True)
            f.write("\n")

        logger.info(f"Helm templates dumped into {launch_path}")

    def _is_ray_job(self):
        """
        Detect if this is a Ray job that should use Ray templates.
        Ray can be used for non-RL and RL through a mechanism other than Ray.
        """
        # Check for VERL model type (uses Ray)
        model_type = OmegaConf.select(self.cfg, "recipes.run.model_type")
        verl_model = model_type is not None and str(model_type).lower() == "verl"

        # Check for Ray cluster configuration
        ray_cluster = OmegaConf.select(self.cfg, "recipes.ray_cluster") is not None

        return verl_model and ray_cluster

    # @override - available in Python 3.12 - `template_root` is required by parent implementation
    def _make_k8s_spec_file(
        self, template_root: str, cluster_parameters: Dict, job_path: JobPaths, stage_cfg_path=None
    ):
        """
        Referring from https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L669
        Break the function into 3 parts so we can easily override in different stages
        - Create general k8s configs that will be applicable for all device types and training methods
        - Update stage specific k8s configs
        - Write k8s configs to disk as value.yaml, which will be consumed by helm
        """
        #  Select template based on job type - Ray for RL jobs, training for others
        if self._is_ray_job():
            template_root: Path = ROOT_DIR / "launcher/nemo/k8s_templates/ray"
        else:
            template_root: Path = ROOT_DIR / "launcher/nemo/k8s_templates/training"

        values_template = self.generate_default_k8s_value_template(template_root, cluster_parameters, stage_cfg_path)
        values_template = self.update_stage_specific_k8s_values(values_template)
        self.write_value_template(values_template, job_path)

    # @override - available in Python 3.12 - `template_root` is required by parent implementation
    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        #  Select template directory based on job type - Ray for RL jobs, training for others
        if self._is_ray_job():
            src_template_dir = ROOT_DIR / "launcher/nemo/k8s_templates/ray"
        else:
            src_template_dir = ROOT_DIR / "launcher/nemo/k8s_templates/training"

        super()._copy_k8s_helm_chart(str(src_template_dir), job_path)

    def get_env_vars(self) -> Dict:
        """
        Set up dictionary for environment variables
        By default injecting the EFA env variable when doing multi-node training
        The environment variables from hydra config will be set inside the job scripts.
        For Example:
            Set `env_vars.NVTE_BIAS_DROPOUT_FUSION=1` while calling nemo_launcherlauncher-scripts,
            `NVTE_BIAS_DROPOUT_FUSION=1` will be set while running the job.

        :return: a dictionary of env vars while running the job.
        :rtype: Dict
        """
        env_vars = super().get_env_vars()
        stage_cfg = self.stage_cfg
        nodes = get_num_nodes(stage_cfg)
        if int(nodes) > 1:
            env_vars = set_multinode_envs(env_vars, self.instance_type)
        return env_vars

    def _generate_verl_config_for_sagemaker(self, job_path: Path) -> None:
        """
        Generate VERL config file for SageMaker Ray training jobs.

        For SMTJ, the config file is written to the job root directory as 'verl-config.yaml'
        so it gets packaged with the training job and is accessible in the container at
        /opt/ml/code/verl-config.yaml.
        """
        verl_config = self._generate_verl_config_base()

        # Write config file to SageMaker container path
        config_path = Path(job_path) / "verl-config.yaml"

        # Ensure the directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            OmegaConf.save(verl_config, f)

        logger.info(f"Generated VERL config file at {config_path}")


class SMCustomTraining(SMTraining):
    """
    Base stage for the custom training on Sagemaker.
    """

    @property
    def _entry_script_path(self) -> Path:
        if self.stage_cfg.entry_script:
            return Path(self.stage_cfg.entry_script)
        else:
            return None

    @property
    def _entry_module(self) -> str:
        if self.stage_cfg.get("entry_module", None):
            return self.stage_cfg.entry_module
        else:
            return None

    def setup_stage_vars(self, cfg):
        """Setup the stage vars, i.e. stage name and stage cfg"""
        self.stage_name = "custom_training_sm"
        self.stage_cfg = cfg.get("training_cfg")

    def get_script_args_str(self, stage_cfg_path=None):
        """
        Getting all script args and make it as a str
        """
        arg_str = []
        if self.stage_cfg.get("script_args", None) is not None:
            # script_args will be a list of dict which has key of arg_name and value of arg_value
            for arg in list(self.stage_cfg.script_args):
                for key, val in arg.items():
                    arg_str.append(f"{key} {val} ")
        return "".join(arg_str)

    def update_stage_specific_k8s_values(self, values_template):
        """
        Custom training specifc k8s values
        """
        values_template.trainingConfig.ntasksPerNode = get_ntasks_per_node(self.stage_cfg)
        values_template.trainingConfig.nodes = get_num_nodes(self.stage_cfg)
        values_template.trainingConfig.scriptPath = self.stage_cfg.entry_script
        values_template.trainingConfig.customScript = True
        return values_template

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        #  Need to override the template_root to use our templates
        # [TODO] Currently hard-code it to do the stage as training
        src_training_dir = ROOT_DIR / "launcher/nemo/k8s_templates/training"

        # For custom run, there is no need for training config files
        # Only creating training.yaml, Chart.yaml
        template_file = str(src_training_dir / "training.yaml")
        chart_file = str(src_training_dir / "Chart.yaml")
        training_path = Path(job_path.folder / "k8s_template" / "templates" / "training.yaml")
        training_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path = Path(job_path.folder / "k8s_template" / "Chart.yaml")

        shutil.copy2(template_file, training_path)
        shutil.copy2(chart_file, chart_path)

    def _make_cluster_parameters(self, cluster: str) -> Dict:
        """
        Make a cluster-specific parameters for jobs on different clusters.

        :param str cluster: i.e. `bcm`, `bcp`, `interactive`, etc.
        :return: a dictionary of cluster parameters, e.g. `ntasks_per_node`
        :rtype: Dict
        """
        with omegaconf.open_dict(self.cfg):
            # Patch self.cfg.cluster to align with
            # https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L312
            origin_cluster = self.cfg.cluster
            self.cfg.cluster = self.cfg.cluster.cluster_config
            cluster_parameters = super()._make_cluster_parameters(cluster)
            cluster_type = origin_cluster.get("cluster_type")
            if cluster_type == "k8s":
                env_vars = cluster_parameters.get("env_vars")
                if env_vars and "SLURM_NTASKS_PER_NODE" in env_vars:
                    env_vars.pop("SLURM_NTASKS_PER_NODE")
            self.cfg.cluster = origin_cluster
        return cluster_parameters


class SMCustomTrainingGPU(SMCustomTraining):
    """
    Stage for training with custom stage on GPU
    """

    @property
    def _cuda_visible_devices(self) -> str:
        ntasks_per_node = get_ntasks_per_node(self.stage_cfg)
        if ntasks_per_node is None:
            ntasks_per_node = 8
        return (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
            if ntasks_per_node == 8
            else f"CUDA_VISIBLE_DEVICES={','.join(map(str, range(ntasks_per_node)))}"
        )

    @property
    def _set_ln_sm_margin(self) -> str:
        return ""

    @property
    def _skip_ag_overlap(self) -> str:
        return ""


class SMCustomTrainingCPU(SMCustomTrainingGPU):
    """
    Stage for custom training on CPU
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "cpu"

    @property
    def _cuda_visible_devices(self) -> str:
        return ""


class SMCustomTrainingTrainium(SMCustomTraining):
    """
    Stage for custom training on Trainium
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "trainium"

    def make_stage_command_groups(self, stage_cfg_path: Path) -> List[List[str]]:
        """
        Make the command groups for current stage
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"

        :param Path stage_cfg_path: path to interpolated and saved configuration
        :return: command groups for current stage
        :rtype: List[List[str]]
        """

    def update_stage_specific_k8s_values(self, values_template):
        """
        Custom training specifc k8s values for trainum
        """
        super().update_stage_specific_k8s_values(values_template)
        values_template.trainingConfig.numNeuronDevices = get_num_accelerator_devices(self.instance_type)
        return values_template
