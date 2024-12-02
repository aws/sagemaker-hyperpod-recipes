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


import logging
import shutil
from pathlib import Path
from typing import Dict, List

import omegaconf
from nemo_launcher.core.stages import Training, _hydra_interpolation
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf

from ..accelerator_devices import get_num_accelerator_devices
from ..efa import (
    efa_supported_instance,
    instanceWithMultipleEFAs,
    instanceWithRDMASupport,
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


def set_multinode_envs(env_vars, instance_type):
    # https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md
    if get_num_efa_devices(instance_type) > 0:
        env_vars["FI_PROVIDER"] = "efa"
    env_vars["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
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
        nodes = stage_cfg.get("trainer").get("num_nodes")
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
        script_path = str(self._entry_script_path)
        torchrun_cmd = self._make_torchrun_string()
        script_args_str = self.get_script_args_str(stage_cfg_path)
        command = [torchrun_cmd, script_path, script_args_str]
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
            if self.cfg.recipes.get("model", None) and self.cfg.recipes.model.get("multi_modal", False):
                transformers_upgrade_cmd = "pip install transformers==4.45.2"
                post_launch_commands.append(transformers_upgrade_cmd)

        launch_docker_container_text.append(f'  "{image}" sleep infinity')
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
                assert repo_url_or_path is not None, "`repo_url_or_path` must be defined when setting git config"
                if self.cfg.git.get("token", None) is not None:
                    repo_url_or_path = self.insert_git_token(repo_url_or_path, self.cfg.git.token)
                if self.cfg.git.get("branch", None) is not None:
                    branch = self.cfg.git.branch

            if not self._use_local_repo():
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
                    ]
                )
            else:
                # simply cd to the directory for local repo
                script_text.append(f"cd {repo_url_or_path}")

            if branch is not None:
                script_text.append(f"git checkout {branch}")
            if self.cfg.get("git", None) is not None and self.cfg.git.get("commit", None) is not None:
                script_text.append(f"git fetch origin {self.cfg.git.commit}")
                script_text.append(f"git reset --hard {self.cfg.git.commit}")
        else:
            script_text.append('GIT_CLONE_DIR=""')

        if not OmegaConf.select(self.cfg, "training.run.model_type", default="").startswith("neuron"):
            script_text.append("")
            script_text.append("unset SLURM_NTASKS")

        script_text.append("")
        script_text.append(self._make_custom_call_string(stage_cfg_path))
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
        OmegaConf.save(config=self.cfg.cluster.get("sm_jobs_config"), f=sm_jobs_config_path)
        script_src = Path(ROOT_DIR) / "template" / "sm_jobs.py"
        script_dst = Path(job_folder) / "launch.py"
        shutil.copy(script_src, script_dst)
        # FIXME: Remove transformers requirement when container is updated to include the version
        # required to run multi-modal.
        if self.cfg.recipes.get("model", None) and self.cfg.recipes.model.get("multi_modal", False):
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

    def run(self) -> str:
        """
        Run current stage
        """
        # Setup folders and datasets
        self.setup_folder_and_data()
        # Save stage hydra config
        job_path = self.get_job_path()
        # Identify if launching a trainium job
        is_trainium = self.__class__.__name__ == "SMTrainingTrainiumRecipe"

        is_custom = self.cfg.get("training_cfg") is not None
        if not is_custom:
            stage_cfg_path = SMTraining.save_stage_hydra_config(self.stage_cfg, job_path, self.cfg)
        else:
            stage_cfg_path = job_path.config_file

        if self.cluster == "sm_jobs":
            if is_custom:
                raise RuntimeError("SM jobs launcher is not supported with custom training.")
            cluster_parameters = {"job_name": self.job_name}
            self.create_sm_jobs_script(job_path.folder)
            command_groups = self.make_sm_jobs_command()
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
        if "persistent_volume_claims" in cluster_parameters:
            values_template.trainingConfig.persistentVolumeClaims = cluster_parameters["persistent_volume_claims"]
        if "volumes" in cluster_parameters:
            values_template.trainingConfig.volumes = cluster_parameters["volumes"]
        if cluster_parameters.get("namespace", None) is not None:
            values_template.trainingConfig.namespace = cluster_parameters["namespace"]
        if cluster_parameters.get("annotations", None) is not None:
            values_template.trainingConfig.annotations = cluster_parameters["annotations"]
        if cluster_parameters.get("priority_class_name", None) is not None:
            values_template.trainingConfig.priorityClassName = cluster_parameters["priority_class_name"]
        if cluster_parameters.get("service_account_name") is not None:
            values_template.trainingConfig.serviceAccountName = cluster_parameters["service_account_name"]
        if cluster_parameters.get("custom_labels", None) is not None:
            values_template.trainingConfig.customLabels = cluster_parameters["custom_labels"]
        if cluster_parameters.get("label_selector", None) is not None:
            values_template.trainingConfig.labelSelector = cluster_parameters["label_selector"]
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

        values_template.trainingConfig.device = self.device
        values_template.trainingConfig.scriptArgs = self.get_script_args_str(stage_cfg_path)
        return values_template

    def write_value_template(self, values_template, job_path):
        """
        Write the value template into disk
        """
        k8s_template_path = job_path.folder
        k8s_template_file = Path(k8s_template_path / "k8s_template" / "values.yaml")
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        OmegaConf.save(conf, k8s_template_file)

    def update_stage_specific_k8s_values(self, values_template):
        """
        Update the k8s configs that is related to the current stage
        """
        values_template.trainingConfig.ntasksPerNode = self.stage_cfg.trainer.devices
        values_template.trainingConfig.nodes = self.stage_cfg.trainer.num_nodes
        choice_model_type, _ = self.get_stage_config_choice()
        if self.cfg.git.get("entry_script", None) is not None:
            # Override with entry script provided by the customer
            values_template.trainingConfig.scriptPath = self.cfg.git.entry_script
        else:
            values_template.trainingConfig.scriptPath = str(self._entry_script_path)

        if self.cfg.recipes.get("model", None) and self.cfg.recipes.model.get("multi_modal", False):
            transformers_upgrade_cmd = "pip install transformers==4.45.2"
            values_template.trainingConfig.pre_script.append(transformers_upgrade_cmd)

        return values_template

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
        #  Need to override the template_root to use our templates
        # [TODO] Currently hard-code it to do the stage as training
        template_root: Path = ROOT_DIR / "launcher/nemo/k8s_templates/training"
        values_template = self.generate_default_k8s_value_template(template_root, cluster_parameters, stage_cfg_path)
        values_template = self.update_stage_specific_k8s_values(values_template)
        self.write_value_template(values_template, job_path)

    def _copy_k8s_helm_helper_configs(self, src_training_dir: Path, job_path: JobPaths):
        """
        Copy helper Helm files into results directory
        """
        # copy the Trainium and GPU config files
        gpu_config = "train-script-gpu.yaml"
        trn_config = "train-script-trn.yaml"
        templates_path = Path(job_path.folder / "k8s_template" / "templates")
        shutil.copy2(str(src_training_dir / gpu_config), str(templates_path / gpu_config))
        shutil.copy2(str(src_training_dir / trn_config), str(templates_path / trn_config))

    # @override - available in Python 3.12 - `template_root` is required by parent implementation
    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        #  Need to override the template_root to use our templates
        # [TODO] Currently hard-code it to do the stage as training
        src_training_dir = ROOT_DIR / "launcher/nemo/k8s_templates/training"
        super()._copy_k8s_helm_chart(str(src_training_dir), job_path)
        self._copy_k8s_helm_helper_configs(src_training_dir, job_path)

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


class SMCustomTraining(SMTraining):
    """
    Base stage for the custom training on Sagemaker.
    """

    @property
    def _entry_script_path(self) -> Path:
        return Path(self.stage_cfg.entry_script)

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
        self._copy_k8s_helm_helper_configs(src_training_dir, job_path)

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
