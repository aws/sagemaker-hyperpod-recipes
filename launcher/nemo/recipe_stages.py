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

import shutil
from pathlib import Path
from typing import Dict, List

import omegaconf
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf

from ..accelerator_devices import get_num_accelerator_devices
from .constants import (
    HPCT_ENV_VARS,
    HPCT_MODEL_TASK_TO_CODE_PATH,
    NEMO_REPO,
    NEMO_REPO_TAG,
    NEURONX_CONF_PATH,
    NEURONX_REPO_TAG,
    NEURONX_REPO_URI,
    ROOT_DIR,
    SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH,
    SM_ADAPTER_REPO,
)
from .stages import SMTraining, get_num_nodes, set_multinode_envs


class SMTrainingGPURecipe(SMTraining):
    """
    Stage used to run our GPU recipes
    """

    @property
    def _default_repo(self):
        use_default = self.cfg.get("git", {}).get("use_default", True)
        return SM_ADAPTER_REPO if use_default else None

    @property
    def _entry_script_path(self) -> Path:
        # Use Git entry_script config if available
        cfg_git_entry_script = self.cfg.get("git", {}).get("entry_script", None)
        if cfg_git_entry_script != None:
            return cfg_git_entry_script

        # [TODO] Handle generate the script path from github
        choice_model_type, _ = self.get_stage_config_choice()
        choice_model_type = choice_model_type.split("/")[1]
        # predefined model
        if choice_model_type in SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH:
            return Path(SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH[choice_model_type])

        # custom model
        return Path("examples/custom_model/custom_pretrain.py")

    @property
    def _entry_module(self):
        if self.cfg.get("entry_module", None):
            return self.cfg.entry_module
        return None

    def get_stage_config_choice(self):
        # [TODO] check if need to override
        return super().get_stage_config_choice()


class SMTrainingGPURecipeElastic(SMTrainingGPURecipe):
    """
    Stage used to run elastic training jobs
    """

    @staticmethod
    def save_stage_hydra_config(stage_cfg: OmegaConf, job_path: JobPaths, cfg: OmegaConf) -> Path:
        default_cfg_path = SMTrainingGPURecipe.save_stage_hydra_config(stage_cfg, job_path, cfg)

        remove_keys = {"scale_config", "elastic_policy"}
        basic_config = {key: value for key, value in stage_cfg.items() if key not in remove_keys}
        basic_config = OmegaConf.create(basic_config)

        scale_configs = stage_cfg.get("scale_config", {})
        scale_space = scale_configs.keys()

        for nnodes in scale_space:
            new_stage_cfg = OmegaConf.merge(basic_config, scale_configs[nnodes])
            cfg_save_path = job_path.folder / f"train_config_n{nnodes}.yaml"
            omegaconf.OmegaConf.save(new_stage_cfg, cfg_save_path)

        return default_cfg_path

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        super()._copy_k8s_helm_chart(template_root, job_path)

        train_config_paths = job_path.folder.glob("train_config_n*.yaml")
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")
        for path in train_config_paths:
            shutil.copy(path, hydra_config_path)

    def generate_default_k8s_value_template(self, template_root, cluster_parameters, stage_cfg_path=None):
        values_template = super().generate_default_k8s_value_template(template_root, cluster_parameters, stage_cfg_path)

        elastic_policy = self.cfg.training.elastic_policy
        scale_config = self.cfg.training.get("scale_config", None)

        values_template.trainingConfig.elastic_policy.is_elastic = elastic_policy.is_elastic
        values_template.trainingConfig.elastic_policy.min_nodes = elastic_policy.min_nodes
        values_template.trainingConfig.elastic_policy.max_nodes = elastic_policy.max_nodes

        if scale_config:
            scale_space = list(scale_config.keys())
            values_template.trainingConfig.elastic_policy.replica_space = scale_space
        else:
            assert (
                elastic_policy.get("replica_increment_step", None) is not None
            ), "Either scale_config or replica_increment_step need to be defined"
            values_template.trainingConfig.elastic_policy.replica_increment_step = elastic_policy.replica_increment_step

        use_graceful_shutdown = elastic_policy.get("use_graceful_shutdown", True)
        values_template.trainingConfig.elastic_policy.use_graceful_shutdown = use_graceful_shutdown
        if use_graceful_shutdown:
            values_template.trainingConfig.envVars.HYPERPOD_SIGNAL_COORDINATION = "distributed"

        if elastic_policy.get("scaling_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.scaling_timeout = elastic_policy.scaling_timeout
        if elastic_policy.get("graceful_shutdown_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.graceful_shutdown_timeout = (
                elastic_policy.graceful_shutdown_timeout
            )
        if elastic_policy.get("faulty_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.faulty_timeout = elastic_policy.faulty_timeout

        return values_template

    def get_script_args_str(self, stage_cfg_path: Path) -> str:
        """
        Based on https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L608
        """
        if self.cluster == "k8s":
            return f"--config-path=/config"
        return f"--config-path={stage_cfg_path.parents[0]} --config-name={stage_cfg_path.name}"


class NeMoTraining(SMTraining):
    """
    Stage to run NeMo recipes
    """

    @property
    def _nemo_code_path(self) -> Path:
        return Path("")

    @property
    def _default_repo(self):
        return NEMO_REPO

    @property
    def _default_branch(self):
        return NEMO_REPO_TAG

    @property
    def _entry_script_path(self) -> Path:
        choice_model_type, _ = self.get_stage_config_choice()
        choice_model_type = choice_model_type.split("/")[1]
        code_path = self._get_nemo_code_path(choice_model_type)
        return Path(code_path)

    @property
    def _entry_module(self):
        if self.cfg.get("entry_module", None):
            return self.cfg.entry_module
        return None


class SMTrainingTrainiumRecipe(SMTraining):
    """
    Stage to run our Trainium recipes
    """

    DEFAULT_TRAIN_SCRIPT_PATH = "examples/train.sh"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "trainium"

        # Used by Slurm and K8s. Example: "llama/megatron_llama_7B_config"
        self._training_filename = self.cfg.training_config.rsplit("/", 1)[-1]
        self._temp_training_conf_file = ROOT_DIR / f"tmp/training/{self._training_filename}.yaml"

        if not self._temp_training_conf_file.parent.exists():
            self._temp_training_conf_file.parent.mkdir(parents=True)

    @property
    def _default_repo(self):
        return NEURONX_REPO_URI

    @property
    def _default_branch(self):
        return NEURONX_REPO_TAG

    @property
    def _entry_script_path(self) -> Path:
        cfg_git_entry_script = self.cfg.get("git", {}).get("entry_script")
        entry_script_path = cfg_git_entry_script or self.DEFAULT_TRAIN_SCRIPT_PATH
        return Path(entry_script_path)

    def _make_custom_call_string(self, stage_cfg_path=None):
        """
        Create the command that runs the training script
        """
        compile = OmegaConf.select(self.cfg, "recipes.run.compile", default=0)

        commands: List[str] = [
            "# copy the resolved training config file into the cloned Neuronx repo",
            f"cp -f {self._temp_training_conf_file} {NEURONX_CONF_PATH}",
            "",
            "# training script depends on other files invoked with relative paths, so must cd into it",
            f'cd "$(dirname {self._entry_script_path})"',
            "",
            "# run training script but first define its arguments",
            f"export CONF_FILE={self._training_filename}",
            f"export COMPILE={compile}",
            f'bash ./"$(basename {self._entry_script_path})"',
            "",
        ]
        return "\n".join(commands)

    def update_stage_specific_k8s_values(self, values_template):
        """
        training specifc k8s values for trainum
        """
        super().update_stage_specific_k8s_values(values_template)
        values_template.trainingConfig.numNeuronDevices = get_num_accelerator_devices(self.instance_type)
        return values_template

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


class SMTrainingHPCTRecipe(SMTraining):
    """
    Stage used to run HyperPod Checkpoint-less Training recipes
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        cluster_type = cfg.get("cluster_type")
        if cluster_type != "k8s":
            raise ValueError(
                f"HyperPod checkpointless training recipes only support K8s cluster type, got: {cluster_type}"
            )

    @property
    def _default_repo(self):
        return None  # No repo needed, scripts are in container

    @property
    def _entry_module(self):
        return None

    @property
    def _entry_script_path(self) -> Path:
        training_config = self.cfg.get("training_config")

        # Extract model name from path: training/llama/... -> llama
        path_parts = training_config.split("/")
        model_name = path_parts[1]
        if "fine_tuning" in training_config.lower():
            task = "fine_tuning"
        elif "lora" in training_config.lower():
            task = "lora"
        elif "pretrain" in training_config.lower():
            task = "pretrain"
        else:
            raise ValueError(f"Cannot determine task type from training config: {training_config}")

        model_task_key = f"{model_name}_{task}"

        if model_task_key not in HPCT_MODEL_TASK_TO_CODE_PATH:
            supported_types = list(HPCT_MODEL_TASK_TO_CODE_PATH.keys())
            raise ValueError(
                f"HyperPod checkpointless model-task combination '{model_task_key}' is not supported. "
                f"Supported types: {supported_types}"
            )

        entry_script_path = HPCT_MODEL_TASK_TO_CODE_PATH[model_task_key]
        return Path(entry_script_path)

    def get_env_vars(self) -> Dict:
        """
        Set up HCT-specific environment variables
        Handle all logic without calling parent to avoid conflicts
        """
        return HPCT_ENV_VARS.copy()
