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

from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from ..accelerator_devices import get_num_accelerator_devices
from .constants import (
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
        return SM_ADAPTER_REPO

    @property
    def _entry_script_path(self) -> Path:
        # [TODO] Handle generate the script path from github
        choice_model_type, _ = self.get_stage_config_choice()
        choice_model_type = choice_model_type.split("/")[1]
        # predefined model
        if choice_model_type in SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH:
            return Path(SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH[choice_model_type])
        # custom model
        return Path("examples/custom_model/custom_pretrain.py")

    def get_stage_config_choice(self):
        # [TODO] check if need to override
        return super().get_stage_config_choice()


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
