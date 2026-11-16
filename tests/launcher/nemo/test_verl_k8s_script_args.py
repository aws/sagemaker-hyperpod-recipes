import sys
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from omegaconf import OmegaConf

# Mock nemo_launcher dependencies before importing
sys.modules["nemo_launcher"] = MagicMock()
sys.modules["nemo_launcher.utils"] = MagicMock()
sys.modules["nemo_launcher.utils.job_utils"] = MagicMock()
sys.modules["nemo_launcher.core"] = MagicMock()
sys.modules["nemo_launcher.core.stages"] = MagicMock()

from launcher.nemo.recipe_stages import SMTrainingGPURecipe

TRAINING_CONFIG_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "launcher" / "nemo" / "k8s_templates" / "training" / "training-config.yaml"
)


def _k8s_stage(recipes):
    cfg = OmegaConf.create(
        {
            "instance_type": "p5.48xlarge",
            "cluster": {"instance_type": "p5.48xlarge"},
            "training": {"run": {"name": "test-job"}},
            "recipes": recipes,
        }
    )
    stage = SMTrainingGPURecipe(cfg=cfg)
    stage.cluster = "k8s"
    return stage


def _configmap_keys():
    """Keys of the training ConfigMap, i.e. the file names mounted under /config."""
    # Strip Helm directives so the data section parses as YAML.
    lines = [line for line in TRAINING_CONFIG_TEMPLATE.read_text().splitlines() if "{{" not in line]
    return set(yaml.safe_load("\n".join(lines))["data"])


class TestVerlK8sScriptArgs:
    """Script args the HyperPodPyTorchJob passes to the verl SFT/DPO (non-Ray) entry point on k8s"""

    def test_verl_non_ray_job_names_the_mounted_config_file(self):
        stage = _k8s_stage({"run": {"model_type": "verl"}})

        result = stage.get_script_args_str(Path("/some/path/config.yaml"))

        assert result == "--config-path=/config --config-name=verl_config.yaml"

    def test_verl_non_ray_config_name_matches_configmap_key(self):
        """The config name must be a file the ConfigMap actually mounts, extension included:
        entry scripts like verl_trainer_script.py open config-path/config-name directly."""
        stage = _k8s_stage({"run": {"model_type": "verl"}})

        config_name = stage.get_script_args_str(Path("/some/path/config.yaml")).split("--config-name=")[1]

        assert config_name in _configmap_keys()

    def test_verl_ray_job_keeps_default_args(self):
        stage = _k8s_stage({"run": {"model_type": "verl"}, "ray_cluster": {"head_node": {}}})

        result = stage.get_script_args_str(Path("/some/path/config.yaml"))

        assert result == "--config-path=/config --config-name=config.yaml"

    def test_non_verl_job_keeps_default_args(self):
        stage = _k8s_stage({"run": {"model_type": "llmft"}})

        result = stage.get_script_args_str(Path("/some/path/config.yaml"))

        assert result == "--config-path=/config --config-name=config.yaml"
