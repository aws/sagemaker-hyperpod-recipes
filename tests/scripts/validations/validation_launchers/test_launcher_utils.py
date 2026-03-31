"""Unit tests for launcher_utils.select_validation_launcher."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.validations.validation_launchers.launcher_utils import (
    get_job_parameters,
    select_validation_launcher,
)


class TestSelectValidationLauncher(unittest.TestCase):
    """Test suite for select_validation_launcher function."""

    def test_k8_platform(self):
        """Test K8 platform returns K8sValidationLauncher."""
        from scripts.validations.validation_launchers.k8s_launcher import (
            K8sValidationLauncher,
        )

        result = select_validation_launcher("K8")
        self.assertEqual(result, K8sValidationLauncher)

    def test_slurm_platform(self):
        """Test SLURM platform returns SlurmValidationLauncher."""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        result = select_validation_launcher("SLURM")
        self.assertEqual(result, SlurmValidationLauncher)

    def test_smjobs_platform(self):
        """Test SMJOBS platform returns SageMakerJobsValidationLauncher."""
        from scripts.validations.validation_launchers.smjobs_launcher import (
            SageMakerJobsValidationLauncher,
        )

        result = select_validation_launcher("SMJOBS")
        self.assertEqual(result, SageMakerJobsValidationLauncher)

    def test_serverless_platform(self):
        """Test SERVERLESS platform returns ServerlessValidationLauncher."""
        from scripts.validations.validation_launchers.serverless_launcher import (
            ServerlessValidationLauncher,
        )

        result = select_validation_launcher("SERVERLESS")
        self.assertEqual(result, ServerlessValidationLauncher)

    def test_case_insensitive(self):
        """Test that platform names are case-insensitive."""
        from scripts.validations.validation_launchers.k8s_launcher import (
            K8sValidationLauncher,
        )

        result = select_validation_launcher("k8")
        self.assertEqual(result, K8sValidationLauncher)

    def test_unknown_platform_raises_error(self):
        """Test that unknown platform raises ValueError."""
        with self.assertRaises(ValueError) as context:
            select_validation_launcher("UNKNOWN")
        self.assertIn("Unknown platform: UNKNOWN", str(context.exception))


MOCK_DATASET_INFO = {
    "train_data_name": "tatqa_train",
    "train_data_dir": "/fake/train.jsonl",
    "val_data_name": "tatqa_val",
    "val_data_dir": "/fake/val.jsonl",
}


@patch("scripts.validations.validation_launchers.launcher_utils._get_container_path", return_value="fake-container")
@patch(
    "scripts.validations.validation_launchers.launcher_utils._get_dataset_info",
    return_value=MOCK_DATASET_INFO,
)
@patch("scripts.validations.validation_launchers.launcher_utils._get_model_path", return_value="/fake/model/path")
class TestGetJobParametersSleeperPod(unittest.TestCase):
    """Test suite for sleeper pod resolution in get_job_parameters."""

    def _make_cfg(self, platform):
        return OmegaConf.create(
            {
                "platform": platform,
                "instance_type": "ml.p5.48xlarge",
                "hf": {"access_token": ""},
                "git": {"use_default": False},
                "entry_module": "amzn_awsllm_fine_tuning.train_hp",
                "k8": {"sleeper_deployment": "hp-recipe-validator-sleeper"},
                "models": {
                    "default": {
                        "slurm": "/fsx/hf_pretrained_models/",
                        "k8": "/data/hp-recipe-validator/models/",
                        "smjobs": "s3://fake-bucket/models",
                    }
                },
                "recipe_type_config": {
                    "sft": {
                        "detection_keywords": ["lora", "sft", "fft"],
                        "model_config_key": "default",
                        "container_key": "llmft",
                        "recipe_structure": "llmft",
                    }
                },
            }
        )

    @patch("scripts.validations.validation_launchers.launcher_utils._resolve_general_pod_name")
    def test_slurm_does_not_resolve_sleeper_pod(
        self, mock_resolve_pod, mock_model_path, mock_dataset_info, mock_container_path
    ):
        cfg = self._make_cfg("SLURM")
        result = get_job_parameters(
            cfg,
            "fine-tuning/llama/llmft_llama3_3_70b_instruct_seq4k_gpu_sft_fft.yaml",
            "meta-llama/Llama-3.3-70B-Instruct",
        )
        mock_resolve_pod.assert_not_called()
        self.assertEqual(result["k8_general_pod"], "")

    @patch("scripts.validations.validation_launchers.launcher_utils._resolve_general_pod_name")
    def test_smjobs_does_not_resolve_sleeper_pod(
        self, mock_resolve_pod, mock_model_path, mock_dataset_info, mock_container_path
    ):
        cfg = self._make_cfg("SMJOBS")
        result = get_job_parameters(
            cfg,
            "fine-tuning/llama/llmft_llama3_3_70b_instruct_seq4k_gpu_sft_fft.yaml",
            "meta-llama/Llama-3.3-70B-Instruct",
        )
        mock_resolve_pod.assert_not_called()
        self.assertEqual(result["k8_general_pod"], "")

    @patch(
        "scripts.validations.validation_launchers.launcher_utils._resolve_general_pod_name",
        return_value="sleeper-pod-abc123",
    )
    def test_k8_resolves_sleeper_pod(self, mock_resolve_pod, mock_model_path, mock_dataset_info, mock_container_path):
        cfg = self._make_cfg("K8")
        result = get_job_parameters(
            cfg,
            "fine-tuning/llama/llmft_llama3_3_70b_instruct_seq4k_gpu_sft_fft.yaml",
            "meta-llama/Llama-3.3-70B-Instruct",
        )
        mock_resolve_pod.assert_called_once()
        self.assertEqual(result["k8_general_pod"], "sleeper-pod-abc123")
