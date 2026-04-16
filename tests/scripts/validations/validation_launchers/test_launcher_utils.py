"""Unit tests for launcher_utils.select_validation_launcher."""

import subprocess
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
        mock_resolve_pod.assert_called_once_with(cfg, context_override=None)
        self.assertEqual(result["k8_general_pod"], "sleeper-pod-abc123")

    @patch(
        "scripts.validations.validation_launchers.launcher_utils._resolve_general_pod_name",
        return_value="sleeper-pod-ctx456",
    )
    def test_k8_passes_context_override_to_resolve_pod(
        self, mock_resolve_pod, mock_model_path, mock_dataset_info, mock_container_path
    ):
        cfg = self._make_cfg("K8")
        ctx = ["--context", "arn:aws:eks:us-west-2:123:cluster/test"]
        result = get_job_parameters(
            cfg,
            "fine-tuning/llama/llmft_llama3_3_70b_instruct_seq4k_gpu_sft_fft.yaml",
            "meta-llama/Llama-3.3-70B-Instruct",
            context_override=ctx,
        )
        mock_resolve_pod.assert_called_once_with(cfg, context_override=ctx)
        self.assertEqual(result["k8_general_pod"], "sleeper-pod-ctx456")


class TestGetRunningSleeperPod(unittest.TestCase):
    """Tests for _get_running_sleeper_pod."""

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    def test_returns_pod_name(self, mock_run):
        mock_run.return_value = unittest.mock.Mock(stdout="sleeper-pod-abc\n")
        from scripts.validations.validation_launchers.launcher_utils import (
            _get_running_sleeper_pod,
        )

        result = _get_running_sleeper_pod("my-deployment")
        self.assertEqual(result, "sleeper-pod-abc")

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    def test_returns_empty_when_no_pods(self, mock_run):
        mock_run.return_value = unittest.mock.Mock(stdout="\n")
        from scripts.validations.validation_launchers.launcher_utils import (
            _get_running_sleeper_pod,
        )

        result = _get_running_sleeper_pod("my-deployment")
        self.assertEqual(result, "")

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    def test_context_override_in_command(self, mock_run):
        mock_run.return_value = unittest.mock.Mock(stdout="pod-1\n")
        from scripts.validations.validation_launchers.launcher_utils import (
            _get_running_sleeper_pod,
        )

        _get_running_sleeper_pod("my-dep", context_override=["--context", "my-ctx"])
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[1:3], ["--context", "my-ctx"])

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    def test_no_context_override(self, mock_run):
        mock_run.return_value = unittest.mock.Mock(stdout="pod-1\n")
        from scripts.validations.validation_launchers.launcher_utils import (
            _get_running_sleeper_pod,
        )

        _get_running_sleeper_pod("my-dep")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "kubectl")
        self.assertEqual(cmd[1], "get")


class TestResolveGeneralPodName(unittest.TestCase):
    """Tests for _resolve_general_pod_name with auto-deploy."""

    def _make_cfg(self, sleeper_deployment="hp-recipe-validator-sleeper"):
        return OmegaConf.create({"k8": {"sleeper_deployment": sleeper_deployment}})

    @patch("scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod", return_value="pod-abc")
    def test_returns_existing_pod(self, mock_get_pod):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "pod-abc")
        mock_get_pod.assert_called_once()

    @patch("scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod", return_value="pod-abc")
    def test_passes_context_override(self, mock_get_pod):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        ctx = ["--context", "my-cluster"]
        _resolve_general_pod_name(self._make_cfg(), context_override=ctx)
        mock_get_pod.assert_called_once_with("hp-recipe-validator-sleeper", env=None, context_override=ctx)

    def test_raises_when_no_deployment_configured(self):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        cfg = OmegaConf.create({"k8": {"sleeper_deployment": ""}})
        with self.assertRaises(ValueError):
            _resolve_general_pod_name(cfg)

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    @patch("scripts.validations.validation_launchers.launcher_utils._SLEEPER_YAML")
    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        side_effect=["", "pod-new"],
    )
    def test_auto_deploys_when_no_pod_found(self, mock_get_pod, mock_yaml_path, mock_run):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        mock_yaml_path.exists.return_value = True
        mock_yaml_path.__str__ = lambda self: "/fake/sleeper.yaml"
        mock_run.return_value = unittest.mock.Mock()

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "pod-new")
        # Should have called kubectl apply and kubectl wait
        self.assertEqual(mock_run.call_count, 2)
        apply_cmd = mock_run.call_args_list[0][0][0]
        self.assertIn("apply", apply_cmd)
        wait_cmd = mock_run.call_args_list[1][0][0]
        self.assertIn("wait", wait_cmd)

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    @patch("scripts.validations.validation_launchers.launcher_utils._SLEEPER_YAML")
    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        side_effect=["", ""],
    )
    def test_returns_empty_when_deploy_fails(self, mock_get_pod, mock_yaml_path, mock_run):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        mock_yaml_path.exists.return_value = True
        mock_yaml_path.__str__ = lambda self: "/fake/sleeper.yaml"
        mock_run.return_value = unittest.mock.Mock()

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "")

    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        return_value="",
    )
    @patch("scripts.validations.validation_launchers.launcher_utils._SLEEPER_YAML")
    def test_returns_empty_when_yaml_missing(self, mock_yaml_path, mock_get_pod):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        mock_yaml_path.exists.return_value = False

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "")

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    @patch("scripts.validations.validation_launchers.launcher_utils._SLEEPER_YAML")
    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        side_effect=["", "pod-ctx"],
    )
    def test_auto_deploy_uses_context_override(self, mock_get_pod, mock_yaml_path, mock_run):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        mock_yaml_path.exists.return_value = True
        mock_yaml_path.__str__ = lambda self: "/fake/sleeper.yaml"
        mock_run.return_value = unittest.mock.Mock()

        ctx = ["--context", "my-cluster"]
        result = _resolve_general_pod_name(self._make_cfg(), context_override=ctx)
        self.assertEqual(result, "pod-ctx")
        # Both apply and wait should include context
        for call in mock_run.call_args_list:
            cmd = call[0][0]
            self.assertIn("--context", cmd)

    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        side_effect=subprocess.CalledProcessError(1, "kubectl", stderr="connection refused"),
    )
    def test_returns_empty_on_kubectl_error(self, mock_get_pod):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "")

    @patch("scripts.validations.validation_launchers.launcher_utils.subprocess.run")
    @patch("scripts.validations.validation_launchers.launcher_utils._SLEEPER_YAML")
    @patch(
        "scripts.validations.validation_launchers.launcher_utils._get_running_sleeper_pod",
        return_value="",
    )
    def test_returns_empty_on_apply_error(self, mock_get_pod, mock_yaml_path, mock_run):
        from scripts.validations.validation_launchers.launcher_utils import (
            _resolve_general_pod_name,
        )

        mock_yaml_path.exists.return_value = True
        mock_yaml_path.__str__ = lambda self: "/fake/sleeper.yaml"
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl apply", stderr="forbidden")

        result = _resolve_general_pod_name(self._make_cfg())
        self.assertEqual(result, "")


class TestConstructK8LaunchCommandKubeContext(unittest.TestCase):
    """Tests for kube_context in construct_k8_launch_command."""

    @patch("scripts.validations.validation_launchers.launcher_utils.get_launch_command", return_value=["base_cmd"])
    def test_appends_kube_context(self, mock_get_cmd):
        from scripts.validations.validation_launchers.launcher_utils import (
            construct_k8_launch_command,
        )

        cfg = unittest.mock.Mock()
        run_info = {"k8_general_pod": "pod-1", "kube_context": "my-cluster-ctx"}
        result = construct_k8_launch_command(cfg, run_info)
        self.assertIn("+cluster.kube_context=my-cluster-ctx", result)

    @patch("scripts.validations.validation_launchers.launcher_utils.get_launch_command", return_value=["base_cmd"])
    def test_no_kube_context_when_absent(self, mock_get_cmd):
        from scripts.validations.validation_launchers.launcher_utils import (
            construct_k8_launch_command,
        )

        cfg = unittest.mock.Mock()
        run_info = {"k8_general_pod": "pod-1"}
        result = construct_k8_launch_command(cfg, run_info)
        self.assertFalse(any("kube_context" in item for item in result))

    @patch("scripts.validations.validation_launchers.launcher_utils.get_launch_command", return_value=["base_cmd"])
    def test_no_kube_context_when_none(self, mock_get_cmd):
        from scripts.validations.validation_launchers.launcher_utils import (
            construct_k8_launch_command,
        )

        cfg = unittest.mock.Mock()
        run_info = {"k8_general_pod": "pod-1", "kube_context": None}
        result = construct_k8_launch_command(cfg, run_info)
        self.assertFalse(any("kube_context" in item for item in result))
