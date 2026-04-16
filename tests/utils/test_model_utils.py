import subprocess
from unittest.mock import Mock, patch

from utils.model_utils import copy_script_to_pod, download_model, download_model_on_k8s


def _make_cfg(cluster_type, kube_context=None, run_name="llama-3-1-8b-instruct"):
    cfg = Mock()
    cfg.get = lambda key, default=None: {
        "training_config": "some_recipe",
        "local_model_name_or_path": "/data/models/meta-llama/Llama-3.1-8B-Instruct",
    }.get(key, default)
    cfg.cluster_type = cluster_type
    cfg.cluster.general_pod = "sleeper-pod-abc"
    cfg.cluster.get = lambda key, default=None: kube_context if key == "kube_context" else default
    cfg.__getitem__ = lambda self, key: {"recipes": cfg.recipes}[key]
    cfg.recipes.run.get = lambda key, default=None: run_name if key == "name" else default
    cfg.recipes.__getitem__ = lambda self, key: {"run": cfg.recipes.run}[key]
    cfg.recipes.run.__getitem__ = lambda self, key: {"name": run_name}[key]
    return cfg


class TestDownloadModelKubeContext:
    @patch("utils.model_utils.download_model_on_k8s")
    @patch("utils.model_utils.get_jumpstart_model_id", return_value="meta-textgeneration-llama-3-1-8b-instruct")
    def test_k8s_passes_kube_context(self, mock_js_id, mock_download):
        cfg = _make_cfg("k8s", kube_context="arn:aws:eks:us-west-2:123:cluster/test")
        download_model(cfg)
        mock_download.assert_called_once_with(
            "sleeper-pod-abc",
            "meta-textgeneration-llama-3-1-8b-instruct",
            "/data/models/meta-llama/Llama-3.1-8B-Instruct",
            kube_context="arn:aws:eks:us-west-2:123:cluster/test",
        )

    @patch("utils.model_utils.download_model_on_k8s")
    @patch("utils.model_utils.get_jumpstart_model_id", return_value="meta-textgeneration-llama-3-1-8b-instruct")
    def test_k8s_passes_none_context_when_not_set(self, mock_js_id, mock_download):
        cfg = _make_cfg("k8s")
        download_model(cfg)
        mock_download.assert_called_once_with(
            "sleeper-pod-abc",
            "meta-textgeneration-llama-3-1-8b-instruct",
            "/data/models/meta-llama/Llama-3.1-8B-Instruct",
            kube_context=None,
        )

    @patch("utils.model_utils.download_model_on_k8s")
    @patch("utils.model_utils.get_jumpstart_model_id", return_value=None)
    def test_k8s_skips_download_when_no_jumpstart_id(self, mock_js_id, mock_download):
        cfg = _make_cfg("k8s")
        download_model(cfg)
        mock_download.assert_not_called()


class TestDownloadModelOnK8s:
    @patch("utils.model_utils.subprocess.run")
    @patch("utils.model_utils.copy_script_to_pod")
    def test_passes_kube_context_to_kubectl(self, mock_copy, mock_run):
        mock_run.return_value = Mock(returncode=0)
        download_model_on_k8s("pod-1", "model-id", "/save/path", kube_context="my-context")
        cmd = mock_run.call_args[0][0]
        assert "--context" in cmd
        assert "my-context" in cmd
        mock_copy.assert_called_once_with(
            "pod-1", "/data/hp-recipe-validator/k8s_download_model.py", kube_context="my-context"
        )

    @patch("utils.model_utils.subprocess.run")
    @patch("utils.model_utils.copy_script_to_pod")
    def test_no_context_flag_when_none(self, mock_copy, mock_run):
        mock_run.return_value = Mock(returncode=0)
        download_model_on_k8s("pod-1", "model-id", "/save/path")
        cmd = mock_run.call_args[0][0]
        assert "--context" not in cmd

    @patch("utils.model_utils.subprocess.run")
    @patch("utils.model_utils.copy_script_to_pod")
    def test_returns_false_on_subprocess_failure(self, mock_copy, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl", stderr="access denied")
        result = download_model_on_k8s("pod-1", "model-id", "/save/path")
        assert result is False

    @patch("utils.model_utils.subprocess.run")
    @patch("utils.model_utils.copy_script_to_pod")
    def test_returns_false_on_copy_failure(self, mock_copy, mock_run):
        mock_copy.side_effect = subprocess.CalledProcessError(1, "kubectl cp", stderr="tar not found")
        result = download_model_on_k8s("pod-1", "model-id", "/save/path")
        assert result is False


class TestCopyScriptToPod:
    @patch("utils.model_utils.subprocess.run")
    def test_passes_kube_context(self, mock_run):
        copy_script_to_pod("pod-1", "/remote/path", kube_context="my-context")
        cmd = mock_run.call_args[0][0]
        assert cmd[1:3] == ["--context", "my-context"]

    @patch("utils.model_utils.subprocess.run")
    def test_no_context_when_none(self, mock_run):
        copy_script_to_pod("pod-1", "/remote/path")
        cmd = mock_run.call_args[0][0]
        assert "--context" not in cmd


class TestDownloadModelVerl:
    @patch("utils.model_utils.download_model_on_k8s")
    @patch("utils.model_utils.get_jumpstart_model_id", return_value="some-jumpstart-id")
    def test_verl_recipe_downloads_model(self, mock_js_id, mock_download):
        cfg = _make_cfg("k8s", run_name="verl-grpo-deepseek-r1")
        download_model(cfg)
        mock_download.assert_called_once()

    @patch("utils.model_utils.download_model_on_k8s")
    @patch("utils.model_utils.get_jumpstart_model_id", return_value="some-jumpstart-id")
    def test_verl_sft_recipe_downloads_model(self, mock_js_id, mock_download):
        cfg = _make_cfg("k8s", run_name="verl-sft-qwen-3-8b-lora")
        download_model(cfg)
        mock_download.assert_called_once()
