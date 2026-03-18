import subprocess
from unittest.mock import Mock, mock_open, patch

import pytest

from scripts.validations.validation_launchers.k8s_launcher import (
    HP_PYTORCH_JOB_RESOURCE_NAME,
    K8sValidationLauncher,
)


@pytest.fixture
def mock_config():
    config = Mock()
    config.base_results_dir = "/tmp/results"
    config.platform = "K8"
    config.assume_role_arn = None
    config.get = Mock(side_effect=lambda key, default=None: "/tmp/results" if key == "base_results_dir" else default)
    return config


@pytest.fixture
def mock_job_recorder():
    return Mock()


@pytest.fixture
def launcher(mock_config, mock_job_recorder):
    return K8sValidationLauncher(mock_job_recorder, mock_config)


class TestParseOutput:
    def test_parse_output_pytorch_job(self, launcher):
        launch_output = Mock()
        launch_output.stdout = """
submission file created at /tmp/output
NAME: test-job-123
"""
        with patch.object(launcher, "get_pods", return_value=["test-job-123-worker-0"]), patch.object(
            launcher, "get_head_node", return_value="test-job-123-worker-0"
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock()

            result = launcher._parse_output("test_recipe.yaml", launch_output)

            assert result["job_name"] == "test-job-123"
            assert result["pod_name"] == "test-job-123-worker-0"
            assert result["output_folder_path"] == "/tmp/output"

    def test_parse_output_verl_job(self, launcher):
        launch_output = Mock()
        launch_output.stdout = """
submission file created at /tmp/verl_output
NAME: verl-job-456
"""
        with patch.object(launcher, "get_pods", return_value=["verl-job-456"]), patch.object(
            launcher, "get_head_node", return_value="verl-job-456"
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock()

            result = launcher._parse_output("verl_recipe.yaml", launch_output)

            assert result["job_name"] == "verl-job-456"
            assert result["pod_name"] == "verl-job-456"
            assert result["output_folder_path"] == "/tmp/verl_output"

    def test_parse_output_missing_fields(self, launcher, mock_job_recorder):
        launch_output = Mock()
        launch_output.stdout = "incomplete output"

        result = launcher._parse_output("test_recipe.yaml", launch_output)

        assert result is None
        mock_job_recorder.update_job.assert_called_once()

    def test_parse_output_kubectl_wait_fails(self, launcher, mock_job_recorder):
        launch_output = Mock()
        launch_output.stdout = """
submission file created at /tmp/output
NAME: test-job-123
"""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, "cmd", stderr="timeout"),
                Mock(stdout="job details"),  # describe_resource call
                Mock(),  # clean_up_resource call
            ]

            with pytest.raises(subprocess.CalledProcessError):
                launcher._parse_output("test_recipe.yaml", launch_output)


class TestMonitorJobPyTorch:
    """Tests for _monitor_job with PyTorch jobs"""

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_success(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "output_folder_path": "/tmp/output",
        }
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="Completed"), patch.object(
            launcher, "clean_up_resource"
        ), patch("os.path.exists", return_value=True), patch(
            "builtins.open", mock_open(read_data="logs")
        ), patch.object(
            launcher, "calculate_throughput_from_logs", return_value=100.5
        ):
            result = launcher._monitor_job("test_recipe.yaml", job_details)

            assert result is True
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Complete"
            assert mock_job_recorder.update_job.call_args[1]["tokens_throughput"] == 100.5

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_failure(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "output_folder_path": "/tmp/output",
        }
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="Failed"), patch.object(
            launcher, "clean_up_resource"
        ), patch("os.path.exists", return_value=False):
            result = launcher._monitor_job("test_recipe.yaml", job_details)

            assert result is False
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Failed"

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_exception(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "output_folder_path": "/tmp/output",
        }
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(
            launcher, "_get_job_status", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error")
        ):
            result = launcher._monitor_job("test_recipe.yaml", job_details)

            assert result is False


class TestMonitorJobVERL:
    """Tests for _monitor_job with VERL jobs"""

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_success(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="SUCCEEDED"), patch.object(
            launcher, "clean_up_resource"
        ):
            result = launcher._monitor_job("verl_recipe.yaml", job_details)

            assert result is True
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Complete"

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_failure(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="FAILED"), patch.object(
            launcher, "clean_up_resource"
        ):
            result = launcher._monitor_job("verl_recipe.yaml", job_details)

            assert result is False
            mock_job_recorder.update_job.assert_called_once()

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_cleanup_fails(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = True
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="SUCCEEDED"), patch.object(
            launcher, "clean_up_resource", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="cleanup error")
        ):
            with pytest.raises(subprocess.CalledProcessError):
                launcher._monitor_job("verl_recipe.yaml", job_details)

    @patch("subprocess.run")
    @patch("time.sleep")
    @patch("threading.Thread")
    def test_monitor_job_log_thread_dies(self, mock_thread, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_log_thread = Mock()
        mock_log_thread.is_alive.return_value = False
        mock_thread.return_value = mock_log_thread

        with patch.object(launcher, "_get_job_status", return_value="Running"):
            result = launcher._monitor_job("verl_recipe.yaml", job_details)

            assert result is False

        mock_thread.return_value = mock_log_thread

        with patch.object(
            launcher, "_get_job_status", side_effect=subprocess.CalledProcessError(1, "cmd", stderr="error")
        ):
            result = launcher._monitor_job("test_recipe.yaml", job_details)

            assert result is False


class TestCollectPodLogs:
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("subprocess.Popen")
    def test_collect_pod_logs_pytorch(self, mock_popen, mock_makedirs, mock_file, launcher):
        mock_proc = Mock()
        mock_proc.stdout = ["line1\n", "line2\n"]
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        logs = launcher._collect_pod_logs("test-job", "test-pod", False)

        assert "line1" in logs
        assert "line2" in logs
        mock_popen.assert_called_once()
        assert "kubectl" in mock_popen.call_args[0][0]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("subprocess.Popen")
    def test_collect_pod_logs_verl(self, mock_popen, mock_makedirs, mock_file, launcher):
        mock_proc = Mock()
        mock_proc.stdout = ["verl log\n"]
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        logs = launcher._collect_pod_logs("verl-job", "verl-pod", True)

        assert "verl log" in logs
        assert "-n" in mock_popen.call_args[0][0]
        assert "ray-training" in mock_popen.call_args[0][0]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("subprocess.Popen")
    def test_collect_pod_logs_training_errors(self, mock_popen, mock_makedirs, mock_file, launcher):
        mock_proc = Mock()
        mock_proc.stdout = [
            "normal log\n",
            "Exception occurred during training: OOM\n",
            "more logs\n",
            "Exception occurred during training: Error 2\n",
            "Exception occurred during training: Error 3\n",
        ]
        mock_proc.wait.return_value = None
        mock_popen.return_value = mock_proc

        logs = launcher._collect_pod_logs("test-job", "test-pod", False, training_error_limit=3)

        # Should stop after 3 errors
        assert "normal log" in logs
        mock_proc.terminate.assert_called_once()


class TestWaitForPodStatus:
    @patch("subprocess.run")
    def test_wait_for_pod_status_success(self, mock_run, launcher):
        mock_run.return_value = Mock()

        launcher._wait_for_pod_status("test-pod", "Running")

        mock_run.assert_called_once()
        assert "kubectl" in mock_run.call_args[0][0]
        assert "wait" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_wait_for_pod_status_timeout(self, mock_run, launcher):
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

        with pytest.raises(subprocess.TimeoutExpired):
            launcher._wait_for_pod_status("test-pod", "Running")

    @patch("subprocess.run")
    def test_wait_for_pod_status_calls_describe_on_error(self, mock_run, launcher):
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "cmd", stderr="wait failed"),
            Mock(stdout="pod details"),
        ]

        with pytest.raises(subprocess.CalledProcessError):
            launcher._wait_for_pod_status("test-pod", "Running")

        # Should call: wait (fails), describe pod
        assert mock_run.call_count == 2
        assert "describe" in mock_run.call_args_list[1][0][0]
        assert "pod" in mock_run.call_args_list[1][0][0]

    @patch("subprocess.run")
    def test_wait_for_pod_status_describe_fails_silently(self, mock_run, launcher):
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "cmd", stderr="wait failed"),
            subprocess.CalledProcessError(1, "cmd", stderr="describe pod failed"),
        ]

        with pytest.raises(subprocess.CalledProcessError):
            launcher._wait_for_pod_status("test-pod", "Running")

        # Should call: wait (fails), describe pod (fails and caught)
        assert mock_run.call_count == 2


class TestDescribeResource:
    @patch("subprocess.run")
    def test_describe_resource_success(self, mock_run, launcher):
        mock_result = Mock(stdout="dummy_output")
        mock_run.return_value = mock_result

        result = launcher.describe_resource("pod", "test-pod")

        assert result.stdout == "dummy_output"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["kubectl", "describe", "pod", "test-pod"]

    @patch("subprocess.run")
    def test_describe_resource_failure(self, mock_run, launcher):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="resource not found")

        with pytest.raises(subprocess.CalledProcessError):
            launcher.describe_resource("pod", "test-pod")

    @patch("subprocess.run")
    def test_describe_resource_verl(self, mock_run, launcher):
        mock_run.return_value = Mock(stdout="resource details")

        result = launcher.describe_resource("rayjobs", "verl-job", is_verl=True)

        assert "-n" in mock_run.call_args[0][0]
        assert "ray-training" in mock_run.call_args[0][0]


class TestCleanUpResource:
    @patch("subprocess.run")
    def test_clean_up_resource_success(self, mock_run, launcher):
        launcher.clean_up_resource(HP_PYTORCH_JOB_RESOURCE_NAME, "test-job")

        mock_run.assert_called_once()
        assert "kubectl" in mock_run.call_args[0][0]
        assert "delete" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_clean_up_resource_failure(self, mock_run, launcher):
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        launcher.clean_up_resource(HP_PYTORCH_JOB_RESOURCE_NAME, "test-job")

        mock_run.assert_called_once()


class TestGetPods:
    @patch("subprocess.run")
    def test_get_pods_success(self, mock_run, launcher):
        mock_run.return_value = Mock(stdout="test-job-worker-0\ntest-job-worker-1\nother-pod")

        pods = launcher.get_pods("test-job")

        assert len(pods) == 2
        assert "test-job-worker-0" in pods
        assert "test-job-worker-1" in pods

    @patch("subprocess.run")
    def test_get_pods_timeout(self, mock_run, launcher):
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 10)

        with pytest.raises(subprocess.TimeoutExpired):
            launcher.get_pods("test-job")

    @patch("subprocess.run")
    def test_get_pods_verl(self, mock_run, launcher):
        mock_run.return_value = Mock(stdout="verl-job-pod1\nverl-job-pod2\n")

        pods = launcher.get_pods("verl-job", is_verl=True)

        assert pods == ["verl-job-pod1", "verl-job-pod2"]
        assert "-n" in mock_run.call_args[0][0]
        assert "ray-training" in mock_run.call_args[0][0]


class TestGetHeadNode:
    @patch("time.sleep")
    @patch("subprocess.run")
    def test_get_head_node_single_pod(self, mock_run, mock_sleep, launcher):
        with patch.object(launcher, "get_pods", return_value=["test-job-worker-0"]):
            head_node = launcher.get_head_node("test-job")

            assert head_node == "test-job-worker-0"
            mock_run.assert_not_called()

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_get_head_node_multiple_pods(self, mock_run, mock_sleep, launcher):
        mock_run.return_value = Mock(stdout="group_rank=0")

        head_node = launcher.get_head_node("test-job", all_pods=["test-job-worker-0", "test-job-worker-1"])

        assert head_node == "test-job-worker-0"

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_get_head_node_timeout_then_success(self, mock_run, mock_sleep, launcher):
        # First call times out, second succeeds - need multiple pods to trigger retry logic
        mock_run.side_effect = [subprocess.TimeoutExpired("cmd", 10), Mock(stdout="group_rank=0")]

        head_node = launcher.get_head_node(
            "test-job", all_pods=["test-job-worker-0", "test-job-worker-1"], max_retries=2, delay=0
        )

        assert head_node == "test-job-worker-1"

    @patch("time.sleep")
    @patch("subprocess.run")
    def test_get_head_node_not_found(self, mock_run, mock_sleep, launcher):
        # With multiple pods where none have group_rank=0, should raise RuntimeError
        mock_run.return_value = Mock(stdout="group_rank=1")

        with pytest.raises(RuntimeError, match="Unable to determine head node"):
            launcher.get_head_node(
                "test-job", all_pods=["test-job-worker-0", "test-job-worker-1"], max_retries=1, delay=0
            )

    @patch("subprocess.run")
    def test_get_head_node_verl(self, mock_run, launcher):
        all_pods = ["verl-job-launcher", "verl-job-head", "verl-job-worker"]

        head = launcher.get_head_node("verl-job", all_pods=all_pods, is_verl=True)

        assert head == "verl-job-launcher"


class TestAwsEnvParameter:
    """Test that aws_env is passed to subprocess.run calls."""

    def test_init_creates_aws_env(self, launcher):
        """Test that initialization creates aws_env."""
        assert hasattr(launcher, "aws_env")
        assert isinstance(launcher.aws_env, dict)

    @patch("subprocess.run")
    def test_clean_up_resource_passes_aws_env(self, mock_run, launcher):
        """Test that clean_up_resource passes aws_env to subprocess."""
        mock_run.return_value = Mock()

        launcher.clean_up_resource("pytorchjob", "test-job")

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["env"] == launcher.aws_env

    @patch("subprocess.run")
    def test_get_pods_passes_aws_env(self, mock_run, launcher):
        """Test that get_pods passes aws_env to subprocess."""
        mock_run.return_value = Mock(stdout="test-job-worker-0\n")

        launcher.get_pods("test-job")

        assert mock_run.called
        assert mock_run.call_args.kwargs["env"] == launcher.aws_env


class TestWaitForJobStatus:
    """Tests for _wait_for_job_status method"""

    @patch("subprocess.run")
    def test_wait_for_job_status_pytorch(self, mock_run, launcher):
        mock_run.return_value = Mock()

        launcher._wait_for_job_status("test-job", "Running", is_verl=False)

        assert "hyperpodpytorchjob/test-job" in mock_run.call_args[0][0]
        assert "--for=condition=Running" in mock_run.call_args[0][0]

    @patch("subprocess.run")
    def test_wait_for_job_status_verl(self, mock_run, launcher):
        mock_run.return_value = Mock()

        launcher._wait_for_job_status("verl-job", "Running", is_verl=True)

        assert "rayjobs/verl-job" in mock_run.call_args[0][0]
        assert "--for=jsonpath={.status.jobDeploymentStatus}=Running" in mock_run.call_args[0][0]
        assert "-n" in mock_run.call_args[0][0]
        assert "ray-training" in mock_run.call_args[0][0]


class TestGetJobStatus:
    """Tests for _get_job_status method"""

    @patch("subprocess.run")
    def test_get_job_status_pytorch(self, mock_run, launcher):
        mock_run.return_value = Mock(stdout="Completed True")

        status = launcher._get_job_status("test-job", is_verl=False)

        assert status == "Completed True"
        # Current implementation returns both type and status
        assert "jsonpath={.status.conditions[-1].type} {.status.conditions[-1].status}" in " ".join(
            mock_run.call_args[0][0]
        )

    @patch("subprocess.run")
    def test_get_job_status_verl(self, mock_run, launcher):
        mock_run.return_value = Mock(stdout="RUNNING")

        status = launcher._get_job_status("verl-job", is_verl=True)

        assert status == "RUNNING"
        assert "jsonpath={.status.jobStatus}" in mock_run.call_args[0][0]
        assert "-n" in mock_run.call_args[0][0]
        assert "ray-training" in mock_run.call_args[0][0]
