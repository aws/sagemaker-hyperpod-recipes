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
        result = launcher._parse_output("verl_recipe.yaml", launch_output)

        assert result["job_name"] == "verl-job-456"
        assert result["pod_name"] == "verl-job-456"
        assert result["output_folder_path"] == "/tmp/verl_output"
        assert "all_pods" not in result  # VERL jobs don't set all_pods

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


class TestMonitorJob:
    def test_monitor_job_pytorch(self, launcher):
        job_details = {"job_name": "test-job", "pod_name": "test-pod"}

        with patch.object(launcher, "_monitor_pytorch_job", return_value=True) as mock_monitor:
            result = launcher._monitor_job("test_recipe.yaml", job_details)

            assert result is True
            mock_monitor.assert_called_once_with("test_recipe.yaml", job_details, 10)

    def test_monitor_job_verl(self, launcher):
        job_details = {"job_name": "verl-job"}

        with patch.object(launcher, "_monitor_ray_job", return_value=True) as mock_monitor:
            result = launcher._monitor_job("verl_recipe.yaml", job_details)

            assert result is True
            mock_monitor.assert_called_once_with("verl_recipe.yaml", job_details, 10)


class TestMonitorRayJob:
    @patch("subprocess.run")
    @patch("time.sleep")
    def test_monitor_ray_job_success(self, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        # First call: kubectl get rayjobs (status check) returns "Completed"
        # Second call: kubectl delete rayjobs (cleanup)
        mock_run.side_effect = [
            Mock(stdout="SUCCEEDED"),  # status check
            Mock(),  # cleanup
        ]

        with patch.object(launcher, "_collect_pod_logs", return_value="logs"), patch.object(
            launcher, "calculate_throughput_from_logs", return_value=None
        ):
            result = launcher._monitor_ray_job("verl_recipe.yaml", job_details)

            assert result is True
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Complete"

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_monitor_ray_job_failure(self, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        # Status check returns "Failed"
        mock_run.side_effect = [
            Mock(stdout="FAILED"),  # status check
            Mock(),  # cleanup
        ]

        with patch.object(launcher, "_collect_pod_logs", return_value="logs"):
            result = launcher._monitor_ray_job("verl_recipe.yaml", job_details)

            assert result is False
            mock_job_recorder.update_job.assert_called_once()

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_monitor_ray_job_pod_not_ready(self, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_run.side_effect = [
            Mock(stdout="SUCCEEDED"),  # status check
            Mock(),  # cleanup
        ]

        with patch.object(launcher, "_collect_pod_logs", return_value="logs"), patch.object(
            launcher, "calculate_throughput_from_logs", return_value=None
        ):
            result = launcher._monitor_ray_job("verl_recipe.yaml", job_details)

            assert result is True

    @patch("subprocess.run")
    @patch("time.sleep")
    def test_monitor_ray_job_cleanup_fails(self, mock_sleep, mock_run, launcher, mock_job_recorder):
        job_details = {"job_name": "verl-job", "pod_name": "verl-job", "output_folder_path": "/tmp/output"}
        mock_run.side_effect = [
            Mock(stdout="SUCCEEDED"),  # status check
            subprocess.CalledProcessError(1, "cmd", stderr="cleanup error"),  # cleanup fails
        ]

        with patch.object(launcher, "_collect_pod_logs", return_value="logs"), patch.object(
            launcher, "calculate_throughput_from_logs", return_value=None
        ):
            result = launcher._monitor_ray_job("verl_recipe.yaml", job_details)

            assert result is True  # Job succeeds even if cleanup fails


class TestMonitorPytorchJob:
    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_monitor_pytorch_job_success(self, mock_makedirs, mock_file, mock_run, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "all_pods": ["test-pod"],
            "output_folder_path": "/tmp/output",
        }
        # First call: kubectl get hyperpodpytorchjobs (status check) returns "Completed True"
        # Second call: kubectl delete hyperpodpytorchjobs (cleanup)
        mock_run.side_effect = [
            Mock(stdout="Completed True"),  # status check
            Mock(),  # cleanup
        ]

        with patch.object(launcher, "_wait_for_pod_status"), patch.object(
            launcher, "_collect_pod_logs", return_value="epoch=1 epoch=2 epoch=3 epoch=4 epoch=5 agent is finished"
        ), patch.object(launcher, "calculate_throughput_from_logs", return_value=100.5):
            result = launcher._monitor_pytorch_job("test_recipe.yaml", job_details)

            assert result is True
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Complete"
            assert mock_job_recorder.update_job.call_args[1]["tokens_throughput"] == 100.5

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_monitor_pytorch_job_failure(self, mock_makedirs, mock_file, mock_run, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "all_pods": ["test-pod"],
            "output_folder_path": "/tmp/output",
        }
        mock_run.side_effect = [
            Mock(stdout="Failed True"),  # status check
            Mock(),  # cleanup
        ]

        with patch.object(launcher, "_wait_for_pod_status"), patch.object(
            launcher, "_collect_pod_logs", return_value="error logs"
        ), patch.object(launcher, "calculate_throughput_from_logs", return_value=None):
            result = launcher._monitor_pytorch_job("test_recipe.yaml", job_details)

            assert result is False
            mock_job_recorder.update_job.assert_called_once()
            assert mock_job_recorder.update_job.call_args[1]["status"] == "Failed"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_monitor_pytorch_job_exception(self, mock_makedirs, mock_file, launcher, mock_job_recorder):
        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "all_pods": ["test-pod"],
            "output_folder_path": "/tmp/output",
        }

        with patch.object(launcher, "_wait_for_pod_status") as mock_wait:
            mock_wait.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")
            with patch.object(launcher, "clean_up_resource"):
                result = launcher._monitor_pytorch_job("test_recipe.yaml", job_details)

                assert result is False
                mock_job_recorder.update_job.assert_called_once()
                assert mock_job_recorder.update_job.call_args[1]["status"] == "Failed"


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
