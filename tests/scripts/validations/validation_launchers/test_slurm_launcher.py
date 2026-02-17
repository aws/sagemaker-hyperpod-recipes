from unittest.mock import Mock, mock_open, patch

import pytest


@pytest.fixture
def mock_job_recorder():
    """Mock job recorder"""
    return Mock()


@pytest.fixture
def mock_config_local():
    """Config without slurm_client_config (local execution)"""
    config = Mock(spec=["platform", "assume_role_arn"])
    config.platform = "SLURM"
    config.assume_role_arn = "arn:aws:iam::123456789012:role/test-role"
    return config


@pytest.fixture
def mock_config_remote():
    """Config with slurm_client_config (remote execution)"""
    config = Mock()
    config.platform = "SLURM"
    config.assume_role_arn = "arn:aws:iam::123456789012:role/test-role"
    config.slurm_client_config = Mock()
    config.slurm_client_config.target_id = "mi-test123"
    config.slurm_client_config.cluster_region = "us-west-2"
    return config


class TestSlurmValidationLauncherInit:
    """Test SlurmValidationLauncher initialization"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_init_local_mode(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local):
        """Test initialization in local mode (no slurm_client_config)"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)

        assert launcher.slurm_client is None
        mock_slurm_client.assert_not_called()

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_init_remote_mode(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_remote):
        """Test initialization in remote mode (with slurm_client_config)"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_remote)

        assert launcher.slurm_client is not None
        mock_slurm_client.assert_called_once()


class TestSlurmValidationLauncherExecuteCommand:
    """Test _execute_command method"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    @patch("scripts.validations.validation_launchers.slurm_launcher.subprocess.run")
    def test_execute_command_local(
        self, mock_subprocess, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local
    ):
        """Test command execution in local mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_process = Mock()
        mock_process.stdout = "output"
        mock_subprocess.return_value = mock_process

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        result = launcher._execute_command(["python", "script.py"])

        mock_subprocess.assert_called_once()
        assert result == mock_process

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_execute_command_remote(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_remote):
        """Test command execution in remote mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_client_instance = Mock()
        mock_slurm_client.return_value = mock_client_instance
        mock_process = Mock()
        mock_process.stdout = "output"
        mock_client_instance.launch_job.return_value = mock_process

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_remote)
        result = launcher._execute_command(["python", "script.py"])

        mock_client_instance.launch_job.assert_called_once()
        assert result == mock_process


class TestSlurmValidationLauncherGetJobState:
    """Test _get_job_state method"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    @patch("scripts.validations.validation_launchers.slurm_launcher.subprocess.check_output")
    def test_get_job_state_local(
        self, mock_subprocess, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local
    ):
        """Test getting job state in local mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_subprocess.return_value = "809|RUNNING"

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        state = launcher._get_job_state("809")

        assert state == "RUNNING"
        mock_subprocess.assert_called_once()

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_get_job_state_filters_session_messages(
        self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_remote
    ):
        """Test that session messages are filtered from output"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_client_instance = Mock()
        mock_slurm_client.return_value = mock_client_instance
        mock_process = Mock()
        mock_process.stdout = """Starting session with SessionId: test-session
809|RUNNING
Exiting session with sessionId: test-session."""
        mock_client_instance.run.return_value = mock_process

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_remote)
        state = launcher._get_job_state("809")

        assert state == "RUNNING"

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    @patch("scripts.validations.validation_launchers.slurm_launcher.subprocess.check_output")
    def test_get_job_state_returns_unknown_on_exception(
        self, mock_subprocess, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local
    ):
        """Test that _get_job_state returns UNKNOWN when exception occurs"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_subprocess.side_effect = Exception("Command failed")

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        state = launcher._get_job_state("809")

        assert state == "UNKNOWN"


class TestSlurmValidationLauncherCancelJob:
    """Test _cancel_job method"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    @patch("scripts.validations.validation_launchers.slurm_launcher.subprocess.run")
    def test_cancel_job_local(
        self, mock_subprocess, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local
    ):
        """Test canceling job in local mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        launcher._cancel_job("809")

        mock_subprocess.assert_called_once()

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_cancel_job_remote(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_remote):
        """Test canceling job in remote mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_client_instance = Mock()
        mock_slurm_client.return_value = mock_client_instance

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_remote)
        launcher._cancel_job("809")

        mock_client_instance.run.assert_called_once()


class TestSlurmValidationLauncherCollectLogs:
    """Test _collect_job_logs method"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_collect_logs_local(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local):
        """Test collecting logs in local mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        launcher.slurm_client = None

        # Patch open after launcher is created
        with patch("builtins.open", mock_open(read_data="log content")):
            logs = launcher._collect_job_logs("/path/to/log.txt")
            assert "log content" in logs

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_collect_logs_remote(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_remote):
        """Test collecting logs in remote mode"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_client_instance = Mock()
        mock_slurm_client.return_value = mock_client_instance
        mock_process = Mock()
        mock_process.stdout = "remote log content"
        mock_client_instance.run.return_value = mock_process

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_remote)
        logs = launcher._collect_job_logs("/path/to/log.txt")

        assert logs == "remote log content"


class TestSlurmValidationLauncherParseOutput:
    """Test _parse_output method"""

    @patch("scripts.validations.validation_launchers.base_launcher.boto3")
    @patch("scripts.validations.validation_launchers.slurm_launcher.SlurmClient")
    def test_parse_output_success(self, mock_slurm_client, mock_boto3, mock_job_recorder, mock_config_local):
        """Test parsing successful job submission output"""
        from scripts.validations.validation_launchers.slurm_launcher import (
            SlurmValidationLauncher,
        )

        mock_output = Mock()
        mock_output.stdout = """Job submission file created at /path/to/submission.sh
Job submitted with Job ID 809
Submitted job's logfile path /path/to/log.txt"""

        launcher = SlurmValidationLauncher(mock_job_recorder, mock_config_local)
        result = launcher._parse_output("input.yaml", mock_output)

        assert result["job_id"] == "809"
        assert result["log_file_path"] == "/path/to/log.txt"
        assert result["output_folder_path"] == "/path/to/submission.sh"
