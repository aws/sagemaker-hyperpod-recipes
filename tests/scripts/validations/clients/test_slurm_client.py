import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_boto_session():
    """Mock boto3 session with credentials"""
    session = Mock()
    credentials = Mock()
    credentials.access_key = "test_access_key"
    credentials.secret_key = "test_secret_key"
    credentials.token = "test_token"
    session.get_credentials.return_value = credentials
    session.client.return_value = Mock()
    return session


@pytest.fixture
def slurm_config():
    """Basic SlurmClientConfig"""
    # Import inside fixture to avoid circular import
    from scripts.validations.clients.slurm_client import SlurmClientConfig

    return SlurmClientConfig(
        target_id="sagemaker-cluster:test_controller-i-0123456789abcdef0", cluster_region="us-west-2"
    )


@pytest.fixture
def temp_work_dir():
    """Create temporary work directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestSlurmClientInit:
    """Test SlurmClient initialization"""

    @patch("scripts.validations.clients.slurm_client.SlurmClient._validate_platform_auth")
    @patch("scripts.validations.clients.slurm_client.SlurmClient._setup_workdir")
    @patch("scripts.validations.clients.slurm_client.os.makedirs")
    def test_init_creates_directories(self, mock_makedirs, mock_setup, mock_validate, slurm_config, mock_boto_session):
        """Test that __init__ creates local work directory"""
        from scripts.validations.clients.slurm_client import SlurmClient

        client = SlurmClient(slurm_config, boto_session=mock_boto_session)

        mock_makedirs.assert_called_once()
        assert client.target_id == slurm_config.target_id
        assert client.region == slurm_config.cluster_region

    @patch("scripts.validations.clients.slurm_client.SlurmClient._validate_platform_auth")
    @patch("scripts.validations.clients.slurm_client.SlurmClient._setup_workdir")
    @patch("scripts.validations.clients.slurm_client.os.makedirs")
    def test_init_sets_aws_credentials(self, mock_makedirs, mock_setup, mock_validate, slurm_config, mock_boto_session):
        """Test that AWS credentials are set in process_env"""
        from scripts.validations.clients.slurm_client import SlurmClient

        client = SlurmClient(slurm_config, boto_session=mock_boto_session)

        assert client.process_env["AWS_ACCESS_KEY_ID"] == "test_access_key"
        assert client.process_env["AWS_SECRET_ACCESS_KEY"] == "test_secret_key"
        assert client.process_env["AWS_SESSION_TOKEN"] == "test_token"
        assert client.process_env["AWS_DEFAULT_REGION"] == "us-west-2"


class TestSlurmClientRun:
    """Test SlurmClient.run() method"""

    @patch("scripts.validations.clients.slurm_client.SlurmClient._validate_platform_auth")
    @patch("scripts.validations.clients.slurm_client.SlurmClient._setup_workdir")
    @patch("scripts.validations.clients.slurm_client.os.makedirs")
    @patch("scripts.validations.clients.slurm_client.subprocess.run")
    @patch("scripts.validations.clients.slurm_client.get_project_root")
    def test_run_executes_ssm_command(
        self, mock_root, mock_subprocess, mock_makedirs, mock_setup, mock_validate, slurm_config, mock_boto_session
    ):
        """Test that run() executes SSM command via AWS CLI"""
        from scripts.validations.clients.slurm_client import SlurmClient

        mock_root.return_value = "/local/project"
        mock_process = Mock()
        mock_process.stdout = "Done"
        mock_subprocess.return_value = mock_process

        client = SlurmClient(slurm_config, boto_session=mock_boto_session)
        result = client.run(["squeue"], timeout_in_min=1)

        # Verify subprocess.run was called
        assert mock_subprocess.called
        call_args = mock_subprocess.call_args

        # Verify AWS CLI command structure
        assert call_args[0][0][0] == "aws"
        assert call_args[0][0][1] == "ssm"
        assert call_args[0][0][2] == "start-session"
        assert "--target" in call_args[0][0]
        assert slurm_config.target_id in call_args[0][0]


class TestSlurmClientCleanup:
    """Test SlurmClient cleanup methods"""

    @patch("scripts.validations.clients.slurm_client.SlurmClient._validate_platform_auth")
    @patch("scripts.validations.clients.slurm_client.SlurmClient._setup_workdir")
    @patch("scripts.validations.clients.slurm_client.os.makedirs")
    def test_clean_local_dir(
        self, mock_makedirs, mock_setup, mock_validate, slurm_config, mock_boto_session, temp_work_dir
    ):
        """Test that _clean_local_dir removes all files"""
        from scripts.validations.clients.slurm_client import SlurmClient

        client = SlurmClient(slurm_config, boto_session=mock_boto_session)
        client.local_work_dir = temp_work_dir

        # Create test files
        test_file = os.path.join(temp_work_dir, "test.txt")
        test_dir = os.path.join(temp_work_dir, "testdir")
        os.makedirs(test_dir)
        with open(test_file, "w") as f:
            f.write("test")

        client._clean_local_dir()

        # Verify directory is empty
        assert len(os.listdir(temp_work_dir)) == 0

    @patch("scripts.validations.clients.slurm_client.SlurmClient._validate_platform_auth")
    @patch("scripts.validations.clients.slurm_client.SlurmClient._setup_workdir")
    @patch("scripts.validations.clients.slurm_client.os.makedirs")
    def test_clean_s3_artifacts(self, mock_makedirs, mock_setup, mock_validate, slurm_config, mock_boto_session):
        """Test that _clean_s3_artifacts deletes S3 objects"""
        from scripts.validations.clients.slurm_client import SlurmClient

        mock_s3_client = Mock()
        mock_paginator = Mock()
        mock_s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "key1"}, {"Key": "key2"}]}]

        client = SlurmClient(slurm_config, boto_session=mock_boto_session)
        client.s3_client = mock_s3_client

        client._clean_s3_artifacts()

        # Verify delete_objects was called
        mock_s3_client.delete_objects.assert_called_once()
        call_args = mock_s3_client.delete_objects.call_args
        assert len(call_args[1]["Delete"]["Objects"]) == 2
