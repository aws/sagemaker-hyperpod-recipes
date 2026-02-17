"""Unit tests for ServerlessValidationLauncher CloudWatch logs validation with assumed role support."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from omegaconf import OmegaConf

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.validations.validation_launchers.serverless_launcher import (
    ServerlessValidationLauncher,
)


class TestServerlessLauncherCloudWatchValidation(unittest.TestCase):
    """Test suite for CloudWatch logs validation with assumed role support."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = OmegaConf.create(
            {
                "platform": "SERVERLESS",
                "serverless_config": {
                    "region": "us-west-2",
                    "cloudwatch_log_group": "/aws/sagemaker/TrainingJobs",
                    "hub_account_id": "123456789012",
                    "hub_name": "test-hub",
                    "model_version": "1.0",
                    "role_arn": "arn:aws:iam::123456789012:role/test-role",
                    "s3_output_path": "s3://test-bucket/output",
                    "max_runtime_seconds": 3600,
                    "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
                    "endpoint": "https://api.sagemaker.us-west-2.amazonaws.com",
                    "default_hyper_parameters": {},
                },
                "serverless_dataset_mapping": {
                    "RLAIF": "arn:aws:sagemaker:us-west-2:123456789012:dataset/rlaif-dataset",
                    "DPO": "arn:aws:sagemaker:us-west-2:123456789012:dataset/dpo-dataset",
                },
                "serverless_mlflow_mapping": {
                    "RLAIF": "arn:aws:sagemaker:us-west-2:123456789012:mlflow/rlaif-mlflow",
                    "DPO": "arn:aws:sagemaker:us-west-2:123456789012:mlflow/dpo-mlflow",
                },
            }
        )

        # Create mock job recorder
        self.mock_job_recorder = MagicMock()

        # Create launcher instance
        self.launcher = ServerlessValidationLauncher(self.mock_job_recorder, self.mock_config)

    def tearDown(self):
        """Clean up after tests."""
        # Remove environment variable if set
        if "HP_MODEL_CUSTOMIZATION_ASSUME_ROLE_ARN" in os.environ:
            del os.environ["HP_MODEL_CUSTOMIZATION_ASSUME_ROLE_ARN"]

    def test_validate_cloudwatch_logs(self):
        """Test CloudWatch validation using default credentials (no assume role)."""
        # Setup
        job_name = "test-job-123"
        technique = "RLAIF"

        # Mock CloudWatch logs client
        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        # Mock log streams response
        mock_logs_client.describe_log_streams.return_value = {"logStreams": [{"logStreamName": f"{job_name}/stream1"}]}

        # Mock log events with success message
        mock_logs_client.get_log_events.return_value = {
            "events": [
                {"message": "Training started"},
                {"message": "Saving the merged model"},  # Success message for RLAIF
                {"message": "Training completed"},
            ]
        }

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertTrue(is_valid)
        self.assertIn("Saving the merged model", message)
        self.assertIn("stream1", message)

        # Verify boto3 client was called with correct parameters (no assume role)
        self.launcher.boto_session.client.assert_called_once_with("logs", region_name="us-west-2")

    def test_validate_cloudwatch_logs_no_log_streams(self):
        """Test validation when no log streams are found."""
        # Setup
        job_name = "test-job-789"
        technique = "RLAIF"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        # Mock empty log streams response
        mock_logs_client.describe_log_streams.return_value = {"logStreams": []}

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("No CloudWatch log streams found", message)
        self.assertIn(job_name, message)

    def test_validate_cloudwatch_logs_missing_success_message(self):
        """Test validation when success message is not found in logs."""
        # Setup
        job_name = "test-job-999"
        technique = "RLAIF"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        mock_logs_client.describe_log_streams.return_value = {"logStreams": [{"logStreamName": f"{job_name}/stream1"}]}

        # Mock log events WITHOUT success message
        mock_logs_client.get_log_events.return_value = {
            "events": [
                {"message": "Training started"},
                {"message": "Processing data"},
                {"message": "Training completed"},
            ]
        }

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("Expected messages", message)
        self.assertIn("not found", message)

    def test_validate_cloudwatch_logs_log_group_not_found(self):
        """Test validation when CloudWatch log group doesn't exist."""
        # Setup
        job_name = "test-job-404"
        technique = "DPO"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        # Mock ResourceNotFoundException
        mock_logs_client.describe_log_streams.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "The specified log group does not exist."}},
            "DescribeLogStreams",
        )

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertFalse(is_valid)
        self.assertIn("CloudWatch log validation error", message)
        self.assertIn("ResourceNotFoundException", message)

    def test_validate_cloudwatch_logs_multiple_streams(self):
        """Test validation with multiple log streams."""
        # Setup
        job_name = "test-job-multi"
        technique = "SFT"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        # Mock multiple log streams
        mock_logs_client.describe_log_streams.return_value = {
            "logStreams": [
                {"logStreamName": f"{job_name}/stream1"},
                {"logStreamName": f"{job_name}/stream2"},
                {"logStreamName": f"{job_name}/stream3"},
            ]
        }

        # First two streams don't have success message
        # Third stream has success message
        def get_log_events_side_effect(logGroupName, logStreamName, **kwargs):
            if "stream3" in logStreamName:
                return {"events": [{"message": "Saving the complete model"}]}  # Success for SFT
            return {"events": [{"message": "Some other log message"}]}

        mock_logs_client.get_log_events.side_effect = get_log_events_side_effect

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertTrue(is_valid)
        self.assertIn("Saving the complete model", message)
        self.assertIn("stream3", message)

    def test_validate_cloudwatch_logs_case_insensitive_match(self):
        """Test that success message matching is case-insensitive."""
        # Setup
        job_name = "test-job-case"
        technique = "RLVR"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        mock_logs_client.describe_log_streams.return_value = {"logStreams": [{"logStreamName": f"{job_name}/stream1"}]}

        # Mock log events with different case
        mock_logs_client.get_log_events.return_value = {
            "events": [{"message": "SAVING THE MERGED MODEL"}]  # Uppercase version
        }

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertTrue(is_valid)
        self.assertIn("Saving the merged model", message)

    def test_validate_cloudwatch_logs_default_log_group(self):
        """Test that default log group is used when not specified in config."""
        # Setup - remove cloudwatch_log_group from config
        config_without_log_group = OmegaConf.create(
            {
                "platform": "SERVERLESS",
                "serverless_config": {
                    "region": "us-west-2",
                    # No cloudwatch_log_group specified
                    "hub_account_id": "123456789012",
                    "hub_name": "test-hub",
                    "model_version": "1.0",
                    "role_arn": "arn:aws:iam::123456789012:role/test-role",
                    "s3_output_path": "s3://test-bucket/output",
                    "max_runtime_seconds": 3600,
                    "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
                    "endpoint": "https://api.sagemaker.us-west-2.amazonaws.com",
                    "default_hyper_parameters": {},
                },
                "serverless_dataset_mapping": {},
                "serverless_mlflow_mapping": {},
            }
        )

        launcher = ServerlessValidationLauncher(self.mock_job_recorder, config_without_log_group)

        job_name = "test-job-default"
        technique = "RLAIF"

        mock_logs_client = MagicMock()
        launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        mock_logs_client.describe_log_streams.return_value = {"logStreams": [{"logStreamName": f"{job_name}/stream1"}]}

        mock_logs_client.get_log_events.return_value = {"events": [{"message": "Saving the merged model"}]}

        # Execute
        is_valid, message = launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertTrue(is_valid)

        # Verify describe_log_streams was called with default log group
        mock_logs_client.describe_log_streams.assert_called_once()
        call_kwargs = mock_logs_client.describe_log_streams.call_args[1]
        self.assertEqual(call_kwargs["logGroupName"], "/aws/sagemaker/TrainingJobs")


class TestServerlessLauncherTechniqueMessages(unittest.TestCase):
    """Test suite for technique-specific success messages."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = OmegaConf.create(
            {
                "platform": "SERVERLESS",
                "serverless_config": {
                    "region": "us-west-2",
                    "cloudwatch_log_group": "/aws/sagemaker/TrainingJobs",
                    "hub_account_id": "123456789012",
                    "hub_name": "test-hub",
                    "model_version": "1.0",
                    "role_arn": "arn:aws:iam::123456789012:role/test-role",
                    "s3_output_path": "s3://test-bucket/output",
                    "max_runtime_seconds": 3600,
                    "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
                    "endpoint": "https://api.sagemaker.us-west-2.amazonaws.com",
                    "default_hyper_parameters": {},
                },
                "serverless_dataset_mapping": {},
                "serverless_mlflow_mapping": {},
            }
        )

        self.mock_job_recorder = MagicMock()
        self.launcher = ServerlessValidationLauncher(self.mock_job_recorder, self.mock_config)

    def test_rlaif_success_message(self):
        """Test RLAIF technique expects 'Saving the merged model'."""
        self._test_technique_message("RLAIF", "Saving the merged model")

    def test_rlvr_success_message(self):
        """Test RLVR technique expects 'Saving the merged model'."""
        self._test_technique_message("RLVR", "Saving the merged model")

    def test_dpo_success_message(self):
        """Test DPO technique expects 'Saving the complete model'."""
        self._test_technique_message("DPO", "Saving the complete model")

    def test_sft_success_message(self):
        """Test SFT technique expects 'Saving the complete model'."""
        self._test_technique_message("SFT", "Saving the complete model")

    def _test_technique_message(self, technique, expected_message):
        """Helper method to test technique-specific messages."""
        job_name = f"test-job-{technique.lower()}"

        mock_logs_client = MagicMock()
        self.launcher.boto_session.client = MagicMock(return_value=mock_logs_client)

        mock_logs_client.describe_log_streams.return_value = {"logStreams": [{"logStreamName": f"{job_name}/stream1"}]}

        mock_logs_client.get_log_events.return_value = {"events": [{"message": expected_message}]}

        # Execute
        is_valid, message = self.launcher._validate_cloudwatch_logs(job_name, technique)

        # Assert
        self.assertTrue(is_valid)
        self.assertIn(expected_message, message)


if __name__ == "__main__":
    unittest.main()
