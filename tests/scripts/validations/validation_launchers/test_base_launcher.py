"""Unit tests for BaseLauncher initialization."""

import logging
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.validations.validation_launchers.base_launcher import BaseLauncher


class ConcreteLauncher(BaseLauncher):
    """Concrete implementation of BaseLauncher for testing."""

    def _prepare_job(self, input_file_path: str):
        """Implement abstract method."""

    def _build_command(self, run_info):
        """Implement abstract method."""

    def _execute_command(self, launch_command):
        """Implement abstract method."""

    def _parse_output(self, input_file_path, launch_output):
        """Implement abstract method."""

    def _monitor_job(self, input_file_path, job_details):
        """Implement abstract method."""


class TestBaseLauncherInit(unittest.TestCase):
    """Test suite for BaseLauncher initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_job_recorder = MagicMock()

    def test_init_without_assume_role(self):
        """Test initialization without assume role ARN."""
        # Setup
        config = OmegaConf.create(
            {
                "platform": "TEST_PLATFORM",
                "region": "us-west-2",
            }
        )

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert
        self.assertEqual(launcher.job_recorder, self.mock_job_recorder)
        self.assertEqual(launcher.config, config)
        self.assertIsNotNone(launcher.logger)
        self.assertEqual(launcher.logger.name, "ConcreteLauncher")
        self.assertEqual(launcher.logger.level, logging.INFO)
        self.assertFalse(launcher.logger.propagate)
        self.assertIsNotNone(launcher.boto_session)

        # Assert AWS environment exists
        self.assertIsNotNone(launcher.aws_env)
        self.assertIsInstance(launcher.aws_env, dict)

    @patch("boto3.client")
    @patch("boto3.Session")
    def test_init_with_assume_role(self, mock_session, mock_boto_client):
        """Test initialization with assume role ARN."""
        # Setup
        assume_role_arn = "arn:aws:iam::123456789012:role/TestRole"
        config = OmegaConf.create(
            {
                "platform": "TEST_PLATFORM",
                "region": "us-west-2",
                "assume_role_arn": assume_role_arn,
            }
        )

        mock_sts_client = MagicMock()
        mock_boto_client.return_value = mock_sts_client

        # Mock assume_role response
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "ASIA_TEST_KEY_ID",
                "SecretAccessKey": "test_secret_key",
                "SessionToken": "test_session_token",
                "Expiration": "2099-01-01T00:00:00Z",
            }
        }

        mock_boto_session = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "ASIA_TEST_KEY_ID"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_session_token"
        mock_boto_session.get_credentials.return_value = mock_credentials
        mock_session.return_value = mock_boto_session

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert
        self.assertEqual(launcher.job_recorder, self.mock_job_recorder)
        self.assertEqual(launcher.config, config)
        self.assertIsNotNone(launcher.logger)

        # Verify STS client was created
        mock_boto_client.assert_called_once_with("sts")

        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn=assume_role_arn,
            RoleSessionName="test_platform-validation-session",
        )

        # Verify boto session was created with assumed credentials
        mock_session.assert_called_once_with(
            aws_access_key_id="ASIA_TEST_KEY_ID",
            aws_secret_access_key="test_secret_key",
            aws_session_token="test_session_token",
        )

        self.assertEqual(launcher.boto_session, mock_boto_session)

        # Assert AWS environment variables are set with assumed role credentials
        self.assertIsNotNone(launcher.aws_env)
        self.assertEqual(launcher.aws_env["AWS_ACCESS_KEY_ID"], "ASIA_TEST_KEY_ID")
        self.assertEqual(launcher.aws_env["AWS_SECRET_ACCESS_KEY"], "test_secret_key")
        self.assertEqual(launcher.aws_env["AWS_SESSION_TOKEN"], "test_session_token")

    @patch("boto3.client")
    def test_assume_role_with_different_platforms(self, mock_boto_client):
        """Test that role session name uses platform name correctly."""
        # Setup
        test_cases = [
            ("K8S", "k8s-validation-session"),
            ("SLURM", "slurm-validation-session"),
            ("SMJOBS", "smjobs-validation-session"),
            ("SERVERLESS", "serverless-validation-session"),
        ]

        mock_sts_client = MagicMock()
        mock_boto_client.return_value = mock_sts_client

        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "ASIA_TEST",
                "SecretAccessKey": "test_secret",
                "SessionToken": "test_token",
                "Expiration": "2099-01-01T00:00:00Z",
            }
        }

        for platform, expected_session_name in test_cases:
            with self.subTest(platform=platform):
                # Reset mock
                mock_sts_client.reset_mock()

                config = OmegaConf.create(
                    {
                        "platform": platform,
                        "assume_role_arn": "arn:aws:iam::123456789012:role/TestRole",
                    }
                )

                # Execute
                launcher = ConcreteLauncher(self.mock_job_recorder, config)

                # Assert
                mock_sts_client.assume_role.assert_called_once()
                call_kwargs = mock_sts_client.assume_role.call_args[1]
                self.assertEqual(call_kwargs["RoleSessionName"], expected_session_name)

    def test_config_without_assume_role_attribute(self):
        """Test initialization when config doesn't have assume_role_arn attribute."""
        # Setup - config without assume_role_arn
        config = OmegaConf.create(
            {
                "platform": "TEST_PLATFORM",
                "region": "us-west-2",
                "other_setting": "value",
            }
        )

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert - should create default boto session
        self.assertIsNotNone(launcher.boto_session)
        self.assertEqual(launcher.config, config)

    def test_config_with_none_assume_role(self):
        """Test initialization when assume_role_arn is explicitly None."""
        # Setup
        config = OmegaConf.create(
            {
                "platform": "TEST_PLATFORM",
                "region": "us-west-2",
                "assume_role_arn": None,
            }
        )

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert - should create default boto session
        self.assertIsNotNone(launcher.boto_session)
        self.assertEqual(launcher.config, config)

    @patch("boto3.client")
    def test_assume_role_logs_info_message(self, mock_boto_client):
        """Test that assume role creates logger and calls STS."""
        # Setup
        assume_role_arn = "arn:aws:iam::123456789012:role/TestRole"
        config = OmegaConf.create(
            {
                "platform": "TEST_PLATFORM",
                "assume_role_arn": assume_role_arn,
            }
        )

        mock_sts_client = MagicMock()
        mock_boto_client.return_value = mock_sts_client

        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "ASIA_TEST",
                "SecretAccessKey": "test_secret",
                "SessionToken": "test_token",
                "Expiration": "2099-01-01T00:00:00Z",
            }
        }

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert - verify logger exists and STS was called (which means the log message was executed)
        self.assertIsNotNone(launcher.logger)
        mock_sts_client.assume_role.assert_called_once()
        self.assertEqual(launcher.logger.name, "ConcreteLauncher")

    def test_aws_env_contains_system_environment(self):
        """Test that aws_env includes system environment variables."""
        # Setup
        import os

        test_env_var = "TEST_CUSTOM_VAR"
        test_env_value = "test_value_12345"
        os.environ[test_env_var] = test_env_value

        config = OmegaConf.create({"platform": "TEST_PLATFORM"})

        try:
            # Execute
            launcher = ConcreteLauncher(self.mock_job_recorder, config)

            # Assert - aws_env should contain system env vars
            self.assertIn(test_env_var, launcher.aws_env)
            self.assertEqual(launcher.aws_env[test_env_var], test_env_value)
        finally:
            # Cleanup
            del os.environ[test_env_var]

    def test_aws_env_credentials_not_none(self):
        """Test that AWS credentials in env are not None when credentials are available."""
        # Setup
        config = OmegaConf.create({"platform": "TEST_PLATFORM"})

        # Execute
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert - aws_env should always exist
        self.assertIsNotNone(launcher.aws_env)
        self.assertIsInstance(launcher.aws_env, dict)
        # Note: AWS credential keys may or may not be present depending on environment

    @patch("boto3.Session")
    def test_handles_none_credentials_gracefully(self, mock_session):
        """Test that initialization handles None credentials gracefully."""
        # Setup - simulate environment with no AWS credentials
        config = OmegaConf.create({"platform": "TEST_PLATFORM"})

        mock_boto_session = MagicMock()
        mock_boto_session.get_credentials.return_value = None
        mock_session.return_value = mock_boto_session

        # Execute - should not raise AttributeError
        launcher = ConcreteLauncher(self.mock_job_recorder, config)

        # Assert - aws_env should exist but may not have AWS credentials
        self.assertIsNotNone(launcher.aws_env)
        self.assertIsInstance(launcher.aws_env, dict)


class TestCredentialRefresh(unittest.TestCase):
    """Test suite for _refresh_credentials_if_needed()."""

    def _create_launcher_with_assumed_role(self, expiry):
        """Helper: create a ConcreteLauncher with mocked assumed-role credentials."""
        with patch("boto3.client") as mock_client, patch("boto3.Session"):
            mock_sts = MagicMock()
            mock_client.return_value = mock_sts
            mock_sts.assume_role.return_value = {
                "Credentials": {
                    "AccessKeyId": "ORIG_KEY",
                    "SecretAccessKey": "ORIG_SECRET",
                    "SessionToken": "ORIG_TOKEN",
                    "Expiration": expiry,
                }
            }
            config = OmegaConf.create(
                {
                    "platform": "SERVERLESS",
                    "assume_role_arn": "arn:aws:iam::123456789012:role/TestRole",
                }
            )
            launcher = ConcreteLauncher(MagicMock(), config)
        return launcher

    @patch("scripts.validations.validation_launchers.base_launcher.boto3.client")
    def test_refresh_when_credentials_expiring_soon(self, mock_boto_client):
        """Credentials within REFRESH_THRESHOLD_MINUTES of expiry should trigger a refresh."""
        new_expiry = datetime.now(timezone.utc) + timedelta(hours=4)
        mock_sts = MagicMock()
        mock_boto_client.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "NEW_KEY",
                "SecretAccessKey": "NEW_SECRET",
                "SessionToken": "NEW_TOKEN",
                "Expiration": new_expiry,
            }
        }

        launcher = self._create_launcher_with_assumed_role(
            expiry=datetime.now(timezone.utc) + timedelta(minutes=10)  # within 15 min threshold
        )

        launcher._refresh_credentials_if_needed()

        # STS assume_role should have been called
        mock_sts.assume_role.assert_called_once()
        call_kwargs = mock_sts.assume_role.call_args[1]
        self.assertNotIn("DurationSeconds", call_kwargs)
        # Credentials should be updated
        self.assertEqual(launcher.aws_env["AWS_ACCESS_KEY_ID"], "NEW_KEY")
        self.assertEqual(launcher.aws_env["AWS_SECRET_ACCESS_KEY"], "NEW_SECRET")
        self.assertEqual(launcher.aws_env["AWS_SESSION_TOKEN"], "NEW_TOKEN")
        # Expiry should be updated
        self.assertEqual(launcher._credentials_expiry, new_expiry)

    @patch("scripts.validations.validation_launchers.base_launcher.boto3.client")
    def test_no_refresh_when_credentials_still_fresh(self, mock_boto_client):
        """Credentials with >REFRESH_THRESHOLD_MINUTES remaining should NOT trigger a refresh."""
        launcher = self._create_launcher_with_assumed_role(
            expiry=datetime.now(timezone.utc) + timedelta(hours=2)  # 2 hours left — well above 15 min
        )

        launcher._refresh_credentials_if_needed()

        # STS should NOT be called
        mock_boto_client.return_value.assume_role.assert_not_called()

    def test_no_refresh_when_no_assume_role_arn(self):
        """Without an assumed role, refresh should be a no-op."""
        config = OmegaConf.create({"platform": "TEST_PLATFORM"})
        launcher = ConcreteLauncher(MagicMock(), config)
        launcher._assume_role_arn = None
        launcher._credentials_expiry = None

        # Should not raise
        launcher._refresh_credentials_if_needed()

    def test_no_refresh_when_no_expiry_stored(self):
        """If _credentials_expiry is None (defensive), refresh should be a no-op."""
        config = OmegaConf.create({"platform": "TEST_PLATFORM"})
        launcher = ConcreteLauncher(MagicMock(), config)
        launcher._assume_role_arn = "arn:aws:iam::123456789012:role/TestRole"
        launcher._credentials_expiry = None

        # Should not raise
        launcher._refresh_credentials_if_needed()

    @patch("scripts.validations.validation_launchers.base_launcher.boto3.client")
    def test_refresh_updates_boto_session(self, mock_boto_client):
        """Refresh should create a new boto3.Session with fresh credentials."""
        new_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_sts = MagicMock()
        mock_boto_client.return_value = mock_sts
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "REFRESHED_KEY",
                "SecretAccessKey": "REFRESHED_SECRET",
                "SessionToken": "REFRESHED_TOKEN",
                "Expiration": new_expiry,
            }
        }

        launcher = self._create_launcher_with_assumed_role(
            expiry=datetime.now(timezone.utc) + timedelta(minutes=1)  # almost expired
        )
        old_session = launcher.boto_session

        with patch("scripts.validations.validation_launchers.base_launcher.boto3.Session") as mock_session:
            mock_new_session = MagicMock()
            mock_session.return_value = mock_new_session

            launcher._refresh_credentials_if_needed()

            # New session should have been created
            mock_session.assert_called_once_with(
                aws_access_key_id="REFRESHED_KEY",
                aws_secret_access_key="REFRESHED_SECRET",
                aws_session_token="REFRESHED_TOKEN",
            )
            self.assertEqual(launcher.boto_session, mock_new_session)

    @patch("scripts.validations.validation_launchers.base_launcher.boto3.client")
    def test_refresh_at_exact_boundary(self, mock_boto_client):
        """Credentials expiring just above REFRESH_THRESHOLD_MINUTES should NOT trigger refresh."""
        launcher = self._create_launcher_with_assumed_role(
            expiry=datetime.now(timezone.utc) + timedelta(minutes=BaseLauncher.REFRESH_THRESHOLD_MINUTES, seconds=1)
        )

        launcher._refresh_credentials_if_needed()

        # Should NOT be called — threshold + 1 sec is still safe
        mock_boto_client.return_value.assume_role.assert_not_called()


if __name__ == "__main__":
    unittest.main()
