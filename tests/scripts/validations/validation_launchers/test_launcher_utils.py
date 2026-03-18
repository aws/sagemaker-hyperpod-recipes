"""Unit tests for launcher_utils.select_validation_launcher."""

import sys
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.validations.validation_launchers.launcher_utils import (
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
