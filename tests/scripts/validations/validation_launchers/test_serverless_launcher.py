"""
Unit tests for Hub content override in ServerlessValidationLauncher
"""
import json
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from scripts.validations.validation_launchers.serverless_launcher import (
    ServerlessValidationLauncher,
)


class TestHubContentOverride(unittest.TestCase):
    """Test Hub content override functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OmegaConf.create(
            {
                "serverless_config": {
                    "region": "us-west-2",
                    "endpoint": "beta",
                    "hub_account_id": "123456789012",
                    "hub_name": "TestPrivateHub",
                    "model_version": "2.7.14",
                },
                "container_info": {
                    "llmft": {"smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0"},
                    "verl": {
                        "smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.1.0-smtj"
                    },
                },
            }
        )

        self.job_recorder = Mock()

    @patch("boto3.Session")
    @patch("boto3.client")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_override_hub_content_always_updates(
        self, mock_open, mock_map_path, mock_recipes_folder, mock_boto_client, mock_boto_session
    ):
        """Test that override always pushes local config to hub, even when images already match"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")

        # Mock model mapping
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        )

        # Mock SageMaker client
        mock_sagemaker = Mock()
        mock_boto_client.return_value = mock_sagemaker

        # Hub already has the SAME image (matching config)
        hub_doc = {
            "RecipeCollection": [
                {
                    "CustomizationTechnique": "SFT",
                    "SmtjImageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0",
                }
            ]
        }
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(hub_doc),
            "HubContentDisplayName": "Test Model",
        }

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        with patch.object(launcher, "_extract_run_name_from_recipe", return_value="llama-3-1-8b-instruct"):
            launcher._override_hub_content(recipe="fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml")

        # Verify import_hub_content was called (always override)
        mock_sagemaker.import_hub_content.assert_called_once()

    @patch("boto3.Session")
    @patch("boto3.client")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_override_hub_content_updates_mismatched_image(
        self, mock_open, mock_map_path, mock_recipes_folder, mock_boto_client, mock_boto_session
    ):
        """Test that override updates hub content when images differ"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")

        # Mock model mapping
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        )

        # Mock SageMaker client
        mock_sagemaker = Mock()
        mock_boto_client.return_value = mock_sagemaker

        # Hub has an OLD image (mismatched)
        hub_doc = {"RecipeCollection": [{"CustomizationTechnique": "SFT", "SmtjImageUri": "old-image:v1.0.0"}]}
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(hub_doc),
            "HubContentDisplayName": "Test Model",
        }

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        with patch.object(launcher, "_extract_run_name_from_recipe", return_value="llama-3-1-8b-instruct"):
            launcher._override_hub_content(recipe="fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml")

        # Verify import_hub_content was called
        mock_sagemaker.import_hub_content.assert_called_once()

        # Verify the updated document contains the new image
        call_args = mock_sagemaker.import_hub_content.call_args
        updated_doc = json.loads(call_args[1]["HubContentDocument"])
        self.assertEqual(
            updated_doc["RecipeCollection"][0]["SmtjImageUri"],
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0",
        )

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_override_hub_content_all_models(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test that override targets all models when no target_run_name specified"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")

        # Mock model mapping with multiple models
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct",
                "verl-grpo-gpt-oss-20b-lora": "openai-reasoning-gpt-oss-20b",
            }
        )

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        with patch.object(launcher, "_auto_update_hub_content", return_value=True) as mock_auto_update:
            launcher._override_hub_content()

            # Verify auto-update was called with ALL models
            mock_auto_update.assert_called_once()
            called_models = mock_auto_update.call_args[0][0]
            self.assertIn("llama-3-1-8b-instruct", called_models)
            self.assertIn("verl-grpo-gpt-oss-20b-lora", called_models)

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_override_hub_content_target_model_not_found(
        self, mock_open, mock_map_path, mock_recipes_folder, mock_boto
    ):
        """Test that override logs warning when target model not in mapping"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")

        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        )

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        with patch.object(launcher, "_auto_update_hub_content") as mock_auto_update, patch.object(
            launcher, "_extract_run_name_from_recipe", return_value="nonexistent-model"
        ):
            launcher._override_hub_content(recipe="fine-tuning/llama/fake_nonexistent_recipe.yaml")

            # Should NOT call auto_update since model not in mapping
            mock_auto_update.assert_not_called()

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_no_container_info(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test when config has no container_info"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")

        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({})

        # Config without container_info
        config_no_container = OmegaConf.create({"serverless_config": {"region": "us-west-2", "endpoint": "beta"}})

        # Should create launcher without errors
        launcher = ServerlessValidationLauncher(self.job_recorder, config_no_container)
        self.assertIsNotNone(launcher)

        # Override should return early without error
        with patch.object(launcher, "_auto_update_hub_content") as mock_auto_update:
            launcher._override_hub_content()
            mock_auto_update.assert_not_called()


class TestGetExpectedImage(unittest.TestCase):
    """Test expected image resolution"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OmegaConf.create(
            {
                "serverless_config": {
                    "region": "us-west-2",
                    "endpoint": "beta",
                    "hub_account_id": "123456789012",
                    "hub_name": "TestPrivateHub",
                    "model_version": "2.7.14",
                },
                "container_info": {
                    "llmft": {"smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0"},
                    "verl": {
                        "smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.1.0-smtj"
                    },
                },
            }
        )
        self.job_recorder = Mock()

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_get_expected_image_llmft(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test getting expected image for LLMFT recipe type"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({})

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        image = launcher._get_expected_image("llmft")
        self.assertEqual(image, "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0")

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_get_expected_image_verl(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test getting expected image for VERL recipe type"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({})

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        image = launcher._get_expected_image("verl")
        self.assertEqual(image, "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.1.0-smtj")

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_get_expected_image_unknown(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test getting expected image for unknown recipe type"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({})

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        image = launcher._get_expected_image("unknown")
        self.assertIsNone(image)


class TestUpdateSingleHubContent(unittest.TestCase):
    """Test single model hub content override"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = OmegaConf.create(
            {
                "serverless_config": {
                    "region": "us-west-2",
                    "endpoint": "beta",
                    "hub_account_id": "123456789012",
                    "hub_name": "TestPrivateHub",
                    "model_version": "2.7.14",
                },
                "container_info": {
                    "llmft": {"smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0"},
                    "verl": {
                        "smjobs": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.1.0-smtj"
                    },
                },
            }
        )
        self.job_recorder = Mock()

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_update_single_always_imports(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test that _update_single_hub_content always calls import even when images match"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        )

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        mock_sagemaker = Mock()
        # Hub already has the correct image
        hub_doc = {
            "RecipeCollection": [
                {
                    "CustomizationTechnique": "SFT",
                    "SmtjImageUri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0",
                }
            ]
        }
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(hub_doc),
            "HubContentDisplayName": "Test Model",
        }

        result = launcher._update_single_hub_content(mock_sagemaker, "llama-3-1-8b-instruct")

        # Should still import even though images match
        self.assertTrue(result)
        mock_sagemaker.import_hub_content.assert_called_once()

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_update_single_sets_correct_image(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test that _update_single_hub_content sets the correct expected image"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        )

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        mock_sagemaker = Mock()
        hub_doc = {
            "RecipeCollection": [
                {"CustomizationTechnique": "SFT", "SmtjImageUri": "old-image:v1.0.0"},
                {"CustomizationTechnique": "DPO", "SmtjImageUri": "old-image:v1.0.0"},
            ]
        }
        mock_sagemaker.describe_hub_content.return_value = {
            "HubContentDocument": json.dumps(hub_doc),
        }

        launcher._update_single_hub_content(mock_sagemaker, "llama-3-1-8b-instruct")

        # Verify all recipes got the expected image
        call_args = mock_sagemaker.import_hub_content.call_args
        updated_doc = json.loads(call_args[1]["HubContentDocument"])
        for recipe in updated_doc["RecipeCollection"]:
            self.assertEqual(
                recipe["SmtjImageUri"],
                "123456789012.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.1.0",
            )

    @patch("boto3.Session")
    @patch("scripts.validations.validation_launchers.serverless_launcher.get_recipes_folder")
    @patch("scripts.validations.validation_launchers.serverless_launcher.JUMPSTART_MODEL_ID_MAP_PATH")
    @patch("builtins.open")
    def test_update_single_model_not_in_mapping(self, mock_open, mock_map_path, mock_recipes_folder, mock_boto):
        """Test that update returns False when model not in mapping"""
        mock_recipes_folder.return_value = "/fake/path"
        mock_map_path.__str__ = Mock(return_value="/fake/map.json")
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({})

        launcher = ServerlessValidationLauncher(self.job_recorder, self.config)

        mock_sagemaker = Mock()
        result = launcher._update_single_hub_content(mock_sagemaker, "nonexistent-model")

        self.assertFalse(result)
        mock_sagemaker.import_hub_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
