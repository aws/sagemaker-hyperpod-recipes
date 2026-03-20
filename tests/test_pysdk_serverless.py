"""Unit tests for scripts/validations/pysdk_serverless.py"""

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

mock_sft_trainer = MagicMock(__name__="SFTTrainer")
mock_dpo_trainer = MagicMock(__name__="DPOTrainer")
mock_rlaif_trainer = MagicMock(__name__="RLAIFTrainer")
mock_rlvr_trainer = MagicMock(__name__="RLVRTrainer")

mock_training_type = MagicMock()
mock_training_type.FULL = "FULL"
mock_training_type.LORA = "LORA"

mock_train_module = ModuleType("sagemaker.train")
mock_train_module.SFTTrainer = mock_sft_trainer
mock_train_module.DPOTrainer = mock_dpo_trainer
mock_train_module.RLAIFTrainer = mock_rlaif_trainer
mock_train_module.RLVRTrainer = mock_rlvr_trainer

mock_train_common_module = ModuleType("sagemaker.train.common")
mock_train_common_module.TrainingType = mock_training_type

if "sagemaker.train" not in sys.modules or "sagemaker.core" not in sys.modules:
    sys.modules.setdefault("sagemaker", ModuleType("sagemaker"))
    sys.modules["sagemaker.train"] = mock_train_module
    sys.modules["sagemaker.train.common"] = mock_train_common_module

from scripts.validations.pysdk_serverless import (
    TRAINER_MAPPING,
    convert_to_jumpstart_model_id,
    create_trainer,
    detect_lora_or_full_from_filename,
    detect_training_type_from_filename,
    get_trainer_class,
    get_training_type_enum,
    launch_training,
)


class TestDetectTrainingTypeFromFilename(unittest.TestCase):
    """Tests for detect_training_type_from_filename."""

    def test_rlvr_detection(self):
        self.assertEqual(detect_training_type_from_filename("verl_grpo_rlvr_qwen_7b_lora.yaml"), "rlvr")

    def test_rlaif_detection(self):
        self.assertEqual(detect_training_type_from_filename("verl_grpo_rlaif_llama_8b_fft.yaml"), "rlaif")

    def test_dpo_detection(self):
        self.assertEqual(detect_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_dpo.yaml"), "dpo")

    def test_sft_detection(self):
        self.assertEqual(
            detect_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml"), "sft"
        )

    def test_default_to_sft(self):
        """Recipes without specific keywords default to sft."""
        self.assertEqual(detect_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_lora.yaml"), "sft")

    def test_none_input(self):
        self.assertEqual(detect_training_type_from_filename(None), "sft")

    def test_empty_string(self):
        self.assertEqual(detect_training_type_from_filename(""), "sft")

    def test_rlvr_takes_priority_over_other_keywords(self):
        """RLVR should match before DPO or SFT."""
        self.assertEqual(detect_training_type_from_filename("rlvr_dpo_sft_mixed.yaml"), "rlvr")

    def test_rlaif_takes_priority_over_dpo(self):
        """RLAIF should match before DPO."""
        self.assertEqual(detect_training_type_from_filename("rlaif_dpo_test.yaml"), "rlaif")

    def test_case_insensitive(self):
        self.assertEqual(detect_training_type_from_filename("LLMFT_RLVR_TEST.yaml"), "rlvr")


class TestDetectLoraOrFullFromFilename(unittest.TestCase):
    """Tests for detect_lora_or_full_from_filename."""

    def test_fft_detection(self):
        self.assertEqual(detect_lora_or_full_from_filename("llmft_llama3_1_8b_sft_fft.yaml"), "full")

    def test_full_fine_tuning_detection(self):
        self.assertEqual(detect_lora_or_full_from_filename("llmft_llama3_full_fine_tuning.yaml"), "full")

    def test_lora_default(self):
        self.assertEqual(detect_lora_or_full_from_filename("llmft_llama3_1_8b_sft_lora.yaml"), "lora")

    def test_none_input(self):
        self.assertEqual(detect_lora_or_full_from_filename(None), "lora")

    def test_empty_string(self):
        self.assertEqual(detect_lora_or_full_from_filename(""), "lora")

    def test_no_fft_keyword(self):
        self.assertEqual(detect_lora_or_full_from_filename("llmft_llama3_1_8b_sft.yaml"), "lora")


class TestGetTrainerClass(unittest.TestCase):
    """Tests for get_trainer_class."""

    def test_sft_returns_sft_trainer(self):
        self.assertEqual(get_trainer_class("sft"), mock_sft_trainer)

    def test_dpo_returns_dpo_trainer(self):
        self.assertEqual(get_trainer_class("dpo"), mock_dpo_trainer)

    def test_rlaif_returns_rlaif_trainer(self):
        self.assertEqual(get_trainer_class("rlaif"), mock_rlaif_trainer)

    def test_rlvr_returns_rlvr_trainer(self):
        self.assertEqual(get_trainer_class("rlvr"), mock_rlvr_trainer)

    def test_case_insensitive(self):
        self.assertEqual(get_trainer_class("SFT"), mock_sft_trainer)

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_trainer_class("invalid")
        self.assertIn("Unknown training type", str(ctx.exception))
        self.assertIn("invalid", str(ctx.exception))

    def test_trainer_mapping_has_all_types(self):
        """Verify TRAINER_MAPPING covers all expected types."""
        expected = {"sft", "dpo", "rlaif", "rlvr"}
        self.assertEqual(set(TRAINER_MAPPING.keys()), expected)


class TestGetTrainingTypeEnum(unittest.TestCase):
    """Tests for get_training_type_enum."""

    def test_full_returns_full_type(self):
        self.assertEqual(get_training_type_enum("full"), mock_training_type.FULL)

    def test_lora_returns_lora_type(self):
        self.assertEqual(get_training_type_enum("lora"), mock_training_type.LORA)

    def test_anything_else_returns_lora(self):
        self.assertEqual(get_training_type_enum("something"), mock_training_type.LORA)


class TestConvertToJumpstartModelId(unittest.TestCase):
    """Tests for convert_to_jumpstart_model_id."""

    @patch("scripts.validations.pysdk_serverless.load_jumpstart_model_id_map")
    def test_known_model_returns_jumpstart_id(self, mock_load):
        mock_load.return_value = {"llama-3-2-1b-instruct": "meta-textgeneration-llama-3-2-1b-instruct"}
        result = convert_to_jumpstart_model_id("llama-3-2-1b-instruct")
        self.assertEqual(result, "meta-textgeneration-llama-3-2-1b-instruct")

    @patch("scripts.validations.pysdk_serverless.load_jumpstart_model_id_map")
    def test_unknown_model_returns_original(self, mock_load):
        mock_load.return_value = {}
        result = convert_to_jumpstart_model_id("unknown-model")
        self.assertEqual(result, "unknown-model")


class TestCreateTrainer(unittest.TestCase):
    """Tests for create_trainer."""

    def setUp(self):
        """Reset mock call records before each test."""
        mock_sft_trainer.reset_mock()
        mock_dpo_trainer.reset_mock()
        mock_rlaif_trainer.reset_mock()
        mock_rlvr_trainer.reset_mock()

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_sft_trainer_created(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
        )

        mock_sft_trainer.assert_called_once()
        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertEqual(call_kwargs["model"], "js-model-id")
        self.assertEqual(call_kwargs["training_dataset"], "s3://bucket/data")
        self.assertEqual(call_kwargs["model_package_group"], "test-group")

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_dpo_trainer_created(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="dpo",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
        )

        mock_dpo_trainer.assert_called_once()

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_rlaif_gets_custom_reward_function(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="rlaif",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            custom_reward_function="arn:aws:sagemaker:us-west-2:123:evaluator/test",
        )

        call_kwargs = mock_rlaif_trainer.call_args[1]
        self.assertEqual(call_kwargs["custom_reward_function"], "arn:aws:sagemaker:us-west-2:123:evaluator/test")

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_rlvr_gets_custom_reward_function(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="rlvr",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            custom_reward_function="arn:aws:lambda:us-west-2:123:function:reward",
        )

        call_kwargs = mock_rlvr_trainer.call_args[1]
        self.assertEqual(call_kwargs["custom_reward_function"], "arn:aws:lambda:us-west-2:123:function:reward")

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_none_values_filtered_out(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            validation_dataset=None,
            mlflow_resource_arn=None,
            s3_output_path=None,
        )

        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertNotIn("validation_dataset", call_kwargs)
        self.assertNotIn("mlflow_resource_arn", call_kwargs)
        self.assertNotIn("s3_output_path", call_kwargs)

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_accept_eula_passed_through(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            accept_eula=True,
        )

        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertTrue(call_kwargs["accept_eula"])

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_lora_training_type(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            lora_or_full="lora",
        )

        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertEqual(call_kwargs["training_type"], mock_training_type.LORA)

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_full_training_type(self, mock_convert):
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            lora_or_full="full",
        )

        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertEqual(call_kwargs["training_type"], mock_training_type.FULL)

    @patch("scripts.validations.pysdk_serverless.convert_to_jumpstart_model_id")
    def test_sft_does_not_get_custom_reward_function(self, mock_convert):
        """SFT trainer should not receive custom_reward_function even if provided."""
        mock_convert.return_value = "js-model-id"

        create_trainer(
            model="test-model",
            training_type="sft",
            training_dataset="s3://bucket/data",
            model_package_group="test-group",
            custom_reward_function="arn:aws:some:evaluator",
        )

        call_kwargs = mock_sft_trainer.call_args[1]
        self.assertNotIn("custom_reward_function", call_kwargs)


class TestLaunchTraining(unittest.TestCase):
    """Tests for launch_training."""

    def test_launch_with_wait(self):
        mock_trainer = MagicMock()
        mock_job = MagicMock()
        mock_job.training_job_name = "test-job-123"
        mock_trainer.train.return_value = mock_job

        result = launch_training(trainer=mock_trainer, wait=True)

        mock_trainer.train.assert_called_once_with(wait=True)
        self.assertEqual(result.training_job_name, "test-job-123")

    def test_launch_without_wait(self):
        mock_trainer = MagicMock()
        mock_job = MagicMock()
        mock_job.training_job_name = "test-job-456"
        mock_trainer.train.return_value = mock_job

        result = launch_training(trainer=mock_trainer, wait=False)

        mock_trainer.train.assert_called_once_with(wait=False)
        self.assertEqual(result.training_job_name, "test-job-456")

    def test_launch_with_dataset_overrides(self):
        mock_trainer = MagicMock()
        mock_job = MagicMock()
        mock_job.training_job_name = "test-job-789"
        mock_trainer.train.return_value = mock_job

        launch_training(
            trainer=mock_trainer,
            training_dataset="s3://override/train",
            validation_dataset="s3://override/val",
            wait=False,
        )

        mock_trainer.train.assert_called_once_with(
            wait=False,
            training_dataset="s3://override/train",
            validation_dataset="s3://override/val",
        )

    def test_launch_without_dataset_overrides(self):
        """When no dataset overrides, train() should only get wait kwarg."""
        mock_trainer = MagicMock()
        mock_job = MagicMock()
        mock_job.training_job_name = "test-job-no-override"
        mock_trainer.train.return_value = mock_job

        launch_training(trainer=mock_trainer, wait=True)

        mock_trainer.train.assert_called_once_with(wait=True)


class TestLoadJumpstartModelIdMap(unittest.TestCase):
    """Tests for load_jumpstart_model_id_map."""

    def test_loads_valid_json(self):
        from scripts.validations.pysdk_serverless import load_jumpstart_model_id_map

        result = load_jumpstart_model_id_map()
        self.assertIsInstance(result, dict)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_missing_file_returns_empty_dict(self, mock_open):
        from scripts.validations.pysdk_serverless import load_jumpstart_model_id_map

        result = load_jumpstart_model_id_map()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
