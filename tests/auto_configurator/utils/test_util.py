# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import json
from unittest.mock import MagicMock, patch

import pytest

from auto_configurator.utils.util import (
    MODEL_ARCHITECTURES,
    AutoConfiguratorLogger,
    OptimizerType,
    copy_file,
    get_gpu_memory_gb,
    get_optimizer_type,
    get_sequence_length_range,
    prettify,
)


class TestOptimizerType:
    def test_llmft_value(self):
        assert OptimizerType.LLMFT.value == "llmft"

    def test_verl_value(self):
        assert OptimizerType.VERL.value == "verl"


class TestGetOptimizerType:
    def test_llmft_recipe(self):
        assert get_optimizer_type("llmft_llama3_8b") == OptimizerType.LLMFT

    def test_verl_recipe(self):
        assert get_optimizer_type("verl_qwen_7b") == OptimizerType.VERL

    def test_case_insensitive(self):
        assert get_optimizer_type("LLMFT_Model") == OptimizerType.LLMFT
        assert get_optimizer_type("VERL_Model") == OptimizerType.VERL

    def test_unknown_recipe(self):
        with pytest.raises(ValueError, match="Unable to determine optimizer type"):
            get_optimizer_type("unknown_recipe")


class TestModelArchitectures:
    def test_llama_3_1_8b_exists(self):
        assert "meta-llama/Llama-3.1-8B-Instruct" in MODEL_ARCHITECTURES

    def test_model_params_structure(self):
        model = MODEL_ARCHITECTURES["meta-llama/Llama-3.1-8B-Instruct"]
        assert model.vocab_size == 128256
        assert model.hidden_width == 4096
        assert model.num_heads == 32
        assert model.num_layers == 32
        assert model.moe is False

    def test_moe_model(self):
        model = MODEL_ARCHITECTURES["openai/gpt-oss-120b-bf16"]
        assert model.moe is True
        assert model.num_local_experts == 128
        assert model.num_experts_per_tok == 4


@patch("auto_configurator.utils.util.boto3.client")
class TestGetGpuMemoryGb:
    def test_p5_instance(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [{"GpuInfo": {"Gpus": [{"MemoryInfo": {"SizeInMiB": 81920}}]}}]
        }
        memory = get_gpu_memory_gb("ml.p5.48xlarge")
        assert memory == 80.0
        mock_ec2.describe_instance_types.assert_called_once_with(InstanceTypes=["p5.48xlarge"])

    def test_strips_ml_prefix(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [{"GpuInfo": {"Gpus": [{"MemoryInfo": {"SizeInMiB": 40960}}]}}]
        }
        get_gpu_memory_gb("ml.p4d.24xlarge")
        mock_ec2.describe_instance_types.assert_called_once_with(InstanceTypes=["p4d.24xlarge"])


class TestPrettify:
    def test_dict_formatting(self):
        data = {"key": "value", "num": 42}
        result = prettify(data)
        assert json.loads(result) == data
        assert "\n" in result  # Check indentation

    def test_list_formatting(self):
        data = [1, 2, 3]
        result = prettify(data)
        assert json.loads(result) == data


class TestAutoConfiguratorLogger:
    def test_logger_creation(self):
        logger = AutoConfiguratorLogger()


class TestCopyFile:
    def test_copy_file_success(self, tmp_path):
        """Test successful file copy"""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "subdir" / "dest.txt"

        result = copy_file(str(source), str(dest))

        assert result == str(dest)
        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_copy_file_source_not_exists(self, tmp_path):
        """Test copy_file returns empty string when source doesn't exist"""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"

        result = copy_file(str(source), str(dest))

        assert result == ""
        assert not dest.exists()

    def test_copy_file_empty_source_path(self, tmp_path):
        """Test copy_file returns empty string when source path is empty"""
        dest = tmp_path / "dest.txt"

        result = copy_file("", str(dest))

        assert result == ""
        assert not dest.exists()

    def test_copy_file_creates_parent_dirs(self, tmp_path):
        """Test copy_file creates parent directories"""
        source = tmp_path / "source.txt"
        source.write_text("test content")
        dest = tmp_path / "a" / "b" / "c" / "dest.txt"

        result = copy_file(str(source), str(dest))

        assert result == str(dest)
        assert dest.exists()
        assert dest.parent.exists()

    def test_logger_level(self):
        import logging

        logger = AutoConfiguratorLogger()
        assert logger.get_logger().level == logging.DEBUG


class TestGetSequenceLengthRange:
    def test_returns_default_range(self):
        from unittest.mock import MagicMock

        auto_config = MagicMock()
        auto_config.autotune_config.get.return_value = "auto"

        result = get_sequence_length_range(auto_config)

        assert result == [4096, 8192, 16384, 32768, 65536, 131072]

    def test_returns_custom_sequence_lengths(self):
        from unittest.mock import MagicMock

        auto_config = MagicMock()
        auto_config.autotune_config.get.return_value = [4096, 8192]

        result = get_sequence_length_range(auto_config)

        assert result == [4096, 8192]
