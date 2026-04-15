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
    _model_params_cache,
    copy_file,
    get_gpu_info,
    get_gpu_memory_gb,
    get_optimizer_type,
    get_sequence_length_range,
    load_model_params,
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
class TestGetGpuInfo:
    def test_returns_memory_and_count(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [
                {"GpuInfo": {"TotalGpuMemoryInMiB": 655360, "Gpus": [{"MemoryInfo": {"SizeInMiB": 81920}}]}}
            ]
        }
        memory, count = get_gpu_info("ml.p5.48xlarge")
        assert memory == 80.0
        assert count == 8

    def test_g5_12xlarge_4_gpus(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [
                {"GpuInfo": {"TotalGpuMemoryInMiB": 98304, "Gpus": [{"MemoryInfo": {"SizeInMiB": 24576}}]}}
            ]
        }
        memory, count = get_gpu_info("ml.g5.12xlarge")
        assert memory == 24.0
        assert count == 4


@patch("auto_configurator.utils.util.boto3.client")
class TestGetGpuMemoryGb:
    def test_p5_instance(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [
                {"GpuInfo": {"TotalGpuMemoryInMiB": 655360, "Gpus": [{"MemoryInfo": {"SizeInMiB": 81920}}]}}
            ]
        }
        memory = get_gpu_memory_gb("ml.p5.48xlarge")
        assert memory == 80.0
        mock_ec2.describe_instance_types.assert_called_once_with(InstanceTypes=["p5.48xlarge"])

    def test_strips_ml_prefix(self, mock_boto_client):
        mock_ec2 = MagicMock()
        mock_boto_client.return_value = mock_ec2
        mock_ec2.describe_instance_types.return_value = {
            "InstanceTypes": [
                {"GpuInfo": {"TotalGpuMemoryInMiB": 327680, "Gpus": [{"MemoryInfo": {"SizeInMiB": 40960}}]}}
            ]
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

    def test_returns_single_value_as_list(self):
        from unittest.mock import MagicMock

        auto_config = MagicMock()
        auto_config.autotune_config.get.return_value = 8192

        result = get_sequence_length_range(auto_config)

        assert result == [8192]


class TestLoadModelParams:
    def setup_method(self):
        _model_params_cache.clear()

    def _mock_hf_config(self, **kwargs):
        """Create a mock HF config with given attributes."""
        defaults = {
            "vocab_size": 128256,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "max_position_embeddings": 131072,
        }
        defaults.update(kwargs)
        cfg = MagicMock(spec=[])
        for k, v in defaults.items():
            setattr(cfg, k, v)
        return cfg

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_fetches_from_hf(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config()

        result = load_model_params("some-org/some-model")

        mock_from_pretrained.assert_called_once_with("some-org/some-model", trust_remote_code=True, token=None)
        assert result.vocab_size == 128256
        assert result.hidden_width == 4096
        assert result.num_heads == 32
        assert result.num_key_value_heads == 8
        assert result.num_layers == 32
        assert result.intermediate_size == 14336
        assert result.max_context_width == 131072
        assert result.moe is False

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_passes_hf_token(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config()

        load_model_params("some-org/some-model", hf_token="hf_test123")

        mock_from_pretrained.assert_called_once_with("some-org/some-model", trust_remote_code=True, token="hf_test123")

    @patch.dict("os.environ", {"HF_TOKEN": "hf_env_token"})
    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_uses_env_token_when_no_explicit_token(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config()

        load_model_params("some-org/some-model")

        mock_from_pretrained.assert_called_once_with(
            "some-org/some-model", trust_remote_code=True, token="hf_env_token"
        )

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_caches_result(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config()

        load_model_params("some-org/some-model")
        load_model_params("some-org/some-model")

        assert mock_from_pretrained.call_count == 1

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_returns_copy_from_cache(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config()

        result1 = load_model_params("some-org/some-model")
        result2 = load_model_params("some-org/some-model")

        assert result1 == result2
        assert result1 is not result2

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_multimodal_uses_text_config(self, mock_from_pretrained):
        text_cfg = self._mock_hf_config(num_hidden_layers=40)
        outer_cfg = MagicMock()
        outer_cfg.text_config = text_cfg
        mock_from_pretrained.return_value = outer_cfg

        result = load_model_params("some-org/vision-model")

        assert result.num_layers == 40

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_moe_model(self, mock_from_pretrained):
        mock_from_pretrained.return_value = self._mock_hf_config(
            num_local_experts=32,
            num_experts_per_tok=4,
        )

        result = load_model_params("some-org/moe-model")

        assert result.moe is True
        assert result.num_local_experts == 32
        assert result.num_experts_per_tok == 4

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_falls_back_to_static_map_on_hf_error(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = Exception("not found")

        result = load_model_params("meta-llama/Llama-3.1-8B-Instruct")

        assert result.vocab_size == 128256
        assert result.hidden_width == 4096

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_static_map_fallback_substring_match(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = Exception("not found")

        result = load_model_params("/fsx/models/meta-llama/Llama-3.1-8B-Instruct")

        assert result.hidden_width == 4096

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_raises_when_unknown_and_hf_fails(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = Exception("not found")

        with pytest.raises(ValueError, match="Unknown model.*totally-unknown"):
            load_model_params("totally-unknown/model")

    @patch("auto_configurator.utils.util.AutoConfig.from_pretrained")
    def test_num_key_value_heads_defaults_to_num_heads(self, mock_from_pretrained):
        cfg = self._mock_hf_config()
        del cfg.num_key_value_heads
        mock_from_pretrained.return_value = cfg

        result = load_model_params("some-org/no-gqa-model")

        assert result.num_key_value_heads == 32  # falls back to num_attention_heads
