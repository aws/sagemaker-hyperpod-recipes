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

"""
Unit tests for VERL sequence length extraction and formatting logic.
"""

import pytest
from omegaconf import OmegaConf

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)


@pytest.fixture
def processor(tmp_path, monkeypatch):
    """Create a VerlRecipeTemplateProcessor with minimal dependencies stubbed out."""
    template_file = tmp_path / "template.json"
    template_file.write_text('{"templates": {}}')

    regional_file = tmp_path / "regional.json"
    regional_file.write_text('{"verl": {}}')

    model_map_file = tmp_path / "model_map.json"
    model_map_file.write_text("{}")

    monkeypatch.setattr(
        VerlRecipeTemplateProcessor,
        "_load_template",
        lambda self: None,
    )

    proc = VerlRecipeTemplateProcessor(staging_cfg={}, template_path=str(template_file))
    return proc


class TestExtractSequenceLength:
    """Tests for VerlRecipeTemplateProcessor._extract_sequence_length."""

    def test_rl_recipe_sums_prompt_and_response_length(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_prompt_length": 4096,
                        "max_response_length": 8192,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result == 4096 + 8192

    def test_sft_recipe_uses_max_length(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_length": 2048,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result == 2048

    def test_falls_back_to_max_length_when_max_prompt_length_is_none(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_prompt_length": None,
                        "max_length": 16384,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result == 16384

    def test_falls_back_to_max_length_when_max_response_length_is_none(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_prompt_length": 4096,
                        "max_response_length": None,
                        "max_length": 8192,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result == 8192

    def test_returns_none_when_both_fields_missing(self, processor):
        recipe_cfg = OmegaConf.create({"training_config": {"data": {}}})
        result = processor._extract_sequence_length(recipe_cfg)
        assert result is None

    def test_returns_none_when_all_fields_explicitly_none(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_prompt_length": None,
                        "max_response_length": None,
                        "max_length": None,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result is None

    def test_does_not_sum_when_only_max_prompt_length_present(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_prompt_length": 4096,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result is None

    def test_ignores_max_response_length_without_max_prompt_length(self, processor):
        recipe_cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {
                        "max_response_length": 8192,
                        "max_length": 4096,
                    }
                }
            }
        )
        result = processor._extract_sequence_length(recipe_cfg)
        assert result == 4096


class TestFormatSequenceLength:
    """Tests for BaseRecipeTemplateProcessor.format_sequence_length."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (1024, "1K"),
            (2048, "2K"),
            (4096, "4K"),
            (8192, "8K"),
            (16384, "16K"),
            (32768, "32K"),
            (65536, "64K"),
            (131072, "128K"),
        ],
    )
    def test_exact_matches(self, processor, input_val, expected):
        assert processor.format_sequence_length(input_val) == expected

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (10240, "8K"),
            (5000, "4K"),
            (3000, "2K"),
            (1500, "1K"),
            (100000, "64K"),
            (50000, "32K"),
            (20000, "16K"),
        ],
    )
    def test_rounds_down_to_nearest_bucket(self, processor, input_val, expected):
        assert processor.format_sequence_length(input_val) == expected

    def test_value_smaller_than_minimum_returns_1k(self, processor):
        assert processor.format_sequence_length(512) == "1K"
        assert processor.format_sequence_length(1) == "1K"

    def test_value_larger_than_maximum_returns_128k(self, processor):
        assert processor.format_sequence_length(200000) == "128K"
        assert processor.format_sequence_length(131073) == "128K"

    def test_zero_returns_1k(self, processor):
        assert processor.format_sequence_length(0) == "1K"

    def test_negative_value_returns_1k(self, processor):
        assert processor.format_sequence_length(-1) == "1K"
