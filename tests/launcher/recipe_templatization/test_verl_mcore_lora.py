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
Unit tests for Mcore/Nemotron LoRA rank backfill and nested lora.rank PEFT detection
in VerlRecipeTemplateProcessor.
"""

from unittest.mock import patch

import pytest

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)


@pytest.fixture
def processor(tmp_path, monkeypatch):
    """Create a VerlRecipeTemplateProcessor with minimal dependencies stubbed out."""
    template_file = tmp_path / "template.json"
    template_file.write_text('{"templates": {}}')

    monkeypatch.setattr(
        VerlRecipeTemplateProcessor,
        "_load_template",
        lambda self: None,
    )

    proc = VerlRecipeTemplateProcessor(staging_cfg={}, template_path=str(template_file))
    proc.template_data = {"templates": {}}
    proc.recipe_jumpstart_model_id_mapping = {}
    proc._original_regional_parameters = {}
    return proc


class TestExtractPeftTypeNestedLora:
    """Tests for _extract_peft_type_from_config with nested model.lora.rank (Mcore/Nemotron)."""

    def test_peft_type_from_nested_lora_rank_rl_recipe(self, processor):
        """When lora_rank is 0 but model.lora.rank > 0, returns LoRA (RL recipe structure)."""
        recipe_cfg = {
            "training_config": {
                "actor_rollout_ref": {
                    "model": {
                        "lora_rank": 0,
                        "lora": {
                            "rank": 128,
                            "alpha": 256,
                        },
                    }
                }
            }
        }
        result = processor._extract_peft_type_from_config(recipe_cfg)
        assert result == "LoRA"

    def test_peft_type_from_nested_lora_rank_sft_recipe(self, processor):
        """When lora_rank is 0 but model.lora.rank > 0, returns LoRA (SFT recipe structure)."""
        recipe_cfg = {
            "training_config": {
                "model": {
                    "lora_rank": 0,
                    "lora": {
                        "rank": 128,
                        "alpha": 256,
                    },
                }
            }
        }
        result = processor._extract_peft_type_from_config(recipe_cfg)
        assert result == "LoRA"

    def test_peft_type_none_when_no_lora(self, processor):
        """When lora_rank is 0 and no nested lora config, returns None (FFT)."""
        recipe_cfg = {
            "training_config": {
                "actor_rollout_ref": {
                    "model": {
                        "lora_rank": 0,
                    }
                }
            }
        }
        result = processor._extract_peft_type_from_config(recipe_cfg)
        assert result is None

    def test_peft_type_from_top_level_lora_rank(self, processor):
        """When lora_rank > 0, returns LoRA without checking nested config."""
        recipe_cfg = {
            "training_config": {
                "actor_rollout_ref": {
                    "model": {
                        "lora_rank": 64,
                    }
                }
            }
        }
        result = processor._extract_peft_type_from_config(recipe_cfg)
        assert result == "LoRA"

    def test_peft_type_nested_lora_rank_zero(self, processor):
        """When lora_rank is 0 and nested lora.rank is also 0, returns None."""
        recipe_cfg = {
            "training_config": {
                "model": {
                    "lora_rank": 0,
                    "lora": {
                        "rank": 0,
                    },
                }
            }
        }
        result = processor._extract_peft_type_from_config(recipe_cfg)
        assert result is None


class TestProcessRecipeLoraRankBackfill:
    """Tests for process_recipe lora_rank backfill from model.lora.rank (Mcore/Nemotron)."""

    def test_backfill_lora_rank_from_nested_lora_config(self, processor, tmp_path):
        """When lora_rank default is 0, backfill from model.lora.rank in the recipe."""
        # Create a recipe file with model.lora.rank = 128 but model.lora_rank = 0
        recipe_content = """
training_config:
  model:
    lora_rank: 0
    lora:
      rank: 128
      alpha: 256
"""
        recipe_file = tmp_path / "test_recipe.yaml"
        recipe_file.write_text(recipe_content)

        # Set up processor state to simulate process_recipe context
        processor.recipe_override_parameters = {
            "lora_rank": {"default": 0, "min": 8, "max": 128},
        }

        # Mock the parent process_recipe and _load_recipe_config
        with patch.object(VerlRecipeTemplateProcessor.__bases__[0], "process_recipe", return_value={}), patch.object(
            processor,
            "_load_recipe_config",
            return_value={
                "training_config": {
                    "model": {
                        "lora_rank": 0,
                        "lora": {"rank": 128, "alpha": 256},
                    }
                }
            },
        ):
            processor.process_recipe(recipe_file_path=str(recipe_file))

        assert processor.recipe_override_parameters["lora_rank"]["default"] == 128

    def test_no_backfill_when_lora_rank_already_set(self, processor, tmp_path):
        """When lora_rank default is already > 0, no backfill occurs."""
        processor.recipe_override_parameters = {
            "lora_rank": {"default": 64, "min": 8, "max": 128},
        }

        with patch.object(VerlRecipeTemplateProcessor.__bases__[0], "process_recipe", return_value={}):
            processor.process_recipe(recipe_file_path=str(tmp_path / "dummy.yaml"))

        # Should remain unchanged
        assert processor.recipe_override_parameters["lora_rank"]["default"] == 64

    def test_backfill_from_actor_rollout_ref_model(self, processor, tmp_path):
        """Backfill works for RL recipes where model is under actor_rollout_ref."""
        processor.recipe_override_parameters = {
            "lora_rank": {"default": 0, "min": 8, "max": 128},
        }

        with patch.object(VerlRecipeTemplateProcessor.__bases__[0], "process_recipe", return_value={}), patch.object(
            processor,
            "_load_recipe_config",
            return_value={
                "training_config": {
                    "actor_rollout_ref": {
                        "model": {
                            "lora_rank": 0,
                            "lora": {"rank": 128, "alpha": 256},
                        }
                    }
                }
            },
        ):
            processor.process_recipe(recipe_file_path=str(tmp_path / "dummy.yaml"))

        assert processor.recipe_override_parameters["lora_rank"]["default"] == 128

    def test_no_backfill_when_no_nested_lora(self, processor, tmp_path):
        """When no nested lora config exists, default stays 0."""
        processor.recipe_override_parameters = {
            "lora_rank": {"default": 0, "min": 8, "max": 128},
        }

        with patch.object(VerlRecipeTemplateProcessor.__bases__[0], "process_recipe", return_value={}), patch.object(
            processor,
            "_load_recipe_config",
            return_value={
                "training_config": {
                    "model": {
                        "lora_rank": 0,
                    }
                }
            },
        ):
            processor.process_recipe(recipe_file_path=str(tmp_path / "dummy.yaml"))

        assert processor.recipe_override_parameters["lora_rank"]["default"] == 0
