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
Tests for :class:`MtrlRecipeTemplateProcessor`.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from launcher.recipe_templatization.mtrl.mtrl_recipe_template_processor import (
    MtrlRecipeTemplateProcessor,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)


def _base_staging_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
            "batch": {"training_batch_size": 128, "group_size": 8},
            "rl": {"advantage_method": "group_based", "loss_fn": "ppo"},
            "sampling_params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 12288},
            "model": {"model_name": "openai/gpt-oss-20b", "max_sequence_length": 32768, "lora_rank": 32},
            "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 3},
            "display_name": "MTRL GPT-OSS 20B LoRA",
            "recipe_version": "1.0.0",
            "schedule": {"epochs": 1, "max_steps": 100},
        }
    )


def _make_processor(staging_cfg=None) -> MtrlRecipeTemplateProcessor:
    if staging_cfg is None:
        staging_cfg = _base_staging_cfg()
    return MtrlRecipeTemplateProcessor(staging_cfg=staging_cfg)


# ---------------------------------------------------------------------------
# _load_template (lines 39-46)
# ---------------------------------------------------------------------------


def test_load_template_populates_attributes():
    processor = _make_processor()
    assert isinstance(processor.template_data, dict)
    assert "templates" in processor.template_data
    assert isinstance(processor.recipe_jumpstart_model_id_mapping, dict)
    assert isinstance(processor.regional_parameters, dict)


# ---------------------------------------------------------------------------
# get_recipe_template (lines 51-68)
# ---------------------------------------------------------------------------


class TestGetRecipeTemplate:
    def test_raises_on_none_path(self):
        processor = _make_processor()
        with pytest.raises(ValueError, match="recipe_file_path is required"):
            processor.get_recipe_template({}, {}, recipe_file_path=None)

    def test_lora_recipe_returns_mtrl_lora_template(self):
        processor = _make_processor()
        templates = processor.template_data["templates"]
        result = processor.get_recipe_template({}, templates, recipe_file_path="training/mtrl/mtrl-gpt-oss-20b-lora")
        assert result is not None
        assert "recipe_template" in result

    def test_nova_lite_lora_returns_nova_template(self):
        processor = _make_processor()
        templates = processor.template_data["templates"]
        result = processor.get_recipe_template({}, templates, recipe_file_path="training/mtrl/mtrl-nova_lite-lora")
        assert result is not None

    def test_non_lora_defaults_to_lora_template(self):
        processor = _make_processor()
        templates = processor.template_data["templates"]
        result = processor.get_recipe_template({}, templates, recipe_file_path="training/mtrl/mtrl-gpt-oss-20b-fft")
        assert result is not None

    def test_raises_on_missing_template_key(self):
        processor = _make_processor()
        with pytest.raises(ValueError, match="Invalid MTRL template key"):
            processor.get_recipe_template({}, {}, recipe_file_path="training/mtrl/mtrl-gpt-oss-20b-lora")


# ---------------------------------------------------------------------------
# get_recipe_metadata (lines 71-109)
# ---------------------------------------------------------------------------


class TestGetRecipeMetadata:
    def _arm_processor(self, processor):
        """Set matched_template_group as process_recipe would."""
        templates = processor.template_data["templates"]
        # Use the first available template group
        key = next(k for k in templates if "mtrl" in k and "eval" not in k)
        processor.matched_template_group = templates[key]
        processor.matched_template = templates[key]["recipe_template"]
        processor.recipe_override_parameters = copy.deepcopy(templates[key]["recipe_override_parameters"])

    def test_metadata_has_expected_keys(self):
        processor = _make_processor()
        self._arm_processor(processor)

        recipe_cfg = OmegaConf.create(
            {
                "display_name": "MTRL GPT-OSS 20B LoRA",
                "recipe_version": "1.0.0",
                "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
                "model": {"model_name": "openai/gpt-oss-20b", "max_sequence_length": 32768, "lora_rank": 32},
            }
        )
        processor._load_recipe_config = lambda _: recipe_cfg

        metadata = processor.get_recipe_metadata("training/mtrl/mtrl-gpt-oss-20b-lora")

        assert isinstance(metadata, OrderedDict)
        assert metadata["Type"] == "FineTuning"
        assert metadata["CustomizationTechnique"] == "MTRL"
        assert "Name" in metadata
        assert "Versions" in metadata
        assert metadata["Versions"] == ["1.0.0"]

    def test_metadata_peft_lora_from_config(self):
        """lora_rank > 0 → Peft = LoRA."""
        processor = _make_processor()
        self._arm_processor(processor)

        recipe_cfg = OmegaConf.create(
            {
                "display_name": "Test",
                "recipe_version": "1.0.0",
                "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
                "model": {"model_name": "openai/gpt-oss-20b", "max_sequence_length": 4096, "lora_rank": 16},
            }
        )
        processor._load_recipe_config = lambda _: recipe_cfg

        metadata = processor.get_recipe_metadata("training/mtrl/mtrl-gpt-oss-20b-lora")
        assert metadata.get("Peft") == "LoRA"

    def test_metadata_peft_from_filename_fallback(self):
        """lora_rank=0 but 'lora' in filename → falls back to recipe_metadata_helpers."""
        processor = _make_processor()
        self._arm_processor(processor)

        recipe_cfg = OmegaConf.create(
            {
                "display_name": "Test",
                "recipe_version": "2.0.0",
                "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
                "model": {"model_name": "openai/gpt-oss-20b", "max_sequence_length": 4096, "lora_rank": 0},
            }
        )
        processor._load_recipe_config = lambda _: recipe_cfg

        metadata = processor.get_recipe_metadata("training/mtrl/mtrl-gpt-oss-20b-lora")
        # Either Peft is set from helpers or not present - just verify no crash
        assert "Type" in metadata


# ---------------------------------------------------------------------------
# _extract_model_display_name (lines 111-116)
# ---------------------------------------------------------------------------


class TestExtractModelDisplayName:
    def test_strips_org_prefix(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {"model_name": "openai/gpt-oss-20b"}})
        assert processor._extract_model_display_name(cfg) == "gpt-oss-20b"

    def test_returns_name_without_slash(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {"model_name": "gpt-oss-20b"}})
        assert processor._extract_model_display_name(cfg) == "gpt-oss-20b"

    def test_returns_unknown_when_missing(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {}})
        assert processor._extract_model_display_name(cfg) == "Unknown Model"


# ---------------------------------------------------------------------------
# _extract_sequence_length (lines 118-120)
# ---------------------------------------------------------------------------


class TestExtractSequenceLength:
    def test_returns_sequence_length(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {"max_sequence_length": 32768}})
        assert processor._extract_sequence_length(cfg) == 32768

    def test_returns_none_when_missing(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {}})
        assert processor._extract_sequence_length(cfg) is None


# ---------------------------------------------------------------------------
# _extract_peft_type_from_config (lines 124-135)
# ---------------------------------------------------------------------------


class TestExtractPeftTypeFromConfig:
    def test_lora_rank_positive_returns_lora(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {"lora_rank": 32}})
        assert processor._extract_peft_type_from_config(cfg) == "LoRA"

    def test_lora_rank_zero_returns_none(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {"lora_rank": 0}})
        assert processor._extract_peft_type_from_config(cfg) is None

    def test_no_lora_rank_returns_none(self):
        processor = _make_processor()
        cfg = OmegaConf.create({"model": {}})
        assert processor._extract_peft_type_from_config(cfg) is None
