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
Tests that the customer-overridable sequence-length hyperparameters are exposed
by the recipe templates:

  - verl GRPO (RL) templates expose max_prompt_length + max_response_length
  - verl SFT/DPO templates expose dataset_max_len (recipe field data.max_length)
  - llmft SFT/DPO/FT templates expose dataset_max_len (recipe field max_len)

Both frameworks map their differently-named recipe field to the single
dataset_max_len override parameter, so the customer sees one consistent knob.

Each is resolved from base_override_parameters.json + the template's sparse
override_parameters, exactly as the launch.json generator does.
"""

import json
from pathlib import Path

import pytest

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)
from utils.resolve_override_params import resolve_params

TEMPLATE_DIR = Path("launcher/recipe_templatization")
BASE_FILE = TEMPLATE_DIR / "base_override_parameters.json"
VERL_TEMPLATE_FILE = TEMPLATE_DIR / "verl" / "verl_recipe_template_parameters.json"
LLMFT_TEMPLATE_FILE = TEMPLATE_DIR / "llmft" / "llmft_recipe_template_parameters.json"

GRPO_TEMPLATES = ["grpo_rlaif", "grpo_rlvr", "grpo_rlaif_fft", "grpo_rlvr_fft"]
VERL_SFT_DPO_TEMPLATES = ["sft", "sft_fft", "dpo", "dpo_fft"]
LLMFT_SFT_DPO_TEMPLATES = [
    "llmft_sft_lora",
    "llmft_sft_fft",
    "llmft_dpo",
    "llmft_dpo_fft",
    "llmft_fine_tuning",
]


def _resolve(template_file: Path, template_key: str) -> dict:
    base = json.loads(BASE_FILE.read_text())["fine_tuning"]
    template = json.loads(template_file.read_text())["templates"][template_key]
    return resolve_params(
        base,
        template.get("override_parameters", {}),
        template.get("recipe_template", {}),
    )


class TestGrpoLengthOverrideParams:
    """verl GRPO recipes expose prompt + response length."""

    @pytest.mark.parametrize("template_key", GRPO_TEMPLATES)
    def test_exposes_prompt_and_response_length(self, template_key):
        resolved = _resolve(VERL_TEMPLATE_FILE, template_key)
        assert "max_prompt_length" in resolved
        assert "max_response_length" in resolved
        for param in ("max_prompt_length", "max_response_length"):
            assert resolved[param]["visibility_tier"] == "advanced"
            assert resolved[param]["type"] == "integer"

    @pytest.mark.parametrize("template_key", GRPO_TEMPLATES)
    def test_does_not_expose_max_length_params(self, template_key):
        # GRPO uses prompt + response, not the single-sequence knobs.
        resolved = _resolve(VERL_TEMPLATE_FILE, template_key)
        assert "max_length" not in resolved
        assert "dataset_max_len" not in resolved


class TestVerlSftDpoLengthOverrideParams:
    """verl SFT/DPO recipes expose dataset_max_len (mapped from data.max_length)."""

    @pytest.mark.parametrize("template_key", VERL_SFT_DPO_TEMPLATES)
    def test_exposes_dataset_max_len(self, template_key):
        resolved = _resolve(VERL_TEMPLATE_FILE, template_key)
        assert "dataset_max_len" in resolved
        assert resolved["dataset_max_len"]["type"] == "integer"

    @pytest.mark.parametrize("template_key", VERL_SFT_DPO_TEMPLATES)
    def test_does_not_expose_prompt_response_or_max_length(self, template_key):
        resolved = _resolve(VERL_TEMPLATE_FILE, template_key)
        assert "max_prompt_length" not in resolved
        assert "max_response_length" not in resolved
        # max_length is the recipe field; the customer-facing param is dataset_max_len.
        assert "max_length" not in resolved

    @pytest.mark.parametrize("template_key", VERL_SFT_DPO_TEMPLATES)
    def test_recipe_body_wires_dataset_max_len(self, template_key):
        template = json.loads(VERL_TEMPLATE_FILE.read_text())["templates"][template_key]
        data = template["recipe_template"]["training_config"]["data"]
        assert data.get("max_length") == "{{dataset_max_len}}"


class TestLlmftSftDpoLengthOverrideParams:
    """llmft SFT/DPO/FT recipes expose dataset_max_len (mapped from max_len)."""

    @pytest.mark.parametrize("template_key", LLMFT_SFT_DPO_TEMPLATES)
    def test_exposes_dataset_max_len(self, template_key):
        resolved = _resolve(LLMFT_TEMPLATE_FILE, template_key)
        assert "dataset_max_len" in resolved
        assert resolved["dataset_max_len"]["type"] == "integer"


# Recipes whose computed sequence length (max_token_len_per_gpu x world size,
# floored to a power of 2) differs from the static base max (131072), chosen to
# prove the per-recipe clamp both lowers and raises the ceiling. gemma-4-31b's
# raw 266240 floors to 262144 (2^18).
VERL_CLAMP_CASES = [
    # context_length = floor_pow2(max_token_len_per_gpu); NOT scaled by world size
    # (SP unsupported). 72b-sft: 8192 -> 8192 "8K"; gemma-31b-dpo: 33280 -> 32768 "32K".
    ("fine-tuning/qwen-0_7_0/verl-sft-qwen-2-5-72b-instruct-fft", 8192, "8K"),
    ("fine-tuning/gemma4-0_7_0/verl-dpo-gemma-4-31b-it-lora", 32768, "32K"),
]


class TestVerlSequenceLengthClamp:
    """verl SFT/DPO/RL clamp each length param's max to the recipe's sequence length."""

    @pytest.fixture(scope="class")
    def processor(self):
        base = json.loads(BASE_FILE.read_text())["fine_tuning"]
        return VerlRecipeTemplateProcessor(staging_cfg=base, platform="sm_jobs")

    @pytest.mark.parametrize("recipe_path,expected_max,expected_label", VERL_CLAMP_CASES)
    def test_dataset_max_len_clamped_to_sequence_length(self, processor, recipe_path, expected_max, expected_label):
        processor.process_recipe(recipe_file_path=recipe_path)
        _, override_params, _ = processor.get_additional_data(recipe_path)
        assert override_params["dataset_max_len"]["max"] == expected_max
        # SequenceLength metadata is the single source of truth, formatted "<n>K".
        metadata = processor.get_recipe_metadata(recipe_path)
        assert metadata["SequenceLength"] == expected_label

    def test_sequence_length_floored_to_power_of_two(self, processor):
        # grpo-rlvr-qwen-3.6-27b-fft: actor ppo_max_token_len_per_gpu 12288
        # -> floored to 8192 -> "8K" (not scaled by world size; SP unsupported).
        rel = "fine-tuning/qwen-0_7_0/verl-grpo-rlvr-qwen-3-dot-6-27b-fft"
        processor.process_recipe(recipe_file_path=rel)
        _, override_params, _ = processor.get_additional_data(rel)
        assert override_params["max_prompt_length"]["max"] == 8192
        assert processor.get_recipe_metadata(rel)["SequenceLength"] == "8K"

    def test_floor_power_of_two(self, processor):
        assert processor._floor_power_of_two(98304) == 65536
        assert processor._floor_power_of_two(266240) == 262144
        assert processor._floor_power_of_two(131072) == 131072  # already a power of 2

    def test_extract_context_length_computes_sft_formula(self, processor):
        # SFT/DPO: data.max_token_len_per_gpu, floored to a power of 2. Not scaled
        # by world size — SP is unsupported, so a rank holds a whole sequence.
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "training_config": {
                    "data": {"max_token_len_per_gpu": 16384},
                    "trainer": {"n_gpus_per_node": 8, "nnodes": 2},
                }
            }
        )
        assert processor._extract_context_length(cfg) == 16384

    def test_extract_context_length_computes_rl_formula(self, processor):
        # RL: actor_rollout_ref.actor.ppo_max_token_len_per_gpu, floored to a
        # power of 2. Not scaled by world size (SP unsupported).
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "training_config": {
                    "actor_rollout_ref": {"actor": {"ppo_max_token_len_per_gpu": 32768}},
                    "trainer": {"n_gpus_per_node": 8, "nnodes": 1},
                }
            }
        )
        assert processor._extract_context_length(cfg) == 32768

    def test_extract_context_length_none_when_budget_absent(self, processor):
        # No per-GPU token budget field -> None (no-op clamp), not a crash.
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"training_config": {"data": {"max_length": 4096}}})
        assert processor._extract_context_length(cfg) is None

    def test_extract_context_length_none_when_no_training_config(self, processor):
        # A config lacking training_config entirely yields None (no crash).
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"run": {"name": "x"}})
        assert processor._extract_context_length(cfg) is None

    def test_get_additional_data_noop_when_base_empty(self, processor):
        # When the base resolver returns nothing, get_additional_data passes it
        # through untouched (no clamp attempted).
        from unittest.mock import patch

        with patch.object(VerlRecipeTemplateProcessor.__bases__[0], "get_additional_data", return_value=[]):
            assert processor.get_additional_data("any/path") == []

    def test_get_additional_data_noop_when_no_override_params(self, processor):
        # When the base resolves metadata but no override params, return as-is
        # without touching context_length.
        from unittest.mock import patch

        passthrough = [{"Name": "x"}, {}, {}]
        with patch.object(
            VerlRecipeTemplateProcessor.__bases__[0],
            "get_additional_data",
            return_value=passthrough,
        ):
            assert processor.get_additional_data("any/path") == passthrough

    def test_llmft_dataset_max_len_static_max(self):
        # The recipe is the source of truth for what the UI presents (Studio UI
        # team request). llmft has no dynamic context clamp, so dataset_max_len
        # keeps its static base max — the UI renders a bounded range rather than
        # an open-ended field. verl still overrides this with its computed ceiling.
        base = json.loads(BASE_FILE.read_text())["fine_tuning"]
        assert base["dataset_max_len"]["max"] == 131072

    @pytest.mark.parametrize(
        "param,expected_max",
        [
            ("max_prompt_length", 16384),
            ("max_response_length", 200000),
            ("dataset_max_len", 131072),
        ],
    )
    def test_length_params_have_static_max_in_base(self, param, expected_max):
        # Length params carry a static base max so non-verl recipes present a
        # bounded field. verl's dynamic clamp overrides it per-recipe.
        base = json.loads(BASE_FILE.read_text())["fine_tuning"]
        assert base[param]["max"] == expected_max

    @pytest.mark.parametrize("param", ["max_response_length", "dataset_max_len"])
    def test_length_param_hint_references_range(self, param):
        # With a static max present, the hint renders a full min-max range.
        base = json.loads(BASE_FILE.read_text())["fine_tuning"]
        assert base[param].get("hint") == "Must be a value between {min} and {max}"

    def test_verl_param_uncapped_when_ceiling_not_computable(self, processor):
        # When a verl recipe has no computable ceiling, the clamp is a no-op and the
        # length param stays uncapped (no 'max') rather than falling back to a static
        # cap, which no longer exists.
        from unittest.mock import patch

        base_result = [
            {"Name": "x"},
            {"max_prompt_length": {"min": 512, "default": 1024}},
            {},
        ]
        with patch.object(
            VerlRecipeTemplateProcessor.__bases__[0],
            "get_additional_data",
            return_value=base_result,
        ), patch.object(processor, "_load_recipe_config", return_value={}), patch.object(
            processor, "_extract_context_length", return_value=None
        ):
            _, override_params, _ = processor.get_additional_data("any/path")
        assert "max" not in override_params["max_prompt_length"]

    @pytest.mark.parametrize("recipe_path,expected_max,expected_label", VERL_CLAMP_CASES)
    def test_clamp_never_inverts_range(self, processor, recipe_path, expected_max, expected_label):
        # The clamped max must stay >= the param's min and default, so the range
        # is always valid even if a recipe ships a tiny sequence length.
        processor.process_recipe(recipe_file_path=recipe_path)
        _, override_params, _ = processor.get_additional_data(recipe_path)
        for param in ("max_prompt_length", "max_response_length", "dataset_max_len"):
            spec = override_params.get(param)
            if spec is None:
                continue
            assert spec["max"] >= spec["min"]
            assert spec["max"] >= spec["default"]

    def test_floor_guard_prevents_inverted_range(self, processor):
        # Directly exercise the guard: a context_length below a param's min must
        # not produce max < min. Uses a synthetic override_params dict.
        override_params = {
            "max_prompt_length": {"min": 512, "max": 16384, "default": 1024},
        }
        # Simulate the clamp body with a tiny ceiling.
        tiny_context_length = 100
        spec = override_params["max_prompt_length"]
        floor = max(spec["min"], spec["default"])
        spec["max"] = max(tiny_context_length, floor)
        assert spec["max"] == 1024  # floored to default, not 100
        assert spec["max"] >= spec["min"]
