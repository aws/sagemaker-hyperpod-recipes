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

import pytest
from omegaconf import OmegaConf

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)

# Path to a real DPO recipe so the processor can load and resolve defaults
DPO_LORA_RECIPE = "fine-tuning/qwen-0_7_0/verl-dpo-qwen-3-dot-5-4b-lora"
DPO_FFT_RECIPE = "fine-tuning/qwen-0_7_0/verl-dpo-qwen-3-dot-5-4b-fft"
DPO_LORA_RECIPE_FULL = "recipes_collection/recipes/fine-tuning/qwen-0_7_0/verl-dpo-qwen-3-dot-5-4b-lora.yaml"


@pytest.fixture
def processor():
    """Build a VerlRecipeTemplateProcessor against the real DPO LoRA recipe."""
    cfg = OmegaConf.load(DPO_LORA_RECIPE_FULL)
    wrapped = OmegaConf.create({"recipes": cfg})
    OmegaConf.update(wrapped, "base_results_dir", "/tmp/test_results", merge=False)
    OmegaConf.resolve(wrapped)
    staging_cfg = OmegaConf.to_container(wrapped.recipes, resolve=True)
    return VerlRecipeTemplateProcessor(staging_cfg, platform="k8s")


class TestExtractAlgorithmType:
    """Covers the DPO detection branch in _extract_algorithm_type (line 235)."""

    def test_detects_dpo_when_trainer_has_beta(self, processor):
        recipe_cfg = {"training_config": {"trainer": {"beta": 0.01}}}
        assert processor._extract_algorithm_type(recipe_cfg) == "dpo"

    def test_detects_sft_when_no_algorithm_and_no_beta(self, processor):
        recipe_cfg = {"training_config": {"trainer": {"total_epochs": 1}}}
        assert processor._extract_algorithm_type(recipe_cfg) == "sft"

    def test_detects_sft_when_trainer_missing(self, processor):
        recipe_cfg = {"training_config": {}}
        assert processor._extract_algorithm_type(recipe_cfg) == "sft"

    def test_detects_grpo_via_adv_estimator(self, processor):
        recipe_cfg = {"training_config": {"algorithm": {"adv_estimator": "grpo"}}}
        assert processor._extract_algorithm_type(recipe_cfg) == "grpo"

    def test_returns_sft_when_adv_estimator_none(self, processor):
        recipe_cfg = {"training_config": {"algorithm": {}}}
        assert processor._extract_algorithm_type(recipe_cfg) == "sft"


class TestDpoTemplateRouting:
    """Covers the DPO branches in get_recipe_template (lines 60-64)."""

    def test_dpo_lora_recipe_uses_dpo_template(self, processor):
        template = processor.template_data["templates"]
        result = processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=DPO_LORA_RECIPE)
        assert result is template["dpo"]
        assert processor.algorithm_type == "dpo"

    def test_dpo_fft_recipe_uses_dpo_fft_template(self, processor):
        template = processor.template_data["templates"]
        result = processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=DPO_FFT_RECIPE)
        assert result is template["dpo_fft"]
        assert processor.algorithm_type == "dpo"


class TestVerl090RewardKwargsPath:
    """verl 0.9 RLAIF recipes take the judge settings at training_config.reward.reward_kwargs."""

    RLAIF_07 = "fine-tuning/qwen-0_7_0/verl-grpo-rlaif-qwen-3-dot-6-27b-lora"
    # No 0.9 GRPO recipe ships in this repo, so these paths are served by _grpo_0_9_recipes below.
    RLAIF_09 = "fine-tuning/qwen-0_9_0/verl-grpo-rlaif-qwen-3-dot-6-27b-lora"
    RLVR_09 = "fine-tuning/qwen-0_9_0/verl-grpo-rlvr-qwen-3-dot-5-4b-lora"
    JUDGE_KWARGS = {
        "model_id": "{{judge_model_id}}",
        "llmj_prompt_template_path": "{{judge_prompt_template}}",
    }

    @pytest.fixture(autouse=True)
    def _grpo_0_9_recipes(self, processor, monkeypatch):
        """Serve minimal 0.9 GRPO recipe configs; template selection reads only these fields."""
        recipes = {
            self.RLAIF_09: {
                "display_name": "Qwen 3.6 27B GRPO RLAIF Fine-Tuning with LoRA (verl 0.9)",
                "training_config": {"algorithm": {"adv_estimator": "grpo"}},
            },
            self.RLVR_09: {
                "display_name": "Qwen 3.5 4B GRPO RLVR Fine-Tuning with LoRA (verl 0.9)",
                "training_config": {"algorithm": {"adv_estimator": "grpo"}},
            },
        }
        load = processor._load_recipe_config
        monkeypatch.setattr(processor, "_load_recipe_config", lambda path: recipes.get(path) or load(path))

    def test_0_9_rlaif_template_writes_judge_settings_under_reward(self, processor):
        template = processor.template_data["templates"]
        result = processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=self.RLAIF_09)
        training_config = result["recipe_template"]["training_config"]
        assert training_config["reward"]["reward_kwargs"]["kwargs"] == self.JUDGE_KWARGS
        assert "reward_model" not in training_config

    def test_0_9_relocation_does_not_mutate_the_shared_template(self, processor):
        template = processor.template_data["templates"]
        processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=self.RLAIF_09)
        shared = template["grpo_rlaif"]["recipe_template"]["training_config"]
        assert shared["reward_model"]["reward_kwargs"]["kwargs"] == self.JUDGE_KWARGS
        assert "reward" not in shared

    def test_0_7_rlaif_template_is_unchanged(self, processor):
        template = processor.template_data["templates"]
        result = processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=self.RLAIF_07)
        assert result is template["grpo_rlaif"]

    def test_0_9_template_without_reward_kwargs_is_returned_as_is(self, processor):
        template = processor.template_data["templates"]
        result = processor.get_recipe_template(yaml_data={}, template=template, recipe_file_path=self.RLVR_09)
        assert result is template["grpo_rlvr"]
