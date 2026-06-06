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
