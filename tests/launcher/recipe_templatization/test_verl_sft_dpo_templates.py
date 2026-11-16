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

"""The verl SFT/DPO templates, the Nemotron copies, and the recipes routed to each.

The ``sft``, ``sft_fft``, ``dpo`` and ``dpo_fft`` templates expose the llmft override names, so a
job spec written for an llmft recipe is still accepted by the verl recipe replacing it. The one
llmft name left out on purpose is ``merge_weights``: verl LoRA recipes always merge
(``tests/test_verl_recipes_constraints.py`` pins ``merge_lora_on_final_save: true``), so the
templates do not let a job turn merging off.

The Nemotron recipes run in their own container, which does not support the llmft keys yet, so
they use ``<key>_nemotron`` copies of the templates without those fields. The Megatron-engine GLM
recipes (verl 0.9) use them too, because their ``{{min_lr}}`` is an absolute ``optim.min_lr``.

This module pins three things:

1. Each Nemotron template is the default template minus the llmft fields: same override
   parameters and metadata helpers, a display name marked "Nemotron", every Nemotron
   ``recipe_template`` entry present in the default template with the same placeholder, and the
   default template adds exactly the pinned llmft paths. The only override text allowed to differ
   is listed in ``DEFAULT_ONLY_DESCRIPTIONS``. A change to one template that is not mirrored in the
   other fails here.
2. Every verl SFT/DPO recipe under ``nemotron-0_7_0/`` or ``glm-0_9_0/`` uses a Nemotron template,
   and every other verl SFT/DPO recipe uses a default template. GRPO routing is unchanged.
3. Only the verl SFT/DPO recipes of the Llama 3.x, DeepSeek R1 distill, Qwen 2.5 and Qwen 3 models
   in ``POW2_CAP_MODELS`` publish the power-of-2 length cap, and the count per (technique, model) is
   pinned.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = REPO_ROOT / "launcher" / "recipe_templatization"
VERL_TEMPLATE_FILE = TEMPLATE_DIR / "verl" / "verl_recipe_template_parameters.json"
BASE_FILE = TEMPLATE_DIR / "base_override_parameters.json"
RECIPES_DIR = REPO_ROOT / "recipes_collection" / "recipes"

SUFFIX = VerlRecipeTemplateProcessor.NEMOTRON_TEMPLATE_SUFFIX

# recipe_template paths the default template adds on top of its Nemotron copy (the llmft
# override names), and the placeholder each one carries.
_COMMON_LLMFT_FIELDS = {
    "training_config.trainer.logging_steps": "{{logging_steps}}",
    "training_config.model.generation_cfg.temperature": "{{temperature}}",
    "training_config.model.generation_cfg.max_new_tokens": "{{max_response_length}}",
    "training_config.optim.clip_grad": "{{gradient_clipping_threshold}}",
    "training_config.optim.gradient_clipping": "{{gradient_clipping}}",
    "training_config.data.train_dataset_name": "{{training_data_name}}",
    "training_config.data.val_dataset_name": "{{validation_data_name}}",
    "training_config.data.seed": "{{seed}}",
}
_LORA_LLMFT_FIELDS = {
    **_COMMON_LLMFT_FIELDS,
    "training_config.model.lora_dropout": "{{lora_dropout}}",
}
EXPECTED_LLMFT_FIELDS = {
    "sft": _LORA_LLMFT_FIELDS,
    "sft_fft": _COMMON_LLMFT_FIELDS,
    "dpo": _LORA_LLMFT_FIELDS,
    "dpo_fft": _COMMON_LLMFT_FIELDS,
}

# Override descriptions the default template may word differently from its Nemotron copy. Every
# recipe on the default templates runs the FSDP engine, where min_lr lands on
# optim.min_lr_ratio, a fraction of the peak learning rate. The Nemotron recipes run Megatron,
# where min_lr lands on optim.min_lr, an absolute learning rate, so their templates cannot use
# the fraction wording. test_verl_placeholder_landing_invariant.py pins where min_lr lands.
DEFAULT_ONLY_DESCRIPTIONS = {"min_lr"}

NEMOTRON_DIR = "fine-tuning/nemotron-0_7_0/"
# The Megatron-engine recipes, routed to the Nemotron templates by engine.strategy, not by path.
MEGATRON_ENGINE_DIR = "fine-tuning/glm-0_9_0/"


def _without_default_only_descriptions(override_parameters):
    stripped = json.loads(json.dumps(override_parameters))
    for name in DEFAULT_ONLY_DESCRIPTIONS:
        stripped.get(name, {}).pop("description", None)
    return stripped


def _flatten(node, path=""):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                out.update(_flatten(value, new_path))
            else:
                out[new_path] = value
    return out


@pytest.fixture(scope="module")
def templates():
    return json.loads(VERL_TEMPLATE_FILE.read_text())["templates"]


@pytest.fixture(scope="module")
def processor():
    base = json.loads(BASE_FILE.read_text())["fine_tuning"]
    return VerlRecipeTemplateProcessor(staging_cfg=base, platform="sm_jobs")


def test_no_llmft_parity_templates_left(templates):
    assert not [key for key in templates if "parity" in key]


@pytest.mark.parametrize("base_key", sorted(EXPECTED_LLMFT_FIELDS))
class TestNemotronTemplateIsDefaultWithoutLlmftFields:
    def test_nemotron_template_exists(self, templates, base_key):
        assert base_key + SUFFIX in templates

    def test_non_recipe_template_sections_identical(self, templates, base_key):
        default, nemotron = templates[base_key], templates[base_key + SUFFIX]
        assert set(nemotron) == set(default)
        for section in default:
            if section == "override_parameters":
                assert _without_default_only_descriptions(nemotron[section]) == _without_default_only_descriptions(
                    default[section]
                ), f"{base_key}{SUFFIX}.{section} drifted"
            elif section not in ("recipe_template", "display_name"):
                assert nemotron[section] == default[section], f"{base_key}{SUFFIX}.{section} drifted"

    def test_default_template_does_not_expose_merge_weights(self, templates, base_key):
        default = templates[base_key]
        assert "{{merge_weights}}" not in json.dumps(default["recipe_template"])
        assert "merge_weights" not in default["override_parameters"]

    def test_display_name_marks_nemotron(self, templates, base_key):
        # Only docs/HYPERPARAMETERS.md reads a template's display_name; a distinct label keeps
        # the Nemotron template from appearing as a second section with the default name.
        default = templates[base_key]["display_name"]
        assert "parity" not in default
        assert templates[base_key + SUFFIX]["display_name"] == default[:-1] + ", Nemotron)"

    def test_nemotron_entries_kept_unchanged(self, templates, base_key):
        nemotron = _flatten(templates[base_key + SUFFIX]["recipe_template"])
        default = _flatten(templates[base_key]["recipe_template"])
        changed = {p: (v, default.get(p)) for p, v in nemotron.items() if default.get(p) != v}
        assert not changed, f"{base_key} differs from {base_key}{SUFFIX} at: {changed}"

    def test_default_adds_only_the_llmft_fields(self, templates, base_key):
        nemotron = _flatten(templates[base_key + SUFFIX]["recipe_template"])
        default = _flatten(templates[base_key]["recipe_template"])
        added = {p: v for p, v in default.items() if p not in nemotron}
        assert added == EXPECTED_LLMFT_FIELDS[base_key]

    def test_nemotron_template_has_no_llmft_fields(self, templates, base_key):
        nemotron = _flatten(templates[base_key + SUFFIX]["recipe_template"])
        leaked = set(EXPECTED_LLMFT_FIELDS[base_key]) & set(nemotron)
        assert not leaked, f"llmft field(s) {sorted(leaked)} are on the {base_key}{SUFFIX!r} template"


@pytest.mark.parametrize(
    "recipe, expected_key",
    [
        ("fine-tuning/llama-0_7_0/verl-sft-llama-3-dot-2-1b-instruct-g6-48xl-17k-tt-lora", "sft"),
        ("fine-tuning/llama-0_7_0/verl-sft-llama-3-dot-3-70b-instruct-p5-48xl-8k-tt-fft", "sft_fft"),
        ("fine-tuning/llama-0_7_0/verl-dpo-llama-3-dot-1-8b-instruct-p4de-24xl-24k-tt-lora", "dpo"),
        ("fine-tuning/llama-0_7_0/verl-dpo-llama-3-dot-2-3b-instruct-g6-48xl-6k-tt-fft", "dpo_fft"),
        ("fine-tuning/deepseek-0_7_0/verl-sft-deepseek-r1-distilled-llama-8b-g5-12xl-5k-tt-lora", "sft"),
        ("fine-tuning/deepseek-0_7_0/verl-dpo-deepseek-r1-distilled-llama-70b-p5-48xl-4k-tt-fft", "dpo_fft"),
        ("fine-tuning/qwen-0_7_0/verl-dpo-qwen-2-5-14b-instruct-p4d-24xl-8k-tt-lora", "dpo"),
        ("fine-tuning/qwen-0_7_0/verl-sft-qwen-2-5-14b-instruct-fft", "sft_fft"),
        ("fine-tuning/gemma4-0_7_0/verl-sft-gemma-4-31b-it-lora", "sft"),
        # The Nemotron recipes use the Nemotron copies.
        ("fine-tuning/nemotron-0_7_0/verl-sft-nemotron-3-nano-30b-lora", "sft" + SUFFIX),
        ("fine-tuning/nemotron-0_7_0/verl-sft-nemotron-3-nano-30b-fft", "sft_fft" + SUFFIX),
        ("fine-tuning/nemotron-0_7_0/verl-dpo-nemotron-3-super-120b-lora", "dpo" + SUFFIX),
        ("fine-tuning/nemotron-0_7_0/verl-dpo-nemotron-3-5-lightning-30b-fft", "dpo_fft" + SUFFIX),
        # So do the Megatron-engine GLM recipes.
        ("fine-tuning/glm-0_9_0/verl-sft-glm-5-dot-3-flash-p5-48xl-29k-tt-lora", "sft" + SUFFIX),
        ("fine-tuning/glm-0_9_0/verl-sft-glm-5-dot-3-flash-p5en-48xl-64k-tt-fft", "sft_fft" + SUFFIX),
        # GRPO routing is unchanged, Nemotron included.
        ("fine-tuning/nemotron-0_7_0/verl-grpo-rlvr-nemotron-3-nano-30b-lora", "grpo_rlvr"),
        ("fine-tuning/nemotron-0_7_0/verl-grpo-rlaif-nemotron-3-nano-30b-fft", "grpo_rlaif_fft"),
    ],
)
def test_template_routing(processor, templates, recipe, expected_key):
    selected = processor.get_recipe_template({}, templates, recipe)
    assert selected is templates[expected_key]


def _sft_dpo_key(recipe_path: Path):
    """Base template key (sft/sft_fft/dpo/dpo_fft) for a verl SFT/DPO recipe, else None."""
    recipe = yaml.safe_load(recipe_path.read_text())
    training_config = recipe.get("training_config", {})
    algorithm = training_config.get("algorithm")
    if algorithm and algorithm.get("adv_estimator"):
        return None
    algo = "dpo" if "beta" in training_config.get("trainer", {}) else "sft"
    return f"{algo}_fft" if "fft" in recipe_path.stem.lower() else algo


# Verl SFT/DPO recipes in recipes_collection routed to the Nemotron templates, per template key
# (Nemotron, then the Megatron-engine GLM recipes). A routed recipe added or removed changes a
# count, so the change is deliberate.
EXPECTED_NEMOTRON_ROUTED_COUNTS = {"sft": 3 + 1, "sft_fft": 2 + 2, "dpo": 3, "dpo_fft": 2}


def test_routed_sets(processor, templates):
    """Every Nemotron and GLM verl SFT/DPO recipe uses a *_nemotron template; every other one the default."""
    nemotron_counts = Counter()
    wrong = []
    for path in sorted(RECIPES_DIR.rglob("verl-*.yaml")):
        base_key = _sft_dpo_key(path)
        if base_key is None:
            continue
        rel = path.relative_to(RECIPES_DIR).as_posix()[: -len(".yaml")]
        selected = processor.get_recipe_template({}, templates, rel)
        is_nemotron = rel.startswith((NEMOTRON_DIR, MEGATRON_ENGINE_DIR))
        expected = base_key + SUFFIX if is_nemotron else base_key
        if selected is not templates[expected]:
            wrong.append(f"{rel}: expected {expected}")
        if is_nemotron:
            nemotron_counts[base_key] += 1
    assert not wrong, "recipes routed to the wrong template:\n  " + "\n  ".join(wrong)
    assert dict(nemotron_counts) == EXPECTED_NEMOTRON_ROUTED_COUNTS


def test_only_nemotron_recipes_are_detected_as_nemotron():
    detected = sorted(
        path.relative_to(RECIPES_DIR).as_posix()
        for path in RECIPES_DIR.rglob("verl-*.yaml")
        if VerlRecipeTemplateProcessor._is_nemotron_recipe(path.relative_to(RECIPES_DIR).as_posix())
    )
    assert detected
    assert all(rel.startswith(NEMOTRON_DIR) for rel in detected), detected


def test_grpo_llama_recipe_not_capped():
    # The prefixes are SFT/DPO-only.
    assert not VerlRecipeTemplateProcessor._uses_pow2_length_cap("verl-grpo-rlaif-llama-3-dot-1-8b-instruct-lora")


# The only models whose verl SFT/DPO recipes publish the power-of-2 length cap. Written out again
# rather than read from the processor, so widening the processor's prefixes fails here until this
# list is updated too.
POW2_CAP_MODELS = (
    "llama-3-dot-2-1b-instruct",
    "llama-3-dot-2-3b-instruct",
    "llama-3-dot-1-8b-instruct",
    "llama-3-dot-3-70b-instruct",
    "deepseek-r1-distilled-llama-8b",
    "deepseek-r1-distilled-llama-70b",
    "deepseek-r1-distilled-qwen-1-dot-5b",
    "deepseek-r1-distilled-qwen-7b",
    "deepseek-r1-distilled-qwen-14b",
    "deepseek-r1-distilled-qwen-32b",
    "qwen-2-5-7b-instruct",
    "qwen-2-5-14b-instruct",
    "qwen-2-5-32b-instruct",
    "qwen-2-5-72b-instruct",
    "qwen-3-0-dot-6b",
    "qwen-3-1-dot-7b",
    "qwen-3-4b",
    "qwen-3-8b",
    "qwen-3-14b",
    "qwen-3-32b",
)
POW2_CAP_RECIPE_DIRS = ("fine-tuning/llama-0_7_0/", "fine-tuning/deepseek-0_7_0/", "fine-tuning/qwen-0_7_0/")
# Recipes in recipes_collection that publish the power-of-2 cap, per (technique, model). A recipe
# added for one of these models changes a count, so adding it is a deliberate change.
EXPECTED_POW2_CAP_RECIPE_COUNTS = {
    ("sft", "llama-3-dot-2-1b-instruct"): 11,
    ("sft", "llama-3-dot-2-3b-instruct"): 10,
    ("sft", "llama-3-dot-1-8b-instruct"): 7,
    ("sft", "llama-3-dot-3-70b-instruct"): 2,
    ("dpo", "llama-3-dot-2-1b-instruct"): 7,
    ("dpo", "llama-3-dot-2-3b-instruct"): 7,
    ("dpo", "llama-3-dot-1-8b-instruct"): 6,
    ("dpo", "llama-3-dot-3-70b-instruct"): 2,
    ("sft", "deepseek-r1-distilled-llama-8b"): 7,
    ("sft", "deepseek-r1-distilled-llama-70b"): 2,
    ("sft", "deepseek-r1-distilled-qwen-1-dot-5b"): 7,  # includes the 8-GPU g6-48xl-11k-tt-fft rung
    ("sft", "deepseek-r1-distilled-qwen-7b"): 7,
    ("sft", "deepseek-r1-distilled-qwen-14b"): 6,
    ("sft", "deepseek-r1-distilled-qwen-32b"): 6,
    ("dpo", "deepseek-r1-distilled-llama-8b"): 8,
    ("dpo", "deepseek-r1-distilled-llama-70b"): 2,
    ("dpo", "deepseek-r1-distilled-qwen-1-dot-5b"): 7,
    ("dpo", "deepseek-r1-distilled-qwen-7b"): 11,
    ("dpo", "deepseek-r1-distilled-qwen-14b"): 6,
    ("dpo", "deepseek-r1-distilled-qwen-32b"): 6,
    ("sft", "qwen-2-5-7b-instruct"): 4,
    ("sft", "qwen-2-5-14b-instruct"): 6,
    ("sft", "qwen-2-5-32b-instruct"): 2,
    ("sft", "qwen-2-5-72b-instruct"): 3,
    ("sft", "qwen-3-0-dot-6b"): 7,
    ("sft", "qwen-3-1-dot-7b"): 5,
    ("sft", "qwen-3-4b"): 6,  # includes the single-GPU g5-2xl-4k-tt-lora rung
    ("sft", "qwen-3-8b"): 6,
    ("sft", "qwen-3-14b"): 8,
    ("sft", "qwen-3-32b"): 4,
    ("dpo", "qwen-2-5-7b-instruct"): 5,
    ("dpo", "qwen-2-5-14b-instruct"): 6,
    ("dpo", "qwen-2-5-32b-instruct"): 5,
    ("dpo", "qwen-2-5-72b-instruct"): 2,
    ("dpo", "qwen-3-0-dot-6b"): 7,
    ("dpo", "qwen-3-1-dot-7b"): 5,
    ("dpo", "qwen-3-4b"): 8,
    ("dpo", "qwen-3-8b"): 3,
    ("dpo", "qwen-3-14b"): 7,
    ("dpo", "qwen-3-32b"): 4,
}


def _pow2_cap_slot(stem):
    for technique in ("sft", "dpo"):
        for model in POW2_CAP_MODELS:
            if stem.startswith(f"verl-{technique}-{model}-"):
                return technique, model
    return None


def test_only_the_pow2_cap_models_are_capped():
    capped = {
        path.relative_to(RECIPES_DIR).as_posix(): path.stem
        for path in RECIPES_DIR.rglob("*.yaml")
        if VerlRecipeTemplateProcessor._uses_pow2_length_cap(path.stem)
    }
    outside = sorted(rel for rel in capped if not rel.startswith(POW2_CAP_RECIPE_DIRS))
    assert not outside, f"recipes outside {POW2_CAP_RECIPE_DIRS} get the power-of-2 cap: {outside}"
    other_models = sorted(stem for stem in capped.values() if _pow2_cap_slot(stem) is None)
    assert not other_models, f"recipes of models not in POW2_CAP_MODELS get the power-of-2 cap: {other_models}"
    counts = Counter(_pow2_cap_slot(stem) for stem in capped.values())
    assert dict(counts) == EXPECTED_POW2_CAP_RECIPE_COUNTS


@pytest.mark.parametrize(
    "recipe_name",
    [
        # Other Llama 3.x models, real or hypothetical, are not capped.
        "verl-sft-llama-3-dot-2-11b-vision-instruct-lora",
        "verl-dpo-llama-3-dot-1-70b-instruct-fft",
        "verl-sft-llama-3-dot-1-405b-instruct-lora",
        # A model name that only extends a capped model's name.
        "verl-sft-llama-3-dot-2-1b-instructx-lora",
        "verl-sft-deepseek-r1-distilled-qwen-7bx-lora",
        # DeepSeek models other than the six R1 distills.
        "verl-sft-deepseek-r1-distilled-qwen-3b-lora",
        "verl-dpo-deepseek-v3-fft",
        # Other families on the default templates.
        "verl-dpo-gemma-4-31b-it-lora",
        # Qwen models other than the Qwen 2.5 and Qwen 3 ones in POW2_CAP_MODELS.
        "verl-sft-qwen-3-dot-5-4b-lora",
        "verl-dpo-qwen-3-dot-6-27b-fft",
        "verl-dpo-qwen-2-5-3b-instruct-lora",
        "verl-sft-qwen-3-8bx-lora",
    ],
)
def test_other_models_not_capped(recipe_name):
    assert not VerlRecipeTemplateProcessor._uses_pow2_length_cap(recipe_name)
