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
Tests for verl recipe constraints.
"""

import pytest
import yaml

from launcher.nemo.constants import ROOT_DIR

RECIPES_DIR = ROOT_DIR / "recipes_collection" / "recipes" / "fine-tuning"

MAX_TOTAL_EPOCHS = 10


def _collect_verl_recipe_files():
    """Collect all verl recipe YAML files under the fine-tuning recipes directory."""
    return sorted(RECIPES_DIR.rglob("verl-*.yaml"))


def _get_recipe_id(path):
    """Return a short human-readable identifier for the recipe file."""
    return str(path.relative_to(RECIPES_DIR))


VERL_RECIPE_FILES = _collect_verl_recipe_files()
VERL_GRPO_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if "grpo" in str(f)]
VERL_LORA_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if "-lora" in f.stem]


@pytest.mark.parametrize(
    "recipe_path",
    VERL_GRPO_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_GRPO_RECIPE_FILES],
)
def test_max_num_batched_tokens_positive(recipe_path):
    """
    Verify that max_num_batched_tokens is a positive integer for every verl recipe.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    rollout = config["training_config"]["actor_rollout_ref"]["rollout"]
    max_num_batched_tokens = rollout["max_num_batched_tokens"]

    assert isinstance(max_num_batched_tokens, int) and max_num_batched_tokens > 0, (
        f"max_num_batched_tokens ({max_num_batched_tokens}) must be a positive integer "
        f"in {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "recipe_path",
    VERL_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_RECIPE_FILES],
)
def test_total_epochs_within_bounds(recipe_path):
    """
    Verify that total_epochs is a positive integer not exceeding MAX_TOTAL_EPOCHS.

    Catches accidental large values (e.g. 100) that would waste compute.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = config["training_config"]["trainer"]
    total_epochs = trainer["total_epochs"]

    assert isinstance(total_epochs, int) and 1 <= total_epochs <= MAX_TOTAL_EPOCHS, (
        f"total_epochs ({total_epochs}) must be an integer between 1 and {MAX_TOTAL_EPOCHS} "
        f"in {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "recipe_path",
    VERL_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_RECIPE_FILES],
)
def test_save_freq_present_and_valid(recipe_path):
    """
    Verify that save_freq is present and set to a valid value.

    Valid values are positive integers or the string 'after_each_epoch'.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = config["training_config"]["trainer"]

    assert "save_freq" in trainer, f"save_freq is missing from trainer config in {recipe_path.relative_to(ROOT_DIR)}"

    save_freq = trainer["save_freq"]
    is_valid = save_freq == "after_each_epoch" or (isinstance(save_freq, int) and save_freq > 0)
    assert is_valid, (
        f"save_freq ({save_freq!r}) must be a positive integer or 'after_each_epoch' "
        f"in {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "recipe_path",
    VERL_LORA_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_LORA_RECIPE_FILES],
)
def test_merge_lora_on_final_save_true(recipe_path):
    """
    Verify that merge_lora_on_final_save is true for all LoRA recipes.

    This ensures both LoRA adapters and merged weights are saved for
    inference and evaluation respectively.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = config["training_config"]["trainer"]
    merge_lora = trainer.get("merge_lora_on_final_save")

    assert merge_lora is True, (
        f"merge_lora_on_final_save must be true for LoRA recipes, "
        f"got {merge_lora!r} in {recipe_path.relative_to(ROOT_DIR)}"
    )
