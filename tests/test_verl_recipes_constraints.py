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


def _collect_verl_recipe_files():
    """Collect all verl recipe YAML files under the fine-tuning recipes directory."""
    return sorted(RECIPES_DIR.rglob("verl-*.yaml"))


def _get_recipe_id(path):
    """Return a short human-readable identifier for the recipe file."""
    return str(path.relative_to(RECIPES_DIR))


VERL_RECIPE_FILES = _collect_verl_recipe_files()
VERL_GRPO_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if "grpo" in str(f)]


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
