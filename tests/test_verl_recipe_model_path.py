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
Tests to verify that verl recipe YAML files have a model path consistent
with the model identity advertised in the filename and display_name.

This catches copy-paste errors where a recipe file is duplicated for a
different model but the internal ``model.path`` is not updated (e.g.
filename says "qwen-3-5-9b" while the path still points to "Qwen3.5-4B").

Both *resolved* recipes (``recipes_collection/recipes/``) and *source*
recipes (``hyperpod_recipes/recipes_src/``) are validated.
"""

import re

import pytest
import yaml

from launcher.nemo.constants import ROOT_DIR

RESOLVED_RECIPES_DIR = ROOT_DIR / "recipes_collection" / "recipes" / "fine-tuning"
SOURCE_RECIPES_DIR = ROOT_DIR / "hyperpod_recipes" / "recipes_src" / "fine-tuning"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Known algorithm / training keywords in display_name that are NOT part
# of the model identity.
_DISPLAY_NAME_STOP_WORDS = re.compile(
    r"\b(?:GRPO|SFT|RLAIF|RLVR|Fine-Tuning|Fine Tuning|with|LoRA|FFT|Full)\b",
    re.IGNORECASE,
)

# Algorithm prefixes to strip from recipe filenames.
_ALGO_PREFIX_RE = re.compile(r"^verl-(?:grpo-(?:rlvr|rlaif)-|sft-)")

# Tuning-method suffixes to strip from recipe filenames.
_TUNING_SUFFIX_RE = re.compile(r"-(lora|fft)$")

# HuggingFace suffixes that are not part of the model identity.
_HF_STRIP_RE = re.compile(r"-bf16$", re.IGNORECASE)


def _to_tokens(s):
    """Split a string into lowercase alphanumeric tokens.

    Dots and hyphens are treated as separators. The word "dot" is
    replaced by "." before splitting so that "3-dot-1" and "3.1" yield
    the same tokens.  "distilled" is normalised to "distill".

    Additionally, boundaries between letters and digits are split so
    that "Qwen3" becomes ["qwen", "3"] and "Qwen2.5" becomes
    ["qwen", "2", "5"].  This ensures HuggingFace names like
    "Qwen3-8B" normalise the same way as filenames like "qwen-3-8b".
    """
    s = s.lower()
    s = re.sub(r"-dot-", ".", s)
    s = s.replace("distilled", "distill")
    # Insert a separator at letter-to-digit boundaries:
    #   "qwen3" -> "qwen-3", "r1" -> "r-1"  (but keep "8b" as-is since
    #   the size token "8b" is a single logical unit).
    # We only split where a letter is followed by a digit, not the
    # reverse (so "8b" stays together).
    s = re.sub(r"([a-z])(\d)", r"\1-\2", s)
    # Split on any non-alphanumeric-dot character, then split remaining
    # dots that sit between digits into separate tokens.
    parts = re.split(r"[^a-z0-9.]+", s)
    tokens = []
    for p in parts:
        if not p:
            continue
        # "3.5" -> ["3", "5"];  "0.6b" -> ["0", "6b"]
        sub = re.split(r"(?<=\d)\.(?=\d)", p)
        tokens.extend(sub)
    return tokens


def _model_id_from_path(model_path):
    """Extract a canonical model identity from a HuggingFace model path.

    "Qwen/Qwen3.5-9B"                       -> "qwen3 5 9b"
    "meta-llama/Llama-3.1-8B-Instruct"       -> "llama 3 1 8b instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" -> "deepseek r1 distill qwen 7b"
    "openai/gpt-oss-20b-bf16"                -> "gpt oss 20b"
    """
    name = model_path.rsplit("/", 1)[-1]
    name = _HF_STRIP_RE.sub("", name)
    return " ".join(_to_tokens(name))


def _model_id_from_filename(filename):
    """Extract a canonical model identity from a recipe filename.

    "verl-grpo-rlvr-qwen-3-5-9b-lora.yaml"  -> "qwen 3 5 9b"
    "verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft.yaml"
                                              -> "llama 3 1 8b instruct"
    """
    name = filename.removesuffix(".yaml")
    name = _ALGO_PREFIX_RE.sub("", name)
    name = _TUNING_SUFFIX_RE.sub("", name)
    return " ".join(_to_tokens(name))


def _model_id_from_display_name(display_name):
    """Extract a canonical model identity from a display_name.

    "Qwen 3.5 9B GRPO RLVR Fine-Tuning with LoRA" -> "qwen 3 5 9b"
    "Llama 3.1 8B GRPO RLAIF Fine-Tuning"          -> "llama 3 1 8b"
    "Deepseek R1 Distilled Qwen 7B GRPO ..."        -> "deepseek r1 distill qwen 7b"
    """
    # Truncate at the first algorithm/training keyword.
    match = _DISPLAY_NAME_STOP_WORDS.search(display_name)
    model_part = display_name[: match.start()].strip() if match else display_name
    return " ".join(_to_tokens(model_part))


# ---------------------------------------------------------------------------
# Collect recipe files
# ---------------------------------------------------------------------------


def _collect_resolved_verl_recipes():
    """Resolved recipes with a non-null, non-local model path."""
    recipes = []
    for path in sorted(RESOLVED_RECIPES_DIR.rglob("verl-*.yaml")):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config.get("training_config", {}).get("actor_rollout_ref", {}).get("model", {}).get("path")
        if model_path and not str(model_path).startswith("/"):
            recipes.append(path)
    return recipes


def _collect_source_verl_recipes():
    """Source recipes (excluding hydra_config) with an inline model path."""
    recipes = []
    for path in sorted(SOURCE_RECIPES_DIR.rglob("verl-*.yaml")):
        if "hydra_config" in str(path):
            continue
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config.get("training_config", {}).get("actor_rollout_ref", {}).get("model", {}).get("path")
        if model_path and not str(model_path).startswith("/"):
            recipes.append(path)
    return recipes


def _collect_model_configs():
    """Hydra model config files that define a model path."""
    configs = []
    for path in sorted(SOURCE_RECIPES_DIR.rglob("model_config/*.yaml")):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        model_path = None
        if "training_config" in config:
            model_path = config.get("training_config", {}).get("actor_rollout_ref", {}).get("model", {}).get("path")
        elif "actor_rollout_ref" in config:
            model_path = config.get("actor_rollout_ref", {}).get("model", {}).get("path")
        if model_path and not str(model_path).startswith("/"):
            configs.append((path, model_path))
    return configs


def _get_id(path):
    return str(path.relative_to(ROOT_DIR))


RESOLVED_RECIPES = _collect_resolved_verl_recipes()
SOURCE_RECIPES = _collect_source_verl_recipes()
MODEL_CONFIGS = _collect_model_configs()


# ---------------------------------------------------------------------------
# Tests for resolved recipes (recipes_collection/recipes/)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe_path",
    RESOLVED_RECIPES,
    ids=[_get_id(p) for p in RESOLVED_RECIPES],
)
def test_resolved_recipe_model_path_matches_display_name(recipe_path):
    """The model identity in ``model.path`` must match ``display_name``."""
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    display_name = config.get("display_name", "")
    model_path = config["training_config"]["actor_rollout_ref"]["model"]["path"]

    path_id = _model_id_from_path(model_path)
    display_id = _model_id_from_display_name(display_name)

    assert path_id == display_id or path_id.startswith(display_id + " "), (
        f"Model identity mismatch in {recipe_path.relative_to(ROOT_DIR)}:\n"
        f"  display_name '{display_name}' -> '{display_id}'\n"
        f"  model.path   '{model_path}' -> '{path_id}'"
    )


@pytest.mark.parametrize(
    "recipe_path",
    RESOLVED_RECIPES,
    ids=[_get_id(p) for p in RESOLVED_RECIPES],
)
def test_resolved_recipe_model_path_matches_filename(recipe_path):
    """The model identity in ``model.path`` must match the filename."""
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["training_config"]["actor_rollout_ref"]["model"]["path"]

    path_id = _model_id_from_path(model_path)
    filename_id = _model_id_from_filename(recipe_path.name)

    assert path_id == filename_id, (
        f"Model identity mismatch in {recipe_path.relative_to(ROOT_DIR)}:\n"
        f"  filename   '{recipe_path.name}' -> '{filename_id}'\n"
        f"  model.path '{model_path}' -> '{path_id}'"
    )


# ---------------------------------------------------------------------------
# Tests for source recipes (hyperpod_recipes/recipes_src/)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recipe_path",
    SOURCE_RECIPES,
    ids=[_get_id(p) for p in SOURCE_RECIPES],
)
def test_source_recipe_model_path_matches_display_name(recipe_path):
    """For source recipes with an inline model path, the identity in
    ``model.path`` must match ``display_name``."""
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    display_name = config.get("display_name", "")
    model_path = config["training_config"]["actor_rollout_ref"]["model"]["path"]

    path_id = _model_id_from_path(model_path)
    display_id = _model_id_from_display_name(display_name)

    if not display_id:
        pytest.skip(f"No display_name in {recipe_path.name}")

    assert path_id == display_id or path_id.startswith(display_id + " "), (
        f"Model identity mismatch in {recipe_path.relative_to(ROOT_DIR)}:\n"
        f"  display_name '{display_name}' -> '{display_id}'\n"
        f"  model.path   '{model_path}' -> '{path_id}'"
    )


@pytest.mark.parametrize(
    "recipe_path",
    SOURCE_RECIPES,
    ids=[_get_id(p) for p in SOURCE_RECIPES],
)
def test_source_recipe_model_path_matches_filename(recipe_path):
    """For source recipes with an inline model path, the identity in
    ``model.path`` must match the filename."""
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["training_config"]["actor_rollout_ref"]["model"]["path"]

    path_id = _model_id_from_path(model_path)
    filename_id = _model_id_from_filename(recipe_path.name)

    assert path_id == filename_id, (
        f"Model identity mismatch in {recipe_path.relative_to(ROOT_DIR)}:\n"
        f"  filename   '{recipe_path.name}' -> '{filename_id}'\n"
        f"  model.path '{model_path}' -> '{path_id}'"
    )


# ---------------------------------------------------------------------------
# Tests for Hydra model configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config_path,model_path",
    MODEL_CONFIGS,
    ids=[_get_id(p) for p, _ in MODEL_CONFIGS],
)
def test_model_config_path_matches_config_filename(config_path, model_path):
    """The model identity in a Hydra model config's ``path`` field must
    match the config filename."""
    path_id = _model_id_from_path(model_path)
    filename_id = _model_id_from_filename(config_path.name)

    assert path_id == filename_id, (
        f"Model identity mismatch in {config_path.relative_to(ROOT_DIR)}:\n"
        f"  filename   '{config_path.name}' -> '{filename_id}'\n"
        f"  model.path '{model_path}' -> '{path_id}'"
    )
