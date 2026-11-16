"""Repo-wide check of where the verl SFT/DPO template placeholders land.

For every verl recipe under `recipes_collection/recipes/fine-tuning/*/verl-*.yaml` and its matched
template (same selection rule as `VerlRecipeTemplateProcessor.get_recipe_template`: algorithm type,
`fft` in the filename, and the `_nemotron` suffix for the recipes
`VerlRecipeTemplateProcessor._uses_nemotron_templates` selects: Nemotron), a placeholder that lands on none of its
template paths is a placeholder the recipe does not carry. `_prune_unlanded_placeholder_params`
drops it from that recipe's advertised overrides at generation time.

The default sft/sft_fft/dpo/dpo_fft templates carry the llmft override names, and only the Llama 3.x,
DeepSeek R1 distill, Qwen 2.5 and Qwen 3 recipes carry every llmft key, so an unlanded placeholder is
expected on the other families. This module still catches a template mistake by pinning, per family:

- every placeholder of the matched template lands on every Llama 3.x, DeepSeek, Qwen 2.5 and Qwen 3 recipe;
- every placeholder of the matched `*_nemotron` template lands on every Nemotron recipe;
- on every other family, the only unlanded placeholders are llmft keys (and the pruning removes
  them from the published overrides, which `test_unlanded_placeholder_pruning.py` covers);
- `{{min_lr}}` lands on `optim.min_lr_ratio` (a fraction of the peak learning rate, as the default
  templates' description says) on every recipe on the default templates, and on `optim.min_lr` on
  the `*_nemotron` ones.

GRPO/PPO recipes (with `training_config.algorithm.adv_estimator`) use their own templates and are
out of scope.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES_DIR = REPO_ROOT / "recipes_collection" / "recipes" / "fine-tuning"
TEMPLATE_PATH = REPO_ROOT / "launcher" / "recipe_templatization" / "verl" / "verl_recipe_template_parameters.json"

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

# Families whose recipes carry every llmft key, so every placeholder must land.
FULL_LANDING_DIRS = ("llama-0_7_0", "deepseek-0_7_0", "nemotron-0_7_0")
# Models in other family directories whose recipes also carry every llmft key: the Qwen 2.5 and
# Qwen 3 models that replace llmft recipes on JumpStart. qwen-0_7_0 also holds Qwen 3.5/3.6/3.8,
# which do not, so these are matched by recipe name.
FULL_LANDING_NAME_PREFIXES = tuple(
    f"verl-{technique}-{model}-"
    for technique in ("sft", "dpo")
    for model in (
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
)


def _full_landing(family: str, name: str) -> bool:
    # The Qwen match covers ladder rungs ("-tt-lora" / "-tt-fft"), which inherit the edited bases.
    # The few standalone qwen-0_7_0 recipes with the same model names (version 2.x, never on
    # JumpStart) compose their own config and are held to the other-family rule instead.
    if family in FULL_LANDING_DIRS:
        return True
    return name.startswith(FULL_LANDING_NAME_PREFIXES) and re.search(r"-tt-(lora|fft)\.yaml$", name) is not None


# The placeholders the default templates add for llmft compatibility (the only ones a recipe of
# another family may lack). Kept in step with EXPECTED_LLMFT_FIELDS in test_verl_sft_dpo_templates.py.
LLMFT_PLACEHOLDERS = {
    "logging_steps",
    "temperature",
    "max_response_length",
    "gradient_clipping_threshold",
    "gradient_clipping",
    "training_data_name",
    "validation_data_name",
    "seed",
    "lora_dropout",
}


def _flatten(node, path=""):
    """Flatten a template/recipe dict to dotted paths -> leaf value."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                out.update(_flatten(value, new_path))
            else:
                out[new_path] = value
    return out


def _path_exists(recipe: dict, dotted: str) -> bool:
    cur = recipe
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _template_key_for(recipe: dict, recipe_rel: str):
    """Mirror VerlRecipeTemplateProcessor.get_recipe_template's key selection.

    Returns None for GRPO/PPO (adv_estimator set), which use grpo_* templates keyed on
    reward mechanism/display name -- out of scope here.
    """
    training_config = recipe.get("training_config", {})
    algorithm = training_config.get("algorithm")
    if algorithm and algorithm.get("adv_estimator"):
        return None
    trainer = training_config.get("trainer", {})
    algo = "dpo" if "beta" in trainer else "sft"
    key = f"{algo}_fft" if "fft" in Path(recipe_rel).name.lower() else algo
    if VerlRecipeTemplateProcessor._uses_nemotron_templates(recipe_rel, recipe):
        key += VerlRecipeTemplateProcessor.NEMOTRON_TEMPLATE_SUFFIX
    return key


def _sft_dpo_recipes():
    """(family dir, recipe, template key) for every verl SFT/DPO recipe."""
    out = []
    for path in sorted(RECIPES_DIR.rglob("verl-*.yaml")):
        recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(recipe, dict):
            continue
        rel = path.relative_to(RECIPES_DIR.parent).as_posix()[: -len(".yaml")]
        key = _template_key_for(recipe, rel)
        if key is not None:
            out.append((path.parent.name, path.name, recipe, key))
    return out


SFT_DPO_RECIPES = _sft_dpo_recipes()


@pytest.fixture(scope="module")
def templates():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["templates"]


def _unlanded(recipe: dict, recipe_template: dict) -> dict:
    """Placeholder name -> template paths, for each placeholder that lands on none of its paths."""
    paths_by_placeholder: dict = {}
    for tpath, value in _flatten(recipe_template).items():
        if isinstance(value, str):
            m = PLACEHOLDER.search(value)
            if m:
                paths_by_placeholder.setdefault(m.group(1), []).append(tpath)
    return {
        name: sorted(paths)
        for name, paths in paths_by_placeholder.items()
        if not any(_path_exists(recipe, p) for p in paths)
    }


def test_verl_recipes_exist():
    assert SFT_DPO_RECIPES, f"No verl SFT/DPO recipes found under {RECIPES_DIR}"
    families = {family for family, _, _, _ in SFT_DPO_RECIPES}
    # Guard against the checks below silently matching nothing (e.g. after a directory rename).
    assert set(FULL_LANDING_DIRS) <= families
    assert any(key.endswith(VerlRecipeTemplateProcessor.NEMOTRON_TEMPLATE_SUFFIX) for _, _, _, key in SFT_DPO_RECIPES)


def test_every_placeholder_lands_on_llama_deepseek_and_nemotron(templates):
    """The Llama 3.x, DeepSeek, Qwen 2.5 and Qwen 3 recipes carry every llmft key; Nemotron carries
    every key its templates name."""
    violations = []
    checked_qwen = 0
    for family, name, recipe, key in SFT_DPO_RECIPES:
        if not _full_landing(family, name):
            continue
        checked_qwen += family not in FULL_LANDING_DIRS
        assert key in templates, f"{name}: template key {key!r} not in template file"
        for placeholder, paths in _unlanded(recipe, templates[key]["recipe_template"]).items():
            violations.append(f"[{key}] {family}/{name}: {{{{{placeholder}}}}} lands on none of {paths}")
    assert checked_qwen, f"no recipe matched {FULL_LANDING_NAME_PREFIXES[0]}... (renamed?)"
    if violations:
        pytest.fail("Template placeholder(s) that a recipe does not carry:\n  " + "\n  ".join(sorted(violations)))


def test_other_families_only_lack_llmft_keys(templates):
    """On every other family a placeholder may be unlanded only if it is one of the llmft keys."""
    violations = []
    for family, name, recipe, key in SFT_DPO_RECIPES:
        if _full_landing(family, name):
            continue
        assert not key.endswith(VerlRecipeTemplateProcessor.NEMOTRON_TEMPLATE_SUFFIX), f"{family}/{name} -> {key}"
        for placeholder, paths in _unlanded(recipe, templates[key]["recipe_template"]).items():
            if placeholder not in LLMFT_PLACEHOLDERS:
                violations.append(f"[{key}] {family}/{name}: {{{{{placeholder}}}}} lands on none of {paths}")
    if violations:
        pytest.fail("Non-llmft placeholder(s) that a recipe does not carry:\n  " + "\n  ".join(sorted(violations)))


def test_min_lr_lands_on_the_ratio_outside_nemotron():
    """{{min_lr}} is a fraction of the peak LR (optim.min_lr_ratio) on the default templates' recipes."""
    wrong = []
    for family, name, recipe, key in SFT_DPO_RECIPES:
        nemotron = key.endswith(VerlRecipeTemplateProcessor.NEMOTRON_TEMPLATE_SUFFIX)
        has_ratio = _path_exists(recipe, "training_config.optim.min_lr_ratio")
        has_absolute = _path_exists(recipe, "training_config.optim.min_lr")
        if nemotron and not (has_absolute and not has_ratio):
            wrong.append(f"{family}/{name}: recipe on a Nemotron template should carry optim.min_lr only")
        if not nemotron and not (has_ratio and not has_absolute):
            wrong.append(f"{family}/{name}: should carry optim.min_lr_ratio only")
    assert not wrong, "\n  ".join(wrong)
