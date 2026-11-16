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

"""llmft -> verl override-parameter parity for the released Llama verl recipes.

The Llama JumpStart entries move from the ``llmft_*`` recipes to the verl SFT/DPO
recipes. A customer job spec carried over from an llmft recipe may still send the
llmft override NAMES; the SDK rejects unknown override names, so every override the
llmft recipe advertised must also be advertised by its verl counterpart, with an
identical name (values are current-behaviour defaults, set elsewhere -- this test
never touches a value).

This module pins three things for each of the 16 released Llama pairs
(4 model sizes x {SFT, DPO} x {LoRA, FFT}):

1. The pairing itself -- an explicit ``(model, technique, peft) -> (llmft recipe,
   verl recipe)`` table -- so a future recipe rename on either side fails loudly
   here instead of silently dropping a customer-facing override.

2. Name parity -- every override NAME the llmft recipe advertises is also advertised
   by the verl counterpart. Extra overrides on the verl side are fine (verl exposes
   ``min_lr``/``warmup_steps`` that llmft does not). The one deliberate exception is
   ``merge_weights`` on the LoRA pairs, listed in ``ALLOWLISTED_MISSING_NAMES``.

3. Placeholder landing -- for every SHARED override whose verl advertisement comes
   from a ``{{placeholder}}`` in the matched verl template, that placeholder lands on
   at least one dotted path that actually exists in the verl recipe body. This rules
   out a phantom override: a name advertised (because the placeholder is in the
   matched template) but never applied to the recipe. This is the same landing rule
   ``test_verl_placeholder_landing_invariant.py`` enforces repo-wide, pinned here to
   the exact shared-parity names on the exact Llama recipes.

Advertised override sets are resolved through the repo's REAL resolution path:
``resolve_params(base, template.override_parameters, template.recipe_template)`` for
the template-level view (as ``test_verl_length_override_params.py`` does), and the
concrete ``LLMFTRecipeTemplateProcessor`` / ``VerlRecipeTemplateProcessor`` for the
per-recipe view -- the latter runs ``_prune_unlanded_placeholder_params``, so its
output is the set the SDK actually sees.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from launcher.recipe_templatization.llmft.llmft_recipe_template_processor import (
    LLMFTRecipeTemplateProcessor,
)
from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)
from utils.resolve_override_params import resolve_params

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_DIR = REPO_ROOT / "launcher" / "recipe_templatization"
BASE_FILE = TEMPLATE_DIR / "base_override_parameters.json"
LLMFT_TEMPLATE_FILE = TEMPLATE_DIR / "llmft" / "llmft_recipe_template_parameters.json"
VERL_TEMPLATE_FILE = TEMPLATE_DIR / "verl" / "verl_recipe_template_parameters.json"
RECIPES_DIR = REPO_ROOT / "recipes_collection" / "recipes"

PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")

# Recipe directories (relative to recipes_collection/recipes), no .yaml suffix.
_LLMFT_DIR = "fine-tuning/llama"
_VERL_DIR = "fine-tuning/llama-0_7_0"


class Pair:
    """One (model, technique, peft) coordinate and the two recipes it binds.

    ``llmft_key`` / ``verl_key`` are the template keys the two processors resolve for
    these recipes (llmft keys on the filename technique+peft; verl on algorithm +
    ``fft`` in the name; every recipe outside Nemotron uses the default sft/dpo templates).
    They are pinned here so a template-selection regression is caught alongside a
    rename.
    """

    def __init__(self, model, technique, peft, llmft, verl, llmft_key, verl_key):
        self.model = model
        self.technique = technique
        self.peft = peft
        self.llmft = f"{_LLMFT_DIR}/{llmft}"
        self.verl = f"{_VERL_DIR}/{verl}"
        self.llmft_key = llmft_key
        self.verl_key = verl_key

    @property
    def id(self):
        return f"{self.model}-{self.technique}-{self.peft}"


# Explicit pairing table: model size x technique x PEFT -> (llmft recipe, verl recipe).
# Sourced from PARITY_ROUND2_BRIEF.md. A rename on either side breaks the file-exists
# test below rather than silently changing what a customer can override.
PAIRS = [
    # 3.2-1b
    Pair(
        "3.2-1b",
        "sft",
        "lora",
        "llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
        "verl-sft-llama-3-dot-2-1b-instruct-g6-48xl-17k-tt-lora",
        "llmft_sft_lora",
        "sft",
    ),
    Pair(
        "3.2-1b",
        "sft",
        "fft",
        "llmft_llama3_2_1b_instruct_seq4k_gpu_sft_fft",
        "verl-sft-llama-3-dot-2-1b-instruct-g6-48xl-15k-tt-fft",
        "llmft_sft_fft",
        "sft_fft",
    ),
    Pair(
        "3.2-1b",
        "dpo",
        "lora",
        "llmft_llama3_2_1b_instruct_seq4k_gpu_dpo",
        "verl-dpo-llama-3-dot-2-1b-instruct-g5-48xl-8k-tt-lora",
        "llmft_dpo",
        "dpo",
    ),
    Pair(
        "3.2-1b",
        "dpo",
        "fft",
        "llmft_llama3_2_1b_instruct_seq4k_gpu_dpo_fft",
        "verl-dpo-llama-3-dot-2-1b-instruct-g6-48xl-4k-tt-fft",
        "llmft_dpo_fft",
        "dpo_fft",
    ),
    # 3.2-3b
    Pair(
        "3.2-3b",
        "sft",
        "lora",
        "llmft_llama3_2_3b_instruct_seq4k_gpu_sft_lora",
        "verl-sft-llama-3-dot-2-3b-instruct-g6-48xl-6k-tt-lora",
        "llmft_sft_lora",
        "sft",
    ),
    Pair(
        "3.2-3b",
        "sft",
        "fft",
        "llmft_llama3_2_3b_instruct_seq4k_gpu_sft_fft",
        "verl-sft-llama-3-dot-2-3b-instruct-g6-48xl-13k-tt-fft",
        "llmft_sft_fft",
        "sft_fft",
    ),
    Pair(
        "3.2-3b",
        "dpo",
        "lora",
        "llmft_llama3_2_3b_instruct_seq4k_gpu_dpo",
        "verl-dpo-llama-3-dot-2-3b-instruct-g5-48xl-6k-tt-lora",
        "llmft_dpo",
        "dpo",
    ),
    Pair(
        "3.2-3b",
        "dpo",
        "fft",
        "llmft_llama3_2_3b_instruct_seq4k_gpu_dpo_fft",
        "verl-dpo-llama-3-dot-2-3b-instruct-g6-48xl-6k-tt-fft",
        "llmft_dpo_fft",
        "dpo_fft",
    ),
    # 3.1-8b
    Pair(
        "3.1-8b",
        "sft",
        "lora",
        "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
        "verl-sft-llama-3-dot-1-8b-instruct-g6-48xl-11k-tt-lora",
        "llmft_sft_lora",
        "sft",
    ),
    Pair(
        "3.1-8b",
        "sft",
        "fft",
        "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft",
        "verl-sft-llama-3-dot-1-8b-instruct-g6-48xl-5k-tt-fft",
        "llmft_sft_fft",
        "sft_fft",
    ),
    Pair(
        "3.1-8b",
        "dpo",
        "lora",
        "llmft_llama3_1_8b_instruct_seq4k_gpu_dpo",
        "verl-dpo-llama-3-dot-1-8b-instruct-g5-48xl-6k-tt-lora",
        "llmft_dpo",
        "dpo",
    ),
    Pair(
        "3.1-8b",
        "dpo",
        "fft",
        "llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_fft",
        "verl-dpo-llama-3-dot-1-8b-instruct-p5-48xl-12k-tt-fft",
        "llmft_dpo_fft",
        "dpo_fft",
    ),
    # 3.3-70b
    Pair(
        "3.3-70b",
        "sft",
        "lora",
        "llmft_llama3_3_70b_instruct_seq4k_gpu_sft_lora",
        "verl-sft-llama-3-dot-3-70b-instruct-p5-48xl-8k-tt-lora",
        "llmft_sft_lora",
        "sft",
    ),
    Pair(
        "3.3-70b",
        "sft",
        "fft",
        "llmft_llama3_3_70b_instruct_seq4k_gpu_sft_fft",
        "verl-sft-llama-3-dot-3-70b-instruct-p5-48xl-8k-tt-fft",
        "llmft_sft_fft",
        "sft_fft",
    ),
    Pair(
        "3.3-70b",
        "dpo",
        "lora",
        "llmft_llama3_3_70b_instruct_seq4k_gpu_dpo",
        "verl-dpo-llama-3-dot-3-70b-instruct-p4de-24xl-4k-tt-lora",
        "llmft_dpo",
        "dpo",
    ),
    Pair(
        "3.3-70b",
        "dpo",
        "fft",
        "llmft_llama3_3_70b_instruct_seq4k_gpu_dpo_fft",
        "verl-dpo-llama-3-dot-3-70b-instruct-p5-48xl-4k-tt-fft",
        "llmft_dpo_fft",
        "dpo_fft",
    ),
]

# Deliberate name-parity exceptions: llmft override names that the verl counterpart is
# NOT required to advertise. Add a name here only with a comment citing why verl does not
# expose it -- never to paper over a real gap.
#
# merge_weights (LoRA pairs): verl LoRA recipes always merge the adapter into the base model
# (tests/test_verl_recipes_constraints.py pins merge_lora_on_final_save: true), and verl does
# not support adapter-only output yet: with merging off the job fails after its last step. So
# the verl recipes do not offer the switch, and a job spec that sends merge_weights is rejected.
ALLOWLISTED_MISSING_NAMES: dict = {
    (pair.model, pair.technique, pair.peft): {"merge_weights"} for pair in PAIRS if pair.peft == "lora"
}

# Sixteen pairs, four coordinates each on two axes -- guards the table itself.
EXPECTED_PAIR_COUNT = 16
EXPECTED_MODELS = {"3.2-1b", "3.2-3b", "3.1-8b", "3.3-70b"}


def _flatten(node, path=""):
    """Flatten a nested dict to {dotted.path: leaf_value}. Mirrors the landing-invariant test."""
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


def _load_yaml(rel_no_suffix: str) -> dict:
    with open(RECIPES_DIR / (rel_no_suffix + ".yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _template_resolved_names(template_file: Path, template_key: str) -> set:
    """Advertised override names for a template, via the real resolution path."""
    base = json.loads(BASE_FILE.read_text())["fine_tuning"]
    template = json.loads(template_file.read_text())["templates"][template_key]
    return set(
        resolve_params(
            base,
            template.get("override_parameters", {}),
            template.get("recipe_template", {}),
        )
    )


@pytest.fixture(scope="module")
def llmft_processor():
    base = json.loads(BASE_FILE.read_text())["fine_tuning"]
    return LLMFTRecipeTemplateProcessor(staging_cfg=base, platform="sm_jobs")


@pytest.fixture(scope="module")
def verl_processor():
    base = json.loads(BASE_FILE.read_text())["fine_tuning"]
    return VerlRecipeTemplateProcessor(staging_cfg=base, platform="sm_jobs")


def _resolve_recipe_names(processor, recipe_rel: str) -> set:
    """Per-recipe advertised override names, AFTER unlanded-placeholder pruning."""
    processor.process_recipe(recipe_file_path=recipe_rel)
    return set(processor.recipe_override_parameters or {})


PAIR_IDS = [p.id for p in PAIRS]


class TestPairingTable:
    """Pin the pairing itself so a recipe rename fails loudly here."""

    def test_pair_count_and_coverage(self):
        assert len(PAIRS) == EXPECTED_PAIR_COUNT
        assert {p.model for p in PAIRS} == EXPECTED_MODELS
        # Every model must carry all four technique x PEFT coordinates.
        for model in EXPECTED_MODELS:
            coords = {(p.technique, p.peft) for p in PAIRS if p.model == model}
            assert coords == {("sft", "lora"), ("sft", "fft"), ("dpo", "lora"), ("dpo", "fft")}, model
        # Coordinates are unique (no accidental duplicate row).
        assert len({p.id for p in PAIRS}) == EXPECTED_PAIR_COUNT

    @pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
    def test_both_recipes_exist(self, pair):
        assert (RECIPES_DIR / (pair.llmft + ".yaml")).is_file(), f"missing llmft recipe: {pair.llmft}"
        assert (RECIPES_DIR / (pair.verl + ".yaml")).is_file(), f"missing verl recipe: {pair.verl}"

    @pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
    def test_template_keys_match_processor_selection(self, pair, llmft_processor, verl_processor):
        # The template each processor actually selects for the recipe must equal the
        # key pinned in the table -- catches a template-routing regression.
        llmft_all = json.loads(LLMFT_TEMPLATE_FILE.read_text())["templates"]
        verl_all = json.loads(VERL_TEMPLATE_FILE.read_text())["templates"]

        llmft_tmpl = llmft_processor.get_recipe_template({}, llmft_all, pair.llmft)
        assert llmft_tmpl is llmft_all[pair.llmft_key], f"{pair.id}: llmft key drift"

        verl_yaml = _load_yaml(pair.verl)
        verl_tmpl = verl_processor.get_recipe_template(verl_yaml, verl_all, pair.verl)
        assert verl_tmpl is verl_all[pair.verl_key], f"{pair.id}: verl key drift"


class TestTemplateLevelNameParity:
    """Fast template-level view: every llmft-template name is on the verl counterpart."""

    @pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
    def test_llmft_names_subset_of_verl(self, pair):
        llmft_names = _template_resolved_names(LLMFT_TEMPLATE_FILE, pair.llmft_key)
        verl_names = _template_resolved_names(VERL_TEMPLATE_FILE, pair.verl_key)
        allow = ALLOWLISTED_MISSING_NAMES.get((pair.model, pair.technique, pair.peft), set())
        missing = llmft_names - verl_names - allow
        assert not missing, (
            f"{pair.id}: verl template {pair.verl_key!r} does not advertise llmft override "
            f"name(s) {sorted(missing)} -- a customer job spec carrying these would be rejected"
        )


class TestPerRecipeNameParity:
    """Authoritative view: names the SDK actually sees, after unlanded-placeholder pruning."""

    @pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
    def test_llmft_recipe_names_subset_of_verl_recipe(self, pair, llmft_processor, verl_processor):
        llmft_names = _resolve_recipe_names(llmft_processor, pair.llmft)
        verl_names = _resolve_recipe_names(verl_processor, pair.verl)
        allow = ALLOWLISTED_MISSING_NAMES.get((pair.model, pair.technique, pair.peft), set())
        missing = llmft_names - verl_names - allow
        assert not missing, (
            f"{pair.id}: verl recipe {Path(pair.verl).name} does not advertise llmft override "
            f"name(s) {sorted(missing)} after pruning -- either the verl recipe body lacks the "
            f"key the placeholder lands on (phantom -> pruned) or the templates diverged. "
            f"This is a recipe/template gap to fix, not a value to change."
        )


class TestSharedPlaceholdersLand:
    """Every shared override that is a verl-template placeholder lands on the verl recipe body.

    Guards against a phantom override: a name advertised because the placeholder sits in
    the matched template, but never substituted because the recipe lacks the dotted path.
    """

    @pytest.mark.parametrize("pair", PAIRS, ids=PAIR_IDS)
    def test_shared_param_placeholders_land_on_verl_recipe(self, pair):
        llmft_names = _template_resolved_names(LLMFT_TEMPLATE_FILE, pair.llmft_key)
        verl_names = _template_resolved_names(VERL_TEMPLATE_FILE, pair.verl_key)
        shared = llmft_names & verl_names

        verl_template = json.loads(VERL_TEMPLATE_FILE.read_text())["templates"][pair.verl_key]
        flat_template = _flatten(verl_template.get("recipe_template", {}))

        # placeholder name -> [template dotted paths it appears at]
        paths_by_placeholder: dict = {}
        for tpath, value in flat_template.items():
            if not isinstance(value, str):
                continue
            m = PLACEHOLDER.search(value)
            if m:
                paths_by_placeholder.setdefault(m.group(1), []).append(tpath)

        recipe = _load_yaml(pair.verl)
        phantom = []
        for name in sorted(shared):
            paths = paths_by_placeholder.get(name)
            if not paths:
                # Shared param advertised via the template's override_parameters block,
                # not a {{placeholder}} -> no landing path to check.
                continue
            if not any(_path_exists(recipe, p) for p in paths):
                phantom.append(f"{name} lands on none of {sorted(paths)}")

        assert not phantom, (
            f"{pair.id}: shared override(s) advertised on verl recipe {Path(pair.verl).name} "
            f"but their placeholder lands on no recipe path (phantom override):\n  " + "\n  ".join(phantom)
        )
