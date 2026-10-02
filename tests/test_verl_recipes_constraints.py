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

import hashlib
import re

import pytest
import yaml

from launcher.nemo.constants import ROOT_DIR

RECIPES_DIR = ROOT_DIR / "recipes_collection" / "recipes" / "fine-tuning"

MAX_TOTAL_EPOCHS = 10

# Ladder rungs — one file per measured token budget for a (model, technique, tuning) slot,
# named `verl-<sft|dpo>-<model>-<instance>-<N>k-tt-<tuning>`. The length label N is
# `training_config.data.max_length // 1024`; the instance segment names the instance the
# sweep was measured on. Kept in step with scripts/enforce_recipe_instance_conventions.py,
# which is the executable statement of the instance-type conventions on the same files.
# Deliberately NOT anchored on the tuning suffix: a new tuning (say `-tt-qlora`) must fail
# these checks rather than silently fall out of the collection and go unchecked.
LADDER_STEM_RE = re.compile(r"^verl-(?:sft|dpo)-.*-(\d+)k-tt-")
LADDER_INSTANCE_SEG_RE = re.compile(r"-(?:g5|g6|g6e|p4d|p4de|p5|p5e|p5en)-(\d+)xl-")
# The `(Nk)` suffix every rung's display_name carries — the customer-visible length.
DISPLAY_NAME_LABEL_RE = re.compile(r"\((\d+)k\)\s*$")
TOKENS_PER_LABEL_STEP = 1024
GPUS_BY_SIZE = {"48": 8, "24": 8, "12": 4}


def _collect_verl_recipe_files():
    """Collect all verl recipe YAML files under the fine-tuning recipes directory."""
    return sorted(RECIPES_DIR.rglob("verl-*.yaml"))


def _get_recipe_id(path):
    """Return a short human-readable identifier for the recipe file."""
    return str(path.relative_to(RECIPES_DIR))


def _ladder_length_label(stem):
    """The N from a rung's `-Nk-tt-` segment."""
    return int(LADDER_STEM_RE.match(stem).group(1))


def _ladder_gpu_count(stem):
    """GPUs per node for the instance a rung's filename names, or None if unrecognised.

    `*.12xlarge` is 4 GPUs, `*.24xlarge` and `*.48xlarge` are 8. An unknown suffix is
    reported rather than guessed, matching scripts/enforce_recipe_instance_conventions.py.
    """
    match = LADDER_INSTANCE_SEG_RE.search(stem)
    return GPUS_BY_SIZE.get(match.group(1)) if match else None


def _fingerprint_without_display_name(text):
    """Digest of a resolved recipe with its `display_name:` line removed.

    display_name is the one field that carries the length label in prose, so it is the one
    field that differs between two rungs that are otherwise the very same training job.
    Stripping it turns "these two look similar" into "these two are the same recipe".
    """
    body = "\n".join(line for line in text.splitlines() if not line.startswith("display_name:"))
    return hashlib.md5(body.encode()).hexdigest()


def _group_ladder_rungs_by_slot():
    """{(run.name, gpus_per_node): [(path, max_length, fingerprint)]} for every ladder rung.

    The GPU count is part of the key on purpose: two rungs at the same length on a 4-GPU
    and an 8-GPU instance are two distinct measurements, not a duplicated one.
    """
    slots = {}
    for path in VERL_LADDER_RECIPE_FILES:
        text = path.read_text()
        config = yaml.safe_load(text)
        gpus = _ladder_gpu_count(path.stem)
        if gpus is None:
            # test_ladder_rung_names_a_known_gpu_class fails on these; skipping here keeps
            # the grouping honest instead of lumping unrelated rungs together.
            continue
        key = (config["run"]["name"], gpus)
        slots.setdefault(key, []).append(
            (path, config["training_config"]["data"]["max_length"], _fingerprint_without_display_name(text))
        )
    return slots


VERL_RECIPE_FILES = _collect_verl_recipe_files()
VERL_GRPO_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if "grpo" in str(f)]
VERL_LORA_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if "-lora" in f.stem]
VERL_LADDER_RECIPE_FILES = [f for f in VERL_RECIPE_FILES if LADDER_STEM_RE.match(f.stem)]

LADDER_SLOTS = _group_ladder_rungs_by_slot()
# Slots with a single rung cannot collide, so only multi-rung slots are worth a test case.
LADDER_MULTI_RUNG_SLOTS = sorted(slot for slot, rungs in LADDER_SLOTS.items() if len(rungs) > 1)


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


@pytest.mark.parametrize(
    "recipe_path",
    VERL_LADDER_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_LADDER_RECIPE_FILES],
)
def test_ladder_rung_names_a_known_gpu_class(recipe_path):
    """
    Verify every ladder rung's filename names an instance whose GPU count is known.

    The duplicate-rung tests below group rungs by GPU count. A rung whose size suffix is
    unmapped would be dropped from that grouping instead of failing it, so the grouping is
    only as trustworthy as this assertion.
    """
    gpus = _ladder_gpu_count(recipe_path.stem)

    assert gpus is not None, (
        f"Ladder rung filename does not name a recognised instance size "
        f"(expected one of {sorted(GPUS_BY_SIZE)}xlarge): {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "recipe_path",
    VERL_LADDER_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_LADDER_RECIPE_FILES],
)
def test_ladder_length_label_matches_max_length(recipe_path):
    """
    Verify the `-Nk-tt-` length label in a ladder rung's filename matches data.max_length.

    The label is the only place the length appears in the rung's name, and
    scripts/generate_recipes_doc.py derives docs/RECIPES.md's length column from the
    filename. What the job actually trains at is training_config.data.max_length, which is
    also what the template processor reports to the hub as SequenceLength
    (VerlRecipeTemplateProcessor._extract_sequence_length). A rung whose max_length was
    lowered to clear an OOM but kept its old label therefore misrepresents itself in both
    the docs and the hub metadata while training at something else entirely.

    The check is the floor rule, not equality: sweep-measured ceilings are not all
    multiples of 1024 (25000, 10752, 17920 are all real), so the label is
    `max_length // 1024` and the recipe is correct when max_length falls anywhere inside
    that label's 1024-token band.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    label = _ladder_length_label(recipe_path.stem)
    max_length = config["training_config"]["data"]["max_length"]
    band_start = label * TOKENS_PER_LABEL_STEP
    band_end = band_start + TOKENS_PER_LABEL_STEP

    assert band_start <= max_length < band_end, (
        f"Ladder rung length label disagrees with what it trains at: label '{label}k' "
        f"claims {band_start} <= max_length < {band_end}, but max_length={max_length} "
        f"(label should be {max_length // TOKENS_PER_LABEL_STEP}k) "
        f"in {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "recipe_path",
    VERL_LADDER_RECIPE_FILES,
    ids=[_get_recipe_id(p) for p in VERL_LADDER_RECIPE_FILES],
)
def test_ladder_display_name_label_matches_filename(recipe_path):
    """
    Verify a ladder rung's display_name `(Nk)` suffix matches its filename length label.

    display_name is the customer-visible string the hub renders, so it is the one place a
    stale label is read by a human rather than by a tool. It is written by hand alongside
    the filename and drifts in lockstep with it, which is why it needs its own assertion
    rather than being trusted once the filename is checked.
    """
    with open(recipe_path, "r") as f:
        config = yaml.safe_load(f)

    label = _ladder_length_label(recipe_path.stem)
    display_name = config.get("display_name")
    match = DISPLAY_NAME_LABEL_RE.search(display_name or "")

    assert match, (
        f"Ladder rung display_name is missing the '(Nk)' length suffix every rung carries: "
        f"got {display_name!r} in {recipe_path.relative_to(ROOT_DIR)}"
    )
    assert int(match.group(1)) == label, (
        f"Ladder rung display_name length disagrees with its filename: filename says "
        f"'{label}k', display_name says '{match.group(1)}k' ({display_name!r}) "
        f"in {recipe_path.relative_to(ROOT_DIR)}"
    )


@pytest.mark.parametrize(
    "slot",
    LADDER_MULTI_RUNG_SLOTS,
    ids=[f"{name}-{gpus}gpu" for name, gpus in LADDER_MULTI_RUNG_SLOTS],
)
def test_no_two_ladder_rungs_in_a_slot_share_a_max_length(slot):
    """
    Verify no two ladder rungs in one (run.name, GPU-count) slot train at the same length.

    A rung exists to offer one measured token budget. Two rungs at the same length in the
    same GPU class are the same offer twice: whichever the customer picks, they get the
    same job. This is how lowering one rung's max_length onto another's value shows up.

    The GPU count is part of the slot deliberately — the same length on a 4-GPU and an
    8-GPU instance are two genuine measurements and must not fail here.
    """
    by_length = {}
    for path, max_length, _fingerprint in LADDER_SLOTS[slot]:
        by_length.setdefault(max_length, []).append(path.stem)

    collisions = {length: sorted(stems) for length, stems in by_length.items() if len(stems) > 1}

    assert not collisions, (
        f"Slot {slot[0]} ({slot[1]} GPUs) has rungs sharing a max_length — merge them, "
        f"keeping the lower, and record the retired rung's measured ceiling in "
        f"scripts/measured_ceilings.json first:\n"
        + "\n".join(f"  max_length={length}: {', '.join(stems)}" for length, stems in sorted(collisions.items()))
    )


@pytest.mark.parametrize(
    "slot",
    LADDER_MULTI_RUNG_SLOTS,
    ids=[f"{name}-{gpus}gpu" for name, gpus in LADDER_MULTI_RUNG_SLOTS],
)
def test_no_two_ladder_rungs_in_a_slot_are_byte_identical(slot):
    """
    Verify no two ladder rungs in one slot are byte-identical apart from display_name.

    The stronger form of the check above. Once the instance-type conventions have run,
    every rung in a slot and GPU class that reaches a given length advertises the same
    instance_types, so two rungs driven onto the same length become the same file except
    for the `(Nk)` prose in display_name. Comparing the digest of the file minus that one
    line is how the real duplicate was found.
    """
    by_fingerprint = {}
    for path, _max_length, fingerprint in LADDER_SLOTS[slot]:
        by_fingerprint.setdefault(fingerprint, []).append(path.stem)

    duplicates = [sorted(stems) for stems in by_fingerprint.values() if len(stems) > 1]

    assert not duplicates, (
        f"Slot {slot[0]} ({slot[1]} GPUs) has rungs that are the same recipe once "
        f"display_name is stripped — merge them, keeping the lower max_length, and record "
        f"the retired rung's measured ceiling in scripts/measured_ceilings.json first:\n"
        + "\n".join(f"  {', '.join(stems)}" for stems in sorted(duplicates))
    )


def test_ladder_instance_types_match_measured_ceilings():
    """Every ladder rung already advertises exactly the instances the rules allow.

    Locks the output of scripts/enforce_recipe_instance_conventions.py so a recipe cannot
    drift from it by hand, and — the reason this test exists — so that deleting a rung file
    cannot silently drop a measured instance. A byte-identical duplicate rung is safe to
    delete only if its sweep is first recorded in scripts/measured_ceilings.json; delete the
    file without the entry (or delete the entry later) and the instance stops being
    admissible on every rung it covered, which shows up here as drift.
    """
    from scripts.enforce_recipe_instance_conventions import (
        load_ladders,
        load_manifest,
        target_state,
    )

    plan, problems = target_state(load_ladders(), load_manifest())
    assert problems == [], "instance-convention problems:\n  " + "\n  ".join(problems)
    assert plan, "no ladder rungs found — the discovery glob is broken"

    drift = [
        f"{rung['stem']}: has {sorted(rung['instance_types'])}, rules allow {want}"
        for rung, want, _gpus in plan
        if sorted(rung["instance_types"]) != want
    ]
    assert drift == [], (
        f"{len(drift)} rung(s) drifted from the measured-ceiling rules. Run "
        "`python scripts/enforce_recipe_instance_conventions.py --check` for the full diff:\n  " + "\n  ".join(drift)
    )
