"""
Test that override parameter type families are consistent within each template
category (fine_tuning, evaluation, checkpointless).

Type families:
  - "numeric": integer, float, number (all compatible)
  - "string": string
  - "boolean": boolean

The test auto-discovers all *_recipe_template_parameters.json files under
launcher/recipe_templatization/ and validates every override parameter against
the baseline at baseline_artifacts/override_param_types.json.

It fails if:
  - A parameter exists in templates but not in the baseline (missing)
  - A parameter's type family doesn't match the baseline (conflict)
  - A parameter exists in the baseline but not in any template (stale)

Environment variables:
  GOLDEN_TEST_WRITE=1: Auto-update the baseline file with current state.
                       Run this after adding new parameters.

Usage:
  pytest tests/launcher/recipe_templatization/test_override_param_type_consistency.py
  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_override_param_type_consistency.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path("launcher/recipe_templatization")
BASELINE_PATH = Path("tests/launcher/recipe_templatization/baseline_artifacts/override_param_types.json")
GOLDEN_WRITE = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")

# Nova eval template names — everything else in nova is train
NOVA_EVAL_TEMPLATES = {
    "nova_general_text_benchmark_eval",
    "nova_general_text_benchmark_2_0_eval",
    "nova_general_multi_modal_benchmark_eval",
    "nova_general_multi_modal_benchmark_2_0_eval",
    "nova_llm_judge_eval",
    "nova_bring_your_own_dataset_eval",
    "nova_bring_your_own_dataset_2_0_eval",
}

TYPE_FAMILIES = {
    "integer": "numeric",
    "float": "numeric",
    "number": "numeric",
    "string": "string",
    "boolean": "boolean",
}

# Map each framework to its category and optional nova filter
# The framework name is derived from the subdirectory name
FRAMEWORK_CATEGORIES = {
    "llmft": ("fine_tuning", None),
    "verl": ("fine_tuning", None),
    "nova": None,  # Special: split into train and eval
    "evaluation": ("evaluation", None),
    "checkpointless": ("checkpointless", None),
}


def _type_family(raw_type: str) -> str:
    return TYPE_FAMILIES.get(raw_type, raw_type)


def _get_category_for_template(framework: str, template_name: str) -> str:
    """Determine which category a (framework, template) belongs to."""
    if framework == "nova":
        return "evaluation" if template_name in NOVA_EVAL_TEMPLATES else "fine_tuning"
    mapping = FRAMEWORK_CATEGORIES.get(framework)
    if mapping is None:
        return "unknown"
    return mapping[0]


# ---------------------------------------------------------------------------
# Auto-discover all template parameter files
# ---------------------------------------------------------------------------


def _discover_template_files() -> Dict[str, Path]:
    """
    Find all *_recipe_template_parameters.json under launcher/recipe_templatization/.
    Returns {framework_name: path}.
    """
    files = {}
    for subdir in TEMPLATE_DIR.iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob("*_recipe_template_parameters.json"):
            framework = subdir.name
            files[framework] = f
    return files


# ---------------------------------------------------------------------------
# Collect all param instances from all templates
# ---------------------------------------------------------------------------


def _collect_all_instances() -> List[Tuple[str, str, str, str, str]]:
    """
    Returns list of (category, framework, template_name, param_name, type_family).
    """
    instances = []
    template_files = _discover_template_files()

    for framework, fpath in sorted(template_files.items()):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tname, tdata in data.get("templates", {}).items():
            category = _get_category_for_template(framework, tname)
            for pname, pdef in tdata.get("recipe_override_parameters", {}).items():
                raw_type = pdef.get("type", "MISSING")
                family = _type_family(raw_type)
                instances.append((category, framework, tname, pname, family))

    return instances


# ---------------------------------------------------------------------------
# Generate baseline from current state
# ---------------------------------------------------------------------------


def _generate_baseline(instances: List[Tuple[str, str, str, str, str]]) -> Dict:
    """Build baseline dict from collected instances."""
    # Group by category -> param -> {families, frameworks}
    cat_data = defaultdict(lambda: defaultdict(lambda: {"families": defaultdict(set), "frameworks": set()}))

    for category, framework, tname, pname, family in instances:
        cat_data[category][pname]["families"][family].add(framework)
        cat_data[category][pname]["frameworks"].add(framework)

    baseline = {
        "_comment": [
            "Override Parameter Type Consistency Baseline",
            "",
            "This file defines the expected type_family for every override parameter",
            "within each template category (fine_tuning, evaluation, checkpointless).",
            "",
            "Fields:",
            "  type_family: The expected type family ('numeric', 'string', 'boolean').",
            "               'numeric' covers integer, float, and number types.",
            "  frameworks:  List of frameworks where this parameter appears.",
            "",
            "The test ensures that within each category, a parameter's type family",
            "is consistent across all frameworks. For example, 'learning_rate' must",
            "be numeric in llmft, verl, and nova - it cannot be string in one and",
            "numeric in another.",
            "",
            "To auto-update this file when adding new parameters:",
            "  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_override_param_type_consistency.py",
        ]
    }

    for category in sorted(cat_data.keys()):
        cat_baseline = {}
        for pname in sorted(cat_data[category].keys()):
            info = cat_data[category][pname]
            families = info["families"]
            # Use the majority family if there's a conflict
            best_family = max(families.keys(), key=lambda f: len(families[f]))
            cat_baseline[pname] = {
                "type_family": best_family,
                "frameworks": sorted(info["frameworks"]),
            }
        baseline[category] = cat_baseline

    return baseline


# ---------------------------------------------------------------------------
# Load / write baseline
# ---------------------------------------------------------------------------


def _load_baseline() -> Dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_baseline(baseline: Dict):
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=4)
        f.write("\n")


# ---------------------------------------------------------------------------
# Module-level collection and golden write
# ---------------------------------------------------------------------------

ALL_INSTANCES = _collect_all_instances()

# If GOLDEN_TEST_WRITE is set, regenerate baseline at import time
# so all tests (including parametrized ones) see the updated baseline.
if GOLDEN_WRITE:
    _write_baseline(_generate_baseline(ALL_INSTANCES))
    print(f"GOLDEN_TEST_WRITE: Updated {BASELINE_PATH}")


# ---------------------------------------------------------------------------
# Test 1: Every param's type family must match the baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,framework,template_name,param_name,actual_family",
    ALL_INSTANCES,
    ids=[f"{i[0]}|{i[1]}::{i[2]}::{i[3]}" for i in ALL_INSTANCES],
)
def test_param_type_family_matches_baseline(category, framework, template_name, param_name, actual_family):
    """Each override parameter's type family must match the baseline for its category."""
    baseline = _load_baseline()

    assert category in baseline, f"Category '{category}' not in baseline. " f"Run with GOLDEN_TEST_WRITE=1 to update."

    cat_baseline = baseline[category]
    assert param_name in cat_baseline, (
        f"Parameter '{param_name}' in {framework}::{template_name} ({category}) "
        f"is NOT in the baseline. "
        f"Run with GOLDEN_TEST_WRITE=1 to update."
    )

    expected_family = cat_baseline[param_name]["type_family"]
    assert actual_family == expected_family, (
        f"Type family mismatch for '{param_name}' in {framework}::{template_name} ({category}): "
        f"expected '{expected_family}', got '{actual_family}'. "
        f"This means the parameter type is inconsistent across frameworks."
    )


# ---------------------------------------------------------------------------
# Test 2: No cross-framework type family conflicts within each category
# ---------------------------------------------------------------------------


def _get_categories() -> List[str]:
    return sorted(set(i[0] for i in ALL_INSTANCES))


@pytest.mark.parametrize("category", _get_categories())
def test_no_type_family_conflicts_within_category(category):
    """
    For every param within a category, all instances must share the same
    type family regardless of which framework they come from.
    """
    instances = [i for i in ALL_INSTANCES if i[0] == category]
    param_families = defaultdict(lambda: defaultdict(list))

    for _, framework, tname, pname, family in instances:
        param_families[pname][family].append(f"{framework}::{tname}")

    conflicts = []
    for pname, families in sorted(param_families.items()):
        if len(families) > 1:
            detail = {f: sorted(srcs) for f, srcs in families.items()}
            conflicts.append(f"  {pname}: {detail}")

    assert not conflicts, (
        f"Type family conflicts in '{category}':\n"
        + "\n".join(conflicts)
        + f"\n\nThis means the same parameter has incompatible types across frameworks."
    )


# ---------------------------------------------------------------------------
# Test 3: Baseline covers all params in templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _get_categories())
def test_baseline_covers_all_params(category):
    """Every param in templates must exist in the baseline for its category."""
    baseline = _load_baseline()
    cat_baseline = baseline.get(category, {})
    all_names = set(i[3] for i in ALL_INSTANCES if i[0] == category)
    baseline_names = set(cat_baseline.keys())

    missing = all_names - baseline_names
    assert not missing, (
        f"Parameters in {category} templates but NOT in baseline: {sorted(missing)}. "
        f"Run with GOLDEN_TEST_WRITE=1 to update."
    )


# ---------------------------------------------------------------------------
# Test 4: No stale baseline entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _get_categories())
def test_no_stale_baseline_entries(category):
    """Every param in the baseline must exist in at least one template."""
    baseline = _load_baseline()
    cat_baseline = baseline.get(category, {})
    all_names = set(i[3] for i in ALL_INSTANCES if i[0] == category)
    baseline_names = set(cat_baseline.keys())

    stale = baseline_names - all_names
    assert not stale, (
        f"Parameters in baseline but NOT in any {category} template: {sorted(stale)}. "
        f"Remove them from {BASELINE_PATH} or run with GOLDEN_TEST_WRITE=1."
    )
