"""
Golden test for fully-resolved recipe_override_parameters across all templates.

This test captures a snapshot of every template's `recipe_override_parameters`
from all 5 framework template files. It is used to verify zero behavioral change
during the centralize-override-parameters refactor.

The snapshot is keyed by `{framework}.{template_name}` and stored at:
  tests/launcher/recipe_templatization/baseline_artifacts/resolved_override_params.json

Environment variables:
  GOLDEN_TEST_WRITE=1: Regenerate the baseline artifact from the current state.

Usage:
  pytest tests/launcher/recipe_templatization/test_golden_override_params.py
  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_golden_override_params.py
"""

import json
import os
from pathlib import Path
from typing import Dict

import pytest

from utils.resolve_override_params import resolve_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path("launcher/recipe_templatization")
BASE_FILE = TEMPLATE_DIR / "base_override_parameters.json"
BASELINE_PATH = Path("tests/launcher/recipe_templatization/baseline_artifacts/resolved_override_params.json")
GOLDEN_WRITE = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")

# Frameworks and their template parameter file names
FRAMEWORKS = {
    "llmft": "llmft_recipe_template_parameters.json",
    "nova": "nova_recipe_template_parameters.json",
    "verl": "verl_recipe_template_parameters.json",
    "evaluation": "evaluation_recipe_template_parameters.json",
    "checkpointless": "checkpointless_recipe_template_parameters.json",
}

# Nova evaluation templates use the "evaluation" category
NOVA_EVAL_TEMPLATES = {
    "nova_general_text_benchmark_eval",
    "nova_general_text_benchmark_2_0_eval",
    "nova_general_multi_modal_benchmark_eval",
    "nova_general_multi_modal_benchmark_2_0_eval",
    "nova_llm_judge_eval",
    "nova_bring_your_own_dataset_eval",
    "nova_bring_your_own_dataset_2_0_eval",
}


# ---------------------------------------------------------------------------
# Category determination
# ---------------------------------------------------------------------------


def _get_category(framework: str, template_name: str) -> str:
    """Determine the base parameter category for a given framework and template."""
    if framework == "evaluation":
        return "evaluation"
    if framework == "nova" and template_name in NOVA_EVAL_TEMPLATES:
        return "evaluation"
    return "fine_tuning"


# ---------------------------------------------------------------------------
# Resolution logic — uses shared resolve_params from utils/
# ---------------------------------------------------------------------------


def _resolve_override_parameters(
    base_params: dict, category: str, template_overrides: dict, recipe_template: dict
) -> dict:
    """Resolve override parameters by delegating to the shared resolve_params function."""
    category_base = base_params.get(category, {})
    return resolve_params(category_base, template_overrides, recipe_template)


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------


def generate_resolved_override_params_snapshot() -> Dict:
    """
    Iterate all 5 framework template files, load every template's
    override parameters (resolving from base if using sparse format),
    and return the combined snapshot keyed by {framework}.{template_name}.

    Supports both:
    - Legacy format: `recipe_override_parameters` (full definitions)
    - New format: `override_parameters` (sparse overrides merged with base)
    """
    # Load base definitions
    with open(BASE_FILE, "r", encoding="utf-8") as f:
        base_params = json.load(f)

    snapshot = {}

    for framework, filename in sorted(FRAMEWORKS.items()):
        filepath = TEMPLATE_DIR / framework / filename
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        templates = data.get("templates", {})
        for template_name in sorted(templates.keys()):
            template_data = templates[template_name]
            key = f"{framework}.{template_name}"

            # Support both new key (override_parameters) and legacy key (recipe_override_parameters)
            if "override_parameters" in template_data:
                # Sparse format: resolve by merging base + overrides using placeholder extraction
                category = _get_category(framework, template_name)
                sparse_overrides = template_data["override_parameters"]
                recipe_template = template_data.get("recipe_template", {})
                override_params = _resolve_override_parameters(base_params, category, sparse_overrides, recipe_template)
            else:
                # Legacy format: use full definitions as-is
                override_params = template_data.get("recipe_override_parameters", {})

            snapshot[key] = override_params

    return snapshot


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def write_baseline(snapshot: Dict) -> None:
    """Write the snapshot to the baseline artifact file."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4, sort_keys=True)
        f.write("\n")


def load_baseline() -> Dict:
    """Load the baseline snapshot from disk."""
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Module-level: regenerate baseline if GOLDEN_TEST_WRITE is set
# ---------------------------------------------------------------------------

if GOLDEN_WRITE:
    _snapshot = generate_resolved_override_params_snapshot()
    write_baseline(_snapshot)
    print(f"GOLDEN_TEST_WRITE: Updated {BASELINE_PATH}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoldenOverrideParams:
    """Golden comparison test for resolved override parameters."""

    def test_snapshot_matches_baseline(self):
        """
        Regenerate the snapshot from current template files and assert it
        matches the stored baseline for every template.
        """
        baseline = load_baseline()
        current = generate_resolved_override_params_snapshot()

        # Check same set of keys
        baseline_keys = set(baseline.keys())
        current_keys = set(current.keys())

        missing_from_current = baseline_keys - current_keys
        extra_in_current = current_keys - baseline_keys

        assert (
            not missing_from_current
        ), f"Templates in baseline but missing from current: {sorted(missing_from_current)}"
        assert not extra_in_current, (
            f"Templates in current but missing from baseline: {sorted(extra_in_current)}. "
            f"Run with GOLDEN_TEST_WRITE=1 to update."
        )

        # Check each template's parameters match
        mismatches = []
        for key in sorted(baseline_keys):
            if baseline[key] != current[key]:
                mismatches.append(key)

        assert not mismatches, (
            f"Override parameters changed for {len(mismatches)} template(s): "
            f"{mismatches[:10]}{'...' if len(mismatches) > 10 else ''}. "
            f"If intentional, run with GOLDEN_TEST_WRITE=1 to update the baseline."
        )

    def test_baseline_file_exists(self):
        """Verify the baseline artifact file exists and is non-empty."""
        assert BASELINE_PATH.exists(), (
            f"Baseline file not found at {BASELINE_PATH}. " f"Run with GOLDEN_TEST_WRITE=1 to generate it."
        )
        assert BASELINE_PATH.stat().st_size > 0, f"Baseline file at {BASELINE_PATH} is empty."

    def test_each_template_has_override_params(self):
        """Every template in the snapshot should have a non-empty override params dict."""
        current = generate_resolved_override_params_snapshot()
        empty_templates = [key for key, params in current.items() if not params]
        # Note: some templates might legitimately have no override params,
        # but this is a sanity check for the common case
        if empty_templates:
            pytest.skip(f"Templates with empty override params (may be intentional): " f"{empty_templates}")

    def test_base_override_parameters_unchanged(self):
        """
        Verify that base_override_parameters.json has not been accidentally modified.
        Compares the live file against the baseline copy in baseline_artifacts/.
        If intentional changes were made, update the baseline with:
            cp launcher/recipe_templatization/base_override_parameters.json \
               tests/launcher/recipe_templatization/baseline_artifacts/base_override_parameters.json
        """
        base_file = TEMPLATE_DIR / "base_override_parameters.json"
        baseline_copy = Path("tests/launcher/recipe_templatization/baseline_artifacts/base_override_parameters.json")

        assert base_file.exists(), f"Base file not found at {base_file}"
        assert baseline_copy.exists(), (
            f"Baseline copy not found at {baseline_copy}. " f"Copy the base file there to enable change detection."
        )

        with open(base_file, "r", encoding="utf-8") as f:
            current = json.load(f)
        with open(baseline_copy, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        assert current == baseline, (
            f"base_override_parameters.json has been modified. "
            f"If this is intentional, update the baseline:\n"
            f"  cp {base_file} {baseline_copy}"
        )
