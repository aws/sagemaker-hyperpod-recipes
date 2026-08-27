"""
Golden test for the recipes excluded from JumpStart publishing.

`recipes_collection/jumpstart_exclusions.yaml` holds regexes that are matched
(via `re.search`) against recipe ids — the path of a recipe under
`recipes_collection/recipes/` without the `.yaml` suffix, e.g.
`fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_lora`. Any recipe
matching a pattern is skipped by `scripts/generate_launch_jsons.py` and by the
JumpStart publish Lambda.

A regex change (or a new recipe landing under an existing pattern) silently
changes which recipes reach customers. This test snapshots the resolved
exclusion set so that change shows up as a reviewable diff:

  tests/launcher/recipe_templatization/baseline_artifacts/jumpstart_excluded_recipes.json

Environment variables:
  GOLDEN_TEST_WRITE=1: Regenerate the baseline artifact from the current state.

Usage:
  pytest tests/launcher/recipe_templatization/test_jumpstart_exclusions.py
  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_jumpstart_exclusions.py
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

GOLDEN_TEST_WRITE = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES_DIR = REPO_ROOT / "recipes_collection" / "recipes"
EXCLUSIONS_PATH = REPO_ROOT / "recipes_collection" / "jumpstart_exclusions.yaml"
BASELINE_PATH = (
    REPO_ROOT
    / "tests"
    / "launcher"
    / "recipe_templatization"
    / "baseline_artifacts"
    / "jumpstart_excluded_recipes.json"
)

# Directories under recipes_collection/recipes that hold Hydra fragments or
# build noise rather than standalone recipes (mirrors hyperpod_recipes.list_recipes).
SKIP_DIRS = {"__pycache__", "hydra_config"}

BASELINE_COMMENT = [
    "GENERATED FILE - do not hand-edit.",
    "Snapshot of every recipe excluded from JumpStart publishing by the regexes in",
    "recipes_collection/jumpstart_exclusions.yaml. Recipe ids are paths relative to",
    "recipes_collection/recipes/ with the .yaml suffix stripped, matched with re.search.",
    "matches_by_pattern is keyed by the raw pattern, in the order the YAML file lists them;",
    "an empty list means the pattern currently matches no recipe.",
    "Regenerate: GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_jumpstart_exclusions.py",
]


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------


def load_exclusion_patterns() -> List[str]:
    """Return the raw exclusion patterns, in the order the YAML file lists them."""
    with open(EXCLUSIONS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("exclusion_patterns", []) or [])


def list_recipe_ids() -> List[str]:
    """Return every recipe id under recipes_collection/recipes/, sorted."""
    ids = []
    for path in RECIPES_DIR.rglob("*.yaml"):
        if SKIP_DIRS.intersection(path.relative_to(RECIPES_DIR).parts):
            continue
        ids.append(path.relative_to(RECIPES_DIR).with_suffix("").as_posix())
    return sorted(ids)


def generate_snapshot() -> Dict:
    """Match every recipe id against every exclusion pattern."""
    patterns = load_exclusion_patterns()
    recipe_ids = list_recipe_ids()

    matches_by_pattern = {}
    excluded = set()
    for pattern in patterns:
        compiled = re.compile(pattern)
        matched = [rid for rid in recipe_ids if compiled.search(rid)]
        matches_by_pattern[pattern] = matched
        excluded.update(matched)

    return {
        "_comment": BASELINE_COMMENT,
        "excluded_recipes": sorted(excluded),
        "matches_by_pattern": matches_by_pattern,
    }


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def write_baseline(snapshot: Dict) -> None:
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=4)
        f.write("\n")


def load_baseline() -> Dict:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Module-level: regenerate baseline if GOLDEN_TEST_WRITE is set
# ---------------------------------------------------------------------------

if GOLDEN_TEST_WRITE:
    write_baseline(generate_snapshot())
    print(f"GOLDEN_TEST_WRITE: Updated {BASELINE_PATH}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJumpStartExclusions:
    """Golden comparison for the JumpStart exclusion set."""

    def test_baseline_file_exists(self):
        assert BASELINE_PATH.exists(), (
            f"Baseline file not found at {BASELINE_PATH}. " f"Run with GOLDEN_TEST_WRITE=1 to generate it."
        )
        assert BASELINE_PATH.stat().st_size > 0, f"Baseline file at {BASELINE_PATH} is empty."

    def test_patterns_are_valid_regexes(self):
        invalid = []
        for pattern in load_exclusion_patterns():
            try:
                re.compile(pattern)
            except re.error as exc:
                invalid.append(f"{pattern!r}: {exc}")
        assert not invalid, "Invalid regex(es) in jumpstart_exclusions.yaml:\n  " + "\n  ".join(invalid)

    def test_excluded_recipes_match_baseline(self):
        """Every recipe matching an exclusion pattern must be listed in the baseline."""
        baseline = set(load_baseline()["excluded_recipes"])
        current = set(generate_snapshot()["excluded_recipes"])

        newly_excluded = sorted(current - baseline)
        no_longer_excluded = sorted(baseline - current)

        if newly_excluded or no_longer_excluded:
            lines = ["", "=" * 70, "JumpStart exclusion set DIFFERS from baseline", "=" * 70]
            if newly_excluded:
                lines.append(
                    f"\nNewly excluded — matched a pattern but missing from the baseline ({len(newly_excluded)}):"
                )
                lines.extend(f"  + {rid}" for rid in newly_excluded)
            if no_longer_excluded:
                lines.append(
                    f"\nNo longer excluded — in the baseline but matched by no pattern ({len(no_longer_excluded)}):"
                )
                lines.extend(f"  - {rid}" for rid in no_longer_excluded)
            lines.append(
                "\nIf intentional, regenerate the baseline:\n"
                "  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_jumpstart_exclusions.py"
            )
            lines.append("=" * 70)
            pytest.fail("\n".join(lines))

    def test_matches_by_pattern_match_baseline(self):
        """Catch per-pattern drift that leaves the overall exclusion set unchanged."""
        baseline = load_baseline()["matches_by_pattern"]
        current = generate_snapshot()["matches_by_pattern"]

        added_patterns = sorted(set(current) - set(baseline))
        removed_patterns = sorted(set(baseline) - set(current))
        changed_patterns = sorted(p for p in set(baseline) & set(current) if baseline[p] != current[p])

        if added_patterns or removed_patterns or changed_patterns:
            lines = ["", "=" * 70, "Per-pattern exclusion matches DIFFER from baseline", "=" * 70]
            for pattern in added_patterns:
                lines.append(f"\nPattern added ({len(current[pattern])} matches): {pattern!r}")
                lines.extend(f"  + {rid}" for rid in current[pattern])
            for pattern in removed_patterns:
                lines.append(f"\nPattern removed (had {len(baseline[pattern])} matches): {pattern!r}")
            for pattern in changed_patterns:
                lines.append(f"\nPattern matches changed: {pattern!r}")
                lines.extend(f"  + {rid}" for rid in sorted(set(current[pattern]) - set(baseline[pattern])))
                lines.extend(f"  - {rid}" for rid in sorted(set(baseline[pattern]) - set(current[pattern])))
            lines.append(
                "\nIf intentional, regenerate the baseline:\n"
                "  GOLDEN_TEST_WRITE=1 pytest tests/launcher/recipe_templatization/test_jumpstart_exclusions.py"
            )
            lines.append("=" * 70)
            pytest.fail("\n".join(lines))
