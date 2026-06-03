#!/usr/bin/env python3
"""
Check inference (hosting) coverage for fine-tuning recipes touched in a PR.

Given a list of changed recipe files (from git diff), checks whether each one
has a corresponding hosting configuration file in
``utils/inference_configs/hosting-<recipe_stem>.json``.

The hosting config filename is derived directly from the recipe filename:
    Recipe:  ``recipes_collection/recipes/fine-tuning/qwen/llmft_qwen3_4b_seq4k_gpu_dpo.yaml``
    Config:  ``utils/inference_configs/hosting-llmft_qwen3_4b_seq4k_gpu_dpo.json``

Only recipes that are both (a) touched in the PR and (b) missing a hosting
config will be reported — this avoids generating noise for recipe
modifications that don't require inference-side changes.

Usage:
    # Check specific changed files (PR mode)
    python .github/scripts/check_inference_coverage.py \\
        --changed-files "recipes_collection/recipes/fine-tuning/llama/sft.yaml
    recipes_collection/recipes/fine-tuning/qwen/dpo.yaml"

    # Check all recipes (fallback / local testing)
    python .github/scripts/check_inference_coverage.py --all

Outputs (via GITHUB_OUTPUT):
    has_missing: "true" or "false"
    missing_recipes: JSON array of {recipe_path, run_name, js_model_id, reason} — one per recipe
    missing_count: number of recipes missing hosting configuration
    has_unmapped_models: "true" or "false"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Reuse shared helpers from eval coverage checker
from check_eval_coverage import (
    _set_github_output,
    extract_run_name,
    filter_recipe_files,
    find_recipes,
    load_model_id_map,
)

logger = logging.getLogger(__name__)

# Directory containing hosting configuration files
DEFAULT_INFERENCE_CONFIGS_DIR = "utils/inference_configs"


def get_hosting_config_path(recipe_path: str, inference_configs_dir: str = DEFAULT_INFERENCE_CONFIGS_DIR) -> str:
    """Derive the expected hosting config file path from a recipe path.

    The convention is ``hosting-<recipe_stem>.json`` where *recipe_stem*
    is the recipe filename without its ``.yaml`` extension.

    Args:
        recipe_path: Full path to the recipe YAML file.
        inference_configs_dir: Directory containing hosting config JSON files.

    Returns:
        Expected path to the hosting configuration file.

    Examples:
        >>> get_hosting_config_path("recipes_collection/recipes/fine-tuning/qwen/llmft_qwen3_4b_seq4k_gpu_dpo.yaml")
        'utils/inference_configs/hosting-llmft_qwen3_4b_seq4k_gpu_dpo.json'
    """
    stem = os.path.splitext(os.path.basename(recipe_path))[0]
    return os.path.join(inference_configs_dir, f"hosting-{stem}.json")


def load_hosting_config_stems(inference_configs_dir: str) -> set[str]:
    """Load the set of recipe stems that have hosting configs.

    Scans ``inference_configs_dir`` for files matching ``hosting-*.json``
    and returns the set of stems (the part between ``hosting-`` and ``.json``).

    Args:
        inference_configs_dir: Directory containing hosting config JSON files.

    Returns:
        Set of stems (e.g. ``{"llmft_qwen3_4b_seq4k_gpu_dpo", ...}``).
    """
    stems: set[str] = set()
    if not os.path.isdir(inference_configs_dir):
        return stems
    for fname in os.listdir(inference_configs_dir):
        if fname.startswith("hosting-") and fname.endswith(".json"):
            stem = fname[len("hosting-") : -len(".json")]
            stems.add(stem)
    return stems


def check_inference_coverage_for_recipes(
    recipe_paths: list[str],
    model_id_map_path: str,
    inference_configs_dir: str = DEFAULT_INFERENCE_CONFIGS_DIR,
) -> list[dict[str, str]]:
    """
    Check a list of recipe files for inference (hosting) coverage.

    For each recipe:
    1. Verify ``run.name`` exists and maps to a JumpStart model ID.
    2. Check that ``utils/inference_configs/hosting-<stem>.json`` exists.

    Args:
        recipe_paths: List of recipe file paths to check.
        model_id_map_path: Path to ``jumpstart_model-id_map.json``.
        inference_configs_dir: Directory containing hosting config JSON files.

    Returns:
        A list of dicts for recipes missing hosting configs::

            {recipe_path, run_name, js_model_id, reason}
    """
    model_id_map = load_model_id_map(model_id_map_path)
    hosting_stems = load_hosting_config_stems(inference_configs_dir)

    missing = []
    for recipe_path in recipe_paths:
        run_name = extract_run_name(recipe_path)
        if run_name is None:
            # No run.name field — skip (may be a config/cluster file)
            continue

        js_model_id = model_id_map.get(run_name)
        if js_model_id is None:
            missing.append(
                {
                    "recipe_path": recipe_path,
                    "run_name": run_name,
                    "js_model_id": None,
                    "reason": "run.name not found in jumpstart_model-id_map.json",
                }
            )
            continue

        # Derive the expected hosting config stem from the recipe filename
        stem = os.path.splitext(os.path.basename(recipe_path))[0]
        if stem not in hosting_stems:
            missing.append(
                {
                    "recipe_path": recipe_path,
                    "run_name": run_name,
                    "js_model_id": js_model_id,
                    "reason": f"Hosting configuration not found: hosting-{stem}.json",
                }
            )

    return missing


def check_inference_coverage(
    recipes_dir: str,
    model_id_map_path: str,
    inference_configs_dir: str = DEFAULT_INFERENCE_CONFIGS_DIR,
) -> list[dict[str, str]]:
    """
    Check all recipes in a directory for inference (hosting) coverage.
    Convenience wrapper that finds all recipes then checks them.
    """
    recipes = find_recipes(recipes_dir)
    return check_inference_coverage_for_recipes(recipes, model_id_map_path, inference_configs_dir)


def main():
    parser = argparse.ArgumentParser(description="Check inference hosting coverage for fine-tuning recipes")
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Newline-separated list of changed recipe file paths from git diff. "
        "Only these files will be checked for inference coverage.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all recipes in --recipes-dir (ignores --changed-files)",
    )
    parser.add_argument(
        "--recipes-dir",
        default="recipes_collection/recipes/fine-tuning",
        help="Path to fine-tuning recipes directory",
    )
    parser.add_argument(
        "--model-id-map",
        default="launcher/recipe_templatization/jumpstart_model-id_map.json",
        help="Path to jumpstart_model-id_map.json",
    )
    parser.add_argument(
        "--inference-configs-dir",
        default=DEFAULT_INFERENCE_CONFIGS_DIR,
        help="Path to directory containing hosting config JSON files",
    )
    args = parser.parse_args()

    print("=== Checking Inference (Hosting) Coverage ===")
    print(f"Model ID map: {args.model_id_map}")
    print(f"Inference configs dir: {args.inference_configs_dir}")

    if args.all or args.changed_files is None:
        # Scan all recipes in directory
        print(f"Mode: scanning all recipes in {args.recipes_dir}")
        print()
        missing = check_inference_coverage(args.recipes_dir, args.model_id_map, args.inference_configs_dir)
    else:
        # Only check the changed files
        changed = [f for f in args.changed_files.split("\n") if f.strip()]
        recipe_files = filter_recipe_files(changed, args.recipes_dir)
        print(f"Mode: checking {len(recipe_files)} changed recipe file(s)")
        for rf in recipe_files:
            print(f"  - {rf}")
        print()

        if not recipe_files:
            print("✓ No eligible recipe files in changed files. Nothing to check.")
            _set_github_output("has_missing", "false")
            _set_github_output("missing_count", "0")
            _set_github_output("missing_recipes", "[]")
            _set_github_output("has_unmapped_models", "false")
            return 0

        missing = check_inference_coverage_for_recipes(recipe_files, args.model_id_map, args.inference_configs_dir)

    has_missing = len(missing) > 0

    if has_missing:
        print(f"⚠ Found {len(missing)} recipe(s) missing hosting configuration:\n")
        for entry in missing:
            print(f"  Recipe: {entry['recipe_path']}")
            print(f"  run.name: {entry['run_name']}")
            print(f"  JumpStart ID: {entry['js_model_id'] or 'NOT MAPPED'}")
            print(f"  Reason: {entry['reason']}")
            print()
    else:
        print("✓ All checked recipes have hosting configuration.")

    # Check if any recipes have a run.name not in jumpstart_model-id_map.json
    has_unmapped = any(entry["js_model_id"] is None for entry in missing)

    _set_github_output("has_missing", str(has_missing).lower())
    _set_github_output("missing_count", str(len(missing)))
    _set_github_output("missing_recipes", json.dumps(missing, indent=2))
    _set_github_output("has_unmapped_models", str(has_unmapped).lower())

    return 0


if __name__ == "__main__":
    sys.exit(main())
