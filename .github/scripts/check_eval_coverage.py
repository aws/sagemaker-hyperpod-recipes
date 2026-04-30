#!/usr/bin/env python3
"""
Check eval coverage for fine-tuning recipes touched in a PR.

Given a list of changed recipe files (from git diff), checks whether each one
has eval instance mapping data in
launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json.

Only recipes that are both (a) touched in the PR and (b) missing eval data
will be reported — this avoids generating noise for recipe modifications
that don't require eval-side changes.

Usage:
    # Check specific changed files (PR mode)
    python .github/scripts/check_eval_coverage.py \
        --changed-files "recipes_collection/recipes/fine-tuning/llama/sft.yaml
    recipes_collection/recipes/fine-tuning/qwen/dpo.yaml"

    # Check all recipes (fallback / local testing)
    python .github/scripts/check_eval_coverage.py --all

Outputs (via GITHUB_OUTPUT):
    has_missing: "true" or "false"
    missing_recipes: JSON array of {recipe_path, run_name, js_model_id, reason}
    missing_count: integer count
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)

# Directories to exclude from eval coverage checks.
# Nova recipes use generic run.name values (e.g., "my-lora-run") that don't
# map to JumpStart model IDs. Training/evaluation recipes are not fine-tuning.
EXCLUDED_SUBDIRS = {"nova", "training", "evaluation"}


def load_model_id_map(path: str) -> dict[str, str]:
    """Load the jumpstart_model-id_map.json mapping run.name -> JumpStart model ID."""
    with open(path, "r") as f:
        return json.load(f)


def load_eval_instance_mapping(path: str) -> set[str]:
    """Load the set of JumpStart model IDs that have eval instance mapping."""
    with open(path, "r") as f:
        data = json.load(f)
    return set(data.get("js_model_name_instance_mapping", {}).keys())


def find_recipes(recipes_dir: str) -> list[str]:
    """
    Find all recipe YAML files under the recipes directory,
    excluding subdirectories in EXCLUDED_SUBDIRS.
    """
    all_yamls = sorted(glob.glob(os.path.join(recipes_dir, "**", "*.yaml"), recursive=True))
    recipes = []
    for path in all_yamls:
        # Get the first subdirectory under recipes_dir
        rel = os.path.relpath(path, recipes_dir)
        top_dir = rel.split(os.sep)[0] if os.sep in rel else ""
        if top_dir not in EXCLUDED_SUBDIRS:
            recipes.append(path)
    return recipes


def filter_recipe_files(changed_files: list[str], recipes_dir: str) -> list[str]:
    """
    Filter a list of changed file paths to only include recipe YAML files
    under the recipes directory, excluding EXCLUDED_SUBDIRS.

    Args:
        changed_files: List of file paths from git diff
        recipes_dir: Base recipes directory (e.g., recipes_collection/recipes/fine-tuning)

    Returns:
        Filtered list of recipe file paths that exist and are eligible for eval checking
    """
    filtered = []
    for filepath in changed_files:
        filepath = filepath.strip()
        if not filepath or not filepath.endswith(".yaml"):
            continue

        # Must be under the recipes_dir
        try:
            rel = os.path.relpath(filepath, recipes_dir)
        except ValueError:
            continue

        # Skip if it's not actually under recipes_dir (relpath goes up with ..)
        if rel.startswith(".."):
            continue

        # Check excluded subdirectories
        top_dir = rel.split(os.sep)[0] if os.sep in rel else ""
        if top_dir in EXCLUDED_SUBDIRS:
            continue

        # Only include files that exist (new files in PR may not exist on disk in some CI setups)
        if os.path.isfile(filepath):
            filtered.append(filepath)

    return sorted(filtered)


def extract_run_name(recipe_path: str) -> str | None:
    """Extract run.name from a recipe YAML file. Returns None if not found."""
    try:
        with open(recipe_path, "r") as f:
            data = yaml.safe_load(f)
        if data and isinstance(data, dict):
            return data.get("run", {}).get("name")
    except Exception:
        logger.debug("Failed to extract run.name from %s", recipe_path, exc_info=True)
    return None


def check_eval_coverage_for_recipes(
    recipe_paths: list[str],
    model_id_map_path: str,
    eval_params_path: str,
) -> list[dict[str, str]]:
    """
    Check a list of recipe files for eval coverage.

    Args:
        recipe_paths: List of recipe file paths to check
        model_id_map_path: Path to jumpstart_model-id_map.json
        eval_params_path: Path to evaluation_regional_parameters.json

    Returns a list of dicts for recipes missing eval data:
        {recipe_path, run_name, js_model_id, reason}
    """
    model_id_map = load_model_id_map(model_id_map_path)
    eval_models = load_eval_instance_mapping(eval_params_path)

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
        elif js_model_id not in eval_models:
            missing.append(
                {
                    "recipe_path": recipe_path,
                    "run_name": run_name,
                    "js_model_id": js_model_id,
                    "reason": "JumpStart model ID not in eval js_model_name_instance_mapping",
                }
            )

    return missing


def check_eval_coverage(
    recipes_dir: str,
    model_id_map_path: str,
    eval_params_path: str,
) -> list[dict[str, str]]:
    """
    Check all recipes in a directory for eval coverage.
    Convenience wrapper that finds all recipes then checks them.
    """
    recipes = find_recipes(recipes_dir)
    return check_eval_coverage_for_recipes(recipes, model_id_map_path, eval_params_path)


def deduplicate_by_model(missing: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Deduplicate missing entries so we report each missing *model* only once,
    with all affected recipe paths and run names listed.

    The key is (js_model_id, reason) when js_model_id is known, which
    correctly groups different recipe run names (e.g. LLMFT vs VERL variants)
    that map to the same underlying JumpStart model.  When js_model_id is
    None (model not yet in jumpstart_model-id_map.json), we fall back to
    (run_name, reason) since there is no shared model ID to group on.
    """
    seen: dict[tuple[str | None, str], dict] = {}
    for entry in missing:
        js_id = entry["js_model_id"]
        if js_id is not None:
            key = (js_id, entry["reason"])
        else:
            key = (entry["run_name"], entry["reason"])
        if key not in seen:
            seen[key] = {
                "run_names": [],
                "js_model_id": js_id,
                "reason": entry["reason"],
                "recipe_paths": [],
            }
        if entry["run_name"] not in seen[key]["run_names"]:
            seen[key]["run_names"].append(entry["run_name"])
        seen[key]["recipe_paths"].append(entry["recipe_path"])
    return list(seen.values())


def _set_github_output(name: str, value: str) -> None:
    """Set a GitHub Actions output variable."""
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            # Use heredoc syntax for multiline values
            if "\n" in value:
                f.write(f"{name}<<GHEOF\n{value}\nGHEOF\n")
            else:
                f.write(f"{name}={value}\n")


def main():
    parser = argparse.ArgumentParser(description="Check eval coverage for fine-tuning recipes")
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Newline-separated list of changed recipe file paths from git diff. "
        "Only these files will be checked for eval coverage.",
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
        "--eval-params",
        default="launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json",
        help="Path to evaluation_regional_parameters.json",
    )
    args = parser.parse_args()

    print("=== Checking Eval Coverage ===")
    print(f"Model ID map: {args.model_id_map}")
    print(f"Eval params: {args.eval_params}")

    if args.all or args.changed_files is None:
        # Scan all recipes in directory
        print(f"Mode: scanning all recipes in {args.recipes_dir}")
        print()
        missing = check_eval_coverage(args.recipes_dir, args.model_id_map, args.eval_params)
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
            return 0

        missing = check_eval_coverage_for_recipes(recipe_files, args.model_id_map, args.eval_params)

    deduped = deduplicate_by_model(missing)
    has_missing = len(deduped) > 0

    if has_missing:
        print(f"⚠ Found {len(deduped)} model(s) missing eval instance mapping ({len(missing)} recipe files):\n")
        for entry in deduped:
            run_names = entry["run_names"]
            print(f"  Model: {run_names[0]}" if len(run_names) == 1 else f"  Models: {', '.join(run_names)}")
            print(f"  JumpStart ID: {entry['js_model_id'] or 'NOT MAPPED'}")
            print(f"  Reason: {entry['reason']}")
            print(f"  Recipes ({len(entry['recipe_paths'])}):")
            for rp in entry["recipe_paths"]:
                print(f"    - {rp}")
            print()
    else:
        print("✓ All checked recipes have eval instance mapping.")

    # Check if any models are unmapped (missing from jumpstart_model-id_map.json)
    has_unmapped = any(entry["js_model_id"] is None for entry in deduped)

    # Set GitHub outputs
    _set_github_output("has_missing", str(has_missing).lower())
    _set_github_output("missing_count", str(len(deduped)))
    _set_github_output("missing_recipes", json.dumps(deduped, indent=2))
    _set_github_output("has_unmapped_models", str(has_unmapped).lower())

    return 0


if __name__ == "__main__":
    sys.exit(main())
