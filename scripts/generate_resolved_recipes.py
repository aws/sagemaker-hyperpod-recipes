#!/usr/bin/env python3
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
Generator for self-contained recipe YAMLs in recipes_collection/recipes/.

This script resolves Hydra composition for recipes in hyperpod_recipes/recipes_src/
and generates fully-resolved YAML files without defaults or interpolations.

Usage:
    python scripts/generate_resolved_recipes.py
    python scripts/generate_resolved_recipes.py --check
    python scripts/generate_resolved_recipes.py --check --diff
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

# Add parent directory to path to import hyperpod_recipes
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from hyperpod_recipes import list_recipes
from hyperpod_recipes.recipe import RECIPES_DIR, Recipe

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "recipes_collection" / "recipes"

# Recipes under these paths are self-contained YAMLs without Hydra composition.
# They are directly copied to preserve comments.
DIRECT_COPY_PATHS = [
    Path(RECIPES_DIR) / "training" / "nova",
    Path(RECIPES_DIR) / "fine-tuning" / "nova",
    Path(RECIPES_DIR) / "evaluation" / "nova",
]


class ResolvedRecipeError(Exception):
    """Custom exception for resolved recipe errors."""


def list_resolved_files() -> set[Path]:
    resolved = set()
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() or item.suffix != ".yaml":
            continue
        resolved.add(item)
    return resolved


def _is_direct_copy_path(recipe_path: str) -> bool:
    return any(recipe_path.startswith(str(p)) for p in DIRECT_COPY_PATHS)


def _resolve_one(recipe_path: str) -> tuple[str, str]:
    """Resolve a single recipe to (name, yaml_content). Runs in a worker process."""
    recipe = Recipe(recipe_path)
    if _is_direct_copy_path(recipe_path):
        content = Path(recipe_path).read_text()
    else:
        content = OmegaConf.to_yaml(recipe.config, sort_keys=True)
    return recipe.name, content


def generate_resolved_recipes(write=False):
    """Generate all resolved recipe YAMLs."""
    recipes = list_recipes()
    recipe_paths = [r.path for r in recipes]

    if write:
        print(f"Generating {len(recipes)} resolved recipe YAMLs...")
    else:
        print(f"Checking {len(recipes)} resolved recipe YAMLs...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    max_workers = min(len(recipe_paths), os.cpu_count() or 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_resolve_one, recipe_paths))
    elapsed = time.perf_counter() - t0
    print(f"Resolved {len(results)} recipes in {elapsed:.1f}s ({max_workers} workers)")

    checked_files = set()

    for name, expected_content in results:
        output_path = OUTPUT_DIR / f"{name}.yaml"
        checked_files.add(output_path)

        if write:
            print(f"Writing recipe {output_path}")
            output_path.write_text(expected_content)
        else:
            print(f"Checking recipe {output_path}")
            if output_path.exists():
                actual_content = output_path.read_text()
                if actual_content != expected_content:
                    import difflib

                    diff = difflib.unified_diff(
                        actual_content.splitlines(keepends=True),
                        expected_content.splitlines(keepends=True),
                        fromfile=f"{output_path} (on disk)",
                        tofile=f"{output_path} (expected from source)",
                    )
                    diff_str = "".join(diff)
                    raise ResolvedRecipeError(
                        f"Mismatch in resolved recipe: {output_path}\n"
                        f"Run `python scripts/generate_resolved_recipes.py` (without --check) to regenerate.\n"
                        f"Diff:\n{diff_str}"
                    )
            else:
                raise ResolvedRecipeError(
                    f"Missing resolved recipe file: {output_path}\n"
                    f"Run `python scripts/generate_resolved_recipes.py` (without --check) to regenerate."
                )

    for to_delete in list_resolved_files().difference(checked_files):
        if write:
            print(f"Removing deleted recipe {to_delete}")
            to_delete.unlink()
        else:
            raise ResolvedRecipeError(
                f"Unexpected resolved recipe file (no longer has a source): {to_delete}\n"
                "Run `python scripts/generate_resolved_recipes.py` (without --check) to regenerate."
            )

    if write:
        print("Generation successful")
    else:
        print("Check passed")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate self-contained recipe YAMLs from Hydra compositions")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated YAMLs match disk (exit 1 if mismatches)",
    )
    args = parser.parse_args()

    generate_resolved_recipes(write=not args.check)


if __name__ == "__main__":
    main()
