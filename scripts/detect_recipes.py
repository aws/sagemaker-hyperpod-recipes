"""Detect new and functionally changed recipes in a PR.

Compares recipes in the PR branch against the base branch.
Filters out non-functional changes (dataset paths, model paths, etc.).
"""

import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.recipe_diff import is_functional_change

RECIPES_PREFIX = "recipes_collection/recipes"


def _get_file_at_ref(path, ref):
    """Get file content at a git ref. Returns None if file doesn't exist."""
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def detect_recipes(base_branch="main", include_modified=True):
    """Detect new and functionally changed recipes vs base branch.

    Args:
        base_branch: Branch to compare against.
        include_modified: If True, include modified recipes with functional changes.
            If False, only return newly added recipes.

    Returns list of recipe paths (relative to repo root) that are either:
    - Newly added (not in base branch)
    - Modified with functional changes (not just dataset/model paths) — only if include_modified=True
    """
    diff_filter = "AM" if include_modified else "A"
    # Get recipe files matching the filter
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD", "--name-only", f"--diff-filter={diff_filter}", "--", RECIPES_PREFIX],
        capture_output=True,
        text=True,
        check=True,
    )
    changed_files = [f for f in result.stdout.strip().split("\n") if f and f.endswith(".yaml")]

    if not changed_files:
        return []

    functional = []
    for recipe_path in changed_files:
        old_content = _get_file_at_ref(recipe_path, base_branch)

        # New file — always include
        if old_content is None:
            functional.append(recipe_path)
            continue

        # Modified — check if functional
        new_content = _get_file_at_ref(recipe_path, "HEAD")
        if new_content is None:
            functional.append(recipe_path)
            continue

        try:
            old_yaml = yaml.safe_load(old_content)
            new_yaml = yaml.safe_load(new_content)
        except Exception:
            functional.append(recipe_path)
            continue

        if is_functional_change(old_yaml, new_yaml):
            functional.append(recipe_path)

    return functional


if __name__ == "__main__":  # pragma: no cover
    import os

    base = os.environ.get("BASE_BRANCH", "main")
    recipes = detect_recipes(base)
    print(f"Detected {len(recipes)} recipe(s) with functional changes:")
    for r in recipes:
        print(f"  {r}")
