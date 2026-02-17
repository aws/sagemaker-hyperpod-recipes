#!/usr/bin/env python3
"""
Helper functions for EKS recipe validation workflow
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

# platform-instance constraints
PLATFORM_INSTANCE_CONSTRAINTS = {
    "K8": ["ml.p5.48xlarge"],
    "SLURM": ["ml.p5.48xlarge"],
    "SMJOBS": ["ml.p4d.24xlarge", "ml.p4de.24xlarge", "ml.p5.48xlarge", "ml.g5.48xlarge"],
}

PLATFORMS = ["K8", "SLURM", "SMJOBS"]
RECIPES_PREFIX = "recipes_collection/recipes/"


def detect_modified_recipes(base_branch="main", recipes_dir="recipes_collection/recipes/"):
    """
    Detect modified recipe files compared to base branch

    Returns:
        list: List of modified .yaml recipe files, or empty list if none
    """

    try:
        # Fetch base branch
        subprocess.run(["git", "fetch", "origin", f"{base_branch}:{base_branch}"], check=True, capture_output=True)

        # Find modified/added .yaml files
        result = subprocess.run(
            ["git", "diff", f"{base_branch}...HEAD", "--name-only", "--diff-filter=AM", "--", recipes_dir],
            capture_output=True,
            text=True,
            check=True,
        )

        # Filter for .yaml files
        modified_files = [f for f in result.stdout.strip().split("\n") if f and f.endswith(".yaml")]

        # Write to GITHUB_OUTPUT
        github_output = os.getenv("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                if not modified_files:
                    print("No modified recipe files detected")
                    f.write("has_changes=false\n")
                else:
                    print(f"Found {len(modified_files)} modified recipes:")
                    for file in modified_files:
                        print(f"  - {file}")

                    f.write("has_changes=true\n")
                    f.write("changed_files<<EOF\n")
                    f.write("\n".join(modified_files) + "\n")
                    f.write("EOF\n")

        return modified_files

    except subprocess.CalledProcessError as e:
        print(f"❌ Error detecting changes: {e.stderr}", file=sys.stderr)
        return None


def create_validation_matrix(
    base_branch: str = "main",
    recipes_dir: str = "recipes_collection/recipes/",
    custom_recipes: list = None,
    custom_instances: list = None,
    custom_platform: list = None,
    custom_dataset: str = None,
    batch_size: int = 15,
    max_workers: int = 5,
) -> dict:
    """
    Create validation matrix and batch into platform-consistent groups.

    Args:
        base_branch: Git branch to compare for detecting changes
        recipes_dir: Directory containing recipes
        custom_recipes: Optional list of recipes (overrides git detection)
        custom_instances: Optional list of instances (overrides all instance logic)
        custom_platform: Optional list of platforms (overrides default list of 3)
        custom_dataset: Optional dataset name
        batch_size: Max runs per batch (default 15)
        max_workers: Upper limit on number of workers to scale up (default 5)

    Returns:
        dict with batches, num_batches, total_runs
    """
    if custom_recipes:
        recipes = custom_recipes
    else:
        modified_files = detect_modified_recipes(base_branch, recipes_dir)
        if not modified_files:
            return {"batches": [], "num_batches": 0, "total_runs": 0}
        recipes = modified_files
        # Detect which are newly added (vs modified)
        new_recipes = _detect_new_recipes(base_branch, recipes_dir)

    matrix = []

    for recipe in recipes:
        # Normalize path: strip prefix if present
        normalized_recipe = recipe
        if normalized_recipe.startswith(RECIPES_PREFIX):
            normalized_recipe = normalized_recipe[len(RECIPES_PREFIX) :]

        for platform in PLATFORMS:
            # Determine instance types for this recipe+platform
            instances = _get_instance_types(
                recipe=recipe,
                platform=platform,
                custom_instances=custom_instances,
            )

            for instance_type in instances:
                matrix.append(
                    {
                        "recipe": normalized_recipe,
                        "platform": platform,
                        "instance_type": instance_type,
                    }
                )

    batches = _batch_by_platform(matrix, batch_size)

    # Cap the number of active batches to max_workers
    overflow_runs = []
    if len(batches) > max_workers:
        active_batches = batches[:max_workers]
        overflow_batches = batches[max_workers:]
        # Flatten overflow batch runs into a single list
        for batch in overflow_batches:
            for run in batch["runs"]:
                overflow_runs.append(
                    {
                        "recipe": run["recipe"],
                        "instance_type": run["instance_type"],
                        "platform": batch["platform"],
                    }
                )
        print(
            f"⚠️ Batches ({len(batches)}) exceed max_workers ({max_workers}). "
            f"Capping to {max_workers} active batches, {len(overflow_runs)} overflow runs saved."
        )
        batches = active_batches

    # Write overflow runs to file if any
    has_overflow = len(overflow_runs) > 0
    if has_overflow:
        with open("overflow_runs.json", "w") as f:
            json.dump(overflow_runs, f)
        print(f"📁 Wrote {len(overflow_runs)} overflow runs to overflow_runs.json")

    result = {
        "batches": batches,
        "num_batches": len(batches),
        "total_runs": len(matrix),
        "overflow_runs": len(overflow_runs),
    }

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"num_batches={len(batches)}\n")
            f.write(f"has_overflow={str(has_overflow).lower()}\n")
            f.write("batches_json<<EOF\n")
            f.write(json.dumps(batches) + "\n")
            f.write("EOF\n")

    return result


def _detect_new_recipes(base_branch: str, recipes_dir: str) -> set:
    """Detect newly added recipes (not just modified)."""
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD", "--name-only", "--diff-filter=A", "--", recipes_dir],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(f for f in result.stdout.strip().split("\n") if f and f.endswith(".yaml"))


def _get_instance_types(recipe: str, platform: str, custom_instances: list) -> list:
    """
    Determine instance types for a recipe+platform combination.

    Priority:
    1. Custom instances passed via input → use those (filtered by platform)
    2. Recipe has instance_types field → use recipe's list (filtered by platform)
    3. Recipe has no instance_types field → use platform defaults
    """
    platform_defaults = PLATFORM_INSTANCE_CONSTRAINTS.get(platform, [])

    # Custom instances override everything
    if custom_instances:
        filtered = [i for i in custom_instances if i in platform_defaults]
        return filtered if filtered else platform_defaults

    # Read recipe's instance_types field
    recipe_instances = _read_recipe_instance_types(recipe)

    if not recipe_instances:
        # No instance_types in recipe → use platform defaults
        return platform_defaults

    # Filter recipe instances by platform constraints
    filtered = [i for i in recipe_instances if i in platform_defaults]
    return filtered if filtered else platform_defaults


def _read_recipe_instance_types(recipe_path: str) -> list:
    """Read instance_types field from recipe YAML."""
    try:
        with open(recipe_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get("instance_types", [])
    except Exception:
        return []


def _batch_by_platform(matrix: list, batch_size: int) -> list:
    """
    Batch matrix entries, keeping platform consistent per batch.

    Returns list of batches, each with:
    - batch_id: int
    - platform: str
    - runs: list of {recipe, instance_type}
    """
    # Group by platform
    by_platform = {}
    for entry in matrix:
        platform = entry["platform"]
        if platform not in by_platform:
            by_platform[platform] = []
        by_platform[platform].append(
            {
                "recipe": entry["recipe"],
                "instance_type": entry["instance_type"],
            }
        )

    # Chunk each platform group
    batches = []
    batch_id = 1

    for platform in PLATFORMS:
        runs = by_platform.get(platform, [])
        for i in range(0, len(runs), batch_size):
            chunk = runs[i : i + batch_size]
            batches.append(
                {
                    "batch_id": batch_id,
                    "platform": platform,
                    "runs": chunk,
                }
            )
            batch_id += 1

    return batches


def update_validation_config(platform_type, recipe_files, config_path, instance_type):
    """
    Update validation config with recipe list, platform, and container image

    Args:
        recipe_files: List of recipe file paths (relative to recipes_collection/recipes/)
        config_path: Path to common_validation_config.yaml
        instance_type: Instance type to be used
        container_image: Container image to use
    """
    try:
        # Read the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update recipe_list
        config["recipe_list"] = recipe_files

        # Set the platform
        config["platform"] = platform_type

        # Update instance type
        config["instance_type_list"] = [it.strip() for it in instance_type.split(",") if it.strip()]

        # Write back the config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Updated config with {len(recipe_files)} recipe(s)")
        print(f"   Platform: {config['platform']}")
        print(f"   Recipes: {recipe_files}")

        return True

    except Exception as e:
        print(f"❌ Error updating validation config: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Helper functions for EKS recipe validation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # detect-changes command
    detect_parser = subparsers.add_parser("detect-changes", help="Detect modified recipe files")
    detect_parser.add_argument("--base-branch", default="main", help="Base branch to compare against")
    detect_parser.add_argument("--recipes-dir", default="recipes_collection/recipes/", help="Recipes directory path")

    # create-matrix command
    create_parser = subparsers.add_parser("create-matrix", help="Create validation matrix and batch")
    create_parser.add_argument("--base-branch", default="main")
    create_parser.add_argument("--recipes-dir", default="recipes_collection/recipes/")
    create_parser.add_argument("--recipes", help="Comma-separated recipe list (optional)")
    create_parser.add_argument("--instances", help="Comma-separated instance types (optional)")
    create_parser.add_argument("--batch-size", type=int, default=25)
    create_parser.add_argument(
        "--max-workers", type=int, default=5, help="Upper limit on number of workers to scale up"
    )

    # update-config command
    update_parser = subparsers.add_parser("update-config", help="Update validation configuration")
    update_parser.add_argument("--recipe-files", nargs="+", required=True, help="List of recipe files to validate")
    update_parser.add_argument("--config-path", required=True, help="Path to common_validation_config.yaml")
    update_parser.add_argument("--instance-type", required=True, help="Instance type to use for validation")
    update_parser.add_argument("--platform", required=True, help="Platform type to use for validation")

    args = parser.parse_args()

    # TODO additional validation helper functions

    if args.command == "detect-changes":
        modified_files = detect_modified_recipes(args.base_branch, args.recipes_dir)
        sys.exit(0 if modified_files is not None else 1)

    elif args.command == "create-matrix":
        custom_recipes = args.recipes.split(",") if args.recipes else None
        custom_instances = args.instances.split(",") if args.instances else None
        result = create_validation_matrix(
            base_branch=args.base_branch,
            recipes_dir=args.recipes_dir,
            custom_recipes=custom_recipes,
            custom_instances=custom_instances,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )
        print(f"Created {result['num_batches']} batches with {result['total_runs']} total runs")
        if result.get("overflow_runs", 0) > 0:
            print(f"⚠️ {result['overflow_runs']} overflow runs written to overflow_runs.json")

    elif args.command == "update-config":
        success = update_validation_config(args.platform, args.recipe_files, args.config_path, args.instance_type)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
