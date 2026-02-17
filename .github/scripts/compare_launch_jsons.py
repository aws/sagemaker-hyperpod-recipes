"""
Script to compare launch.jsons between the current branch and release branch
and bump versions for changed/new recipes.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.generate_launch_jsons import LaunchJsonGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare launch.jsons between current branch and release branch")
    parser.add_argument(
        "--release-branch", default="release", help="Name of the release branch to compare against (default: release)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/launch_json_comparison",
        help="Output directory for generated launch.jsons (default: results/launch_json_comparison)",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="Skip version validation and exit successfully after generating comparison results.",
    )
    parser.add_argument(
        "--fail-on-changes",
        action="store_true",
        help="Fail if any recipes have changes",
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["llmft", "nova", "verl"],
        help="Recipe prefixes to process (default: llmft). Options: llmft, nova, verl, evaluation, checkpointless",
    )
    parser.add_argument(
        "--job-types",
        nargs="+",
        default=["k8s", "sm_jobs"],
        help="Supported frameworks for recipes. Options: k8s, sm_jobs",
    )
    return parser.parse_args()


def normalize_json_for_comparison(data):
    """
    Normalize JSON data by removing random 5-character hash suffixes
    from run names to compare the JSON content.

    Pattern: model-name-XXXXX where XXXXX is a 5-char alphanumeric hash
    Examples:
      - deepseek-r1-distilled-llama-8b-6aqt6 -> deepseek-r1-distilled-llama-8b
      - deepseek-r1-distilled-llama-8b-86s5x -> deepseek-r1-distilled-llama-8b
    """
    hash_pattern = re.compile(r'(-[a-z0-9]{5})(?=_hydra\.yaml|"|\s|$)')

    def normalize_value(obj):
        if isinstance(obj, dict):
            return {k: normalize_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [normalize_value(item) for item in obj]
        elif isinstance(obj, str):
            # Replace 5-char hash suffixes
            return hash_pattern.sub("", obj)
        else:
            return obj

    return normalize_value(data)


def compare_json_content(json_path1, json_path2):
    """
    Deep compare two JSON files, ignoring random hash suffixes in run names.
    Returns True if content is identical (after normalization), False otherwise.
    """
    try:
        with open(json_path1, "r") as f:
            data1 = json.load(f)
        with open(json_path2, "r") as f:
            data2 = json.load(f)

        normalized1 = normalize_json_for_comparison(data1)
        normalized2 = normalize_json_for_comparison(data2)

        return normalized1 == normalized2

    except Exception as e:
        print(f"  Error comparing JSON files: {e}")
        return False


def compare_launch_jsons(current_results, old_results):
    new_recipes = []
    changed_recipes = []
    unchanged_recipes = []

    for recipe_file, current_paths in current_results.items():
        if recipe_file not in old_results:
            new_recipes.append(recipe_file)
        else:
            old_paths = old_results[recipe_file]

            # Compare each job type
            is_changed = False
            for job_type, current_path in current_paths.items():
                if current_path is None:
                    continue
                old_path = old_paths.get(job_type)
                if old_path is None:
                    is_changed = True
                    break
                if not compare_json_content(str(current_path), str(old_path)):
                    is_changed = True
                    break

            if is_changed:
                changed_recipes.append(recipe_file)
            else:
                unchanged_recipes.append(recipe_file)

    removed_recipes = [r for r in old_results if r not in current_results]

    return {"new": new_recipes, "changed": changed_recipes, "unchanged": unchanged_recipes, "removed": removed_recipes}


def get_recipe_version(recipe_file, working_dir=None):
    """Extract version from a recipe YAML file."""
    file_path = recipe_file
    if working_dir:
        file_path = os.path.join(working_dir, recipe_file)

    try:
        with open(file_path, "r") as f:
            content = f.read()

        version_match = re.search(r'^version:\s*["\']?(\d+\.\d+\.\d+)["\']?\s*$', content, re.MULTILINE)
        if version_match:
            return version_match.group(1)
    except Exception as e:
        print(f"  Warning: Could not read version from {recipe_file}: {e}")

    return None


def compare_versions(version1, version2):
    """
    Compare two semantic versions.
    Returns:
      - positive number if version1 > version2
      - 0 if version1 == version2
      - negative number if version1 < version2
    Returns None if either version is None.
    """
    if version1 is None or version2 is None:
        return None

    def parse_version(v):
        return tuple(int(x) for x in v.split("."))

    try:
        v1 = parse_version(version1)
        v2 = parse_version(version2)

        for a, b in zip(v1, v2):
            if a != b:
                return a - b
        return 0
    except Exception:
        return None


def setup_worktree(branch_name, temp_dir):
    """Create a git worktree for the specified branch and initialize submodules."""
    # Create worktree
    cmd = ["git", "worktree", "add", temp_dir, branch_name]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr}")
        sys.exit(1)

    # Initialize and update submodules in the worktree
    submodule_result = subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=temp_dir, capture_output=True, text=True
    )

    if submodule_result.returncode != 0:
        print(f"  Failed to initialize submodules in worktree of release branch: {submodule_result.stderr[:200]}")
        sys.exit(1)

    return temp_dir


def cleanup_worktree(temp_dir):
    """Remove the git worktree."""
    result = subprocess.run(["git", "worktree", "remove", temp_dir, "--force"], capture_output=True)

    if result.returncode != 0:
        print(f"  Warning: Failed to remove worktree cleanly: {result.stderr[:200]}")


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Launch.json Comparison Tool")
    print("=" * 60)

    # Step 1: Find all fine-tuning recipes
    print("\n[1/5] Finding recipes...")
    generator = LaunchJsonGenerator()
    recipes = generator.discover_recipes(prefixes=args.prefixes)

    if not recipes:
        print("No recipe files found. Exiting.")
        sys.exit(1)

    # Step 2: Generate launch.jsons for current branch
    print(f"\n[2/5] Generating launch.jsons for current branch...")
    current_output_dir = os.path.join(args.output_dir, "current")
    try:
        current_results = generator.generate_all_launch_jsons(
            recipes=recipes,
            output_dir=current_output_dir,
            job_types=args.job_types,
        )
    except Exception as e:
        print("Failed to generate launch.jsons for the current branch.")
        sys.exit(1)

    current_results_normalized = {}
    project_root = str(generator.working_dir)

    current_versions = {}
    for k, v in current_results.items():
        # Normalize the key to match current_results format
        normalized_key = k.replace(project_root + "/", "")
        current_results_normalized[normalized_key] = v

        # Get version from current branch's recipe
        current_versions[normalized_key] = get_recipe_version(normalized_key, working_dir=project_root)

    # Step 3: Setup worktree for release branch
    print(f"\n[3/5] Setting up worktree for '{args.release_branch}' branch...")
    temp_dir = tempfile.mkdtemp(prefix="launch_json_compare_")

    # Store release versions before worktree cleanup
    release_versions = {}

    try:
        setup_worktree(args.release_branch, temp_dir)
        print(f"  Worktree created at: {temp_dir}")

        # Step 4: Generate launch.jsons for release branch
        print(f"\n[4/5] Generating launch.jsons for '{args.release_branch}' branch...")

        # Create generator for the release branch worktree
        release_generator = LaunchJsonGenerator(working_dir=temp_dir)
        release_recipes = release_generator.discover_recipes(prefixes=args.prefixes)

        release_output_dir = os.path.abspath(os.path.join(args.output_dir, "old"))
        old_results = release_generator.generate_all_launch_jsons(
            recipes=release_recipes,
            output_dir=release_output_dir,
            job_types=args.job_types,
        )

        # Normalize old_results keys (convert absolute paths to relative)
        old_results_normalized = {}
        for k, v in old_results.items():
            # Normalize the key to match current_results format
            normalized_key = k.replace(temp_dir + "/", "")
            old_results_normalized[normalized_key] = v

            # Get version from release branch's recipe
            release_versions[normalized_key] = get_recipe_version(normalized_key, working_dir=temp_dir)

        release_success_count = sum(
            1 for result in old_results_normalized.values() for path in result.values() if path is not None
        )
        print(f"  Generated {release_success_count} launch.jsons")

    except Exception as e:
        print("Failed to set up branch worktree and generate launch.json files.")
        sys.exit(1)

    finally:
        # Cleanup worktree
        print(f"\n  Cleaning up worktree...")
        cleanup_worktree(temp_dir)

    # Step 5: Compare results
    print(f"\n[5/5] Comparing launch.jsons...")
    comparison = compare_launch_jsons(current_results_normalized, old_results_normalized)

    # If skip-version-check, exit early without version validation
    if args.skip_version_check:
        print("Skipping version check as requested.")
        sys.exit(0)

    # Print results
    print("COMPARISON RESULTS")

    if comparison["new"]:
        print(f"\nNEW recipes ({len(comparison['new'])}):")
        for r in sorted(comparison["new"]):
            print(f"  + {Path(r).name}")

    if comparison["changed"]:
        print(f"\nCHANGED recipes ({len(comparison['changed'])}):")
        for r in sorted(comparison["changed"]):
            print(f"  ~ {Path(r).name}")

    if comparison["removed"]:
        print(f"\nREMOVED recipes ({len(comparison['removed'])}):")
        for r in sorted(comparison["removed"]):
            print(f"  - {Path(r).name}")

    if comparison["unchanged"]:
        print(f"\nUNCHANGED recipes: {len(comparison['unchanged'])}")

    # Determine which recipes actually need version bumps
    # Changed recipes need bump only if current version <= release version
    recipes_needing_bump = []
    recipes_already_bumped = []

    for recipe_file in comparison["changed"]:
        current_ver = current_versions.get(recipe_file)
        release_ver = release_versions.get(recipe_file)

        version_cmp = compare_versions(current_ver, release_ver)

        if version_cmp is None:
            # Could not compare versions, assume needs bump
            recipes_needing_bump.append((recipe_file, current_ver, release_ver, "unable to compare"))
        elif version_cmp > 0:
            # Current version is higher, already bumped
            recipes_already_bumped.append((recipe_file, current_ver, release_ver))
        else:
            # Current version is same or lower, needs bump
            recipes_needing_bump.append((recipe_file, current_ver, release_ver, "version not bumped"))

    # New recipes (if they don't exist in release, they're new)
    new_recipes_info = []
    for recipe_file in comparison["new"]:
        current_ver = current_versions.get(recipe_file)
        new_recipes_info.append((recipe_file, current_ver))

    # Print version comparison details
    if recipes_already_bumped:
        print(f"\nCHANGED recipes with version already bumped ({len(recipes_already_bumped)}):")
        for recipe_file, current_ver, release_ver in recipes_already_bumped:
            print(f"  ✓ {Path(recipe_file).name}: {release_ver} -> {current_ver}")

    if recipes_needing_bump:
        print(f"\nCHANGED recipes needing version bump ({len(recipes_needing_bump)}):")
        for recipe_file, current_ver, release_ver, reason in recipes_needing_bump:
            print(f"  ✗ {Path(recipe_file).name}: {current_ver} (release: {release_ver}) - {reason}")

    if recipes_needing_bump and args.fail_on_changes:
        print("\n" + "-" * 60)
        print("ERROR: Recipe changes detected without version bumps!")
        print("=" * 60)
        print("\nThe following recipes have changes and need version updates:")
        for recipe_file, current_ver, release_ver, reason in recipes_needing_bump:
            print(f"  - {recipe_file}")
            print(f"    Current branch: {current_ver}, Release branch to JumpStart: {release_ver}")
        print(
            "\nPlease update the 'version' field in these recipe files before merging. This could be a major or minor version bump."
        )
        print(
            "NOTE: for Nova recipe version bumps, you should edit the versions in launcher/recipe_templatization/nova/nova_metadata.json"
        )

    # Summary
    print("SUMMARY")
    print(f"  New:              {len(comparison['new'])}")
    print(f"  Changed:          {len(comparison['changed'])}")
    print(f"    - Already bumped: {len(recipes_already_bumped)}")
    print(f"    - Needs bump:     {len(recipes_needing_bump)}")
    print(f"  Unchanged:        {len(comparison['unchanged'])}")
    print(f"  Removed:          {len(comparison['removed'])}")

    results_file = os.path.join(args.output_dir, "comparison_results.json")
    comparison["needs_version_bump"] = [r[0] for r in recipes_needing_bump]
    comparison["version_already_bumped"] = [r[0] for r in recipes_already_bumped]
    with open(results_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    # Exit with error code if there are recipes needing bumps and fail_on_changes is set
    if recipes_needing_bump and args.fail_on_changes:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
