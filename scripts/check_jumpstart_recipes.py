#!/usr/bin/env python3
"""
check_jumpstart_recipes.py

Checks which HyperPod recipes are available on JumpStart per AWS region.

For each region, the script uses the region-specific JumpStart S3 bucket:
    https://jumpstart-cache-prod-{region}.s3.{region}.amazonaws.com/

Steps:
  1. Load all local recipes via list_recipes()
  2. Load the launcher/recipe_templatization/jumpstart_model-id_map.json mapping
  3. For each region:
       a. Fetch the region's JumpStart models manifest
       b. Find the highest-version spec_key per mapped model ID
       c. Download each model spec and collect all recipe_file_path values
  4. Print a table: rows = recipes, columns = regions, cells = ✓ / ✗
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so `hyperpod_recipes` can be imported
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hyperpod_recipes import list_recipes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAPPING_PATH = os.path.join(
    _REPO_ROOT,
    "launcher",
    "recipe_templatization",
    "jumpstart_model-id_map.json",
)

EXCLUSIONS_PATH = os.path.join(_REPO_ROOT, "recipes_collection", "jumpstart_exclusions.yaml")

# Regions to check
REGIONS = ["us-west-2", "us-east-1", "ap-northeast-1", "eu-west-1"]

TICK = "✓"
CROSS = "✗"
EXCLUDED = "⊘"
ERROR = "✗!"


# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------


def load_exclusion_patterns() -> List[re.Pattern]:
    """Load and compile exclusion patterns from jumpstart_exclusions.yaml."""
    if not os.path.exists(EXCLUSIONS_PATH):
        print(f"  WARNING: Exclusions file not found: {EXCLUSIONS_PATH}")
        return []
    with open(EXCLUSIONS_PATH, "r") as f:
        data = yaml.safe_load(f) or {}
    patterns = data.get("exclusion_patterns", []) or []
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p))
        except re.error as e:
            print(f"  WARNING: Invalid regex '{p}': {e}")
    return compiled


def is_excluded(recipe_id: str, patterns: List[re.Pattern]) -> bool:
    """Return True if recipe_id matches any exclusion pattern."""
    return any(p.search(recipe_id) for p in patterns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def js_base_url(region: str) -> str:
    """Return the JumpStart S3 base URL for the given region."""
    return f"https://jumpstart-cache-prod-{region}.s3.{region}.amazonaws.com"


def fetch_json(url: str) -> Optional[Any]:
    """
    Download a JSON file from *url* and return the parsed object.
    Returns None if the URL returns a 404 (model/spec not present in that region).
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def parse_version(version_str: str) -> Tuple:
    """
    Simple version parser returning a comparable tuple of ints.
    Falls back to (0,) for unparseable strings.
    """
    try:
        return tuple(int(x) for x in version_str.split("."))
    except (ValueError, AttributeError):
        return (0,)


def get_highest_version_spec_keys(manifest: List[dict], model_ids: Set[str]) -> Dict[str, str]:
    """
    Given a JumpStart manifest list and a set of model_ids we care about,
    return {model_id -> spec_key} for the highest non-deprecated version.
    """
    best: Dict[str, Tuple] = {}  # model_id -> (parsed_version, spec_key)

    for entry in manifest:
        mid = entry.get("model_id", "")
        if mid not in model_ids:
            continue
        if entry.get("deprecated", False):
            continue
        spec_key = entry.get("spec_key", "")
        parsed = parse_version(entry.get("version", "0"))

        if mid not in best or parsed > best[mid][0]:
            best[mid] = (parsed, spec_key)

    return {mid: info[1] for mid, info in best.items()}


def get_available_recipe_paths_for_region(
    region: str,
    unique_js_model_ids: Set[str],
) -> Tuple[Set[str], Dict[str, str]]:
    """
    For a single region, fetch its JumpStart manifest + relevant model specs.

    Returns:
        (recipe_paths, recipe_path_to_js_model_id)
        recipe_paths            – set of all recipe_file_path values found
        recipe_path_to_js_model – mapping recipe_file_path -> JS model_id
    """
    base = js_base_url(region)
    manifest_url = f"{base}/models_manifest.json"

    print(f"  [{region}] Fetching manifest...", flush=True)
    manifest = fetch_json(manifest_url)
    if manifest is None:
        print(f"  [{region}] WARNING: manifest not found (404). Skipping region.", flush=True)
        return set(), {}

    spec_keys = get_highest_version_spec_keys(manifest, unique_js_model_ids)

    not_found = unique_js_model_ids - set(spec_keys.keys())
    if not_found:
        print(
            f"  [{region}] WARNING: {len(not_found)} model ID(s) not in manifest: " + ", ".join(sorted(not_found)),
            flush=True,
        )
    print(f"  [{region}] Found {len(spec_keys)} model spec(s) to fetch.", flush=True)

    recipe_paths: Set[str] = set()
    recipe_path_to_js_model: Dict[str, str] = {}
    total = len(spec_keys)
    for i, (model_id, spec_key) in enumerate(spec_keys.items(), start=1):
        url = f"{base}/{spec_key}"
        try:
            spec = fetch_json(url)
        except Exception as exc:
            print(f"  [{region}] WARNING: failed to fetch {model_id}: {exc}", flush=True)
            continue

        if spec is None:
            # 404 — spec not present in this region
            print(f"  [{region}]   [{i}/{total}] {model_id}: not present (404)", flush=True)
            continue

        found = 0
        for recipe_entry in spec.get("recipe_collection", []):
            rfp = recipe_entry.get("recipe_file_path", "")
            if rfp:
                recipe_paths.add(rfp)
                recipe_path_to_js_model[rfp] = model_id
                found += 1
        print(f"  [{region}]   [{i}/{total}] {model_id}: {found} recipe(s)", flush=True)

    print(f"  [{region}] Total recipe paths found: {len(recipe_paths)}", flush=True)
    return recipe_paths, recipe_path_to_js_model


def print_table(
    all_recipe_ids: List[str],
    region_recipe_paths: Dict[str, Set[str]],
    regions: List[str],
    recipe_to_js_model: Dict[str, str],
    exclusion_patterns: List[re.Pattern],
) -> bool:
    """
    Print a table showing recipe availability per region with exclusion awareness.

    Cells show one of four states:
      ✓   recipe is present on JumpStart
      ✗   recipe is missing and does NOT match any exclusion rule
      ⊘   recipe is missing but matches an exclusion rule (expected)
      ✗!  recipe is present but SHOULD have been excluded (error)

    Returns list of error recipes (✗!) found.
    """
    NO_MODEL = "(no mapping)"

    def sort_key(rid: str) -> tuple:
        model = recipe_to_js_model.get(rid, NO_MODEL)
        return (0 if model != NO_MODEL else 1, model, rid)

    sorted_ids = sorted(all_recipe_ids, key=sort_key)

    recipe_col_w = max((len(r) for r in sorted_ids), default=20) + 2
    region_col_w = max(max(len(r) for r in regions) + 2, 6)

    header = f"{'Recipe':<{recipe_col_w}}" + "".join(f"{r:^{region_col_w}}" for r in regions)
    sep = "=" * len(header)
    thin_sep = "-" * len(header)

    print()
    print(sep)
    print(header)
    print(sep)

    counts = {"present": 0, "missing_excluded": 0, "missing_unexpected": 0, "present_should_exclude": 0}
    prev_model: Optional[str] = None
    error_recipes: List[str] = []

    for recipe_id in sorted_ids:
        model = recipe_to_js_model.get(recipe_id, NO_MODEL)
        excluded = is_excluded(recipe_id, exclusion_patterns)

        if prev_model is not None and model != prev_model:
            print(thin_sep)
            print(f"  [{model}]")
        elif prev_model is None:
            print(f"  [{model}]")
        prev_model = model

        js_path = f"recipes/{recipe_id}.yaml"
        cells = []
        for r in regions:
            present = js_path in region_recipe_paths.get(r, set())
            if present and excluded:
                cells.append(ERROR)
                counts["present_should_exclude"] += 1
                if recipe_id not in error_recipes:
                    error_recipes.append(recipe_id)
            elif present:
                cells.append(TICK)
                counts["present"] += 1
            elif excluded:
                cells.append(EXCLUDED)
                counts["missing_excluded"] += 1
            else:
                cells.append(CROSS)
                counts["missing_unexpected"] += 1

        row = "".join(f"{c:^{region_col_w}}" for c in cells)
        print(f"{recipe_id:<{recipe_col_w}}{row}")

    print(sep)

    total = len(sorted_ids) * len(regions)
    print(
        f"\n  Legend:  {TICK} present   {EXCLUDED} missing (excluded)   {CROSS} missing (unexpected)   {ERROR} present but should be excluded (ERROR)"
    )
    print(
        f"\n  Cells: {counts['present']} present, {counts['missing_excluded']} excluded, {counts['missing_unexpected']} missing, {counts['present_should_exclude']} errors  (total {total})"
    )
    print(sep)
    print()

    return error_recipes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load local recipes
    # ------------------------------------------------------------------
    print("\n=== Step 1: Loading local recipes ===")
    local_recipes = list_recipes()
    sorted_recipe_ids = sorted(r.recipe_id for r in local_recipes)
    print(f"  Found {len(sorted_recipe_ids)} local recipe(s).")

    # ------------------------------------------------------------------
    # 2. Load recipe_name -> JS model_id mapping
    # ------------------------------------------------------------------
    print("\n=== Step 2: Loading JumpStart model-ID mapping ===")
    with open(MAPPING_PATH, "r") as fh:
        recipe_to_js_model: Dict[str, str] = json.load(fh)
    unique_js_model_ids: Set[str] = set(recipe_to_js_model.values())
    print(f"  Loaded {len(recipe_to_js_model)} entries → {len(unique_js_model_ids)} unique JS model IDs.")

    # ------------------------------------------------------------------
    # 2b. Load exclusion patterns
    # ------------------------------------------------------------------
    print("\n=== Step 2b: Loading exclusion patterns ===")
    exclusion_patterns = load_exclusion_patterns()
    print(f"  Loaded {len(exclusion_patterns)} exclusion pattern(s).")

    # ------------------------------------------------------------------
    # 3. For each region, fetch manifest + specs
    # ------------------------------------------------------------------
    print(f"\n=== Step 3: Checking availability per region: {', '.join(REGIONS)} ===")
    region_recipe_paths: Dict[str, Set[str]] = {}
    # recipe_path_to_js_model: union across all regions so every recipe path
    # seen anywhere gets a JS model label for grouping in the table.
    recipe_path_to_js_model: Dict[str, str] = {}
    for region in REGIONS:
        paths, path_model_map = get_available_recipe_paths_for_region(region, unique_js_model_ids)
        region_recipe_paths[region] = paths
        recipe_path_to_js_model.update(path_model_map)
        print()

    # Build a recipe_id -> JS model_id lookup used by print_table for grouping.
    # Use recipe.config.run.name (the short model name, e.g. "qwen-2d5-14b") to
    # look up the JS model ID in the recipe_to_js_model map.
    recipe_id_to_js_model: Dict[str, str] = {}

    for recipe in local_recipes:
        rid = recipe.recipe_id
        try:
            run_name = recipe.config.run.name
            if run_name and run_name in recipe_to_js_model:
                recipe_id_to_js_model[rid] = recipe_to_js_model[run_name]
        except Exception:
            pass

    # Fill in any remaining gaps from the JumpStart recipe_file_path mapping
    for js_path, model_id in recipe_path_to_js_model.items():
        if js_path.startswith("recipes/") and js_path.endswith(".yaml"):
            rid = js_path[len("recipes/") : -len(".yaml")]
            if rid not in recipe_id_to_js_model:
                recipe_id_to_js_model[rid] = model_id

    # ------------------------------------------------------------------
    # 4. Print table
    # ------------------------------------------------------------------
    print("=== Step 4: Results table ===")
    error_recipes = print_table(
        sorted_recipe_ids, region_recipe_paths, REGIONS, recipe_id_to_js_model, exclusion_patterns
    )

    # ------------------------------------------------------------------
    # 5. Per-region summary
    # ------------------------------------------------------------------
    print("=== Per-region summary ===")
    total = len(sorted_recipe_ids)
    for region in REGIONS:
        available = sum(
            1 for rid in sorted_recipe_ids if f"recipes/{rid}.yaml" in region_recipe_paths.get(region, set())
        )
        excluded = sum(1 for rid in sorted_recipe_ids if is_excluded(rid, exclusion_patterns))
        print(
            f"  {region:<20}  available: {available:>3}   excluded: {excluded:>3}   missing: {total - available - excluded:>3}"
        )
    print()

    if error_recipes:
        print(f"ERROR: {len(error_recipes)} recipe(s) present on JumpStart but should be excluded:\n")
        for rid in error_recipes:
            print(f"    - {rid}")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
