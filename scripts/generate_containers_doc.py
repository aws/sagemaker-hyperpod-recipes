#!/usr/bin/env python3
"""Generate CONTAINERS.md documentation from regional parameter JSON files.

Shows container images available by region for each recipe. Only includes recipes
that exist in recipes_collection/recipes/.

Container data is sourced from *_regional_parameters.json files under
launcher/recipe_templatization/.

Usage:
    python scripts/generate_containers_doc.py           # Generate docs/CONTAINERS.md
    python scripts/generate_containers_doc.py --check   # Validate docs/CONTAINERS.md is up-to-date
    python scripts/generate_containers_doc.py --check --diff  # Show what changed
"""

import argparse
import difflib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import new utility
from hyperpod_recipes import list_recipes_by_model

# Import helpers from recipes doc generator
from scripts.generate_recipes_doc import _display_family, _get, scan_recipes

# =============================================================================
# PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATIZATION_DIR = REPO_ROOT / "launcher" / "recipe_templatization"
OUTPUT_FILE = REPO_ROOT / "docs" / "CONTAINERS.md"

# =============================================================================
# FRAMEWORK REGISTRY — maps to regional_parameters.json files
# =============================================================================
FRAMEWORK_REGIONAL_FILES = {
    "llmft": "llmft/llmft_regional_parameters.json",
    "checkpointless": "checkpointless/checkpointless_regional_parameters.json",
    "evaluation": "evaluation/evaluation_regional_parameters.json",
    "verl": "verl/verl_regional_parameters.json",
    "nova": "nova/nova_regional_parameters.json",
    "mtrl": "mtrl/mtrl_regional_parameters.json",
    "mtrl_eval": "mtrl_eval/mtrl_eval_regional_parameters.json",
}

# =============================================================================
# REGION ORDERING — consistent order for table columns
# =============================================================================
REGION_ORDER = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ap-northeast-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "eu-central-1",
    "eu-north-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "sa-east-1",
]

# =============================================================================
# MODEL FAMILY ORDERING — determines display order of model families
# =============================================================================
FAMILY_ORDER = [
    "llama",
    "qwen",
    "deepseek",
    "gpt_oss",
    "nova",
    "open-source",
    "custom_model",
]


# =============================================================================
# LOAD REGIONAL PARAMETERS
# =============================================================================


def _normalize_regional_entry(recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize regional parameter entry to a consistent k8s/sm_jobs structure.

    Some files (e.g., evaluation) use container_image/smtj_container_image keys
    instead of k8s/sm_jobs pipeline wrappers. Normalize to the k8s/sm_jobs format.
    """
    if "k8s" in recipe_data or "sm_jobs" in recipe_data:
        return recipe_data

    normalized = {}
    if "container_image" in recipe_data:
        normalized["k8s"] = {"container_image": recipe_data["container_image"]}
    if "smtj_container_image" in recipe_data:
        normalized["sm_jobs"] = {"container_image": recipe_data["smtj_container_image"]}
    return normalized if normalized else recipe_data


def _load_regional_parameters() -> Dict[str, Dict[str, Any]]:
    """Load all regional parameter JSON files.

    Returns:
        Dict mapping recipe_key to regional data, e.g.:
        {
            "llmft": {...},
            "nova_lite_2_0_p5_gpu_lora_sft": {...}
        }
    """
    regional_data = {}

    for framework, rel_path in FRAMEWORK_REGIONAL_FILES.items():
        json_path = TEMPLATIZATION_DIR / rel_path
        if not json_path.exists():
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Handle recipe_container_mapping wrapper (used by evaluation)
            if "recipe_container_mapping" in data:
                data = data["recipe_container_mapping"]

            # Merge all recipe entries from this file
            for recipe_key, recipe_data in data.items():
                if recipe_key.startswith("_"):  # Skip comments
                    continue
                if not isinstance(recipe_data, dict):
                    continue
                regional_data[recipe_key] = _normalize_regional_entry(recipe_data)

        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}", file=sys.stderr)

    return regional_data


def _match_recipe_to_regional_key(recipe_filename: str, regional_keys: Set[str]) -> Optional[str]:
    """Match a recipe filename to its regional parameter key.

    Regional keys use patterns like:
    - "llmft" (generic fallback)
    - "nova_lite_2_0_p5_gpu_lora_sft" (specific recipe)

    Args:
        recipe_filename: e.g., "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml"
        regional_keys: Set of all keys from regional_parameters.json files

    Returns:
        Matching regional key or None
    """
    # Remove .yaml/.yml extension and try exact match
    base = recipe_filename.replace(".yaml", "").replace(".yml", "")

    # Try exact match first
    if base in regional_keys:
        return base

    # Try framework-level fallback (e.g., "llmft", "nova")
    for framework in FRAMEWORK_REGIONAL_FILES.keys():
        if base.startswith(framework) and framework in regional_keys:
            return framework

    # Try substring matching for Nova recipes with technique suffixes
    # e.g., "nova_lite_2_0_p5_gpu_lora_sft" should match various Nova recipe files
    for key in regional_keys:
        if len(key) > 10 and key in base:
            return key

    # Open-source eval recipes all use the same container
    if base.startswith("open_source_") and base.endswith("_eval"):
        if "open_source_deterministic_eval" in regional_keys:
            return "open_source_deterministic_eval"

    return None


def _get_container_for_region(recipe_data: Dict[str, Any], pipeline: str, region: str) -> Optional[str]:
    """Extract container image URI for a specific pipeline stage and region.

    Args:
        recipe_data: Regional parameter data for one recipe
        pipeline: "k8s" or "sm_jobs"
        region: AWS region code

    Returns:
        Container image URI or None
    """
    container_data = _get(recipe_data, pipeline, "container_image", "prod", region)
    return container_data


def _format_container_cell(k8s_uri: Optional[str], sm_uri: Optional[str], service_only: bool = False) -> str:
    """Format a table cell showing container URIs for k8s and/or sm_jobs.

    Args:
        k8s_uri: Container URI for k8s pipeline
        sm_uri: Container URI for sm_jobs pipeline
        service_only: If True, prefix cell with "SERVICE ONLY"

    Returns:
        Formatted cell content
    """
    if not k8s_uri and not sm_uri:
        return "—"

    if service_only:
        uri = sm_uri or k8s_uri
        return f"SERVICE ONLY {uri}"

    if k8s_uri == sm_uri:
        return k8s_uri

    parts = []
    if k8s_uri:
        parts.append(f"K8S: {k8s_uri}")
    if sm_uri:
        parts.append(f"SM: {sm_uri}")

    return " / ".join(parts)


# =============================================================================
# TABLE GENERATION
# =============================================================================


def _get_recipe_display_name(recipe: Any) -> str:
    """Get display name for a recipe in the container table.

    Args:
        recipe: Recipe object from scan_recipes() or hyperpod_recipes.Recipe

    Returns:
        Display name (from display_name field, model_short_name, or model)
    """
    # Try multiple approaches to get a display name
    if hasattr(recipe, "config"):
        # hyperpod_recipes.Recipe object
        display = recipe.config.get("display_name", "")
        if display:
            return display

    # scan_recipes() Recipe object
    if hasattr(recipe, "model_short_name") and recipe.model_short_name:
        return recipe.model_short_name
    if hasattr(recipe, "model") and recipe.model:
        return recipe.model

    # Fallback to recipe ID or name
    if hasattr(recipe, "recipe_id"):
        return recipe.recipe_id
    if hasattr(recipe, "name"):
        return recipe.name

    return "Unknown"


def _build_container_table(recipes: List[Any], regions: List[str], model_id: str = "") -> List[str]:
    """Build a markdown table showing containers by region.

    Args:
        recipes: List of Recipe objects for this group
        regions: List of region codes to show as columns
        model_id: JumpStart model ID for this group (used for model-specific regional lookups)

    Returns:
        List of markdown table lines
    """
    if not recipes:
        return []

    # Load all regional parameters once
    regional_params = _load_regional_parameters()
    regional_keys = set(regional_params.keys())

    # Build table rows
    rows = []
    for recipe in recipes:
        # Get recipe filename
        if hasattr(recipe, "recipe_path"):
            recipe_filename = Path(recipe.recipe_path).name
        elif hasattr(recipe, "path"):
            recipe_filename = Path(recipe.path).name
        else:
            continue

        is_mtrl = recipe_filename.startswith("mtrl")

        # For mtrl eval recipes, prefer model-specific regional key, fall back to generic
        # (mirrors MtrlEvalRecipeTemplateProcessor which matches by model ID substring)
        if is_mtrl and "eval" in recipe_filename:
            if model_id in regional_keys:
                regional_key = model_id
            else:
                regional_key = _match_recipe_to_regional_key(recipe_filename, regional_keys)
        else:
            regional_key = _match_recipe_to_regional_key(recipe_filename, regional_keys)

        if not regional_key:
            # No container data for this recipe
            continue

        recipe_data = regional_params[regional_key]
        row_cells = [_get_recipe_display_name(recipe)]

        for region in regions:
            k8s_uri = _get_container_for_region(recipe_data, "k8s", region)
            sm_uri = _get_container_for_region(recipe_data, "sm_jobs", region)
            cell = _format_container_cell(k8s_uri, sm_uri, service_only=is_mtrl)
            row_cells.append(cell)

        rows.append(row_cells)

    if not rows:
        return []

    rows.sort(key=lambda r: r[0])

    # Build table header
    header_cells = ["Recipe"] + regions
    header = "| " + " | ".join(header_cells) + " |"
    separator = "|" + "|".join(["-----"] * len(header_cells)) + "|"

    # Build table rows
    table_rows = ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]

    return [header, separator] + table_rows


def _get_model_family_from_recipe(recipe: Any) -> str:
    """Extract model family from a recipe object.

    Args:
        recipe: Recipe object (either from scan_recipes or hyperpod_recipes.Recipe)

    Returns:
        Family name (e.g., "llama", "qwen", "nova"), with version suffixes stripped
    """
    # Try scan_recipes() Recipe object
    if hasattr(recipe, "family"):
        family = recipe.family
    elif hasattr(recipe, "path"):
        # Try hyperpod_recipes.Recipe object - extract from path
        path = Path(recipe.path)
        # Path structure: .../fine-tuning/llama/recipe.yaml or .../nova/SFT/recipe.yaml
        parts = path.parts
        family = "unknown"
        for i, part in enumerate(parts):
            if part in ("fine-tuning", "training", "evaluation"):
                if i + 1 < len(parts):
                    family = parts[i + 1]
                    break
    else:
        family = "unknown"

    # Strip version suffixes like "qwen-0_7_0" → "qwen", "gpt_oss-0_7_0" → "gpt_oss"
    # But preserve model names like "gemma4" (the 4 is part of the model name, not a version)
    family_lower = family.lower()
    # Only strip if it matches the specific pattern "-0_7_0" (recipe version suffix)
    if "-0_7_0" in family_lower:
        family = family_lower.split("-0_7_0")[0]

    # Normalize gemma variants: "gemma" and "gemma4" both refer to Gemma 4 models
    # Check the stripped value, not the original
    if family.lower() in ("gemma", "gemma4"):
        family = "gemma"

    return family


def _get_active_regions_for_recipes(recipes: List[Any]) -> List[str]:
    """Get list of AWS regions that have containers for the given recipes.

    Args:
        recipes: List of Recipe objects

    Returns:
        List of region codes in standard order
    """
    regional_params = _load_regional_parameters()
    regional_keys = set(regional_params.keys())
    active_regions = set()

    for recipe in recipes:
        # Get recipe filename
        if hasattr(recipe, "recipe_path"):
            recipe_filename = Path(recipe.recipe_path).name
        elif hasattr(recipe, "path"):
            recipe_filename = Path(recipe.path).name
        else:
            continue

        regional_key = _match_recipe_to_regional_key(recipe_filename, regional_keys)
        if regional_key and regional_key in regional_params:
            recipe_data = regional_params[regional_key]
            for pipeline in ["k8s", "sm_jobs"]:
                prod_data = _get(recipe_data, pipeline, "container_image", "prod")
                if isinstance(prod_data, dict):
                    active_regions.update(prod_data.keys())

    # Return regions in standard order
    return [r for r in REGION_ORDER if r in active_regions]


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================


def generate_markdown(recipes: List[Any]) -> str:
    """Generate complete CONTAINERS.md content organized by Model Family > Model > Recipe.

    Args:
        recipes: All recipes from scan_recipes()

    Returns:
        Complete markdown document as string
    """
    lines = [
        "# Amazon SageMaker HyperPod Recipe Containers",
        "",
        "This document shows container images available by AWS region for each recipe,",
        "organized by model family and JumpStart model ID.",
        "",
        "Container images are shown for both Kubernetes (K8S) and SageMaker Jobs (SM) pipelines. "
        "When both pipelines use the same container, only one URI is shown.",
        "",
        "## Table of Contents",
        "",
    ]

    # Get recipes grouped by JumpStart model ID
    try:
        recipes_by_model_id = list_recipes_by_model()
    except Exception as e:
        print(f"Warning: Could not load recipes by model: {e}", file=sys.stderr)
        recipes_by_model_id = {}

    # Group models by family
    models_by_family = defaultdict(list)
    for model_id, model_recipes in recipes_by_model_id.items():
        if not model_recipes:
            continue
        # Get family from first recipe
        family = _get_model_family_from_recipe(model_recipes[0])
        models_by_family[family].append((model_id, model_recipes))

    # Sort families according to FAMILY_ORDER
    ordered_families = []
    for family in FAMILY_ORDER:
        if family in models_by_family:
            ordered_families.append(family)
    # Add any remaining families not in the order
    for family in sorted(models_by_family.keys()):
        if family not in ordered_families:
            ordered_families.append(family)

    # Generate TOC
    for family in ordered_families:
        display_family = _display_family(family)
        anchor = display_family.lower().replace(" ", "-")
        lines.append(f"- [{display_family}](#{anchor})")

    lines.append("")

    # Generate sections for each family
    for family in ordered_families:
        display_family = _display_family(family)
        lines.extend([f"## {display_family}", ""])

        # Sort models by model ID
        family_models = sorted(models_by_family[family], key=lambda x: x[0])

        for model_id, model_recipes in family_models:
            # Get regions for this model's recipes
            regions = _get_active_regions_for_recipes(model_recipes)
            if not regions:
                # Skip models with no container data
                continue

            # Use model ID as section title
            lines.extend([f"### {model_id}", ""])

            # Build table for this model
            table = _build_container_table(model_recipes, regions, model_id=model_id)
            if table:
                lines.extend(table)
                lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def run_generation():
    """Generate CONTAINERS.md file."""
    print("Scanning recipes...")
    recipes = scan_recipes()
    print(f"Found {len(recipes)} recipes")

    print("Loading regional parameters...")
    regional_params = _load_regional_parameters()
    print(f"Loaded {len(regional_params)} regional parameter entries")

    print("Generating CONTAINERS.md...")
    content = generate_markdown(recipes)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"Generated: {OUTPUT_FILE}")


def check_generation(show_diff: bool = False) -> bool:
    """Check if CONTAINERS.md is up-to-date.

    Args:
        show_diff: Whether to show unified diff

    Returns:
        True if up-to-date, False otherwise
    """
    print("Generating expected content...")
    recipes = scan_recipes()
    expected = generate_markdown(recipes)

    if not OUTPUT_FILE.exists():
        if show_diff:
            print(f"MISSING: {OUTPUT_FILE}")
        return False

    actual = OUTPUT_FILE.read_text(encoding="utf-8")
    if actual == expected:
        return True

    if show_diff:
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile="a/docs/CONTAINERS.md",
            tofile="b/docs/CONTAINERS.md",
            lineterm="",
        )
        print("".join(diff))

    return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate CONTAINERS.md documentation from regional parameter JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s          Generate CONTAINERS.md\n  %(prog)s --check  Check if up-to-date",
    )
    parser.add_argument("--check", action="store_true", help="Check if generated content matches disk")
    parser.add_argument("--diff", action="store_true", help="Show unified diff (requires --check)")
    args = parser.parse_args()

    if args.check:
        if check_generation(show_diff=args.diff):
            print("✓ CONTAINERS.md is up-to-date.")
            sys.exit(0)
        print(f"\n{'=' * 70}\nERROR: CONTAINERS.md is out of sync\n{'=' * 70}\n" f"To fix: python {sys.argv[0]}\n")
        sys.exit(1)

    run_generation()


if __name__ == "__main__":
    main()
