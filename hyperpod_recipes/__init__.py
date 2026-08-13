import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from hyperpod_recipes.recipe import RECIPES_DIR, Recipe


@lru_cache(maxsize=1)
def list_recipes() -> List[Recipe]:
    """
    Returns a list of Recipe objects
    """
    if not os.path.exists(RECIPES_DIR):
        raise FileNotFoundError(f"Recipes directory not found: {RECIPES_DIR}")

    # Skip __pycache__ and hydra_config directories
    # hydra_config contains Hydra composition components, not standalone recipes
    skip_dirs = {"__pycache__", "hydra_config"}

    recipes = []
    for root, dirs, files in os.walk(RECIPES_DIR):
        # Prune directories we don't want to traverse
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for f in files:
            # Only include .yaml files
            if not f.endswith(".yaml"):
                continue
            abs_path = os.path.join(root, f)
            recipes.append(Recipe(abs_path))
    recipes.sort(key=lambda r: r.name)
    return recipes


def get_recipe(recipe_id: str) -> Recipe:
    """
    Return a single Recipe object by its recipe_id.
    Raises KeyError if not found.
    """
    possible_path = os.path.join(RECIPES_DIR, recipe_id + ".yaml")

    if not os.path.exists(possible_path):
        raise KeyError(f"Recipe not found: '{recipe_id}'.")

    return Recipe(possible_path)


def list_recipes_by_model() -> Dict[str, List[Recipe]]:
    """
    [EXPERIMENTAL] Returns a dictionary mapping JumpStart model IDs to lists of Recipe objects.

    This function groups recipes by their associated JumpStart model ID, allowing you to find
    all recipes applicable to a specific model. The model ID is extracted from the recipe's
    `run.name` field and mapped through the JumpStart model ID mapping.

    Returns:
        Dictionary where keys are JumpStart model IDs (e.g., "meta-textgeneration-llama-3-1-8b-instruct")
        and values are lists of Recipe objects that use that model.

    Example:
        >>> recipes_by_model = list_recipes_by_model()
        >>> llama_recipes = recipes_by_model.get("meta-textgeneration-llama-3-1-8b-instruct", [])
        >>> for recipe in llama_recipes:
        ...     print(recipe.name)

    Note:
        This function is experimental and the API may change in future versions.
        Recipes without a valid JumpStart model ID mapping are excluded from the results.
    """
    # Load JumpStart model ID mapping
    repo_root = Path(RECIPES_DIR).parent.parent
    model_id_map_path = repo_root / "launcher" / "recipe_templatization" / "jumpstart_model-id_map.json"

    if not model_id_map_path.exists():
        raise FileNotFoundError(f"JumpStart model ID map not found: {model_id_map_path}")

    with open(model_id_map_path) as f:
        model_id_map = json.load(f)

    # Group recipes by JumpStart model ID
    recipes_by_model_id = defaultdict(list)
    open_source_eval_recipes = []
    mtrl_eval_recipes = []

    for recipe in list_recipes():
        try:
            # Collect open-source eval recipes separately
            if "evaluation/open-source" in recipe.path:
                open_source_eval_recipes.append(recipe)
                continue

            # Collect mtrl eval recipes separately
            if "evaluation/mtrl" in recipe.path:
                mtrl_eval_recipes.append(recipe)
                continue

            run_config = recipe.config.get("run", {})

            # Try run.name first
            model_name = run_config.get("name")
            jumpstart_id = model_id_map.get(model_name) if model_name else None

            # If run.name didn't map, try run.model_type (for Nova recipes)
            if not jumpstart_id:
                model_type = run_config.get("model_type")
                if model_type:
                    jumpstart_id = model_id_map.get(model_type)

            if jumpstart_id:
                recipes_by_model_id[jumpstart_id].append(recipe)
        except Exception:
            # Skip recipes that can't be loaded or don't have the expected structure
            continue

    # Add open-source eval recipes to any model that has llmft or verl training recipes
    # Add mtrl eval recipes to any model that has mtrl training recipes
    for model_id, recipes in recipes_by_model_id.items():
        has_llmft_or_verl = any(Path(r.path).name.startswith(("llmft", "verl")) for r in recipes)
        if has_llmft_or_verl and open_source_eval_recipes:
            recipes.extend(open_source_eval_recipes)

        has_mtrl = any(Path(r.path).name.startswith("mtrl") for r in recipes)
        if has_mtrl and mtrl_eval_recipes:
            recipes.extend(mtrl_eval_recipes)

    return dict(recipes_by_model_id)
