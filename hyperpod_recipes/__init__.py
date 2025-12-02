import os
from typing import List

from hyperpod_recipes.recipe import RECIPES_DIR, Recipe


def list_recipes() -> List[Recipe]:
    """
    Returns a list of Recipe objects
    """
    if not os.path.exists(RECIPES_DIR):
        raise FileNotFoundError(f"Recipes directory not found: {RECIPES_DIR}")

    recipes = []
    for root, _, files in os.walk(RECIPES_DIR):
        for f in files:
            abs_path = os.path.join(root, f)
            recipes.append(Recipe(abs_path))
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
