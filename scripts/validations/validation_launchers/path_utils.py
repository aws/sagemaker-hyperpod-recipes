import os
from pathlib import Path


def get_project_root():
    """Get the project root directory (where the recipes_collection folder is located)."""
    current_file = Path(__file__).resolve()
    # Navigate up from scripts/validations/validation_launchers/ to project root
    return current_file.parent.parent.parent.parent


def get_recipes_folder():
    """Get the absolute path to recipes_collection/recipes/"""
    return os.path.join(get_project_root(), "recipes_collection", "recipes")


def get_common_config():
    """Get the absolute path to common validation config"""
    return os.path.join(get_project_root(), "scripts", "validations", "common_validation_config.yaml")
