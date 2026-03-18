import logging
import os
import shutil
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
    """Get the absolute path to common validation config.

    If common_validation_config.yaml does not exist, it is automatically created
    by copying common_validation_config.example.yaml on first usage.
    """
    project_root = get_project_root()
    config_path = os.path.join(project_root, "scripts", "validations", "common_validation_config.yaml")
    example_path = os.path.join(project_root, "scripts", "validations", "common_validation_config.example.yaml")

    if not os.path.exists(config_path):
        if not os.path.exists(example_path):
            raise FileNotFoundError(
                f"Neither '{config_path}' nor the example file '{example_path}' were found. "
                "Cannot initialize validation config."
            )
        shutil.copy2(example_path, config_path)
        logging.info(
            f"Created '{config_path}' from example. " "Edit this file to customize your validation configuration."
        )

    return config_path
