import json
import os
from pathlib import Path

import yaml


def get_recipe_metadata(file_path):
    """Extract recipe metadata from YAML file"""
    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Get relative path from recipes_collection/recipes/
        rel_path = str(Path(file_path).relative_to("recipes_collection/recipes/"))
        recipe_name = rel_path.replace(".yaml", "")

        # Extract latest version
        version = data.get("version")

        return {"recipe": recipe_name, "version": version}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    changed_files = os.environ.get("CHANGED_FILES", "").strip().split("\n")

    changed_files = [f for f in changed_files if f.strip()]

    if not changed_files:
        print("No files to process")
        return

    updated_recipes = []

    for file_path in changed_files:
        file_path = file_path.strip()
        if not file_path or not file_path.endswith(".yaml"):
            continue

        if os.path.exists(file_path):
            metadata = get_recipe_metadata(file_path)
            if metadata:
                updated_recipes.append(metadata)
        else:
            print(f"File deleted: {file_path}")

    # Output the metadata in the required format
    output = {"updatedRecipes": updated_recipes}

    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write("recipe_metadata<<EOF\n")
        f.write(json.dumps(output, indent=2))
        f.write("\nEOF\n")


if __name__ == "__main__":
    main()
