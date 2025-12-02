"""
Unit test for nova_metadata.json validation.

This test validates that the production nova_metadata.json file is in sync
with the baseline artifact, ensuring no unintended changes are introduced.
"""

import json
import unittest
from pathlib import Path

import yaml


class TestNovaMetadataBaseline(unittest.TestCase):
    """Test that nova_metadata.json matches the baseline."""

    def test_nova_metadata_matches_baseline(self):
        """
        Compare nova_metadata.json against the baseline artifact.

        This test validates that the production nova_metadata.json file is in sync
        with the baseline test artifact. If they differ, the test fails with detailed
        information about what changed, and instructs the user to update manually.

        The metadata (display_name, version, instance_types) has been moved out of
        recipe YAML files and into nova_metadata.json for centralized management.

        Note: assertIsInstance(obj, type) checks if obj is an instance of type.
        For example, assertIsInstance(x, str) verifies x is a string.
        """
        project_root = Path(__file__).parent.parent
        production_file = project_root / "launcher" / "recipe_templatization" / "nova" / "nova_metadata.json"
        baseline_file = (
            project_root
            / "tests"
            / "launcher"
            / "recipe_templatization"
            / "baseline_artifacts"
            / "nova_recipes_mapping.json"
        )

        # Fail if baseline doesn't exist
        if not baseline_file.exists():
            self.fail(f"Baseline file not found: {baseline_file}\n" f"Cannot validate without baseline artifact.")

        # Fail if production file doesn't exist
        if not production_file.exists():
            self.fail(
                f"nova_metadata.json not found at: {production_file}\n"
                f"This file should exist and contain recipe metadata."
            )

        # Load both files
        with open(baseline_file, "r") as f:
            baseline_mapping = json.load(f)

        with open(production_file, "r") as f:
            production_mapping = json.load(f)

        # Validate structure of production file
        self.assertGreater(len(production_mapping), 0, "nova_metadata.json is empty")

        # Verify each entry has correct structure
        for recipe_name, metadata in production_mapping.items():
            # Recipe name should not have .yaml extension
            self.assertFalse(recipe_name.endswith(".yaml"), f"Recipe name should not end with .yaml: {recipe_name}")

            # Recipe name should start with nova_
            self.assertTrue(recipe_name.startswith("nova_"), f"Recipe name should start with nova_: {recipe_name}")

            # Verify required fields exist
            self.assertIn("display_name", metadata, f"Missing display_name in {recipe_name}")
            self.assertIn("version", metadata, f"Missing version in {recipe_name}")
            self.assertIn("instance_types", metadata, f"Missing instance_types in {recipe_name}")

            # Verify field types - assertIsInstance checks if object is instance of a type
            self.assertIsInstance(metadata["display_name"], str, f"display_name must be a string in {recipe_name}")
            self.assertIsInstance(metadata["version"], str, f"version must be a string in {recipe_name}")
            self.assertIsInstance(metadata["instance_types"], list, f"instance_types must be a list in {recipe_name}")

            # Verify instance types are strings starting with ml.
            for instance_type in metadata["instance_types"]:
                self.assertIsInstance(instance_type, str, f"Instance type must be string in {recipe_name}")
                if instance_type:
                    self.assertTrue(
                        instance_type.startswith("ml."),
                        f"Instance type should start with ml. in {recipe_name}: {instance_type}",
                    )

        # Compare production against baseline
        if production_mapping != baseline_mapping:
            # Identify differences
            added_recipes = set(production_mapping.keys()) - set(baseline_mapping.keys())
            removed_recipes = set(baseline_mapping.keys()) - set(production_mapping.keys())
            modified_recipes = []

            for recipe_name in set(production_mapping.keys()) & set(baseline_mapping.keys()):
                if production_mapping[recipe_name] != baseline_mapping[recipe_name]:
                    modified_recipes.append(recipe_name)

            # Build detailed failure message
            changes = []
            if added_recipes:
                changes.append(f"\nAdded recipes ({len(added_recipes)}):")
                for recipe in sorted(added_recipes):
                    changes.append(f"  + {recipe}")

            if removed_recipes:
                changes.append(f"\nRemoved recipes ({len(removed_recipes)}):")
                for recipe in sorted(removed_recipes):
                    changes.append(f"  - {recipe}")

            if modified_recipes:
                changes.append(f"\nModified recipes ({len(modified_recipes)}):")
                for recipe in sorted(modified_recipes):
                    baseline_meta = baseline_mapping.get(recipe, {})
                    production_meta = production_mapping.get(recipe, {})
                    changes.append(f"  ~ {recipe}:")
                    if baseline_meta.get("display_name") != production_meta.get("display_name"):
                        changes.append(
                            f"      display_name: '{baseline_meta.get('display_name')}' -> '{production_meta.get('display_name')}'"
                        )
                    if baseline_meta.get("version") != production_meta.get("version"):
                        changes.append(
                            f"      version: '{baseline_meta.get('version')}' -> '{production_meta.get('version')}'"
                        )
                    if baseline_meta.get("instance_types") != production_meta.get("instance_types"):
                        changes.append(
                            f"      instance_types: {baseline_meta.get('instance_types')} -> {production_meta.get('instance_types')}"
                        )

            change_summary = "\n".join(changes)

            # Fail with instructions
            self.fail(
                f"\n{'='*70}\n"
                f"nova_metadata.json is OUT OF SYNC with baseline!\n"
                f"{'='*70}\n"
                f"{change_summary}\n"
                f"\nSummary:\n"
                f"  Baseline recipes: {len(baseline_mapping)}\n"
                f"  Production recipes: {len(production_mapping)}\n"
                f"\nProduction file: {production_file}\n"
                f"Baseline file: {baseline_file}\n"
                f"\n"
                f"This test ensures no unintended changes are introduced to\n"
                f"nova_metadata.json. If the changes above are intentional:\n"
                f"\n"
                f"1. Review the changes carefully\n"
                f"2. Update the baseline to match:\n"
                f"   cp {production_file} {baseline_file}\n"
                f"3. Commit both files together\n"
                f"\n"
                f"If the changes are NOT intentional, revert nova_metadata.json\n"
                f"to match the baseline.\n"
                f"{'='*70}"
            )

    def test_nova_recipes_metadata_compliance(self):
        """
        Verify nova recipe metadata compliance:
        1. Every nova recipe file has a corresponding entry in nova_metadata.json
        2. No nova recipe YAML contains display_name, version, or instance_types fields

        These fields have been moved to nova_metadata.json for centralized management.
        """

        project_root = Path(__file__).parent.parent
        recipes_dir = project_root / "recipes_collection" / "recipes"
        metadata_file = project_root / "launcher" / "recipe_templatization" / "nova" / "nova_metadata.json"

        if not recipes_dir.exists():
            self.skipTest("Recipes directory not found")

        if not metadata_file.exists():
            self.fail(f"nova_metadata.json not found at: {metadata_file}")

        # Load metadata
        with open(metadata_file, "r") as f:
            metadata_mapping = json.load(f)

        # Find all nova recipe files
        nova_recipe_files = list(recipes_dir.rglob("nova_*.yaml"))
        self.assertGreater(len(nova_recipe_files), 0, "No nova recipe files found")

        # Check each recipe in one pass
        missing_metadata = []
        files_with_metadata_fields = []

        for recipe_file in nova_recipe_files:
            recipe_name = recipe_file.stem  # filename without .yaml

            # Check 1: Recipe has metadata entry
            if recipe_name not in metadata_mapping:
                missing_metadata.append(str(recipe_file.relative_to(project_root)))

            # Check 2: Recipe YAML doesn't have metadata fields
            try:
                with open(recipe_file, "r") as f:
                    recipe_data = yaml.safe_load(f)

                if recipe_data:
                    found_fields = []
                    if "display_name" in recipe_data:
                        found_fields.append("display_name")
                    if "version" in recipe_data:
                        found_fields.append("version")
                    if "instance_types" in recipe_data:
                        found_fields.append("instance_types")

                    if found_fields:
                        files_with_metadata_fields.append(
                            {"file": str(recipe_file.relative_to(project_root)), "fields": found_fields}
                        )
            except Exception:
                # Skip files that can't be parsed
                continue

        # Build failure message if any issues found
        errors = []

        if missing_metadata:
            errors.append(
                f"Found {len(missing_metadata)} recipe(s) WITHOUT metadata entries:\n"
                + "\n".join(f"  - {path}" for path in sorted(missing_metadata))
                + f"\n\nPlease add metadata entries to: {metadata_file}"
            )

        if files_with_metadata_fields:
            details = []
            for item in sorted(files_with_metadata_fields, key=lambda x: x["file"]):
                fields_str = ", ".join(item["fields"])
                details.append(f"  - {item['file']}\n    Fields: {fields_str}")

            errors.append(
                f"Found {len(files_with_metadata_fields)} recipe(s) WITH metadata fields:\n"
                + "\n".join(details)
                + f"\n\nThese fields (display_name, version, instance_types) should ONLY\n"
                f"exist in nova_metadata.json, not in recipe YAML files.\n"
                f"Please remove these fields from the recipe files."
            )

        if errors:
            self.fail(
                f"\n{'='*70}\n"
                f"Nova Recipe Metadata Compliance Issues\n"
                f"{'='*70}\n\n" + "\n\n".join(errors) + f"\n{'='*70}"
            )


def run_tests():
    """Run all tests and return the result."""
    import sys

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
