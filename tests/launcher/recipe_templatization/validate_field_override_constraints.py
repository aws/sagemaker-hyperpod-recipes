"""
Generic validation for field override constraints in launch.json files.

Validates that specified fields in launch.json files:
1. Are not templatized (no {{field}} format)
2. Match the default values from the original recipe

This validator is generic and can be used with any set of field names.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


def get_nested_value(data: Dict, path: str):
    """
    Get a value from nested dictionary using dot notation path.

    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "run.replicas" or "training_config.mlflow.tracking_uri")

    Returns:
        The value at the path, or None if not found
    """
    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None

    return current


def find_field_in_yaml(yaml_content: str, field_name: str) -> List[Tuple[int, str]]:
    """
    Find all occurrences of a field in YAML content and return line numbers and values.

    Args:
        yaml_content: The YAML content as a string
        field_name: The field name to search for

    Returns:
        List of tuples (line_number, field_value) for each occurrence
    """
    occurrences = []
    lines = yaml_content.split("\n")

    # Pattern to match field assignments (handles both direct and nested)
    # Matches: "field_name: value" or "field_name: '{{template}}'"
    pattern = rf"^\s*{re.escape(field_name)}:\s*(.+)$"

    for line_num, line in enumerate(lines, 1):
        match = re.search(pattern, line)
        if match:
            value = match.group(1).strip()
            occurrences.append((line_num, value))

    return occurrences


def validate_fields_not_overridden(
    launch_json: Dict,
    original_recipe_path: Path,
    fields_to_validate: Set[str],
    yaml_content_key: str = "training_recipe.yaml",
    validation_label: str = "Field Override",
) -> Tuple[bool, List[str]]:
    """
    Generic validator that checks specified fields are not templatized and match defaults.

    This function validates that specified fields in a launch.json:
    1. Are not templatized (don't contain {{field}} format)
    2. Match the default values from the original recipe

    Args:
        launch_json: Loaded launch.json data
        original_recipe_path: Path to the original recipe YAML file
        fields_to_validate: Set of field names to validate (e.g., DONT_OVERRIDE_IN_SMTJ)
        yaml_content_key: Key in launch_json containing the YAML content to check
                         (e.g., "training_recipe.yaml" for SM Jobs)
        validation_label: Label to use in error messages for context

    Returns:
        Tuple of (is_valid, list_of_errors)

    Example usage:
        # For SM Jobs Nova recipes with DONT_OVERRIDE_IN_SMTJ
        is_valid, errors = validate_fields_not_overridden(
            launch_json=my_launch_json,
            original_recipe_path=Path("recipe.yaml"),
            fields_to_validate=DONT_OVERRIDE_IN_SMTJ,
            yaml_content_key="training_recipe.yaml",
            validation_label="DONT_OVERRIDE_IN_SMTJ"
        )
    """
    errors = []

    # Get YAML content to validate
    yaml_content = launch_json.get(yaml_content_key, "")
    if not yaml_content:
        # No content with this key - skip validation (not applicable)
        return True, []

    # Load original recipe for comparison
    try:
        with open(original_recipe_path, "r") as f:
            original_recipe_content = yaml.safe_load(f)
    except Exception as e:
        errors.append(f"[{validation_label}] Failed to load original recipe: {e}")
        return False, errors

    # Get processed recipe from JSON if available (easier to parse)
    json_content_key = yaml_content_key.replace(".yaml", ".json")
    training_recipe_json = launch_json.get(json_content_key, {})

    if training_recipe_json and "recipes" in training_recipe_json:
        processed_recipe = training_recipe_json["recipes"]
    else:
        # Parse YAML if JSON not available
        try:
            processed_recipe = yaml.safe_load(yaml_content)
        except Exception as e:
            errors.append(f"[{validation_label}] Failed to parse {yaml_content_key}: {e}")
            return False, errors

    # Validate each field
    for field_name in fields_to_validate:
        # 1. Check if field is templatized in the YAML content
        occurrences = find_field_in_yaml(yaml_content, field_name)

        for line_num, value in occurrences:
            # Check for template syntax: {{...}}
            if "{{" in value and "}}" in value:
                errors.append(
                    f"[{validation_label}] Field '{field_name}' is templatized at line {line_num}: {value}. "
                    f"This field should not be templatized."
                )

        # 2. Check if field value matches the original recipe (if both exist)
        # Common paths where fields might appear
        field_paths = [
            f"run.{field_name}",
            f"training_config.mlflow.{field_name}",
            f"training_config.{field_name}",
            field_name,  # Top-level
        ]

        # Try to find the field in both original and processed recipes
        original_value = None
        processed_value = None

        for path in field_paths:
            if original_value is None:
                original_value = get_nested_value(original_recipe_content, path)
            if processed_value is None:
                processed_value = get_nested_value(processed_recipe, path)

        # Only validate value match if field exists in both recipes
        if original_value is not None and processed_value is not None:
            # Normalize values for comparison (handle different types)
            orig_str = str(original_value).strip()
            proc_str = str(processed_value).strip()

            if orig_str != proc_str:
                errors.append(
                    f"[{validation_label}] Field '{field_name}' value mismatch. "
                    f"Original: '{orig_str}', Processed: '{proc_str}'. "
                    f"Field should match the default value."
                )

    if errors:
        return False, errors

    logger.info(f"✓ PASS: All {validation_label} fields validated successfully")
    return True, []
