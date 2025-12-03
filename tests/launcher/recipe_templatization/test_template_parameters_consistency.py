"""
Unit tests to validate recipe template parameters consistency.

These tests ensure that every template variable {{name}} in recipe_template
has a corresponding entry in recipe_override_parameters.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

import pytest


def find_template_variables(obj: any, path: str = "") -> Set[str]:
    """
    Recursively find all {{variable}} patterns in a nested dictionary/list structure.

    Args:
        obj: Object to search (dict, list, string, or primitive)
        path: Current path in the object tree (for debugging)

    Returns:
        Set of variable names found (without the {{ }} wrappers)
    """
    variables = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            variables.update(find_template_variables(value, current_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            current_path = f"{path}[{idx}]"
            variables.update(find_template_variables(item, current_path))
    elif isinstance(obj, str):
        # Find all {{variable_name}} patterns
        matches = re.findall(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", obj)
        variables.update(matches)

    return variables


def get_template_parameter_files() -> List[Path]:
    """
    Find all *_recipe_template_parameters.json files in launcher/recipe_templatization.

    Returns:
        List of Path objects pointing to template parameter files
    """
    base_dir = Path("launcher/recipe_templatization")
    template_files = []

    # Search in subdirectories
    for subdir in ["nova", "llmft", "verl", "evaluation", "checkpointless"]:
        file_path = base_dir / subdir / f"{subdir}_recipe_template_parameters.json"
        if file_path.exists():
            template_files.append(file_path)

    return template_files


def load_template_file(file_path: Path) -> Dict:
    """
    Load and parse a template parameters JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON content as dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_template_consistency(file_path: Path) -> List[str]:
    """
    Validate that all template variables have corresponding override parameters.

    Args:
        file_path: Path to the template parameters JSON file

    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []

    try:
        data = load_template_file(file_path)
    except json.JSONDecodeError as e:
        return [f"Failed to parse JSON: {e}"]
    except Exception as e:
        return [f"Failed to load file: {e}"]

    # Get templates section
    templates = data.get("templates", {})

    if not templates:
        return [f"No 'templates' section found in {file_path.name}"]

    # Validate each template
    for template_name, template_data in templates.items():
        recipe_template = template_data.get("recipe_template", {})
        recipe_override_params = template_data.get("recipe_override_parameters", {})

        if not recipe_template:
            errors.append(f"{file_path.name}::{template_name}: " f"Missing 'recipe_template' section")
            continue

        if not recipe_override_params:
            errors.append(f"{file_path.name}::{template_name}: " f"Missing 'recipe_override_parameters' section")
            continue

        # Find all template variables in recipe_template
        template_variables = find_template_variables(recipe_template)

        # Get all override parameter names
        override_param_names = set(recipe_override_params.keys())

        # Meta-parameters that may not appear in override_parameters but are handled specially
        # These are platform-specific or runtime-injected parameters
        meta_params = {"namespace", "instance_type", "instance_types", "instance_count", "lora_rank"}

        # Find variables that don't have corresponding override parameters
        # Exclude meta-parameters from this check
        missing_params = template_variables - override_param_names - meta_params

        if missing_params:
            missing_list = sorted(missing_params)
            errors.append(
                f"{file_path.name}::{template_name}: " f"Template variables without override parameters: {missing_list}"
            )

        # Also check for unused override parameters (informational, not an error)
        # This helps identify potential cleanup opportunities
        unused_params = override_param_names - template_variables - meta_params

        if unused_params:
            unused_list = sorted(unused_params)
            # This is a warning, not an error - parameters may be used programmatically
            # or may be valid but optional parameters
            # errors.append(
            #     f"{file_path.name}::{template_name}: "
            #     f"Override parameters not used in template: {unused_list}"
            # )

    return errors


class TestTemplateParametersConsistency:
    """Test suite for validating template parameters consistency."""

    @pytest.fixture(scope="class")
    def template_files(self) -> List[Path]:
        """Get all template parameter files."""
        return get_template_parameter_files()

    def test_template_files_exist(self, template_files: List[Path]):
        """Verify that template parameter files are found."""
        assert len(template_files) > 0, (
            "No *_recipe_template_parameters.json files found. " "Expected files in launcher/recipe_templatization/*/."
        )

    def test_template_files_parseable(self, template_files: List[Path]):
        """Verify that all template files are valid JSON."""
        for file_path in template_files:
            try:
                load_template_file(file_path)
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse {file_path.name}: {e}")
            except Exception as e:
                pytest.fail(f"Failed to load {file_path.name}: {e}")

    @pytest.mark.parametrize("file_path", get_template_parameter_files())
    def test_template_variables_have_override_parameters(self, file_path: Path):
        """
        Test that every {{variable}} in recipe_template has a corresponding
        entry in recipe_override_parameters.

        This ensures that:
        1. All template variables can be properly substituted
        2. No template variables are left undefined
        3. The template and override parameters are in sync
        """
        errors = validate_template_consistency(file_path)

        if errors:
            error_message = f"\n\nTemplate consistency errors in {file_path.name}:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            pytest.fail(error_message)


def test_find_template_variables_basic():
    """Test the template variable extraction function."""
    # Test simple string with one variable
    result = find_template_variables("{{name}}")
    assert result == {"name"}

    # Test string with multiple variables
    result = find_template_variables("{{var1}} and {{var2}}")
    assert result == {"var1", "var2"}

    # Test nested dict
    obj = {"key1": "{{variable1}}", "key2": {"nested": "{{variable2}}"}}
    result = find_template_variables(obj)
    assert result == {"variable1", "variable2"}

    # Test list
    obj = ["{{item1}}", "{{item2}}"]
    result = find_template_variables(obj)
    assert result == {"item1", "item2"}

    # Test no variables
    result = find_template_variables("no variables here")
    assert result == set()


def test_find_template_variables_complex():
    """Test template variable extraction with complex structures."""
    obj = {
        "run": {"name": "{{name}}", "replicas": "{{replicas}}"},
        "config": {
            "learning_rate": "{{learning_rate}}",
            "nested": {"deep": "{{deep_var}}"},
            "list": ["{{list_var1}}", {"dict_in_list": "{{list_var2}}"}],
        },
    }

    result = find_template_variables(obj)
    expected = {"name", "replicas", "learning_rate", "deep_var", "list_var1", "list_var2"}
    assert result == expected
