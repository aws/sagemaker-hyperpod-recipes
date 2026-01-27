"""
Recipe and Override Parameters Validators for Launch.json

These validators ensure:
1. All recipe fields are present in training.yaml
2. Override parameters are properly templatized
3. Default values match original recipe values
4. Override parameter constraints are internally consistent
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def validate_no_omegaconf_artifacts(training_yaml_content: str, recipe_file_path: str) -> List[str]:
    """
    Validate that training.yaml has no OmegaConf artifacts from failed resolution.

    Checks for patterns like:
    - ${...} (unresolved interpolations)
    - oc.select(...)
    - OmegaConf.select(...)
    - _target_: (Hydra instantiation patterns that shouldn't be in final output)

    Args:
        training_yaml_content: The content of training.yaml from launch.json as a string
        recipe_file_path: Path to the recipe file for error reporting

    Returns:
        List of validation errors
    """
    errors = []

    # Patterns that indicate unresolved OmegaConf operations
    omegaconf_patterns = [
        ("${", "Unresolved OmegaConf interpolation"),
        ("oc.select", "OmegaConf select artifact"),
        ("OmegaConf.select", "OmegaConf select artifact"),
        ("omegaconf.select", "OmegaConf select artifact (lowercase)"),
    ]

    for pattern, description in omegaconf_patterns:
        if pattern in training_yaml_content:
            # Find line numbers for better error reporting
            lines_with_pattern = []
            for line_num, line in enumerate(training_yaml_content.split("\n"), 1):
                if pattern in line:
                    lines_with_pattern.append(f"Line {line_num}: {line.strip()[:80]}")
                    if len(lines_with_pattern) >= 3:  # Show max 3 examples
                        break

            error_msg = f"{description} '{pattern}' found in training.yaml:\n  " + "\n  ".join(lines_with_pattern)
            errors.append(error_msg)

    return errors


def validate_recipe_fields_presence_in_training_yaml(
    original_recipe: Dict, training_yaml_content: str, recipe_file_path: str
) -> List[str]:
    """
    Validate that all fields from the original recipe file are present in training.yaml.

    This validation uses STRING-BASED comparison instead of YAML parsing because:
    - training.yaml contains template variables like {{name}} which aren't valid YAML
    - PPO recipes have multiple YAML documents separated by --- which complicate parsing

    Args:
        original_recipe: The original non-templatized recipe configuration (as dict)
        training_yaml_content: The content of training.yaml from launch.json as a string
        recipe_file_path: Path to the recipe file for error reporting

    Returns:
        List of validation errors
    """
    errors = []

    # Exception list: Fields that exist in recipe but not in training config files
    # Nova PPO: Replica counts and top-level sections don't appear as keys in training-config.yaml
    # - Replica counts: meta-configuration for K8s job creation
    # - PPO sections: become separate ConfigMaps, content is merged into unified training_config
    # Hydra: defaults directive is processed during composition and not in final output
    nova_ppo_exception_fields = {
        # Hydra defaults directive - processed during composition, not in output
        "defaults",
        # Replica count meta-fields
        "actor_train_replicas",
        "rm_replicas",
        "cm_replicas",
        "am_replicas",
        "actor_generation_replicas",
        # Top-level PPO section names (split into separate ConfigMaps)
        "lora_rank",
    }

    special_mappings = {"ppo_reward": "ppo_rm", "ppo_critic": "ppo_cm", "ppo_actor_generation": "ppo_actor_gen"}

    def check_field_presence_as_string(original_dict, path=""):
        """
        Recursively check that all field keys from original recipe appear as strings in training.yaml.
        Uses simple string search since YAML parsing fails on template variables.
        """
        local_errors = []

        if not isinstance(original_dict, dict):
            return local_errors

        for key, original_value in original_dict.items():
            current_path = f"{path}.{key}" if path else key

            # Skip exception fields (recipe-specific meta fields)
            if key in nova_ppo_exception_fields:
                continue

            # Check if key exists as "key:" pattern in the YAML string (as a key)
            key_pattern = f"{key}:"
            key_found_as_key = key_pattern in training_yaml_content

            # Fallback: Check if key exists anywhere in the string (could be a value)
            key_found_as_value = str(key) in training_yaml_content

            # Fail only if key not found in either form
            if not key_found_as_key and not key_found_as_value:
                if key not in special_mappings or special_mappings[key] not in training_yaml_content:
                    local_errors.append(
                        f"Field key '{key}' (path: '{current_path}') not found in training.yaml (neither as key nor value)"
                    )
                continue

            # If value is a dict, recurse to check nested fields
            if isinstance(original_value, dict):
                local_errors.extend(check_field_presence_as_string(original_value, current_path))

        return local_errors

    # Start checking from root level
    errors.extend(check_field_presence_as_string(original_recipe))

    return errors


def validate_override_parameters_templatization(
    override_params: Dict, training_yaml_content: str, platform: str, recipe_file_path: str, recipe_type: str = None
) -> List[str]:
    """
    Validate that override parameters are properly templatized in training.yaml.

    For each parameter in recipe_override_parameters, this checks that it appears
    in training.yaml as "{{parameter_name}}" (the template variable format).

    Args:
        override_params: The recipe_override_parameters dict from launch.json
        training_yaml_content: The content of training.yaml from launch.json as a string
        platform: Platform type ("k8s" or "sm_jobs")
        recipe_file_path: Path to the recipe file for error reporting
        recipe_type: Recipe type ("nova", "llmft", "verl", "evaluation") - if not provided, inferred from recipe_file_path

    Returns:
        List of validation errors
    """
    errors = []

    # Infer recipe type from file path if not provided
    if recipe_type is None:
        recipe_file_path_lower = recipe_file_path.lower()
        if "nova" in recipe_file_path_lower:
            recipe_type = "nova"
        elif "llmft" in recipe_file_path_lower:
            recipe_type = "llmft"
        elif "verl" in recipe_file_path_lower:
            recipe_type = "verl"
        elif "eval" in recipe_file_path_lower:
            recipe_type = "evaluation"
        else:
            recipe_type = "unknown"

    # Platform-specific skip parameters
    # K8s-specific parameters
    k8s_skip_params = {
        "namespace",  # K8s specific, not in recipe content
    }

    # SM Jobs-specific parameters
    sm_jobs_skip_params = {
        "instance_count",  # SM Jobs specific
    }

    # Recipe-type specific skip parameters
    # replicas is ONLY in Nova recipes (in run.replicas), not in LLMFT/VERL/open_source
    non_nova_skip_params = {
        "replicas",  # Only present in Nova recipes
    }

    # Common skip parameters (apply to all)
    common_skip_params = {"instance_type", "lora_rank", "max_context_length"}  # Not always in recipe content

    # Build skip set based on platform and recipe type
    skip_params = common_skip_params.copy()

    # Add platform-specific skips
    if platform == "k8s":
        skip_params |= k8s_skip_params
    elif platform == "sm_jobs":
        skip_params |= sm_jobs_skip_params

    # Add recipe-type-specific skips
    # For non-Nova recipes, skip 'replicas' since it's not in their recipe structure
    if recipe_type in ["llmft", "verl", "evaluation", "unknown"]:
        skip_params |= non_nova_skip_params

    for param_name in override_params.keys():
        # Skip parameters that aren't expected in training.yaml
        if param_name in skip_params:
            continue

        # Check if template variable appears in training.yaml
        template_var = "{{" + f"{param_name}" + "}}"

        if template_var not in training_yaml_content:
            errors.append(
                f"Override parameter '{param_name}' not found as template variable '{template_var}' in training.yaml"
            )

    return errors


def validate_override_parameter_default_values(
    original_recipe: Dict, override_params: Dict, recipe_template: Dict, recipe_file_path: str
) -> List[str]:
    """
    Validate that default values in override parameters match the original recipe.

    This ensures that the default values specified in recipe_override_parameters
    match the actual values from the non-templatized recipe file.

    Uses the recipe_template to discover the actual paths of parameters in the recipe.

    Args:
        original_recipe: The original non-templatized recipe configuration
        override_params: The recipe_override_parameters dict from launch.json
        recipe_template: The recipe_template section showing structure with {{variables}}
        recipe_file_path: Path to the recipe file for error reporting

    Returns:
        List of validation errors
    """
    errors = []

    def find_template_variable_path(template_dict, target_var, path=[]):
        """
        Recursively find the path to a template variable in the recipe template.
        Returns the path as a list of keys.
        """
        if isinstance(template_dict, dict):
            for key, value in template_dict.items():
                current_path = path + [key]

                if isinstance(value, str) and value == f"{{{{{target_var}}}}}":
                    return current_path

                result = find_template_variable_path(value, target_var, current_path)
                if result:
                    return result

        return None

    def get_nested_value(d: Dict, path: List[str]):
        """Get value from nested dict using path list."""
        current = d
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    for param_name, param_config in override_params.items():
        # HARD RULE: default must ALWAYS be present
        if "default" not in param_config:
            errors.append(f"Override parameter '{param_name}' missing required 'default' field")
            continue

        default_value = param_config["default"]
        param_type = param_config.get("type")

        # Skip empty string defaults (often used for paths that must be provided)
        if default_value == "":
            continue

        # Find the path to this parameter in the recipe using the template
        recipe_path = find_template_variable_path(recipe_template, param_name)

        if recipe_path is None:
            # Can't validate if parameter not in template
            continue

        # Get the actual value from the original recipe
        actual_value = get_nested_value(original_recipe, recipe_path)

        if actual_value is None:
            # Field doesn't exist in recipe, skip validation
            continue

        # Compare default with actual value
        # For floats, NO TOLERANCE - must be exact match
        if not values_match_exact(default_value, actual_value, param_type):
            errors.append(
                f"Override parameter '{param_name}' default value {default_value} "
                f"doesn't match recipe value {actual_value} at path {'.'.join(recipe_path)}"
            )

    return errors


def values_match_exact(default_value, actual_value, param_type: Optional[str] = None) -> bool:
    """
    Check if two values match exactly with NO tolerance.

    Args:
        default_value: Default value from override parameters
        actual_value: Actual value from recipe
        param_type: Parameter type hint (integer, float, string, etc.)

    Returns:
        True if values match exactly
    """
    # Direct equality check
    if default_value == actual_value:
        return True

    # Type-aware comparison for numeric types
    if param_type in ["integer", "number", "float"]:
        try:
            # For integers, allow int/float equivalence if they're mathematically equal
            return float(default_value) == float(actual_value)
        except (ValueError, TypeError):
            return False

    return False


def validate_override_parameter_constraints(override_params: Dict) -> List[str]:
    """
    Validate override parameter constraints for internal consistency.

    HARD RULES enforced:
    - conditional_constraints should NEVER be present in launch.json (only in template files)
    - For numeric types (integer/number/float):
      * min < max (if both present)
      * default >= min (if min present)
      * default <= max (if max present)
      * min, max, default are never None when they should exist
    - For all types:
      * default value is in enum list (if enum present)
      * For numeric enums: all values within min/max range
      * For string enums: default is one of the enum values

    Args:
        override_params: The recipe_override_parameters dict from launch.json

    Returns:
        List of validation errors
    """
    errors = []

    for param_name, param_config in override_params.items():
        # CRITICAL: conditional_constraints should NEVER appear in launch.json
        # They should only be in *_recipe_template_parameters.json and resolved before launch.json generation
        if "conditional_constraints" in param_config:
            errors.append(
                f"Parameter '{param_name}': 'conditional_constraints' found in launch.json - "
                f"should have been resolved during template processing"
            )

        param_type = param_config.get("type")
        default_value = param_config.get("default")
        required = param_config.get("required", False)
        min_value = param_config.get("min", None)
        max_value = param_config.get("max", None)
        enum_values = param_config.get("enum", None)
        null_exceptions = {"data_path", "validation_data_path", "resume_from_path"}
        # HARD RULE: default must NEVER be None/null
        if default_value is None:
            if param_name not in null_exceptions:
                errors.append(
                    f"Parameter '{param_name}': default value is None/null - all parameters must have a non-null default"
                )
            # Continue to check other constraints

        # Validate constraints for numeric types only
        if param_type in ["integer", "number", "float"]:
            # HARD RULE: For numeric types, if min/max exist, they cannot be None
            if min_value is not None and max_value is not None:
                # HARD RULE: min < max
                if not (min_value < max_value):
                    errors.append(
                        f"Parameter '{param_name}': min value {min_value} must be less than max value {max_value}"
                    )

            # HARD RULE: default >= min (if min present and default not None)
            if min_value is not None and default_value is not None:
                if default_value < min_value:
                    errors.append(f"Parameter '{param_name}': default value {default_value} must be >= min {min_value}")

            # HARD RULE: default <= max (if max present and default not None)
            if max_value is not None and default_value is not None:
                if default_value > max_value:
                    errors.append(f"Parameter '{param_name}': default value {default_value} must be <= max {max_value}")

            # Validate enum values are within min/max range (if all specified)
            if enum_values is not None:
                if min_value is not None or max_value is not None:
                    for enum_val in enum_values:
                        if min_value is not None and enum_val < min_value:
                            errors.append(
                                f"Parameter '{param_name}': enum value {enum_val} is less than min {min_value}"
                            )

                        if max_value is not None and enum_val > max_value:
                            errors.append(
                                f"Parameter '{param_name}': enum value {enum_val} is greater than max {max_value}"
                            )

        # Validate default value is in enum list (for all types)
        if enum_values is not None and default_value is not None and default_value != "":
            if param_type in ["integer", "number", "float", "string"]:
                # For numeric and string enums, default must be in the list
                if default_value not in enum_values:
                    errors.append(
                        f"Parameter '{param_name}': default value {default_value} not in enum list {enum_values}"
                    )
            elif param_type == "array":
                # For array types, if enum is specified, default should be a subset
                if not isinstance(default_value, list):
                    errors.append(
                        f"Parameter '{param_name}': default value must be array, got {type(default_value).__name__}"
                    )
                else:
                    if all([i == j for i, j in zip(sorted(enum_val) != sorted(default_value))]):
                        errors.append(
                            f"Parameter '{param_name}': default array value '{default_value}' not equal to enum list {enum_values}"
                        )

    return errors
