"""
Utility functions for template processing and manipulation.
"""


def remove_quotes_from_numeric_params(template_content: str, override_spec: dict) -> str:
    """
    Remove quotes from numeric template variables in Jinja template.

    This ensures numeric parameters render as numbers (not strings) in YAML
    by removing quotes around template variables before substitution.

    Args:
        template_content: The Jinja template content
        override_spec: Dictionary mapping parameter names to their specifications,
                      including 'type' field indicating parameter type

    Returns:
        Modified template content with quotes removed from numeric variables
    """

    # Handle None or empty override_spec gracefully
    if not override_spec:
        return template_content

    for param_name, param_def in override_spec.items():
        param_type = param_def.get("type", "").lower()

        # Only process numeric types (integer, float, number)
        if param_type in ["integer", "float", "number"]:
            # Template variable format: {{param_name}}
            template_var = f"{{{{{param_name}}}}}"

            # Remove single quotes
            quoted_single = f"'{template_var}'"
            if quoted_single in template_content:
                template_content = template_content.replace(quoted_single, template_var)

            # Remove double quotes
            quoted_double = f'"{template_var}"'
            if quoted_double in template_content:
                template_content = template_content.replace(quoted_double, template_var)

    return template_content
