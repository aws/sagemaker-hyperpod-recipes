"""
Special Override Parameters Processor

Utility class to dynamically compute enum values for special override parameters
like global_batch_size, rollout, etc. based on their default values (which are set
from the recipe file during template application).

This replaces hardcoded per-recipe conditional_constraints with dynamic computation.
"""

from typing import Dict, List, Union


class SpecialOverrideParametersProcessor:
    """
    Processor for dynamically computing enum values for special override parameters.

    Each framework processor (VerlRecipeTemplateProcessor, LlmftRecipeTemplateProcessor, etc.)
    can instantiate this class with its framework type.

    Example:
        >>> processor = SpecialOverrideParametersProcessor(framework="verl")
        >>> updated_params = processor.process(override_parameters)
    """

    # Define potential values for parameters that need dynamic enum generation
    POTENTIAL_VALUES = {"global_batch_size": [8, 16, 32, 64, 128, 256, 512, 1024]}

    # Special parameters configuration organized by framework
    # "default" applies to all frameworks, framework-specific configs override or add to default
    SPECIAL_PARAMS_CONFIG = {
        "default": {
            "global_batch_size": "min_from_default",
        },
        "verl": {
            # Add verl-specific parameters here
        },
        "llmft": {
            # Add llmft-specific parameters here
        },
        "nova": {
            # Add nova-specific parameters here
        },
    }

    def __init__(self, framework: str = "default"):
        """
        Initialize the processor for a specific framework.

        Args:
            framework: Framework name ('verl', 'llmft', 'nova', or 'default')
        """
        self.framework = framework
        self.params_config = self._get_params_config()

    def _get_params_config(self) -> Dict[str, str]:
        """
        Get the combined params config for the framework.

        Merges default config with framework-specific config.

        Returns:
            Combined config dict mapping param names to computation types
        """
        config = dict(self.SPECIAL_PARAMS_CONFIG.get("default", {}))
        framework_config = self.SPECIAL_PARAMS_CONFIG.get(self.framework, {})
        config.update(framework_config)
        return config

    @staticmethod
    def compute_enum_from_min(
        default_value: Union[int, float],
        potential_values: List[Union[int, float]],
    ) -> List[Union[int, float]]:
        """
        Compute enum values starting from the default value.

        The default value becomes the minimum allowed value.

        Args:
            default_value: The default value (from recipe file)
            potential_values: List of all potential values (must be sorted ascending)

        Returns:
            List of allowed values (default_value and all values >= it)
        """
        return [v for v in potential_values if v >= default_value]

    @staticmethod
    def compute_enum_exact(
        default_value: Union[int, float],
        potential_values: List[Union[int, float]],
    ) -> List[Union[int, float]]:
        """
        Compute enum with only the exact default value.

        Args:
            default_value: The default value (from recipe file)
            potential_values: List of all potential values (unused, kept for interface consistency)

        Returns:
            List containing only the default value
        """
        return [default_value]

    def _compute_enum(
        self,
        computation_type: str,
        default_value: Union[int, float],
        potential_values: List[Union[int, float]],
    ) -> List[Union[int, float]]:
        """
        Compute enum based on computation type using pattern matching.

        Args:
            computation_type: Type of computation to perform
            default_value: The default value from recipe
            potential_values: List of potential enum values

        Returns:
            Computed enum list, or None if computation type is unknown
        """
        match computation_type:
            case "min_from_default":
                return self.compute_enum_from_min(default_value, potential_values)
            case "exact":
                return self.compute_enum_exact(default_value, potential_values)
            case _:
                return None

    def process_single_param(self, param_name: str, param_config: Dict) -> None:
        """
        Process a single special override parameter to compute dynamic enum values.

        Called by the base processor for params that don't have conditional_constraints.
        Modifies param_config in place.

        Args:
            param_name: Name of the parameter (e.g., 'global_batch_size', 'rollout')
            param_config: Parameter configuration dict (modified in place)

        Example:
            >>> processor = SpecialOverrideParametersProcessor(framework="verl")
            >>> param_config = {"type": "integer", "default": 256, "enum": [...]}
            >>> processor.process_single_param("global_batch_size", param_config)
            >>> param_config["enum"]
            [256, 512, 1024]  # Computed from default
        """
        # Check if this param is configured for processing
        computation_type = self.params_config.get(param_name)
        if computation_type is None:
            return

        default_value = param_config.get("default")
        if default_value is None:
            return

        # Skip parameters that use min/max instead of enum
        if "enum" not in param_config:
            return

        potential_values = self.POTENTIAL_VALUES.get(param_name)
        if potential_values is None:
            return

        # Compute new enum
        new_enum = self._compute_enum(computation_type, default_value, potential_values)
        if new_enum is None:
            return

        # Respect max constraint if present
        max_value = param_config.get("max")
        if max_value is not None:
            new_enum = [v for v in new_enum if v <= max_value]

        # Respect min constraint if present
        min_value = param_config.get("min")
        if min_value is not None:
            new_enum = [v for v in new_enum if v >= min_value]

        # Update the enum in place
        param_config["enum"] = new_enum

    def process(self, override_parameters: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Process all special override parameters to compute dynamic enum values.

        Iterates through all params and processes those that are configured.

        Args:
            override_parameters: The recipe_override_parameters dict with updated defaults

        Returns:
            Updated override_parameters with dynamically computed enums

        Note:
            This method is kept for backwards compatibility but the preferred approach
            is to use process_single_param() from the base processor's routing loop.
        """
        for param_name in self.params_config.keys():
            if param_name not in override_parameters:
                continue

            param_config = override_parameters[param_name]

            # Skip params with conditional_constraints (they have explicit constraints)
            if "conditional_constraints" in param_config:
                continue

            self.process_single_param(param_name, param_config)

        return override_parameters
