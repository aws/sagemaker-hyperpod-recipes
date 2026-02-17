#!/usr/bin/env python3
import copy
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Optional

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils.recipe_utils import load_recipe_with_hydra

from ..nemo.constants import ROOT_DIR
from .process_special_override_parameters import SpecialOverrideParametersProcessor

logger = logging.getLogger(__name__)


class ServerlessMeteringType(Enum):
    TOKEN_BASED = "Token-based"
    HOURLY = "Hourly"


class BaseRecipeTemplateProcessor(ABC):
    """Abstract base class for recipe template processors."""

    # Framework type for special parameter processing. Subclasses should override this.
    framework_type: str = "default"

    def __init__(self, staging_cfg: dict):
        self.staging_cfg = staging_cfg
        self.template_data = None
        self.matched_template = None
        self.matched_template_group = None
        self.recipe_override_parameters = None
        self.regional_parameters = None
        self.recipe_jumpstart_model_id_mapping = None
        self.avoid_default_val_update_attributes = {"results_directory"}
        self._load_template()

    @abstractmethod
    def _load_template(self):
        """Load template files. Must be implemented by subclasses."""

    @abstractmethod
    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for the recipe. Must be implemented by subclasses."""

    @abstractmethod
    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate recipe metadata. Must be implemented by subclasses."""

    # Finds the matching template for a given recipe file and then apply templatization to it.
    # Returns the templatized recipe as an output
    def process_recipe(self, recipe_file_path: str = None, output_path: str = None, dont_override_list=None) -> dict:
        """Process recipe file with template matching."""
        recipe_templates = self.template_data["templates"]
        self.matched_template_group = self.get_recipe_template(self.staging_cfg, recipe_templates, recipe_file_path)

        if self.matched_template_group:
            self.matched_template = self.matched_template_group["recipe_template"]
            self.recipe_override_parameters = self.matched_template_group["recipe_override_parameters"]
            result = self.apply_template(self.staging_cfg, self.matched_template, dont_override_list)

            # Save result
            if output_path:
                with open(output_path, "w") as f:
                    yaml_content = yaml.dump(result, default_flow_style=False)
                    f.write(yaml_content)
            return result

        raise ValueError("No matching template found for the YAML structure")

    # Apply template first flattens the template and then overrides the values
    # with the respective override parameters from the *_recipe_template_parameters.json
    def apply_template(self, yaml_data: dict, template: dict, dont_override_list=None) -> dict:
        """Apply template substitutions to YAML data."""
        flattened = self.flatten_template(template, "all")

        def _substitute(data, path=""):
            if isinstance(data, (dict, DictConfig)):
                result = {}
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    if current_path in flattened:
                        if dont_override_list and key in dont_override_list:
                            result[key] = _substitute(value, current_path)
                            continue
                        current_value = data[key]
                        result[key] = flattened[current_path]
                        template_override = flattened[current_path][
                            2:-2
                        ]  # Get the override key without the '{{' and '}}'
                        if template_override in self.recipe_override_parameters:
                            # Dont update the default value for override parameter if it is in deny list
                            if template_override in self.avoid_default_val_update_attributes:
                                continue
                            else:
                                # Update default value for the override attribute with the value in the recipe file
                                self.recipe_override_parameters[template_override]["default"] = (
                                    OmegaConf.to_container(current_value, resolve=True)
                                    if isinstance(current_value, (ListConfig))
                                    else current_value
                                )
                    else:
                        result[key] = _substitute(value, current_path)
                return result
            return data

        return _substitute(yaml_data)

    # Flattening the template essentially converts the json nesting into dot notation
    # Our recipe files can have any level of nesting. To make it easier to traverse this nesting during
    # templatization, the nesting is converted into dot notation like in the example below.
    # Ex: {a:{b:{c:"value"}}} -> a.b.c = value
    def flatten_template(self, template_dict: dict, section_type: str = "all") -> dict:
        """Flatten template dict, optionally filtering by section type."""
        flattened = {}

        def _flatten(obj, path="", in_section=None):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {"Required", "Optional"}:
                        _flatten(value, path, key)
                    else:
                        new_path = f"{path}.{key}" if path else key
                        if isinstance(value, dict):
                            _flatten(value, new_path, in_section)
                        else:
                            if section_type == "all" or in_section == section_type:
                                flattened[new_path] = value

        _flatten(template_dict)
        return flattened

    # Returns the metadata, recipe_override_parameters, regional_parameters for the launch_json
    def get_additional_data(self, recipe_file_path: str) -> list:
        """Get additional data including metadata, resolved override parameters, and regional parameters."""
        logger.info(f"Found recipe file path: {recipe_file_path}")

        # Handle case where recipe_file_path is None
        if recipe_file_path is None:
            return [{}, {}, {}]

        # Generate initial metadata using template helpers (without instance types)
        recipe_metadata = self.get_recipe_metadata(recipe_file_path)

        # Check container availability before proceeding
        if not self._check_container_availability(recipe_metadata):
            return None

        # Resolve conditional constraints using the generated metadata
        resolved_override_parameters = self._resolve_constraints_using_metadata(recipe_metadata)

        # Update the instance variable so it's used in the main processing
        self.recipe_override_parameters = resolved_override_parameters

        # Fetch regional parameters for the specific recipe only
        recipe_name = recipe_metadata.get("Name")
        regional_parameters = self._get_regional_parameters(recipe_name) if recipe_name else {}

        return [recipe_metadata, resolved_override_parameters, regional_parameters]

    # Containers are regional objects. They are only present in some regions and support only some platforms
    # Information related to that is present in the *_regional_parameters.json
    def _check_container_availability(self, recipe_metadata: dict) -> bool:
        """Check if containers are available for the given recipe name and platform."""

        # Get recipe name from metadata
        recipe_name = recipe_metadata.get("Name")
        if not recipe_name:
            logger.error("No recipe name found in metadata, skipping container availability check")
            return False

        try:
            regional_parameters = self._get_regional_parameters(recipe_name)
            if regional_parameters == {}:
                logging.error(
                    "No regional parameters found, skipping container availability check and launch json generation"
                )
                return False

            # Check for container availability based on platform
            container_available = False
            if self.platform == "k8s":
                container_available = "hp_eks_regional_ecr_uri" in regional_parameters
            elif self.platform == "sm_jobs":
                container_available = "smtj_regional_ecr_uri" in regional_parameters
            else:
                # For other platforms, check if smtj_regional_ecr_uri exists (as it's added by default)
                container_available = "smtj_regional_ecr_uri" in regional_parameters

            if not container_available:
                logging.error(f"No container image found for platform {self.platform}, skipping launch json generation")
                return False

        except KeyError as e:
            logging.error(f"Regional parameter matching failed: {e}, skipping launch json generation")
            return False

        return True

    # Fetches the regional parameters from *_regional_parameters.json for a given platform and recipe_name
    def _get_regional_parameters(self, recipe_name: str) -> dict:
        """Get regional parameters for the given recipe name."""

        regional_parameters = {}
        common_regional_parameters = self.regional_parameters.get("all", {})
        regional_parameter_option = None
        for recipe_name_pattern in self.regional_parameters:
            if recipe_name_pattern in recipe_name:
                regional_parameter_option = self.regional_parameters[recipe_name_pattern]
                break
        if not regional_parameter_option:
            raise KeyError(f"Regional parameter option not found for recipe {recipe_name}")
        if self.platform == "k8s":
            if regional_parameter_option.get(self.platform, None):
                for regional_parameter in regional_parameter_option.get("k8s", {}):
                    if regional_parameter == "container_image":
                        # Get regional parameter values for different pipeline stages (beta, gamma etc)
                        regional_parameters["hp_eks_regional_ecr_uri"] = regional_parameter_option[self.platform][
                            regional_parameter
                        ]
                    else:
                        regional_parameters[regional_parameter] = regional_parameter_option[self.platform][
                            regional_parameter
                        ]

                # Add the common regional parameters for K8s
                for common_regional_parameter in common_regional_parameters.get("k8s", {}):
                    regional_parameters[common_regional_parameter] = common_regional_parameters["k8s"][
                        common_regional_parameter
                    ]
            else:
                logging.error(f"No supported regional parameters for platform: {self.platform}")
                return regional_parameters

        # By default even in the case of hyperpod as a platform
        for regional_parameter in regional_parameter_option.get("sm_jobs", {}):
            if regional_parameter == "container_image":
                regional_parameters["smtj_regional_ecr_uri"] = regional_parameter_option["sm_jobs"][regional_parameter]
            else:
                regional_parameters[regional_parameter] = regional_parameter_option["sm_jobs"][regional_parameter]
        # Add the common regional parameters for sm_jobs
        for common_regional_parameter in common_regional_parameters.get("sm_jobs", {}):
            regional_parameters[common_regional_parameter] = common_regional_parameters["sm_jobs"][
                common_regional_parameter
            ]

        return regional_parameters

    def extract_sequence_length(self, recipe_name: str) -> Optional[str]:
        """Extract sequence length from recipe name."""
        match = re.search(r"seq(\d+)k", recipe_name)
        if match:
            return f"{match.group(1)}K"
        return None

    def format_sequence_length(self, seq_length: int) -> str:
        """
        Format integer sequence length to string format.
        Rounds down to the nearest valid sequence length if exact match not found.

        Args:
            seq_length: Integer sequence length (e.g., 1024, 4096, 16384, 10240)

        Returns:
            Formatted sequence length string (e.g., "1K", "4K", "8K")
            One of: "1K" | "2K" | "4K" | "8K" | "16K" | "32K" | "64K" | "128K"

        Examples:
            1024 -> "1K"
            4096 -> "4K"
            10240 -> "8K" (rounded down)
            16384 -> "16K"
            100000 -> "64K" (rounded down)
        """
        # Valid sequence lengths mapping
        valid_lengths = {
            1024: "1K",
            2048: "2K",
            4096: "4K",
            8192: "8K",
            16384: "16K",
            32768: "32K",
            65536: "64K",
            131072: "128K",
        }

        # If exact match, return it
        if seq_length in valid_lengths:
            return valid_lengths[seq_length]

        # Otherwise, round down to the nearest valid length
        sorted_lengths = sorted(valid_lengths.keys())

        # Find the largest valid length that is less than seq_length
        for length in reversed(sorted_lengths):
            if seq_length > length:
                return valid_lengths[length]

        # If smaller than all valid lengths, return the minimum
        return valid_lengths[sorted_lengths[0]]

    def extract_model_size(self, recipe_name: str) -> Optional[str]:
        """Extract model size from recipe name."""
        # Look for patterns like 8b, 70b, 405b, etc.
        match = re.search(r"(\d+)b", recipe_name)
        if match:
            return f"{match.group(1)}B"
        return None

    # Constraint resolution involves resolving values for the sub-fields in overridable parameters. Examples are included
    # in the _comment in the respective *_recipe_template_parameters.json. We use info in metadata to resolve constraints.
    # Example conditional_constraint:- overridable_parameter:{conditional_constraints:{sub-field : {metadata-field-key: {metadata-field-value: "resolved value 1", ....}}}}
    # Example: instance_type:{conditional_constraints:{'enum':{'Model_ID':{nova-micro: p5, nova-pro: g6}}}}
    # The general workflow is as follows:-
    # - Get a overridable parameter that has conditional_constraints
    # - Get the sub-field that has constraints
    # - Get the metadata key upon which the sub-field is constrained upon
    # - Find the metadata value from the recipe file to resolve the constraint
    # - Resolve constraints based on the metadata value and delete the conditional_constraints section
    #
    # Why do we need conditional constraints? Helps us use a single template for multiple recipe files with overlapping recipe contents
    def _resolve_constraints_using_metadata(self, metadata: dict) -> dict:
        """Resolve conditional constraints using already generated metadata.

        For each parameter:
        - If it has conditional_constraints: resolve using metadata
        - Else if it's a special param (global_batch_size, rollout): use dynamic computation

        This ensures params with explicit constraints use those constraints,
        while params without constraints get dynamic computation based on their default values.
        """
        if not self.recipe_override_parameters:
            return {}

        # Create a copy of the parameters to modify
        resolved_parameters = copy.deepcopy(self.recipe_override_parameters)

        logger.info(f"Resolving constraints using metadata: {metadata}")

        # Initialize special params processor for params without conditional_constraints
        special_params_processor = SpecialOverrideParametersProcessor(framework=self.framework_type)
        special_param_names = set(special_params_processor.params_config.keys())

        # Process each parameter - route to appropriate handler
        for param_name, param_config in resolved_parameters.items():
            if "conditional_constraints" in param_config:
                # Route 1: Has conditional_constraints - resolve using metadata
                self._resolve_conditional_constraints(param_name, param_config, metadata)
            elif param_name in special_param_names:
                # Route 2: Special param without conditional_constraints - use dynamic computation
                special_params_processor.process_single_param(param_name, param_config)

        logging.info("Constraints resolved")
        return resolved_parameters

    def _resolve_conditional_constraints(self, param_name: str, param_config: dict, metadata: dict) -> None:
        """Resolve conditional constraints for a single parameter using metadata.

        Args:
            param_name: Name of the parameter
            param_config: Parameter configuration dict (modified in place)
            metadata: Recipe metadata dict
        """
        conditional_constraints = param_config["conditional_constraints"]

        for params_to_be_changed, constraint_map in conditional_constraints.items():
            if len(constraint_map) > 1:
                raise ValueError(
                    f"Multiple constraints found for {param_name} : {params_to_be_changed}. Only one constraint is allowed."
                )
            condition_key, resolver_map = next(iter(constraint_map.items()))
            if condition_key not in metadata:
                raise KeyError(f"Condition key {condition_key} not found in metadata")

            resolved_values = None

            condition_value = metadata[condition_key]

            # Look for exact match first, then fall back to "default"
            for constraint in resolver_map:
                if constraint in condition_value:
                    resolved_values = resolver_map[constraint]
                    break

            if not resolved_values:
                resolved_values = resolver_map["default"]

            # Resolve the constraints
            param_config[params_to_be_changed] = resolved_values

        # Cleanup the conditional constraints section from the override_parameters
        del param_config["conditional_constraints"]

    def get_recipe_name_from_path(self, recipe_file_path: str) -> str:
        """
        Extract recipe name from file path with null checks.

        :param recipe_file_path: Path to the recipe file
        :return: Recipe name without .yaml extension
        :raises ValueError: If recipe_file_path is None or empty
        """
        if not recipe_file_path:
            raise ValueError("Recipe file path cannot be None or empty")

        # Handle both full paths and just filenames
        recipe_name = recipe_file_path.split("/")[-1] if "/" in recipe_file_path else recipe_file_path

        # Remove .yaml extension if present
        if recipe_name.endswith(".yaml"):
            recipe_name = recipe_name[:-5]

        if not recipe_name:
            raise ValueError("Invalid recipe file path - no filename found")

        return recipe_name

    def load_hosting_config(self, recipe_name: str):
        """
        Loads the hosting config json for the given recipe name
        at utils/inference_configs/hosting-<recipe_name>.json

        :param recipe_name: Name of the recipe
        :return: Parsed hosting configs
        """
        config_path = ROOT_DIR / "utils" / "inference_configs" / f"hosting-{recipe_name}.json"

        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                try:
                    hosting_configs = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {config_path}: {e}")
        else:
            hosting_configs = None
            logging.info(f"No hosting config found for recipe: {recipe_name}")

        return hosting_configs

    def _load_recipe_config(self, recipe_file_path: str):
        return load_recipe_with_hydra(recipe_file_path)
