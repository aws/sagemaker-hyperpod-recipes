#!/usr/bin/env python3
import copy
import json
import os
from collections import OrderedDict
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from ..base_recipe_template_processor import BaseRecipeTemplateProcessor


class EvaluationRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """Evaluation-specific recipe template processor."""

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/evaluation/evaluation_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load evaluation template file."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)

        with open("./launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for evaluation recipes."""
        # Check for LLMAJ recipe first
        if recipe_file_path and "llmaj" in recipe_file_path:
            return template.get("open_source_llmaj_eval")
        if "llm_as_judge" in yaml_data:
            return template.get("open_source_llmaj_eval")

        # Check for deterministic recipe
        if recipe_file_path and "deterministic" in recipe_file_path:
            return template.get("open_source_deterministic_eval")
        if "evaluation" in yaml_data:
            return template.get("open_source_deterministic_eval")

        # Fallback for evaluation recipes
        if recipe_file_path and "evaluation" in recipe_file_path:
            return template.get("open_source_deterministic_eval")

        return None

    def get_instance_type(self, cfg, recipe_file_path):
        instance_type = None
        if "open_source_deterministic_eval" in recipe_file_path:
            with open("./launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json", "r") as f:
                regional_parameters = json.load(f)
                instance_map = regional_parameters.get("js_model_name_instance_mapping", {})
                base_model_name = cfg.recipes.run.get("base_model_name", "")
                instance_type = instance_map.get(base_model_name, ["ml.p5.48xlarge"])
                return instance_type
            return ["ml.p5.48xlarge"]
        else:
            return ["ml.t3.large"]

    def get_suffix(self, recipe_file_path):
        if "open_source_deterministic_eval" in recipe_file_path:
            return "deterministic"
        return "llmaj"

    def get_evaluation_type(self, recipe_file_path):
        if "open_source_deterministic_eval" in recipe_file_path:
            return "DeterministicEvaluation"
        return "LLMAJEvaluation"

    def get_hardware(self, recipe_file_path):
        if "open_source_deterministic_eval" in recipe_file_path:
            return "GPU"
        return "CPU"

    def get_recipe_metadata(self, recipe_file_path: str, cfg: DictConfig = None) -> OrderedDict:
        """Generate metadata for evaluation recipes."""
        metadata = OrderedDict()

        try:
            recipe_cfg = OmegaConf.load(os.path.join("./recipes_collection/recipes", recipe_file_path + ".yaml"))
        except Exception as e:
            raise Exception(f"Error loading recipe config: {e}")

        # Get metadata helpers from the matched template
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Extract basic info from recipe file name or use run.name
        base_model_name = cfg.recipes.run.get("base_model_name", "")
        job_name = cfg.recipes.run.get("name", "")

        # Generate model-specific metadata
        metadata["Name"] = f"open-source-eval-{base_model_name}-" + self.get_suffix(recipe_file_path)
        metadata["DefaultModelNameOrPath"] = base_model_name
        metadata["DisplayName"] = self._generate_eval_display_name_for_model(recipe_file_path, base_model_name)
        metadata["Hardware"] = self.get_hardware(recipe_file_path)
        metadata["InstanceTypes"] = self.get_instance_type(cfg, recipe_file_path)
        metadata["Model_ID"] = base_model_name
        metadata["Type"] = "Evaluation"
        metadata["EvaluationType"] = self.get_evaluation_type(recipe_file_path)
        metadata["Versions"] = ["1.0"]

        # Extract recipe name from recipe_file_path for hosting config
        recipe_name = self.get_recipe_name_from_path(recipe_file_path)
        hosting_configs = self.load_hosting_config(recipe_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        return metadata

    def _extract_evaluation_type(self, recipe_name: str) -> Optional[str]:
        """Extract evaluation type from recipe name."""
        if "general_text_benchmark" in recipe_name:
            return "general_text_benchmark"
        elif "general_multi_modal_benchmark" in recipe_name:
            return "general_multi_modal_benchmark"
        elif "llm_judge" in recipe_name:
            return "llm_judge"
        elif "bring_your_own_dataset" in recipe_name:
            return "bring_your_own_dataset"
        return "general_text_benchmark"  # default

    def _extract_hardware_type(self, recipe_name: str) -> Optional[str]:
        """Extract hardware type from recipe name."""
        if "cpu" in recipe_name:
            return "cpu"
        elif any(hw in recipe_name for hw in ["p5", "p4", "g5", "g6"]) or "gpu" in recipe_name:
            return "gpu"
        return "gpu"  # default

    def _generate_eval_display_name(self, recipe_name: str) -> str:
        """Generate a human-readable display name from evaluation recipe filename."""
        parts = recipe_name.split("_")

        # Extract evaluation type
        eval_type = ""
        if "general" in parts and "text" in parts and "benchmark" in parts:
            eval_type = "General Text Benchmark"
        elif "general" in parts and "multi" in parts and "modal" in parts and "benchmark" in parts:
            eval_type = "General Multi-Modal Benchmark"
        elif "llm" in parts and "judge" in parts:
            eval_type = "LLM Judge"
        elif "bring" in parts and "your" in parts and "own" in parts and "dataset" in parts:
            eval_type = "Bring Your Own Dataset"
        else:
            eval_type = "Evaluation"

        # Extract hardware
        hardware = ""
        for part in parts:
            if part in ["p5", "p4", "g5", "g6"]:
                hardware = part.upper()
                break

        hardware_str = "CPU" if "cpu" in recipe_name else "GPU"

        if hardware:
            return f"Open Source {eval_type} Evaluation on {hardware} {hardware_str}"
        else:
            return f"Open Source {eval_type} Evaluation on {hardware_str}"

    def _generate_eval_display_name_for_model(self, recipe_name: str, model_name: str) -> str:
        """Generate a human-readable display name for a specific model."""
        base_display_name = self._generate_eval_display_name(recipe_name)

        # Clean up model name for display
        display_model_name = model_name.replace("-", " ").title()

        return f"{base_display_name} - {display_model_name}"

    def _resolve_constraints_using_metadata(self, metadata: dict) -> dict:
        """Resolve conditional constraints using already generated metadata."""
        if not self.recipe_override_parameters:
            return {}

        # Create a copy of the parameters to modify
        resolved_parameters = copy.deepcopy(self.recipe_override_parameters)

        # Process each parameter that has conditional_constraints
        for param_name, param_config in resolved_parameters.items():
            if "conditional_constraints" in param_config:
                conditional_constraints = param_config["conditional_constraints"]

                # Process each conditioning key
                for condition_key, constraint_map in conditional_constraints.items():
                    resolved_enum = None

                    if condition_key == "RecipeFilename":
                        # Extract pattern from recipe name
                        recipe_name = metadata.get("Name", "")
                        filename_pattern = self._extract_filename_pattern(recipe_name)

                        if filename_pattern and filename_pattern in constraint_map:
                            resolved_enum = constraint_map[filename_pattern]
                        elif "default" in constraint_map:
                            resolved_enum = constraint_map["default"]
                    elif condition_key in metadata:
                        condition_value = metadata[condition_key]

                        # Look for exact match first, then fall back to "default"
                        if condition_value in constraint_map:
                            resolved_enum = constraint_map[condition_value]
                        elif "default" in constraint_map:
                            resolved_enum = constraint_map["default"]

                    if resolved_enum:
                        # Update the enum with the resolved values
                        if "enum" in resolved_enum:
                            param_config["enum"] = resolved_enum["enum"]
                        if "default" in resolved_enum:
                            param_config["default"] = resolved_enum["default"]

                        # Remove conditional_constraints from the final output
                        del param_config["conditional_constraints"]
                        break  # Only process the first matching condition key

        return resolved_parameters

    def _extract_filename_pattern(self, recipe_name: str) -> Optional[str]:
        """Extract filename pattern for conditional constraints matching."""
        patterns = [
            ("p5_48xl", ["p5", "48xl"]),
            ("p4_24xl", ["p4", "24xl"]),
            ("g5_48xl", ["g5", "48xl"]),
        ]

        recipe_lower = recipe_name.lower()

        for pattern_name, keywords in patterns:
            if all(keyword in recipe_lower for keyword in keywords):
                return pattern_name

        return None

    def get_additional_data(self, recipe_file_path: str, cfg: DictConfig = None) -> list:
        """Get additional data including metadata, resolved override parameters, and regional parameters."""
        if recipe_file_path is None:
            return [{}, {}, {}]

        # Process recipe first to set up matched_template_group
        try:
            self.process_recipe(recipe_file_path)
        except Exception as e:
            print(f"Warning: Failed to process recipe for metadata generation: {e}")
            return [{}, {}, {}]

        # Generate initial metadata
        recipe_metadata = self.get_recipe_metadata(recipe_file_path, cfg)

        # Check container availability
        if not self._check_container_availability(recipe_metadata):
            return None

        # Resolve conditional constraints using the generated metadata
        resolved_override_parameters = self._resolve_constraints_using_metadata(recipe_metadata)

        # Update the instance variable
        self.recipe_override_parameters = resolved_override_parameters

        # Remove platform-specific parameters for sm_jobs
        if self.platform == "sm_jobs":
            if "custom_labels" in self.recipe_override_parameters:
                del self.recipe_override_parameters["custom_labels"]
            if "replicas" in self.recipe_override_parameters:
                del self.recipe_override_parameters["replicas"]
            if "namespace" in self.recipe_override_parameters:
                del self.recipe_override_parameters["namespace"]

        # Update metadata with resolved instance types
        recipe_metadata["InstanceTypes"] = self.get_instance_type(cfg, recipe_file_path)

        # Get regional parameters for the specific recipe
        recipe_name = recipe_metadata.get("Name")
        regional_parameters = self._get_regional_parameters(recipe_name) if recipe_name else {}

        return [recipe_metadata, resolved_override_parameters, regional_parameters]

    def _check_container_availability(self, recipe_metadata: dict) -> bool:
        """Check if containers are available for the given recipe type and platform."""
        import logging

        logger = logging.getLogger(__name__)

        # For open source evaluation recipes, check based on recipe type rather than specific name
        recipe_type = "open_source_deterministic_eval"  # Default for deterministic eval

        if "recipe_container_mapping" not in self.regional_parameters:
            logger.warning("Recipe container mapping not found in regional parameters")
            return True

        recipe_mapping = self.regional_parameters["recipe_container_mapping"]

        if recipe_type not in recipe_mapping:
            logger.warning(f"Recipe type {recipe_type} not found in container mapping, allowing by default")
            return True

        recipe_config = recipe_mapping[recipe_type]

        if self.platform == "k8s":
            container_field = "container_image"
        elif self.platform == "sm_jobs":
            container_field = "smtj_container_image"
        else:
            logger.warning(f"Unknown platform: {self.platform}")
            return True

        if container_field in recipe_config:
            container_config = recipe_config[container_field]
            if container_config == "none":
                logger.info(f"Container not available for recipe type {recipe_type} on platform {self.platform}")
                return False
            elif isinstance(container_config, dict) and container_config:
                logger.info(f"Container available for recipe type {recipe_type} on platform {self.platform}")
                return True
            else:
                logger.warning(
                    f"Invalid container configuration for recipe type {recipe_type} on platform {self.platform}"
                )
                return False
        else:
            logger.warning(f"Container field {container_field} not found for recipe type {recipe_type}")
            return True

    def _get_regional_parameters(self, recipe_name: str) -> dict:
        """Get regional parameters for the given recipe type."""
        import logging

        logger = logging.getLogger(__name__)

        regional_parameters = {}
        # For open source evaluation recipes, use recipe type rather than specific name
        recipe_type = "open_source_deterministic_eval"  # Default for deterministic eval

        if "recipe_container_mapping" not in self.regional_parameters:
            logger.warning("Recipe container mapping not found in regional parameters")
            return regional_parameters
        recipe_mapping = self.regional_parameters["recipe_container_mapping"]
        if recipe_type not in recipe_mapping:
            logger.warning(f"Recipe type {recipe_type} not found in container mapping")
            return regional_parameters

        recipe_config = recipe_mapping[recipe_type]
        if self.platform == "k8s":
            # For k8s platform: return both container_image and smtj_container_image
            for container_field in ["container_image", "smtj_container_image"]:
                if container_field in recipe_config:
                    container_config = recipe_config[container_field]
                    if container_config != "none" and isinstance(container_config, dict):
                        regional_parameters[container_field] = container_config
        elif self.platform == "sm_jobs":
            # For sm_jobs platform: return only container_image but set its value to smtj_container_image
            if "smtj_container_image" in recipe_config:
                smtj_container_config = recipe_config["smtj_container_image"]
                if smtj_container_config != "none" and isinstance(smtj_container_config, dict):
                    regional_parameters["container_image"] = smtj_container_config

        return regional_parameters
