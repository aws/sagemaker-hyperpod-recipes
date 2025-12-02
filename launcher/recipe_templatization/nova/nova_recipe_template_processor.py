#!/usr/bin/env python3
import json
import logging
import os
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)

logger = logging.getLogger(__name__)


class NovaRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """Nova-specific recipe template processor for both fine-tuning and evaluation."""

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/nova/nova_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load Nova unified template file."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)

        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/nova/nova_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)
        with open("./launcher/recipe_templatization/nova/nova_metadata.json", "r") as f:
            self.nova_metadata = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for Nova recipes by determining the specific recipe type."""

        recipe_file_name = recipe_file_path.split("/")[-1]
        if recipe_file_name == "" or recipe_file_name == None:
            raise ValueError(f"Invalid recipe filename for recipe file path:- {recipe_file_path}")

        # Get nova recipe metadata to get info for display_name, version, and instance_types
        nova_recipe_metadata = self.nova_metadata[recipe_file_name]

        # Get DisplayName. Raise error if its not found in the recipe file
        displayName = nova_recipe_metadata.get("display_name", None)
        if not displayName:
            raise ValueError("Display name not found in nova metadata")

        if "Data Distillation" in displayName:
            return template.get("nova_distill")
        elif "Evaluation" in displayName:
            eval_type = self._get_evaluation_type(recipe_file_path)
            computed_template_name = f"nova_{eval_type}_eval"
            if computed_template_name in template:
                return template.get(computed_template_name)
            else:
                raise ValueError(f"Template not found for evaluation type: {eval_type}")
        elif recipe_file_path:
            # Search based on recipe type
            recipe_type = self._determine_recipe_type(yaml_data, recipe_file_path)
            computed_template_name = f"nova_{recipe_type}"
            if computed_template_name in template:
                return template.get(computed_template_name)

            # Search based on recipe filename
            for template_key, matched_template in template.items():
                if template_key in recipe_file_name:
                    return matched_template
            raise ValueError(f"Template not found for recipe file: {recipe_file_name}")
        else:
            raise KeyError(f"Template not found and recipe_file_path is None")

    # Returns the metadata, recipe_override_parameters, regional_parameters for the launch_json
    def get_additional_data(self, recipe_file_path: str) -> list:
        """Get additional data including metadata, resolved override parameters, and regional parameters."""
        logger.info(f"Found recipe file path {recipe_file_path}")

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
        resolved_override_parameters["instance_type"] = {
            "type": "string",
            "required": False,
            "enum": recipe_metadata["InstanceTypes"],
            "default": recipe_metadata["InstanceTypes"][0],
        }

        if self.platform == "sm_jobs":
            if "namespace" in resolved_override_parameters:
                del resolved_override_parameters["namespace"]
            if "instance_type" in resolved_override_parameters:
                del resolved_override_parameters["instance_type"]
            if "replicas" in resolved_override_parameters:
                del resolved_override_parameters["replicas"]
            if "instance_count" in resolved_override_parameters:
                del resolved_override_parameters["instance_count"]
            for entry in ["mlflow_tracking_uri", "mlflow_experiment_name", "mlflow_run_name"]:
                if entry in resolved_override_parameters:
                    del resolved_override_parameters[entry]

        # Fetch regional parameters for the specific recipe only
        recipe_name = recipe_metadata.get("Name")
        regional_parameters = self._get_regional_parameters(recipe_name) if recipe_name else {}

        return [recipe_metadata, resolved_override_parameters, regional_parameters]

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for Nova recipes using template metadata helpers."""
        logger.info(f"Loading recipe metadata for: {recipe_file_path}")
        metadata = OrderedDict()

        try:
            recipe_cfg = OmegaConf.load(os.path.join("./recipes_collection/recipes", recipe_file_path + ".yaml"))
            logger.info(f"Recipe config loaded successfully")
        except Exception as e:
            logger.error(f"Error loading recipe config: {e}")
            raise

        # Metadata helpers are used to generate metadata based on some pre-defined values
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Extract basic info from recipe file name
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        if recipe_file_name == "" or recipe_file_name == None:
            raise ValueError(f"Invalid recipe filename for recipe file path:- {recipe_file_path}")

        # Get nova recipe metadata to get info for display_name, version, and instance_types
        nova_recipe_metadata = self.nova_metadata[recipe_file_name]

        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"
        metadata["DisplayName"] = nova_recipe_metadata.get("display_name", None)
        if not metadata["DisplayName"]:
            raise ValueError(f"Display name not found in recipe: {recipe_file_name}")

        model_type = recipe_cfg["run"].get("model_type", None)
        if not model_type:
            raise KeyError(f"Model type not found for recipe: {recipe_file_name}")
        jumpstart_model_id = self.recipe_jumpstart_model_id_mapping.get(model_type, None)
        if not jumpstart_model_id:
            raise KeyError(f"Jumpstart model ID not found for model type {model_type}")

        if hasattr(recipe_cfg.run, "model_name_or_path"):
            metadata["DefaultModelNameOrPath"] = recipe_cfg.run["model_name_or_path"]
            metadata["nova_model_name"] = recipe_cfg.run["model_name_or_path"]
        else:
            raise KeyError(f"model_name_or_path not found in recipe {recipe_file_name}")

        # Get Hardware Type from template helpers and set InstanceTypes
        hardware_type = self._extract_hardware_type(recipe_file_name)
        if hardware_type and hardware_type in recipe_metadata_helpers["hardware_types"]:
            metadata["Hardware"] = hardware_type.upper()

            # Get Instance Types directly from the recipe file
            assert nova_recipe_metadata["instance_types"] is not None, "InstanceTypes not found in nova metadata"
            metadata["InstanceTypes"] = nova_recipe_metadata["instance_types"]

        else:
            raise ValueError(f"Hardware type not found: {hardware_type}")

        # Check if this is an evaluation recipe
        is_evaluation = (
            hasattr(recipe_cfg, "evaluation") and recipe_cfg.evaluation and "Evaluation" in metadata["DisplayName"]
        )
        if is_evaluation:
            # Handle evaluation-specific metadata
            metadata["Type"] = "Evaluation"

            # Extract evaluation type from recipe name and use helper
            eval_type = self._get_evaluation_type(recipe_file_name)
            if eval_type and eval_type in recipe_metadata_helpers["evaluation_types"]:
                metadata["EvaluationType"] = recipe_metadata_helpers["evaluation_types"][eval_type]
            else:
                raise ValueError(f"Evaluation type not found: {eval_type}")

            # Add Serverless Metering for evaluation recipes
            serverlessMeteringType = self.get_serverless_metering_type("eval", recipe_file_name)
            if serverlessMeteringType != None:
                metadata["ServerlessMeteringType"] = serverlessMeteringType

        else:
            # Handle fine-tuning-specific metadata
            metadata["Type"] = "FineTuning"

            # Extract training technique from recipe name and use helper
            training_technique = self._extract_training_technique(recipe_file_name)
            if training_technique and training_technique in recipe_metadata_helpers["training_techniques"]:
                if training_technique == "rft":
                    if "RLAIF" in metadata["DisplayName"]:
                        metadata["CustomizationTechnique"] = "RLAIF"
                    elif "RLVR" in metadata["DisplayName"]:
                        metadata["CustomizationTechnique"] = "RLVR"
                    else:
                        raise ValueError(
                            f"Customization technique not found for Nova rft technique with displayname: {metadata['DisplayName']}"
                        )
                else:
                    metadata["CustomizationTechnique"] = training_technique
            else:
                raise ValueError(f"Training technique not found: {training_technique}")

            # Extract PEFT technique if available
            peft_technique = self._extract_peft_technique(recipe_file_name)
            if peft_technique and peft_technique in recipe_metadata_helpers.get("peft_techniques", {}):
                metadata["Peft"] = peft_technique

            # Add Serverless Metering
            serverlessMeteringType = self.get_serverless_metering_type(training_technique, recipe_file_name)
            if serverlessMeteringType != None:
                metadata["ServerlessMeteringType"] = serverlessMeteringType

            # Get Instance Count (number of nodes)
            if hardware_type != "cpu":  # uses nodes
                metadata["InstanceCount"] = self._determine_instance_count(
                    training_technique, recipe_cfg, recipe_file_name
                )

            # Get input sequence length
            if training_technique == "ppo":
                seq_length = recipe_cfg["ppo_actor_train"]["max_length"]
            elif training_technique == "distill":
                seq_length = int(recipe_cfg["training_config"]["maxStudentModelFineTuningContextLengthInTokens"])
            else:
                seq_length = recipe_cfg["training_config"]["max_length"]
            assert seq_length is not None, "Sequence length not found in recipe config"
            metadata["SequenceLength"] = self.format_sequence_length(seq_length)

        # Get Jumpstart model-id, fail if mapping is not present
        try:
            original_model_id = recipe_cfg.run.model_type
            metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[original_model_id]
        except Exception as e:
            logger.error(f"Error getting model_type: {e}")
            raise ValueError(f"Unable to determine model type for recipe: {recipe_file_name}")

        # Get version from pre-fetched
        version = nova_recipe_metadata.get("version", None)
        if not version:
            raise KeyError(f"Version not found in nova metadata")
        metadata["Versions"] = [version]

        # Default values
        metadata["OutputConfig"] = {"SageMakerInferenceRecipeName": "default"}

        # Get hosting configs
        hosting_configs = self.load_hosting_config(recipe_file_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        logger.info(f"Generated metadata successfully: {metadata}")
        return metadata

    def _extract_training_technique(self, recipe_name: str) -> Optional[str]:
        """Extract training technique from recipe name."""
        techniques = ["sft", "dpo", "ppo", "distill", "pretrain", "rft"]
        for technique in techniques:
            if technique in recipe_name:
                if technique == "pretrain":
                    return "CPT"
                return technique
        return None

    def _get_evaluation_type(self, recipe_name: str) -> Optional[str]:
        """Extract evaluation type from recipe name."""
        if "general_text_benchmark" in recipe_name:
            if "_2_0" in recipe_name:
                return "general_text_benchmark_2_0"
            return "general_text_benchmark"
        elif "general_multi_modal_benchmark" in recipe_name:
            if "_2_0" in recipe_name:
                return "general_multi_modal_benchmark_2_0"
            return "general_multi_modal_benchmark"
        elif "llm_judge" in recipe_name:
            return "llm_judge"
        elif "bring_your_own_dataset" in recipe_name:
            if "_2_0" in recipe_name:
                return "bring_your_own_dataset_2_0"
            return "bring_your_own_dataset"
        else:
            raise ValueError(f"Unable to determine evaluation type for recipe: {recipe_name}")

    def _extract_peft_technique(self, recipe_name: str) -> Optional[str]:
        """Extract PEFT technique from recipe name."""
        if "lora" in recipe_name:
            return "lora"
        return None

    def get_serverless_metering_type(self, training_technique: str, recipe_file_name: str) -> Optional[str]:
        if training_technique in ["sft", "dpo", "distill"]:
            return ServerlessMeteringType.TOKEN_BASED.value
        elif training_technique in ["rft"]:
            return ServerlessMeteringType.HOURLY.value
        elif "eval" in recipe_file_name:
            return ServerlessMeteringType.TOKEN_BASED.value
        else:
            return None

    def _extract_hardware_type(self, recipe_name: str) -> Optional[str]:
        """Extract hardware type from recipe name."""
        hardware_type = None

        if "gpu" in recipe_name:
            hardware_type = "gpu"
        elif "trn" in recipe_name:
            hardware_type = "trainium"
        elif "cpu" in recipe_name:
            hardware_type = "cpu"
        else:
            if "eval" in recipe_name or "rft" in recipe_name:
                hardware_type = "gpu"
            else:
                raise ValueError(f"Unable to determine hardware type for recipe: {recipe_name}")
        return hardware_type

    def _determine_recipe_type(self, yaml_data: dict, recipe_file_path: str = None) -> str:
        """Determine the fine-tuning recipe type from the recipe filename."""
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to determine recipe type")

        # Extract recipe type directly from filename patterns
        if "pretrain" in recipe_file_path and "2_0" in recipe_file_path:
            return "pretrain_2_0"
        elif "pretrain" in recipe_file_path:
            return "pretrain"
        elif "distill" in recipe_file_path:
            return "distill"
        elif "lora" in recipe_file_path and "dpo" in recipe_file_path:
            return "lora_dpo"
        elif "lora" in recipe_file_path and "sft" in recipe_file_path:
            if "_2_0" in recipe_file_path:
                return "lora_sft_2_0"
            return "lora_sft"
        elif "dpo" in recipe_file_path:
            return "dpo"
        elif "ppo" in recipe_file_path:
            return "ppo"
        elif "sft" in recipe_file_path:
            if "_2_0" in recipe_file_path:
                return "sft_2_0"
            return "sft"
        elif "smtj" in recipe_file_path and "rft" in recipe_file_path:
            if "lora" in recipe_file_path:
                return "smtj_lora_rft"
            return "smtj_rft"
        elif "rft" in recipe_file_path:
            if "lora" in recipe_file_path:
                return "lora_rft"
            return "rft"
        else:
            raise ValueError(f"Unable to determine recipe type from filename: {recipe_file_path}")

    def _determine_instance_count(self, training_technique: str, recipe_cfg: OmegaConf, recipe_file_name: str):
        # Nova rft recipes use different field structure than PPO-style recipes
        if training_technique == "rft":
            if "smtj" not in recipe_file_name:
                # Nova rft uses: replicas, generation_replicas, rollout_worker_replicas
                assert recipe_cfg["run"]["replicas"] is not None, f"replicas not found for recipe: {recipe_file_name}"
                assert (
                    recipe_cfg["run"]["generation_replicas"] is not None
                ), f"generation_replicas not found for recipe: {recipe_file_name}"
                assert (
                    recipe_cfg["run"]["rollout_worker_replicas"] is not None
                ), f"rollout_worker_replicas not found for recipe: {recipe_file_name}"

                # Core replicas from recipe configuration
                total_instances = (
                    recipe_cfg["run"]["replicas"]
                    + recipe_cfg["run"]["generation_replicas"]
                    + recipe_cfg["run"]["rollout_worker_replicas"]
                )

                # Additional service replicas (not mapped from recipe fields)
                # These services each have 1 replica: rewardFunction, hub, promptRbs, natsServer
                additional_services = 6  # rewardFunctio - 1, hub - 1, promptRbs - 1, natsServer - 3

                # Redis is only enabled when rollout.delegate is not none and set to true
                delegate = OmegaConf.select(recipe_cfg, "run.rollout.delegate")
                if delegate is not None and delegate is True:
                    additional_services += 1  # redis (1 replica when enabled)

                total_instances += additional_services
                return total_instances
            else:
                return recipe_cfg.run["replicas"]

        elif training_technique == "ppo":
            # Nova PPO uses VERL-style fields
            assert (
                recipe_cfg["run"]["actor_train_replicas"] is not None
            ), f"actor_train_replicas not found for recipe: {recipe_file_name}"
            assert recipe_cfg["run"]["rm_replicas"] is not None, f"rm_replicas not found for recipe: {recipe_file_name}"
            assert recipe_cfg["run"]["cm_replicas"] is not None, f"cm_replicas not found for recipe: {recipe_file_name}"
            assert (
                recipe_cfg["run"]["actor_generation_replicas"] is not None
            ), f"actor_generation_replicas not found for recipe: {recipe_file_name}"
            assert recipe_cfg["run"]["am_replicas"] is not None, f"am_replicas not found for recipe: {recipe_file_name}"

            total_instances = (
                recipe_cfg["run"]["actor_train_replicas"]
                + recipe_cfg["run"]["rm_replicas"]
                + recipe_cfg["run"]["cm_replicas"]
                + recipe_cfg["run"]["actor_generation_replicas"]
                + recipe_cfg["run"]["am_replicas"]
            )
            return total_instances

        else:
            # For other techniques (sft, dpo, distill, pretrain), use standard replicas field
            if hasattr(recipe_cfg.run, "replicas"):
                return recipe_cfg.run["replicas"]
            else:
                raise ValueError(f"Number of instances/replicas not found in recipe: {recipe_file_name}")
