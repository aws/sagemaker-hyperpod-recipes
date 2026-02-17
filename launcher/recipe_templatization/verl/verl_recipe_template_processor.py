#!/usr/bin/env python3
import json
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)


class VerlRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """VERL-specific recipe template processor for reinforcement learning fine-tuning."""

    framework_type = "verl"

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/verl/verl_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load VERL template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/verl/verl_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for VERL recipes based on algorithm type and reward mechanism."""
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        self.algorithm_type = self._extract_algorithm_type(recipe_cfg)

        # Determine template key based on algorithm and reward mechanism
        template_key = self.algorithm_type
        if self.algorithm_type == "grpo":
            display_name = recipe_cfg.get("display_name", "")
            customization_technique = self._training_technique(self.algorithm_type, display_name)
            if customization_technique == "RLAIF":
                if "fft" in recipe_file_name.lower():
                    template_key = "grpo_rlaif_fft"
                else:
                    template_key = "grpo_rlaif"
            elif customization_technique == "RLVR":
                if "fft" in recipe_file_name.lower():
                    template_key = "grpo_rlvr_fft"
                else:
                    template_key = "grpo_rlvr"

        if template_key in template:
            return template[template_key]

        raise ValueError(f"Invalid VERL template key: {template_key}")

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for VERL recipes."""
        metadata = OrderedDict()
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Basic metadata
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name is not None, "Recipe file name not found in recipe config"
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        # Get Display Name
        assert recipe_cfg["display_name"] is not None, "Recipe display name not found in recipe config"
        metadata["DisplayName"] = recipe_cfg["display_name"]

        # Get Job Type - All Verl recipes are FineTuning
        metadata["Type"] = "FineTuning"

        # Extract algorithm type from config
        self.algorithm_type = self._extract_algorithm_type(recipe_cfg)

        # Extract VERL-specific technique using helper function
        if self.algorithm_type in recipe_metadata_helpers["rl_algorithms"]:
            if self.algorithm_type == "grpo":
                metadata["CustomizationTechnique"] = self._training_technique(
                    self.algorithm_type, metadata["DisplayName"]
                )
            else:
                metadata["CustomizationTechnique"] = recipe_metadata_helpers["rl_algorithms"][self.algorithm_type]
        else:
            raise ValueError(f"Customization technique not found: {self.algorithm_type}")

        # Extract model info from the composed configuration
        model_name = self._extract_model_name(recipe_cfg)
        if model_name in self.recipe_jumpstart_model_id_mapping:
            metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[model_name]
        else:
            raise ValueError(f"Model ID not found: {model_name}")

        # Hardware and instance types
        hardware_type = self._extract_hardware_type(recipe_file_name)
        if hardware_type in recipe_metadata_helpers["hardware_types"]:
            metadata["Hardware"] = hardware_type.upper()

            # Get Instance Types directly from the recipe file
            assert recipe_cfg["instance_types"] is not None, "Recipe InstanceTypes not found in recipe config"
            metadata["InstanceTypes"] = OmegaConf.to_container(recipe_cfg["instance_types"], resolve=True)

        # Extract reward function if present (only for templates that support it)
        if "preset_reward_function" in recipe_metadata_helpers:
            reward_function_name = self._extract_reward_function_name(recipe_cfg)
            if reward_function_name and reward_function_name in recipe_metadata_helpers["preset_reward_function"]:
                metadata["PresetRewardFunction"] = recipe_metadata_helpers["preset_reward_function"][
                    reward_function_name
                ]

        # Set PEFT technique based on recipe name or config
        peft_technique = self._extract_peft_type_from_config(recipe_cfg)
        if peft_technique is None:
            for peft_key, peft_value in recipe_metadata_helpers.get("peft_techniques", {}).items():
                if peft_key in recipe_file_name.lower():
                    peft_technique = peft_value
                    break
        if peft_technique:
            metadata["Peft"] = peft_technique

        # Get Recipe Version
        assert recipe_cfg["version"] is not None, "Recipe version not found in recipe file"
        metadata["Versions"] = [recipe_cfg["version"]]

        # Default OutputConfig
        metadata["OutputConfig"] = {"SageMakerInferenceRecipeName": "default"}

        # Get hosting configs
        hosting_configs = self.load_hosting_config(recipe_file_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        # Add Serverless Metering
        metadata["ServerlessMeteringType"] = ServerlessMeteringType.HOURLY.value

        # Get Instance Count (number of nodes)
        num_nodes = self._extract_num_nodes(recipe_cfg)
        assert num_nodes is not None, "Number of nodes not found in recipe config"
        metadata["InstanceCount"] = num_nodes

        # Get input sequence length
        seq_length = self._extract_sequence_length(recipe_cfg)
        assert seq_length is not None, "Sequence length not found in recipe config"
        metadata["SequenceLength"] = self.format_sequence_length(seq_length)

        return metadata

    def _extract_algorithm_type(self, recipe_cfg) -> Optional[str]:
        """Extract RL algorithm type from recipe configuration."""
        training_config = recipe_cfg.get("training_config")
        algorithm = training_config.get("algorithm")
        return algorithm.get("adv_estimator")

    def _extract_model_name(self, recipe_cfg) -> Optional[str]:
        """Extract model name for jumpstart mapping lookup."""
        run_config = recipe_cfg.get("run")
        if run_config and "name" in run_config:
            return run_config.get("name")
        return None

    def _extract_hardware_type(self, recipe_name: str) -> str:
        """Extract hardware type from recipe name."""
        return "gpu"

    def _extract_reward_function_name(self, recipe_cfg) -> Optional[str]:
        """Extract reward function name from recipe configuration."""
        training_config = recipe_cfg.get("training_config")
        custom_reward_function = training_config.get("custom_reward_function")
        return custom_reward_function.get("name")

    def _extract_peft_type_from_config(self, recipe_cfg) -> Optional[str]:
        """Extract PEFT type from recipe configuration."""
        training_config = recipe_cfg.get("training_config")
        actor_rollout_ref = training_config.get("actor_rollout_ref")
        model = actor_rollout_ref.get("model")
        lora_rank = model.get("lora_rank")

        # If lora_rank > 0, it's LoRA; if 0, it's FFT
        if lora_rank and int(lora_rank) > 0:
            return "LoRA"
        return None

    def _extract_num_nodes(self, recipe_cfg) -> Optional[int]:
        """Extract number of nodes from recipe configuration."""
        trainer = recipe_cfg.get("trainer")
        if trainer and "num_nodes" in trainer:
            return trainer.get("num_nodes")

        ray_cluster = recipe_cfg.get("ray_cluster")
        worker_nodes = ray_cluster.get("worker_nodes")
        return worker_nodes.get("replicas")

    def _extract_sequence_length(self, recipe_cfg) -> Optional[int]:
        """Extract sequence length from recipe configuration."""
        training_config = recipe_cfg.get("training_config")
        data = training_config.get("data")
        return data.get("max_prompt_length")

    def _training_technique(self, algorithm_type: str, display_name: str) -> str:
        """Determine training technique based on algorithm type and display name."""
        if algorithm_type == "grpo":
            if "RLAIF" in display_name:
                return "RLAIF"
            elif "RLVR" in display_name:
                return "RLVR"
            else:
                raise ValueError(
                    f"Customization technique not found for GRPO technique with display name: {display_name}"
                )
        else:
            # For other algorithms like GAE, return the algorithm type as technique
            return algorithm_type.upper()
