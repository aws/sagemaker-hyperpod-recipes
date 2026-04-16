#!/usr/bin/env python3
import json
import logging
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)

logger = logging.getLogger(__name__)


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
        self._verl_regional_key: Optional[str] = None
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load VERL template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/verl/verl_regional_parameters.json", "r") as f:
            self._original_regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for VERL recipes based on algorithm type and reward mechanism."""
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        self.algorithm_type = self._extract_algorithm_type(recipe_cfg)

        logger.debug(f"Detected verl algorithm type: {self.algorithm_type}")

        # Determine template key based on algorithm and reward mechanism
        template_key = self.algorithm_type
        if self.algorithm_type == "sft":
            if "fft" in recipe_file_name.lower():
                template_key = "sft_fft"
            else:
                template_key = "sft"
        elif self.algorithm_type == "grpo":
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

    def _get_regional_parameters(self, recipe_name: str, recipe_metadata: dict) -> dict:
        """Get regional parameters using the detected verl version key.

        Overrides the base class implementation because detecting verl version requires having the recipe contents.
        It uses the cached self._verl_regional_key from the last run of self.get_recipe_metadata()
        """
        if self._verl_regional_key is None:
            raise ValueError("Failed to get verl regional key")

        if self._verl_regional_key not in self._original_regional_parameters:
            raise ValueError("Invalid verl regional key")

        # Temporarily patch regional_parameters so the base implementation only has the matching version
        patched = {"verl": self._original_regional_parameters[self._verl_regional_key]}
        self.regional_parameters = patched
        return super()._get_regional_parameters(recipe_name, recipe_metadata)

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for VERL recipes."""
        metadata = OrderedDict()
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Check for verl-0.7.0
        # SFT 0.7.0 recipes have "profiler" directly under training_config,
        # while RL 0.7.0 recipes (GRPO/PPO) have "global_profiler" instead.
        is_verl_0_7_0 = "profiler" in recipe_cfg.training_config or "global_profiler" in recipe_cfg.training_config
        self._verl_regional_key = "verl-0.7.0" if is_verl_0_7_0 else "verl"
        logger.debug(f"Detected verl recipe version: {self._verl_regional_key}")

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
        is_sft = (
            "sft_algorithms" in recipe_metadata_helpers
            and self.algorithm_type in recipe_metadata_helpers["sft_algorithms"]
        )
        if is_sft:
            metadata["CustomizationTechnique"] = recipe_metadata_helpers["sft_algorithms"][self.algorithm_type]
        elif (
            "rl_algorithms" in recipe_metadata_helpers
            and self.algorithm_type in recipe_metadata_helpers["rl_algorithms"]
        ):
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
        if is_sft:
            metadata["ServerlessMeteringType"] = ServerlessMeteringType.TOKEN_BASED.value
        else:
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
        """Extract algorithm type from recipe configuration.

        For RL recipes (e.g. GRPO), the type is found at
        training_config.algorithm.adv_estimator. For SFT recipes there is no
        algorithm section, so we return "sft" in that case.
        """
        training_config = recipe_cfg.get("training_config")
        algorithm = training_config.get("algorithm")
        if algorithm is None:
            return "sft"
        adv_estimator = algorithm.get("adv_estimator")
        if adv_estimator is None:
            return "sft"
        return adv_estimator

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
        """Extract reward function name from recipe configuration.

        Returns None for SFT recipes which have no custom_reward_function section.
        """
        training_config = recipe_cfg.get("training_config")
        custom_reward_function = training_config.get("custom_reward_function")
        if custom_reward_function is None:
            return None
        return custom_reward_function.get("name")

    def _extract_peft_type_from_config(self, recipe_cfg) -> Optional[str]:
        """Extract PEFT type from recipe configuration.

        Handles both RL recipes (lora_rank under actor_rollout_ref.model) and
        SFT recipes (lora_rank directly under model).
        """
        training_config = recipe_cfg.get("training_config")

        # SFT recipes have the model config directly under training_config
        actor_rollout_ref = training_config.get("actor_rollout_ref")
        if actor_rollout_ref is not None:
            model = actor_rollout_ref.get("model")
        else:
            model = training_config.get("model")

        if model is None:
            raise ValueError("Could not determine PEFT type")

        lora_rank = model.get("lora_rank")

        # If lora_rank > 0, it's LoRA; if 0 or absent, it's FFT
        if lora_rank and int(lora_rank) > 0:
            return "LoRA"
        return None

    def _extract_num_nodes(self, recipe_cfg) -> Optional[int]:
        """Extract number of nodes from recipe configuration.

        SFT recipes use trainer.num_nodes; RL recipes use
        ray_cluster.worker_nodes.replicas (with trainer.num_nodes as fallback).
        """
        trainer = recipe_cfg.get("trainer")
        if trainer and "num_nodes" in trainer:
            return trainer.get("num_nodes")

        ray_cluster = recipe_cfg.get("ray_cluster")
        if ray_cluster is None:
            return None
        worker_nodes = ray_cluster.get("worker_nodes")
        if worker_nodes is None:
            return None
        return worker_nodes.get("replicas")

    def _extract_sequence_length(self, recipe_cfg) -> Optional[int]:
        """Extract sequence length from recipe configuration.

        RL recipes use data.max_prompt_length; SFT recipes use data.max_length.
        Falls back to max_length if max_prompt_length is not present.
        """
        training_config = recipe_cfg.get("training_config")
        data = training_config.get("data")
        max_prompt_length = data.get("max_prompt_length")
        if max_prompt_length is not None:
            return max_prompt_length
        return data.get("max_length")

    def _training_technique(self, algorithm_type: str, display_name: str) -> str:
        """Determine training technique based on algorithm type and display name."""
        if algorithm_type == "sft":
            return "SFT"
        elif algorithm_type == "grpo":
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
