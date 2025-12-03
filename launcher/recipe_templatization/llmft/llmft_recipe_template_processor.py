#!/usr/bin/env python3
import json
import os
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)


class LLMFTRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """LLMFT-specific recipe template processor."""

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/llmft/llmft_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load LLMFT template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/llmft/llmft_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for LLMFT recipes based on recipe file path and technique."""
        if recipe_file_path is None:
            return None

        # Extract technique from recipe file path to determine template
        recipe_name = recipe_file_path.split("/")[-1] if "/" in recipe_file_path else recipe_file_path

        # Map recipe patterns to template names
        if "fft" in recipe_name:
            if "sft" in recipe_name:
                template_name = "llmft_sft_fft"
            elif "dpo" in recipe_name:
                template_name = "llmft_dpo_fft"
            else:
                raise KeyError(f"Matching template not found for recipe: {recipe_name}")
        else:
            if "sft_lora" in recipe_name:
                template_name = "llmft_sft_lora"
            elif "dpo" in recipe_name:
                template_name = "llmft_dpo"
            elif "fine_tuning" in recipe_name:
                template_name = "llmft_fine_tuning"
            else:
                raise KeyError(f"Matching template not found for recipe: {recipe_name}")

        return template.get(template_name)

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """
        Generate metadata for LLMFT recipes.

        Metadata Format:-
        "Name": "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
        "DisplayName": "Llama 3.1 8B Instruct SFT LoRA Fine-tuning on GPU, 4K sequence length",
        "Type": "FineTuning",
        "Framework": "LLMFT",
        "CustomizationTechnique": "SFT", // SFT, DPO
        "Peft": "LoRA",
        "SequenceLength": "4K",
        "ServerlessMeteringType": "Token-based",
        "Model_ID": "meta-textgeneration-llama-3-1-8b-instruct",
        "Hardware": "GPU",
        "Versions": [2.0, 1.0],
        "OutputConfig": {
        "SageMakerInferenceRecipeName": "default"
        },
        "InstanceCount": 1,
        "InstanceTypes": ["ml.p4de.24xlarge", "ml.p5.48xlarge"],
        "HostingConfigs": [{
            "InstanceType": "ml.p5.48xlarge",
            "Profile": "Default",
            "EcrAddress": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128",
            "Environment": {
            "OPTION_ENABLE_LORA": "true",
            "OPTION_MAX_LORAS": "8",
            "OPTION_MAX_CPU_LORAS": "64",
            "OPTION_TENSOR_PARALLEL_DEGREE": "8",
            "OPTION_ROLLING_BATCH": "disable",
            "OPTION_MAX_ROLLING_BATCH_SIZE": "1",
            "OPTION_ASYNC_MODE": "true",
            "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
            "SAGEMAKER_MAX_NUMBER_OF_ADAPTERS_IN_MEMORY": "128",
            "SAGEMAKER_ENABLE_LOAD_AWARE": "1"
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": 1024000,
                "NumberOfCpuCoresRequired": 100,
                "NumberOfAcceleratorDevicesRequired": 8
            },
        }],
        """
        metadata = OrderedDict()
        recipe_cfg = OmegaConf.load(os.path.join("./recipes_collection/recipes", recipe_file_path + ".yaml"))
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Get Name
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name is not None, "Recipe file name not found in recipe config"
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        # Get Display Name
        assert recipe_cfg["display_name"] is not None, "Recipe display name not found in recipe config"
        metadata["DisplayName"] = recipe_cfg["display_name"]

        # Get Job Type - All LLMFT recipes are FineTuning
        metadata["Type"] = "FineTuning"

        # Get Finetuning Technique from template metadata helpers
        technique = None
        peft_technique = None

        # Extract technique from recipe metadata helpers
        for tech_key, tech_value in recipe_metadata_helpers.get("fine_tuning_techniques", {}).items():
            if tech_key in recipe_file_name.lower():
                technique = tech_value
                break

        # Extract PEFT technique from recipe metadata helpers
        peft_type_from_config = self._extract_peft_type_from_config(recipe_cfg)
        if peft_type_from_config:
            # Try to get the display value from metadata helpers using the config key
            peft_techniques_map = recipe_metadata_helpers.get("peft_techniques", {})
            peft_technique = peft_techniques_map.get(peft_type_from_config.lower())

        # If peft technique not found in config, try matching filename with metadata helpers
        if peft_technique is None:
            for peft_key, peft_value in recipe_metadata_helpers.get("peft_techniques", {}).items():
                if peft_key in recipe_file_name.lower():
                    peft_technique = peft_value
                    break

        # Fallback: infer from recipe file name patterns
        if technique is None:
            if "dpo" in recipe_file_name.lower():
                technique = "DPO"
            elif "sft" in recipe_file_name.lower() or "fine_tuning" in recipe_file_name.lower():
                technique = "SFT"

        assert technique is not None, f"Training Technique not found for recipe: {recipe_file_name}"
        metadata["CustomizationTechnique"] = technique

        if peft_technique:
            metadata["Peft"] = peft_technique

        # Get Sequence length
        sequence_length = self.extract_sequence_length(metadata["Name"])
        assert sequence_length is not None, f"Sequence length not found for recipe: {recipe_file_name}"
        metadata["SequenceLength"] = sequence_length

        # Get Model ID
        assert (
            recipe_cfg.run.name in self.recipe_jumpstart_model_id_mapping
        ), f"Model name '{recipe_cfg.run.name}' not found in Recipe-Jumpstart ModelId mapping"
        metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[recipe_cfg.run.name]

        # Get Hardware - infer from recipe file name
        hardware = None
        if "gpu" in recipe_file_name.lower():
            hardware = "GPU"
        elif "trn" in recipe_file_name.lower() or "trainium" in recipe_file_name.lower():
            hardware = "TRN"
        else:
            hardware = "GPU"  # Default to GPU

        metadata["Hardware"] = hardware

        # Get Instance Types from the recipe
        assert recipe_cfg["instance_types"] is not None, "Recipe InstanceTypes not found in recipe config"
        metadata["InstanceTypes"] = OmegaConf.to_container(recipe_cfg["instance_types"], resolve=True)

        # Get Recipe Versions
        assert recipe_cfg["version"] is not None, "Recipe version not found in recipe file"
        metadata["Versions"] = [recipe_cfg["version"]]

        # Default OutputConfig
        metadata["OutputConfig"] = {"SageMakerInferenceRecipeName": "default"}

        # Get hosting configs
        hosting_configs = self.load_hosting_config(recipe_file_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        # Add Serverless Metering
        metadata["ServerlessMeteringType"] = ServerlessMeteringType.TOKEN_BASED.value

        # Get Instance Count (number of nodes)
        assert recipe_cfg["trainer"]["num_nodes"] is not None, "Number of nodes not found in recipe config"
        metadata["InstanceCount"] = recipe_cfg["trainer"]["num_nodes"]

        return metadata

    def _extract_peft_type_from_config(self, recipe_cfg) -> Optional[str]:
        """Extract peft type name from recipe configuration."""
        training_config = recipe_cfg.get("training_config") or {}
        model_config = training_config.get("model_config") or {}
        peft_config = model_config.get("peft_config") or {}
        peft_type = peft_config.get("peft_type", None)

        # Validate that peft_type is a non-empty string
        if peft_type and isinstance(peft_type, str):
            return peft_type.strip()

        return None
