#!/usr/bin/env python3
import json
import logging
from collections import OrderedDict
from typing import Optional

from ..base_recipe_template_processor import BaseRecipeTemplateProcessor

logger = logging.getLogger(__name__)


class MtrlRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """MTRL-specific recipe template processor for agentic RFT (Reinforcement Fine-Tuning)."""

    framework_type = "mtrl"

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/mtrl/mtrl_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load MTRL template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/mtrl/mtrl_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for MTRL recipes based on PEFT type."""
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)

        # MTRL recipes are primarily LoRA-based, determine template key
        if "lora" in recipe_file_name.lower():
            if "nova_lite" in recipe_file_name.lower():
                template_key = "mtrl_nova_lite_lora"
            else:
                template_key = "mtrl_lora"
        else:
            # Default to LoRA template for MTRL
            template_key = "mtrl_lora"

        if template_key in template:
            return template[template_key]

        raise ValueError(f"Invalid MTRL template key: {template_key}")

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for MTRL recipes."""
        metadata = OrderedDict()
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Basic metadata
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name is not None, "Recipe file name not found in recipe config"
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        # Get Display Name from recipe config
        assert recipe_cfg.get("display_name") is not None, "Recipe display name not found in recipe config"
        metadata["DisplayName"] = recipe_cfg["display_name"]

        # Get Model_ID from JS map
        metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[recipe_cfg.run.name]

        # Get Description - generated from display name or recipe config
        metadata["Description"] = f"Agentic RFT training recipe for {self._extract_model_display_name(recipe_cfg)}"

        # Job Type - All MTRL recipes are FineTuning
        metadata["Type"] = "FineTuning"

        # CustomizationTechnique - MTRL for all MTRL recipes
        metadata["CustomizationTechnique"] = "MTRL"

        # Extract PEFT technique
        peft_technique = self._extract_peft_type_from_config(recipe_cfg)
        if peft_technique is None:
            for peft_key, peft_value in recipe_metadata_helpers.get("peft_techniques", {}).items():
                if peft_key in recipe_file_name.lower():
                    peft_technique = peft_value
                    break
        if peft_technique:
            metadata["Peft"] = peft_technique

        # Get sequence length from model config
        seq_length = self._extract_sequence_length(recipe_cfg)
        assert seq_length is not None, "Sequence length not found in recipe config"
        metadata["SequenceLength"] = self.format_sequence_length(seq_length)

        # Get Recipe Versions
        version = recipe_cfg.get("recipe_version") or recipe_cfg.get("version")
        assert version is not None, "Recipe version not found in recipe file"
        metadata["Versions"] = [version]

        # Get hosting configs
        hosting_configs = self.load_hosting_config(recipe_file_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        return metadata

    def _extract_model_display_name(self, recipe_cfg) -> str:
        """Extract a human-readable model display name."""
        model_config = recipe_cfg.get("model", {})
        model_name = model_config.get("model_name", "Unknown Model")
        # Extract just the model part after the org
        if "/" in model_name:
            return model_name.split("/")[-1]
        return model_name

    def _extract_sequence_length(self, recipe_cfg) -> Optional[int]:
        """Extract sequence length from recipe configuration."""
        model_config = recipe_cfg.get("model", {})
        return model_config.get("max_sequence_length")

    def _extract_peft_type_from_config(self, recipe_cfg) -> Optional[str]:
        """Extract PEFT type from recipe configuration."""
        model_config = recipe_cfg.get("model", {})
        lora_rank = model_config.get("lora_rank")

        # If lora_rank > 0, it's LoRA; otherwise FFT
        if lora_rank and int(lora_rank) > 0:
            return "LoRA"
        return None
