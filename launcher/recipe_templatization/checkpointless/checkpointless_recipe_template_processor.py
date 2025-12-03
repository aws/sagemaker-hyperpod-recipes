#!/usr/bin/env python3
import json
import os
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import BaseRecipeTemplateProcessor


class CheckpointlessRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """Checkpointless-specific recipe template processor."""

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/checkpointless/checkpointless_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load checkpointless template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/checkpointless/checkpointless_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for checkpointless recipes."""
        if recipe_file_path is None:
            return None

        recipe_name = recipe_file_path.split("/")[-1] if "/" in recipe_file_path else recipe_file_path

        # Map recipe patterns to template names
        if "fine_tuning" in recipe_name:
            template_name = "checkpointless_fine_tuning"
        elif "pretrain" in recipe_name:
            template_name = "checkpointless_pretrain"
        else:
            raise KeyError(f"Matching template not found for recipe: {recipe_name}")

        return template.get(template_name)

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for checkpointless recipes."""
        metadata = OrderedDict()
        recipe_cfg = OmegaConf.load(os.path.join("./recipes_collection/recipes", recipe_file_path + ".yaml"))
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Get Name
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name is not None, "Recipe file name not found in recipe config"
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        # Get Display Name (construct from recipe name if not present)
        display_name = getattr(recipe_cfg, "display_name", None)
        if display_name is None:
            # Construct display name from recipe file name
            display_name = recipe_file_name.replace("_", " ").title()
        metadata["DisplayName"] = display_name

        # Determine job type
        if "fine_tuning" in recipe_file_name:
            metadata["Type"] = "FineTuning"
        else:
            metadata["Type"] = "PreTraining"

        # Extract sequence length
        sequence_length = self.extract_sequence_length(metadata["Name"])
        assert sequence_length is not None, f"Sequence length not found for recipe: {recipe_file_name}"
        metadata["SequenceLength"] = sequence_length

        # Get Model ID from run.name
        model_path = recipe_cfg.run.name
        if model_path in self.recipe_jumpstart_model_id_mapping:
            metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[model_path]
        else:
            raise ValueError(f"Model ID not found: {model_path}")

        # Determine hardware
        if "gpu" in recipe_file_name.lower():
            metadata["Hardware"] = "GPU"
        elif "trn" in recipe_file_name.lower():
            metadata["Hardware"] = "TRN"
        else:
            metadata["Hardware"] = "GPU"

        # Get instance count
        assert recipe_cfg["trainer"]["num_nodes"] is not None, "Number of nodes not found in recipe config"
        metadata["InstanceCount"] = recipe_cfg.trainer.num_nodes

        # Get instance type
        assert recipe_cfg["instance_types"] is not None, "Recipe InstanceTypes not found in recipe config"
        metadata["InstanceTypes"] = OmegaConf.to_container(recipe_cfg["instance_types"], resolve=True)

        # Add checkpointless-specific metadata
        metadata["ModelType"] = "hyperpod_checkpointless_nemo"

        return metadata
