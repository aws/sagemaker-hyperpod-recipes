#!/usr/bin/env python3
import json
import logging
from collections import OrderedDict
from typing import Optional

from omegaconf import DictConfig

from ..base_recipe_template_processor import BaseRecipeTemplateProcessor

logger = logging.getLogger(__name__)


class MtrlEvalRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """MTRL evaluation recipe template processor.

    Parallel to MtrlRecipeTemplateProcessor but targets MTRL evaluation recipes.
    Only deterministic evaluation is supported and SM Jobs is the only platform.
    """

    framework_type = "mtrl_eval"

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/mtrl_eval/mtrl_eval_recipe_template_parameters.json",
        platform: str = "sm_jobs",
    ):
        self.template_path = template_path
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        """Load MTRL eval template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/mtrl_eval/mtrl_eval_regional_parameters.json", "r") as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for MTRL eval recipes.

        MTRL evaluation has a single deterministic flavor.
        """
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        if "mtrl_eval" in template:
            return template["mtrl_eval"]

        raise ValueError("Invalid MTRL eval template key: mtrl_eval")

    def get_recipe_metadata(self, recipe_file_path: str, cfg: DictConfig = None) -> OrderedDict:
        """Generate metadata for MTRL eval recipes.

        Returns an OrderedDict whose keys in order are:
            Name, RecipeFilePath, DisplayName, Description, Type, EvaluationType, Versions

        CustomizationTechnique / Hardware / InstanceTypes are omitted — the eval
        operator filters on Type == "Evaluation" rather than CustomizationTechnique.
        """
        metadata = OrderedDict()
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Read base_model_name from the resolved Hydra config (which has CLI
        # overrides applied) when available, falling back to the raw recipe YAML.
        if cfg is not None:
            base_model_name = cfg.recipes.run.get("base_model_name", "")
        else:
            base_model_name = recipe_cfg.run.get("base_model_name", "")

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name, "Recipe file name not found"

        metadata["Name"] = f"mtrl-eval-{base_model_name}"
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        assert recipe_cfg.get("display_name") is not None, "display_name missing in recipe"
        metadata["DisplayName"] = recipe_cfg["display_name"]

        metadata["Model_ID"] = base_model_name

        metadata["Type"] = "Evaluation"
        # MTRL eval has a single deterministic flavor; the first key of the
        # evaluation_types helper map is the EvaluationType enum value.
        eval_types_map = recipe_metadata_helpers["evaluation_types"]
        metadata["EvaluationType"] = next(iter(eval_types_map))

        version = recipe_cfg.get("recipe_version") or recipe_cfg.get("version")
        assert version, "Recipe version not found"
        metadata["Versions"] = [version]

        return metadata

    def get_additional_data(self, recipe_file_path: str, cfg: DictConfig = None) -> list:
        """Get additional data including metadata, resolved override parameters, and regional parameters.

        Overrides the base class to pass ``cfg`` (the resolved Hydra config) to
        ``get_recipe_metadata`` so that CLI overrides like ``base_model_name``
        are reflected in the metadata.
        """
        if recipe_file_path is None:
            return [{}, {}, {}]

        try:
            self.process_recipe(recipe_file_path)
        except Exception as e:
            logger.warning(f"Failed to process recipe for metadata generation: {e}")
            return [{}, {}, {}]

        recipe_metadata = self.get_recipe_metadata(recipe_file_path, cfg)

        if not self._check_container_availability(recipe_metadata):
            return None

        resolved_override_parameters = self._resolve_constraints_using_metadata(recipe_metadata)
        self.recipe_override_parameters = resolved_override_parameters

        recipe_name = recipe_metadata.get("Name")
        regional_parameters = self._get_regional_parameters(recipe_name, recipe_metadata) if recipe_name else {}

        return [recipe_metadata, resolved_override_parameters, regional_parameters]
