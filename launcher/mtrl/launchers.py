import json
import logging
import os
import shutil
from pathlib import Path

import omegaconf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ..recipe_templatization.mtrl.mtrl_recipe_template_processor import (
    MtrlRecipeTemplateProcessor,
)
from ..recipe_templatization.mtrl_eval.mtrl_eval_recipe_template_processor import (
    MtrlEvalRecipeTemplateProcessor,
)

logger = logging.getLogger(__name__)


def get_recipe_file_path():
    """Get recipe file path from Hydra config."""
    hydra_cfg = HydraConfig.get()
    for override in hydra_cfg.overrides.task:
        if "recipes" in override:
            return override.split("=")[1]
    raise KeyError("Recipe file path not found in hydra config")


def _is_mtrl_eval_recipe_path(recipe_file_path: str) -> bool:
    """Return True if the recipe path is under evaluation/mtrl/."""
    return bool(recipe_file_path) and "evaluation/mtrl" in recipe_file_path


class MtrlSMTJLauncher:
    """Launcher for MTRL jobs on SM Jobs (SageMaker Training Jobs).

    Handles both MTRL fine-tuning recipes and MTRL evaluation recipes. The
    template processor is selected at ``_save_hydra_config`` time based on
    the Hydra-resolved recipe file path:

    - ``evaluation/mtrl/*`` → ``MtrlEvalRecipeTemplateProcessor``
    - anything else         → ``MtrlRecipeTemplateProcessor``
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._job_name = cfg.recipes.run["name"]
        self._output_dir = Path(cfg["base_results_dir"]) / self._job_name
        self._launch_json = cfg.get("launch_json", False)
        self.recipe_file_path = None
        self._recipe_template_processor = None

    def _prepare_output_dir(self):
        """Create output directory for launch artifacts."""
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

    def _interpolate_hydra(self):
        """Resolve Hydra interpolations in config."""

        def interpolate(cfg):
            if isinstance(cfg, omegaconf.DictConfig):
                for k, v in cfg.items():
                    cfg[k] = interpolate(v)
            elif isinstance(cfg, list):
                for i, v in enumerate(cfg):
                    cfg[i] = interpolate(v)
            return cfg

        interpolate(self.cfg.recipes)

    def _save_hydra_config(self):
        """Save Hydra config with recipe templatization.

        Picks the template processor based on the recipe file path so a single
        launcher handles both MTRL training and MTRL eval recipes.
        """
        self._interpolate_hydra()
        config_path = self._output_dir / "config"
        config_path.mkdir(parents=True, exist_ok=True)
        config_file = config_path / f"{self._job_name}_hydra.yaml"
        self.recipe_file_path = config_file

        if self._launch_json:
            try:
                recipe_file_path = get_recipe_file_path()
                logger.info(f"Found recipe file path: {recipe_file_path}")
                processor_cls = (
                    MtrlEvalRecipeTemplateProcessor
                    if _is_mtrl_eval_recipe_path(recipe_file_path)
                    else MtrlRecipeTemplateProcessor
                )
                self._recipe_template_processor = processor_cls(self.cfg.recipes, platform=self.cfg.cluster_type)
                self._recipe_template_processor = processor_cls(self.cfg.recipes, platform=self.cfg.cluster_type)
                templatized_recipe = self._recipe_template_processor.process_recipe(recipe_file_path)
                OmegaConf.save(templatized_recipe, config_file)
            except Exception as e:
                logger.warning(f"Recipe templatization failed: {e}")
                OmegaConf.save(self.cfg.recipes, config_file)
                self._recipe_template_processor = None
        else:
            OmegaConf.save(self.cfg.recipes, config_file)
            self._recipe_template_processor = None

    def _create_launch_json(self):
        """Create launch.json for MTRL SM Jobs recipes (training and eval)."""
        launch_json = {}
        recipe_file_path = get_recipe_file_path()

        # Get metadata from template processor
        if self._recipe_template_processor:
            if isinstance(self._recipe_template_processor, MtrlEvalRecipeTemplateProcessor):
                additional_data = self._recipe_template_processor.get_additional_data(recipe_file_path, self.cfg)
            else:
                additional_data = self._recipe_template_processor.get_additional_data(recipe_file_path)
            if additional_data:
                (
                    launch_json["metadata"],
                    launch_json["recipe_override_parameters"],
                    launch_json["regional_parameters"],
                ) = additional_data
            else:
                logger.warning(
                    "No regional parameters found, skipping container availability check and launch json generation"
                )
                return

        # Add templatized recipe YAML content. The key name ``training_recipe.yaml``
        # is the fixed SMTJ schema key for the payload template (see
        # ``launcher.recipe_templatization.launch_json_validation._SMTJ_FIXED_KEYS``);
        # it is used for both training and eval recipes.
        with open(self.recipe_file_path, "r") as f:
            launch_json["training_recipe.yaml"] = f.read()

        # Add the original (non-templatized) recipe wrapped under ``recipes`` key.
        # ``training_recipe.json`` is likewise the fixed SMTJ schema key regardless
        # of training vs eval.
        full_recipe_path = f"./recipes_collection/recipes/{recipe_file_path}.yaml"
        untemplated_recipe = OmegaConf.load(full_recipe_path)
        wrapped_recipe = OmegaConf.create({"recipes": untemplated_recipe})
        recipe_container = OmegaConf.to_container(wrapped_recipe, resolve=True)
        launch_json["training_recipe.json"] = recipe_container

        # Write launch.json
        launch_path = self._output_dir / "launch.json"
        with open(launch_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=False)
            f.write("\n")
        logger.info(f"launch.json created at {launch_path}")

    def run(self):
        """Run the MTRL SM Jobs launcher."""
        # Require launch_json=True for MTRL recipes to run
        if not self._launch_json:
            raise ValueError("MTRL recipes only support launch.json generation. Please run with launch_json=true")

        self._prepare_output_dir()
        self._save_hydra_config()

        self._create_launch_json()
        logger.info(f"MTRL launch.json generated at {self._output_dir}")
