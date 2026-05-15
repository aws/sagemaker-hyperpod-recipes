"""
Unit tests to validate that algorithm-relevant recipe override parameters
have 'category' and 'description' metadata fields.

These tests ensure that hyperparameter params are properly tagged so the
UI can distinguish tunable hyperparameters from config/admin params and
display descriptions in tooltips.
"""

import json
from pathlib import Path
from typing import Dict, List, Set

import pytest

# Params that must have category="hyperparameter" and a description.
# This is the canonical list of algorithm-relevant params across all recipes.
REQUIRED_HYPERPARAMETER_PARAMS: Set[str] = {
    # Optimizer & LR
    "learning_rate",
    "lr_scheduler",
    "lr_warmup_steps_ratio",
    "weight_decay",
    "warmup_steps",
    "min_lr",
    "learning_rate_ratio",
    # Training schedule
    "max_epochs",
    "max_steps",
    "seed",
    "logging_steps",
    "save_steps",
    # Batch & sequence
    "global_batch_size",
    "dataset_max_len",
    "max_context_length",
    "max_prompt_length",
    "max_response_length",
    # Regularization
    "gradient_clipping",
    "gradient_clipping_threshold",
    # LoRA
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "merge_weights",
    "fine_tuned_model",
    # Generation
    "temperature",
    # RL-specific
    "clip_ratio",
    "clip_ratio_high",
    "clip_ratio_low",
    "kl_loss_coef",
    "use_kl_loss",
    "rollout_n",
    "rollout_temperature",
    # DPO-specific
    "adam_beta",
    # Data
    "train_val_split_ratio",
    # Model
    "reasoning_enabled",
}

EXPECTED_CATEGORY_VALUE = "hyperparameter"


def get_template_parameter_files() -> List[Path]:
    """Find all recipe template parameter JSON files."""
    base_dir = Path("launcher/recipe_templatization")
    template_files = []
    for subdir in ["nova", "llmft", "verl"]:
        file_path = base_dir / subdir / f"{subdir}_recipe_template_parameters.json"
        if file_path.exists():
            template_files.append(file_path)
    return template_files


def load_template_file(file_path: Path) -> Dict:
    """Load and parse a template parameters JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestHyperparameterCategoryMetadata:
    """Test suite for validating hyperparameter category and description metadata."""

    @pytest.mark.parametrize("file_path", get_template_parameter_files())
    def test_hyperparameter_params_have_category(self, file_path: Path):
        """
        Verify that algorithm-relevant params have category='hyperparameter'.

        Only checks params that exist in the template AND are in the
        REQUIRED_HYPERPARAMETER_PARAMS set. Params not in the set are
        config/admin params and should NOT have a category field.
        """
        data = load_template_file(file_path)
        errors = []

        for template_name, template_data in data.get("templates", {}).items():
            override_params = template_data.get("recipe_override_parameters", {})

            for param_name, param_config in override_params.items():
                if param_name not in REQUIRED_HYPERPARAMETER_PARAMS:
                    continue

                if "category" not in param_config:
                    errors.append(f"{template_name}::{param_name}: missing 'category' field")
                elif param_config["category"] != EXPECTED_CATEGORY_VALUE:
                    errors.append(
                        f"{template_name}::{param_name}: expected category "
                        f"'{EXPECTED_CATEGORY_VALUE}', got '{param_config['category']}'"
                    )

        if errors:
            pytest.fail(
                f"\n\nMissing or incorrect category in {file_path.name}:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @pytest.mark.parametrize("file_path", get_template_parameter_files())
    def test_hyperparameter_params_have_description(self, file_path: Path):
        """
        Verify that algorithm-relevant params have a non-empty description.
        """
        data = load_template_file(file_path)
        errors = []

        for template_name, template_data in data.get("templates", {}).items():
            override_params = template_data.get("recipe_override_parameters", {})

            for param_name, param_config in override_params.items():
                if param_name not in REQUIRED_HYPERPARAMETER_PARAMS:
                    continue

                if "description" not in param_config:
                    errors.append(f"{template_name}::{param_name}: missing 'description' field")
                elif not param_config["description"].strip():
                    errors.append(f"{template_name}::{param_name}: 'description' is empty")

        if errors:
            pytest.fail(
                f"\n\nMissing or empty description in {file_path.name}:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @pytest.mark.parametrize("file_path", get_template_parameter_files())
    def test_config_params_do_not_have_category(self, file_path: Path):
        """
        Verify that config/admin params (paths, mlflow, namespace, etc.)
        do NOT have a category field. This prevents non-hyperparameter
        params from showing up in the UI's hyperparameter sections.
        """
        data = load_template_file(file_path)
        errors = []

        for template_name, template_data in data.get("templates", {}).items():
            override_params = template_data.get("recipe_override_parameters", {})

            for param_name, param_config in override_params.items():
                if param_name in REQUIRED_HYPERPARAMETER_PARAMS:
                    continue

                if "category" in param_config:
                    errors.append(
                        f"{template_name}::{param_name}: config param should "
                        f"not have 'category' field, but has '{param_config['category']}'"
                    )

        if errors:
            pytest.fail(
                f"\n\nConfig params with unexpected category in {file_path.name}:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
