# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
Comprehensive launch.json generation and validation test.

Generates launch.json for all recipes with prefixes: nova, llmft, verl, open_source
Validates metadata against js_schema.json and regional_parameters structure.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
from omegaconf import OmegaConf

# Import the reusable generator from scripts
from scripts.generate_launch_jsons import LaunchJsonGenerator
from tests.launcher.recipe_templatization.launch_json_validators import (
    validate_launch_json_content,
    validate_metadata_schema,
)
from tests.launcher.recipe_templatization.recipe_and_override_params_validators import (
    validate_no_omegaconf_artifacts,
    validate_override_parameter_constraints,
    validate_override_parameter_default_values,
    validate_override_parameters_templatization,
    validate_recipe_fields_presence_in_training_yaml,
)
from tests.launcher.recipe_templatization.validate_field_override_constraints import (
    validate_fields_not_overridden,
)
from tests.launcher.recipe_templatization.validate_replica_overrides import (
    validate_launch_json_replica_overrides,
)

logger = logging.getLogger(__name__)

# Import DONT_OVERRIDE_IN_SMTJ for SM Jobs validation
try:
    from launcher.nova.launchers import DONT_OVERRIDE_IN_SMTJ
except ImportError:
    DONT_OVERRIDE_IN_SMTJ = {"replicas"}
    logger.warning("Could not import DONT_OVERRIDE_IN_SMTJ from launchers.py, using fallback")

# Configuration
RECIPE_PREFIXES = ["llmft", "nova", "verl", "evaluation"]

# DEBUG: Filter recipes by path substring (e.g., "evaluation", "lite", "sft_lora")
# Set to None to test all recipes
DEBUG_RECIPE_PATH = None  # Example: "evaluation/nova" or "nova_lite_v2_sft_lora"

# Valid stages/environments in regional_parameters
VALID_REGIONAL_STAGES = {"prod", "gamma", "beta"}  # For llmft and verl
VALID_REGIONAL_STAGES_NOVA = {"prod", "gamma"}  # Nova excludes beta

# Valid parameter keys in regional_parameters
VALID_REGIONAL_PARAMS = {
    "hp_eks_regional_ecr_uri",
    "smtj_regional_ecr_uri",
    "container_image",
    "smtj_container_image",
    "init_container_image",
    "actor_generation_container_image",
    "rft_generation_container_image",
    "rft_nats_server_container_image",
    "rft_nats_reloader_container_image",
    "rft_storm_container_image",
    "rft_redis_container_image",
}


class LaunchJsonGeneratorWithValidation(LaunchJsonGenerator):
    """
    Extends LaunchJsonGenerator with validation methods for testing.

    Inherits all generation functionality from scripts/generate_launch_jsons.py
    and adds test-specific validation methods.
    """

    def __init__(self):
        super().__init__(working_dir=Path.cwd())
        # Load schema for validation
        self.script_dir = self.working_dir / "tests" / "launcher" / "recipe_templatization"
        self.schema_path = self.script_dir / "js_schema.json"
        with open(self.schema_path) as f:
            self.schema = json.load(f)

    def validate_launch_json(self, launch_json_path: Path, recipe_prefix: str) -> List[str]:
        """Validate launch.json metadata and structure using validators module."""
        validation_errors = []

        try:
            with open(launch_json_path) as f:
                launch_data = json.load(f)

            # Validate metadata section using validators module
            if "metadata" not in launch_data:
                validation_errors.append("Missing 'metadata' section")
            else:
                metadata = launch_data["metadata"]
                validation_errors.extend(validate_metadata_schema(metadata))

            # Validate regional_parameters structure
            if "regional_parameters" in launch_data:
                validation_errors.extend(
                    self.validate_regional_params(launch_data["regional_parameters"], recipe_prefix)
                )

        except json.JSONDecodeError as e:
            validation_errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            validation_errors.append(f"Validation error: {e}")

        return validation_errors

    def validate_regional_params(self, regional_params: Dict, recipe_prefix: str) -> List[str]:
        """Validate regional_parameters structure."""
        errors = []

        for param_name, param_value in regional_params.items():
            if param_name not in VALID_REGIONAL_PARAMS:
                errors.append(f"Unknown regional parameter: {param_name}")
                continue

            if not isinstance(param_value, dict):
                errors.append(f"Regional parameter {param_name} must be dict, got {type(param_value).__name__}")
                continue

            if recipe_prefix in ["llmft", "verl"]:
                for stage_name in param_value.keys():
                    if stage_name not in VALID_REGIONAL_STAGES:
                        errors.append(f"Invalid stage in {param_name}: {stage_name}. Valid: {VALID_REGIONAL_STAGES}")
            elif recipe_prefix == "nova":
                for stage_name in param_value.keys():
                    if stage_name not in VALID_REGIONAL_STAGES_NOVA:
                        errors.append(
                            f"Invalid stage in {param_name}: {stage_name}. Valid: {VALID_REGIONAL_STAGES_NOVA}"
                        )

        return errors

    def validate_serverless_metering_type(self, launch_json: Dict, recipe_name: str) -> List[str]:
        """
        Validate ServerlessMeteringType in launch.json metadata for Nova recipes.

        Validates based on get_serverless_metering_type() which returns:
        - sft, dpo, distill -> TOKEN_BASED
        - rft -> HOURLY
        - eval recipes -> TOKEN_BASED
        - ppo, pretrain -> raises KeyError (field should not be present)
        """
        errors = []
        metadata = launch_json.get("metadata", {})
        serverless_metering_type = metadata.get("ServerlessMeteringType")
        recipe_name_lower = recipe_name.lower()

        training_technique = None
        expected_metering_type = None

        if "sft" in recipe_name_lower:
            training_technique = "sft"
            expected_metering_type = "Token-based"
        elif "dpo" in recipe_name_lower:
            training_technique = "dpo"
            expected_metering_type = "Token-based"
        elif "distill" in recipe_name_lower:
            training_technique = "distill"
            expected_metering_type = "Token-based"
        elif "rft" in recipe_name_lower:
            training_technique = "rft"
            expected_metering_type = "Hourly"
        elif "eval" in recipe_name_lower:
            training_technique = "eval"
            expected_metering_type = "Token-based"
        elif "ppo" in recipe_name_lower:
            training_technique = "ppo"
            expected_metering_type = None
        elif "pretrain" in recipe_name_lower:
            training_technique = "pretrain"
            expected_metering_type = None

        if expected_metering_type is not None:
            if serverless_metering_type is None:
                errors.append(
                    f"[Serverless Metering] ServerlessMeteringType missing for {training_technique} recipe. "
                    f"Expected '{expected_metering_type}'"
                )
            elif serverless_metering_type != expected_metering_type:
                errors.append(
                    f"[Serverless Metering] ServerlessMeteringType mismatch for {training_technique}: "
                    f"expected '{expected_metering_type}', got '{serverless_metering_type}'"
                )
            else:
                logger.info(f"✓ ServerlessMeteringType='{serverless_metering_type}' for {training_technique} recipe")
        else:
            if serverless_metering_type is not None:
                errors.append(
                    f"[Serverless Metering] ServerlessMeteringType should NOT be present for {training_technique} "
                    f"(raises KeyError in get_serverless_metering_type), but found: '{serverless_metering_type}'"
                )
            else:
                logger.info(f"✓ ServerlessMeteringType correctly absent for {training_technique} recipe")

        return errors


def get_test_parameters():
    """Discover recipes and generate test parameters for pytest.mark.parametrize."""
    generator = LaunchJsonGeneratorWithValidation()
    recipes = generator.discover_recipes(
        prefixes=RECIPE_PREFIXES,
        recipe_filter=DEBUG_RECIPE_PATH,
    )

    test_params = []
    for recipe_info in recipes:
        recipe_path, prefix, model_name = recipe_info
        for job_type in ["k8s", "sm_jobs"]:
            if model_name:
                test_id = f"{recipe_path.stem}_{model_name}_{job_type}"
            else:
                test_id = f"{recipe_path.stem}_{job_type}"
            test_params.append((recipe_info, job_type, test_id))

    return test_params


@pytest.fixture
def temp_dir_fixture():
    """Pytest fixture for temporary directory management."""
    temp_dir = tempfile.mkdtemp(prefix="launch_json_test_")
    logger.info(f"Using temporary directory: {temp_dir}")
    yield temp_dir
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


class TestLaunchJsonGenerationAndValidation:
    """Comprehensive test for launch.json generation and validation."""

    @pytest.mark.parametrize(
        "recipe_info,job_type,test_id",
        get_test_parameters(),
        ids=lambda param: param[2] if len(param) > 2 else str(param),
    )
    def test_recipe_generates_valid_launch_json(self, recipe_info, job_type, test_id, temp_dir_fixture):
        """Test that a single recipe generates a valid launch.json file."""
        recipe_path, prefix, model_name = recipe_info
        generator = LaunchJsonGeneratorWithValidation()

        # Generate launch.json - now returns (path, status) tuple
        result = generator.generate_launch_json(recipe_path, job_type, temp_dir_fixture, model_name)

        # Handle the tuple return value
        if result is None:
            pytest.fail("generate_launch_json returned None unexpectedly")

        launch_json_path, status = result

        # Handle status
        if status == "skipped":
            pytest.skip(f"Recipe not supported on {job_type} platform")
        elif status.startswith("failed:"):
            pytest.fail(f"Generation failed: {status}")

        # Assert generation succeeded
        assert launch_json_path is not None, "launch.json path should not be None for successful generation"
        assert launch_json_path.exists(), f"launch.json not found at {launch_json_path}"

        # Validate the generated launch.json using both schema and content validators
        validation_errors = generator.validate_launch_json(launch_json_path, prefix)

        # Load launch.json for content validation
        with open(launch_json_path) as f:
            launch_json = json.load(f)

        # Perform content validation using the validators module
        content_errors = validate_launch_json_content(launch_json, job_type)
        validation_errors.extend(content_errors)

        # SERVERLESS METERING TYPE VALIDATION: For Nova fine-tuning and eval recipes
        recipe_relative_path = str(recipe_path.relative_to(generator.recipes_dir)).lower()
        is_nova_recipe_metering = "nova" in recipe_relative_path

        if is_nova_recipe_metering and job_type == "sm_jobs":
            logger.info(f"Running serverless metering type validation for Nova SM Jobs recipe: {recipe_path.stem}")
            metering_errors = generator.validate_serverless_metering_type(
                launch_json=launch_json, recipe_name=recipe_path.stem
            )
            if metering_errors:
                validation_errors.extend(metering_errors)

        # REPLICA OVERRIDE VALIDATION: For Nova K8s recipes only
        recipe_relative_path = str(recipe_path.relative_to(generator.recipes_dir)).lower()
        is_nova_recipe = "nova" in recipe_relative_path
        is_distill_recipe = "distill" in recipe_relative_path
        is_nova_eval = "eval" in recipe_relative_path

        if is_nova_recipe and job_type == "k8s" and not is_distill_recipe and not is_nova_eval:
            logger.info(f"Running replica override validation for Nova K8s recipe: {recipe_path.stem}")
            replica_is_valid, replica_errors = validate_launch_json_replica_overrides(launch_json_path)
            if not replica_is_valid:
                validation_errors.extend([f"[Replica Override] {e}" for e in replica_errors])
        elif is_distill_recipe:
            logger.info(f"Skipping replica override validation for distill recipe: {recipe_path.stem}")

        # EVAL RECIPE EXCLUSION VALIDATION
        if is_nova_eval and job_type == "k8s":
            logger.info(f"Validating eval recipe has NO {{{{replicas}}}} placeholder: {recipe_path.stem}")
            training_yaml = launch_json.get("training.yaml", "")
            training_config_yaml = launch_json.get("training-config.yaml", "")
            if (
                "{{replicas}}" in training_yaml
                or "replicas_placeholder" in training_yaml
                or "{{replicas}}" in training_config_yaml
                or "replicas_placeholder" in training_config_yaml
            ):
                validation_errors.append(
                    "[Eval Recipe Exclusion] Eval recipe should NOT have {{replicas}} placeholder in training.yaml"
                )

        # SM_JOBS EXCLUSION VALIDATION
        if job_type == "sm_jobs":
            logger.info(f"Validating sm_jobs recipe has NO {{{{replicas}}}} placeholder: {recipe_path.stem}")
            training_recipe_yaml = launch_json.get("training_recipe.yaml", "")
            if "{{replicas}}" in training_recipe_yaml:
                validation_errors.append(
                    "[SM Jobs Exclusion] SM Jobs recipe should NOT have {{replicas}} placeholder in training_recipe.yaml"
                )

        # DONT_OVERRIDE_IN_SMTJ VALIDATION: For Nova SM Jobs recipes only
        recipe_relative_path = str(recipe_path.relative_to(generator.recipes_dir)).lower()
        is_nova_recipe_smtj = "nova" in recipe_relative_path

        if is_nova_recipe_smtj and job_type == "sm_jobs":
            logger.info(f"Running DONT_OVERRIDE_IN_SMTJ validation for Nova SM Jobs recipe: {recipe_path.stem}")
            fields_valid, field_errors = validate_fields_not_overridden(
                launch_json=launch_json,
                original_recipe_path=recipe_path,
                fields_to_validate=DONT_OVERRIDE_IN_SMTJ,
                yaml_content_key="training_recipe.yaml",
                validation_label="DONT_OVERRIDE_IN_SMTJ",
            )
            if not fields_valid:
                validation_errors.extend(field_errors)

        # NEW VALIDATORS: Recipe and override parameter validation
        is_nova_ppo = "ppo" in str(recipe_path).lower() and "nova" in str(recipe_path).lower()

        if job_type == "k8s":
            training_content_key = "training-config.yaml"
        else:
            training_content_key = "training_recipe.yaml"

        if training_content_key in launch_json:
            try:
                recipe_cfg = OmegaConf.load(recipe_path)
                wrapped_recipe = OmegaConf.create({"recipes": recipe_cfg})
                OmegaConf.update(wrapped_recipe, "base_results_dir", "/tmp/test_results", merge=False)
                OmegaConf.resolve(wrapped_recipe)
                original_recipe = OmegaConf.to_container(wrapped_recipe.recipes, resolve=True)

                training_yaml_content = launch_json.get(training_content_key, "")
                override_params = launch_json.get("recipe_override_parameters", {})
                recipe_file_str = str(recipe_path)

                omegaconf_errors = validate_no_omegaconf_artifacts(training_yaml_content, recipe_file_str)
                validation_errors.extend(omegaconf_errors)

                field_presence_errors = validate_recipe_fields_presence_in_training_yaml(
                    original_recipe, training_yaml_content, recipe_file_str
                )
                validation_errors.extend(field_presence_errors)

                content_to_check = training_yaml_content
                if is_nova_ppo and "training.yaml" in launch_json:
                    content_to_check = launch_json.get("training.yaml", "") + training_yaml_content

                templatization_errors = validate_override_parameters_templatization(
                    override_params, content_to_check, job_type, recipe_file_str
                )
                validation_errors.extend(templatization_errors)

                is_nova_distill = "distill" in recipe_file_str.lower() and "nova" in recipe_file_str.lower()

                if is_nova_distill:
                    logger.warning(f"Skipping constraint validation for Nova distill recipe: {recipe_path.stem}")
                else:
                    constraint_errors = validate_override_parameter_constraints(override_params)
                    validation_errors.extend(constraint_errors)

                try:
                    from launcher.recipe_templatization.evaluation.evaluation_recipe_template_processor import (
                        EvaluationRecipeTemplateProcessor,
                    )
                    from launcher.recipe_templatization.llmft.llmft_recipe_template_processor import (
                        LLMFTRecipeTemplateProcessor,
                    )
                    from launcher.recipe_templatization.nova.nova_recipe_template_processor import (
                        NovaRecipeTemplateProcessor,
                    )
                    from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
                        VerlRecipeTemplateProcessor,
                    )

                    recipe_processor = None
                    recipe_template = None

                    if "nova" in recipe_file_str.lower():
                        recipe_processor = NovaRecipeTemplateProcessor(original_recipe, platform=job_type)
                    elif "llmft" in recipe_file_str.lower():
                        recipe_processor = LLMFTRecipeTemplateProcessor(original_recipe, platform=job_type)
                    elif "verl" in recipe_file_str.lower():
                        recipe_processor = VerlRecipeTemplateProcessor(original_recipe, platform=job_type)
                    elif "eval" in recipe_file_str.lower():
                        recipe_processor = EvaluationRecipeTemplateProcessor(original_recipe, platform=job_type)

                    if recipe_processor and hasattr(recipe_processor, "matched_template"):
                        recipe_template = recipe_processor.matched_template

                        if recipe_template:
                            default_value_errors = validate_override_parameter_default_values(
                                original_recipe, override_params, recipe_template, recipe_file_str
                            )
                            validation_errors.extend(default_value_errors)
                except Exception as e:
                    logger.debug(f"Could not validate default values for {recipe_path.stem}: {e}")

            except Exception as e:
                validation_errors.append(f"Recipe/override validators error: {str(e)}")

        # Copy failed launch.json for debugging if there are validation errors
        if validation_errors:
            try:
                failed_launch_json_name = f"failed_launch_json_{recipe_path.stem}_{job_type}.json"
                failed_launch_json_dest = generator.working_dir / failed_launch_json_name
                shutil.copy2(launch_json_path, failed_launch_json_dest)
                logger.error(f"Copied failed launch.json to: {failed_launch_json_dest}")
            except Exception as copy_error:
                logger.warning(f"Failed to copy launch.json for debugging: {copy_error}")

        assert not validation_errors, f"Validation errors:\n" + "\n".join(f"  - {err}" for err in validation_errors)
