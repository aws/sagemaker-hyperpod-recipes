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
import os
import random
import shutil
import string
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from omegaconf import OmegaConf

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

# Import DONT_OVERRIDE_IN_SMTJ for SM Jobs validation
try:
    from launcher.nova.launchers import DONT_OVERRIDE_IN_SMTJ
except ImportError:
    # Fallback if import fails
    DONT_OVERRIDE_IN_SMTJ = {"replicas"}
    logger.warning("Could not import DONT_OVERRIDE_IN_SMTJ from launchers.py, using fallback")

logger = logging.getLogger(__name__)

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
    "hp_eks_regional_ecr_uri",  # Nova/LLMFT K8s
    "smtj_regional_ecr_uri",  # Nova/LLMFT SM Jobs
    "container_image",  # Evaluation K8s
    "smtj_container_image",  # Evaluation SM Jobs
    "init_container_image",
    "actor_generation_container_image",
    "rft_generation_container_image",
    "rft_nats_server_container_image",
    "rft_nats_reloader_container_image",
    "rft_storm_container_image",
    "rft_redis_container_image",
}


class LaunchJsonTestResult:
    """Track test results for a single recipe."""

    def __init__(self, recipe_name: str, job_type: str):
        self.recipe_name = recipe_name
        self.job_type = job_type
        self.success = False
        self.skipped = False  # New field for platform-specific skips
        self.error_message = None
        self.launch_json_path = None
        self.validation_errors = []

    def to_dict(self):
        return {
            "recipe_name": self.recipe_name,
            "job_type": self.job_type,
            "success": self.success,
            "skipped": self.skipped,
            "error_message": self.error_message,
            "launch_json_path": str(self.launch_json_path) if self.launch_json_path else None,
            "validation_errors": self.validation_errors,
        }


class LaunchJsonGenerator:
    """Generate and validate launch.json files for all recipes."""

    def __init__(self):
        self.current_dir = Path.cwd()
        self.script_dir = self.current_dir / "tests" / "launcher" / "recipe_templatization"
        self.recipes_dir = self.current_dir / "recipes_collection" / "recipes"
        self.schema_path = self.script_dir / "js_schema.json"
        self.unsupported_recipes_path = (
            self.current_dir / "launcher" / "recipe_templatization" / "unsupported_recipes_list.json"
        )
        self.eval_regional_params_path = (
            self.current_dir
            / "launcher"
            / "recipe_templatization"
            / "evaluation"
            / "evaluation_regional_parameters.json"
        )
        self.temp_dir = None

        # Load schema
        with open(self.schema_path) as f:
            self.schema = json.load(f)

        # Load unsupported recipes list
        self.unsupported_recipes = set()
        if self.unsupported_recipes_path.exists():
            with open(self.unsupported_recipes_path) as f:
                unsupported_data = json.load(f)
                # Flatten all unsupported recipes from all categories
                for category_recipes in unsupported_data.values():
                    for recipe_path in category_recipes:
                        # Convert to Path for comparison
                        self.unsupported_recipes.add(Path(recipe_path))

        # Load evaluation regional parameters for model mappings
        self.eval_model_mapping = {}
        if self.eval_regional_params_path.exists():
            with open(self.eval_regional_params_path) as f:
                eval_data = json.load(f)
                self.eval_model_mapping = eval_data.get("js_model_name_instance_mapping", {})

    def discover_recipes(self) -> List[Tuple[Path, str, Optional[str]]]:
        """
        Discover all recipes matching the prefixes.

        Returns list of tuples: (recipe_path, prefix, model_name)
        - For most recipes: model_name is None
        - For open_source evaluation recipes: expanded into one tuple per model
        """
        recipes = []
        excluded_count = 0

        for prefix in RECIPE_PREFIXES:
            # Handle special case for evaluation - search only in open-source directory
            if prefix == "evaluation":
                search_dir = self.recipes_dir / "evaluation" / "open-source"
                # For evaluation, search all yaml files in open-source directory
                recipe_files = search_dir.glob("*.yaml")
            else:
                # For other prefixes, recursively search for files matching the prefix
                search_dir = self.recipes_dir
                recipe_files = search_dir.rglob(f"*{prefix}*.yaml")

            if not search_dir.exists():
                continue

            # Process found recipe files
            for recipe_file in recipe_files:
                # Apply DEBUG_RECIPE_PATH filter if set
                if DEBUG_RECIPE_PATH is not None:
                    # Check if path contains the debug filter string
                    if DEBUG_RECIPE_PATH not in str(recipe_file):
                        continue

                # Check if recipe is in unsupported list
                # Compare as relative path from project root for matching
                recipe_relative = recipe_file.relative_to(self.current_dir)
                if recipe_relative in self.unsupported_recipes:
                    excluded_count += 1
                    logger.info(f"Excluding unsupported recipe: {recipe_relative}")
                    continue

                # For evaluation prefix (open-source), expand into one test per model
                if prefix == "evaluation":
                    # Expand into one test per model
                    for model_name in self.eval_model_mapping.keys():
                        recipes.append((recipe_file, prefix, model_name))
                else:
                    # Regular recipe - no model expansion
                    recipes.append((recipe_file, prefix, None))

        # Count unique recipe files vs total tests after expansion
        unique_recipe_files = len(set(r[0] for r in recipes))
        total_tests = len(recipes)

        logger.info(f"Discovered {unique_recipe_files} recipe file(s)")
        logger.info(f"Generated {total_tests} recipe tests after expansion (recipes × models × platforms)")
        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} unsupported recipes from testing")
        if DEBUG_RECIPE_PATH:
            logger.info(f"DEBUG_RECIPE_PATH filter active: '{DEBUG_RECIPE_PATH}'")
        return sorted(recipes, key=lambda x: (x[0].name, x[2] or ""))

    @staticmethod
    def generate_valid_k8s_name(recipe_name: str) -> str:
        """
        Generate a K8s-compliant run name from recipe filename.

        Follows recipe naming pattern (e.g., llama-3-1-8b-instruct) but ensures:
        - DNS-1123 compliant (lowercase alphanumeric + hyphens only)
        - Max 53 characters (Helm limitation)
        - Adds random 5-char suffix to prevent collisions

        Since valid_run_name() in main.py doesn't run in pytest mode, we handle it here.
        """
        # Generate 5-char random hash (same as main.py's valid_run_name)
        random_hash = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))

        # Convert recipe name to K8s-compliant format
        # Replace underscores with hyphens, convert to lowercase
        k8s_name = recipe_name.lower().replace("_", "-")

        # Remove any non-DNS-1123 characters (keep only lowercase alphanumeric and hyphens)
        k8s_name = "".join(c if c.isalnum() or c == "-" else "-" for c in k8s_name)

        # Remove consecutive hyphens
        while "--" in k8s_name:
            k8s_name = k8s_name.replace("--", "-")

        # Strip leading/trailing hyphens
        k8s_name = k8s_name.strip("-")

        # Truncate to 47 chars to leave room for "-{hash}" (47 + 1 + 5 = 53 total)
        if len(k8s_name) > 47:
            k8s_name = k8s_name[:47].rstrip("-")

        # Append random hash (same pattern as main.py)
        return f"{k8s_name}-{random_hash}"

    def get_nova_params(
        self,
        recipe_path: Path,
        recipe_rel_path: Path,
        job_type: str,
        run_name: str,
        recipe_temp_dir: str,
        instance_type: str,
    ) -> List[str]:
        """Generate Nova-specific parameters."""
        is_evaluation = "/evaluation/" in str(recipe_path)

        # For Nova evaluation recipes, use minimal parameters
        if is_evaluation:
            if job_type == "k8s":
                return [
                    f"recipes={recipe_rel_path}",
                    f"instance_type={instance_type}",
                    f"base_results_dir={recipe_temp_dir}",
                    "cluster=k8s",
                    "cluster_type=k8s",
                    "launch_json=true",
                    "container=test_container",
                ]
            else:  # sm_jobs
                sm_instance_type = (
                    instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
                )
                return [
                    f"recipes={recipe_rel_path}",
                    f"instance_type={sm_instance_type}",
                    f"base_results_dir={recipe_temp_dir}",
                    "cluster=sm_jobs",
                    "cluster_type=sm_jobs",
                    "launch_json=true",
                    "container=test_container",
                ]

        # For Nova fine-tuning/training recipes
        if job_type == "k8s":
            return [
                f"recipes={recipe_rel_path}",
                f"instance_type={instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={recipe_temp_dir}",
                "container=test_container",
                "cluster=k8s",
                "cluster_type=k8s",
                "launch_json=true",
            ]
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            return [
                f"recipes={recipe_rel_path}",
                "cluster=sm_jobs",
                "cluster_type=sm_jobs",
                f"instance_type={sm_instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={recipe_temp_dir}",
                "+cluster.sm_jobs_config.output_path=s3://test_path",
                "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                "container=test_container",
                "+env_vars.NEMO_LAUNCHER_DEBUG=1",
                "git.use_default=false",
                "git.entry_script=/app/src/train_hp.py",
                "launch_json=true",
            ]

    def get_llmft_params(
        self, recipe_rel_path: Path, job_type: str, run_name: str, recipe_temp_dir: str, instance_type: str
    ) -> List[str]:
        """Generate LLMFT-specific parameters."""
        if job_type == "k8s":
            return [
                f"recipes={recipe_rel_path}",
                f"instance_type={instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={recipe_temp_dir}",
                "container=test_container",
                "git.use_default=false",
                "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
                "+cluster.persistent_volume_claims.0.mountPath=/data",
                "cluster=k8s",
                "cluster_type=k8s",
                "launch_json=true",
            ]
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            return [
                f"recipes={recipe_rel_path}",
                "cluster=sm_jobs",
                "cluster_type=sm_jobs",
                f"instance_type={sm_instance_type}",
                f"recipes.run.name={run_name}",
                f"base_results_dir={recipe_temp_dir}",
                "+cluster.sm_jobs_config.output_path=s3://test_path",
                "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                "container=test_container",
                "+env_vars.NEMO_LAUNCHER_DEBUG=1",
                "git.use_default=false",
                "git.entry_script=/app/src/train_hp.py",
                "launch_json=true",
            ]

    def get_open_source_eval_params(
        self, recipe_path: Path, recipe_rel_path: Path, job_type: str, recipe_temp_dir: str, model_name: str
    ) -> List[str]:
        """Generate open source evaluation-specific parameters."""
        # Get instance type for this model
        instance_types = self.eval_model_mapping.get(model_name, ["ml.p5.48xlarge"])
        instance_type = instance_types[0]  # Use first supported instance type

        # Generate eval job name
        eval_job_name = f"eval-{model_name}-job"

        # Base parameters for open source evaluation
        base_params = [
            f"recipes={recipe_rel_path}",
            f"cluster_type={job_type}",
            f"cluster={job_type}",
            f"base_results_dir={recipe_temp_dir}",
            f"recipes.run.base_model_name={model_name}",
            f"recipes.run.name={eval_job_name}",
            "recipes.output.eval_results_dir=''",
            "container=test_container",
            "git.use_default=false",
            "launch_json=true",
        ]

        # Recipe-specific parameters
        recipe_name = recipe_path.stem.lower()
        if "llmaj" in recipe_name:
            base_params.append("recipes.run.inference_data_s3_path=''")
        elif "deterministic" in recipe_name:
            base_params.extend(
                [
                    "recipes.run.model_name_or_path=''",
                    "recipes.run.data_s3_path=''",
                ]
            )

        # Platform-specific parameters
        if job_type == "k8s":
            base_params.extend(
                [
                    f"instance_type={instance_type}",
                    "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
                    "+cluster.persistent_volume_claims.0.mountPath=/data",
                ]
            )
        else:  # sm_jobs
            sm_instance_type = instance_type.replace("ml.", "") if instance_type.startswith("ml.") else instance_type
            base_params.extend(
                [
                    f"instance_type={sm_instance_type}",
                    "+cluster.sm_jobs_config.output_path=s3://test_path",
                    "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
                    "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
                ]
            )

        return base_params

    def get_job_params(
        self, recipe_path: Path, job_type: str, run_name: str, recipe_temp_dir: str, model_name: Optional[str] = None
    ) -> List[str]:
        """Generate job parameters for launch.json generation."""
        recipe_rel_path = recipe_path.relative_to(self.recipes_dir).with_suffix("")

        # Route to recipe-specific parameter generation
        recipe_name = recipe_path.stem.lower()

        # Check if this is an open source evaluation recipe
        if model_name is not None:
            return self.get_open_source_eval_params(recipe_path, recipe_rel_path, job_type, recipe_temp_dir, model_name)

        # For non-open-source recipes, read recipe file to extract instance_type
        with open(recipe_path, "r") as f:
            recipe_data = yaml.safe_load(f)

        # Get instance_type from recipe (use first one if multiple)
        instance_types = recipe_data.get("instance_types", ["ml.p5.48xlarge"])
        instance_type = instance_types[0] if instance_types else "ml.p5.48xlarge"

        if "nova" in recipe_name:
            return self.get_nova_params(
                recipe_path, recipe_rel_path, job_type, run_name, recipe_temp_dir, instance_type
            )
        elif "llmft" in recipe_name:
            return self.get_llmft_params(recipe_rel_path, job_type, run_name, recipe_temp_dir, instance_type)
        else:
            # Default parameters for other recipe types
            return self.get_llmft_params(recipe_rel_path, job_type, run_name, recipe_temp_dir, instance_type)

    def generate_launch_json(
        self, recipe_path: Path, job_type: str, model_name: Optional[str] = None
    ) -> LaunchJsonTestResult:
        """
        Generate launch.json for a single recipe using subprocess for parallel execution.

        Uses subprocess instead of calling main() directly to avoid GlobalHydra conflicts.
        Each subprocess has its own GlobalHydra instance, enabling true parallel execution.

        Args:
            recipe_path: Path to the recipe file
            job_type: Either "k8s" or "sm_jobs"
            model_name: For open source eval recipes, the model to test with
        """
        recipe_name = recipe_path.stem
        # For open source eval, include model in the name
        if model_name:
            display_name = f"{recipe_name}_{model_name}"
        else:
            display_name = recipe_name

        # Generate short K8s-compliant name
        base_name = display_name.lower().replace("_", "-")
        # Truncate to 53 chars max
        if len(base_name) > 53:
            base_name = base_name[:53].rstrip("-")
        run_name = base_name
        result = LaunchJsonTestResult(display_name, job_type)

        # Create unique temp directory for this recipe within main temp dir
        # This prevents prefix collision when multiple recipes have similar names
        recipe_temp_dir = tempfile.mkdtemp(prefix=f"{recipe_name[:20]}_{job_type}_", dir=self.temp_dir)

        try:
            job_params = self.get_job_params(recipe_path, job_type, run_name, recipe_temp_dir, model_name)
            cmd = ["python3", "main.py"] + job_params

            env = os.environ.copy()
            env["HYDRA_FULL_ERROR"] = "1"
            env["AWS_REGION"] = "us-east-1"

            proc_result = subprocess.run(
                cmd, cwd=self.current_dir, capture_output=True, text=True, env=env, timeout=180
            )

            # Log subprocess output
            if proc_result.stdout:
                logger.debug(f"STDOUT from {display_name}:\n{proc_result.stdout}")
            if proc_result.stderr:
                logger.debug(f"STDERR from {display_name}:\n{proc_result.stderr}")

            # Check if recipe is not supported on this specific platform
            combined_output = proc_result.stdout + proc_result.stderr
            if (
                "No regional parameters found" in combined_output
                or "No supported regional parameters for platform" in combined_output
                or "No container image found for platform" in combined_output
                or "has not been implemented yet" in combined_output
            ):
                result.success = True
                result.skipped = True
                result.error_message = f"Skipped: Recipe not supported on {job_type} platform"
                return result

            if proc_result.returncode != 0:
                result.error_message = f"Generation failed: {proc_result.stderr[:500]}"
                return result

            # Parse output directory from stdout if present
            # Format: "Outputs are in: /path/to/directory"
            output_dir = None
            for line in proc_result.stdout.split("\n"):
                if "Outputs are in:" in line:
                    output_dir = line.split("Outputs are in:")[1].strip()
                    break

            # Find generated directory in recipe-specific temp dir
            recipe_temp_path = Path(recipe_temp_dir)

            if output_dir:
                # Use the directory from stdout
                run_dir = Path(output_dir)
                if not run_dir.exists():
                    # Directory doesn't exist - this is a skip scenario
                    result.success = True
                    result.skipped = True
                    result.error_message = (
                        f"Skipped: Output directory not created (recipe printed path but didn't generate files)"
                    )
                    return result
            else:
                # Fallback: search for created directories
                created_dirs = [d for d in recipe_temp_path.iterdir() if d.is_dir()]

                if not created_dirs:
                    result.error_message = (
                        f"No directory created in {recipe_temp_dir}\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"Return code: {proc_result.returncode}\n"
                        f"STDOUT: {proc_result.stdout[:500]}\n"
                        f"STDERR: {proc_result.stderr[:500]}"
                    )
                    return result

                # Should only be one directory per recipe
                run_dir = created_dirs[0]

            # Debug: Check what's in the run directory
            run_dir_contents = list(run_dir.iterdir())

            # For K8s, look in k8s_templates subdirectory (note: plural), but also check alternatives
            if job_type == "k8s":
                # Try k8s_templates (plural) first
                launch_json_path = run_dir / "k8s_templates" / "launch.json"
                if not launch_json_path.exists():
                    # Try k8s_template (singular)
                    launch_json_path = run_dir / "k8s_template" / "launch.json"
                    if not launch_json_path.exists():
                        # Try direct in run_dir
                        launch_json_path = run_dir / "launch.json"
                        if not launch_json_path.exists():
                            # Debug info
                            result.error_message = (
                                f"launch.json not found. Tried:\n"
                                f"  1. {run_dir / 'k8s_templates' / 'launch.json'}\n"
                                f"  2. {run_dir / 'k8s_template' / 'launch.json'}\n"
                                f"  3. {run_dir / 'launch.json'}\n"
                                f"run_dir contents: {[str(p.name) for p in run_dir_contents]}"
                            )
                            return result
            else:
                launch_json_path = run_dir / "launch.json"
                if not launch_json_path.exists():
                    result.error_message = (
                        f"launch.json not found at {launch_json_path}\n"
                        f"run_dir contents: {[str(p.name) for p in run_dir_contents]}"
                    )
                    return result

            result.launch_json_path = launch_json_path
            result.success = True

        except subprocess.TimeoutExpired:
            result.error_message = "Timeout after 3 minutes"
        except Exception as e:
            result.error_message = f"Exception: {str(e)}"

        return result

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
            # Check if param is valid
            if param_name not in VALID_REGIONAL_PARAMS:
                errors.append(f"Unknown regional parameter: {param_name}")
                continue

            if not isinstance(param_value, dict):
                errors.append(f"Regional parameter {param_name} must be dict, got {type(param_value).__name__}")
                continue

            # Validate stages based on recipe type
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

        Args:
            launch_json: The complete launch.json content
            recipe_name: The recipe filename (stem)

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        metadata = launch_json.get("metadata", {})
        serverless_metering_type = metadata.get("ServerlessMeteringType")
        recipe_name_lower = recipe_name.lower()

        # Determine training technique and expected metering type
        # Match the order of checks in get_serverless_metering_type()
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

        # Validation
        if expected_metering_type is not None:
            # ServerlessMeteringType should be present with correct value
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
            # ServerlessMeteringType should NOT be present
            if serverless_metering_type is not None:
                errors.append(
                    f"[Serverless Metering] ServerlessMeteringType should NOT be present for {training_technique} "
                    f"(raises KeyError in get_serverless_metering_type), but found: '{serverless_metering_type}'"
                )
            else:
                logger.info(f"✓ ServerlessMeteringType correctly absent for {training_technique} recipe")

        return errors

    def process_recipe(self, recipe_info: Tuple[Path, str, Optional[str]], job_type: str) -> LaunchJsonTestResult:
        """Process a single recipe: generate and validate launch.json."""
        recipe_path, prefix, model_name = recipe_info
        result = self.generate_launch_json(recipe_path, job_type, model_name)

        if result.success and result.launch_json_path and not result.skipped:
            # Validate the generated launch.json using both schema and content validators
            validation_errors = self.validate_launch_json(result.launch_json_path, prefix)

            # Load launch.json for content validation
            try:
                with open(result.launch_json_path) as f:
                    launch_json = json.load(f)

                # Perform content validation using the validators module
                content_errors = validate_launch_json_content(launch_json, job_type)
                validation_errors.extend(content_errors)

                # SERVERLESS METERING TYPE VALIDATION: For Nova fine-tuning and eval recipes
                # Validates ServerlessMeteringType based on get_serverless_metering_type() logic
                recipe_relative_path = str(recipe_path.relative_to(self.recipes_dir)).lower()
                is_nova_recipe_metering = "nova" in recipe_relative_path

                # Run validation ONLY for Nova SM Jobs recipes (serverless metering is SM Jobs specific)
                if is_nova_recipe_metering and job_type == "sm_jobs":
                    logger.info(
                        f"Running serverless metering type validation for Nova SM Jobs recipe: {recipe_path.stem}"
                    )
                    metering_errors = self.validate_serverless_metering_type(
                        launch_json=launch_json, recipe_name=recipe_path.stem
                    )
                    if metering_errors:
                        validation_errors.extend(metering_errors)

                # REPLICA OVERRIDE VALIDATION: For Nova K8s recipes only
                # Validates Master replicas always = 1 and Worker replicas are correctly templatized
                # Skip distill recipes (they use CPU instances and have different structure)
                # Check only the recipe's relative path from recipes directory, not absolute path
                # to avoid matching directory names in the filesystem path
                recipe_relative_path = str(recipe_path.relative_to(self.recipes_dir)).lower()
                is_nova_recipe = "nova" in recipe_relative_path
                is_distill_recipe = "distill" in recipe_relative_path
                is_nova_eval = "eval" in recipe_relative_path

                if is_nova_recipe and job_type == "k8s" and not is_distill_recipe and not is_nova_eval:
                    logger.info(f"Running replica override validation for Nova K8s recipe: {recipe_path.stem}")
                    replica_is_valid, replica_errors = validate_launch_json_replica_overrides(result.launch_json_path)
                    if not replica_is_valid:
                        validation_errors.extend([f"[Replica Override] {e}" for e in replica_errors])
                elif is_distill_recipe:
                    logger.info(f"Skipping replica override validation for distill recipe: {recipe_path.stem}")

                # EVAL RECIPE EXCLUSION VALIDATION: Ensure eval recipes don't have {{replicas}} placeholder
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

                # SM_JOBS EXCLUSION VALIDATION: Ensure sm_jobs recipes don't have {{replicas}} placeholder
                if job_type == "sm_jobs":
                    logger.info(f"Validating sm_jobs recipe has NO {{{{replicas}}}} placeholder: {recipe_path.stem}")
                    # For sm_jobs, check training_recipe.yaml if it exists
                    training_recipe_yaml = launch_json.get("training_recipe.yaml", "")
                    if "{{replicas}}" in training_recipe_yaml:
                        validation_errors.append(
                            "[SM Jobs Exclusion] SM Jobs recipe should NOT have {{replicas}} placeholder in training_recipe.yaml"
                        )

                # DONT_OVERRIDE_IN_SMTJ VALIDATION: For Nova SM Jobs recipes only
                # Ensures fields in DONT_OVERRIDE_IN_SMTJ are not templatized and match default values
                # Check only the recipe's relative path from recipes directory, not absolute path
                recipe_relative_path = str(recipe_path.relative_to(self.recipes_dir)).lower()
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
                # For K8s: use training.yaml, For SM Jobs: use training_recipe.yaml
                # For Nova PPO: use training-config.yaml (recipe content is in ConfigMaps)
                is_nova_ppo = "ppo" in str(recipe_path).lower() and "nova" in str(recipe_path).lower()

                if job_type == "k8s":
                    training_content_key = "training-config.yaml"
                else:
                    training_content_key = "training_recipe.yaml"

                if training_content_key in launch_json:
                    try:
                        # Load original recipe and wrap under "recipes" key for proper interpolation
                        # Interpolations like ${recipes.run.rm_replicas} require this structure
                        recipe_cfg = OmegaConf.load(recipe_path)
                        wrapped_recipe = OmegaConf.create({"recipes": recipe_cfg})
                        # Set base_results_dir for interpolation resolution (LLMFT recipes use ${base_results_dir})
                        OmegaConf.update(wrapped_recipe, "base_results_dir", "/tmp/test_results", merge=False)
                        OmegaConf.resolve(wrapped_recipe)
                        original_recipe = OmegaConf.to_container(wrapped_recipe.recipes, resolve=True)

                        # Get data from launch.json
                        training_yaml_content = launch_json.get(training_content_key, "")
                        override_params = launch_json.get("recipe_override_parameters", {})

                        recipe_file_str = str(recipe_path)

                        # 1. Check for OmegaConf artifacts in training yaml
                        omegaconf_errors = validate_no_omegaconf_artifacts(training_yaml_content, recipe_file_str)
                        validation_errors.extend(omegaconf_errors)

                        # 2. Validate all recipe fields are present in training yaml
                        field_presence_errors = validate_recipe_fields_presence_in_training_yaml(
                            original_recipe, training_yaml_content, recipe_file_str
                        )
                        validation_errors.extend(field_presence_errors)

                        # 3. Validate override parameters are properly templatized
                        # Use both training.yaml and training-config.yaml for templatization check
                        content_to_check = training_yaml_content
                        if is_nova_ppo and "training.yaml" in launch_json:
                            # For PPO, check both files since params can be in either
                            content_to_check = launch_json.get("training.yaml", "") + training_yaml_content

                        templatization_errors = validate_override_parameters_templatization(
                            override_params, content_to_check, job_type, recipe_file_str
                        )
                        validation_errors.extend(templatization_errors)

                        # 4. Validate override parameter constraints
                        # Skip for Nova distill recipes (known type mismatch: string defaults with integer type)
                        is_nova_distill = "distill" in recipe_file_str.lower() and "nova" in recipe_file_str.lower()

                        if is_nova_distill:
                            logger.warning(
                                f"Skipping constraint validation for Nova distill recipe: {recipe_path.stem}"
                            )
                        else:
                            constraint_errors = validate_override_parameter_constraints(override_params)
                            validation_errors.extend(constraint_errors)

                        # 5. Validate default values match original recipe (requires recipe_template)
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

                            # Determine recipe type and create processor
                            if "nova" in recipe_file_str.lower():
                                recipe_processor = NovaRecipeTemplateProcessor(original_recipe, platform=job_type)
                            elif "llmft" in recipe_file_str.lower():
                                recipe_processor = LLMFTRecipeTemplateProcessor(original_recipe, platform=job_type)
                            elif "verl" in recipe_file_str.lower():
                                recipe_processor = VerlRecipeTemplateProcessor(original_recipe, platform=job_type)
                            elif "eval" in recipe_file_str.lower():
                                recipe_processor = EvaluationRecipeTemplateProcessor(original_recipe, platform=job_type)

                            # Get matched template
                            if recipe_processor and hasattr(recipe_processor, "matched_template"):
                                recipe_template = recipe_processor.matched_template

                                if recipe_template:
                                    # Validate default values
                                    default_value_errors = validate_override_parameter_default_values(
                                        original_recipe, override_params, recipe_template, recipe_file_str
                                    )
                                    validation_errors.extend(default_value_errors)
                        except Exception as e:
                            # Don't fail test if default value validation has issues
                            logger.debug(f"Could not validate default values for {recipe_path.stem}: {e}")

                    except Exception as e:
                        validation_errors.append(f"Recipe/override validators error: {str(e)}")

            except Exception as e:
                validation_errors.append(f"Content validation error: {str(e)}")

            if validation_errors:
                result.success = False
                result.validation_errors = validation_errors

                # Copy failed launch.json to project root for easier debugging
                try:
                    failed_launch_json_name = f"failed_launch_json_{result.recipe_name}_{job_type}.json"
                    failed_launch_json_dest = self.current_dir / failed_launch_json_name
                    shutil.copy2(result.launch_json_path, failed_launch_json_dest)
                    logger.error(f"Copied failed launch.json to: {failed_launch_json_dest}")
                except Exception as copy_error:
                    logger.warning(f"Failed to copy launch.json for debugging: {copy_error}")

        return result

    def run_all_tests(self):
        """Run tests sequentially, stopping on first failure for easier debugging."""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="launch_json_test_")
        logger.info(f"Using temporary directory: {self.temp_dir}")

        try:
            recipes = self.discover_recipes()

            # Create test jobs for both k8s and sm_jobs
            test_jobs = []
            for recipe_info in recipes:
                for job_type in ["k8s", "sm_jobs"]:
                    test_jobs.append((recipe_info, job_type))

            logger.info(f"Total test jobs to run: {len(test_jobs)}")

            results = []
            total_jobs = len(test_jobs)

            # Track recipe-to-platform mapping
            # Key: recipe file path (stem), Value: set of successful platforms
            recipe_platform_mapping: Dict[str, Set[str]] = defaultdict(set)

            # Run tests sequentially and stop on first ACTUAL failure (not skips)
            for idx, (recipe_info, job_type) in enumerate(test_jobs, 1):
                # Use print() so progress shows in pytest output without -s flag
                recipe_path, prefix, model_name = recipe_info
                display_name = f"{recipe_path.stem}_{model_name}" if model_name else recipe_path.stem
                print(f"[{idx}/{total_jobs}] Testing {display_name} ({job_type})...")
                logger.info(f"[{idx}/{total_jobs}] Testing {display_name} ({job_type})...")
                result = self.process_recipe(recipe_info, job_type)
                results.append(result)

                # Track successful platforms for this recipe file
                if result.success and not result.skipped:
                    recipe_platform_mapping[recipe_path.stem].add(job_type)

                if result.success:
                    if result.skipped:
                        print(f"⊘ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type}) - SKIPPED")
                        logger.info(f"⊘ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type}) - SKIPPED")
                    else:
                        print(f"✓ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type})")
                        logger.info(f"✓ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type})")
                else:
                    # Actual failure - stop immediately
                    print(f"✗ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type}): {result.error_message}")
                    logger.error(
                        f"✗ [{idx}/{total_jobs}] {result.recipe_name} ({result.job_type}): {result.error_message}"
                    )
                    if result.validation_errors:
                        for err in result.validation_errors:
                            print(f"  Validation error: {err}")
                            logger.error(f"  Validation error: {err}")
                    # Stop immediately on first ACTUAL failure
                    print(f"Stopping on first failure after {idx}/{total_jobs} tests")
                    logger.error(f"Stopping on first failure after {idx}/{total_jobs} tests")
                    break

            return results, recipe_platform_mapping

        finally:
            # Cleanup temp directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    def generate_report(
        self, results: List[LaunchJsonTestResult], recipe_platform_mapping: Dict[str, Set[str]]
    ) -> Dict:
        """Generate test report with recipe coverage analysis."""
        successful = [r for r in results if r.success and not r.skipped]
        skipped = [r for r in results if r.success and r.skipped]
        failed = [r for r in results if not r.success]

        # Analyze recipe coverage
        recipes_with_k8s = [recipe for recipe, platforms in recipe_platform_mapping.items() if "k8s" in platforms]
        recipes_with_sm_jobs = [
            recipe for recipe, platforms in recipe_platform_mapping.items() if "sm_jobs" in platforms
        ]
        recipes_with_both = [recipe for recipe, platforms in recipe_platform_mapping.items() if len(platforms) == 2]
        recipes_with_one = [recipe for recipe, platforms in recipe_platform_mapping.items() if len(platforms) == 1]
        recipes_unsupported = [recipe for recipe, platforms in recipe_platform_mapping.items() if len(platforms) == 0]

        report = {
            "total": len(results),
            "successful": len(successful),
            "skipped": len(skipped),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "recipe_coverage": {
                "total_unique_recipes": len(recipe_platform_mapping),
                "recipes_with_k8s_support": len(recipes_with_k8s),
                "recipes_with_sm_jobs_support": len(recipes_with_sm_jobs),
                "recipes_with_both_platforms": len(recipes_with_both),
                "recipes_with_one_platform": len(recipes_with_one),
                "recipes_unsupported_on_all_platforms": len(recipes_unsupported),
                "recipe_platform_mapping": {
                    recipe: sorted(list(platforms)) for recipe, platforms in sorted(recipe_platform_mapping.items())
                },
                "unsupported_recipes": recipes_unsupported,
            },
            "skipped_items": [
                {"recipe": r.recipe_name, "job_type": r.job_type, "reason": r.error_message} for r in skipped
            ],
            "failures": [
                {
                    "recipe": r.recipe_name,
                    "job_type": r.job_type,
                    "error": r.error_message,
                    "validation_errors": r.validation_errors,
                }
                for r in failed
            ],
        }

        return report


class TestLaunchJsonGenerationAndValidation:
    """Comprehensive test for launch.json generation and validation."""

    def test_all_recipes_generate_valid_launch_json(self):
        """Test that all matching recipes generate valid launch.json files (sequential, stops on first failure)."""
        # Set up file logging FIRST before any operations
        log_file = Path("launch_json_test.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # Configure the root logger to capture all messages
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Also ensure console output
        if not logger.handlers or len([h for h in logger.handlers if isinstance(h, logging.StreamHandler)]) == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

        try:
            logger.info("=" * 60)
            logger.info("Starting comprehensive launch.json test (sequential mode - stops on first failure)")
            logger.info(f"Log file: {log_file.absolute()}")
            logger.info("=" * 60)

            generator = LaunchJsonGenerator()

            # Store recipe file count before running tests
            recipes = generator.discover_recipes()
            unique_recipe_files = len(set(r[0] for r in recipes))

            results, recipe_platform_mapping = generator.run_all_tests()
            report = generator.generate_report(results, recipe_platform_mapping)

            # Add recipe file count to report
            report["recipe_files_discovered"] = unique_recipe_files
            report["tests_per_recipe_file"] = report["total"] / unique_recipe_files if unique_recipe_files > 0 else 0

            # Save detailed report
            report_path = Path("launch_json_test_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Print summary (using both print and logger for visibility)
            summary_lines = [
                f"\n{'='*60}",
                "LAUNCH.JSON TEST SUMMARY",
                f"{'='*60}",
                f"Recipe files discovered: {report.get('recipe_files_discovered', 'N/A')}",
                f"Total tests generated: {report['total']} ({report.get('tests_per_recipe_file', 'N/A'):.1f} tests per recipe file)",
                f"Successful: {report['successful']}",
                f"Skipped (platform-specific): {report['skipped']}",
                f"Failed: {report['failed']}",
                f"Success rate: {report['success_rate']:.1f}%",
                f"Detailed report saved to: {report_path}",
                f"{'='*60}",
            ]

            for line in summary_lines:
                print(line)  # Print so it shows in pytest output
                logger.info(line)

            # Print recipe coverage analysis
            coverage = report.get("recipe_coverage", {})
            if coverage:
                coverage_lines = [
                    f"\n{'='*60}",
                    "RECIPE PLATFORM COVERAGE ANALYSIS",
                    f"{'='*60}",
                    f"Total unique recipes: {coverage['total_unique_recipes']}",
                    f"Recipes with K8s support: {coverage['recipes_with_k8s_support']}",
                    f"Recipes with SM Jobs support: {coverage['recipes_with_sm_jobs_support']}",
                    f"Recipes with both platforms: {coverage['recipes_with_both_platforms']}",
                    f"Recipes with one platform only: {coverage['recipes_with_one_platform']}",
                    f"Recipes with zero platform support: {coverage['recipes_unsupported_on_all_platforms']}",
                    f"{'='*60}",
                ]

                for line in coverage_lines:
                    print(line)
                    logger.info(line)

                # Print unsupported recipes if any (these will cause test failure)
                if coverage["unsupported_recipes"]:
                    logger.error("\nRECIPES WITH ZERO PLATFORM SUPPORT (TEST WILL FAIL):")
                    for recipe in coverage["unsupported_recipes"]:
                        logger.error(f"  ⚠ {recipe}")

            # Print skipped items
            if report.get("skipped_items"):
                logger.info("\nSKIPPED (Platform-specific):")
                for skipped in report["skipped_items"][:10]:  # Show first 10
                    logger.info(f"  ⊘ {skipped['recipe']} ({skipped['job_type']}): {skipped['reason']}")

            # Print failures
            if report["failures"]:
                logger.info("\nFAILURES:")
                for failure in report["failures"][:10]:  # Show first 10
                    logger.info(f"  ✗ {failure['recipe']} ({failure['job_type']})")
                    logger.info(f"    Error: {failure['error']}")
                    if failure["validation_errors"]:
                        for err in failure["validation_errors"]:
                            logger.info(f"      - {err}")

            # Verify all recipes are supported on at least one platform
            # Recipes with only one platform support are acceptable
            unsupported_count = coverage.get("recipes_unsupported_on_all_platforms", 0)
            assert unsupported_count == 0, (
                f"Found {unsupported_count} recipe(s) with no platform support. "
                f"All recipes must be supported on at least one platform (k8s or sm_jobs). "
                f"See {report_path} for details."
            )

            # Assert no failures
            assert report["failed"] == 0, (
                f"Found {report['failed']} failures out of {report['total']} recipes. "
                f"See {report_path} for details."
            )
        finally:
            # Clean up file handler
            logger.removeHandler(file_handler)
            file_handler.close()
            print(f"\nDetailed log saved to: {log_file.absolute()}")
