import json
import random
import string
import subprocess
import threading
import time
from pathlib import Path

from .base_launcher import BaseLauncher
from .launcher_utils import (
    _get_recipe_type_info,
    get_peft_type_from_filename,
    load_recipe_with_hydra,
)
from .path_utils import get_recipes_folder

# Path to the jumpstart model ID mapping file
JUMPSTART_MODEL_ID_MAP_PATH = (
    Path(__file__).parent.parent.parent.parent / "launcher" / "recipe_templatization" / "jumpstart_model-id_map.json"
)


class ServerlessValidationLauncher(BaseLauncher):
    """Serverless SageMaker Training Jobs launcher using aws sagemaker CLI"""

    def __init__(self, job_recorder, config):
        super().__init__(job_recorder, config)

        # Get recipes folder path
        self.recipes_folder = get_recipes_folder()

        # Load model mapping from jumpstart_model-id_map.json
        self.model_mapping = self._load_model_mapping()

        # Load mappings from config file
        self.dataset_mapping = (
            dict(config.serverless_dataset_mapping) if hasattr(config, "serverless_dataset_mapping") else {}
        )
        self.evaluator_mapping = (
            dict(config.serverless_evaluator_mapping) if hasattr(config, "serverless_evaluator_mapping") else {}
        )
        self.mlflow_mapping = (
            dict(config.serverless_mlflow_mapping) if hasattr(config, "serverless_mlflow_mapping") else {}
        )

        # Track which models have had their Hub content overridden (by run_name)
        self._hub_content_overridden_models = set()
        self._hub_override_lock = threading.Lock()

    def _load_model_mapping(self) -> dict:
        """Load model mapping from jumpstart_model-id_map.json"""
        try:
            with open(JUMPSTART_MODEL_ID_MAP_PATH, "r") as f:
                model_mapping = json.load(f)
            self.logger.info(f"Loaded {len(model_mapping)} model mappings from {JUMPSTART_MODEL_ID_MAP_PATH}")
            return model_mapping
        except FileNotFoundError:
            self.logger.error(f"Model mapping file not found: {JUMPSTART_MODEL_ID_MAP_PATH}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse model mapping JSON: {e}")
            return {}

    def _override_hub_content(self, recipe=None):
        """Always override Hub content with local config values (images, recipes, config).
        Args:
            recipe: Optional recipe path (e.g., 'fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml').
                    If provided, extracts run_name from the recipe and overrides that model only.
                    If None, overrides all models in the model mapping.
        """
        if not hasattr(self.config, "container_info"):
            self.logger.warning("No container_info in config, skipping Hub content override")
            return

        has_llmft = hasattr(self.config.container_info, "llmft")
        has_verl = hasattr(self.config.container_info, "verl")

        if not has_llmft and not has_verl:
            self.logger.info("No smjobs images configured, skipping Hub content override")
            return

        if recipe:
            # Extract run_name from the recipe file
            run_name = self._extract_run_name_from_recipe(recipe)
            if run_name not in self.model_mapping:
                self.logger.warning(f"Run name '{run_name}' from recipe '{recipe}' not found in model mapping")
                return
            models_to_override = [run_name]
            self.logger.info(f"Overriding Hub content for model '{run_name}' (from recipe: {recipe})")
        else:
            models_to_override = list(self.model_mapping.keys())
            self.logger.info(f"Overriding Hub content for all {len(models_to_override)} models")

        success = self._auto_update_hub_content(models_to_override)

        if success:
            self.logger.info("✓ Hub content override completed successfully")
        else:
            # Provide manual command as fallback in the error message
            hub_name = getattr(self.config.serverless_config, "hub_name", "TestPrivateHub")
            region = self.config.serverless_config.region
            endpoint = self.config.serverless_config.endpoint

            manual_cmd = (
                f"python scripts/model_hub/update_private_hub.py "
                f"--hub-name {hub_name} --region {region} --endpoint {endpoint} "
                f"--s3-bucket <your-bucket> --recipe-regex '.*llama.*'"
            )
            raise RuntimeError(
                f"Hub content override failed for models: {models_to_override}. " f"Manual fallback: {manual_cmd}"
            )

    def _auto_update_hub_content(self, model_names: list) -> bool:
        """
        Override Hub content for the specified models with local config values.

        Args:
            model_names: List of model run_names to update

        Returns:
            True if all updates succeeded, False if any failed

        Raises:
            Exception: If initialization fails (e.g., bad config, auth failure)
        """
        if not model_names:
            return True

        # Get configuration
        hub_name = getattr(self.config.serverless_config, "hub_name", "TestPrivateHub")
        region = self.config.serverless_config.region

        # Get endpoint URL directly from config
        endpoint_url = self.config.serverless_config.endpoint

        # Create SageMaker client once for all updates
        client_kwargs = {"region_name": region}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        sagemaker = self.boto_session.client("sagemaker", **client_kwargs)

        self.logger.info(f"Updating {len(model_names)} models in Hub: {hub_name}")

        success_count = 0
        failed_models = []

        for model_name in model_names:
            try:
                if self._update_single_hub_content(sagemaker, model_name):
                    success_count += 1
                else:
                    failed_models.append(model_name)

            except Exception as e:
                self.logger.error(f"  ✗ Failed to update {model_name}: {e}")
                failed_models.append(model_name)

        # Report results
        if failed_models:
            self.logger.warning(f"Updated {success_count}/{len(model_names)} models. Failed: {failed_models}")
            return False
        else:
            self.logger.info(f"✓ Successfully updated all {success_count} models")
            return True

    def _update_single_hub_content(self, sagemaker_client, model_name: str) -> bool:
        """
        Override Hub content for a single model with local config values.

        Args:
            sagemaker_client: Boto3 SageMaker client
            model_name: Model run_name to update

        Returns:
            True if update succeeded, False otherwise
        """
        # Get the model ID from the model_mapping
        if model_name not in self.model_mapping:
            self.logger.warning(f"  Model {model_name} not found in model_mapping, skipping")
            return False

        model_info = self.model_mapping[model_name]

        # Handle both old format (string) and new format (dict)
        model_id = model_info if isinstance(model_info, str) else model_info.get("model_id")

        if not model_id:
            self.logger.warning(f"  No model_id found for {model_name}, skipping")
            return False

        # Infer recipe type from run_name
        recipe_type = "verl" if any(x in model_name.lower() for x in ["verl", "grpo", "rlvr", "rlaif"]) else "llmft"

        # Get expected image based on recipe type
        expected_image = self._get_expected_image(recipe_type)

        if not expected_image:
            self.logger.warning(
                f"  Could not determine expected image for {model_name} (recipe_type: {recipe_type}), skipping"
            )
            return False

        self.logger.info(f"  Overriding {model_name} ({recipe_type}) with image: {expected_image}")

        # Build model ARN and extract components
        model_arn = self._build_model_arn(model_id)
        arn_parts = model_arn.split("/")
        hub_name_from_arn = arn_parts[1]
        model_name_from_arn = arn_parts[3]

        # Get current Hub content
        response = sagemaker_client.describe_hub_content(
            HubName=hub_name_from_arn, HubContentType="Model", HubContentName=model_name_from_arn
        )

        # Parse and update the Hub content document
        hub_doc = json.loads(response.get("HubContentDocument", "{}"))
        recipes = hub_doc.get("RecipeCollection", [])

        # Always set SmtjImageUri to the expected value in all recipes
        override_count = 0
        for recipe in recipes:
            if "SmtjImageUri" in recipe:
                old_image = recipe["SmtjImageUri"]
                recipe["SmtjImageUri"] = expected_image
                override_count += 1
                if old_image != expected_image:
                    self.logger.info(f"    Changed recipe image: {old_image} -> {expected_image}")
                else:
                    self.logger.debug(f"    Recipe image already matches: {expected_image}")

        if override_count == 0:
            self.logger.warning(f"  No recipes with SmtjImageUri found for {model_name}")

        # Update the Hub content document
        hub_doc["RecipeCollection"] = recipes

        # Prepare import parameters
        import_params = {
            "HubName": hub_name_from_arn,
            "HubContentType": "Model",
            "HubContentName": model_name_from_arn,
            "HubContentDocument": json.dumps(hub_doc),
            "DocumentSchemaVersion": "2.4.0",
        }

        # Preserve optional metadata fields (exclude version to auto-generate new version)
        for field in [
            "HubContentDisplayName",
            "HubContentDescription",
            "HubContentMarkdown",
            "HubContentSearchKeywords",
        ]:
            if field in response:
                import_params[field] = response[field]

        # Import updated content (creates new version)
        sagemaker_client.import_hub_content(**import_params)

        self.logger.info(f"  ✓ Successfully overrode {model_name} ({override_count} recipes)")
        return True

    def _get_expected_image(self, recipe_type: str) -> str:
        """
        Get expected image URI for a recipe type.

        Args:
            recipe_type: Recipe type ('llmft' or 'verl')

        Returns:
            Expected image URI or None if not found
        """
        if recipe_type == "llmft" and hasattr(self.config.container_info, "llmft"):
            return self.config.container_info.llmft.smjobs
        elif recipe_type == "verl" and hasattr(self.config.container_info, "verl"):
            return self.config.container_info.verl.smjobs
        return None

    def _extract_customization_technique(self, recipe_path: str) -> str:
        """Extract customization technique from recipe filename"""
        recipe_lower = recipe_path.lower()
        if "rlaif" in recipe_lower:
            return "RLAIF"
        elif "rlvr" in recipe_lower:
            return "RLVR"
        elif "dpo" in recipe_lower:
            return "DPO"
        elif "sft" in recipe_lower:
            return "SFT"
        else:
            raise ValueError(f"Could not determine customization technique from recipe: {recipe_path}")

    def _extract_run_name_from_recipe(self, recipe_path: str) -> str:
        """Extract run.name from the recipe YAML file using Hydra composition"""
        try:
            # Load the recipe with Hydra to resolve all defaults
            recipe_cfg = load_recipe_with_hydra(recipe_path)

            # Extract the run.name field
            run_name = recipe_cfg.run.name

            if not run_name:
                raise ValueError(f"No run.name found in recipe: {recipe_path}")

            self.logger.info(f"Extracted run.name '{run_name}' from recipe: {recipe_path}")
            return run_name

        except Exception as e:
            raise ValueError(f"Failed to extract run.name from recipe {recipe_path}: {str(e)}")

    def _generate_job_name(self, run_name: str, technique: str) -> str:
        """Generate unique job name with format: {run_name}-{technique}-{5_char_hash}"""
        hash_chars = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
        job_name = f"{run_name}-{technique}-{hash_chars}"

        # Ensure it meets SageMaker naming requirements (max 63 chars)
        if len(job_name) > 63:
            # Truncate run_name to fit
            max_name_len = 63 - len(technique) - 6  # 6 for technique + hash + hyphens
            run_name = run_name[:max_name_len]
            job_name = f"{run_name}-{technique}-{hash_chars}"

        return job_name

    def _build_serverless_config(self, technique: str, model_arn: str, peft_type: str | None) -> str:
        """
        Build serverless job config JSON string.

        Args:
            technique: Customization technique (SFT, DPO, RLAIF, RLVR)
            model_arn: The model ARN
            peft_type: PEFT type - "LORA" for LoRA, None for FFT (full fine-tuning)
        """
        config = {
            "BaseModelArn": model_arn,
            "AcceptEula": True,
            "JobType": "FineTuning",
            "CustomizationTechnique": technique,  # At top level as working commands show
        }

        # Only include Peft field if it's LORA
        # For FFT (full fine-tuning), omit the Peft field entirely (API expects null/absence)
        if peft_type is not None:
            config["Peft"] = peft_type

        # Add evaluator if applicable and not empty
        if technique in self.evaluator_mapping and self.evaluator_mapping[technique]:
            config["EvaluatorArn"] = self.evaluator_mapping[technique]

        # Add judge_model_id only for RLAIF jobs
        if technique == "RLAIF":
            if hasattr(self.config.serverless_config, "default_hyper_parameters") and hasattr(
                self.config.serverless_config.default_hyper_parameters, "judge_model_id"
            ):
                config["JudgeModelId"] = self.config.serverless_config.default_hyper_parameters.judge_model_id
                self.logger.info(f"Added judge_model_id for RLAIF job: {config['JudgeModelId']}")

        return json.dumps(config)

    def _build_model_arn(self, model_name: str) -> str:
        """Build model ARN from config values"""
        region = self.config.serverless_config.region
        account_id = self.config.serverless_config.hub_account_id
        hub_name = self.config.serverless_config.hub_name
        model_version = self.config.serverless_config.model_version

        base_arn = f"arn:aws:sagemaker:{region}:{account_id}:hub-content/{hub_name}/Model/{model_name}"

        if model_version:
            return f"{base_arn}/{model_version}"
        return base_arn

    def _build_hyper_parameters(self, recipe_path: str) -> str:
        """Build hyper-parameters JSON from config based on recipe type.

        Uses the recipe_structure from recipe_type_config to look up
        technique-specific hyper-parameters in serverless_hyper_parameters section.

        Args:
            recipe_path: Path to the recipe file (used to determine recipe type)

        Returns:
            JSON string of hyper-parameters (default + technique-specific merged)
        """
        hyper_params = {}

        # Add default hyper-parameters
        if hasattr(self.config.serverless_config, "default_hyper_parameters"):
            hyper_params = dict(self.config.serverless_config.default_hyper_parameters)

        # Get recipe type info to find recipe_structure (used as hyper-parameters key)
        try:
            recipe_type_info = _get_recipe_type_info(self.config, recipe_path)
            recipe_structure = recipe_type_info.get("recipe_structure", "llmft")

            # Look up technique-specific hyper-parameters using recipe_structure as key
            if hasattr(self.config, "serverless_hyper_parameters"):
                technique_params = getattr(self.config.serverless_hyper_parameters, recipe_structure, None)
                if technique_params:
                    technique_params_dict = dict(technique_params)
                    if technique_params_dict:  # Only log if there are actual parameters
                        hyper_params.update(technique_params_dict)
                        self.logger.info(
                            f"Added technique-specific hyper-parameters for '{recipe_structure}': {technique_params_dict}"
                        )
        except ValueError as e:
            self.logger.warning(f"Could not determine recipe type for hyper-parameters: {e}")

        return json.dumps(hyper_params)

    def launch_job_async(self, recipe: str) -> tuple[str, str]:
        """
        Launch a serverless training job and return immediately with job name.
        Does NOT wait for job completion - use for async monitoring scenarios.

        Args:
            recipe: Path to recipe file

        Returns:
            tuple: (job_name, technique) - Training job name and customization technique

        Raises:
            Exception: If job launch fails
        """
        try:
            # Extract information from recipe
            technique = self._extract_customization_technique(recipe)
            run_name = self._extract_run_name_from_recipe(recipe)

            # Get model ARN using run.name
            if run_name not in self.model_mapping:
                raise ValueError(f"Run name '{run_name}' from recipe {recipe} not found in model mapping")

            model_name = self.model_mapping[run_name]
            model_arn = self._build_model_arn(model_name)

            # Generate job name using run.name
            job_name = self._generate_job_name(run_name, technique)

            # Get PEFT type (FFT or LORA) - convert FFT to None for API
            peft_type = get_peft_type_from_filename(recipe)
            peft_type = None if peft_type == "FFT" else peft_type

            # Get other required ARNs
            dataset_arn = self.dataset_mapping[technique]
            mlflow_arn = self.mlflow_mapping[technique]
            serverless_config = self._build_serverless_config(technique, model_arn, peft_type)

            # Build hyper-parameters from config (recipe-type-specific)
            hyper_params = self._build_hyper_parameters(recipe)

            # Get region from config
            region = self.config.serverless_config.region

            # Build model package config - include SourceModelPackageArn if available for continuous training
            model_package_config = {"ModelPackageGroupArn": self.config.serverless_config.model_package_group_arn}

            # Check if we have a source model package ARN for continuous training
            if hasattr(self.config, "serverless_config") and hasattr(
                self.config.serverless_config, "source_model_package_arn"
            ):
                source_model_arn = self.config.serverless_config.source_model_package_arn
                model_package_config["SourceModelPackageArn"] = source_model_arn
                self.logger.info(f"Using SourceModelPackageArn for continuous training: {source_model_arn}")

            # Get endpoint URL directly from config
            endpoint_url = self.config.serverless_config.endpoint

            # Build AWS CLI command
            command = [
                "aws",
                "sagemaker",
                "create-training-job",
                "--region",
                region,
            ]

            # Add --endpoint-url if configured (non-prod stages like beta, gamma)
            if endpoint_url:
                command.extend(["--endpoint-url", endpoint_url])

            command.extend(
                [
                    "--training-job-name",
                    job_name,
                    "--role-arn",
                    self.config.serverless_config.role_arn,
                    "--hyper-parameters",
                    hyper_params,
                    "--input-data-config",
                    f'[{{"ChannelName": "train", "DataSource":{{"DatasetSource":{{"DatasetArn": "{dataset_arn}"}}}}}}]',
                    "--output-data-config",
                    f'{{"S3OutputPath": "{self.config.serverless_config.s3_output_path}"}}',
                    "--stopping-condition",
                    f'{{"MaxRuntimeInSeconds": {self.config.serverless_config.max_runtime_seconds}}}',
                    "--mlflow-config",
                    f'{{"MlflowResourceArn": "{mlflow_arn}"}}',
                    "--serverless-job-config",
                    serverless_config,
                    "--model-package-config",
                    json.dumps(model_package_config),
                ]
            )

            # Generate debug info
            timestamp = int(time.time())
            debug_dir = Path("debug_serverless")
            debug_dir.mkdir(exist_ok=True)

            command_file = debug_dir / f"aws_command_{job_name}_{timestamp}.sh"
            log_file = debug_dir / f"aws_output_{job_name}_{timestamp}.log"

            # Write command to shell script for manual testing
            self._write_debug_command(command, command_file)

            self.logger.info(f"Debug command written to: {command_file}")
            self.logger.info(f"You can test manually by running: bash {command_file}")

            # Log job info
            self.logger.info(f"Launching serverless job (async): {job_name}")
            self.logger.info(f"  Recipe: {recipe}")
            self.logger.info(f"  Run Name: {run_name}")
            self.logger.info(f"  Technique: {technique}")
            self.logger.info(f"  Peft Type: {peft_type if peft_type else 'FFT (full fine-tuning)'}")
            self.logger.info(f"  Model ARN: {model_arn}")
            self.logger.info(f"  Dataset ARN: {dataset_arn}")

            # Execute command
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            # Write detailed output to log file
            self._write_debug_output(result, log_file, command)

            if result.returncode != 0:
                error_msg = f"AWS CLI failed with exit code {result.returncode}"
                self.logger.error(f"{error_msg}")
                self.logger.error(f"STDERR: {result.stderr}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"Full debug info written to: {log_file}")
                raise Exception(f"{error_msg}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")

            # Success - return job name immediately without monitoring
            self.logger.info(f"Successfully launched job (async): {job_name}")
            self.logger.info(f"AWS Response: {result.stdout}")

            return job_name, technique

        except Exception as e:
            error_msg = f"Failed to launch serverless job for {recipe}: {str(e)}"
            self.logger.error(error_msg)
            raise

    def wait_for_job_completion(self, recipe: str, job_name: str, stop_event=None) -> bool:
        """
        Wait for a previously launched job to complete and record results.
        Use this after launch_job_async() to monitor job completion.

        Args:
            recipe: Path to recipe file
            job_name: Training job name from launch_job_async()
            stop_event: Optional threading.Event that will be set when job should stop early

        Returns:
            bool: True if job completed successfully, False otherwise
        """
        # Initialize job entry in recorder before monitoring
        # This ensures update_job() calls in _monitor_job() will work
        self.job_recorder.add_job(
            input_filename=recipe, output_path="", status="InProgress", output_log=f"Monitoring job: {job_name}"
        )

        # If stop_event is provided, we're doing early stopping
        # Don't use the synchronous _monitor_job, just wait for the stop event
        if stop_event:
            import time

            self.logger.info(f"Waiting for early stop signal for job: {job_name}")

            # Wait for stop event with timeout (max 2 hours)
            max_wait = 7200  # 2 hours
            start_time = time.time()

            while not stop_event.is_set() and (time.time() - start_time) < max_wait:
                time.sleep(5)  # Check every 5 seconds

            if stop_event.is_set():
                self.logger.info(f"Early stop signal received for job: {job_name}")
                # Job was stopped early, mark as complete for now
                # The actual status will be updated by the monitoring thread
                return True
            else:
                self.logger.warning(f"Timeout waiting for early stop signal for job: {job_name}")
                return False

        # Normal synchronous monitoring
        return self._monitor_job(recipe, job_name)

    def launch_job(self, recipe: str) -> bool:
        """Launch a serverless training job with enhanced debugging"""
        try:
            # Extract information from recipe
            technique = self._extract_customization_technique(recipe)
            run_name = self._extract_run_name_from_recipe(recipe)

            # Override Hub content for this model if not already done
            with self._hub_override_lock:
                if run_name not in self._hub_content_overridden_models:
                    self.logger.info(f"Overriding Hub content for run.name '{run_name}' (recipe: {recipe})")
                    self._override_hub_content(recipe=recipe)
                    self._hub_content_overridden_models.add(run_name)

            # Get model ARN using run.name
            if run_name not in self.model_mapping:
                raise ValueError(f"Run name '{run_name}' from recipe {recipe} not found in model mapping")

            model_name = self.model_mapping[run_name]
            model_arn = self._build_model_arn(model_name)

            # Generate job name using run.name
            job_name = self._generate_job_name(run_name, technique)

            # Get PEFT type (FFT or LORA) - convert FFT to None for API
            peft_type = get_peft_type_from_filename(recipe)
            peft_type = None if peft_type == "FFT" else peft_type

            # Get other required ARNs
            dataset_arn = self.dataset_mapping[technique]
            mlflow_arn = self.mlflow_mapping[technique]
            serverless_config = self._build_serverless_config(technique, model_arn, peft_type)

            # Build hyper-parameters from config (recipe-type-specific)
            hyper_params = self._build_hyper_parameters(recipe)

            # Get region from config
            region = self.config.serverless_config.region

            # Build model package config - include SourceModelPackageArn if available for continuous training
            model_package_config = {"ModelPackageGroupArn": self.config.serverless_config.model_package_group_arn}

            # Check if we have a source model package ARN for continuous training
            if hasattr(self.config, "serverless_config") and hasattr(
                self.config.serverless_config, "source_model_package_arn"
            ):
                source_model_arn = self.config.serverless_config.source_model_package_arn
                model_package_config["SourceModelPackageArn"] = source_model_arn
                self.logger.info(f"Using SourceModelPackageArn for continuous training: {source_model_arn}")

            # Get endpoint URL directly from config
            endpoint_url = self.config.serverless_config.endpoint

            # Build AWS CLI command
            command = [
                "aws",
                "sagemaker",
                "create-training-job",
                "--region",
                region,
            ]

            # Add --endpoint-url if configured (non-prod stages like beta, gamma)
            if endpoint_url:
                command.extend(["--endpoint-url", endpoint_url])

            command.extend(
                [
                    "--training-job-name",
                    job_name,
                    "--role-arn",
                    self.config.serverless_config.role_arn,
                    "--hyper-parameters",
                    hyper_params,
                    "--input-data-config",
                    f'[{{"ChannelName": "train", "DataSource":{{"DatasetSource":{{"DatasetArn": "{dataset_arn}"}}}}}}]',
                    "--output-data-config",
                    f'{{"S3OutputPath": "{self.config.serverless_config.s3_output_path}"}}',
                    "--stopping-condition",
                    f'{{"MaxRuntimeInSeconds": {self.config.serverless_config.max_runtime_seconds}}}',
                    "--mlflow-config",
                    f'{{"MlflowResourceArn": "{mlflow_arn}"}}',
                    "--serverless-job-config",
                    serverless_config,
                    "--model-package-config",
                    json.dumps(model_package_config),
                ]
            )

            # Generate debug info
            timestamp = int(time.time())
            debug_dir = Path("debug_serverless")
            debug_dir.mkdir(exist_ok=True)

            command_file = debug_dir / f"aws_command_{job_name}_{timestamp}.sh"
            log_file = debug_dir / f"aws_output_{job_name}_{timestamp}.log"

            # Write command to shell script for manual testing
            self._write_debug_command(command, command_file)

            self.logger.info(f"Debug command written to: {command_file}")
            self.logger.info(f"You can test manually by running: bash {command_file}")

            # === EXISTING LOGGING (keep this) ===
            self.logger.info(f"Launching serverless job: {job_name}")
            self.logger.info(f"  Recipe: {recipe}")
            self.logger.info(f"  Run Name: {run_name}")
            self.logger.info(f"  Technique: {technique}")
            self.logger.info(f"  Peft Type: {peft_type if peft_type else 'FFT (full fine-tuning)'}")
            self.logger.info(f"  Model ARN: {model_arn}")
            self.logger.info(f"  Dataset ARN: {dataset_arn}")

            # === ENHANCED COMMAND EXECUTION ===

            # Execute with enhanced error capture
            result = subprocess.run(
                command, capture_output=True, text=True, check=False  # Don't raise exception, we'll handle it
            )

            # Write detailed output to log file
            self._write_debug_output(result, log_file, command)

            if result.returncode != 0:
                error_msg = f"AWS CLI failed with exit code {result.returncode}"
                self.logger.error(f"{error_msg}")
                self.logger.error(f"STDERR: {result.stderr}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"Full debug info written to: {log_file}")

                # Include specific debug info in job recorder
                debug_info = f"{error_msg}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}\nCommand file: {command_file}\nLog file: {log_file}"
                self.job_recorder.update_job(input_filename=recipe, status="Failed", output_log=debug_info)
                return False

            # === SUCCESS CASE ===
            self.logger.info(f"Successfully launched job: {job_name}")
            self.logger.info(f"AWS Response: {result.stdout}")

            # Monitor the job (existing code)
            return self._monitor_job(recipe, job_name)

        except Exception as e:
            error_msg = f"Failed to launch serverless job for {recipe}: {str(e)}"
            self.logger.error(error_msg)
            self.job_recorder.update_job(input_filename=recipe, status="Failed", output_log=error_msg)
            return False

    def _monitor_job(self, recipe: str, job_name: str) -> bool:
        """
        Monitor job until completion with updated success criteria:
        1. Job status is 'Completed'
        2. 'OutputModelPackageArn' appears in job details
        3. CloudWatch logs contain success messages based on technique
        4. (Optional) MLflow metrics verification
        """
        try:
            max_polls = 240  # Maximum polling attempts (2 hours at 30s intervals)
            poll_count = 0
            technique = self._extract_customization_technique(recipe)
            start_time = time.time()
            last_logged_status = None

            self.logger.info(f"Starting job monitoring for: {job_name} (technique: {technique})")

            # Get endpoint URL directly from config
            region = self.config.serverless_config.region
            endpoint_url = self.config.serverless_config.endpoint

            while poll_count < max_polls:
                poll_count += 1
                elapsed_minutes = (time.time() - start_time) / 60

                # Check job status
                describe_command = [
                    "aws",
                    "sagemaker",
                    "describe-training-job",
                    "--region",
                    region,
                ]

                # Add --endpoint-url only for non-prod stages (beta, gamma)
                if endpoint_url:
                    describe_command.extend(["--endpoint-url", endpoint_url])

                describe_command.extend(
                    [
                        "--training-job-name",
                        job_name,
                    ]
                )

                result = subprocess.run(describe_command, capture_output=True, text=True, check=True)
                job_details = json.loads(result.stdout)

                status = job_details.get("TrainingJobStatus", "Unknown")
                secondary_status = job_details.get("SecondaryStatus", "N/A")

                # Log status changes or every 2 polls (1 minute since we poll every 30s)
                if status != last_logged_status or poll_count % 2 == 0:
                    self.logger.info(
                        f"Job Status: {status} | Secondary: {secondary_status} | Elapsed: {elapsed_minutes:.1f}min"
                    )
                    last_logged_status = status

                # Handle failure states
                if status in ["Failed", "Stopped"]:
                    failure_reason = job_details.get("FailureReason", "Unknown failure reason")
                    self.job_recorder.update_job(
                        input_filename=recipe,
                        status="Failed",
                        output_log=f"Job failed with status: {status}. Reason: {failure_reason}",
                    )
                    return False

                # Handle completion - run full validation
                elif status == "Completed":
                    self.logger.info(f"Job {job_name} completed. Running success validation...")

                    # Criterion 1: Check for OutputModelPackageArn
                    output_model_package_arn = job_details.get("OutputModelPackageArn")
                    if not output_model_package_arn:
                        self.job_recorder.update_job(
                            input_filename=recipe,
                            status="Failed",
                            output_log="Job completed but 'OutputModelPackageArn' field not found",
                        )
                        return False

                    self.logger.info(f"✓ OutputModelPackageArn found: {output_model_package_arn}")

                    # Criterion 2: Check CloudWatch logs for success messages
                    logs_valid, log_message = self._validate_cloudwatch_logs(job_name, technique)
                    if not logs_valid:
                        self.job_recorder.update_job(
                            input_filename=recipe,
                            status="Failed",
                            output_log=f"Job completed with OutputModelPackageArn but CloudWatch logs validation failed: {log_message}",
                        )
                        return False

                    self.logger.info(f"✓ CloudWatch logs validation passed: {log_message}")

                    # Criterion 3 (Optional): MLflow metrics verification

                    # Success!
                    success_msg = f"Job completed successfully. OutputModelPackageArn: {output_model_package_arn}. Log message: {log_message}."
                    self.job_recorder.update_job(input_filename=recipe, status="Complete", output_log=success_msg)
                    return True

                # Continue monitoring for in-progress jobs
                elif status in ["InProgress", "Starting"]:
                    time.sleep(30)  # Wait 30 seconds between polls
                    continue

                else:
                    self.logger.warning(f"Unknown job status: {status}")
                    poll_count += 1
                    time.sleep(30)

            # Timeout handling
            self.logger.warning(f"Job monitoring timed out after {max_polls} polls")
            self.job_recorder.update_job(
                input_filename=recipe,
                status="Timeout",
                output_log=f"Job monitoring timed out after {max_polls * 60} seconds. Last status: {status}",
            )
            return False

        except Exception as e:
            error_msg = f"Error monitoring job {job_name}: {str(e)}"
            self.logger.error(error_msg)
            self.job_recorder.update_job(input_filename=recipe, status="Failed", output_log=error_msg)
            return False

    def _write_debug_command(self, command: list, command_file: Path):
        """Write the AWS CLI command to a shell script for manual testing"""
        try:
            with open(command_file, "w") as f:
                f.write("#!/bin/bash\n\n")
                f.write("# Generated AWS CLI command for debugging\n")
                f.write("# You can run this manually to test the command\n\n")

                # Write the command with proper escaping and formatting
                # Use single quotes for JSON arguments to avoid escaping issues
                escaped_args = []
                for arg in command:
                    if arg.startswith("{") or arg.startswith("["):
                        # JSON argument - use single quotes to avoid escaping issues
                        escaped_args.append(f"'{arg}'")
                    elif " " in arg:
                        # Regular argument with spaces - use double quotes
                        escaped_args.append(f'"{arg}"')
                    else:
                        # Simple argument - no quotes needed
                        escaped_args.append(arg)
                cmd_str = " \\\n    ".join(escaped_args)
                f.write(f"{cmd_str}\n")

            # Make the script executable
            command_file.chmod(0o755)

        except Exception as e:
            self.logger.error(f"Failed to write debug command file: {e}")

    def _write_debug_output(self, result: subprocess.CompletedProcess, log_file: Path, command: list):
        """Write detailed debug output to log file"""
        try:
            with open(log_file, "w") as f:
                f.write("=== AWS CLI Debug Output ===\n\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return Code: {result.returncode}\n\n")

                f.write("=== Command ===\n")
                for i, arg in enumerate(command):
                    f.write(f"{i}: {arg}\n")
                f.write("\n")

                f.write("=== STDOUT ===\n")
                f.write(result.stdout or "(empty)\n\n")

                f.write("=== STDERR ===\n")
                f.write(result.stderr or "(empty)\n\n")

                f.write("=== Environment Info ===\n")
                f.write(f"AWS CLI Version: ")
                try:
                    aws_version = subprocess.run(["aws", "--version"], capture_output=True, text=True)
                    f.write(f"{aws_version.stdout.strip()}\n")
                except:
                    f.write("Could not determine AWS CLI version\n")

        except Exception as e:
            self.logger.error(f"Failed to write debug log file: {e}")

    def _validate_cloudwatch_logs(self, job_name: str, technique: str) -> tuple[bool, str]:
        """
        Validate CloudWatch logs contain the expected success messages based on technique.

        Returns:
            tuple: (is_valid, message)
        """
        try:
            # Get region and log group from config
            region = self.config.serverless_config.region
            log_group = (
                self.config.serverless_config.cloudwatch_log_group
                if hasattr(self.config.serverless_config, "cloudwatch_log_group")
                else "/aws/sagemaker/TrainingJobs"
            )

            logs_client = self.boto_session.client("logs", region_name=region)

            # Define expected messages based on technique
            expected_messages = {
                "RLVR": ["Training completed successfully"],
                "RLAIF": ["Training completed successfully"],
                "DPO": ["Training completed successfully"],
                "SFT": ["Training completed successfully"],
            }

            messages_to_find = expected_messages.get(technique, ["Training completed successfully"])
            self.logger.info(f"Looking for messages in CloudWatch logs: {messages_to_find}")

            # Get log streams for this job
            streams_response = logs_client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                limit=10,
            )

            log_streams = streams_response.get("logStreams", [])
            if not log_streams:
                return False, f"No CloudWatch log streams found for job {job_name}"

            self.logger.info(f"Found {len(log_streams)} log streams for job {job_name}")

            # Search through all log streams
            for stream in log_streams:
                stream_name = stream["logStreamName"]

                try:
                    # Get recent log events (last 1000 events should be sufficient)
                    events_response = logs_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=stream_name,
                        limit=1000,
                        startFromHead=False,  # Get most recent events first
                    )

                    events = events_response.get("events", [])
                    self.logger.info(f"Checking {len(events)} events in stream {stream_name}")

                    # Search for expected messages
                    for event in events:
                        message = event.get("message", "")
                        for expected_msg in messages_to_find:
                            if expected_msg.lower() in message.lower():
                                return True, f"Found '{expected_msg}' in CloudWatch logs (stream: {stream_name})"

                except Exception as stream_error:
                    self.logger.warning(f"Error reading stream {stream_name}: {stream_error}")
                    continue

            return False, f"Expected messages {messages_to_find} not found in CloudWatch logs"

        except Exception as e:
            return False, f"CloudWatch log validation error: {str(e)}"
