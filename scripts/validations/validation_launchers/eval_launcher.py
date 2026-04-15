"""
Evaluation Validation Launcher for SageMaker Training Jobs (SMJOBS).

This launcher directly constructs and runs the Hydra-based ``python3 main.py``
command with all eval parameters, then monitors the resulting SageMaker
Training Job to completion.

Supports multiple eval jobs via eval_config.eval_jobs list, each with its own
model weights path, base model name, and optional parameter overrides.
Auto-resolves data_s3_path from training_recipe using the existing
recipe_type_config → recipe_dataset_mapping → datasets config chain.

Usage:
    1. Set platform to "EVAL" in common_validation_config.yaml
    2. Configure eval_config section with eval_jobs list
    3. Set recipe_list to the eval recipe(s)
    4. Run: python scripts/validations/validation_script_runner.py
"""

import logging
import os
import random
import string
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

try:
    import sagemaker
except ImportError:
    logging.warning("sagemaker not available...")

from .base_launcher import BaseLauncher
from .launcher_utils import _get_dataset_info
from .smjobs_launcher import SageMakerJobsValidationLauncher

# Default S3 bucket for eval results
DEFAULT_RESULTS_S3_BUCKET = "s3://eval-test-results"

# Shared default keys that can be set at eval_config level and overridden per-job
SHARED_DEFAULT_KEYS = [
    "container_image",
    "task",
    "strategy",
    "metric",
    "subtask",
    "eval_results_dir",
    "eval_tensorboard_results_dir",
    "exp_dir",
]


def _generate_run_id(length=5):
    """Generate a random alphanumeric run ID to prevent result collisions."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class EvalValidationLauncher(BaseLauncher):
    """
    Evaluation launcher that directly constructs and executes the Hydra
    ``python3 main.py`` command, then monitors the resulting SageMaker
    Training Job.

    All eval parameters (model path, task, strategy, etc.) are resolved in
    this launcher — there is no intermediate shell script.
    """

    def __init__(self, job_recorder, config):
        """
        Initialize the eval launcher.

        Args:
            job_recorder: Job recorder instance for tracking results
            config: Configuration object (OmegaConf) with eval_config section
        """
        super().__init__(job_recorder, config)

        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.sagemaker_client = self.boto_session.client("sagemaker")
        self.logs_client = self.boto_session.client("logs")

    def launch_model_group(self, model_name, recipes):
        """
        Override to launch eval jobs in parallel. Iterates over eval_config.eval_jobs
        list, merges configs, then submits all jobs concurrently via ThreadPoolExecutor.
        """
        logging.info(f"Processing eval model group: {model_name} with {len(recipes)} recipe(s)")

        if not recipes:
            logging.warning(f"No eval recipes to launch for model {model_name}")
            return

        # Validate eval_config exists
        if not hasattr(self.config, "eval_config"):
            logging.error(
                "eval_config section not found in configuration. " "Please add eval_config with eval_jobs list."
            )
            return

        eval_cfg = self.config.eval_config
        eval_jobs = list(eval_cfg.eval_jobs)

        if not eval_jobs:
            logging.error("eval_config.eval_jobs is empty. Add at least one eval job.")
            return

        # All eval jobs use the same eval recipe
        recipe = recipes[0]

        logging.info(f"Found {len(eval_jobs)} eval job(s) to launch in parallel")

        jobs = []
        for i, job_cfg in enumerate(eval_jobs):
            try:
                job_config = self._merge_eval_job_config(eval_cfg, job_cfg)

                logging.info(
                    f"Eval job {i + 1}/{len(eval_jobs)}: "
                    f"model={job_config.get('model_weights_s3_path', 'N/A')}, "
                    f"task={job_config.get('task', 'mmlu')}, "
                    f"data={job_config.get('data_s3_path', 'N/A')}"
                )
                jobs.append((i, job_config))
            except Exception as e:
                logging.error(f"Failed to prepare eval job {i + 1}: {e}")
                logging.error(f"Full traceback:\n{traceback.format_exc()}")
                job_label = f"{recipe} (job {i + 1})"
                self.job_recorder.update_job(input_filename=job_label, status="Failed", output_log=f"Config error: {e}")

        if not jobs:
            logging.error("No eval jobs could be prepared. Aborting.")
            return

        # Launch all prepared jobs in parallel
        max_workers = min(len(jobs), 10)
        logging.info(f"Launching {len(jobs)} eval job(s) with {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._launch_single_eval_job, recipe, job, job_index=i): (i, job) for i, job in jobs
            }
            for future in futures:
                i, job = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Unhandled exception for eval job {i + 1}: {e}")
                    logging.error(f"Full traceback:\n{traceback.format_exc()}")
                    job_label = f"{recipe} (job {i + 1})"
                    self.job_recorder.update_job(
                        input_filename=job_label, status="Failed", output_log=f"Unhandled exception: {e}"
                    )

    def _merge_eval_job_config(self, eval_cfg, job_cfg):
        """
        Merge shared eval_config defaults with per-job overrides.

        Per-job fields take priority over shared defaults.

        Args:
            eval_cfg: Top-level eval_config OmegaConf object (shared defaults)
            job_cfg: Per-job OmegaConf object from eval_jobs list

        Returns:
            dict: Merged configuration for a single eval job
        """
        merged = {}

        # Start with shared defaults
        for key in SHARED_DEFAULT_KEYS:
            merged[key] = str(getattr(eval_cfg, key, ""))

        # Apply per-job fields (override shared defaults if present)
        for key in job_cfg:
            value = job_cfg[key]
            if value is not None and str(value) != "":
                merged[key] = str(value)

        return merged

    def _launch_single_eval_job(self, input_file_path, job_cfg, job_index=0):
        """
        Launch a single eval job by directly running the Hydra python3 main.py command.

        Args:
            input_file_path: Recipe file path (e.g. evaluation/open-source/open_source_deterministic_eval.yaml)
            job_cfg: Merged eval job config dict
            job_index: Index of this job in eval_jobs list (for logging)

        Returns:
            bool: True if job completed successfully, False otherwise
        """
        # Use a unique label for job recording when running multiple eval jobs
        base_model = job_cfg.get("base_model_name", "")
        model_weights = job_cfg.get("model_weights_s3_path", "")
        job_label = f"{input_file_path} [{base_model}]"
        if base_model:
            job_label = f"{input_file_path} [{base_model}]"
        elif model_weights:
            # Create a more descriptive label for multi-job tracking
            short_model = model_weights.rstrip("/").split("/")[-1] if "/" in model_weights else model_weights
            job_label = f"{input_file_path} [{short_model}]"

        # Register this job in the recorder
        if job_label not in self.job_recorder.jobs:
            self.job_recorder.add_job(input_filename=job_label)

        try:
            if not model_weights:
                raise ValueError(
                    f"model_weights_s3_path is required for eval job {job_index + 1}. "
                    "Set it in eval_config.eval_jobs."
                )

            # Build the Hydra command directly
            command = self._build_eval_command(job_cfg)

            logging.info(f"Launching eval job {job_index + 1} for recipe '{input_file_path}'")
            logging.info(f"  Model weights: {model_weights}")
            logging.info(f"  Base model: {base_model}")
            logging.info(f"  Task: {job_cfg.get('task', 'mmlu')}")
            logging.info(f"  Data S3 path: {job_cfg.get('data_s3_path', 'N/A')}")
            logging.info(f"  Command: {' '.join(command)}")

            # Execute the command directly
            env = self.aws_env.copy()
            env["HYDRA_FULL_ERROR"] = "1"

            launch_output = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )

            logging.info(f"Eval launch output for '{job_label}':\n{launch_output.stdout}")

            # Parse output to get SMTJ job name
            launch_stdout = launch_output.stdout.split("\n")
            training_job_name, output_folder_path, status = self._parse_output(launch_stdout)

            if training_job_name == "" or status in ["Failed", "Stopped"]:
                self.job_recorder.update_job(
                    input_filename=job_label, status=status or "Failed", output_log=launch_output.stdout
                )
                return False

            logging.info(f"Eval SMTJ job submitted: {training_job_name}")

            # Monitor job to completion
            return self._monitor_job(job_label, training_job_name, output_folder_path)

        except subprocess.CalledProcessError as e:
            error_msg = f"Eval command failed for '{job_label}': {e.stderr}"
            logging.error(error_msg)
            self.job_recorder.update_job(input_filename=job_label, status="Failed", output_log=error_msg)
            return False
        except Exception as e:
            error_msg = f"Eval job launch failed for '{job_label}': {e}"
            logging.error(error_msg)
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            self.job_recorder.update_job(input_filename=job_label, status="Failed", output_log=str(e))
            return False

    def _build_eval_command(self, job_cfg):
        """
        Build the Hydra ``python3 main.py`` command list for an eval job.

        All eval parameters are resolved here -- no intermediate shell script
        is involved.

        Args:
            job_cfg: Merged eval job config dict

        Returns:
            list[str]: Command list suitable for ``subprocess.run``

        Raises:
            ValueError: If required ``base_model_name`` is not provided.
        """
        from scripts.validations.validation_launchers.path_utils import get_project_root

        project_root = str(get_project_root())

        # --- required fields ------------------------------------------------
        model_weights = job_cfg.get("model_weights_s3_path", "")

        base_model = job_cfg.get("base_model_name", "")
        if not base_model:
            raise ValueError(
                "base_model_name is required for eval jobs. "
                "Set it in eval_config.eval_jobs or as a shared default in eval_config."
            )

        # --- unique run id to avoid result collisions -----------------------
        run_id = _generate_run_id()
        logging.info(f"Generated eval run ID: {run_id}")

        # --- optional fields (defaults come from eval_config in common_validation_config.yaml) ---
        image = job_cfg.get("container_image")
        exp_dir = job_cfg.get("exp_dir") or os.path.join(project_root, "results", "eval_experiments")
        task = job_cfg.get("task", "")
        strategy = job_cfg.get("strategy", "")
        metric = job_cfg.get("metric", "")
        subtask = job_cfg.get("subtask", "")
        data_s3_path = job_cfg.get("data_s3_path", "")
        if not data_s3_path:
            try:
                dataset_info = _get_dataset_info(self.config, job_cfg.get("training_recipe", ""), "smjobs")
                data_s3_path = dataset_info["val_data_dir"]
                logging.info(f"Auto-resolved data_s3_path from training_recipe: {data_s3_path}")
            except Exception as e:
                logging.warning(f"Failed to auto-resolve data_s3_path: {e}")
                data_s3_path = ""

        eval_results_dir = (
            job_cfg.get("eval_results_dir") or f"{DEFAULT_RESULTS_S3_BUCKET}/{base_model}-{run_id}/eval-results"
        )
        eval_tb_dir = (
            job_cfg.get("eval_tensorboard_results_dir")
            or f"{DEFAULT_RESULTS_S3_BUCKET}/{base_model}-{run_id}/tensorboard"
        )

        logging.info(f"  eval_results_dir: {eval_results_dir}")

        # --- build the command ----------------------------------------------
        cmd = [
            "python3",
            os.path.join(project_root, "main.py"),
            "hydra.job.chdir=True",
            f"base_results_dir={exp_dir}",
            "recipes=evaluation/open-source/open_source_deterministic_eval",
            'recipes.run.name="deterministic-eval-job"',
            f"recipes.run.model_name_or_path={model_weights}",
            f"recipes.run.base_model_name={base_model}",
            f"recipes.run.data_s3_path=/opt/ml/input/data/data/",
            f"recipes.evaluation.task={task}",
            f"recipes.evaluation.strategy={strategy}",
            f"recipes.evaluation.metric={metric}",
            f"recipes.evaluation.subtask={subtask}",
            f"recipes.output.eval_results_dir={eval_results_dir}",
            f"recipes.output.eval_tensorboard_results_dir={eval_tb_dir}",
            f"container={image}",
            "cluster_type=sm_jobs",
            "launch_json=false",
            f"+cluster.sm_jobs_config.additional_estimator_kwargs.image_uri={image}",
            f"++cluster.sm_jobs_config.output_path={eval_results_dir}",
            f"++cluster.sm_jobs_config.inputs.s3.model={model_weights}",
            f"++cluster.sm_jobs_config.inputs.s3.data={data_s3_path}",
            "++cluster.sm_jobs_config.additional_estimator_kwargs.volume_size=100",
            "++cluster.sm_jobs_config.additional_estimator_kwargs.max_run=86400",
            "++cluster.sm_jobs_config.additional_estimator_kwargs.container_log_level=20",
        ]

        return cmd

    def _parse_output(self, launch_stdout):
        """
        Parse the eval command output to extract the SMTJ job name.

        Delegates to SageMakerJobsValidationLauncher._parse_output which
        implements the same regex + list_training_jobs logic.

        """
        return SageMakerJobsValidationLauncher._parse_output(self, launch_stdout)

    def _monitor_job(self, recipe, training_job_name, output_folder_path):
        """
        Monitor the eval SMTJ job until completion.

        Args:
            recipe: Job label for the job recorder
            training_job_name: SageMaker Training Job name
            output_folder_path: Path to output folder

        Returns:
            bool: True if job completed successfully
        """
        job_successful = False
        output_log_cloudwatch = f"Check cloudwatch logs at /aws/sagemaker/TrainingJobs/{training_job_name}"

        # Get output S3 path from the training job description
        output_s3_path = ""
        try:
            job_desc = self.sagemaker_session.describe_training_job(training_job_name)
            output_s3_path = job_desc.get("OutputDataConfig", {}).get("S3OutputPath", "")
            if output_s3_path:
                logging.info(f"Eval job output S3 path: {output_s3_path}")
        except Exception as e:
            logging.warning(f"Could not get output S3 path for job {training_job_name}: {e}")

        try:
            while True:
                time.sleep(300)  # Wait 5 minutes before each check
                status = self.sagemaker_session.describe_training_job(training_job_name)
                current_status = status["TrainingJobStatus"]
                secondary_status = status.get("SecondaryStatus", "Unknown")

                logging.info(f"Eval job {training_job_name}: Primary={current_status}, Secondary={secondary_status}")

                if current_status in ["Failed", "Stopped"]:
                    failure_reason = status.get("FailureReason", "Unknown")
                    logging.error(f"Eval job failed: {failure_reason}")
                    break
                elif current_status == "Completed":
                    job_successful = True
                    break
                elif current_status == "InProgress":
                    if secondary_status == "Training":
                        continue

                time.sleep(120)  # Additional wait between checks

        except Exception as e:
            if hasattr(e, "stderr"):
                logging.error(f"Error monitoring eval job: {e.stderr}")
            else:
                logging.error(f"Error monitoring eval job: {e}")

        if job_successful:
            logging.info(f"Eval job {training_job_name} completed successfully")
            self.job_recorder.update_job(
                input_filename=recipe,
                output_path=output_s3_path or output_folder_path,
                status="Complete",
                output_log=output_log_cloudwatch,
                tokens_throughput="N/A (evaluation job)",
            )
        else:
            self.job_recorder.update_job(
                input_filename=recipe,
                output_path=output_s3_path or output_folder_path,
                status="Failed",
                output_log=output_log_cloudwatch,
            )

        return job_successful
