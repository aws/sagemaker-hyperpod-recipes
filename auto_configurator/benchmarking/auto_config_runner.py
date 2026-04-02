"""
Job runner for auto-configurator benchmarking.

Integrates with validation launcher infrastructure.
"""
import logging
import os
import re
import time
import traceback

from omegaconf import OmegaConf

from auto_configurator.utils.util import prettify
from hyperpod_recipes import get_recipe
from scripts.validations.job_recorder import JobRecorder
from scripts.validations.validation_launchers.base_launcher import BaseLauncher
from scripts.validations.validation_launchers.launcher_utils import (
    select_validation_launcher,
)
from scripts.validations.validation_launchers.path_utils import get_common_config

# Load base config from validation launcher library
COMMON_CONFIG_PATH = get_common_config()
BASE_CONFIG = OmegaConf.load(COMMON_CONFIG_PATH)

# Regex pattern for parsing dry run artifact directory
DRY_RUN_ARTIFACT_PATTERN = re.compile(r"\[DRY_RUN\]\s+Artifacts generated at:\s*(.+)", re.IGNORECASE)


class AutoConfigRunner:
    """Runs training jobs for auto-configurator benchmarking"""

    def __init__(self, auto_cfg):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.auto_config = auto_cfg
        self.__run_identifier = f"autoconfigurator-{self.auto_config.instance_type.replace('.', '_')}"

        self.cfg = self._generate_validation_config()

        os.makedirs(self.cfg.base_results_dir, exist_ok=True)

        instance_type: str = auto_cfg.instance_type
        platform: str = auto_cfg.platform
        self._job_recorder: JobRecorder = JobRecorder(
            instance_type=instance_type,
            platform=platform,
        )

        self._job_launcher: BaseLauncher = self.__configure_launcher(platform)

        self.__base_launch_command = self.__generate_base_launch_command()
        self.base_recipe = self._get_recipe(self.cfg.recipe)

    def _generate_validation_config(self, overrides={}):
        """Generate validation script config"""

        # Merge with auto_config overrides
        return OmegaConf.merge(BASE_CONFIG, self.auto_config, overrides)

    def _get_recipe(self, input_file_path):
        self.logger.info(f"Getting recipe: {input_file_path}")

        try:
            recipe_id = input_file_path.removesuffix(".yaml")

            return get_recipe(recipe_id).config
        except Exception as e:
            error_msg = f"Failed to get recipe: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e

    def __generate_base_launch_command(self):
        """
        Prepare base launch command.
        """
        self.logger.info(f"Preparing base launch command for {self.cfg.recipe}")

        try:
            input_file_path = self.cfg.recipe
            run_info = self._job_launcher._prepare_job(input_file_path)

            return self._job_launcher._build_command(run_info)
        except Exception as e:
            error_msg = f"Failed to generate base launch command: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e

    def __find_hydra_config(self, artifact_dir: str) -> str:
        """Find hydra config file in artifact directory"""
        import glob

        # Find the hydra config file (pattern: *_hydra.yaml)
        pattern = f"{artifact_dir}/*_hydra.yaml"
        hydra_files = glob.glob(pattern)
        if not hydra_files:
            raise FileNotFoundError(f"No hydra config file found matching pattern: {pattern}")

        config_path = hydra_files[0]  # Take first match
        self.logger.info(f"Found hydra config at {config_path}")
        return config_path

    def launch(self, override_commands: list[str] = [], dryrun: bool = False, max_retry=2) -> tuple[dict, bool]:
        """
        Submit job and wait for completion.

        Args:
            override_commands: additional commands to append to launch
            dryrun: flag to enable dryrun for launch, default false
            max_retry: Maximum number of retry attempts, default 2

        Returns:
            job_details
            dryrun:
                {
                    "config_path": "/path/to/base/recipe/hydra/artifact/llama-3-1-8b-instruct-z5kpa_hydra.yaml"
                }

            non dryrun:
                {
                    "job_name": "llama-3-1-8b-instruct-qvtmq",
                    "pod_name": "llama-3-1-8b-instruct-qvtmq-1-nodes-7987d68bdf-x9vcv",
                    "output_folder_path": "'/path/to/output/llama-3-1-8b-instruct-qvtmq/llama-3-1-8b-instruct-qvtmq_submission.sh'",
                    "all_pods": [
                        "llama-3-1-8b-instruct-qvtmq-1-nodes-7987d68bdf-x9vcv"
                    ],
                    "log_path": "path/to/output/pod.logs"
                }
        """
        for attempt in range(max_retry):
            try:
                return self.__execute_launch(override_commands, dryrun)
            except Exception as e:
                if attempt < max_retry - 1:
                    backoff_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s...
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                else:
                    error_msg = f"Failed after {max_retry} attempts with overrides {override_commands}: {e}"
                    self.logger.error(error_msg)
                    self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                    raise

    def __execute_launch(self, overrides: list[str], dryrun: bool) -> tuple[dict, bool]:
        """Execute a single launch attempt"""
        self.logger.debug(f"Executing with overrides: {prettify(overrides)}")
        launch_command = [
            *self.__base_launch_command,
            "++recipes.training_config.training_args.log_gpu_memory=True",
            f"++base_results_dir={self.cfg.base_results_dir}",
            f"+cluster.custom_labels.run_identifier={self.__run_identifier}",
            *overrides,
        ]

        if dryrun:
            launch_command.append("++dry_run=True")

        self.logger.debug(f"Executing launch command: {prettify(launch_command)}")

        output = self._job_launcher._execute_command(launch_command)

        if dryrun:
            artifact_dir = ""
            for line in output.stdout.split("\n"):
                match = DRY_RUN_ARTIFACT_PATTERN.search(line)
                if match:
                    artifact_dir = match.group(1).strip()
                    break

            if not artifact_dir:
                raise RuntimeError("Artifact directory not defined. Unable to find base recipe config.")

            return {"config_path": self.__find_hydra_config(artifact_dir)}, True
        else:
            job_details = self.__get_job_details(output)

            self.logger.info(f"Launch details: {prettify(job_details)}")

            job_success = self.__monitor_job(job_details)

            if not os.path.exists(job_details["log_path"]):
                raise FileNotFoundError(f"Log file not found: {job_details['log_path']}")

            return job_details, job_success

    def __monitor_job(self, job_details):
        self.logger.info(f"Waiting for job completion: {job_details['job_name']}")

        # Add job to recorder before monitoring
        self._job_recorder.add_job(
            input_filename=self.auto_config.recipe,
            output_path=job_details.get("output_folder_path", ""),
        )

        return self._job_launcher._monitor_job(self.cfg.recipe, job_details)

    def __get_job_details(self, output):
        job_details = self._job_launcher._parse_output(self.auto_config.recipe, output)
        # Add config_path from output_folder_path
        output_folder_path = job_details.get("output_folder_path", "")
        if output_folder_path:
            artifact_dir = os.path.dirname(output_folder_path.strip("'\""))
            job_details["config_path"] = self.__find_hydra_config(artifact_dir)

            # Add log path - logs are saved to base_results_dir/{job_name}/{pod_name}.logs
            job_name = job_details.get("job_name", None)
            pod_name = job_details.get("pod_name", None)
            if job_name and pod_name:
                job_details["log_path"] = os.path.join(self.cfg.base_results_dir, job_name, f"{pod_name}.logs")
            else:
                raise ValueError("Unable to determine log_path for job")

        return job_details

    def __configure_launcher(self, platform: str, log_level=logging.ERROR) -> BaseLauncher:
        """Configure and return launcher with suppressed logs"""
        launcher = select_validation_launcher(platform)(self._job_recorder, self.cfg)
        launcher.logger.setLevel(log_level)

        return launcher
