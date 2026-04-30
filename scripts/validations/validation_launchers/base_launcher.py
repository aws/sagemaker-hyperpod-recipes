import logging
import os
import subprocess
import traceback
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict

import boto3

from scripts.validations.validation_launchers.launcher_utils import (
    construct_k8_launch_command,
    construct_slurm_launch_command,
    construct_smjobs_launch_command,
    pre_launch_setup,
)


class BaseLauncher(ABC):
    """Abstract base class for platform-specific job launchers"""

    REFRESH_THRESHOLD_MINUTES = 15  # Refresh when this many minutes remain

    @staticmethod
    def assume_role_credentials(role_arn, session_name, logger):
        """Call STS AssumeRole and return (credentials_dict, expiry, boto3.Session)."""
        logger.info(f"Assuming role {role_arn} (session: {session_name})")
        sts_client = boto3.client("sts")
        assumed_role = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
        )
        creds = assumed_role["Credentials"]
        expiry = creds["Expiration"]
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
        return creds, expiry, session

    def __init__(self, job_recorder, config):
        self.job_recorder = job_recorder
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # Configure logger to include line numbers
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(filename)s:%(lineno)d: %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent duplicate logging

        assume_role_arn = getattr(config, "assume_role_arn", None)
        self._assume_role_arn = assume_role_arn
        if assume_role_arn:
            creds, self._credentials_expiry, self.boto_session = self.assume_role_credentials(
                role_arn=assume_role_arn,
                session_name=f"{config.platform.lower()}-validation-session",
                logger=self.logger,
            )
        else:
            self.logger.info("Creating default boto session")
            self.boto_session = boto3.Session()

        credentials = self.boto_session.get_credentials()
        self.aws_env = os.environ.copy()

        if credentials:
            self.aws_env.update(
                {
                    "AWS_ACCESS_KEY_ID": credentials.access_key,
                    "AWS_SECRET_ACCESS_KEY": credentials.secret_key,
                    "AWS_SESSION_TOKEN": credentials.token if credentials.token else "",
                }
            )

    def _refresh_credentials_if_needed(self):
        if not self._assume_role_arn or not self._credentials_expiry:
            return

        now = datetime.now(timezone.utc)
        if (self._credentials_expiry - now) < timedelta(minutes=self.REFRESH_THRESHOLD_MINUTES):
            self.logger.info("Credentials expiring soon, refreshing assumed role session...")
            creds, self._credentials_expiry, self.boto_session = self.assume_role_credentials(
                role_arn=self._assume_role_arn,
                session_name=f"{self.config.platform.lower()}-validation-refresh",
                logger=self.logger,
            )
            self.aws_env.update(
                {
                    "AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
                    "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"],
                    "AWS_SESSION_TOKEN": creds["SessionToken"],
                }
            )
            self.logger.info("✓ Credentials refreshed successfully")

    def launch_model_group(self, model_name, recipes):
        """Launch all jobs in parallel"""

        logging.info(f"Processing model group: {model_name} with {len(recipes)} recipes")

        if not recipes:
            logging.warning(f"No recipes to launch for model {model_name}")
            return

        # Launch all jobs in parallel
        logging.info(f"Launching all {len(recipes)} jobs for model {model_name} in parallel")
        with ThreadPoolExecutor(max_workers=min(len(recipes), 10)) as executor:
            futures = {executor.submit(self.launch_job, recipe): recipe for recipe in recipes}
            for future in futures:
                recipe = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Unhandled exception for recipe '{recipe}': {e}")
                    logging.error(f"Full traceback:\n{traceback.format_exc()}")
                    self.job_recorder.update_job(
                        input_filename=recipe, status="Failed", output_log=f"Unhandled exception: {e}"
                    )

    def launch_job(self, input_file_path: str) -> bool:
        """Main job launch workflow - template method"""
        try:
            run_info = self._prepare_job(input_file_path)
            launch_command = self._build_command(run_info)
            launch_output = self._execute_command(launch_command)
            job_details = self._parse_output(input_file_path, launch_output)
            if not job_details:
                return False
            logging.info(f"Job details:- {job_details}")
            job_success_status = self._monitor_job(input_file_path, job_details)
            return job_success_status
        except Exception as e:
            error_msg = f"Job launch failed for {input_file_path}: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self.job_recorder.update_job(input_filename=input_file_path, status="Failed", output_log=str(e))
            return False

    def _prepare_job(self, input_file_path: str) -> Dict:
        """Prepare job configuration"""
        return pre_launch_setup(self.config, input_file_path)

    def _build_command(self, run_info: Dict) -> list:
        """Build platform-specific launch command"""
        match self.config.platform:
            case "SMJOBS":
                command = construct_smjobs_launch_command(self.config, run_info)
            case "SLURM":
                command = construct_slurm_launch_command(self.config, run_info)
            case "K8":
                command = construct_k8_launch_command(self.config, run_info)
            case _:
                command = []

        # Append convergence training overrides if present
        # These override smoke-test limits (e.g., max_epochs=1, data limit=200)
        # with convergence params (e.g., total_epochs=15, no data limits)
        convergence_overrides = getattr(self.config, "_convergence_overrides", None)
        if convergence_overrides:
            self.logger.info("Applying convergence training overrides:")
            for key, value in convergence_overrides.items():
                override = f"{key}={value}"
                command.append(override)
                self.logger.info(f"  {override}")

        return command

    def _parse_output(self, input_file_path: str, launch_output) -> Dict:
        """Parse launch output to extract job details"""

    def _monitor_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 60) -> bool:
        """Monitor job until completion"""

    def _execute_command(self, command: list, cwd=None):
        """Execute launch command"""
        try:
            # Join command list into string and use shell=True to handle special chars in Hydra overrides
            cmd_str = " ".join(command)
            process = subprocess.run(
                cmd_str, capture_output=True, text=True, check=True, shell=True, cwd=cwd, env=self.aws_env
            )
            return process
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing command: {e.stderr}")
            raise e

    def _validate_logs(self, logs: str, threshold: int = 5) -> bool:
        """Common log validation logic"""
        epoch_count = logs.lower().count("epoch=")
        error_count = logs.lower().count("error:")
        return epoch_count >= threshold and error_count == 0

    def _should_compute_throughput(self, recipe):
        """Check if throughput should be computed for this recipe based on config keywords.

        Recipes matching any keyword in keywords_to_compute_throughput_for will have throughput computed.
        Recipes not matching will be marked as success if training job completes without computing throughput.
        """
        keywords = getattr(self.config, "keywords_to_compute_throughput_for", [])
        if not keywords:
            return True  # Default to computing throughput if no keywords configured

        recipe_lower = recipe.lower()
        return any(keyword.lower() in recipe_lower for keyword in keywords)

    def calculate_throughput_from_logs(self, logs, job_name, job_success_status=None, input_filename=None):
        """Parse throughput from dictionary pattern in logs and store in job recorder"""
        import re

        try:
            # Look for the dictionary pattern in logs
            # Pattern matches: {'status': 'Valid', 'Model': '...', 'Dataset': '...', ...}
            pattern = r"\{\s*'status':\s*'([^']+)'[^}]*\}"

            # Find all matches in the logs
            matches = re.finditer(pattern, logs, re.MULTILINE | re.DOTALL)

            for match in matches:
                dict_str = match.group(0)
                self.logger.info(f"Found throughput dictionary: {dict_str}")

                try:
                    # Parse the dictionary-like string
                    throughput_data = self._parse_throughput_dict(dict_str)

                    if throughput_data:
                        # Store the full throughput_data in job recorder (for duration, tokens, etc.)
                        if input_filename:
                            self.job_recorder.update_job(
                                input_filename=input_filename,
                                throughput_data=throughput_data,
                                job_success_status=job_success_status,
                            )
                            self.logger.info(
                                f"Stored throughput_data in job recorder for {input_filename}: {throughput_data}"
                            )

                        # Return throughput value or "invalid" based on status
                        status = throughput_data.get("status", "").lower()
                        if "invalid" in status:
                            self.logger.info(f"Job {job_name} has invalid status, returning 'invalid'")
                            return "invalid"

                        # Check if throughput is present
                        throughput_str = throughput_data.get("throughput", "")
                        if not throughput_str:
                            self.logger.error(f"No throughput field found for job {job_name} - treating as failure")
                            return None

                        # Extract numeric throughput value
                        if "tokens/second" in throughput_str:
                            try:
                                throughput_value = float(throughput_str.split()[0])
                                self.logger.info(
                                    f"Calculated throughput: {throughput_value} tokens/sec for job {job_name}"
                                )
                                return throughput_value
                            except (ValueError, IndexError) as e:
                                self.logger.error(f"Failed to parse throughput value '{throughput_str}': {e}")
                                return None
                        else:
                            self.logger.error(f"Invalid throughput format '{throughput_str}' for job {job_name}")
                            return None

                except Exception as e:
                    self.logger.error(f"Error parsing throughput dictionary: {e}")
                    continue

            self.logger.warning(f"No throughput dictionary pattern found in logs for job {job_name}")
            return None

        except Exception as e:
            self.logger.error(f"Error calculating throughput for job {job_name}: {e}")
            return None

    def _store_throughput_data(self, data, job_name, success, input_file):
        self.job_recorder.update_job(
            input_filename=input_file,
            throughput_model=data.get("model"),
            throughput_duration=data.get("duration"),
            throughput_tokens=data.get("tokens"),
            throughput_status=data.get("status"),
        )

    def _parse_throughput_dict(self, dict_str):
        """Parse the dictionary string and extract key-value pairs"""
        import re

        try:
            self.logger.info(f"Parsing throughput dictionary: {dict_str}")

            # Dictionary to store parsed values
            result = {}

            # Pattern to match key-value pairs in the dictionary
            # Handles both 'key': 'value' and 'key': value formats
            kv_pattern = r"'([^']+)':\s*'([^']*)'"

            matches = re.findall(kv_pattern, dict_str)
            self.logger.info(f"Found {len(matches)} key-value matches")

            for i, match in enumerate(matches):
                try:
                    self.logger.debug(f"Processing match {i}: {match}")

                    # With the simplified pattern, we only have 2 groups: key and value
                    if len(match) >= 2 and match[0] and match[1] is not None:
                        key = match[0].lower()
                        value = match[1].strip()
                        self.logger.debug(f"Parsed pair: '{key}' = '{value}'")
                        result[key] = value
                    else:
                        self.logger.warning(f"Skipping invalid match: {match}")
                        continue

                except Exception as e:
                    self.logger.error(f"Error processing match {i} ({match}): {e}")
                    self.logger.error(f"Match traceback:\n{traceback.format_exc()}")
                    continue

            self.logger.info(f"Successfully parsed {len(result)} key-value pairs: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in _parse_throughput_dict: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {}
