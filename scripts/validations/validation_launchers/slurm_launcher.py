import subprocess
import time
from typing import Dict, Optional

from ..clients.slurm_client import SlurmClient
from .base_launcher import BaseLauncher


class SlurmValidationLauncher(BaseLauncher):
    """Slurm-specific job launcher"""

    def __init__(self, job_recorder, config):
        """
        Initialize Slurm launcher.

        Args:
            job_recorder: Job recorder instance
            config: Configuration object
        """
        super().__init__(job_recorder, config)

        self.slurm_client: Optional[SlurmClient] = None
        slurm_client_config = getattr(self.config, "slurm_client_config", None)
        if slurm_client_config:
            self.logger.info("Initializing Slurm launcher with remote execution")
            slurm_client_config = self.config.slurm_client_config

            self.slurm_client = SlurmClient(slurm_client_config, logger=self.logger, boto_session=self.boto_session)
        else:
            self.logger.info("Initializing Slurm launcher with local execution")

    def _execute_command(self, command: list):
        """Execute command"""
        cmd_str = " ".join(command)
        try:
            return (
                self.slurm_client.launch_job([cmd_str])
                if self.slurm_client
                else subprocess.run(cmd_str, capture_output=True, text=True, check=True, shell=True)
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing command: {e.stderr}")
            if self.slurm_client:
                self.slurm_client.cleanup()
            raise e

    def _parse_output(self, input_file_path: str, launch_output) -> Dict:
        job_id = ""
        log_file_path = ""
        output_folder_path = ""

        for line in launch_output.stdout.split("\n"):
            if "submission file created at" in line:
                output_folder_path = line.split(" ")[-1]
            elif "submitted with Job ID" in line:
                job_id = line.split(" ")[-1]
            elif "Submitted job's logfile path" in line:
                log_file_path = line.split(" ")[-1]

        if not all([job_id, log_file_path, output_folder_path]):
            self.job_recorder.update_job(
                input_file_path, output_path=output_folder_path, status="Failed", output_log=log_file_path
            )

        return {"job_id": job_id, "log_file_path": log_file_path, "output_folder_path": output_folder_path}

    def _get_job_state(self, job_id: str) -> str:
        """Get job state using wrapper or local subprocess."""
        cmd = ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "JobID,State"]
        try:
            sq = (
                self.slurm_client.run([" ".join(cmd)]).stdout
                if self.slurm_client
                else subprocess.check_output(cmd, text=True).strip()
            )

            for line in sq.split("\n"):
                line = line.strip()
                if line and "|" in line and not line.startswith("Starting") and not line.startswith("Exiting"):
                    _, job_state = line.split("|")
                    return job_state
        except Exception as e:
            self.logger.error(f"Failed to get job status: {e}")

        return "UNKNOWN"

    def _cancel_job(self, job_id: str) -> None:
        """Cancel job using wrapper or local subprocess."""
        try:
            cmd = ["scancel", job_id]
            self.slurm_client.run([" ".join(cmd)]) if self.slurm_client else subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
        except Exception as e:
            self.logger.error(f"Error cleaning up job: {e}")

    def _monitor_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 60) -> bool:
        job_id = job_details["job_id"]
        log_file_path = job_details["log_file_path"]
        output_folder_path = job_details["output_folder_path"]

        if not job_id:
            return False

        job_successful = False
        tokens_per_sec = None
        try:
            while True:
                job_state = self._get_job_state(job_id)

                if not job_state or job_state == "UNKNOWN":
                    break

                if job_state == "COMPLETED" or job_state == "FAILED":
                    current_job_logs = self._collect_job_logs(log_file_path)

                    if self._validate_logs(current_job_logs):
                        self.logger.info(f"Job {job_id} completed successfully")

                        # Extract and record throughput data using new method
                        tokens_per_sec = self.calculate_throughput_from_logs(
                            current_job_logs, job_id, True, input_file_path
                        )

                        job_successful = True
                        break

                    if "srun: error:" in current_job_logs.lower():
                        break

                    time.sleep(poll_sec)
                else:
                    self.logger.info(f"Job {job_id} state: {job_state}")
                    time.sleep(poll_sec)

        except Exception as e:
            self.logger.error(f"Error monitoring job: {e}")

        # Cleanup
        self._cancel_job(job_id)
        self.slurm_client.cleanup() if self.slurm_client else True

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=log_file_path,
            tokens_throughput=tokens_per_sec if tokens_per_sec else "N/A",
        )
        return job_successful

    def _collect_job_logs(self, log_file: str) -> str:
        if self.slurm_client:
            logs = self.slurm_client.run([f"cat {log_file}"]).stdout
            self.logger.info(f"Job logs: {logs}")
        else:
            logs = " ".join(open(log_file, "r").readlines())
        return logs

    def _validate_logs(self, logs):
        # Checking for errors
        error_count = logs.lower().count("srun: error:")
        return error_count == 0
