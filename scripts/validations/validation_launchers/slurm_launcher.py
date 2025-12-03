import subprocess
import time
from typing import Dict

from .base_launcher import BaseLauncher


class SlurmValidationLauncher(BaseLauncher):
    """Slurm-specific job launcher"""

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
            self.job_recorder.update_job(input_file_path, output_folder_path, "Failed", log_file_path)

        return {"job_id": job_id, "log_file_path": log_file_path, "output_folder_path": output_folder_path}

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
                sq = subprocess.check_output(
                    ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "JobID,State"], text=True
                ).strip()
                if not sq:
                    break
                time.sleep(poll_sec)
                _, job_state = sq.split("|")
                if job_state in ["COMPLETED"]:
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
                    self.logger.info(f"Job {job_id} not deployed yet")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error monitoring job: {e}")

        # Cleanup
        try:
            subprocess.run(["scancel", job_id], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up job: {e}")

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=log_file_path,
            tokens_throughput=tokens_per_sec if tokens_per_sec else "N/A",
        )
        return job_successful

    def _collect_job_logs(self, log_file: str) -> list:
        logs = " ".join(open(log_file, "r").readlines())
        return logs

    def _validate_logs(self, logs, threshold=5):
        epoch_count = logs.lower().count("epoch=")
        error_count = logs.lower().count("srun: error:")
        return epoch_count >= threshold and error_count == 0
