import logging
import os
import re
import subprocess
import time

from .base_launcher import BaseLauncher


class SageMakerJobsValidationLauncher(BaseLauncher):
    """SageMaker Training Jobs launcher"""

    def launch_job(self, recipe) -> bool:
        """Launch a single Slurm job and return True if successful, False otherwise"""
        run_info = self._prepare_job(recipe)

        launch_command = self._build_command(run_info)
        try:
            # Capture bash output after submitting the job
            launch_output = subprocess.run(launch_command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Launcher script '{recipe}' failed because of :- {e.stderr}")
            self.job_recorder.update_job(
                input_filename=recipe, status="Failed", output_log="Job failed. Error msg printed to cli"
            )
            return False
        except Exception as e:
            logging.error(f"Error launching job for recipe '{recipe}': {e}")
            self.job_recorder.update_job(
                input_filename=recipe, status="Failed", output_log="Job failed. Error msg printed to cli"
            )
            return False

        logging.info(f"Launch output for recipe '{recipe}' :- {launch_output.stdout}")
        launch_stdout = launch_output.stdout.split("\n")

        training_job_name, output_folder_path, status = self._parse_output(launch_stdout)

        if training_job_name == "" or status in ["Failed", "Stopped"]:
            self.job_recorder.update_job(input_filename=recipe, status=status, output_log=launch_output.stdout)
            return False
        return self._monitor_job(recipe, training_job_name, output_folder_path)

    def _parse_output(self, launch_stdout):
        job_name = ""
        training_job_name = ""

        for line in launch_stdout:
            match = re.search(r"Job (.*?) submission file created at '(.*?)'", line)
            if match:
                job_name = match.group(1)
                output_folder_path = os.path.dirname(match.group(2))
                break

        # truncate to fit sagemakers limits
        if len(job_name) > 39:
            job_name = job_name[:39]

        time.sleep(30)  # Wait for a while to ensure the training job is created
        results_dict = self.sagemaker_client.list_training_jobs(NameContains=f"{job_name}")
        if results_dict["TrainingJobSummaries"] == []:
            logging.error(f"No training jobs found with name containing '{job_name}'")
            return "", output_folder_path, "Failed"
        print("results_dict", results_dict)
        training_job_name = results_dict["TrainingJobSummaries"][0]["TrainingJobName"]
        status = results_dict["TrainingJobSummaries"][0]["TrainingJobStatus"]

        return training_job_name, output_folder_path, status

    def _monitor_job(self, recipe, training_job_name, output_folder_path):
        """Wait for the job to complete 5 epochs"""

        job_successful = False
        output_log_cloudwatch = f"Check cloudwatch logs at /aws/sagemaker/TrainingJobs/{training_job_name}"
        tokens_per_sec = None
        try:
            while True:
                time.sleep(300)  # Wait before next check
                status = self.sagemaker_session.describe_training_job(training_job_name)
                current_status = status["TrainingJobStatus"]
                secondary_status = status["SecondaryStatus"]

                logging.info(
                    f"Job {training_job_name}: Primary job status is: {current_status}, secondary job status is {secondary_status}"
                )

                if current_status in ["Failed", "Stopped"]:
                    break
                elif current_status in ["InProgress", "Completed"]:
                    if secondary_status in ["Training"]:
                        continue
                    elif current_status == "Completed":
                        job_successful = True
                        break
                time.sleep(600)  # Wait before next check
        except Exception as e:
            if hasattr(e, "stderr"):
                logging.error(f"Error waiting for job to complete: {e.stderr}")
            else:
                logging.error(f"Error waiting for job to complete: {e}")

        if job_successful and self._validate_logs(training_job_name):
            # Extract throughput data from CloudWatch logs using new method
            tokens_per_sec = self.get_job_throughput_from_cloudwatch(training_job_name, recipe)

            # Record the job as complete with throughput data
            self.job_recorder.update_job(
                input_filename=recipe,
                output_path=output_folder_path,
                status="Complete",
                output_log=output_log_cloudwatch,
                tokens_throughput=tokens_per_sec,
            )

            # Also use the record_throughput method for logging
            if tokens_per_sec:
                self.record_throughput(recipe, tokens_per_sec)
        else:
            self.job_recorder.update_job(
                input_filename=recipe, output_path=output_folder_path, status="Failed", output_log=output_log_cloudwatch
            )
        return True

    def _validate_logs(self, training_job_name, threshold=5):
        log_group = f"/aws/sagemaker/TrainingJobs"
        epoch_count = 0
        response = self.logs_client.filter_log_events(
            logGroupName=log_group,
            logStreamNamePrefix=training_job_name,
            startTime=int((time.time() - 900) * 1000),
            endTime=int(time.time() * 1000),
            filterPattern="step=",
        )
        for event in response["events"]:
            message = event["message"].lower()
            if "step=" in message:
                epoch_count += 1
        return epoch_count >= threshold

    def get_job_throughput_from_cloudwatch(self, training_job_name, recipe=None):
        """Extract throughput data from SageMaker CloudWatch logs"""
        log_group = f"/aws/sagemaker/TrainingJobs"

        try:
            # Get logs from the last 15 minutes (900 seconds)
            response = self.logs_client.filter_log_events(
                logGroupName=log_group,
                logStreamNamePrefix=training_job_name,
                startTime=int((time.time() - 900) * 1000),
                endTime=int(time.time() * 1000),
                filterPattern="",
            )

            # Collect all log messages
            all_logs = []
            for event in response["events"]:
                all_logs.append(event["message"])

            # Join all logs into a single string
            combined_logs = "\n".join(all_logs)

            # Use the new calculate_throughput_from_logs method
            tokens_per_sec = self.calculate_throughput_from_logs(combined_logs, training_job_name, True, recipe)

            if tokens_per_sec:
                self.logger.info(
                    f"Found throughput {tokens_per_sec} tokens/sec in CloudWatch logs for job {training_job_name}"
                )
                return tokens_per_sec
            else:
                self.logger.warning(f"No throughput data found in CloudWatch logs for job {training_job_name}")
                return None

        except Exception as e:
            self.logger.error(f"Error retrieving throughput from CloudWatch logs for job {training_job_name}: {e}")
            return None
