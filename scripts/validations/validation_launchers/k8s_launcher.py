import os
import re
import subprocess
import threading
import time
from typing import Dict

from scripts.validations.validation_launchers.path_utils import get_project_root

from .base_launcher import BaseLauncher

HP_PYTORCH_JOB_RESOURCE_NAME = "hyperpodpytorchjob"
RAY_JOBS_RESOURCE_NAME = "rayjobs"
RAY_TRAINING_NAMESPACE = "ray-training"

# Keywords that indicate Ray-based verl jobs (RLVR/RLAIF use Ray, SFT uses torchrun)
VERL_RAY_KEYWORDS = ["rlvr", "rlaif"]


def _is_verl_ray_job(input_file_path: str) -> bool:
    path_lower = input_file_path.lower()
    return any(kw in path_lower for kw in VERL_RAY_KEYWORDS)


class K8sValidationLauncher(BaseLauncher):
    """Kubernetes-specific job launcher"""

    def _parse_output(self, input_file_path: str, launch_output) -> Dict:
        job_name = ""
        output_folder_path = ""
        pod_name = ""

        for line in launch_output.stdout.split("\n"):
            if "submission file created at" in line:
                output_folder_path = line.split(" ")[-1]
            elif "NAME:" in line:
                job_name = line.split(" ")[-1]
                pod_name = job_name

        # Dynamically find the pod name after job is created
        if job_name:
            is_verl = _is_verl_ray_job(input_file_path)

            self._wait_for_job_status(job_name, is_verl=is_verl)

            all_pods = self.get_pods(job_name, is_verl=is_verl)
            pod_name = self.get_head_node(job_name, all_pods=all_pods, is_verl=is_verl)

        if not all([job_name, pod_name, output_folder_path]):
            print(f"\n{'='*80}")
            print(f"❌ JOB LAUNCH FAILED: {input_file_path}")
            print(f"{'='*80}")
            print(f"job_name: {job_name or 'MISSING'}")
            print(f"pod_name: {pod_name or 'MISSING'}")
            print(f"output_folder_path: {output_folder_path or 'MISSING'}")
            print(f"\nLaunch stdout:")
            print(launch_output.stdout if hasattr(launch_output, "stdout") else str(launch_output))
            print(f"\nLaunch stderr:")  # ADD THIS
            print(launch_output.stderr if hasattr(launch_output, "stderr") else "N/A")  # ADD THIS
            print(f"{'='*80}\n")

            self.job_recorder.update_job(
                input_file_path, output_path=output_folder_path, status="Failed", output_log=pod_name
            )
            return None

        result = {
            "job_name": job_name,
            "pod_name": pod_name,
            "output_folder_path": output_folder_path,
        }

        if "all_pods" in locals():
            result["all_pods"] = all_pods

        return result

    def _monitor_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """Monitor job until completion"""
        is_verl = _is_verl_ray_job(input_file_path)

        job_name = job_details["job_name"]
        pod_name = job_details["pod_name"]
        output_folder_path = job_details["output_folder_path"]
        job_successful = False
        tokens_per_sec = None

        pod_timeout = 45 if is_verl else 10
        self._wait_for_pod_status(pod_name, "Running", is_verl, timeout_in_minutes=pod_timeout)

        # Start log collection in background
        log_thread = threading.Thread(target=self._collect_pod_logs, args=(job_name, pod_name, is_verl))
        log_thread.daemon = True
        log_thread.start()

        resource_name = RAY_JOBS_RESOURCE_NAME if is_verl else HP_PYTORCH_JOB_RESOURCE_NAME
        while True:
            try:
                # Check job status
                status = self._get_job_status(job_name, is_verl)

                if any(s in status for s in ["SUCCEEDED", "Completed"]):
                    job_successful = True
                    break
                elif any(s in status for s in ["FAILED", "Failed"]):
                    job_successful = False
                    break

                # Check if log thread stopped (max errors reached)
                if not log_thread.is_alive():
                    self.logger.info(f"Fast-failing. Log collection stopped for {pod_name}")
                    job_successful = False
                    break

                time.sleep(poll_sec)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error monitoring job: {e.stderr}")
                break

        # Wait for log thread to finish
        log_thread.join(timeout=5)

        # Read logs for throughput calculation
        log_save_path = self._construct_log_path(job_name, pod_name)
        if not is_verl and job_successful and os.path.exists(log_save_path):
            with open(log_save_path, "r") as f:
                logs = f.read()
            tokens_per_sec = self.calculate_throughput_from_logs(logs, job_name, job_successful, input_file_path)

        # Cleanup job
        self.clean_up_resource(resource_name, job_name, is_verl=is_verl)

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=pod_name,
            tokens_throughput=tokens_per_sec if tokens_per_sec else "N/A",
        )
        return job_successful

    def _collect_pod_logs(self, job_name: str, pod_name: str, is_verl: bool = False, training_error_limit: int = 3):
        """Collect and save pod logs"""

        logs_command = ["kubectl", "logs", "-f", pod_name]
        if is_verl:
            logs_command += ["-n", RAY_TRAINING_NAMESPACE]

        proc = subprocess.Popen(
            logs_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=self.aws_env,
        )
        logs = []
        log_save_path = self._construct_log_path(job_name, pod_name)
        os.makedirs(os.path.dirname(log_save_path), exist_ok=True)

        with open(log_save_path, "w") as log_file:
            try:
                error_count = 0
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    logs.append(line)

                    if any(
                        s in line.lower()
                        for s in ["exception occurred during training:", "training related exception occurred"]
                    ):
                        error_count += 1
                        self.logger.info(f"Detected error in {job_name}: {line}")
                        if error_count >= training_error_limit:
                            self.logger.info(
                                f"Training error limit ({training_error_limit}) reached, stopping log stream."
                            )
                            break
            finally:
                proc.terminate()
                proc.wait()

        self.logger.info(f"Saved K8s Job:{job_name} logs at '{log_save_path}'")
        return " ".join(logs)

    def _wait_for_pod_status(self, pod_name: str, status: str, is_verl=False, timeout_in_minutes: int = 10) -> None:
        self.logger.debug(f"Waiting for pod {pod_name} to reach status {status}")
        wait_command = [
            "kubectl",
            "wait",
            f"--for=jsonpath={{.status.phase}}={status}",
            f"pod/{pod_name}",
            f"--timeout={timeout_in_minutes}m",
        ]
        if is_verl:
            wait_command += ["-n", RAY_TRAINING_NAMESPACE]
        try:
            subprocess.run(
                wait_command,
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
            self.logger.debug(f"Pod {pod_name} has reached status: {status}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"Failed to wait for {pod_name} to reach status {status}: {e.stderr}")
            try:
                self.logger.info(self.describe_resource("pod", pod_name).stdout)
            except:
                pass

            raise e

    def _get_job_status(self, job_name: str, is_verl: bool) -> str:
        """Get job status string."""
        resource_name = self._get_resource_name(is_verl)
        cmd = ["kubectl", "get", resource_name, job_name, "-o"]

        if is_verl:
            cmd += ["jsonpath={.status.jobStatus}", "-n", RAY_TRAINING_NAMESPACE]
        else:
            cmd += ["jsonpath={.status.conditions[-1].type} {.status.conditions[-1].status}"]

        return subprocess.run(cmd, capture_output=True, text=True, check=True, env=self.aws_env).stdout.strip()

    def _wait_for_job_status(self, job_name: str, status: str = "Running", is_verl: bool = False, timeout: str = "30m"):
        """Wait for job to reach specified status"""
        resource_name = self._get_resource_name(is_verl)
        self.logger.info(f"Waiting for job {job_name} to reach {status}")

        cmd = ["kubectl", "wait", f"{resource_name}/{job_name}", f"--timeout={timeout}"]
        try:
            if is_verl:
                # RayJobs use jobDeploymentStatus, not conditions
                cmd += [
                    f"--for=jsonpath={{.status.jobDeploymentStatus}}={status}",
                    "-n",
                    RAY_TRAINING_NAMESPACE,
                ]
            else:
                cmd += [
                    f"--for=condition={status}",
                ]

            subprocess.run(cmd, capture_output=True, text=True, check=True, env=self.aws_env)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"Failed to wait for job to reach {status}: {job_name}")
            try:
                self.logger.info(self.describe_resource(resource_name, job_name, is_verl=is_verl).stdout)
            except:
                pass
            self.clean_up_resource(resource_name, job_name, is_verl=is_verl)
            raise e

    def describe_resource(self, resource_type: str, resource_name: str, is_verl: bool = False):
        try:
            cmd = ["kubectl", "describe", resource_type, resource_name]
            if is_verl:
                cmd.extend(["-n", RAY_TRAINING_NAMESPACE])

            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to describe {resource_type} {resource_name}: {e.stderr}")
            raise e

    def clean_up_resource(self, resource, job_name: str, is_verl: bool = False):
        self.logger.info(f"Cleaning up {resource} job: {job_name}")
        try:
            cmd = ["kubectl", "delete", resource, job_name]
            if is_verl:
                cmd.extend(["-n", RAY_TRAINING_NAMESPACE])

            subprocess.run(cmd, capture_output=True, text=True, check=True, env=self.aws_env)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up {resource} {job_name}: {e.stderr}")

    def get_pods(self, job_name, is_verl=False):
        """Returns the pods for a job."""
        self.logger.info(f"Fetching pods for job: {job_name}")

        try:
            cmd = ["kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name"]
            if is_verl:
                cmd.extend(["-n", RAY_TRAINING_NAMESPACE])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                env=self.aws_env,
            )

            pods = [p.strip() for p in result.stdout.split("\n") if p.strip().startswith(job_name)]
            self.logger.info(f"Found pods for job: {pods}")

            return pods
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"Failed to get pods for job {job_name}")
            raise e

    def get_head_node(self, job_name, all_pods=None, max_retries=6, delay=10, is_verl=False):
        """Returns the head node (group_rank=0) for a job."""
        pod_names = self.get_pods(job_name, is_verl=is_verl) if not all_pods else all_pods

        if len(pod_names) == 1:
            return pod_names[0]

        # For VERL RayJobs, head pod has '-head-' in the name
        if is_verl:
            head_pods = [p for p in pod_names if "-head-" not in p and "-worker-" not in p]
            if head_pods:
                self.logger.info(f"Returning VERL launcher pod: {head_pods[0]}")
                return head_pods[0]

        # For non-VERL jobs, find head by group_rank=0 in logs
        for attempt in range(max_retries):
            for pod in pod_names:
                try:
                    result = subprocess.run(
                        ["kubectl", "logs", pod],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        env=self.aws_env,
                    )
                    if result.stdout.strip():
                        match = re.search(r"group_rank=(\d+)", result.stdout)
                        if match and match.group(1) == "0":
                            self.logger.info(f"Returning head node: {pod}")
                            return pod
                except subprocess.TimeoutExpired:
                    continue

            self.logger.info(f"Logs not ready yet, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)

        raise RuntimeError(f"Unable to determine head node for job {job_name}")

    def _construct_log_path(self, job_name: str, pod_name: str) -> str:
        """Construct log file path for a job"""
        return os.path.join(
            self.config.get("base_results_dir", os.path.join(get_project_root(), "results")),
            job_name,
            f"{pod_name}.logs",
        )

    def _get_resource_name(self, is_verl: bool):
        return RAY_JOBS_RESOURCE_NAME if is_verl else HP_PYTORCH_JOB_RESOURCE_NAME
