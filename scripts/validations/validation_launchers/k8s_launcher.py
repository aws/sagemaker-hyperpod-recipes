import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict

from .base_launcher import BaseLauncher

HP_PYTORCH_JOB_RESOURCE_NAME = "hyperpodpytorchjob"


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
            if "verl" in input_file_path:
                pod_name = job_name
            else:
                try:
                    subprocess.run(
                        [
                            "kubectl",
                            "wait",
                            "--for=condition=Running",
                            f"{HP_PYTORCH_JOB_RESOURCE_NAME}/{job_name}",
                            "--timeout=30m",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        env=self.aws_env,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    self.logger.error(f"Failed to wait for job to each Running: {job_name}")
                    try:
                        self.logger.info(self.describe_resource(HP_PYTORCH_JOB_RESOURCE_NAME, job_name).stdout)
                    except:
                        pass
                    self.clean_up_resource(HP_PYTORCH_JOB_RESOURCE_NAME, job_name)
                    raise e

                all_pods = self.get_pods(job_name)
                pod_name = self.get_head_node(job_name, all_pods=all_pods)

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
        """monitoring job dispatcher method"""
        if "verl" in input_file_path:
            return self._monitor_ray_job(input_file_path, job_details, poll_sec)
        else:
            return self._monitor_pytorch_job(input_file_path, job_details, poll_sec)

    def _monitor_ray_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """Monitor a Ray/VERL job until completion"""
        job_name = job_details["job_name"]
        pod_name = job_details["pod_name"]
        output_folder_path = job_details["output_folder_path"]
        job_successful = False

        time.sleep(45)  # Initial wait for Ray cluster to start
        tokens_per_sec = None
        while True:
            try:
                output = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "rayjobs",
                        "-n",
                        "ray-training",
                        job_name,
                        "-o",
                        "jsonpath={.status.jobStatus} ",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=self.aws_env,
                ).stdout.strip()
                cond_status = output or None

                if cond_status == "SUCCEEDED":
                    logs = self._collect_pod_logs(job_name, pod_name, False)
                    job_successful = True
                    # Extract and record throughput data
                    tokens_per_sec = self.calculate_throughput_from_logs(
                        logs, job_name, job_successful, input_file_path
                    )
                    break

                if cond_status == "FAILED":
                    logs = self._collect_pod_logs(job_name, pod_name, False)
                    job_successful = False

                    break

                time.sleep(poll_sec)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error monitoring Ray job: {e.stderr}")
                break

        # Cleanup Ray job
        try:
            subprocess.run(
                ["kubectl", "delete", "rayjobs", f"{job_name}", "-n", "ray-training"],
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up Ray job {job_name}: {e.stderr}")

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=job_name if "job_name" in locals() else "",
        )
        return job_successful

    def _monitor_pytorch_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """Monitor a HyperPod job until completion"""
        job_name = job_details["job_name"]
        pod_name = job_details["pod_name"]
        output_folder_path = job_details["output_folder_path"]
        job_successful = False

        tokens_per_sec = None
        while True:
            try:
                self._wait_for_pod_status(pod_name, "Running")  # wait for pod to start
                output = subprocess.run(
                    [
                        "kubectl",
                        "get",
                        "hyperpodpytorchjobs",
                        job_name,
                        "-o",
                        "jsonpath={.status.conditions[-1].type} {.status.conditions[-1].status}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=self.aws_env,
                ).stdout.strip()
                if not output:
                    cond_type, cond_status = None, None
                else:
                    cond_type, cond_status = output.split()

                if cond_type == "Completed" and cond_status == "True":
                    logs = self._collect_pod_logs(job_name, pod_name, False)
                    job_successful = True
                    # Extract and record throughput data
                    tokens_per_sec = self.calculate_throughput_from_logs(
                        logs, job_name, job_successful, input_file_path
                    )
                    break

                if cond_type == "Failed" and cond_status == "True":
                    logs = self._collect_pod_logs(job_name, pod_name, False)
                    job_successful = False

                    break

                time.sleep(poll_sec)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error monitoring HyperPod job: {e.stderr}")
                break

        # Cleanup HyperPod job
        try:
            subprocess.run(
                ["kubectl", "delete", "hyperpodpytorchjobs", job_name],
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up HyperPod job {job_name}: {e.stderr}")

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=pod_name,
            tokens_throughput=tokens_per_sec if tokens_per_sec else "N/A",
        )
        return job_successful

    def _collect_pod_logs(self, job_name: str, pod_name: str, is_verl: bool) -> list:
        """Collect and save pod logs"""

        logs_command = ["kubectl", "logs"]
        if is_verl:
            logs_command += [pod_name, "-n", "ray-training"]
        else:
            logs_command += [pod_name]

        proc = subprocess.Popen(
            logs_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        logs = []
        try:
            for line in proc.stdout:
                logs.append(line)
        finally:
            proc.wait()
        # Save logs
        base_results_dir = os.path.join(Path.cwd(), "results")
        log_save_path = os.path.join(base_results_dir, f"{job_name}.logs")
        self.logger.info(f"Saving K8s Job:{job_name} logs at '{log_save_path}'")
        with open(log_save_path, "w") as f:
            f.writelines(logs)

        logs = " ".join(logs)
        self.logger.info(f"Logs for pod {pod_name}:\n{logs}")
        return logs

    def _wait_for_pod_status(self, pod_name: str, status: str, timeout_in_minutes: int = 10) -> None:
        self.logger.info(f"Waiting for pod {pod_name} to reach status {status}")
        try:
            subprocess.run(
                [
                    "kubectl",
                    "wait",
                    f"--for=jsonpath={{.status.phase}}={status}",
                    f"pod/{pod_name}",
                    f"--timeout={timeout_in_minutes}m",
                ],
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
            self.logger.info(f"Pod {pod_name} has reached status: {status}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.logger.error(f"Failed to wait for {pod_name} to reach status {status}: {e.stderr}")
            try:
                self.logger.info(self.describe_resource("pod", pod_name).stdout)
            except:
                pass

            raise e

    def describe_resource(self, resource_type: str, resource_name: str):
        try:
            return subprocess.run(
                [
                    "kubectl",
                    "describe",
                    resource_type,
                    resource_name,
                ],
                capture_output=True,
                text=True,
                check=True,
                env=self.aws_env,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to describe {resource_type} {resource_name}: {e.stderr}")
            raise e

    def clean_up_resource(self, resource, job_name: str):
        self.logger.info(f"Cleaning up {resource} job: {job_name}")
        try:
            subprocess.run(
                ["kubectl", "delete", resource, job_name], capture_output=True, text=True, check=True, env=self.aws_env
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up {resource} {job_name}: {e.stderr}")

    def get_pods(self, job_name):
        """Returns the pods for a job."""
        self.logger.info(f"Fetching pods for job: {job_name}")

        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name"],
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

    def get_head_node(self, job_name, all_pods=None, max_retries=6, delay=10):
        """Returns the head node (group_rank=0) for a job."""
        pod_names = self.get_pods(job_name) if not all_pods else all_pods

        if len(pod_names) == 1:
            return pod_names[0]

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
