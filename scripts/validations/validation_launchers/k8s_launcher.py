import os
import subprocess
import time
from typing import Dict

from .base_launcher import BaseLauncher

VALID_POD_STATUS = {"Pending", "ContainerCreating", "deployed", "Running"}


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
                # For VERL/Ray jobs, the pod naming convention is different
                if "verl" in input_file_path:
                    pod_name = job_name
                else:
                    pod_name = job_name + "-worker-0"

        if not all([job_name, pod_name, output_folder_path]):
            self.job_recorder.update_job(
                input_file_path, output_path=output_folder_path, status="Failed", output_log=pod_name
            )
            return None

        return {"job_name": job_name, "pod_name": pod_name, "output_folder_path": output_folder_path}

    def _monitor_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """monitoring job dispatcher method"""
        if "verl" in input_file_path:
            return self._monitor_ray_job(input_file_path, job_details, poll_sec)
        else:
            return self._monitor_pytorch_job(input_file_path, job_details, poll_sec)

    def _monitor_ray_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """Monitor a Ray/VERL job until completion"""
        job_name = job_details["job_name"]
        output_folder_path = job_details["output_folder_path"]
        job_successful = False

        # get ray job pod
        ray_cluster_check_cmd = (
            f"kubectl get pods -n ray-training --no-headers -o custom-columns=:metadata.name "
            f"| grep '{job_name}-' "
            "| grep -vE 'head|worker'"
        )

        time.sleep(45)  # Initial wait for Ray cluster to start
        while True:
            try:
                result = subprocess.check_output(ray_cluster_check_cmd, shell=True, text=True).strip()
                job_pod_name = str(result)
                if job_pod_name:
                    # Try to get logs from the node, if it exists
                    try:
                        logs = self._collect_pod_logs(job_name, job_pod_name, True)
                        job_successful = self._validate_logs(logs, job_pod_name)
                        break
                    except subprocess.CalledProcessError:
                        self.logger.warning(f"Job node for {job_name} not ready yet, waiting...")

                time.sleep(poll_sec)
            except subprocess.CalledProcessError:
                self.logger.error(f"Error checking Ray cluster status for {job_name}")
                break

        # Cleanup Ray job
        try:
            subprocess.run(
                ["kubectl", "delete", "rayjobs", f"{job_name}", "-n", "ray-training"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up Ray job {job_name}: {e.stderr}")

        status = "Complete" if job_successful else "Failed"
        self.job_recorder.update_job(
            input_filename=input_file_path,
            output_path=output_folder_path,
            status=status,
            output_log=job_pod_name if "job_pod_name" in locals() else "",
        )
        return job_successful

    def _monitor_pytorch_job(self, input_file_path: str, job_details: Dict, poll_sec: int = 10) -> bool:
        """Monitor a PyTorch job until completion"""
        job_name = job_details["job_name"]
        pod_name = job_details["pod_name"]
        output_folder_path = job_details["output_folder_path"]
        job_successful = False

        time.sleep(30)  # Initial wait for pod to start
        tokens_per_sec = None
        while True:
            try:
                current_pod_status = subprocess.run(
                    ["kubectl", "get", "pods", pod_name], capture_output=True, text=True, check=True
                ).stdout

                if not any(status in current_pod_status for status in VALID_POD_STATUS):
                    break

                if "Running" in current_pod_status:
                    logs = self._collect_pod_logs(job_name, pod_name, False)
                    job_successful = self._validate_logs(logs, pod_name)

                    # Extract and record throughput data
                    tokens_per_sec = self.calculate_throughput_from_logs(
                        logs, job_name, job_successful, input_file_path
                    )
                    break
                time.sleep(poll_sec)

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error monitoring PyTorch job: {e.stderr}")
                break

        # Cleanup PyTorch job
        try:
            subprocess.run(["kubectl", "delete", "pytorchJob", job_name], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning up PyTorch job {job_name}: {e.stderr}")

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

        logs_command = ["kubectl", "logs", "-f"]
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
        log_save_path = os.path.join(self.config.base_results_dir, f"{job_name}.logs")
        self.logger.info(f"Saving K8s Job:{job_name} logs at '{log_save_path}'")
        with open(log_save_path, "w") as f:
            f.writelines(logs)

        return " ".join(logs)

    def _validate_logs(self, logs, pod_name, threshold=5):
        epoch_count = logs.lower().count("epoch=")

        if "verl" in pod_name:
            # For VERL jobs, check for step
            epoch_count = logs.lower().count("step:")

        isComplete = False
        try:
            pod_status = self.get_pod_status(pod_name)
            isComplete = pod_status == "Succeeded"
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting pod status: {e.stderr}")
        return epoch_count >= threshold and isComplete

    def get_pod_status(self, pod_name: str) -> str:
        result = subprocess.run(
            ["kubectl", "get", "pod", pod_name, "--output", "jsonpath={.status.phase}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
