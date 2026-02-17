"""
Slurm client to support remote slurm execution.
Executes commands remotely via SSM start-session.
"""
import json
import logging
import os
import shlex
import shutil
import subprocess
import tarfile
import uuid
from dataclasses import dataclass
from typing import List

import boto3
import pathspec
from omegaconf import MISSING

from scripts.validations.validation_launchers.path_utils import get_project_root


@dataclass
class SlurmClientConfig:
    target_id: str = MISSING
    cluster_region: str = MISSING


def __init__(self, config: SlurmClientConfig):
    self.target_id = config.target_id
    self.region = config.cluster_region


class SlurmClient:
    """Ssm wrapper for remote execution of Slurm commands."""

    def __init__(
        self,
        config: SlurmClientConfig,
        logger: logging.Logger = logging.getLogger(__name__),
        boto_session: boto3.Session = boto3.Session(),
    ):
        """
        Initialize Slurm client wrapper.

        Args:
            config: SlurmClientConfig,
            logger:
            boto_session:
        """
        id = f"hyperpod_recipes_{uuid.uuid4()}"

        self.controller_work_dir = f"/fsx/validation_run/{id}"

        self.local_work_dir = f"/tmp/validation_run/{id}"

        self.s3_artifact_bucket = "hyperpod-recipes-validation-artifacts"
        self.s3_artifact_key_prefix = f"slurm_remote_validation_run/{id}"

        os.makedirs(self.local_work_dir, exist_ok=True)

        self.logger = logger
        self.target_id = config.target_id
        self.region = config.cluster_region
        self.boto_session = boto_session

        # Store AWS credentials from assumed session for CLI usage
        credentials = self.boto_session.get_credentials()
        self.process_env = {
            **os.environ.copy(),
            "AWS_ACCESS_KEY_ID": credentials.access_key,
            "AWS_SECRET_ACCESS_KEY": credentials.secret_key,
            "AWS_SESSION_TOKEN": credentials.token,
            "AWS_DEFAULT_REGION": config.cluster_region,
        }
        self.s3_client = self.boto_session.client("s3")
        self.logger.info(f"Successfully initialized SSM client for {config.target_id} in {config.cluster_region}")

        self._validate_platform_auth()
        self._setup_workdir()

    def launch_job(self, command: List[str], timeout_in_min: int = 60):
        command = [
            f"cd {self.controller_work_dir}",
            "source venv/bin/activate",
            " ".join(command),
        ]
        return self.run(command, timeout_in_min)

    def run(self, command: List[str], timeout_in_min: int = 5):
        command = [cmd.replace(str(get_project_root()), self.controller_work_dir) for cmd in command]

        self.logger.info(f"Executing command: {command}")
        command.append("echo Done")

        cmd_str = " && ".join(command)

        bash_wrapped = f"/bin/bash -c {shlex.quote(cmd_str)}"

        return self.__ssm_execute([bash_wrapped], timeout_in_min)

    def __ssm_execute(self, command: List[str], timeout_in_min: int = 5):
        parameters = json.dumps({"command": [" ".join(command)]})

        ssm_command = [
            "aws",
            "ssm",
            "start-session",
            "--region",
            self.region,
            "--target",
            self.target_id,
            "--document-name",
            "AWS-StartInteractiveCommand",
            "--parameters",
            parameters,
        ]
        self.logger.debug(f"Executing ssm command: {ssm_command}")
        try:
            process = subprocess.run(
                ssm_command,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_in_min * 60,
                env=self.process_env,
            )
            self.logger.debug(f"Process output: {process.stdout}")
            return process
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with error: {e.stderr}")
            raise

    def cleanup(self):
        self.logger.info("Cleaning up artifacts")
        self._clean_remote_dir()
        self._clean_s3_artifacts()
        self._clean_local_dir()

    def _clean_remote_dir(self):
        try:
            self.run([f"rm -rf {self.controller_work_dir}"])
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to clean remote working directory: {self.controller_work_dir}")

    def __del__(self):
        self.cleanup()

    def _setup_workdir(self):
        try:
            self.logger.info(f"Setting up work directory in target: {self.controller_work_dir}")
            self.run([f"mkdir -p {self.controller_work_dir}"])
            self.__sync_repository()

            # install dependencies
            self.run(
                [
                    "dpkg -l | grep -q python3.10-venv || apt install -y python3.10-venv",
                    f"python3 -m venv {self.controller_work_dir}/venv --copies",
                    f"source {self.controller_work_dir}/venv/bin/activate && pip3 install -r {self.controller_work_dir}/requirements.txt",
                ]
            )
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            self.cleanup()
            raise e

    def __sync_repository(self):
        self.logger.info("Syncing hyperpod recipes repository to slurm target")

        tarball = self.__create_repo_tarball()

        s3_key = f"{self.s3_artifact_key_prefix}/hyperpod-recipes.tar.gz"

        self.s3_client.upload_file(tarball, self.s3_artifact_bucket, s3_key)

        self.logger.info(f"Successfully uploaded to at s3://{self.s3_artifact_bucket}/{s3_key}")

        setup_commands = [
            f"aws s3 cp s3://{self.s3_artifact_bucket}/{s3_key} {self.controller_work_dir}/repo.tar.gz",
            f"tar -xzf {self.controller_work_dir}/repo.tar.gz -C {self.controller_work_dir}",
            f"rm {self.controller_work_dir}/repo.tar.gz",
        ]
        self.run(setup_commands)

        self.logger.info(f"Successfully copied repository to controller path: {self.controller_work_dir}")

    def __create_repo_tarball(self):
        repo_root = get_project_root()
        tarball = f"{self.local_work_dir}/hyperpod-recipes.tar.gz"

        self.logger.info(f"Creating repo tarball at {tarball}")

        # Create tarball
        with open(os.path.join(repo_root, ".gitignore"), "r") as f:
            spec = pathspec.PathSpec.from_lines("gitignore", f)

        def exclude_filter(tarinfo):
            rel_path = tarinfo.name
            if spec.match_file(rel_path):
                return None
            return tarinfo

        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(repo_root, arcname=".", filter=exclude_filter)

        return tarball

    def _validate_platform_auth(self):
        self.logger.info("Validating connection to slurm cluster controller")
        self.run(["squeue"])

    def _clean_local_dir(self):
        """Clean up local artifacts"""
        for item in os.listdir(self.local_work_dir):
            path = os.path.join(self.local_work_dir, item)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

    def _clean_s3_artifacts(self):
        try:
            self.logger.info(f"Cleaning up s3 artifacts: s3://{self.s3_artifact_bucket}/{self.s3_artifact_key_prefix}")

            # List all objects with this prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.s3_artifact_bucket, Prefix=self.s3_artifact_key_prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    objects_to_delete.extend([{"Key": obj["Key"]} for obj in page["Contents"]])

            # Delete all objects
            if objects_to_delete:
                self.s3_client.delete_objects(Bucket=self.s3_artifact_bucket, Delete={"Objects": objects_to_delete})
                self.logger.info(f"Deleted {len(objects_to_delete)} objects from S3")
        except Exception as e:
            self.logger.warning(f"S3 cleanup failed: {e}")
