# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
import shutil
from pathlib import Path
from typing import Any, List, Union

import nemo_launcher.utils.job_utils as job_utils
from nemo_launcher.core.launchers import SlurmLauncher
from nemo_launcher.core.logger import logger

NEMO_LAUNCHER_DEBUG = os.getenv("NEMO_LAUNCHER_DEBUG", "False").lower() in (
    "true",
    "t",
    "1",
)


class SMJobPaths(job_utils.JobPaths):
    """
    Our launcher contains an extra entry script called train_script.sh
    This class is used to specify its path
    """

    @property
    def train_script_file(self) -> Path:
        return self._folder / f"train_script.sh"

    @property
    def launch_docker_container_file(self) -> Path:
        return self._folder / f"launch_docker_container.sh"

    @property
    def docker_exec_script_file(self) -> Path:
        return self._folder / f"docker_exec_script.sh"


class SMSlurmLauncher(SlurmLauncher):
    """
    Launcher for SM training jobs using slurm.
    This launcher will launch the job using `torchrun`, unlike the NeMo slurm launcher which use Pytorch lightning
    to handle the torch.distributed. This launcher will create a separate train_script.sh with proper `torchrun` distributed arg prepared.
    Checking `_make_train_script_text` function in stage.py for more details.
    """

    def __init__(self, folder: Union[Path, str], job_name: str, **kwargs: Any) -> None:
        # We need to handle this ntasks_per_node specifically
        # Since we are using torchrun to launch custom jobs, we can not use ntasks_per_node in sbatch command
        self.ntasks_per_node = kwargs.pop("ntasks_per_node", 8)
        if "train_script_text" in kwargs:
            self.train_script_text = kwargs.pop("train_script_text")
        else:
            raise ValueError(f"Missing train_script_text from launcher kwargs {kwargs}")
        self.launch_docker_container_text = kwargs.pop("launch_docker_container_text", None)
        self.docker_exec_script_text = kwargs.pop("docker_exec_script_text", None)
        self.slurm_create_submission_file_only = kwargs.pop("slurm_create_submission_file_only", False)
        if "hostfile" in kwargs:
            self.hostfile = kwargs.pop("hostfile")
        else:
            raise ValueError(f"Missing hostfile from launcher kwargs {kwargs}")
        if "slurm_docker_cfg" in kwargs:
            kwargs.pop("slurm_docker_cfg")
        super(SlurmLauncher, self).__init__(folder, job_name)
        self.parameters = {}
        self._update_parameters(job_name=job_name, **kwargs)
        if shutil.which("srun") is None and not NEMO_LAUNCHER_DEBUG and not self.slurm_create_submission_file_only:
            raise RuntimeError('Could not detect "srun", are you indeed on a slurm cluster?')

    def _make_train_script_file(self):
        """
        Create the custom train_script.sh
        Optional create launch_docker_container.sh to launch docker container on every node
        """
        job_paths = SMJobPaths(folder=self.folder, job_name=self.job_name)
        folder = job_paths.folder
        folder.mkdir(parents=True, exist_ok=True)
        train_script_file_path = job_paths.train_script_file
        with train_script_file_path.open("w") as f:
            f.write(self.train_script_text)
        if self.launch_docker_container_text is not None:
            launch_docker_container_file = job_paths.launch_docker_container_file
            with launch_docker_container_file.open("w") as f:
                f.write(self.launch_docker_container_text)
        if self.docker_exec_script_text is not None:
            docker_exec_script_file = job_paths.docker_exec_script_file
            with docker_exec_script_file.open("w") as f:
                f.write(self.docker_exec_script_text)

    def launch(self, command_groups: List[List[str]]) -> str:
        # Create the custom train_script.sh before launching the real job
        self._make_train_script_file()

        # Same as upstream, but exposing extra control for submission through slurm_create_submission_file_only
        submission_file_path = self._make_submission_file(command_groups)
        logger.info(f"Job {self.job_name} submission file created at '{submission_file_path}'")
        job_id = ""
        if not NEMO_LAUNCHER_DEBUG and not self.slurm_create_submission_file_only:
            job_id = self._submit_command(submission_file_path)
            if job_id:
                logger.info(f"Job {self.job_name} submitted with Job ID {job_id}")
                with open(self.folder / "launcher.log", "w") as f:
                    f.write(f"Submitted batch job {job_id}")
        else:
            logger.info(f"To submit your job on Slurm, run `sbatch {submission_file_path}`")

        return job_id

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        The submission file will be responsible for the following
        - Handle sbatch config (implemented in upstream)
        - Handle env variables (implemented in upstream)
        - Handle storing distribution information which will be consumed by train_script.sh
        - Call train_script.sh with proper srun command
        """
        origin_sbatch_str = super()._make_submission_file_text(command_groups)
        origin_sbatch_str = origin_sbatch_str.split("\n")
        assert origin_sbatch_str[0] == "#!/bin/bash", origin_sbatch_str[0]
        command_idx = None
        for idx, sbatch_str in enumerate(origin_sbatch_str):
            if sbatch_str.startswith("# command"):
                command_idx = idx
                break
        assert command_idx is not None, f"Can not find command in the submission file str: {origin_sbatch_str}"
        distributed_strs = [
            "",
            "# Prepare distributed files",
            f'srun -l bash -c "scontrol show hostnames | sort > {self.hostfile}"',
            "",
        ]
        if self.launch_docker_container_text is None:
            updated_sbatch_str = origin_sbatch_str[:command_idx] + distributed_strs + origin_sbatch_str[command_idx:]
        else:
            updated_sbatch_str = origin_sbatch_str[:command_idx] + distributed_strs + command_groups[0]

        return "\n".join(updated_sbatch_str)
