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

from pathlib import Path
from typing import List

import nemo_launcher.utils.job_utils as job_utils
from nemo_launcher.core.launchers import AutoLauncher, K8SLauncher, Launcher

from .slurm_launcher import SMSlurmLauncher


class SMAutoLauncher(AutoLauncher):
    """
    AutoLauncher object for Sagemaker
    """

    @staticmethod
    def get_launchers():
        """Returns supported launchers as a dictionary from launcher name to launcher class"""
        return {
            "bcm": SMSlurmLauncher,
            "k8s": SMK8SLauncher,
            "sm_jobs": SMJobsLauncher,
        }


class SMK8SLauncher(K8SLauncher):
    """
    Launcher for SM training jobs using K8s.
    """

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Generate the script to launch the Helm chart.
        A very simple bash script is generated which runs `helm install` for the
        Helm chart that was generated.

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        helm_charts = paths.folder / "k8s_template"
        job_name = self.job_name.replace("_", "-")

        extra_helm_args = ""
        if self.parameters.get("namespace", None):
            extra_helm_args += f" --namespace {self.parameters['namespace']}"

        # Apply a timeout of 15min in case images take a long time to bring up
        # or pre-install hooks take a while
        return f"#!/bin/bash\nhelm install --timeout=15m --wait {extra_helm_args} {job_name} {helm_charts}\n"


class SMJobsLauncher(Launcher):
    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Given the command groups, generate submission script file's text.
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"
        On interactive cluster, multi-gpu python scripts are launched with `torchrun --nproc_per_node=??`

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        # now create
        lines = ["#!/bin/bash", ""]

        for group_ind, command_group in enumerate(command_groups):
            command = "\n".join(command_group)
            lines.append(command)
        return "\n".join(lines)

    def _submit_command(self, submission_file_path: Path) -> str:
        command_list = ["bash", submission_file_path]
        # run
        job_utils.CommandFunction(command_list, ret_stdout=False, verbose=False)()  # explicit errors
        return ""
