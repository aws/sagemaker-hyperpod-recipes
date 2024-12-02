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

import re

from omegaconf import DictConfig


class ValueValidator:
    def __init__(self, config: DictConfig):
        self.config = config

    def validate(self) -> None:
        # For all below validations, we will check if the argument is present => it should pass the validation

        # Mandatory arguments for all workflows
        _validate_all_mandatory_argument(self.config)

        # Cluster type argument check for all workflows
        _validate_cluster_type_argument(self.config)

        # PV argument check for all workflows
        _validate_pv_arguments(self.config)

        # Volume argument check for all workflows
        _validate_volume_arguments(self.config)

        # Pull policy argument check for all workflows
        _validate_pull_policy_argument(self.config)

        # Restart policy argument check for all workflows
        _validate_restart_policy_argument(self.config)

        # Namespace argument regex check for k8 workflows
        _validate_namespace_argument(self.config)

        # Check for mandatory arguments for k8 custom script workflow
        _validate_k8_custom_script_workflow_mandatory_argument(self.config)

        # Check the git url setting
        _validate_git_url(self.config)


def _validate_mandatory_argument(argument, argument_name: str) -> None:
    """
    Check if the mandatory input argument is not None.

    Parameters:
    argument : The argument to validate.
    argument_name : The name of argument to validate.
    """
    if argument is None:
        raise ValueError("Missing mandatory argument " + argument_name + " is not provided")


def get_argument(config: DictConfig, argument_name: str):
    """
    Return the value of given argument_name from config

    Parameters:
    config : Configuration dictionary
    argument_name : The name of argument to fetch.

    Returns:
    bool: Argument if present else return None
    argument_name can have a nested key like key1.key2 in which case we will return config[key1][key2].
    If  config[key1] is None for above case we will return None
    """
    argument_name_splits = argument_name.split(".")
    if len(argument_name_splits) > 1:
        subconfig = config.get(argument_name_splits[0])
        if subconfig is None:
            return None
        argument_name_splits.pop(0)
        remaining_argument_name = ".".join(argument_name_splits)
        return get_argument(subconfig, remaining_argument_name)
    return config.get(argument_name)


def _validate_pv_arguments(config: DictConfig) -> None:
    """
    Check all the information needed for persistentVolumeClaim is provided if it is not None

    Parameters:
    config (DictConfig): Configuration dictionary
    """
    cluster_config_name = "cluster.cluster_config"
    cluster_config = get_argument(config, cluster_config_name)
    pv_argument_name = "persistentVolumeClaims"
    exception_message: str = "claimName and mountPath should be provided for persistentVolumeClaim"
    if cluster_config is not None and pv_argument_name in cluster_config:
        pv_arguments = cluster_config.get(pv_argument_name)
        claim_name = "claimName"
        mount_path = "mountPath"
        for pv_argument in pv_arguments:
            if pv_argument is None or claim_name not in pv_argument or mount_path not in pv_argument:
                raise ValueError(exception_message)
            claim_name_argument = pv_argument.get(claim_name)
            mount_path_argument = pv_argument.get(mount_path)
            if claim_name_argument is None or mount_path_argument is None:
                raise ValueError(exception_message)


def _validate_volume_arguments(config: DictConfig) -> None:
    """
    Check all the information needed for volume is provided if it is not None
    Parameters:
    config (DictConfig): Configuration dictionary
    """
    cluster_config_name = "cluster.cluster_config"
    cluster_config = get_argument(config, cluster_config_name)
    volumes_argument_name = "volumes"
    exception_message: str = "hostPath, mountPath, volumeName should be provided for volumes"
    if cluster_config is not None and volumes_argument_name in cluster_config:
        volume_arguments = cluster_config.get(volumes_argument_name)
        if volume_arguments is None:
            return
        host_path = "hostPath"
        mount_path = "mountPath"
        volume_name = "volumeName"
        print(volume_arguments)
        for volume_argument in volume_arguments:
            if (
                volume_argument is None
                or host_path not in volume_argument
                or mount_path not in volume_argument
                or volume_name not in volume_argument
            ):
                raise ValueError(exception_message)
            host_path_argument = volume_argument.get(host_path)
            mount_path_argument = volume_argument.get(mount_path)
            volume_name_argument = volume_argument.get(volume_name)
            if host_path_argument is None or mount_path_argument is None or volume_name_argument is None:
                raise ValueError(exception_message)


def _validate_pull_policy_argument(config: DictConfig) -> None:
    """
    Check only valid pullPolicy is provided if it is not None

    Parameters:
    config (DictConfig): Configuration dictionary
    """
    pull_policy_argument_name = "cluster.cluster_config.pullPolicy"
    pull_policy_argument = get_argument(config, pull_policy_argument_name)
    if pull_policy_argument is not None:
        supported_pull_policies = ["Always", "IfNotPresent", "Never"]
        if pull_policy_argument not in supported_pull_policies:
            raise ValueError("Provided pullPolicy is not supported")


def _validate_restart_policy_argument(config: DictConfig) -> None:
    """
    Check only valid restartPolicy is provided if it is not None

    Parameters:
    config (DictConfig): Configuration dictionary
    """
    restart_policy_argument_name = "cluster.cluster_config.restartPolicy"
    restart_policy_argument = get_argument(config, restart_policy_argument_name)
    if restart_policy_argument is not None:
        supported_restart_policies = ["Always", "OnFailure", "Never", "ExitCode"]
        if restart_policy_argument not in supported_restart_policies:
            raise ValueError("Provided restartPolicy is not supported")


def _validate_cluster_type_argument(config: DictConfig) -> None:
    """
    Check only valid cluster_type is provided

    Parameters:
    config (DictConfig): Configuration dictionary
    """
    cluster_type_argument_name = "cluster.cluster_type"
    cluster_type = get_argument(config, cluster_type_argument_name)
    supported_cluster_types = ["slurm", "k8s", "sm_jobs"]
    if cluster_type is not None and cluster_type not in supported_cluster_types:
        raise ValueError("Provided cluster_type is not supported")


def _validate_namespace_argument(config: DictConfig) -> None:
    """
    Check only valid kubectl namespace is provided
    Naming Convention of Kubernetes Namespaces is
    You can create a name with a maximum length of 253 characters using only alphanumeric characters and hyphens.
    Names cannot start with a hyphen and the alpha characters can only be lowercase.

    Parameters:
    config (DictConfig): Configuration dictionary
    """

    """
    Here's a breakdown of the regex pattern:

    ^ - Asserts the position at the start of the string.
    (?!-) - A negative lookahead assertion to ensure the string does not start with a hyphen.
    [a-z0-9-] - Matches any lowercase letter, digit, or hyphen.
    {1,253} - Specifies that the string must be between 1 and 253 characters long.
    $ - Asserts the position at the end of the string.
    This pattern will ensure the string is constructed using only lowercase alphanumeric characters and hyphens,
    is no longer than 253 characters, and does not start with a hyphen.
    """
    namespace_regex = r"^(?!-)[a-z0-9-]{1,253}$"
    namespace_argument_name = "cluster.cluster_config.namespace"
    namespace_argument = get_argument(config, namespace_argument_name)
    if namespace_argument is not None and not re.match(namespace_regex, namespace_argument):
        raise ValueError(
            "Provided namespace is not valid, Kindly provide Kubernetes Namespace "
            "with a maximum length of 253 characters using only alphanumeric characters and hyphens. "
            "Names cannot start with a hyphen and the alpha characters can only be lowercase."
        )


def _validate_all_mandatory_argument(config: DictConfig) -> None:
    mandatory_arguments_for_all_workflows = ["base_results_dir"]
    for argument_name in mandatory_arguments_for_all_workflows:
        argument = get_argument(config, argument_name)
        _validate_mandatory_argument(argument, argument_name)


def _validate_k8_custom_script_workflow_mandatory_argument(config: DictConfig) -> None:
    cluster_type = get_argument(config, "cluster.cluster_type")
    training_cfg = get_argument(config, "training_cfg")
    if cluster_type == "k8s" and training_cfg is not None:
        k8_custom_script_mandatory_arguments = [
            "container",
            "env_vars",
            "training_cfg",
            "training_cfg.entry_script",
            "training_cfg.run",
            "training_cfg.run.name",
            "training_cfg.run.nodes",
            "cluster",
            "cluster.cluster_type",
            "cluster.instance_type",
            "cluster.cluster_config",
        ]
        for argument_name in k8_custom_script_mandatory_arguments:
            argument = get_argument(config, argument_name)
            _validate_mandatory_argument(argument, argument_name)


def _validate_git_url(config: DictConfig) -> None:
    repo_url_or_path = get_argument(config, "git.repo_url_or_path")
    if repo_url_or_path is not None:
        if repo_url_or_path.startswith("git@"):
            raise ValueError("Currently we do not support to clone repo use ssh, please use http with token instead")
