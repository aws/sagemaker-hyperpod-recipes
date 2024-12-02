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
from typing import Any

from omegaconf import DictConfig, OmegaConf

from launcher.config_validator.value_validator import get_argument


class TypeValidator:
    # Define the list of key-type pairs
    types = [
        ("hydra.output_subdir", "path"),
        ("hydra.run.dir", "path"),
        ("git.repo_url_or_path", "string"),
        ("git.branch", "string"),
        ("git.commit", "string"),
        ("training_cfg.entry_script", "path"),
        ("training_cfg.script_args", "list_dict"),
        ("training_cfg.run.name", "string"),
        ("training_cfg.run.nodes", "positive_integer"),
        ("training_cfg.run.ntasks_per_node", "positive_integer"),
        ("cluster.cluster_type", "string"),
        ("cluster.instance_type", "string"),
        ("cluster.cluster_config", "dict"),
        ("cluster.cluster_config.namespace", "string"),
        ("cluster.cluster_config.custom_labels", "dict"),
        ("cluster.cluster_config.annotations", "dict"),
        ("cluster.cluster_config.priority_class_name", "string"),
        ("cluster.cluster_config.label_selector", "dict"),
        ("cluster.cluster_config.persistentVolumeClaims", "list_dict"),
        ("cluster.cluster_config.volumes", "list_dict"),
        ("cluster.cluster_config.pullPolicy", "string"),
        ("cluster.cluster_config.restartPolicy", "string"),
        ("base_results_dir", "path"),
        ("container_mounts", "list_path"),
        ("container", "string"),
        ("env_vars", "dict"),
    ]

    def __init__(self, config: DictConfig):
        self.config = config

    def validate(self):
        for key, type in self.types:
            argument = get_argument(self.config, key)
            _check_types(argument, type, key)


def _is_valid_path(path) -> bool:
    """
    Check if the input string is a valid file path.

    Parameters:
    path (str): The path to validate.

    Returns:
    bool: True if the path is valid, False otherwise.
    """
    if not isinstance(path, str):
        return False

    try:
        normalized_path = os.path.normpath(path)
        return True
    except Exception as e:
        return False


def _is_positive_integer(argument) -> bool:
    try:
        val = int(argument)
        if val < 1:
            return False
    except ValueError:
        return False
    return True


def _get_base_omega_conf_container(argument) -> (Any, bool):
    try:
        argument = OmegaConf.to_container(argument, resolve=True)
    except Exception as e:
        return None, False
    return argument, True


def _is_list_of_dicts(argument) -> bool:
    argument, status = _get_base_omega_conf_container(argument)
    if status is False:
        return False
    return isinstance(argument, list) and all(item is None or isinstance(item, dict) for item in argument)


def _is_list_of_strings(argument) -> bool:
    argument, status = _get_base_omega_conf_container(argument)
    if status is False:
        return False
    return isinstance(argument, list) and all(item is None or isinstance(item, str) for item in argument)


def _is_list_of_paths(argument) -> bool:
    argument, status = _get_base_omega_conf_container(argument)
    if status is False:
        return False
    return isinstance(argument, list) and all(item is None or _is_valid_path(item) for item in argument)


def _is_dict(argument) -> bool:
    argument, status = _get_base_omega_conf_container(argument)
    if status is False:
        return False
    return isinstance(argument, dict)


def _check_types(argument, type, argument_name) -> None:
    if argument is None:
        return

    if type == "string" and not isinstance(argument, str):
        raise TypeError("{} with val {} is not a string".format(argument_name, argument))

    if type == "path" and not _is_valid_path(argument):
        raise TypeError("{} with val {} is not a valid path".format(argument_name, argument))

    if type == "list_string" and not _is_list_of_strings(argument):
        raise TypeError("{} with val {} is not a list of string".format(argument_name, argument))

    if type == "list_dict" and not _is_list_of_dicts(argument):
        raise TypeError("{} with val {} is not a list of dictionary".format(argument_name, argument))

    if type == "list_path" and not _is_list_of_paths(argument):
        raise TypeError("{} with val {} is not a list of paths".format(argument_name, argument))

    if type == "positive_integer" and not _is_positive_integer(argument):
        raise TypeError("{} with val {} is not a positive integer".format(argument_name, argument))

    if type == "dict" and not _is_dict(argument):
        raise TypeError("{} with val {} is not a dictionary".format(argument_name, argument))
