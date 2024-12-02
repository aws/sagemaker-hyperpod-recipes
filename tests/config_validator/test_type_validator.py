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

import pytest
from omegaconf import OmegaConf

from launcher.config_validator.type_validator import (
    TypeValidator,
    _check_types,
    _is_dict,
    _is_list_of_dicts,
    _is_list_of_paths,
    _is_list_of_strings,
    _is_positive_integer,
    _is_valid_path,
)

# Sample test configuration
sample_test_config = OmegaConf.create(
    {
        "hydra": {"output_subdir": None, "run": {"dir": "."}},
        "git": {"repo_url_or_path": "https://example.com/repo", "branch": "main", "commit": "abc123"},
        "training_cfg": {
            "entry_script": "/path/to/script.py",
            "script_args": [{"arg1": "value1"}, {"arg2": "value2"}],
            "run": {"name": "experiment", "nodes": 2, "ntasks_per_node": 4},
        },
        "cluster": {
            "cluster_type": "k8s",
            "instance_type": "p5.48xlarge",
            "cluster_config": {
                "namespace": "default",
                "custom_labels": {"env": "dev"},
                "annotations": {"annotation1": "value1"},
                "priority_class_name": "high",
                "label_selector": {"key": "value"},
                "persistentVolumeClaim": {"claimName": "claim1", "mountPath": "/mount/path"},
                "pullPolicy": "Always",
                "restartPolicy": "OnFailure",
            },
        },
        "base_results_dir": "/results",
        "container_mounts": ["/mnt/data", "/mnt/other"],
        "container": "my_container",
        "env_vars": {"VAR1": "value1"},
    }
)


@pytest.fixture
def sample_config():
    return sample_test_config


def test_type_validator_validation(sample_config):
    validator = TypeValidator(sample_config)
    try:
        validator.validate()
    except TypeError:
        pytest.fail("Type validation failed unexpectedly")


def test_is_valid_path():
    assert _is_valid_path("/valid/path") is True
    assert _is_valid_path(None) is False


def test_is_positive_integer():
    assert _is_positive_integer(1) is True
    assert _is_positive_integer("10") is True
    assert _is_positive_integer("0") is False
    assert _is_positive_integer(-1) is False
    assert _is_positive_integer("string") is False


def test_is_list_of_dicts():
    assert _is_list_of_dicts(OmegaConf.create([{"key": "value"}, {"key2": "value2"}])) is True
    assert _is_list_of_dicts(OmegaConf.create([{"key": "value"}, "string"])) is False
    assert _is_list_of_dicts(OmegaConf.create("string")) is False
    assert _is_list_of_dicts(OmegaConf.create(None)) is False


def test_is_list_of_strings():
    assert _is_list_of_strings(OmegaConf.create(["string1", "string2"])) is True
    assert _is_list_of_strings(OmegaConf.create(["string1", 2])) is False
    assert _is_list_of_strings(OmegaConf.create("string")) is False
    assert _is_list_of_strings(OmegaConf.create(None)) is False


def test_is_list_of_paths():
    assert _is_list_of_paths(OmegaConf.create(["/valid/path1", "/valid/path2"])) is True
    assert _is_list_of_paths(OmegaConf.create("string")) is False
    assert _is_list_of_paths(OmegaConf.create(None)) is False


def test_is_dict():
    assert _is_dict(OmegaConf.create({"key": "value"})) is True
    assert _is_dict(OmegaConf.create([{"key": "value"}])) is False
    assert _is_dict(OmegaConf.create(None)) is False
    assert _is_dict(None) is False


def test_check_types():
    with pytest.raises(TypeError):
        _check_types(0, "positive_integer", "test_integer")

    with pytest.raises(TypeError):
        _check_types(["string1", 2], "list_string", "test_list")

    with pytest.raises(TypeError):
        _check_types("string", "list_dict", "test_list")

    with pytest.raises(TypeError):
        _check_types("string", "list_path", "test_list")

    with pytest.raises(TypeError):
        _check_types({"key": "value"}, "string", "test_string")

    with pytest.raises(TypeError):
        _check_types({"key": "value"}, "path", "test_path")

    with pytest.raises(TypeError):
        _check_types(OmegaConf.create(None), "dict", "test_dict")
