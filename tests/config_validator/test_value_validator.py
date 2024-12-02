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

from launcher.config_validator.value_validator import ValueValidator

# Sample valid configurations
VALID_CONFIGS = [
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_type": "k8s",
                "instance_type": "p5.48xlarge",
                "cluster_config": {
                    "namespace": "valid-namespace",
                    "pullPolicy": "Always",
                    "restartPolicy": "OnFailure",
                    "persistentVolumeClaims": [{"claimName": "my-claim", "mountPath": "/mount/path"}],
                },
            },
            "training_cfg": {
                "entry_script": "/path/to/entry_script.py",
                "run": {"name": "run_name", "nodes": 2},
                "script_args": [{"arg1": "value1"}],
            },
            "container": "my_container",
            "env_vars": {"VAR1": "value1"},
        }
    ),
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_type": "slurm",
            },
        },
    ),
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_type": "sm_jobs",
            },
        },
    ),
]

# Sample invalid configurations
INVALID_CONFIGS = [
    # Missing mandatory argument base_results_dir
    OmegaConf.create({"cluster": {"cluster_type": "k8s", "cluster_config": {"namespace": "valid-namespace"}}}),
    # Invalid pull policy
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_type": "k8s",
                "cluster_config": {"namespace": "valid-namespace", "pullPolicy": "InvalidPolicy"},
            },
        }
    ),
    # Invalid restart policy
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_type": "k8s",
                "cluster_config": {"namespace": "valid-namespace", "restartPolicy": "InvalidPolicy"},
            },
        }
    ),
    # Invalid cluster type
    OmegaConf.create({"base_results_dir": "/some/path", "cluster": {"cluster_type": "invalid_type"}}),
    # Invalid namespace
    OmegaConf.create(
        {"base_results_dir": "/some/path", "cluster": {"cluster_config": {"namespace": "-invalidnamespace"}}}
    ),
    # Missing persistentVolumeClaim arguments
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {"cluster_config": {"namespace": "valid-namespace", "persistentVolumeClaims": [{}]}},
        }
    ),
    # persistentVolumeClaim set to None
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_config": {
                    "namespace": "valid-namespace",
                    "persistentVolumeClaims": [{"claimName": None, "mountPath": None}],
                }
            },
        }
    ),
    # Missing volume arguments
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {"cluster_config": {"namespace": "valid-namespace", "volumes": [{}]}},
        }
    ),
    # volume arguments set to None
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "cluster": {
                "cluster_config": {
                    "namespace": "valid-namespace",
                    "volumes": [{"hostPath": None, "mountPath": None, "volumeName": None}],
                }
            },
        }
    ),
    # Do not support git clone with ssh
    OmegaConf.create(
        {
            "base_results_dir": "/some/path",
            "git": {"repo_url_or_path": "git@some_repo"},
        }
    ),
]


@pytest.mark.parametrize("config", VALID_CONFIGS)
def test_validate_value_validator_valid_config(config):
    validator = ValueValidator(config)
    try:
        validator.validate()
    except Exception as e:
        pytest.fail(f"Validator raised an exception with valid config: {str(e)}")


@pytest.mark.parametrize("config", INVALID_CONFIGS)
def test_validate_value_validator_invalid_config(config):
    validator = ValueValidator(config)
    with pytest.raises(ValueError):
        validator.validate()
