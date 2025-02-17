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

import difflib
import logging
import os
import shutil
import stat
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from launcher.nemo.constants import ROOT_DIR

logger = logging.getLogger(__name__)

GPUS_PER_NODE = 8


def create_temp_directory():
    """Create a temporary directory and Set full permissions for the directory"""
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return temp_dir


def replace_placeholder(file_path, placeholder, replacement):
    """Replace occurrences of placeholder in a file with the given replacement."""
    with open(file_path, "r") as file:
        content = file.read()

    content = content.replace(placeholder, replacement)

    with open(file_path, "w") as file:
        file.write(content)


def compare_artifacts(artifacts_paths, artifacts_dir, baseline_artifacts_path):
    for artifact_path in artifacts_paths:
        current_dir = os.getcwd()
        actual_artifact_path = artifacts_dir + artifact_path
        baseline_artifact_folder = current_dir + baseline_artifacts_path

        # Make a copy of baseline artifacts to replace placeholders
        baseline_artifact_copy_folder = create_temp_directory()
        shutil.copytree(baseline_artifact_folder, baseline_artifact_copy_folder, dirs_exist_ok=True)
        baseline_artifact_path = baseline_artifact_copy_folder + artifact_path

        results_dir_placeholder = "{$results_dir}"
        replace_placeholder(baseline_artifact_path, results_dir_placeholder, artifacts_dir)
        workspace_dir_placeholder = "{$workspace_dir}"
        replace_placeholder(baseline_artifact_path, workspace_dir_placeholder, current_dir)

        comparison_result = compare_files(baseline_artifact_path, actual_artifact_path)
        if comparison_result is False:
            assert False, baseline_artifact_path + " does not match " + actual_artifact_path


def compare_files(file1_path, file2_path):
    """Compare two files character by character."""
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
        file1_content = file1.readlines()
        file2_content = file2.readlines()

    # Using difflib to compare files
    diff = list(difflib.unified_diff(file1_content, file2_content, fromfile=file1_path, tofile=file2_path))

    if diff:
        diff_block = "\n" + "\n".join(line.strip() for line in diff)
        logger.info(f"Files differ:{diff_block}")
        return False

    logger.info("Files are identical.")
    return True


def compose_hydra_cfg(config_path: str, config_name: str, overrides: List[Any] = []) -> DictConfig:
    """Init and compose a hydra config"""
    with initialize(version_base=None, config_path=config_path):
        return compose(config_name=config_name, overrides=overrides, return_hydra_config=True)


def make_hydra_cfg_instance(config_path: str, config_name: str, overrides=[]) -> DictConfig:
    """Init hydra instance"""
    # Note: This is needed if using compose API and not hydra.main b/c we rely on hydra resolver
    # Open issue tracking fix https://github.com/facebookresearch/hydra/issues/2017
    config = compose_hydra_cfg(config_path, config_name, overrides)
    HydraConfig.instance().set_config(config)
    return config


@lru_cache(maxsize=None)
def get_launcher_run_script_paths() -> List[Path]:
    scripts_dir = ROOT_DIR / "launcher_scripts"
    launch_script_paths = []

    for path in scripts_dir.rglob("*.sh"):
        if not path.is_file():
            continue

        if "run" in path.name:
            launch_script_paths.append(path)

    return launch_script_paths


# adapted from
# https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo/blob/83c6037ad566ccb2584e7becab01da2ca8b6af20/src/hyperpod_nemo_adapter/conf/config_schemas.py#L78
def validate_distributed_degrees(
    shard_degree: Optional[int],
    tensor_model_parallel_degree: Optional[int],
    expert_model_parallel_degree: Optional[int],
    context_parallel_degree: Optional[int],
    num_nodes: Optional[int],
) -> bool:
    """
    Check that the degrees are legal.
    """
    # default param values to 1 if they are missing
    sd = shard_degree or 1
    tp = tensor_model_parallel_degree or 1
    ep = expert_model_parallel_degree or 1
    cp = context_parallel_degree or 1
    degree_mult = sd * tp * ep
    gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", GPUS_PER_NODE))
    world_size = (num_nodes or 1) * gpu_per_node

    # Validate the degree multiplication <= world_size
    degree_mult_is_valid = world_size % degree_mult == 0

    # Validate CP degree <= shard degree
    cp_is_valid = cp <= sd if cp > 1 else True

    return degree_mult_is_valid and cp_is_valid
