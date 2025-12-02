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
import json
import logging
import os
import re
import shutil
import stat
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import yaml
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from launcher.nemo.constants import ROOT_DIR

logger = logging.getLogger(__name__)

GPUS_PER_NODE = 8
REGEX_JOB_RUN_NAME = r"^[a-z0-9-]+$"
PROG_JOB_RUN_NAME = re.compile(REGEX_JOB_RUN_NAME)

BASELINE_CONFIG = {"recipes": {"exp_manager": {"exp_dir": None}}}

GOLDEN_WRITE = bool(os.getenv("GOLDEN_TEST_WRITE", False))


def is_helm_template(file_path: str) -> bool:
    # using values present in the first line of training.yaml (helm file) to heuristically differentiate helm from yaml
    content = Path(file_path).read_text(encoding="utf-8")
    return "{{" in content or "}}" in content or ".Values" in content


def create_temp_directory(dir_name=None):
    """Create a temporary directory and Set full permissions for the directory"""
    if dir_name is not None:
        base_tmp = tempfile.gettempdir()
        temp_dir = os.path.join(base_tmp, dir_name)
        os.makedirs(temp_dir, exist_ok=True)
    else:
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
        original_baseline_artifact_path = baseline_artifact_folder + artifact_path
        baseline_artifact_path = baseline_artifact_copy_folder + artifact_path

        results_dir_placeholder = "{$results_dir}"
        replace_placeholder(baseline_artifact_path, results_dir_placeholder, artifacts_dir)

        # Also replace hardcoded /tmp path that might exist in baseline
        replace_placeholder(baseline_artifact_path, "/tmp/test_sm_jobs_workflow_with_launch_json", artifacts_dir)
        replace_placeholder(baseline_artifact_path, "/tmp/test_recipe_k8s_workflow_with_launch_json", artifacts_dir)

        workspace_dir_placeholder = "{$workspace_dir}"
        replace_placeholder(baseline_artifact_path, workspace_dir_placeholder, current_dir)

        # if GOLDEN_WRITE:
        comparison_result = compare_files(baseline_artifact_path, actual_artifact_path)
        if comparison_result is False:
            if GOLDEN_WRITE:
                shutil.copyfile(actual_artifact_path, original_baseline_artifact_path)
            else:
                assert (
                    False
                ), f"{baseline_artifact_path} does not match {actual_artifact_path}. Run with GOLDEN_TEST_WRITE=1 to update"


def compare_files(file1_path, file2_path):
    """Compare two files as YAML dictionaries if applicable."""
    helm_files = is_helm_template(file1_path) or is_helm_template(file2_path)
    if file1_path.endswith((".yaml", ".yml")) and file2_path.endswith((".yaml", ".yml")) and not helm_files:
        with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
            config1 = OmegaConf.load(file1)
            config2 = OmegaConf.load(file2)

            # Create a default config that provides a value of None for missing keys.
            defaults = OmegaConf.create(BASELINE_CONFIG)

            # Merge defaults into configs.
            config1 = OmegaConf.merge(defaults, config1)
            config2 = OmegaConf.merge(defaults, config2)

            # Resolve interpolations in both configs
            OmegaConf.resolve(config1)
            OmegaConf.resolve(config2)

            # Convert the resolved configs to YAML
            yaml1 = OmegaConf.to_yaml(config1)
            yaml2 = OmegaConf.to_yaml(config2)

        if yaml1 == yaml2:
            logger.info("YAML files are identical.")
            return True
        else:
            diff = list(
                difflib.unified_diff(
                    yaml.dump(yaml1, sort_keys=True).splitlines(),
                    yaml.dump(yaml2, sort_keys=True).splitlines(),
                    fromfile=file1_path,
                    tofile=file2_path,
                )
            )
            diff_block = "\n" + "\n".join(line.strip() for line in diff)
            logger.info(f"YAML files differ:{diff_block}")
            return False

    """Compare two files character by character."""
    print(file1_path, file2_path)
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


def is_job_run_name_valid_for_clusters(run_name: Optional[str]) -> bool:
    """
    Ensure the name is valid for Slurm and Kubernetes clusters.
    Kubernetes clusters accept only lower-case letters, numbers and hyphens.
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-label-names
    """
    # Example: "hf-llama3-70b-lora"
    if run_name is None:
        return False

    return bool(PROG_JOB_RUN_NAME.match(run_name))


def mock_load_hosting_config(recipe_name: str):
    """Mock function that validates the recipe name and returns test hosting config"""
    # Extract the actual recipe name from full recipe paths
    # Handle patterns like "fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft" -> "nova_lite_1_0_p5_p4d_gpu_sft"
    if "/" in recipe_name:
        actual_recipe_name = recipe_name.split("/")[-1]
    else:
        actual_recipe_name = recipe_name

    # List of valid recipe names that tests should use (extracted from full paths)
    valid_recipe_names = [
        # Evaluation recipes
        "open_source_deterministic_eval",
        # Nova recipes (from full paths like "fine-tuning/nova/...")
        "nova_lite_1_0_p5_p4d_gpu_sft",
        "nova_micro_1_0_p5_p4d_gpu_lora_dpo",
        "nova_pro_1_0_p5_gpu_ppo",
        "nova_micro_1_0_p5_48xl_gpu_general_text_benchmark_eval",
        # VERL recipes (from full paths like "fine-tuning/verl/...")
        "verl-ppo-qwen-2-dot-5-0-dot-5b",
        "verl-grpo-rlaif-llama-3-dot-2-1b-instruct-lora",
        "verl-grpo-rlvr-llama-3-dot-1-8b-instruct-lora",
        # LLMFT recipes
        "llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
    ]

    # Validate that we're being called with a valid recipe name
    assert (
        actual_recipe_name in valid_recipe_names
    ), f"Expected one of {valid_recipe_names}, got '{actual_recipe_name}' (from full recipe '{recipe_name}')"

    # Load and return the test hosting config
    test_config_path = ROOT_DIR / "utils" / "inference_configs" / "hosting-test.json"
    with open(test_config_path, "r") as f:
        return json.load(f)
