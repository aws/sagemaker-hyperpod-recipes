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

"""Hosting-config constraints for the Gemma 4 E4B fine-tuning recipes.

An 8B-parameter model that the base JumpStart spec serves on one GPU must not
publish an 8-GPU instance as its only or default hosting option. These tests
pin the Default config to a single-GPU instance, keep every config
self-consistent, and run the resolved recipes through the verl launch.json
generator against a literal expectation of the published instance order.
"""

import json

import pytest

from launcher.nemo.constants import ROOT_DIR
from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)

INFERENCE_CONFIGS_DIR = ROOT_DIR / "utils" / "inference_configs"
RECIPES_DIR = ROOT_DIR / "recipes_collection" / "recipes" / "fine-tuning" / "gemma4-0_7_0"
BASE_OVERRIDE_FILE = ROOT_DIR / "launcher" / "recipe_templatization" / "base_override_parameters.json"
E4B_RECIPES = sorted(p.stem for p in RECIPES_DIR.glob("verl-*-gemma-4-e4b-it-*.yaml"))
GPUS_PER_INSTANCE = {
    "ml.g6e.xlarge": 1,
    "ml.g6e.2xlarge": 1,
    "ml.g6e.4xlarge": 1,
    "ml.g6.12xlarge": 4,
    "ml.g6e.12xlarge": 4,
    "ml.g6e.48xlarge": 8,
    "ml.p5.48xlarge": 8,
    "ml.p5en.48xlarge": 8,
}
EXPECTED_INSTANCE_ORDER = [
    "ml.g6e.2xlarge",
    "ml.g6e.4xlarge",
    "ml.g6e.xlarge",
    "ml.g6e.12xlarge",
    "ml.g6.12xlarge",
    "ml.g6e.48xlarge",
    "ml.p5.48xlarge",
    "ml.p5en.48xlarge",
]
# Base-model minimum host memory per instance, from the published
# huggingface-vlm-gemma-4-e4b-it spec hosting_instance_type_variants.
BASE_SPEC_MIN_MEMORY_MB = {
    "ml.g6e.xlarge": 16384,
    "ml.g6e.2xlarge": 32768,
    "ml.g6e.4xlarge": 65536,
    "ml.g6.12xlarge": 98304,
    "ml.g6e.12xlarge": 196608,
}


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT_DIR)


def _load(recipe_stem):
    path = INFERENCE_CONFIGS_DIR / f"hosting-{recipe_stem}.json"
    with path.open() as fh:
        return json.load(fh)


def test_e4b_recipe_set_is_complete():
    assert len(E4B_RECIPES) == 8, E4B_RECIPES


@pytest.mark.parametrize("recipe_stem", E4B_RECIPES)
def test_default_config_is_a_single_gpu_instance(recipe_stem):
    configs = _load(recipe_stem)
    defaults = [c for c in configs if c.get("Profile") == "Default"]
    assert len(defaults) == 1, f"{recipe_stem}: expected one Default profile, got {len(defaults)}"
    default = defaults[0]
    assert default["InstanceType"] == "ml.g6e.2xlarge"
    assert default["ComputeResourceRequirements"]["NumberOfAcceleratorDevicesRequired"] == 1
    assert default["Environment"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert configs[0] is default, "the Default profile must be the first config so SDK fallbacks pick it"


@pytest.mark.parametrize("recipe_stem", E4B_RECIPES)
def test_single_gpu_configs_cover_the_base_model_instances(recipe_stem):
    instance_types = [c["InstanceType"] for c in _load(recipe_stem)]
    assert len(instance_types) == len(set(instance_types)), f"{recipe_stem}: duplicate instance types"
    for expected in BASE_SPEC_MIN_MEMORY_MB:
        assert expected in instance_types, f"{recipe_stem}: missing {expected}"


@pytest.mark.parametrize("recipe_stem", E4B_RECIPES)
def test_every_config_is_self_consistent(recipe_stem):
    is_lora = recipe_stem.endswith("-lora")
    for config in _load(recipe_stem):
        instance_type = config["InstanceType"]
        gpus = GPUS_PER_INSTANCE[instance_type]
        env = config["Environment"]
        requirements = config["ComputeResourceRequirements"]
        assert env["SM_VLLM_TENSOR_PARALLEL_SIZE"] == str(gpus), f"{recipe_stem} {instance_type}: TP != GPU count"
        assert requirements["NumberOfAcceleratorDevicesRequired"] == gpus, f"{recipe_stem} {instance_type}"
        assert env["SM_VLLM_TOOL_CALL_PARSER"] == "gemma4"
        assert env["SM_VLLM_TRUST_REMOTE_CODE"] == "true"
        assert config["EcrAddress"].endswith("/vllm:0.23.0-gpu-py312-cu130-ubuntu22.04-sagemaker")
        assert ("SM_VLLM_ENABLE_LORA" in env) == is_lora, f"{recipe_stem} {instance_type}: LoRA env mismatch"
        if instance_type in BASE_SPEC_MIN_MEMORY_MB:
            assert requirements["MinMemoryRequiredInMb"] == BASE_SPEC_MIN_MEMORY_MB[instance_type]


@pytest.mark.parametrize("recipe_stem", E4B_RECIPES)
def test_launch_json_metadata_carries_the_single_gpu_default(recipe_stem):
    with BASE_OVERRIDE_FILE.open() as fh:
        staging_cfg = json.load(fh)["fine_tuning"]
    recipe_rel = f"fine-tuning/gemma4-0_7_0/{recipe_stem}"
    processor = VerlRecipeTemplateProcessor(staging_cfg=staging_cfg, platform="sm_jobs")
    processor.process_recipe(recipe_file_path=recipe_rel)
    metadata = processor.get_recipe_metadata(recipe_rel)
    hosting = metadata["HostingConfigs"]
    assert [config["InstanceType"] for config in hosting] == EXPECTED_INSTANCE_ORDER
    assert [config.get("Profile") for config in hosting] == ["Default"] + [None] * (len(EXPECTED_INSTANCE_ORDER) - 1)
    default = hosting[0]
    assert default["ComputeResourceRequirements"] == {
        "MinMemoryRequiredInMb": 32768,
        "NumberOfCpuCoresRequired": 6,
        "NumberOfAcceleratorDevicesRequired": 1,
    }
    assert default["Environment"]["SM_VLLM_TENSOR_PARALLEL_SIZE"] == "1"
    assert default["EcrAddress"].endswith("/vllm:0.23.0-gpu-py312-cu130-ubuntu22.04-sagemaker")
