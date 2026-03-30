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

import logging
import os
from unittest.mock import patch

import yaml
from omegaconf import OmegaConf

from hyperpod_recipes import list_recipes
from launcher.nemo.constants import ROOT_DIR
from main import main

from .test_utils import (
    get_launcher_run_script_paths,
    make_hydra_cfg_instance,
    validate_distributed_degrees,
)

logger = logging.getLogger(__name__)

RUN_SCRIPT_PATHS = get_launcher_run_script_paths()

RECIPES_DIR = ROOT_DIR / "recipes_collection/recipes"


def test_config_for_run_script_exists():
    log_line = lambda script, config: logger.info(
        f"\nlauncher file: {script.relative_to(ROOT_DIR)}" f"\nconfig file: {config.relative_to(ROOT_DIR)}" "\n"
    )

    for run_script_path in RUN_SCRIPT_PATHS:
        with open(run_script_path, "r") as fd:
            for line in fd:
                # this line defines the Yaml configuration file
                #  example: recipes=training/llama/hf_llama3_2_90b_seq8k_gpu_p5x32_pretrain
                if "recipes=" in line:
                    # clean up line
                    line = line.replace(" \\", "")  # remove shell line continuation marker
                    line = line.strip()

                    _, config_path_str = line.split("=")
                    config_path = RECIPES_DIR / (config_path_str + ".yaml")  # append .yaml

                    assert config_path.exists(), log_line(run_script_path, config_path)


def test_config_degree_validation():
    recipes_dir = ROOT_DIR / "recipes_collection/recipes"
    log_config_name = lambda name: logger.info(f"\nFailing Config File: {name}")

    for path in recipes_dir.rglob("*.yaml"):
        if not path.is_file():
            continue

        # Hydra requires relative path definition
        file_path: str = "../" + str(path.relative_to(ROOT_DIR).parent)
        config = make_hydra_cfg_instance(file_path, path.name)

        # plucking values outside the method arguments substantially reduces log output on failure
        shard_degree = OmegaConf.select(config, "model.shard_degree")
        tensor_model_parallel_degree = OmegaConf.select(config, "model.tensor_model_parallel_degree")
        expert_model_parallel_degree = OmegaConf.select(config, "model.expert_model_parallel_degree")
        context_parallel_degree = OmegaConf.select(config, "model.context_parallel_degree")
        num_nodes = OmegaConf.select(config, "trainer.num_nodes")

        assert validate_distributed_degrees(
            shard_degree, tensor_model_parallel_degree, expert_model_parallel_degree, context_parallel_degree, num_nodes
        ), log_config_name(path.name)


def test_dryrun_validation_for_all_recipes(subtests):
    recipes = list_recipes()

    for recipe in recipes:
        overrides = [
            f"recipes={recipe.name}",
            f"base_results_dir=dummy_dir/results",
            f"dry_run=True",
            "cluster_type=k8s",
            "cluster=k8s",
        ]
        if (
            recipe.name.startswith(("training/nova", "fine-tuning/nova", "evaluation/nova"))
            or "checkpointless" in recipe.name
        ):
            overrides.append("cluster_type=k8s")

        with subtests.test(msg=f"Dryrun for recipe {recipe.name}"):
            # with initialize(config_path="../recipes_collection", version_base="1.2"):
            #     cfg = compose(config_name="config", overrides=overrides)
            cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
            OmegaConf.set_struct(cfg, False)
            print("cfg", cfg)
            with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
                main(cfg)


def test_trainer_callbacks_in_openweights_finetuning_recipes(subtests):
    """
    Test that every LLMFT fine-tuning recipe has the required trainer_callbacks field
    with the MeteringCallback configuration.

    Expected structure:
    LLMFT:
    training_config:
      training_args:
        trainer_callbacks:
          - _target_: metering_callback.MeteringCallback
            output_path: /opt/ml/metering

    Note: This only applies to llmft recipes. Verl metering is not supported yet.
    """
    recipes_dir = RECIPES_DIR
    finetuning_dir = recipes_dir / "fine-tuning"

    # Find all fine-tuning recipe files
    recipe_files = list(finetuning_dir.rglob("*.yaml"))

    for recipe_path in recipe_files:
        if not recipe_path.is_file():
            continue

        recipe_rel_path = str(recipe_path.relative_to(recipes_dir))

        # Skip Nova recipes - they have different architecture
        if "nova" in recipe_rel_path.lower():
            continue

        # Skip Verl recipes - they have different architecture (no training_args)
        if "verl" in recipe_rel_path.lower():
            continue

        # Skip checkpointless recipes
        if "checkpointless" in recipe_rel_path.lower():
            continue

        with subtests.test(msg=f"trainer_callbacks in {recipe_rel_path}"):
            with open(recipe_path) as f:
                recipe_data = yaml.safe_load(f)

            # Navigate to trainer_callbacks
            assert "training_config" in recipe_data, f"Recipe {recipe_rel_path} missing 'training_config' section"
            assert (
                "training_args" in recipe_data["training_config"]
            ), f"Recipe {recipe_rel_path} missing 'training_config.training_args' section"
            assert (
                "trainer_callbacks" in recipe_data["training_config"]["training_args"]
            ), f"Recipe {recipe_rel_path} missing 'training_config.training_args.trainer_callbacks' field"

            trainer_callbacks = recipe_data["training_config"]["training_args"]["trainer_callbacks"]

            # Should be a list with at least one callback
            assert isinstance(trainer_callbacks, list), f"Recipe {recipe_rel_path} trainer_callbacks should be a list"
            assert (
                len(trainer_callbacks) >= 1
            ), f"Recipe {recipe_rel_path} trainer_callbacks should have at least one entry"

            # Check that MeteringCallback is present with correct configuration
            metering_callback_found = False
            for callback in trainer_callbacks:
                if callback.get("_target_") == "metering_callback.MeteringCallback":
                    metering_callback_found = True
                    assert (
                        callback.get("output_path") == "/opt/ml/metering"
                    ), f"Recipe {recipe_rel_path} MeteringCallback should have output_path='/opt/ml/metering'"
                    break

            assert metering_callback_found, f"Recipe {recipe_rel_path} missing MeteringCallback in trainer_callbacks"
