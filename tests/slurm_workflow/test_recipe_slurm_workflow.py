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

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)


from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)


def compare_recipe_slurm_artifacts(artifacts_dir):
    logger.info("Comparing recipe slurm artifacts")

    artifacts_paths = [
        "/llama-8b/launch_docker_container.sh",
        # "/llama-8b/llama-8b_hydra.yaml", # Do not test the recipe, this is changing often
        "/llama-8b/sagemaker-llama-8b_submission.sh",
        "/llama-8b/train_script.sh",
        "/llama-8b/docker_exec_script.sh",
    ]

    slurm_baseline_artifacts_path = "/tests/slurm_workflow/slurm_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, slurm_baseline_artifacts_path)


def compare_recipe_slurm_artifacts_trn(artifacts_dir):
    logger.info("Comparing recipe slurm artifacts")

    artifacts_paths = [
        "/hf-llama3-8b/launch_docker_container.sh",
        # "/llama-8b/llama-8b_hydra.yaml", # Do not test the recipe, this is changing often
        "/hf-llama3-8b/sagemaker-hf-llama3-8b_submission.sh",
        # "/llama-7b/train_script.sh",
    ]

    slurm_baseline_artifacts_path = "/tests/slurm_workflow/slurm_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, slurm_baseline_artifacts_path)


def test_recipe_slurm_workflow():
    logger.info("Testing recipe slurm workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=ml.p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_account.dkr.ecr.test_region.amazonaws.com/test_repo:test_tag",
        "cluster.slurm_create_submission_file_only=True",
        "cluster.slurm_docker_cfg.docker_args=[test_docker_cmd]",
        "cluster.slurm_docker_cfg.post_launch_commands=[test_post_launch_cmd]",
    ]

    sample_recipe_slurm_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_slurm_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_slurm_config))

    main(sample_recipe_slurm_config)

    compare_recipe_slurm_artifacts(artifacts_dir)


def test_recipe_slurm_trn_workflow():
    logger.info("Testing recipe slurm workflow for trn")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=trn1.32xlarge",
        "recipes=training/llama/hf_llama3_8b_seq8k_trn1x4_pretrain.yaml",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_account.dkr.ecr.test_region.amazonaws.com/test_repo:test_tag",
        "cluster.slurm_create_submission_file_only=True",
        "cluster.slurm_docker_cfg.docker_args=[test_docker_cmd]",
        "cluster.slurm_docker_cfg.post_launch_commands=[test_post_launch_cmd]",
        "recipes.run.name=hf-llama3-8b",
        "recipes.trainer.num_nodes=4",
        "recipes.data.train_dir=/fake_dataset",
        "recipes.model.model_config=/fake_dataset/config.json",
    ]

    sample_recipe_slurm_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_slurm_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_slurm_config))

    main(sample_recipe_slurm_config)

    compare_recipe_slurm_artifacts_trn(artifacts_dir)
