import logging

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

import pytest

from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)


def compare_recipe_k8s_artifacts(artifacts_dir):
    logger.info("Comparing recipe k8s artifacts")

    artifacts_paths = [
        "/llama-8b/llama-8b_submission.sh",
        # "/llama-8b/llama-8b_hydra.yaml", # Do not test recipe, this changes often
        "/llama-8b/k8s_template/values.yaml",
        "/llama-8b/k8s_template/Chart.yaml",
        # "/llama-8b/k8s_template/config/llama-8b_hydra.yaml", # Do not test recipe, this changes often
        "/llama-8b/k8s_template/templates/training.yaml",
        "/llama-8b/k8s_template/templates/training-config.yaml",
    ]

    k8s_baseline_artifacts_path = "/tests/k8s_workflow/k8s_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, k8s_baseline_artifacts_path)


def test_recipe_k8s_workflow():
    logger.info("Testing recipe k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "git.repo_url_or_path=https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo.git",
        "git.branch=test_branch",
        "git.commit=test_commit",
        "git.token=test_token",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    main(sample_recipe_k8s_config)

    compare_recipe_k8s_artifacts(artifacts_dir)


def test_recipe_k8s_workflow_invalid():
    logger.info("Testing recipe k8s workflow with invalid git config")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "git.repo_url_or_path=/local/path",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    with pytest.raises(ValueError):
        main(sample_recipe_k8s_config)
