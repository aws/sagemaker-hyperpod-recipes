import logging

import pytest

from launcher.nemo.recipe_stages import SMTrainingHPCTRecipe
from main import get_training_stage, main

logger = logging.getLogger(__name__)

from tests.test_utils import create_temp_directory, make_hydra_cfg_instance


def test_checkpointless_k8s_workflow_fine_tuning():
    """Test checkpointless fine-tuning recipe execution with K8s cluster"""
    logger.info("Testing checkpointless fine-tuning k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/llama/checkpointless_llama3_70b_lora",
        "instance_type=p5.48xlarge",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "dry_run=True",
    ]

    config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    stage_class = get_training_stage(config)
    assert stage_class == SMTrainingHPCTRecipe
    main(config)


def test_checkpointless_k8s_workflow_pretrain():
    """Test checkpointless pretrain recipe execution with K8s cluster"""
    logger.info("Testing checkpointless pretrain k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=training/llama/checkpointless_llama3_70b_pretrain",
        "instance_type=p5.48xlarge",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "dry_run=True",
    ]

    config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    stage_class = get_training_stage(config)
    assert stage_class == SMTrainingHPCTRecipe
    main(config)


def test_checkpointless_requires_k8s_cluster():
    """Test that checkpointless recipes fail with non-k8s cluster types"""
    logger.info("Testing checkpointless cluster type validation")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/llama/checkpointless_llama3_70b_lora",
        "instance_type=p5.48xlarge",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "cluster=slurm",
        "cluster_type=slurm",
        "dry_run=True",
    ]

    config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    with pytest.raises(
        ValueError, match="HyperPod checkpointless training recipes only support K8s cluster type, got: bcm"
    ):
        main(config)
