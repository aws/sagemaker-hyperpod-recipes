import logging
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

import pytest
from pydantic import ValidationError

from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
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


def test_hyperpod_pytorch_job_k8s_workflow():
    logger.info("Testing HyperPod PyTorch Job k8s workflow")

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

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    # Add HyperPod PyTorch Job flag to the cluster config
    OmegaConf.set_struct(sample_config, False)
    sample_config.cluster.use_hyperpod_pytorch_job = True

    logger.info("\nsample_hyperpod_pytorch_job_config\n")
    logger.info(OmegaConf.to_yaml(sample_config))

    main(sample_config)

    # Check that useHyperPodPytorchJob is True in the generated values.yaml
    import os

    import yaml

    for root, dirs, files in os.walk(artifacts_dir):
        if "values.yaml" in files and "k8s_template" in root:
            values_path = os.path.join(root, "values.yaml")
            with open(values_path, "r") as f:
                values = yaml.safe_load(f)

            assert values["trainingConfig"]["useHyperPodPytorchJob"] == True, "useHyperPodPytorchJob should be True"
            logger.info("✓ HyperPod PyTorch Job flag correctly set to True")
            return

    raise AssertionError("Could not find values.yaml file in generated artifacts")


@pytest.mark.xfail(reason="Broken by HF recipe removal, need to be fixed")
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
    base_overrides = [
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "recipes.run.model_type=llm_finetuning_aws",
    ]

    overrides = base_overrides + [
        "git.repo_url_or_path=/local/path",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    with pytest.raises(ValueError):
        main(sample_recipe_k8s_config)

    logger.info("Testing recipe k8s workflow fails validation with 0 nodes")

    validate_overrides = base_overrides + [
        "++recipes.training_config.training_args.max_epochs=0",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", validate_overrides)
    with pytest.raises(ValidationError):
        main(sample_recipe_k8s_config)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_recipe_k8s_workflow_with_launch_json(mock_load_hosting_config_patch):
    logger.info("Testing recipe k8s workflow with launch_json")

    artifacts_dir = create_temp_directory("test_recipe_k8s_workflow_with_launch_json")

    overrides = [
        "recipes=fine-tuning/llama/llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "git.use_default=false",
        "git.entry_script=/app/src/train_hp.py",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(sample_recipe_k8s_config, False)
    sample_recipe_k8s_config.cluster.persistent_volume_claims = [{"claimName": "fsx-claim", "mountPath": "/data"}]
    sample_recipe_k8s_config.launch_json = True
    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    del sample_recipe_k8s_config["hydra"]
    main(sample_recipe_k8s_config)

    artifacts_paths = [
        "/llama-3-2-1b-instruct/k8s_template/launch.json",
        "/llama-3-2-1b-instruct/k8s_template/values.yaml",
        "/llama-3-2-1b-instruct/k8s_template/Chart.yaml",
        "/llama-3-2-1b-instruct/k8s_template/templates/training.yaml",
        "/llama-3-2-1b-instruct/k8s_template/templates/training-config.yaml",
    ]

    k8s_baseline_artifacts_path = "/tests/k8s_workflow/k8s_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, k8s_baseline_artifacts_path)
