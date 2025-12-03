import json
import logging
import os
from unittest.mock import patch

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

from tests.test_utils import (
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
)

eval_run_name = "test-eval-job"


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_evaluation_sm_jobs_workflow_with_launch_json(mock_load_hosting):
    logger.info("Testing evaluation SM jobs workflow with launch json")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=evaluation/open-source/open_source_deterministic_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.model_name_or_path=s3://test-bucket/model",
        "recipes.run.base_model_name=meta-textgeneration-llama-3-1-8b-instruct",
        "recipes.run.data_s3_path=s3://test-bucket/data",
        "recipes.evaluation.task=mmlu",
        "recipes.evaluation.strategy=zs_cot",
        "recipes.evaluation.metric=accuracy",
        "recipes.evaluation.subtask=abstract_algebra",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=sm_jobs",
        "+sm_jobs_config.output_path=s3://test-bucket/results",
        "+sm_jobs_config.inputs.s3.data=s3://test-bucket/data",
        "+sm_jobs_config.wait=true",
        "launch_json=true",
    ]

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_config\n")
    logger.info(OmegaConf.to_yaml(sample_config))

    main(sample_config)

    # Verify launch.json exists and has SMTJ content
    launch_json_path = f"{artifacts_dir}/{eval_run_name}/launch.json"
    assert os.path.exists(launch_json_path)

    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    # Check SMTJ-specific files
    assert "sm_jobs_launcher.py" in launch_data
    assert "sm_jobs_config.yaml" in launch_data
    assert "training_recipe.yaml" in launch_data
    assert "metadata" in launch_data
    assert "HostingConfigs" in launch_data["metadata"]
    assert "recipe_override_parameters" in launch_data

    # Verify SMTJ launcher script content
    launcher_script = launch_data["sm_jobs_launcher.py"]
    assert "import sagemaker" in launcher_script
    assert "PyTorch" in launcher_script
    assert "estimator.fit" in launcher_script


def test_evaluation_sm_jobs_config_generation():
    logger.info("Testing evaluation SM jobs config generation")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=evaluation/open-source/open_source_deterministic_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.model_name_or_path=s3://test-bucket/model",
        "recipes.run.base_model_name=meta-textgeneration-llama-3-1-8b-instruct",
        "recipes.run.data_s3_path=s3://test-bucket/data",
        "recipes.evaluation.task=mmlu",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=sm_jobs",
        "+sm_jobs_config.output_path=s3://test-bucket/results",
        "+sm_jobs_config.inputs.s3.data=s3://test-bucket/data",
        "+sm_jobs_config.wait=false",
        "+sm_jobs_config.additional_estimator_kwargs.max_run=7200",
        "launch_json=true",
    ]

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    main(sample_config)

    # Verify launch.json contains proper SMTJ config
    launch_json_path = f"{artifacts_dir}/{eval_run_name}/launch.json"
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    # Check SMTJ config content
    config_yaml = launch_data["sm_jobs_config.yaml"]
    assert "output_path:" in config_yaml
    assert "inputs:" in config_yaml
    assert "wait:" in config_yaml
    assert "additional_estimator_kwargs:" in config_yaml


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_evaluation_sm_jobs_metadata_generation(mock_load_hosting):
    logger.info("Testing evaluation SM jobs metadata generation")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=evaluation/open-source/open_source_deterministic_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.model_name_or_path=s3://test-bucket/model",
        "recipes.run.base_model_name=meta-textgeneration-llama-3-1-8b-instruct",
        "recipes.run.data_s3_path=s3://test-bucket/data",
        "recipes.evaluation.task=mmlu",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=sm_jobs",
        "+sm_jobs_config.output_path=s3://test-bucket/results",
        "+sm_jobs_config.inputs.s3.data=s3://test-bucket/data",
        "+sm_jobs_config.wait=true",
        "launch_json=true",
    ]

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    main(sample_config)

    # Verify metadata structure
    launch_json_path = f"{artifacts_dir}/{eval_run_name}/launch.json"
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    # Check metadata content
    assert "metadata" in launch_data
    metadata = launch_data["metadata"]

    # Verify required metadata fields
    assert "Name" in metadata
    assert "Type" in metadata
    assert metadata["Type"] == "Evaluation"
    assert "InstanceTypes" in metadata
    assert "Hardware" in metadata

    # Check recipe override parameters
    assert "recipe_override_parameters" in launch_data
    params = launch_data["recipe_override_parameters"]

    # Verify common evaluation parameters exist
    common_params = ["name", "base_model_name", "model_name_or_path", "data_s3_path"]
    for param in common_params:
        assert param in params
