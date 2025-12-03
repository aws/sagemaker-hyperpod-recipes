import json
import logging
import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import botocore.exceptions
from omegaconf import OmegaConf

from launcher.evaluation.launchers import SMEvaluationK8SLauncher
from main import main

logger = logging.getLogger(__name__)

import pytest

from tests.test_utils import (
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
)

eval_run_name = "test-eval-job"


@pytest.fixture(autouse=True)
def mock_aws_account_id():
    with patch("launcher.evaluation.launchers.boto3.client") as mock_boto_client:
        mock_boto_client.return_value.get_caller_identity.return_value = {"Account": "123456789012"}
        yield


@pytest.fixture(autouse=True)
def mock_aws_region():
    session_mock = MagicMock()
    session_mock.region_name = "us-east-1"

    with patch("launcher.evaluation.launchers.boto3.session.Session", return_value=session_mock):
        yield


@contextmanager
def mock_aws_account_id_invalid():
    with patch("launcher.evaluation.launchers.boto3.client") as mock_boto_client:
        mock_boto_client.return_value.get_caller_identity.side_effect = botocore.exceptions.NoCredentialsError()
        yield


def test_deterministic_eval_recipe_k8s_workflow_with_launch_json():
    logger.info("Testing deterministic evaluation recipe k8s workflow with launch json")

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
        "cluster_type=k8s",
        "launch_json=true",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    main(sample_recipe_k8s_config)

    # Verify launch.json content
    launch_json_path = f"{artifacts_dir}/{eval_run_name}/launch.json"
    with open(launch_json_path, "r") as f:
        launch_data = json.load(f)

    # Check K8S-specific files
    assert "evaluation.yaml" in launch_data
    assert "evaluation-config.yaml" in launch_data
    assert "metadata" in launch_data
    assert "recipe_override_parameters" in launch_data


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_deterministic_eval_recipe_smtj_workflow_with_launch_json(mock_load_hosting):
    logger.info("Testing deterministic evaluation recipe SMTJ workflow with launch json")

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

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    main(sample_recipe_k8s_config)

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


def test_llmaj_eval_recipe_k8s_workflow():
    logger.info("Testing LLM-as-a-Judge evaluation recipe k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=evaluation/open-source/open_source_llmaj_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.judge_model_id=openai.gpt-oss-120b-1:0",
        "recipes.run.inference_data_s3_path=s3://test-bucket/data",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=k8s",
        "launch_json=true",
    ]

    config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    main(config)

    launch_json_path = f"{artifacts_dir}/{eval_run_name}/launch.json"
    assert os.path.exists(launch_json_path)


def test_unified_launcher_cluster_type_detection():
    """Test that unified launcher correctly detects cluster type"""
    logger.info("Testing unified launcher cluster type detection")

    artifacts_dir = create_temp_directory()

    # Test K8S detection
    k8s_overrides = [
        "recipes=evaluation/open-source/open_source_deterministic_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.data_s3_path=s3://test-bucket/data",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=k8s",
    ]

    k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", k8s_overrides)
    k8s_launcher = SMEvaluationK8SLauncher(k8s_config)
    assert k8s_config.cluster_type == "k8s"

    # Test SMTJ detection via main launcher selection
    from launcher.evaluation.sm_jobs_launcher import SMEvaluationJobsLauncher

    smtj_overrides = [
        "recipes=evaluation/open-source/open_source_deterministic_eval",
        f"recipes.run.name={eval_run_name}",
        "recipes.run.data_s3_path=s3://test-bucket/data",
        "recipes.output.eval_results_dir=s3://test-bucket/results",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster_type=sm_jobs",
        "+sm_jobs_config.output_path=s3://test-bucket/results",
    ]

    smtj_config = make_hydra_cfg_instance("../recipes_collection", "config", smtj_overrides)
    smtj_launcher = SMEvaluationJobsLauncher(smtj_config)
    assert smtj_config.cluster_type == "sm_jobs"
    assert smtj_launcher is not None
