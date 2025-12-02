import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import botocore.exceptions
from omegaconf import OmegaConf

from launcher.nova.launchers import NovaK8SLauncher
from main import main

logger = logging.getLogger(__name__)

import pytest

from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
)

sft_run_name = "nova-lite-sft"
eval_run_name = "nova-lite-eval"


@pytest.fixture(autouse=True)
def mock_aws_account_id():
    with patch("launcher.nova.launchers.boto3.client") as mock_boto_client:
        mock_boto_client.return_value.get_caller_identity.return_value = {"Account": None}
        yield


@pytest.fixture(autouse=True)
def mock_aws_region():
    session_mock = MagicMock()
    session_mock.region_name = "us-east-1"

    with patch("launcher.nova.launchers.boto3.session.Session", return_value=session_mock):
        yield


@contextmanager
def mock_aws_account_id_invalid():
    with patch("launcher.nova.launchers.boto3.client") as mock_boto_client:
        mock_boto_client.return_value.get_caller_identity.side_effect = botocore.exceptions.NoCredentialsError()
        yield


def compare_sft_recipe_k8s_artifacts(artifacts_dir, run_name, launch_json=False):
    logger.info("Comparing sft recipe k8s artifacts")

    artifacts_paths = [
        f"/{run_name}/{run_name}_launch.sh",
        f"/{run_name}/k8s_templates/values.yaml",
        f"/{run_name}/k8s_templates/Chart.yaml",
        f"/{run_name}/k8s_templates/templates/training.yaml",
        f"/{run_name}/k8s_templates/templates/training-config.yaml",
    ]

    if launch_json:
        artifacts_paths.append(f"/{run_name}/launch.json")
        k8s_baseline_artifacts_path = "/tests/nova_k8s_workflow/k8s_baseline_artifacts/with_launch_json"
    else:
        k8s_baseline_artifacts_path = "/tests/nova_k8s_workflow/k8s_baseline_artifacts/without_launch_json"

    compare_artifacts(artifacts_paths, artifacts_dir, k8s_baseline_artifacts_path)


def test_sft_recipe_k8s_workflow():
    logger.info("Testing SFT recipe k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
        "instance_type=p5.48xlarge",
        f"recipes.run.name={sft_run_name}",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+cluster.service_account_name=placeholder_service_account_name",
        "+cluster.priority_class_name=test_pc_name",
        "+cluster.annotations.annotation_key_1=annotation-value-1",
        "+cluster.custom_labels.placeholder=custom_labels",
        "+cluster.label_selector.required.example_label_key=[expected-label-value-1, expected-label-value-2]",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    main(sample_recipe_k8s_config)

    compare_sft_recipe_k8s_artifacts(artifacts_dir, sft_run_name)


def test_sft_recipe_custom_instance_type():
    """Test SFT recipe with a custom instance type."""
    logger.info("Testing SFT recipe with custom instance type")

    artifacts_dir = create_temp_directory()
    custom_instance_type = "ml.g5.48xlarge"

    overrides = [
        f"instance_type={custom_instance_type}",
        "recipes=fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
        f"recipes.run.name={sft_run_name}_custom",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
    ]

    custom_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    logger.info("\nTesting with custom instance type\n")

    main(custom_config)

    values_yaml_path = f"{artifacts_dir}/{sft_run_name}_custom/k8s_templates/values.yaml"
    with open(values_yaml_path, "r") as file:
        values_content = file.read()

    assert f"- {custom_instance_type}" in values_content


def test_recipe_k8s_workflow_invalid():
    logger.info("Testing recipe k8s workflow with invalid git config")
    overrides = [
        "instance_type=p5.48xlarge",
        "recipes=fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
        f"recipes.run.name={sft_run_name}",
        "+cluster.persistent_volume_claims.0.claimName=fsx-claim",
        "+cluster.persistent_volume_claims.0.mountPath=data",
        "cluster=k8s",
        "cluster_type=k8s",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))
    with pytest.raises(ValueError):
        main(sample_recipe_k8s_config)


def test_recipe_env_vars():
    with mock_aws_account_id_invalid():
        logger.info("Testing SFT recipe k8s workflow")

        artifacts_dir = create_temp_directory()
        overrides = [
            "recipes=fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
            "instance_type=p5.48xlarge",
            f"recipes.run.name={sft_run_name}",
            "base_results_dir={}".format(artifacts_dir),
            "container=test_container",
            "cluster=k8s",
            "cluster_type=k8s",
            "+cluster.service_account_name=default",
            "+cluster.priority_class_name=test_pc_name",
            "+cluster.annotations.annotation_key_1=annotation-value-1",
            "+cluster.custom_labels.label_key_1=label-value-1",
            "+cluster.label_selector.required.example_label_key=[expected-label-value-1, expected-label-value-2]",
        ]

        sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

        base_class = NovaK8SLauncher(sample_recipe_k8s_config)
        result = base_class._get_env_vars()
        assert result == {"INSTANCE_TYPE": "ml.p5.48xlarge"}


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_sft_recipe_k8s_workflow_with_launch_json(mock_load_hosting):
    logger.info("Testing SFT recipe k8s workflow with launch json")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/nova/nova_1_0/nova_micro/DPO/nova_micro_1_0_p5_p4d_gpu_lora_dpo",
        "instance_type=ml.p5.48xlarge",
        f"recipes.run.name={sft_run_name}",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+cluster.service_account_name=placeholder_service_account_name",
        "+cluster.priority_class_name=test_pc_name",
        "+cluster.annotations.annotation_key_1=annotation-value-1",
        "+cluster.custom_labels.placeholder=custom_labels",
        "+cluster.label_selector.required.example_label_key=[expected-label-value-1, expected-label-value-2]",
        "launch_json=true",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    main(sample_recipe_k8s_config)

    compare_sft_recipe_k8s_artifacts(artifacts_dir, sft_run_name, launch_json=True)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_eval_recipe_k8s_workflow_with_launch_json(mock_load_hosting):
    logger.info("Testing Eval recipe k8s workflow with launch json")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=evaluation/nova/nova_1_0/nova_micro/nova_micro_1_0_p5_48xl_gpu_general_text_benchmark_eval",
        "instance_type=ml.p5.48xlarge",
        f"recipes.run.name={eval_run_name}",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+cluster.service_account_name=placeholder_service_account_name",
        "+cluster.priority_class_name=test_pc_name",
        "+cluster.annotations.annotation_key_1=annotation-value-1",
        "+cluster.custom_labels.placeholder=custom_labels",
        "+cluster.label_selector.required.example_label_key=[expected-label-value-1, expected-label-value-2]",
        "launch_json=true",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    main(sample_recipe_k8s_config)

    compare_sft_recipe_k8s_artifacts(artifacts_dir, eval_run_name, launch_json=True)


def test_recipe_k8s_workflow_validation():
    logger.info("Testing SFT recipe k8s workflow with invalid inputs")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
        "instance_type=p5.48xlarge",
        f"recipes.run.name={sft_run_name}",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+cluster.service_account_name=default",
        "+cluster.priority_class_name=test_pc_name",
        "+cluster.annotations.annotation_key_1=annotation-value-1",
        "+cluster.custom_labels.label_key_1=label-value-1",
        "+cluster.label_selector.required.example_label_key=[expected-label-value-1, expected-label-value-2]",
        "recipes.training_config.model.hidden_dropout=-1",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))

    with pytest.raises(ValueError):
        main(sample_recipe_k8s_config)
