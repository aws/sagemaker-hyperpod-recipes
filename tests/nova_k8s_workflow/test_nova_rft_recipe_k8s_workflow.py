import logging
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

from tests.test_utils import (
    create_temp_directory,
    make_hydra_cfg_instance,
    replace_placeholder,
)

rft_run_name = "nova-lite-rft"


def compare_rft_artifacts_with_dynamic_names(
    artifacts_mapping, artifacts_dir, baseline_artifacts_path, actual_job_name, baseline_job_name
):
    """Compare RFT artifacts handling dynamic job names."""
    import os
    import shutil

    for actual_path, baseline_path in artifacts_mapping:
        current_dir = os.getcwd()
        actual_artifact_path = artifacts_dir + actual_path
        baseline_artifact_folder = current_dir + baseline_artifacts_path

        # Make a copy of baseline artifacts to replace placeholders
        baseline_artifact_copy_folder = create_temp_directory()
        shutil.copytree(baseline_artifact_folder, baseline_artifact_copy_folder, dirs_exist_ok=True)
        baseline_artifact_path = baseline_artifact_copy_folder + baseline_path

        # Replace placeholders in baseline
        results_dir_placeholder = "{$results_dir}"
        replace_placeholder(baseline_artifact_path, results_dir_placeholder, artifacts_dir)

        # Also replace hardcoded /tmp path that might exist in baseline
        replace_placeholder(baseline_artifact_path, "/tmp/test_sm_jobs_workflow_with_launch_json", artifacts_dir)
        replace_placeholder(baseline_artifact_path, "/tmp/test_recipe_k8s_workflow_with_launch_json", artifacts_dir)

        workspace_dir_placeholder = "{$workspace_dir}"
        replace_placeholder(baseline_artifact_path, workspace_dir_placeholder, current_dir)

        # Replace the baseline job name with the actual job name in the baseline file
        # Only replace if the names are different to avoid issues when they're the same
        if baseline_job_name != actual_job_name:
            replace_placeholder(baseline_artifact_path, baseline_job_name, actual_job_name)

        # Read both files and compare with whitespace normalization
        with open(baseline_artifact_path, "r") as f:
            baseline_content = f.read().rstrip("\n\r\t ")
        with open(actual_artifact_path, "r") as f:
            actual_content = f.read().rstrip("\n\r\t ")

        if baseline_content != actual_content:
            assert (
                False
            ), f"{baseline_artifact_path} does not match {actual_artifact_path}\nBaseline: {repr(baseline_content)}\nActual: {repr(actual_content)}"


@pytest.fixture(autouse=True)
def mock_aws_account_id():
    with patch("launcher.nova.launchers.boto3.client") as mock_boto_client:
        mock_boto_client.return_value.get_caller_identity.return_value = {"Account": "123456789012"}
        yield


@pytest.fixture(autouse=True)
def mock_aws_region():
    session_mock = MagicMock()
    session_mock.region_name = "us-east-1"

    with patch("launcher.nova.launchers.boto3.session.Session", return_value=session_mock):
        yield


def compare_rft_recipe_k8s_artifacts(artifacts_dir, launch_json=False):
    logger.info("Comparing rft recipe k8s artifacts")

    # Find the actual generated directory (it will have a suffix)
    import os

    generated_dirs = [d for d in os.listdir(artifacts_dir) if d.startswith(rft_run_name)]
    if not generated_dirs:
        raise AssertionError(f"No generated directory found starting with {rft_run_name}")

    actual_job_name = generated_dirs[0]
    logger.info(f"Found generated job directory: {actual_job_name}")

    # Create mapping of actual paths to baseline paths
    artifacts_mapping = [
        (f"/{actual_job_name}/{actual_job_name}_launch.sh", f"/{rft_run_name}/{rft_run_name}_launch.sh"),
        (f"/{actual_job_name}/k8s_templates/values.yaml", f"/{rft_run_name}/k8s_templates/values.yaml"),
        (f"/{actual_job_name}/k8s_templates/Chart.yaml", f"/{rft_run_name}/k8s_templates/Chart.yaml"),
        # Service config files
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-training_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-training_hydra.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-vllm-generation_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-vllm-generation_hydra.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-hub_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-hub_hydra.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-nats-bootstrap_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-nats-bootstrap_hydra.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-prompter_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-prompter_hydra.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/config/{actual_job_name}-rbs_hydra.yaml",
            f"/{rft_run_name}/k8s_templates/config/{rft_run_name}-rbs_hydra.yaml",
        ),
        # Template files
        (
            f"/{actual_job_name}/k8s_templates/templates/training.yaml",
            f"/{rft_run_name}/k8s_templates/templates/training.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/templates/vllm-generation.yaml",
            f"/{rft_run_name}/k8s_templates/templates/vllm-generation.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/templates/hub.yaml",
            f"/{rft_run_name}/k8s_templates/templates/hub.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/templates/prompt-rbs.yaml",
            f"/{rft_run_name}/k8s_templates/templates/prompt-rbs.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/templates/nats-server.yaml",
            f"/{rft_run_name}/k8s_templates/templates/nats-server.yaml",
        ),
        (
            f"/{actual_job_name}/k8s_templates/templates/training-config.yaml",
            f"/{rft_run_name}/k8s_templates/templates/training-config.yaml",
        ),
    ]

    if launch_json:
        artifacts_mapping.append((f"/{actual_job_name}/launch.json", f"/{rft_run_name}/launch.json"))

    k8s_baseline_artifacts_path = "/tests/nova_k8s_workflow/k8s_baseline_artifacts/without_launch_json"

    # Use custom comparison logic for RFT artifacts
    compare_rft_artifacts_with_dynamic_names(
        artifacts_mapping, artifacts_dir, k8s_baseline_artifacts_path, actual_job_name, rft_run_name
    )


def test_rft_recipe_k8s_workflow():
    logger.info("Testing RFT recipe k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=fine-tuning/nova/nova_2_0/nova_lite/RFT/nova_lite_2_0_p5_gpu_rft",
        "instance_type=p5.48xlarge",
        f"recipes.run.name={rft_run_name}",
        "recipes.run.replicas=2",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "++cluster.service_account_name=test-service-account",
        "++cluster.namespace=kubeflow",
        "++cluster.region=us-west-2",
        "++cluster.alias=test-user",
        "++cluster.node_type=p5.48xlarge",
        "++cluster.toleration_value=test-toleration",
    ]

    sample_recipe_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_recipe_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_recipe_k8s_config))
    main(sample_recipe_k8s_config)
    compare_rft_recipe_k8s_artifacts(artifacts_dir)
