import logging

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

import os

import pytest
import yaml

from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)


def compare_custom_k8s_artifacts(artifacts_dir):
    logger.info("Comparing custom k8s artifacts")

    artifacts_paths = [
        "/test_custom/test_custom_submission.sh",
        "/test_custom/k8s_template/Chart.yaml",
        "/test_custom/k8s_template/values.yaml",
        "/test_custom/k8s_template/templates/training.yaml",
    ]

    k8s_baseline_artifacts_path = "/tests/k8s_workflow/k8s_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, k8s_baseline_artifacts_path)


def test_custom_k8s_workflow():
    logger.info("Testing k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "training_cfg.entry_script=test.py",
        "cluster.instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "git.repo_url_or_path=https://github.com/example",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
    ]

    sample_custom_k8s_config = make_hydra_cfg_instance("../launcher_scripts/custom_script", "config_k8s", overrides)

    logger.info("\nsample_custom_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_custom_k8s_config))

    main(sample_custom_k8s_config)

    compare_custom_k8s_artifacts(artifacts_dir)


def test_custom_labels_k8s_workflow():
    """Test that custom_labels dict is passed through the k8s workflow without literal_eval."""
    logger.info("Testing custom_labels in k8s workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
        "+cluster.custom_labels.env=prod",
        "+cluster.custom_labels.team=ml",
    ]

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_config with custom_labels\n")
    logger.info(OmegaConf.to_yaml(sample_config))

    main(sample_config)

    # Verify custom_labels appears in generated values.yaml
    for root, dirs, files in os.walk(artifacts_dir):
        if "values.yaml" in files and "k8s_template" in root:
            values_path = os.path.join(root, "values.yaml")
            with open(values_path, "r") as f:
                values = yaml.safe_load(f)

            assert "customLabels" in values["trainingConfig"], "customLabels should be in trainingConfig"
            assert values["trainingConfig"]["customLabels"] == {
                "env": "prod",
                "team": "ml",
            }, "customLabels should match input dict"
            logger.info("custom_labels correctly set in values.yaml")
            return

    pytest.fail("values.yaml not found in artifacts")


def test_custom_labels_not_set_when_missing():
    """Test that customLabels is None in values.yaml when not provided."""
    logger.info("Testing custom_labels not set when missing")

    artifacts_dir = create_temp_directory()
    overrides = [
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
        "cluster=k8s",
        "cluster_type=k8s",
    ]

    sample_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_config without custom_labels\n")
    logger.info(OmegaConf.to_yaml(sample_config))

    main(sample_config)

    # Verify customLabels is None in generated values.yaml
    for root, dirs, files in os.walk(artifacts_dir):
        if "values.yaml" in files and "k8s_template" in root:
            values_path = os.path.join(root, "values.yaml")
            with open(values_path, "r") as f:
                values = yaml.safe_load(f)

            assert values["trainingConfig"]["customLabels"] is None, "customLabels should be None when not provided"
            logger.info("customLabels correctly set to None in values.yaml")
            return

    pytest.fail("values.yaml not found in artifacts")
