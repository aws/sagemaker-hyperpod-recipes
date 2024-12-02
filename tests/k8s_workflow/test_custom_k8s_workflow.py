import logging

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)

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
