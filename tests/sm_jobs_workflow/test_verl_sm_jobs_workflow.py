import logging
import os

import pytest

os.environ["NEMO_LAUNCHER_DEBUG"] = "1"


from main import main

logger = logging.getLogger(__name__)

from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)


def compare_verl_sm_jobs_artifacts(artifacts_dir, prefix, baseline_artifacts_subdir):
    logger.info("Comparing VERL sm_jobs artifacts")

    artifacts_paths = [
        f"/{prefix}/verl-config.yaml",
    ]
    baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/" + baseline_artifacts_subdir
    compare_artifacts(artifacts_paths, artifacts_dir, baseline_artifacts_dir)


@pytest.mark.xfail(reason="Broken by HF recipe removal, need to be fixed")
def test_verl_sm_jobs_workflow_config_generation():
    """Test that VERL SageMaker jobs generate the required verl-config.yaml file"""
    logger.info("Testing VERL sm_jobs workflow config generation")

    artifacts_dir = create_temp_directory()
    overrides = [
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "++cluster.sm_jobs_config.output_path=s3://test_path",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        # VERL-specific configuration
        "++recipes.run.model_type=verl",
        "++recipes.training_config.actor_rollout_ref.model=test_actor_model",
        "++recipes.training_config.trainer.devices=8",
        "++recipes.training_config.trainer.num_nodes=1",
        "++recipes.training_config.data.dataset_path=/test/data",
        "++recipes.training_config.critic.model=test_critic_model",
        "++recipes.training_config.reward_model.model=test_reward_model",
        "++recipes.training_config.algorithm.name=grpo",
        "++recipes.training_config.ray_init.address=auto",
        # Ray cluster configuration required for VERL
        "++recipes.ray_cluster.head_node.cpu=16",
        "++recipes.ray_cluster.worker_nodes.replicas=1",
    ]

    cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    main(cfg)

    # Compare against baseline artifacts (following same pattern as other SM jobs tests)
    compare_verl_sm_jobs_artifacts(artifacts_dir, "llama-8b", "verl")


def test_non_verl_sm_jobs_workflow():
    """Test that non-VERL SageMaker jobs do not generate verl-config.yaml"""
    logger.info("Testing non-VERL sm_jobs workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "++cluster.sm_jobs_config.output_path=s3://test_path",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
    ]

    cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    main(cfg)

    # Find the actual job directory (has random suffix)
    job_dirs = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
    assert len(job_dirs) == 1, f"Expected exactly one job directory, found: {job_dirs}"
    job_dir = os.path.join(artifacts_dir, job_dirs[0])

    # Verify verl-config.yaml was NOT generated
    verl_config_path = os.path.join(job_dir, "verl-config.yaml")
    assert not os.path.exists(verl_config_path), "verl-config.yaml should not be generated for non-VERL jobs"
