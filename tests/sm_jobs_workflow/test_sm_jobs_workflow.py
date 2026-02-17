import logging
import os

import pytest

os.environ["NEMO_LAUNCHER_DEBUG"] = "1"

from unittest.mock import patch

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)


from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
)


def compare_sm_jobs_common_artifacts(artifacts_dir, prefix, baseline_artifacts_subdir):
    logger.info("Comparing sm_jobs common artifacts")

    artifacts_paths = [
        f"/{prefix}/{prefix}_submission.sh",
        f"/{prefix}/{prefix}_hydra.yaml",
        f"/{prefix}/sm_jobs_config.yaml",
        f"/{prefix}/launch.py",
    ]
    baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/" + baseline_artifacts_subdir
    compare_artifacts(artifacts_paths, artifacts_dir, baseline_artifacts_dir)


def is_requirements_file(artifacts_dir, prefix, baseline_artifacts_subdir, reqs):
    logger.info("Checking sm_jobs requirements")

    artifacts_paths = [f"/{prefix}/requirements.txt"]
    if not reqs:
        reqs_file = artifacts_dir + artifacts_paths[0]
        assert not (os.path.exists(reqs_file))
    else:
        baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/" + baseline_artifacts_subdir
        compare_artifacts(artifacts_paths, artifacts_dir, baseline_artifacts_dir)


@pytest.mark.xfail(reason="Broken by HF recipe removal, need to be fixed")
def test_sm_jobs_workflow_no_additional_kwargs():
    logger.info("Testing sm_jobs workflow without additional kwargs")

    artifacts_dir = create_temp_directory()
    overrides = [
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
        "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_sm_jobs_config\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))

    main(sample_sm_jobs_config)

    compare_sm_jobs_common_artifacts(artifacts_dir, "llama-8b", "no_kwargs")
    is_requirements_file(artifacts_dir, "llama-8b", "no_kwargs", False)


@pytest.mark.xfail(reason="Broken by HF recipe removal, need to be fixed")
def test_sm_jobs_workflow_with_additional_kwargs():
    logger.info("Testing sm_jobs workflow with additional kwargs")

    artifacts_dir = create_temp_directory()
    overrides = [
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.inputs.s3.train=s3://test_path",
        "+cluster.sm_jobs_config.inputs.s3.val=s3://test_path",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_sm_jobs_config\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))

    main(sample_sm_jobs_config)

    compare_sm_jobs_common_artifacts(artifacts_dir, "llama-8b", "with_kwargs")
    is_requirements_file(artifacts_dir, "llama-8b", "with_kwargs", False)


@pytest.mark.xfail(reason="Broken by HF recipe removal, need to be fixed")
def test_sm_jobs_workflow_multimodal():
    logger.info("Testing sm_jobs workflow for multi-modal")

    artifacts_dir = create_temp_directory()
    overrides = [
        "recipes=training/llama/hf_llama3_2_11b_seq8k_gpu_p5x4_pretrain",
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
        "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    logger.info("\nsample_sm_jobs_config\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))

    main(sample_sm_jobs_config)

    compare_sm_jobs_common_artifacts(artifacts_dir, "llama3-2-11b", "multimodal")
    is_requirements_file(artifacts_dir, "llama3-2-11b", "multimodal", True)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_sm_jobs_workflow_with_launch_json(mock_load_hosting):
    logger.info("Testing sm_jobs workflow launch json")

    artifacts_dir = create_temp_directory("test_sm_jobs_workflow_with_launch_json")
    overrides = [
        "recipes=fine-tuning/llama/llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
        "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "git.use_default=false",
        "git.entry_script=/app/src/train_hp.py",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(sample_sm_jobs_config, False)
    sample_sm_jobs_config.launch_json = True

    logger.info("\nsample_sm_jobs_config\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))
    del sample_sm_jobs_config["hydra"]
    main(sample_sm_jobs_config)

    baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/with_launch_json"
    compare_artifacts(["/llama-3-2-1b-instruct/launch.json"], artifacts_dir, baseline_artifacts_dir)


# SM Jobs PySdk Launch Executor Tests (Estimator and ModelTrainer)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_sm_jobs_workflow_estimator_api(mock_load_hosting):
    """Test SM Jobs workflow using PyTorch Estimator API (pysdk_launch_executor=estimator)."""
    logger.info("Testing sm_jobs workflow with Estimator API")

    artifacts_dir = create_temp_directory("test_sm_jobs_workflow_estimator")
    overrides = [
        "recipes=fine-tuning/llama/llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "cluster.sm_jobs_config.pysdk_launch_executor=estimator",
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
        "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "git.use_default=false",
        "git.entry_script=/app/src/train_hp.py",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(sample_sm_jobs_config, False)

    logger.info("\nsample_sm_jobs_config (Estimator API)\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))
    del sample_sm_jobs_config["hydra"]
    main(sample_sm_jobs_config)

    # Verify launch.py is generated from template/sm_jobs.py (Estimator template)
    baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/pysdk_launch_executor/estimator"
    compare_artifacts(
        ["/llama-3-2-1b-instruct/launch.py", "/llama-3-2-1b-instruct/sm_jobs_config.yaml"],
        artifacts_dir,
        baseline_artifacts_dir,
    )


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_sm_jobs_workflow_model_trainer_api(mock_load_hosting):
    """Test SM Jobs workflow using ModelTrainer API (pysdk_launch_executor=model_trainer)."""
    logger.info("Testing sm_jobs workflow with ModelTrainer API")

    artifacts_dir = create_temp_directory("test_sm_jobs_workflow_model_trainer")
    overrides = [
        "recipes=fine-tuning/llama/llmft_llama3_2_1b_instruct_seq4k_gpu_sft_lora",
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "instance_type=p5.48xlarge",
        "base_results_dir={}".format(artifacts_dir),
        "cluster.sm_jobs_config.pysdk_launch_executor=model_trainer",
        "+cluster.sm_jobs_config.output_path=s3://test_path",
        "+cluster.sm_jobs_config.tensorboard_config.output_path=s3://test_tensorboard_path",
        "+cluster.sm_jobs_config.tensorboard_config.container_logs_path=/opt/ml/output/tensorboard",
        "container=test_container",
        "+env_vars.NEMO_LAUNCHER_DEBUG=1",
        "git.use_default=false",
        "git.entry_script=/app/src/train_hp.py",
    ]

    sample_sm_jobs_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(sample_sm_jobs_config, False)

    logger.info("\nsample_sm_jobs_config (ModelTrainer API)\n")
    logger.info(OmegaConf.to_yaml(sample_sm_jobs_config))
    del sample_sm_jobs_config["hydra"]
    main(sample_sm_jobs_config)

    # Verify launch.py is generated from template/sm_jobs_model_trainer.py (ModelTrainer template)
    baseline_artifacts_dir = "/tests/sm_jobs_workflow/sm_jobs_baseline_artifacts/pysdk_launch_executor/model_trainer"
    compare_artifacts(
        ["/llama-3-2-1b-instruct/launch.py", "/llama-3-2-1b-instruct/sm_jobs_config.yaml"],
        artifacts_dir,
        baseline_artifacts_dir,
    )
