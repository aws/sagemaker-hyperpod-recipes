import logging
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from main import main
from tests.test_utils import (
    create_temp_directory,
    make_hydra_cfg_instance,
    mock_load_hosting_config,
)

logger = logging.getLogger(__name__)


def test_elastic_k8s_workflow():
    """Test elastic training K8s workflow generates correct artifacts"""
    logger.info("Testing elastic K8s workflow")

    artifacts_dir = create_temp_directory("test_elastic_k8s_workflow")
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

    sample_elastic_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)

    # Manually add elastic policy to test elastic functionality
    OmegaConf.set_struct(sample_elastic_k8s_config, False)
    sample_elastic_k8s_config.recipes.elastic_policy = OmegaConf.create(
        {"is_elastic": True, "min_nodes": 1, "max_nodes": 4, "replica_increment_step": 1}
    )

    logger.info("\nsample_elastic_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_elastic_k8s_config))
    del sample_elastic_k8s_config["hydra"]

    main(sample_elastic_k8s_config)

    # Verify that elastic-specific files are generated
    artifacts_paths = [
        "/llama-3-2-1b-instruct/k8s_template/values.yaml",
        "/llama-3-2-1b-instruct/k8s_template/Chart.yaml",
        "/llama-3-2-1b-instruct/k8s_template/templates/training.yaml",
        "/llama-3-2-1b-instruct/k8s_template/templates/training-config.yaml",
    ]

    # Check that all expected files exist
    for artifact_path in artifacts_paths:
        full_path = Path(artifacts_dir + artifact_path)
        assert full_path.exists(), f"Expected artifact {artifact_path} was not generated"

    # Verify elastic-specific content in generated files
    _verify_elastic_training_yaml(artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/templates/training.yaml")
    _verify_elastic_values_yaml(artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/values.yaml")
    _verify_elastic_training_config_yaml(
        artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/templates/training-config.yaml"
    )


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    side_effect=mock_load_hosting_config,
)
def test_elastic_k8s_workflow_with_launch_json(mock_load_hosting_config_patch):
    """Test elastic training K8s workflow with launch_json generates correct artifacts"""
    logger.info("Testing elastic K8s workflow with launch_json")

    artifacts_dir = create_temp_directory("test_elastic_k8s_workflow_with_launch_json")
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

    sample_elastic_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(sample_elastic_k8s_config, False)

    # Add elastic policy and launch_json
    sample_elastic_k8s_config.recipes.elastic_policy = OmegaConf.create(
        {"is_elastic": True, "min_nodes": 1, "max_nodes": 4, "replica_increment_step": 1}
    )
    sample_elastic_k8s_config.cluster.persistent_volume_claims = [{"claimName": "fsx-claim", "mountPath": "/data"}]
    sample_elastic_k8s_config.launch_json = True

    logger.info("\nsample_elastic_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_elastic_k8s_config))
    del sample_elastic_k8s_config["hydra"]

    main(sample_elastic_k8s_config)

    # Verify that launch.json is generated for elastic training
    launch_json_path = Path(artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/launch.json")
    assert launch_json_path.exists(), "launch.json was not generated for elastic training"

    # Verify elastic-specific content in generated files
    _verify_elastic_training_yaml(artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/templates/training.yaml")
    _verify_elastic_values_yaml(artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/values.yaml")


def test_non_elastic_k8s_workflow():
    """Test that non-elastic training doesn't include elastic policy sections"""
    logger.info("Testing non-elastic K8s workflow")

    artifacts_dir = create_temp_directory("test_non_elastic_k8s_workflow")
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

    sample_non_elastic_k8s_config = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    logger.info("\nsample_non_elastic_k8s_config\n")
    logger.info(OmegaConf.to_yaml(sample_non_elastic_k8s_config))
    OmegaConf.set_struct(sample_non_elastic_k8s_config, False)
    del sample_non_elastic_k8s_config["hydra"]

    main(sample_non_elastic_k8s_config)

    # Verify that elastic policy sections are not present in non-elastic training
    training_yaml_path = artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/templates/training.yaml"
    values_yaml_path = artifacts_dir + "/llama-3-2-1b-instruct/k8s_template/values.yaml"

    _verify_non_elastic_training_yaml(training_yaml_path)
    _verify_non_elastic_values_yaml(values_yaml_path)


def _verify_elastic_training_yaml(file_path):
    """Verify that training.yaml contains elastic policy configuration"""
    with open(file_path, "r") as f:
        content = f.read()

    # Check for elastic policy section
    assert "elasticPolicy:" in content, "training.yaml should contain elasticPolicy section for elastic training"
    assert "minReplicas:" in content, "training.yaml should contain minReplicas for elastic training"
    assert "maxReplicas:" in content, "training.yaml should contain maxReplicas for elastic training"
    assert "replicaIncrementStep:" in content, "training.yaml should contain replicaIncrementStep for elastic training"

    # Check for conditional logic
    assert (
        "{{- if $config.elastic_policy.is_elastic }}" in content
    ), "training.yaml should have conditional elastic policy logic"


def _verify_elastic_values_yaml(file_path):
    """Verify that values.yaml contains elastic policy configuration"""
    with open(file_path, "r") as f:
        content = f.read()

    # Check for elastic policy section
    assert "elastic_policy:" in content, "values.yaml should contain elastic_policy section"
    assert "is_elastic: true" in content, "values.yaml should have is_elastic set to true for elastic training"
    assert "min_nodes:" in content, "values.yaml should contain min_nodes for elastic training"
    assert "max_nodes:" in content, "values.yaml should contain max_nodes for elastic training"


def _verify_elastic_training_config_yaml(file_path):
    """Verify that training-config.yaml contains elastic-specific configuration logic"""
    with open(file_path, "r") as f:
        content = f.read()

    # Check for elastic-specific config logic
    assert (
        "{{- if $config.elastic_policy.is_elastic }}" in content
    ), "training-config.yaml should have conditional elastic policy logic"
    assert "train_config_n*.yaml" in content, "training-config.yaml should reference scale-specific config files"
    assert (
        "{{- range $path, $content := .Files.Glob" in content
    ), "training-config.yaml should have file globbing logic for elastic configs"


def _verify_non_elastic_training_yaml(file_path):
    """Verify that training.yaml does not contain elastic policy for non-elastic training"""
    with open(file_path, "r") as f:
        content = f.read()

    # For non-elastic training, elastic policy section should not be rendered
    # (the conditional logic should prevent it from appearing)
    lines = content.split("\n")
    elastic_policy_lines = [line for line in lines if "elasticPolicy:" in line and not line.strip().startswith("#")]

    # If elasticPolicy appears, it should be within a conditional block that evaluates to false
    if elastic_policy_lines:
        # Check that it's properly conditioned
        assert "{{- if $config.elastic_policy.is_elastic }}" in content, "Elastic policy should be conditional"


def _verify_non_elastic_values_yaml(file_path):
    """Verify that values.yaml has elastic policy disabled for non-elastic training"""
    with open(file_path, "r") as f:
        content = f.read()

    # Check that elastic policy is disabled
    assert "elastic_policy:" in content, "values.yaml should contain elastic_policy section"
    assert "is_elastic: false" in content, "values.yaml should have is_elastic set to false for non-elastic training"
