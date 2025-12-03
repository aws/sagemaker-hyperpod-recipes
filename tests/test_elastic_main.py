import logging
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from launcher.nemo.recipe_stages import (
    SMTrainingGPURecipe,
    SMTrainingGPURecipeElastic,
    SMTrainingTrainiumRecipe,
)
from launcher.nemo.stages import SMCustomTrainingGPU, SMCustomTrainingTrainium
from main import get_training_stage

logger = logging.getLogger(__name__)


class TestElasticMainIntegration:
    """Test suite for elastic training integration in main.py"""

    def test_get_training_stage_elastic_gpu_recipe(self):
        """Test that elastic GPU recipe is selected when elastic policy is enabled"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {
                    "elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4, "replica_increment_step": 1}
                },
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipeElastic

    def test_get_training_stage_non_elastic_gpu_recipe(self):
        """Test that standard GPU recipe is selected when elastic policy is disabled"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": False}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipe

    def test_get_training_stage_missing_elastic_policy_defaults_to_non_elastic(self):
        """Test that missing elastic policy defaults to non-elastic training"""
        cfg = OmegaConf.create({"instance_type": "p5.48xlarge", "cluster_type": "k8s", "training": {}})

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipe

    def test_get_training_stage_empty_training_section_defaults_to_non_elastic(self):
        """Test that empty training section defaults to non-elastic training"""
        cfg = OmegaConf.create({"instance_type": "p5.48xlarge", "cluster_type": "k8s"})

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipe

    def test_get_training_stage_trainium_not_affected_by_elastic_policy(self):
        """Test that Trainium training is not affected by elastic policy"""
        cfg = OmegaConf.create(
            {
                "instance_type": "trn1.32xlarge",
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingTrainiumRecipe

    def test_get_training_stage_custom_gpu_not_affected_by_elastic_policy(self):
        """Test that custom GPU training is not affected by elastic policy"""
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training_cfg": {"custom": "config"},  # This makes it custom training
                "training": {"elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMCustomTrainingGPU

    def test_get_training_stage_custom_trainium_not_affected_by_elastic_policy(self):
        """Test that custom Trainium training is not affected by elastic policy"""
        cfg = OmegaConf.create(
            {
                "instance_type": "trn1.32xlarge",
                "cluster_type": "k8s",
                "training_cfg": {"custom": "config"},  # This makes it custom training
                "training": {"elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMCustomTrainingTrainium

    def test_get_training_stage_elastic_policy_with_different_values(self):
        """Test elastic policy detection with different configuration values"""
        test_cases = [
            # Test case: is_elastic = True
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {
                        "elastic_policy": {
                            "is_elastic": True,
                            "min_nodes": 2,
                            "max_nodes": 8,
                            "replica_increment_step": 2,
                        }
                    },
                },
                "expected": SMTrainingGPURecipeElastic,
            },
            # Test case: is_elastic = False
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": False, "min_nodes": 1, "max_nodes": 1}},
                },
                "expected": SMTrainingGPURecipe,
            },
            # Test case: elastic_policy exists but is_elastic is missing (should default to False)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"min_nodes": 1, "max_nodes": 4}},
                },
                "expected": SMTrainingGPURecipe,
            },
        ]

        for i, test_case in enumerate(test_cases):
            cfg = OmegaConf.create(test_case["config"])
            stage_class = get_training_stage(cfg)
            assert (
                stage_class == test_case["expected"]
            ), f"Test case {i} failed: expected {test_case['expected']}, got {stage_class}"

    def test_get_training_stage_with_omegaconf_select_behavior(self):
        """Test that OmegaConf.select behavior works correctly for nested elastic policy"""
        # Test with deeply nested structure
        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {
                    "model": {"name": "llama"},
                    "elastic_policy": {"is_elastic": True, "config": {"min_nodes": 1, "max_nodes": 4}},
                },
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipeElastic

    def test_get_training_stage_invalid_device_type_with_elastic_policy(self):
        """Test that invalid device type raises error even with elastic policy"""
        cfg = OmegaConf.create(
            {
                "instance_type": "invalid.instance",  # This should result in invalid device type
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": True, "min_nodes": 1, "max_nodes": 4}},
            }
        )

        with pytest.raises(ValueError, match="Recipe only can be run on GPU or Trainium instances"):
            get_training_stage(cfg)

    @patch("main.get_device_type")
    def test_get_training_stage_device_type_detection_with_elastic(self, mock_get_device_type):
        """Test that device type detection works correctly with elastic policy"""
        # Test GPU device type with elastic policy
        mock_get_device_type.return_value = "gpu"

        cfg = OmegaConf.create(
            {
                "instance_type": "p5.48xlarge",
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": True}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingGPURecipeElastic
        mock_get_device_type.assert_called_once_with(cfg)

    @patch("main.get_device_type")
    def test_get_training_stage_trainium_device_type_with_elastic(self, mock_get_device_type):
        """Test that Trainium device type ignores elastic policy"""
        mock_get_device_type.return_value = "trainium"

        cfg = OmegaConf.create(
            {
                "instance_type": "trn1.32xlarge",
                "cluster_type": "k8s",
                "training": {"elastic_policy": {"is_elastic": True}},
            }
        )

        stage_class = get_training_stage(cfg)
        assert stage_class == SMTrainingTrainiumRecipe
        mock_get_device_type.assert_called_once_with(cfg)

    def test_get_training_stage_elastic_policy_edge_cases(self):
        """Test edge cases for elastic policy configuration"""
        edge_cases = [
            # Case 1: is_elastic is None (should default to False)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": None}},
                },
                "expected": SMTrainingGPURecipe,
            },
            # Case 2: is_elastic is 0 (should be falsy)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": 0}},
                },
                "expected": SMTrainingGPURecipe,
            },
            # Case 3: is_elastic is 1 (should be truthy)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": 1}},
                },
                "expected": SMTrainingGPURecipeElastic,
            },
            # Case 4: is_elastic is string "true" (should be truthy)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": "true"}},
                },
                "expected": SMTrainingGPURecipeElastic,
            },
            # Case 5: is_elastic is string "false" (should be truthy because non-empty string)
            {
                "config": {
                    "instance_type": "p5.48xlarge",
                    "cluster_type": "k8s",
                    "training": {"elastic_policy": {"is_elastic": "false"}},
                },
                "expected": SMTrainingGPURecipeElastic,
            },
        ]

        for i, case in enumerate(edge_cases):
            cfg = OmegaConf.create(case["config"])
            stage_class = get_training_stage(cfg)
            assert (
                stage_class == case["expected"]
            ), f"Edge case {i} failed: expected {case['expected']}, got {stage_class}"
