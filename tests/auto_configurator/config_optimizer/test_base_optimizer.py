# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from auto_configurator.config_optimizer.base_optimizer import BaseOptimizer


class ConcreteOptimizer(BaseOptimizer):
    """Concrete implementation for testing"""

    def get_recipe_overrides(self, candidate_params):
        return [f"++param={v}" for v in candidate_params.values()]

    def tune_candidate(self, candidate, error_code):
        return candidate, False

    def _tunable_params(self):
        return ["param1", "param2"]

    def _generate_parameter_ranges(self):
        return {"param1": [1, 2], "param2": [3, 4]}

    def _estimate_memory_per_gpu(self, train_batch_size, candidate):
        # Return memory that scales with batch size to avoid infinite loop
        lower = train_batch_size * 0.5
        upper = train_batch_size * 1.0
        return lower, upper

    def _is_valid_candidate(self, candidate):
        return True


@pytest.fixture
def mock_recipe_cfg():
    return OmegaConf.create(
        {
            "training_config": {
                "model_config": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
                "training_args": {"train_batch_size": 16, "micro_train_batch_size": 2},
            },
            "trainer": {"devices": 8, "num_nodes": 1},
        }
    )


@pytest.fixture
def mock_autotune_config():
    return {}


class TestInit:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_init(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = ConcreteOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        assert optimizer._gpu_memory == 80.0
        assert optimizer.cfg == mock_recipe_cfg
        assert optimizer.num_params > 0


class TestComputeNumParams:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_compute_num_params(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = ConcreteOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        # Llama 3.1 8B should have approximately 8 billion parameters
        assert optimizer.num_params > 7_000_000_000
        assert optimizer.num_params < 9_000_000_000

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_compute_num_params_with_moe(self, mock_gpu_mem, mock_autotune_config):
        """Test _compute_num_params with MoE model"""
        mock_gpu_mem.return_value = 80.0

        # Use a model with MoE - GPT-OSS has MoE
        cfg = OmegaConf.create(
            {
                "training_config": {"model_config": {"model_name_or_path": "openai/gpt-oss-20b"}},
                "trainer": {"devices": 8},
            }
        )

        optimizer = ConcreteOptimizer(mock_autotune_config, cfg, "ml.p5.48xlarge")

        # MoE models should have more parameters
        assert optimizer.num_params > 0


class TestLoadModelParams:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_load_model_params(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = ConcreteOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        model_params = optimizer._load_model_params()

        assert model_params.vocab_size == 128256
        assert model_params.hidden_width == 4096
        assert model_params.num_layers == 32

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_load_model_params_unknown_model(self, mock_gpu_mem, mock_autotune_config):
        mock_gpu_mem.return_value = 80.0

        cfg = OmegaConf.create(
            {
                "training_config": {"model_config": {"model_name_or_path": "unknown/model"}},
                "trainer": {"devices": 8},
            }
        )

        with pytest.raises(ValueError, match="Unknown model"):
            ConcreteOptimizer(mock_autotune_config, cfg, "ml.p5.48xlarge")


class TestIsValidAutoconfig:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_autoconfig_with_non_tunable_param(self, mock_gpu_mem, mock_recipe_cfg):
        """Test _is_valid_autoconfig raises KeyError for non-tunable params"""
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"non_tunable_param": "auto"}

        with pytest.raises(KeyError, match="non_tunable_param is not a tunable parameter"):
            ConcreteOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_autoconfig_with_non_tunable_param_list(self, mock_gpu_mem, mock_recipe_cfg):
        """Test _is_valid_autoconfig raises KeyError for non-tunable params with ListConfig"""
        mock_gpu_mem.return_value = 80.0

        from omegaconf import ListConfig

        autotune_config = {"non_tunable_param": ListConfig([1, 2, 3])}

        with pytest.raises(KeyError, match="non_tunable_param is not a tunable parameter"):
            ConcreteOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_autoconfig_with_tunable_param_auto(self, mock_gpu_mem, mock_recipe_cfg):
        """Test _is_valid_autoconfig succeeds with tunable param set to auto"""
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"param1": "auto"}

        # Should not raise
        optimizer = ConcreteOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._is_valid_autoconfig()

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_autoconfig_with_tunable_param_int(self, mock_gpu_mem, mock_recipe_cfg):
        """Test _is_valid_autoconfig succeeds with tunable param as int"""
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"param1": 123}

        # Should not raise - non-string values are allowed
        optimizer = ConcreteOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._is_valid_autoconfig()

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_autoconfig_with_tunable_param_list(self, mock_gpu_mem, mock_recipe_cfg):
        """Test _is_valid_autoconfig succeeds with tunable param as ListConfig"""
        mock_gpu_mem.return_value = 80.0

        from omegaconf import ListConfig

        autotune_config = {"param1": ListConfig([1, 2, 3])}

        # Should not raise
        optimizer = ConcreteOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._is_valid_autoconfig()


class TestFindBatchSize:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_edge_cases(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """Test _find_batch_size edge cases"""
        mock_gpu_mem.return_value = 80.0

        class EdgeCaseOptimizer(ConcreteOptimizer):
            def __init__(self, *args, memory_scenario="normal", **kwargs):
                self.memory_scenario = memory_scenario
                super().__init__(*args, **kwargs)

            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                if self.memory_scenario == "tight_fit":
                    # Memory just slightly over GPU memory
                    return 0, 85.0
                elif self.memory_scenario == "very_high":
                    # Memory way over limit
                    return 0, 150.0
                else:
                    return train_batch_size * 0.5, train_batch_size * 1.0

        # Test tight fit scenario (returns 1)
        optimizer = EdgeCaseOptimizer(
            mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge", memory_scenario="tight_fit"
        )
        batch_size = optimizer._find_batch_size({})
        assert batch_size == 1

        # Test very high memory scenario (returns 0)
        optimizer = EdgeCaseOptimizer(
            mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge", memory_scenario="very_high"
        )
        batch_size = optimizer._find_batch_size({})
        assert batch_size == 0

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_starts_at_max_valid(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """_find_batch_size should start from train_batch_size // num_gpus"""
        mock_gpu_mem.return_value = 80.0

        checked_sizes = []

        class TrackingOptimizer(ConcreteOptimizer):
            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                checked_sizes.append(train_batch_size)
                return 0, train_batch_size * 5.0  # 5 GB per sample

        mock_recipe_cfg.training_config.training_args = OmegaConf.create(
            {"train_batch_size": 64, "micro_train_batch_size": 2}
        )
        optimizer = TrackingOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        # 64 // 8 = 8, so should start at 8 and halve down
        optimizer._find_batch_size({"max_len": 4096})

        assert checked_sizes[0] == 8

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_halves_until_fits(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """_find_batch_size should halve until memory fits"""
        mock_gpu_mem.return_value = 80.0

        class ScalingOptimizer(ConcreteOptimizer):
            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                return 0, train_batch_size * 20.0  # 20 GB per sample

        mock_recipe_cfg.training_config.training_args = OmegaConf.create(
            {"train_batch_size": 64, "micro_train_batch_size": 2}
        )
        optimizer = ScalingOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        # 8*20=160 > 80, 4*20=80 >= 80, 2*20=40 < 80 → returns 2
        result = optimizer._find_batch_size({"max_len": 4096})

        assert result == 2

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_returns_max_when_all_fit(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """_find_batch_size should return max valid size when all fit in memory"""
        mock_gpu_mem.return_value = 80.0

        class SmallMemOptimizer(ConcreteOptimizer):
            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                return 0, train_batch_size * 1.0  # 1 GB per sample

        mock_recipe_cfg.training_config.training_args = OmegaConf.create(
            {"train_batch_size": 32, "micro_train_batch_size": 2}
        )
        optimizer = SmallMemOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        # 32 // 8 = 4, 4*1=4 < 80 → returns 4 immediately
        result = optimizer._find_batch_size({"max_len": 4096})

        assert result == 4

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_borderline_returns_1(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """_find_batch_size returns 1 when batch=1 exceeds GPU memory but within 25%"""
        mock_gpu_mem.return_value = 80.0

        class BorderlineOptimizer(ConcreteOptimizer):
            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                return 0, 95.0  # Over 80 but under 100 (5/4 * 80)

        optimizer = BorderlineOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        result = optimizer._find_batch_size({"max_len": 4096})

        assert result == 1

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_find_batch_size_way_over_returns_0(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """_find_batch_size returns 0 when even batch=1 far exceeds GPU memory"""
        mock_gpu_mem.return_value = 80.0

        class HugeMemOptimizer(ConcreteOptimizer):
            def _estimate_memory_per_gpu(self, train_batch_size, candidate):
                return 0, 150.0  # Way over 100 (5/4 * 80)

        optimizer = HugeMemOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        result = optimizer._find_batch_size({"max_len": 4096})

        assert result == 0


class TestGenerateCandidateConfigurations:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_candidate_configurations(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = ConcreteOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        candidates = optimizer.generate_candidate_configurations(max_len=2048)

        assert len(candidates) > 0
        assert all(isinstance(c, dict) for c in candidates)
        assert all("max_len" in c for c in candidates)
        assert all(c["max_len"] == 2048 for c in candidates)

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_candidate_configurations_skips_train_batch_size(self, mock_gpu_mem, mock_recipe_cfg):
        """Test that micro_train_batch_size in tunable_params is skipped during iteration"""
        mock_gpu_mem.return_value = 80.0

        class OptimizerWithBatchSize(ConcreteOptimizer):
            def _tunable_params(self):
                return ["param1", "micro_train_batch_size", "param2"]

        autotune_config = {"param1": "auto", "micro_train_batch_size": "auto", "param2": "auto"}
        optimizer = OptimizerWithBatchSize(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        candidates = optimizer.generate_candidate_configurations(max_len=2048)

        # Should still generate candidates and compute micro_train_batch_size
        assert len(candidates) > 0
        assert all("micro_train_batch_size" in c for c in candidates)

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_candidate_configurations_with_non_list_value(self, mock_gpu_mem, mock_recipe_cfg):
        """Test that non-list values in parameter ranges are converted to lists"""
        mock_gpu_mem.return_value = 80.0

        class OptimizerWithNonListValue(ConcreteOptimizer):
            def _generate_parameter_ranges(self):
                return {"param1": 5, "param2": [3, 4]}

        autotune_config = {"param1": "auto", "param2": "auto"}
        optimizer = OptimizerWithNonListValue(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        candidates = optimizer.generate_candidate_configurations(max_len=2048)

        # Should generate candidates with param1=5 and param2 in [3, 4]
        assert len(candidates) > 0
        assert all(c["param1"] == 5 for c in candidates)

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_candidate_with_invalid_candidate(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """Test that invalid candidates are skipped"""
        mock_gpu_mem.return_value = 80.0

        class InvalidCandidateOptimizer(ConcreteOptimizer):
            def _is_valid_candidate(self, candidate):
                # Reject all candidates
                return False

        optimizer = InvalidCandidateOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        candidates = optimizer.generate_candidate_configurations(max_len=2048)

        assert len(candidates) == 0

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_candidate_with_zero_batch_size(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """Test that candidates with zero batch size are skipped"""
        mock_gpu_mem.return_value = 80.0

        class ZeroBatchOptimizer(ConcreteOptimizer):
            def _find_batch_size(self, candidate):
                return 0

        optimizer = ZeroBatchOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        candidates = optimizer.generate_candidate_configurations(max_len=2048)

        assert len(candidates) == 0
