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

from auto_configurator.config_optimizer.llmft_optimizer import LlmftOptimizer
from auto_configurator.evaluation.base_evaluator import ErrorCode


@pytest.fixture
def mock_recipe_cfg():
    return OmegaConf.create(
        {
            "training_config": {
                "model_config": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
                "training_args": {"micro_train_batch_size": 2},
                "datasets": {"val_data": {"limit": 100}},
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

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        assert optimizer._gpu_memory == 80.0
        assert optimizer.cfg == mock_recipe_cfg


class TestTunableParams:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tunable_params(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        params = optimizer._tunable_params()

        assert params == ["train_batch_size", "sharding_strategy", "gradient_checkpointing", "cpu_offload"]


class TestGetRecipeOverrides:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_get_recipe_overrides(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "train_batch_size": 32,
            "micro_train_batch_size": 4,
            "sharding_strategy": "FULL_SHARD",
            "gradient_checkpointing": True,
            "cpu_offload": False,
            "max_len": 2048,
        }

        overrides = optimizer.get_recipe_overrides(candidate)

        assert "++recipes.training_config.training_args.train_batch_size=32" in overrides
        assert "++recipes.training_config.datasets.train_data.limit=480" in overrides  # 32 * 15 (MAX_STEPS)
        assert "++recipes.training_config.training_args.micro_train_batch_size=4" in overrides
        assert "++recipes.training_config.training_args.strategy.fsdp_config.sharding_strategy=FULL_SHARD" in overrides
        assert "++recipes.training_config.training_args.gradient_checkpointing=True" in overrides
        assert "++recipes.training_config.training_args.strategy.fsdp_config.cpu_offload=False" in overrides
        assert "++recipes.training_config.training_args.max_len=2048" in overrides

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_get_recipe_overrides_ignores_unknown_params(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {"train_batch_size": 32, "unknown_param": "value"}

        overrides = optimizer.get_recipe_overrides(candidate)

        assert "++recipes.training_config.training_args.train_batch_size=32" in overrides
        assert "++recipes.training_config.datasets.train_data.limit=480" in overrides  # 32 * 15 (MAX_STEPS)
        assert "unknown_param" not in str(overrides)


class TestGenerateParameterRanges:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_generate_parameter_ranges(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        assert "sharding_strategy" in param_ranges
        assert "gradient_checkpointing" in param_ranges
        assert "cpu_offload" in param_ranges


class TestSetShardingStrategy:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_sharding_strategy_auto(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"sharding_strategy": "auto"}
        optimizer = LlmftOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        assert param_ranges["sharding_strategy"] == ["FULL_SHARD", "HYBRID_SHARD"]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_sharding_strategy_single_value(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        # Use OmegaConf to bypass string validation
        from omegaconf import OmegaConf

        autotune_config = OmegaConf.create({"sharding_strategy": "FULL_SHARD"})

        # Manually set to bypass validation
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = autotune_config
        param_ranges = optimizer._generate_parameter_ranges()

        assert param_ranges["sharding_strategy"] == ["FULL_SHARD"]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_sharding_strategy_list(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        from omegaconf import ListConfig

        autotune_config = OmegaConf.create({"sharding_strategy": ListConfig(["FULL_SHARD", "NO_SHARD"])})
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = autotune_config
        param_ranges = optimizer._generate_parameter_ranges()

        # ListConfig is not isinstance(list), so it gets wrapped in a list
        assert param_ranges["sharding_strategy"] == [["FULL_SHARD", "NO_SHARD"]]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_sharding_strategy_python_list(self, mock_gpu_mem, mock_recipe_cfg):
        """Test with Python list (not ListConfig) - edge case"""
        mock_gpu_mem.return_value = 80.0

        # Use a plain object with Python list attribute to reach the else branch
        class MockConfig:
            sharding_strategy = ["FULL_SHARD", "NO_SHARD"]

        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = MockConfig()
        param_ranges = optimizer._generate_parameter_ranges()

        # Python list is isinstance(list), so it's used directly
        assert param_ranges["sharding_strategy"] == ["FULL_SHARD", "NO_SHARD"]


class TestSetGradientCheckpointing:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_gradient_checkpointing_auto_large_model(self, mock_gpu_mem, mock_recipe_cfg):
        """Test gradient checkpointing for models > 7B"""
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"gradient_checkpointing": "auto"}
        optimizer = LlmftOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        # Llama 3.1 8B is > 7B, should only enable checkpointing
        assert param_ranges["gradient_checkpointing"] == [True]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_gradient_checkpointing_auto_small_model(self, mock_gpu_mem):
        """Test gradient checkpointing for models <= 7B"""
        mock_gpu_mem.return_value = 80.0

        cfg = OmegaConf.create(
            {
                "training_config": {
                    "model_config": {"model_name_or_path": "meta-llama/Llama-3.2-1B-Instruct"},
                    "training_args": {"micro_train_batch_size": 2},
                    "datasets": {"val_data": {"limit": 100}},
                },
                "trainer": {"devices": 8, "num_nodes": 1},
            }
        )

        autotune_config = {"gradient_checkpointing": "auto"}
        optimizer = LlmftOptimizer(autotune_config, cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        # Small model should test both True and False
        assert param_ranges["gradient_checkpointing"] == [True, False]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_gradient_checkpointing_single_value(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        # Use non-string value to bypass validation
        autotune_config = {"gradient_checkpointing": True}
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = OmegaConf.create(autotune_config)
        param_ranges = optimizer._generate_parameter_ranges()

        assert param_ranges["gradient_checkpointing"] == [True]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_gradient_checkpointing_list(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        from omegaconf import ListConfig

        autotune_config = OmegaConf.create({"gradient_checkpointing": ListConfig([True, False])})
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = autotune_config
        param_ranges = optimizer._generate_parameter_ranges()

        # ListConfig is not isinstance(list), so it gets wrapped in a list
        assert param_ranges["gradient_checkpointing"] == [[True, False]]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_gradient_checkpointing_python_list(self, mock_gpu_mem, mock_recipe_cfg):
        """Test with Python list (not ListConfig) - edge case"""
        mock_gpu_mem.return_value = 80.0

        # Use a plain object with Python list attribute to reach the else branch
        class MockConfig:
            gradient_checkpointing = [True, False]

        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = MockConfig()
        param_ranges = optimizer._generate_parameter_ranges()

        # Python list is isinstance(list), so it's used directly
        assert param_ranges["gradient_checkpointing"] == [True, False]


class TestSetCpuOffload:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_cpu_offload_auto_high_memory(self, mock_gpu_mem, mock_recipe_cfg):
        """Test CPU offload when params use <50% GPU memory"""
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"cpu_offload": "auto"}
        optimizer = LlmftOptimizer(autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        # 8B model with 8 GPUs: (8e9 * 2) / 8 / 1e9 = 2GB per GPU < 50% of 80GB
        # Should default to False
        assert param_ranges["cpu_offload"] == [False]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_cpu_offload_auto_low_memory(self, mock_gpu_mem):
        """Test CPU offload when params use >50% GPU memory"""
        mock_gpu_mem.return_value = 80.0

        cfg = OmegaConf.create(
            {
                "training_config": {
                    "model_config": {"model_name_or_path": "meta-llama/Llama-3.3-70B-Instruct"},
                    "training_args": {"micro_train_batch_size": 2},
                    "datasets": {"val_data": {"limit": 100}},
                },
                "trainer": {"devices": 1, "num_nodes": 1},
            }
        )

        autotune_config = {"cpu_offload": "auto"}
        optimizer = LlmftOptimizer(autotune_config, cfg, "ml.p5.48xlarge")
        param_ranges = optimizer._generate_parameter_ranges()

        # 70B model with 1 GPU: (70e9 * 2) / 1 / 1e9 = 140GB > 50% of 80GB
        # Should test both True and False
        assert param_ranges["cpu_offload"] == [True, False]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_cpu_offload_single_value(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        autotune_config = {"cpu_offload": True}
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = OmegaConf.create(autotune_config)
        param_ranges = optimizer._generate_parameter_ranges()

        assert param_ranges["cpu_offload"] == [True]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_cpu_offload_list(self, mock_gpu_mem, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        from omegaconf import ListConfig

        autotune_config = OmegaConf.create({"cpu_offload": ListConfig([True, False])})
        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = autotune_config
        param_ranges = optimizer._generate_parameter_ranges()

        # ListConfig is not isinstance(list), so it gets wrapped in a list
        assert param_ranges["cpu_offload"] == [[True, False]]

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_set_cpu_offload_python_list(self, mock_gpu_mem, mock_recipe_cfg):
        """Test with Python list (not ListConfig) - edge case"""
        mock_gpu_mem.return_value = 80.0

        # Use a plain object with Python list attribute to reach the else branch
        class MockConfig:
            cpu_offload = [True, False]

        optimizer = LlmftOptimizer({}, mock_recipe_cfg, "ml.p5.48xlarge")
        optimizer._autotune_cfg = MockConfig()
        param_ranges = optimizer._generate_parameter_ranges()

        # Python list is isinstance(list), so it's used directly
        assert param_ranges["cpu_offload"] == [True, False]


class TestIsValidCandidate:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_valid(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 32,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        assert optimizer._is_valid_candidate(candidate) is True

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_invalid_sharding(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "INVALID",
            "train_batch_size": 32,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        assert optimizer._is_valid_candidate(candidate) is False

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_batch_size_too_small(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 1,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        assert optimizer._is_valid_candidate(candidate) is False

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_batch_size_less_than_micro_times_gpus(
        self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg
    ):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 8,  # Less than micro_train_batch_size (2) * num_gpus (8) = 16
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        assert optimizer._is_valid_candidate(candidate) is False

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_batch_size_exceeds_val_limit(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 200,  # Exceeds val_limit of 100
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        assert optimizer._is_valid_candidate(candidate) is False

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_is_valid_candidate_cpu_offload_without_checkpointing(
        self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg
    ):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 32,
            "gradient_checkpointing": False,
            "cpu_offload": True,  # Requires gradient_checkpointing=True
        }

        assert optimizer._is_valid_candidate(candidate) is False


class TestTuneCandidate:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tune_candidate_oom_no_shard(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "NO_SHARD",
            "train_batch_size": 32,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        adjusted, should_retry = optimizer.tune_candidate(candidate, ErrorCode.OOM)

        assert should_retry is True
        assert adjusted["sharding_strategy"] == "HYBRID_SHARD"

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tune_candidate_oom_hybrid_shard(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "HYBRID_SHARD",
            "train_batch_size": 32,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        adjusted, should_retry = optimizer.tune_candidate(candidate, ErrorCode.OOM)

        assert should_retry is True
        assert adjusted["sharding_strategy"] == "FULL_SHARD"

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tune_candidate_oom_reduce_batch_size(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 32,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        adjusted, should_retry = optimizer.tune_candidate(candidate, ErrorCode.OOM)

        assert should_retry is True
        assert adjusted["train_batch_size"] == 16

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tune_candidate_low_memory(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "HYBRID_SHARD",
            "train_batch_size": 16,
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        adjusted, should_retry = optimizer.tune_candidate(candidate, ErrorCode.LOW_MEMORY)

        assert should_retry is True
        assert adjusted["train_batch_size"] == 32
        assert adjusted["sharding_strategy"] == "FULL_SHARD"

    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_tune_candidate_no_adjustment(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        mock_gpu_mem.return_value = 80.0

        optimizer = LlmftOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")

        candidate = {
            "sharding_strategy": "FULL_SHARD",
            "train_batch_size": 2,  # Can't reduce further
            "gradient_checkpointing": True,
            "cpu_offload": False,
        }

        adjusted, should_retry = optimizer.tune_candidate(candidate, ErrorCode.OOM)

        assert should_retry is False
        assert adjusted == candidate
