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

from auto_configurator.config_optimizer.verl_optimizer import VerlOptimizer


@pytest.fixture
def mock_recipe_cfg():
    return OmegaConf.create(
        {
            "training_config": {"model_config": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"}},
            "trainer": {"devices": 8, "num_nodes": 1},
        }
    )


@pytest.fixture
def mock_autotune_config():
    return {}


class TestVerlOptimizer:
    @patch("auto_configurator.config_optimizer.base_optimizer.get_gpu_memory_gb")
    def test_init_fails_due_to_not_implemented(self, mock_gpu_mem, mock_autotune_config, mock_recipe_cfg):
        """VERL optimizer cannot be instantiated because methods are not implemented"""
        mock_gpu_mem.return_value = 80.0

        with pytest.raises(NotImplementedError):
            VerlOptimizer(mock_autotune_config, mock_recipe_cfg, "ml.p5.48xlarge")
