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

import logging
from typing import Optional

from omegaconf import OmegaConf

from launcher.nemo.constants import ROOT_DIR

from .test_utils import (
    is_job_run_name_valid_for_clusters,
    make_hydra_cfg_instance,
    validate_distributed_degrees,
)

logger = logging.getLogger(__name__)


def test_configuration_files():
    recipes_dir = ROOT_DIR / "recipes_collection/recipes"
    log_config_name = lambda name: logger.info(f"\nFailing Config File: {name}")

    for path in recipes_dir.rglob("*.yaml"):
        if not path.is_file():
            continue

        if "hydra_config" in str(path):
            continue

        # Hydra requires relative path definition
        file_path: str = "../" + str(path.relative_to(ROOT_DIR).parent)
        config = make_hydra_cfg_instance(file_path, path.name)

        # plucking values outside the method arguments substantially reduces log output on failure
        shard_degree = OmegaConf.select(config, "model.shard_degree")
        tensor_model_parallel_degree = OmegaConf.select(config, "model.tensor_model_parallel_degree")
        expert_model_parallel_degree = OmegaConf.select(config, "model.expert_model_parallel_degree")
        context_parallel_degree = OmegaConf.select(config, "model.context_parallel_degree")
        num_nodes = OmegaConf.select(config, "trainer.num_nodes")

        assert validate_distributed_degrees(
            shard_degree, tensor_model_parallel_degree, expert_model_parallel_degree, context_parallel_degree, num_nodes
        ), log_config_name(path.name)

        job_run_name: Optional[str] = config.get("run", {}).get("name")
        assert is_job_run_name_valid_for_clusters(job_run_name), log_config_name(path.name)
