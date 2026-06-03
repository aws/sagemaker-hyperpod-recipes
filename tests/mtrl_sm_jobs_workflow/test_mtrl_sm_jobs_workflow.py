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

"""
Golden baseline tests for MTRL launch.json generation (training and eval).

Follows the same pattern as tests/sm_jobs_workflow/test_sm_jobs_workflow.py:
runs the full main() flow with Hydra config, generates launch.json, and
compares against checked-in baselines.

To update baselines after intentional changes:
    GOLDEN_TEST_WRITE=1 python -m pytest tests/mtrl_sm_jobs_workflow/
"""

import logging
import os

os.environ["NEMO_LAUNCHER_DEBUG"] = "1"

from unittest.mock import patch

from omegaconf import OmegaConf

from main import main
from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)

logger = logging.getLogger(__name__)

BASELINE_ARTIFACTS_DIR = "/tests/mtrl_sm_jobs_workflow/mtrl_baseline_artifacts/with_launch_json"


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    return_value=None,
)
def test_mtrl_training_gpt_oss_20b_launch_json(mock_load_hosting):
    """Golden test for MTRL training (gpt-oss-20b-lora) launch.json."""
    artifacts_dir = create_temp_directory("test_mtrl_training_gpt_oss_20b")
    overrides = [
        "recipes=fine-tuning/gpt_oss/mtrl-gpt-oss-20b-lora",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "launch_json=true",
    ]

    cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.launch_json = True
    del cfg["hydra"]
    main(cfg)

    compare_artifacts(["/mtrl-gpt-oss-20b-lora/launch.json"], artifacts_dir, BASELINE_ARTIFACTS_DIR)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    return_value=None,
)
def test_mtrl_training_gemma_4_31b_launch_json(mock_load_hosting):
    """Golden test for MTRL training (gemma-4-31b-lora) launch.json."""
    artifacts_dir = create_temp_directory("test_mtrl_training_gemma_4_31b")
    overrides = [
        "recipes=fine-tuning/gemma/mtrl-gemma-4-31b-lora",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "launch_json=true",
    ]

    cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.launch_json = True
    del cfg["hydra"]
    main(cfg)

    compare_artifacts(["/mtrl-gemma-4-31b-lora/launch.json"], artifacts_dir, BASELINE_ARTIFACTS_DIR)


@patch(
    "launcher.recipe_templatization.base_recipe_template_processor.BaseRecipeTemplateProcessor.load_hosting_config",
    return_value=None,
)
def test_mtrl_eval_launch_json(mock_load_hosting):
    """Golden test for MTRL eval launch.json."""
    artifacts_dir = create_temp_directory("test_mtrl_eval")
    overrides = [
        "recipes=evaluation/mtrl/mtrl_eval",
        f"base_results_dir={artifacts_dir}",
        "container=test_container",
        "launch_json=true",
        "recipes.run.base_model_name=huggingface-vlm-gemma-4-31b-it",
    ]

    cfg = make_hydra_cfg_instance("../recipes_collection", "config", overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.launch_json = True
    del cfg["hydra"]
    main(cfg)

    compare_artifacts(["/mtrl-eval/launch.json"], artifacts_dir, BASELINE_ARTIFACTS_DIR)
