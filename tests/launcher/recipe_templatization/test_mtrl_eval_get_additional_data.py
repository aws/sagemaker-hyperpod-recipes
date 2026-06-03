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
Tests for ``MtrlEvalRecipeTemplateProcessor.get_additional_data``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from launcher.recipe_templatization.mtrl_eval.mtrl_eval_recipe_template_processor import (
    MtrlEvalRecipeTemplateProcessor,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)


def _base_staging_cfg():
    return OmegaConf.create(
        {
            "display_name": "MTRL Eval Fixture",
            "recipe_version": "1.0.0",
            "run": {"name": "mtrl-eval-gpt-oss-20b-lora", "model_type": "mtrl"},
            "batch": {"eval_batch_size": None, "eval_group_size": 4, "eval_random_sample": False},
            "eval_metrics_config": {"pass_k_values": [1, 2, 4], "success_threshold": 1.0},
            "eval_sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
            "schedule": {"epochs": 1, "max_steps": 0, "eval_every": 1, "eval_at_start": True},
            "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 3},
        }
    )


def _make_processor():
    return MtrlEvalRecipeTemplateProcessor(staging_cfg=_base_staging_cfg())


class TestGetAdditionalData:
    """Exercise get_additional_data lines 106, 110-112, 117."""

    def test_none_recipe_path_returns_empty_dicts(self):
        """recipe_file_path is None → return [{}, {}, {}]."""
        processor = _make_processor()
        result = processor.get_additional_data(None)
        assert result == [{}, {}, {}]

    def test_process_recipe_exception_returns_empty_dicts(self):
        """Line 110-112: process_recipe raises → return [{}, {}, {}]."""
        processor = _make_processor()

        with patch.object(processor, "process_recipe", side_effect=RuntimeError("boom")):
            result = processor.get_additional_data("evaluation/mtrl/mtrl-eval-gpt-oss-20b-lora")

        assert result == [{}, {}, {}]

    def test_container_unavailable_returns_none(self):
        """Line 117: _check_container_availability returns False → return None."""
        processor = _make_processor()

        with patch.object(processor, "process_recipe"), patch.object(
            processor, "get_recipe_metadata", return_value={"Name": "test", "Type": "Evaluation"}
        ), patch.object(processor, "_check_container_availability", return_value=False):
            result = processor.get_additional_data("evaluation/mtrl/mtrl-eval-gpt-oss-20b-lora")

        assert result is None

    def test_success_returns_metadata_overrides_regional(self):
        """Line 106+: happy path returns [metadata, overrides, regional_params]."""
        processor = _make_processor()

        fake_metadata = {"Name": "mtrl-eval-test", "Type": "Evaluation"}

        with patch.object(processor, "process_recipe"), patch.object(
            processor, "get_recipe_metadata", return_value=fake_metadata
        ), patch.object(processor, "_check_container_availability", return_value=True), patch.object(
            processor, "_resolve_constraints_using_metadata", return_value={"p": "v"}
        ), patch.object(
            processor, "_get_regional_parameters", return_value={"r": "us-east-1"}
        ):
            result = processor.get_additional_data("evaluation/mtrl/mtrl-eval-gpt-oss-20b-lora")

        assert result == [fake_metadata, {"p": "v"}, {"r": "us-east-1"}]
