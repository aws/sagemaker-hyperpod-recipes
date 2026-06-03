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
Tests for ``MtrlSMTJLauncher`` processor dispatch.

The unified launcher picks the template processor based on recipe path:
- ``evaluation/mtrl/*`` → ``MtrlEvalRecipeTemplateProcessor``
- anything else         → ``MtrlRecipeTemplateProcessor``

These tests exercise the ``_is_mtrl_eval_recipe_path`` helper and verify the
processor-selection branch in ``_save_hydra_config`` picks the right class.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from launcher.mtrl import launchers as mtrl_launchers
from launcher.mtrl.launchers import MtrlSMTJLauncher, _is_mtrl_eval_recipe_path
from launcher.recipe_templatization.mtrl.mtrl_recipe_template_processor import (
    MtrlRecipeTemplateProcessor,
)
from launcher.recipe_templatization.mtrl_eval.mtrl_eval_recipe_template_processor import (
    MtrlEvalRecipeTemplateProcessor,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)


class TestIsMtrlEvalRecipePath:
    """Truth table for the processor-dispatch helper."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("evaluation/mtrl/mtrl_eval", True),
            ("evaluation/mtrl/mtrl-eval-gpt-oss-20b", True),
            ("fine-tuning/gpt_oss/mtrl-gpt-oss-20b-lora", False),
            ("fine-tuning/qwen/mtrl-qwen-3-32b-lora", False),
            ("training/nova/nova_sft", False),
            ("", False),
            (None, False),
        ],
    )
    def test_path_classification(self, path, expected):
        assert _is_mtrl_eval_recipe_path(path) is expected


def _eval_recipe_dict() -> dict:
    return {
        "display_name": "MTRL Eval Fixture",
        "recipe_version": "1.0.0",
        "run": {"name": "mtrl-eval", "model_type": "mtrl"},
        "batch": {"eval_batch_size": None, "eval_group_size": 1, "eval_random_sample": False},
        "eval_sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
        "schedule": {"epochs": 1, "max_steps": 0, "eval_every": 1, "eval_at_start": True},
        "rollout": {"max_concurrency": 3, "timeout": 600.0, "max_retries": 3},
    }


def _training_recipe_dict() -> dict:
    """Minimal MTRL training recipe matching the training template's expected shape."""
    return {
        "display_name": "MTRL Training Fixture",
        "recipe_version": "1.0.0",
        "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
        "batch": {
            "training_batch_size": 128,
            "group_size": 8,
            "micro_batch_size": 32,
            "eval_batch_size": None,
            "eval_group_size": 1,
            "eval_random_sample": False,
        },
        "rl": {
            "advantage_method": "group_based",
            "advantage_normalization": True,
            "gamma": 1.0,
            "loss_fn": "ppo",
            "ppo_eps_low": 0.2,
            "ppo_eps_high": 0.2,
        },
        "sampling_params": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 12288},
        "eval_sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 12288},
        "schedule": {
            "epochs": 1,
            "max_steps": 100,
            "save_every": 10,
            "eval_every": 10,
            "eval_at_start": True,
            "seed": 42,
        },
        "model": {
            "model_name": "openai/gpt-oss-20b",
            "max_sequence_length": 32768,
            "lora_rank": 32,
            "lora_alpha": 64,
            "learning_rate": 4e-5,
        },
        "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 3},
    }


def _build_cfg(tmp_path: Path, recipe_dict: dict) -> OmegaConf:
    return OmegaConf.create(
        {
            "base_results_dir": str(tmp_path),
            "cluster_type": "sm_jobs",
            "launch_json": True,
            "recipes": recipe_dict,
        }
    )


class TestProcessorDispatch:
    """``_save_hydra_config`` picks the right processor class for each recipe path."""

    def test_eval_path_constructs_eval_processor(self, tmp_path: Path) -> None:
        """Eval recipe path ⇒ ``MtrlEvalRecipeTemplateProcessor`` is instantiated."""
        cfg = _build_cfg(tmp_path, _eval_recipe_dict())
        launcher = MtrlSMTJLauncher(cfg)

        captured = {}

        def _spy_eval(staging_cfg, platform):
            captured["cls"] = MtrlEvalRecipeTemplateProcessor
            inst = MtrlEvalRecipeTemplateProcessor(staging_cfg, platform=platform)
            return inst

        with patch.object(
            mtrl_launchers,
            "get_recipe_file_path",
            return_value="evaluation/mtrl/mtrl_eval",
        ), patch.object(
            mtrl_launchers,
            "MtrlEvalRecipeTemplateProcessor",
            side_effect=_spy_eval,
        ):
            # Swallow the subsequent process_recipe() call — we only care which
            # processor class was selected.
            try:
                launcher._prepare_output_dir()
                launcher._save_hydra_config()
            except Exception:
                pass

        assert captured.get("cls") is MtrlEvalRecipeTemplateProcessor

    def test_training_path_constructs_training_processor(self, tmp_path: Path) -> None:
        """Training recipe path ⇒ ``MtrlRecipeTemplateProcessor`` is instantiated."""
        cfg = _build_cfg(tmp_path, _training_recipe_dict())
        launcher = MtrlSMTJLauncher(cfg)

        captured = {}

        def _spy_training(staging_cfg, platform):
            captured["cls"] = MtrlRecipeTemplateProcessor
            inst = MtrlRecipeTemplateProcessor(staging_cfg, platform=platform)
            return inst

        with patch.object(
            mtrl_launchers,
            "get_recipe_file_path",
            return_value="fine-tuning/gpt_oss/mtrl-gpt-oss-20b-lora",
        ), patch.object(
            mtrl_launchers,
            "MtrlRecipeTemplateProcessor",
            side_effect=_spy_training,
        ):
            try:
                launcher._prepare_output_dir()
                launcher._save_hydra_config()
            except Exception:
                pass

        assert captured.get("cls") is MtrlRecipeTemplateProcessor

    def test_training_path_does_not_construct_eval_processor(self, tmp_path: Path) -> None:
        """Training recipe path must NOT instantiate the eval processor."""
        cfg = _build_cfg(tmp_path, _training_recipe_dict())
        launcher = MtrlSMTJLauncher(cfg)

        with patch.object(
            mtrl_launchers,
            "get_recipe_file_path",
            return_value="fine-tuning/gpt_oss/mtrl-gpt-oss-20b-lora",
        ), patch.object(
            mtrl_launchers,
            "MtrlEvalRecipeTemplateProcessor",
        ) as eval_cls_mock:
            try:
                launcher._prepare_output_dir()
                launcher._save_hydra_config()
            except Exception:
                pass

        assert (
            eval_cls_mock.call_count == 0
        ), "MtrlEvalRecipeTemplateProcessor must not be constructed for training recipes"

    def test_eval_path_does_not_construct_training_processor(self, tmp_path: Path) -> None:
        """Eval recipe path must NOT instantiate the training processor."""
        cfg = _build_cfg(tmp_path, _eval_recipe_dict())
        launcher = MtrlSMTJLauncher(cfg)

        with patch.object(
            mtrl_launchers,
            "get_recipe_file_path",
            return_value="evaluation/mtrl/mtrl_eval",
        ), patch.object(
            mtrl_launchers,
            "MtrlRecipeTemplateProcessor",
        ) as training_cls_mock:
            try:
                launcher._prepare_output_dir()
                launcher._save_hydra_config()
            except Exception:
                pass

        assert training_cls_mock.call_count == 0, "MtrlRecipeTemplateProcessor must not be constructed for eval recipes"
