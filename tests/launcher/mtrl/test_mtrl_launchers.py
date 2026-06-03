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
Coverage tests for ``launcher/mtrl/launchers.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from launcher.mtrl import launchers as mtrl_launchers
from launcher.mtrl.launchers import MtrlSMTJLauncher, get_recipe_file_path

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_REPO_ROOT)


# ---------------------------------------------------------------------------
# Lines 23-27: get_recipe_file_path()
# ---------------------------------------------------------------------------


class TestGetRecipeFilePath:
    """Exercise get_recipe_file_path() body (lines 23-27)."""

    def test_returns_recipe_path_from_hydra_overrides(self):
        """Found 'recipes' in override → return split value."""
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.overrides.task = ["recipes=training/mtrl/mtrl-gpt-oss-20b-lora"]

        with patch("launcher.mtrl.launchers.HydraConfig.get", return_value=mock_hydra_cfg):
            result = get_recipe_file_path()

        assert result == "training/mtrl/mtrl-gpt-oss-20b-lora"

    def test_raises_key_error_when_no_recipes_override(self):
        """No override contains 'recipes' → raise KeyError."""
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.overrides.task = ["cluster=k8s", "base_results_dir=/tmp"]

        with patch("launcher.mtrl.launchers.HydraConfig.get", return_value=mock_hydra_cfg):
            with pytest.raises(KeyError, match="Recipe file path not found"):
                get_recipe_file_path()

    def test_raises_key_error_when_overrides_empty(self):
        """Empty overrides list → raise KeyError."""
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.overrides.task = []

        with patch("launcher.mtrl.launchers.HydraConfig.get", return_value=mock_hydra_cfg):
            with pytest.raises(KeyError):
                get_recipe_file_path()


# ---------------------------------------------------------------------------
# Lines 68-69: _interpolate_hydra list branch
# ---------------------------------------------------------------------------


class TestInterpolateHydraList:
    """Exercise the list isinstance branch in _interpolate_hydra (lines 68-69)."""

    def test_interpolate_hydra_resolves_list_values(self, tmp_path: Path):
        """Config with a list triggers lines 68-69."""
        cfg = OmegaConf.create(
            {
                "base_results_dir": str(tmp_path),
                "cluster_type": "sm_jobs",
                "launch_json": False,
                "recipes": {
                    "run": {"name": "test-job", "model_type": "mtrl"},
                    "items": [1, 2, 3],
                },
            }
        )
        launcher = MtrlSMTJLauncher(cfg)
        launcher._interpolate_hydra()
        assert OmegaConf.to_container(cfg.recipes)["items"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Lines 99-102: _save_hydra_config success path
# Lines 104-105: _save_hydra_config exception fallback
# ---------------------------------------------------------------------------


class TestSaveHydraConfig:
    """Exercise _save_hydra_config success and exception paths."""

    def _make_cfg(self, tmp_path: Path) -> OmegaConf:
        return OmegaConf.create(
            {
                "base_results_dir": str(tmp_path),
                "cluster_type": "sm_jobs",
                "launch_json": True,
                "recipes": {
                    "run": {"name": "test-job", "model_type": "mtrl"},
                    "model": {"model_name": "test/model"},
                },
            }
        )

    def test_success_path_saves_templatized_recipe(self, tmp_path: Path):
        """Lines 99-102: process_recipe succeeds → OmegaConf.save(templatized)."""
        cfg = self._make_cfg(tmp_path)
        launcher = MtrlSMTJLauncher(cfg)
        launcher._prepare_output_dir()

        mock_processor = MagicMock()
        mock_processor.process_recipe.return_value = {"run": {"name": "templatized"}}

        with patch.object(mtrl_launchers, "get_recipe_file_path", return_value="training/mtrl/test"), patch.object(
            mtrl_launchers, "MtrlRecipeTemplateProcessor", return_value=mock_processor
        ):
            launcher._save_hydra_config()

        config_file = launcher._output_dir / "config" / "test-job_hydra.yaml"
        assert config_file.exists()
        saved = OmegaConf.load(config_file)
        assert saved.run.name == "templatized"
        assert launcher._recipe_template_processor is mock_processor

    def test_exception_fallback_saves_original_config(self, tmp_path: Path):
        """Lines 104-105: process_recipe raises → save original cfg.recipes."""
        cfg = self._make_cfg(tmp_path)
        launcher = MtrlSMTJLauncher(cfg)
        launcher._prepare_output_dir()

        mock_processor = MagicMock()
        mock_processor.process_recipe.side_effect = RuntimeError("template error")

        with patch.object(mtrl_launchers, "get_recipe_file_path", return_value="training/mtrl/test"), patch.object(
            mtrl_launchers, "MtrlRecipeTemplateProcessor", return_value=mock_processor
        ):
            launcher._save_hydra_config()

        config_file = launcher._output_dir / "config" / "test-job_hydra.yaml"
        assert config_file.exists()
        saved = OmegaConf.load(config_file)
        assert saved.run.name == "test-job"
        assert launcher._recipe_template_processor is None


# ---------------------------------------------------------------------------
# Line 117: _create_launch_json training path (non-eval get_additional_data)
# ---------------------------------------------------------------------------


class TestCreateLaunchJsonTrainingPath:
    """Exercise line 117: training processor's get_additional_data(recipe_file_path)."""

    def test_training_processor_get_additional_data_called(self, tmp_path: Path):
        """Non-eval processor → get_additional_data(recipe_file_path) with no cfg arg."""
        cfg = OmegaConf.create(
            {
                "base_results_dir": str(tmp_path),
                "cluster_type": "sm_jobs",
                "launch_json": True,
                "recipes": {"run": {"name": "test-job", "model_type": "mtrl"}},
            }
        )
        launcher = MtrlSMTJLauncher(cfg)
        launcher._prepare_output_dir()

        # Create the config file that _create_launch_json reads
        config_dir = launcher._output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "test-job_hydra.yaml"
        OmegaConf.save(cfg.recipes, config_file)
        launcher.recipe_file_path = config_file

        # Set a mock training processor (not eval)
        from launcher.recipe_templatization.mtrl.mtrl_recipe_template_processor import (
            MtrlRecipeTemplateProcessor,
        )

        mock_processor = MagicMock(spec=MtrlRecipeTemplateProcessor)
        mock_processor.get_additional_data.return_value = [
            {"Name": "test"},
            {"param": "value"},
            {"region": "us-east-1"},
        ]
        launcher._recipe_template_processor = mock_processor

        recipe_slug = "training/mtrl/test-job"

        # Create the recipe YAML file on disk that _create_launch_json loads
        recipe_yaml_path = Path(f"./recipes_collection/recipes/{recipe_slug}.yaml")
        recipe_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_yaml_path.write_text("run:\n  name: test-job\n  model_type: mtrl\n")

        try:
            with patch.object(mtrl_launchers, "get_recipe_file_path", return_value=recipe_slug):
                launcher._create_launch_json()

            mock_processor.get_additional_data.assert_called_once_with(recipe_slug)
        finally:
            if recipe_yaml_path.exists():
                recipe_yaml_path.unlink()
            try:
                recipe_yaml_path.parent.rmdir()
            except OSError:
                pass
