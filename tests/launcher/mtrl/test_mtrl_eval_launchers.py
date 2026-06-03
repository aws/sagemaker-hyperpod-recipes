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
Tests for :class:`MtrlSMTJLauncher` driving MTRL **evaluation** recipes.

The MTRL launcher is a single class that dispatches to the eval or training
template processor based on the recipe file path. These tests exercise the
eval dispatch path (``evaluation/mtrl/*`` recipes) and its associated
``launch.json`` shape / error semantics.

Working-directory note:
    The launcher uses relative paths rooted at the repository root
    (``./recipes_collection/recipes/<path>.yaml`` and ``./launcher/...``).
    Every test chdirs to the repo root via the ``_chdir_repo_root`` autouse
    fixture so the relative paths resolve correctly regardless of invocation cwd.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from launcher.mtrl import launchers as mtrl_launchers
from launcher.mtrl.launchers import MtrlSMTJLauncher

# Resolve the repository root based on this file's location.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Directory where MTRL eval recipe YAMLs live for the launcher to resolve.
_MTRL_EVAL_RECIPES_DIR = _REPO_ROOT / "recipes_collection" / "recipes" / "evaluation" / "mtrl"

# A unique test-only recipe stem so we don't collide with real recipes.
_TEST_RECIPE_STEM = "mtrl-eval-gpt-oss-20b-lora-test-fixture"
_TEST_RECIPE_REL_PATH = f"evaluation/mtrl/{_TEST_RECIPE_STEM}"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure relative paths inside the launcher resolve at repo root."""
    monkeypatch.chdir(_REPO_ROOT)


def _base_recipe_dict(run_name: str = _TEST_RECIPE_STEM) -> dict:
    """A valid MTRL eval recipe dict matching the template placeholders."""
    return {
        "display_name": "MTRL Eval Fixture",
        "recipe_version": "1.0.0",
        "run": {"name": run_name, "model_type": "mtrl"},
        "batch": {
            "eval_batch_size": None,
            "eval_group_size": 1,
            "eval_random_sample": False,
        },
        "eval_sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
        "schedule": {
            "epochs": 1,
            "max_steps": 0,
            "eval_every": 1,
            "eval_at_start": True,
        },
        "rollout": {"max_concurrency": 3, "timeout": 600.0, "max_retries": 3},
    }


def _build_cfg(tmp_path: Path, recipe_dict: dict | None = None, *, launch_json=True) -> OmegaConf:
    """Build a staging config around the given recipe dict."""
    if recipe_dict is None:
        recipe_dict = _base_recipe_dict()
    cfg_dict = {
        "base_results_dir": str(tmp_path),
        "cluster_type": "sm_jobs",
        "recipes": recipe_dict,
    }
    if launch_json is not None:
        cfg_dict["launch_json"] = launch_json
    return OmegaConf.create(cfg_dict)


@pytest.fixture
def recipe_file_on_disk():
    """Create a temporary MTRL eval recipe YAML under the repo's recipes dir.

    The launcher reads ``./recipes_collection/recipes/<path>.yaml`` directly
    via ``OmegaConf.load`` and via the Hydra-backed ``_load_recipe_config``.
    The file is written fresh per test and removed afterwards.
    """
    created_dir = False
    if not _MTRL_EVAL_RECIPES_DIR.exists():
        _MTRL_EVAL_RECIPES_DIR.mkdir(parents=True, exist_ok=True)
        created_dir = True

    written_files: list[Path] = []

    def _write(recipe_dict: dict, stem: str = _TEST_RECIPE_STEM) -> str:
        """Write the recipe YAML and return the path slug used by the launcher."""
        target = _MTRL_EVAL_RECIPES_DIR / f"{stem}.yaml"
        target.write_text(yaml.safe_dump(recipe_dict, default_flow_style=False))
        written_files.append(target)
        return f"evaluation/mtrl/{stem}"

    yield _write

    for f in written_files:
        if f.exists():
            f.unlink()
    # Only remove the dir if this fixture created it AND it's empty.
    if created_dir and _MTRL_EVAL_RECIPES_DIR.exists():
        try:
            _MTRL_EVAL_RECIPES_DIR.rmdir()
        except OSError:
            pass


@pytest.fixture
def patched_recipe_path():
    """Patch ``get_recipe_file_path`` in the launcher module."""
    from contextlib import contextmanager

    @contextmanager
    def _patch(slug: str):
        with patch.object(mtrl_launchers, "get_recipe_file_path", return_value=slug):
            yield

    return _patch


# ---------------------------------------------------------------------------
# run() raises ValueError when launch_json is False / missing / None
# ---------------------------------------------------------------------------


_EXPECTED_ERROR_MSG = "MTRL recipes only support launch.json generation. Please run with launch_json=true"


@pytest.mark.parametrize(
    "launch_json_value,description",
    [
        (False, "launch_json=False"),
        (None, "launch_json=None"),
        ("__MISSING__", "launch_json key missing from cfg"),
    ],
)
def test_run_raises_value_error_when_launch_json_not_true(tmp_path: Path, launch_json_value, description: str) -> None:
    """``run()`` must raise ValueError with the expected message."""
    if launch_json_value == "__MISSING__":
        cfg = _build_cfg(tmp_path, launch_json=None)  # omit key entirely
        if "launch_json" in cfg:
            del cfg["launch_json"]
    else:
        cfg = _build_cfg(tmp_path, launch_json=launch_json_value)

    launcher = MtrlSMTJLauncher(cfg)

    with pytest.raises(ValueError) as excinfo:
        launcher.run()

    assert _EXPECTED_ERROR_MSG in str(excinfo.value), (
        f"Expected message substring missing for case '{description}': " f"got {str(excinfo.value)!r}"
    )


# ---------------------------------------------------------------------------
# Construction does not touch the filesystem
# ---------------------------------------------------------------------------


def test_construction_does_not_touch_filesystem(tmp_path: Path) -> None:
    """Constructing the launcher must not create any files or dirs in base_results_dir."""
    assert list(tmp_path.iterdir()) == []

    cfg = _build_cfg(tmp_path, launch_json=True)
    launcher = MtrlSMTJLauncher(cfg)

    assert list(tmp_path.iterdir()) == [], (
        "Constructor wrote to the filesystem; found: " f"{[p.name for p in tmp_path.iterdir()]}"
    )

    assert launcher._job_name == _TEST_RECIPE_STEM
    assert launcher._output_dir == tmp_path / _TEST_RECIPE_STEM
    assert launcher._launch_json is True
    assert launcher.recipe_file_path is None
    assert launcher._recipe_template_processor is None


# ---------------------------------------------------------------------------
# launch.json shape and contents (eval dispatch path)
# ---------------------------------------------------------------------------

_REQUIRED_LAUNCH_JSON_KEYS = {
    "metadata",
    "recipe_override_parameters",
    "regional_parameters",
    "training_recipe.yaml",
    "training_recipe.json",
}

_SAMPLE_RECIPES = [
    _base_recipe_dict(_TEST_RECIPE_STEM),
    {
        "display_name": "MTRL Eval PBT Fixture",
        "recipe_version": "1.0.0",
        "run": {"name": f"{_TEST_RECIPE_STEM}-v2", "model_type": "mtrl"},
        "batch": {"eval_batch_size": 32, "eval_group_size": 8, "eval_random_sample": True},
        "eval_sampling_params": {"temperature": 1.5, "top_p": 0.5, "max_tokens": 8192},
        "schedule": {"epochs": 3, "max_steps": 0, "eval_every": 5, "eval_at_start": False},
        "rollout": {"max_concurrency": 50, "timeout": 3600.0, "max_retries": 5},
    },
    {
        "display_name": "MTRL Eval Alt Fixture",
        "recipe_version": "2.0.0",
        "run": {"name": f"{_TEST_RECIPE_STEM}-alt", "model_type": "mtrl"},
        "batch": {"eval_batch_size": None, "eval_group_size": 16, "eval_random_sample": False},
        "eval_sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "schedule": {"epochs": 1, "max_steps": 0, "eval_every": 1, "eval_at_start": True},
        "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 0},
    },
]


@pytest.mark.parametrize("recipe_dict", _SAMPLE_RECIPES, ids=lambda r: r["run"]["name"])
def test_launch_json_shape_and_contents(
    recipe_dict: dict,
    tmp_path: Path,
    recipe_file_on_disk,
    patched_recipe_path,
) -> None:
    """``run()`` emits a launch.json with the required five top-level keys and
    faithful YAML / JSON recipe payloads when driven against an eval recipe.
    """
    slug = recipe_file_on_disk(recipe_dict, stem=recipe_dict["run"]["name"])

    cfg = _build_cfg(tmp_path, recipe_dict, launch_json=True)

    with patched_recipe_path(slug):
        launcher = MtrlSMTJLauncher(cfg)
        launcher.run()

    output_dir = tmp_path / recipe_dict["run"]["name"]
    launch_json_path = output_dir / "launch.json"
    assert launch_json_path.is_file(), "launch.json was not produced"

    with launch_json_path.open("r") as f:
        launch_json = json.load(f)

    assert (
        set(launch_json.keys()) >= _REQUIRED_LAUNCH_JSON_KEYS
    ), f"Missing required top-level launch.json keys; got {sorted(launch_json.keys())}"

    hydra_config_path = output_dir / "config" / f"{recipe_dict['run']['name']}_hydra.yaml"
    assert hydra_config_path.is_file(), "Templatized hydra config was not saved"
    assert (
        launch_json["training_recipe.yaml"] == hydra_config_path.read_text()
    ), "training_recipe.yaml in launch.json must equal the on-disk hydra config"

    full_recipe_path = f"./recipes_collection/recipes/{slug}.yaml"
    original = OmegaConf.load(full_recipe_path)
    expected_json = OmegaConf.to_container(OmegaConf.create({"recipes": original}), resolve=True)
    assert (
        launch_json["training_recipe.json"] == expected_json
    ), "training_recipe.json must equal OmegaConf.to_container(wrapped original, resolve=True)"


# ---------------------------------------------------------------------------
# Output directory clean-start
# ---------------------------------------------------------------------------


def test_output_directory_clean_start(
    tmp_path: Path,
    recipe_file_on_disk,
    patched_recipe_path,
) -> None:
    """Pre-existing files in ``_output_dir`` must be removed by ``run()``."""
    recipe_dict = _base_recipe_dict()
    slug = recipe_file_on_disk(recipe_dict, stem=recipe_dict["run"]["name"])

    output_dir = tmp_path / recipe_dict["run"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel_paths = []
    for name in ["alpha", "bravo", "charlie"]:
        p = output_dir / f"sentinel_{name}.txt"
        p.write_text("stale artifact")
        sentinel_paths.append(p)

    for p in sentinel_paths:
        assert p.exists()

    cfg = _build_cfg(tmp_path, recipe_dict, launch_json=True)

    with patched_recipe_path(slug):
        launcher = MtrlSMTJLauncher(cfg)
        launcher.run()

    for p in sentinel_paths:
        assert not p.exists(), f"Stale sentinel survived run(): {p}"

    assert (output_dir / "launch.json").is_file()


# ---------------------------------------------------------------------------
# get_additional_data == None branch: log warning and emit no launch.json
# ---------------------------------------------------------------------------


def test_get_additional_data_none_logs_warning_and_skips_launch_json(
    tmp_path: Path,
    recipe_file_on_disk,
    patched_recipe_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If ``get_additional_data`` returns None, the launcher logs a warning
    and does NOT write ``launch.json``.
    """
    recipe_dict = _base_recipe_dict()
    slug = recipe_file_on_disk(recipe_dict)

    cfg = _build_cfg(tmp_path, recipe_dict, launch_json=True)

    with patched_recipe_path(slug), patch(
        "launcher.recipe_templatization.mtrl_eval.mtrl_eval_recipe_template_processor."
        "MtrlEvalRecipeTemplateProcessor.get_additional_data",
        return_value=None,
    ):
        launcher = MtrlSMTJLauncher(cfg)
        with caplog.at_level(logging.WARNING, logger=mtrl_launchers.logger.name):
            launcher.run()

    output_dir = tmp_path / recipe_dict["run"]["name"]
    assert not (
        output_dir / "launch.json"
    ).exists(), "launch.json should not be emitted when get_additional_data returns None"

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Expected a warning log when get_additional_data returns None"
    assert any(
        "regional parameters" in r.getMessage().lower() or "skipping" in r.getMessage().lower() for r in warning_records
    ), f"Warning message should mention regional parameters / skipping; got {[r.getMessage() for r in warning_records]}"
