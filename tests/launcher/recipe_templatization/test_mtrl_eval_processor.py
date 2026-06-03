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
Tests for :class:`MtrlEvalRecipeTemplateProcessor`.

Covers tasks 2.5 - 2.10 of the mtrl-eval-recipe spec.

Working-directory note:
    ``MtrlEvalRecipeTemplateProcessor._load_template`` opens a handful of JSON
    files via relative paths rooted at ``./launcher/...``. Every test below
    chdirs to the repository root for the duration of the test via the
    ``_chdir_repo_root`` autouse fixture so the relative paths resolve
    correctly regardless of where pytest is invoked from.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from launcher.recipe_templatization.mtrl_eval.mtrl_eval_recipe_template_processor import (
    MtrlEvalRecipeTemplateProcessor,
)

# Resolve the repository root based on this file's location so tests stay stable
# irrespective of the invocation cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure relative template paths inside the processor resolve at repo root."""
    monkeypatch.chdir(_REPO_ROOT)


def _base_staging_cfg() -> dict:
    """A concrete MTRL-eval-like staging config that matches the template."""
    return OmegaConf.create(
        {
            "display_name": "MTRL Eval Fixture",
            "recipe_version": "1.0.0",
            "run": {"name": "mtrl-eval-gpt-oss-20b-lora", "model_type": "mtrl"},
            "batch": {
                "eval_batch_size": None,
                "eval_group_size": 4,
                "eval_random_sample": False,
            },
            "eval_metrics_config": {
                "pass_k_values": [1, 2, 4, 8, 16, 32],
                "success_threshold": 1.0,
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
            "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 3},
        }
    )


def _make_processor(staging_cfg=None) -> MtrlEvalRecipeTemplateProcessor:
    """Build a processor initialised from a copy of the base staging config."""
    if staging_cfg is None:
        staging_cfg = _base_staging_cfg()
    return MtrlEvalRecipeTemplateProcessor(staging_cfg=staging_cfg)


# Canonical set of MTRL eval recipe names used as the "mtrl" substring source.
_MTRL_EVAL_RECIPE_NAMES = [
    "mtrl-eval-gpt-oss-20b-lora",
    "mtrl-eval-qwen-3-4b-instruct-lora",
    "mtrl-eval-qwen-3-32b-lora",
    "mtrl-eval-nova-lite-2-0-lora",
]

# Keys expected (and ordered) in the metadata returned by get_recipe_metadata.
_EXPECTED_METADATA_KEYS = [
    "Name",
    "RecipeFilePath",
    "DisplayName",
    "Model_ID",
    "Type",
    "EvaluationType",
    "Versions",
]


# ---------------------------------------------------------------------------
# 2.5 [UT] _load_template populates all three attributes
# ---------------------------------------------------------------------------


def test_load_template_populates_all_three_attributes() -> None:
    """_load_template must populate template_data, the JS model id map, and regional params."""
    processor = _make_processor()

    assert isinstance(processor.template_data, dict)
    assert "templates" in processor.template_data
    assert "mtrl_eval" in processor.template_data["templates"]

    assert isinstance(processor.recipe_jumpstart_model_id_mapping, dict)
    assert processor.recipe_jumpstart_model_id_mapping, "JS model id map must be non-empty"

    assert isinstance(processor.regional_parameters, dict)
    assert "mtrl" in processor.regional_parameters
    assert "sm_jobs" in processor.regional_parameters["mtrl"]


# ---------------------------------------------------------------------------
# 2.6 [UT] get_recipe_template raises on None path and on missing mtrl_eval key
# ---------------------------------------------------------------------------


def test_get_recipe_template_raises_on_none_path() -> None:
    """None recipe_file_path must raise ValueError."""
    processor = _make_processor()
    templates = processor.template_data["templates"]

    with pytest.raises(ValueError):
        processor.get_recipe_template({}, templates, recipe_file_path=None)


def test_get_recipe_template_raises_on_missing_mtrl_eval_key() -> None:
    """Missing mtrl_eval key must raise ValueError."""
    processor = _make_processor()

    with pytest.raises(ValueError):
        processor.get_recipe_template({}, {}, recipe_file_path="evaluation/mtrl/mtrl-eval-gpt-oss-20b-lora")


# ---------------------------------------------------------------------------
# Helpers for the parametrized tests
# ---------------------------------------------------------------------------


def _install_recipe_config_mock(processor: MtrlEvalRecipeTemplateProcessor, recipe_cfg) -> None:
    """Monkey-patch the instance method so get_recipe_metadata doesn't touch disk."""
    processor._load_recipe_config = lambda _path: recipe_cfg  # type: ignore[method-assign]


def _arm_matched_template_group(processor: MtrlEvalRecipeTemplateProcessor) -> None:
    """Set matched_template_group as process_recipe would."""
    processor.matched_template_group = processor.template_data["templates"]["mtrl_eval"]
    processor.matched_template = processor.matched_template_group["recipe_template"]
    processor.recipe_override_parameters = copy.deepcopy(processor.matched_template_group["recipe_override_parameters"])


# Sample recipes for parametrized tests (replaces hypothesis strategies).
_SAMPLE_RECIPES = [
    {
        "display_name": "MTRL Eval GPT-OSS 20B Lora",
        "recipe_version": "1.0.0",
        "run": {"name": "mtrl-eval-gpt-oss-20b-lora", "model_type": "mtrl"},
        "batch": {"eval_batch_size": None, "eval_group_size": 4, "eval_random_sample": False},
        "eval_metrics_config": {"pass_k_values": [1, 2, 4, 8, 16, 32], "success_threshold": 1.0},
        "eval_sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4096},
        "schedule": {"epochs": 1, "max_steps": 0, "eval_every": 1, "eval_at_start": True},
        "rollout": {"max_concurrency": 96, "timeout": 600.0, "max_retries": 3},
    },
    {
        "display_name": "MTRL Eval Qwen 3 32B Lora",
        "recipe_version": "2.0.0",
        "run": {"name": "mtrl-eval-qwen-3-32b-lora", "model_type": "mtrl"},
        "batch": {"eval_batch_size": 16, "eval_group_size": 8, "eval_random_sample": True},
        "eval_metrics_config": {"pass_k_values": [1, 4, 16], "success_threshold": 0.8},
        "eval_sampling_params": {"temperature": 1.5, "top_p": 0.5, "max_tokens": 8192},
        "schedule": {"epochs": 3, "max_steps": 0, "eval_every": 5, "eval_at_start": False},
        "rollout": {"max_concurrency": 32, "timeout": 3600.0, "max_retries": 5},
    },
    {
        "display_name": "MTRL Eval Nova Lite 2.0 Lora",
        "recipe_version": "1.2.3",
        "run": {"name": "mtrl-eval-nova-lite-2-0-lora", "model_type": "mtrl"},
        "batch": {"eval_batch_size": None, "eval_group_size": 1, "eval_random_sample": False},
        "eval_metrics_config": {"pass_k_values": [1, 2], "success_threshold": 1.0},
        "eval_sampling_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512},
        "schedule": {"epochs": 1, "max_steps": 0, "eval_every": 1, "eval_at_start": True},
        "rollout": {"max_concurrency": 1, "timeout": 60.0, "max_retries": 0},
    },
]


# ---------------------------------------------------------------------------
# 2.7 Metadata identity and field omission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_dict", _SAMPLE_RECIPES, ids=lambda r: r["run"]["name"])
def test_metadata_identity_and_omission(recipe_dict: dict) -> None:
    """get_recipe_metadata returns the expected MTRL eval metadata shape.

    ``CustomizationTechnique``, ``Hardware``, and ``InstanceTypes`` must not
    appear — the eval operator filters on ``Type == "Evaluation"`` instead.
    """
    processor = _make_processor(staging_cfg=OmegaConf.create(recipe_dict))
    _arm_matched_template_group(processor)
    _install_recipe_config_mock(processor, OmegaConf.create(recipe_dict))

    recipe_file_path = f"evaluation/mtrl/{recipe_dict['run']['name']}"
    metadata = processor.get_recipe_metadata(recipe_file_path)

    assert isinstance(metadata, OrderedDict)
    assert list(metadata.keys()) == _EXPECTED_METADATA_KEYS

    assert metadata["Type"] == "Evaluation"
    assert metadata["EvaluationType"] == "MTRLEvaluation"
    assert metadata["RecipeFilePath"] == "recipes/" + recipe_file_path + ".yaml"
    assert metadata["Versions"] == [recipe_dict["recipe_version"]]

    # Forbidden keys must not appear.
    for forbidden in ("CustomizationTechnique", "Hardware", "InstanceTypes"):
        assert forbidden not in metadata


# ---------------------------------------------------------------------------
# 2.8 Template substitution fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_dict", _SAMPLE_RECIPES, ids=lambda r: r["run"]["name"])
def test_template_substitution_fidelity(recipe_dict: dict) -> None:
    """process_recipe writes placeholders at the overridable paths and preserves everything else."""
    original = copy.deepcopy(recipe_dict)
    processor = _make_processor(staging_cfg=OmegaConf.create(recipe_dict))

    recipe_file_path = f"evaluation/mtrl/{recipe_dict['run']['name']}"
    result = processor.process_recipe(recipe_file_path=recipe_file_path)

    # 1. The customer-overridable paths hold the literal {{placeholder}} strings.
    assert result["run"]["name"] == "{{name}}"
    assert result["batch"]["eval_group_size"] == "{{eval_group_size}}"
    assert result["eval_metrics_config"]["pass_k_values"] == "{{pass_k_values}}"
    assert result["eval_metrics_config"]["success_threshold"] == "{{success_threshold}}"
    assert result["eval_sampling_params"]["temperature"] == "{{temperature}}"
    assert result["eval_sampling_params"]["top_p"] == "{{sampling_top_p}}"
    assert result["eval_sampling_params"]["max_tokens"] == "{{sampling_max_tokens}}"
    assert result["rollout"]["max_concurrency"] == "{{rollout_max_concurrency}}"
    assert result["rollout"]["timeout"] == "{{rollout_timeout}}"
    assert result["rollout"]["max_retries"] == "{{rollout_max_retries}}"

    # 2. Non-templated fields flow through from the recipe YAML verbatim.
    assert result["display_name"] == original["display_name"]
    assert result["recipe_version"] == original["recipe_version"]
    assert result["run"]["model_type"] == original["run"]["model_type"]
    assert result["batch"]["eval_batch_size"] == original["batch"]["eval_batch_size"]
    assert result["batch"]["eval_random_sample"] == original["batch"]["eval_random_sample"]
    for key in ("epochs", "max_steps", "eval_every", "eval_at_start"):
        assert result["schedule"][key] == original["schedule"][key]

    # 3. YAML round-trip: serialize + reload yields a structurally-equal dict.
    serialized = yaml.safe_dump(result, default_flow_style=False)
    reloaded = yaml.safe_load(serialized)
    assert reloaded == result


# ---------------------------------------------------------------------------
# 2.9 Override default reflects recipe value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_dict", _SAMPLE_RECIPES, ids=lambda r: r["run"]["name"])
def test_override_default_reflects_recipe_value(recipe_dict: dict) -> None:
    """Each placeholder's override-parameter default equals the recipe's concrete value."""
    processor = _make_processor(staging_cfg=OmegaConf.create(recipe_dict))
    recipe_file_path = f"evaluation/mtrl/{recipe_dict['run']['name']}"
    processor.process_recipe(recipe_file_path=recipe_file_path)

    override_params = processor.recipe_override_parameters
    denylist = processor.avoid_default_val_update_attributes

    # Map override parameter keys to their expected values from the recipe.
    # Note: template uses 'temperature', 'sampling_top_p', 'sampling_max_tokens' as keys.
    path_to_param = {
        "name": recipe_dict["run"]["name"],
        "eval_group_size": recipe_dict["batch"]["eval_group_size"],
        "pass_k_values": recipe_dict["eval_metrics_config"]["pass_k_values"],
        "success_threshold": recipe_dict["eval_metrics_config"]["success_threshold"],
        "temperature": recipe_dict["eval_sampling_params"]["temperature"],
        "sampling_top_p": recipe_dict["eval_sampling_params"]["top_p"],
        "sampling_max_tokens": recipe_dict["eval_sampling_params"]["max_tokens"],
    }

    for key, expected_value in path_to_param.items():
        if key in denylist:
            continue
        assert "default" in override_params[key], f"Missing default for '{key}'"
        assert override_params[key]["default"] == expected_value, (
            f"default for '{key}' should reflect the recipe value "
            f"({expected_value!r}), got {override_params[key]['default']!r}"
        )


# ---------------------------------------------------------------------------
# 2.10 Regional parameters are SM Jobs only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_name", _MTRL_EVAL_RECIPE_NAMES)
def test_regional_parameters_sm_jobs_only(recipe_name: str) -> None:
    """_get_regional_parameters populates smtj_regional_ecr_uri and omits hp_eks."""
    assert "mtrl" in recipe_name

    processor = _make_processor()
    # Minimal metadata - _build_serverless_sku_template reads Type / Model_ID.
    recipe_metadata = {"Name": recipe_name, "Type": "Evaluation", "Model_ID": ""}

    regional = processor._get_regional_parameters(recipe_name, recipe_metadata)

    assert "smtj_regional_ecr_uri" in regional
    assert regional["smtj_regional_ecr_uri"], "smtj_regional_ecr_uri must be populated"
    assert "hp_eks_regional_ecr_uri" not in regional
