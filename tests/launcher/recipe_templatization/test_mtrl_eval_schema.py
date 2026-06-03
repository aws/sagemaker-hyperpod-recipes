"""
Schema sanity tests for the MTRL eval templatization JSON files.

Asserts that both JSON files parse and contain the exact required keys,
bounds, and defaults specified in Requirement 4 (and Requirement 3.1 for
the regional parameters file).
"""

import json
from pathlib import Path

import pytest

TEMPLATE_PATH = Path("launcher/recipe_templatization/mtrl_eval/mtrl_eval_recipe_template_parameters.json")
REGIONAL_PATH = Path("launcher/recipe_templatization/mtrl_eval/mtrl_eval_regional_parameters.json")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _contains_key(obj, key) -> bool:
    """Recursively check whether `key` appears anywhere as a dict key."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, key) for item in obj)
    return False


@pytest.fixture(scope="module")
def template_data() -> dict:
    assert TEMPLATE_PATH.exists(), f"Template parameters file not found at {TEMPLATE_PATH}"
    return _load_json(TEMPLATE_PATH)


@pytest.fixture(scope="module")
def regional_data() -> dict:
    assert REGIONAL_PATH.exists(), f"Regional parameters file not found at {REGIONAL_PATH}"
    return _load_json(REGIONAL_PATH)


def test_template_parameters_file_parses():
    """Both JSON files parse as valid JSON."""
    _load_json(TEMPLATE_PATH)


def test_regional_parameters_file_parses():
    _load_json(REGIONAL_PATH)


def test_templates_mtrl_eval_group_structure(template_data):
    """Requirement 4.1: templates.mtrl_eval with the three required subsections."""
    assert "templates" in template_data
    assert "mtrl_eval" in template_data["templates"]
    group = template_data["templates"]["mtrl_eval"]
    for key in ("recipe_template", "recipe_override_parameters", "recipe_metadata_helpers"):
        assert key in group, f"Missing '{key}' in templates.mtrl_eval"


def test_override_parameters_exact_keys(template_data):
    """recipe_override_parameters contains exactly the customer-overridable keys."""
    override = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]
    expected = {
        "base_model_name",
        "model_name",
        "name",
        "eval_group_size",
        "pass_k_values",
        "success_threshold",
        "temperature",
        "sampling_top_p",
        "sampling_max_tokens",
        "rollout_max_concurrency",
        "rollout_timeout",
        "rollout_max_retries",
    }
    assert set(override.keys()) == expected


def test_eval_group_size_bounds_and_default(template_data):
    """eval_group_size min=1, max=128, default=1."""
    egs = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["eval_group_size"]
    assert egs["type"] == "integer"
    assert egs["min"] == 1
    assert egs["max"] == 128
    assert egs["default"] == 1


def test_pass_k_values_default(template_data):
    """pass_k_values type=array, default=[1, 2, 4, 8, 16, 32]."""
    pkv = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["pass_k_values"]
    assert pkv["type"] == "array"
    assert pkv["default"] == [1, 2, 4, 8, 16, 32]


def test_success_threshold_bounds_and_default(template_data):
    """success_threshold has no min/max bounds, default=1.0."""
    st = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["success_threshold"]
    assert st["type"] == "float"
    assert "min" not in st
    assert "max" not in st
    assert st["default"] == 1.0


def test_sampling_temperature_bounds_and_default(template_data):
    """sampling_temperature min=0, max=2, default=0."""
    temp = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["temperature"]
    assert temp["min"] == 0
    assert temp["max"] == 2
    assert temp["default"] == 0


def test_top_p_bounds_and_default(template_data):
    """top_p min=0, max=1, default=1."""
    top_p = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["sampling_top_p"]
    assert top_p["min"] == 0
    assert top_p["max"] == 1
    assert top_p["default"] == 1


def test_max_tokens_bounds_and_default(template_data):
    """max_tokens min=512, max=8192, default=4096."""
    max_tokens = template_data["templates"]["mtrl_eval"]["recipe_override_parameters"]["sampling_max_tokens"]
    assert max_tokens["min"] == 512
    assert max_tokens["max"] == 8192
    assert max_tokens["default"] == 4096


def test_evaluation_types_mapping(template_data):
    """recipe_metadata_helpers.evaluation_types contains the MTRLEvaluation entry."""
    helpers = template_data["templates"]["mtrl_eval"]["recipe_metadata_helpers"]
    # Map shape is {EvaluationType-enum-value: human-readable-label}, with a
    # single deterministic entry for MTRL eval.
    assert helpers["evaluation_types"] == {"MTRLEvaluation": "MTRL Evaluation"}


def test_no_customization_techniques_key_anywhere(template_data):
    """CustomizationTechnique must not appear anywhere in the MTRL eval schema.

    The eval operator filters on ``Type == "Evaluation"`` so there is no
    customization-technique field to carry.
    """
    assert not _contains_key(template_data, "customization_techniques")
    assert not _contains_key(template_data, "CustomizationTechnique")


def test_regional_parameters_mtrl_sm_jobs_only(regional_data):
    """Requirement 3.1: top-level 'mtrl' key with sm_jobs.container_image map and no k8s block."""
    assert "mtrl" in regional_data
    mtrl = regional_data["mtrl"]
    assert "sm_jobs" in mtrl
    assert "container_image" in mtrl["sm_jobs"]
    assert isinstance(mtrl["sm_jobs"]["container_image"], dict)
    assert mtrl["sm_jobs"]["container_image"], "container_image map must be non-empty"
    # No k8s block anywhere under the 'mtrl' key
    assert "k8s" not in mtrl
    assert not _contains_key(mtrl, "hp_eks")
    assert not _contains_key(mtrl, "hp_eks_regional_ecr_uri")
