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
Integration tests for the MTRL eval feature.

Covers end-to-end ``launch.json`` generation via the unified
``MtrlSMTJLauncher`` driving an MTRL evaluation recipe, and a smoke test of
the per-model ``RecipeCollection`` expansion performed in
``update_private_hub.py --eval``.

Working-directory note:
    The launcher uses paths rooted at the repository root
    (``./recipes_collection/recipes/...`` and ``./launcher/...``).
    An autouse fixture chdirs to the repo root so relative paths resolve
    correctly regardless of invocation cwd.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

# Resolve the repository root based on this file's location.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Make scripts.model_hub importable.
sys.path.insert(0, str(_REPO_ROOT))

from launcher.mtrl import launchers as mtrl_launchers  # noqa: E402
from launcher.mtrl.launchers import MtrlSMTJLauncher  # noqa: E402
from scripts.model_hub.update_private_hub import (  # noqa: E402
    process_mtrl_eval_recipe_metadata,
)

# Path (slug) to the single shared MTRL eval recipe.
_SHARED_RECIPE_SLUG = "evaluation/mtrl/mtrl_eval"
_SHARED_RECIPE_PATH = _REPO_ROOT / "recipes_collection" / "recipes" / "evaluation" / "mtrl" / "mtrl_eval.yaml"

# Forbidden keys in a per-model MTRL eval ``RecipeCollection`` entry.
_FORBIDDEN_ENTRY_KEYS = {
    "CustomizationTechnique",
    "HpEksPayloadTemplateS3Uri",
    "HpEksOverrideParamsS3Uri",
    "Hardware",
    "SupportedInstanceTypes",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure relative paths inside the launcher resolve at repo root."""
    monkeypatch.chdir(_REPO_ROOT)


def _contains_key(obj, key: str) -> bool:
    """Recursively check whether ``key`` appears anywhere as a dict key."""
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key(item, key) for item in obj)
    return False


def _build_cfg(tmp_path: Path) -> OmegaConf:
    """Build a staging cfg mirroring what ``get_mtrl_eval_params`` produces."""
    recipe_dict = OmegaConf.to_container(OmegaConf.load(str(_SHARED_RECIPE_PATH)), resolve=True)
    cfg_dict = {
        "base_results_dir": str(tmp_path),
        "cluster_type": "sm_jobs",
        "launch_json": True,
        "recipes": recipe_dict,
    }
    return OmegaConf.create(cfg_dict)


def _write_launch_json_stub(path: Path, metadata: dict) -> None:
    """Write a minimal launch.json stub for ``process_mtrl_eval_recipe_metadata``."""
    launch_data = {"metadata": metadata, "regional_parameters": {}}
    path.write_text(json.dumps(launch_data))


# ---------------------------------------------------------------------------
# End-to-end launch.json generation via the unified launcher
# ---------------------------------------------------------------------------


def test_mtrl_eval_launch_json_end_to_end(tmp_path: Path) -> None:
    """Drive ``MtrlSMTJLauncher`` against the shared eval recipe and assert
    the produced launch.json has the required eval-specific shape.
    """
    assert _SHARED_RECIPE_PATH.is_file(), (
        f"Shared MTRL eval recipe missing at {_SHARED_RECIPE_PATH}; "
        "expected at recipes_collection/recipes/evaluation/mtrl/mtrl_eval.yaml"
    )

    cfg = _build_cfg(tmp_path)

    with patch.object(
        mtrl_launchers,
        "get_recipe_file_path",
        return_value=_SHARED_RECIPE_SLUG,
    ):
        launcher = MtrlSMTJLauncher(cfg)
        launcher.run()

    # The launcher writes launch.json under <base_results_dir>/<job_name>/.
    job_name = cfg.recipes.run["name"]  # "mtrl-eval" in the shared YAML
    launch_json_path = tmp_path / job_name / "launch.json"
    assert launch_json_path.is_file(), f"launch.json was not produced at {launch_json_path}"

    with launch_json_path.open("r") as f:
        launch_json = json.load(f)

    # metadata.Type / EvaluationType
    metadata = launch_json["metadata"]
    assert metadata["Type"] == "Evaluation", f"metadata.Type must be 'Evaluation'; got {metadata.get('Type')!r}"
    assert (
        metadata["EvaluationType"] == "MTRLEvaluation"
    ), f"metadata.EvaluationType must be 'MTRLEvaluation'; got {metadata.get('EvaluationType')!r}"

    # regional_parameters.smtj_regional_ecr_uri populated, no k8s variant.
    regional_params = launch_json["regional_parameters"]
    smtj_ecr = regional_params.get("smtj_regional_ecr_uri")
    assert (
        isinstance(smtj_ecr, dict) and smtj_ecr
    ), f"regional_parameters.smtj_regional_ecr_uri must be a non-empty dict; got {smtj_ecr!r}"
    assert (
        "hp_eks_regional_ecr_uri" not in regional_params
    ), "MTRL eval launch.json must not carry hp_eks_regional_ecr_uri"

    # CustomizationTechnique must NOT appear anywhere in the launch.json
    # (the eval operator filters on Type == "Evaluation" instead).
    assert not _contains_key(
        launch_json, "CustomizationTechnique"
    ), "CustomizationTechnique must be absent from the MTRL eval launch.json"

    # recipe_override_parameters present.
    assert "recipe_override_parameters" in launch_json, "launch.json is missing recipe_override_parameters"


# ---------------------------------------------------------------------------
# Per-model MTRL eval expansion smoke test
# ---------------------------------------------------------------------------


def test_mtrl_eval_expansion_synthesizes_one_entry_per_trained_model(
    tmp_path: Path,
) -> None:
    """Mirror the per-model MTRL eval expansion block in
    ``update_private_hub.main`` for N=3 MTRL-trained models and assert
    ``process_mtrl_eval_recipe_metadata`` produces correct per-model entries.
    """
    mtrl_models = {
        "openai-reasoning-gpt-oss-20b",
        "huggingface-reasoning-qwen3-4b",
        "deepseek-llm-r1-distilled-llama-8b",
    }
    n = len(mtrl_models)

    # Stub a launch.json on disk (single shared recipe → single file).
    launch_json_path = tmp_path / "launch.json"
    _write_launch_json_stub(
        launch_json_path,
        metadata={
            "DisplayName": "MTRL Evaluation",
            "Name": "mtrl-eval",
            "Versions": ["1.0.0"],
        },
    )

    entries = []
    for model_id in sorted(mtrl_models):
        s3_uris = {
            "sm_jobs_yaml": f"s3://bucket/recipes/mtrl-eval-{model_id}_payload_template_sm_jobs_1.0.0.yaml",
            "sm_jobs_json": f"s3://bucket/recipes/mtrl-eval-{model_id}_override_params_sm_jobs_1.0.0.json",
        }
        stub_ecr_uri = f"123456789012.dkr.ecr.us-west-2.amazonaws.com/rft-trainer:{model_id}"

        with patch(
            "scripts.model_hub.update_private_hub.get_regional_ecr_uri",
            return_value=stub_ecr_uri,
        ):
            entry = process_mtrl_eval_recipe_metadata(
                str(launch_json_path),
                s3_uris,
                model_id=model_id,
                region="us-west-2",
                endpoint="prod",
            )
        entries.append(entry)

    # Exactly N entries from the single shared YAML.
    assert len(entries) == n, f"Expected exactly {n} MTRL eval entries; got {len(entries)}"

    # Each entry has the required shape; forbidden keys absent.
    expected_names = {f"mtrl-eval-{m}" for m in mtrl_models}
    actual_names = {e["Name"] for e in entries}
    assert actual_names == expected_names, f"Entry names must equal {expected_names!r}; got {actual_names!r}"
    assert len(actual_names) == n

    for entry in entries:
        assert entry["Type"] == "Evaluation"
        assert entry["EvaluationType"] == "MTRLEvaluation"

        for key in ("SmtjRecipeTemplateS3Uri", "SmtjOverrideParamsS3Uri", "SmtjImageUri"):
            assert key in entry, f"Missing {key} in entry for {entry['Name']}"
            assert entry[key], f"Empty {key} in entry for {entry['Name']}"

        for forbidden in _FORBIDDEN_ENTRY_KEYS:
            assert forbidden not in entry, f"Forbidden key {forbidden!r} found in entry for {entry['Name']}"
