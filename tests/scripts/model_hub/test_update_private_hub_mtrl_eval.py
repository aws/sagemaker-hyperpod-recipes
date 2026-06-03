"""Tests for MTRL eval recipe helpers in update_private_hub.py.

Covers:
- ``process_mtrl_eval_recipe_metadata`` — per-model RecipeCollection entry shape
- ``get_mtrl_eval_recipe_path`` / ``MTRL_EVAL_RECIPE_PATH``
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.model_hub.update_private_hub import (  # noqa: E402
    MTRL_EVAL_RECIPE_PATH,
    get_mtrl_eval_recipe_path,
    process_mtrl_eval_recipe_metadata,
)

REQUIRED_KEYS = {
    "DisplayName",
    "Name",
    "Type",
    "Versions",
    "EvaluationType",
    "SmtjRecipeTemplateS3Uri",
    "SmtjOverrideParamsS3Uri",
    "SmtjImageUri",
}

FORBIDDEN_KEYS = {
    "CustomizationTechnique",
    "HpEksPayloadTemplateS3Uri",
    "HpEksOverrideParamsS3Uri",
    "Hardware",
    "SupportedInstanceTypes",
}


def _write_launch_json(tmp_path: Path, metadata: dict) -> str:
    """Write a minimal launch.json to tmp_path and return its path."""
    launch_data = {"metadata": metadata, "regional_parameters": {}}
    p = tmp_path / "launch.json"
    p.write_text(json.dumps(launch_data))
    return str(p)


class TestMtrlEvalRecipeCollectionEntryShape(unittest.TestCase):
    """Per-model expansion produces an entry with the required fields only."""

    def _run_entry_shape_test(self, display_name, versions, model_id, sm_jobs_yaml_uri, sm_jobs_json_uri, image_uri):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            metadata = {
                "DisplayName": display_name,
                "Name": "mtrl-eval",
                "Versions": versions,
            }
            launch_json_path = _write_launch_json(tmp_path, metadata)

            s3_uris = {
                "sm_jobs_yaml": sm_jobs_yaml_uri,
                "sm_jobs_json": sm_jobs_json_uri,
            }

            with patch(
                "scripts.model_hub.update_private_hub.get_regional_ecr_uri",
                return_value=image_uri,
            ):
                entry = process_mtrl_eval_recipe_metadata(
                    launch_json_path,
                    s3_uris,
                    model_id=model_id,
                    region="us-west-2",
                    endpoint="prod",
                )

            self.assertEqual(set(entry.keys()), REQUIRED_KEYS)
            for forbidden in FORBIDDEN_KEYS:
                self.assertNotIn(forbidden, entry)

            self.assertEqual(entry["Type"], "Evaluation")
            self.assertEqual(entry["EvaluationType"], "MTRLEvaluation")
            self.assertEqual(entry["Name"], f"mtrl-eval-{model_id}")
            self.assertEqual(entry["DisplayName"], display_name)
            self.assertEqual(entry["Versions"], versions)
            self.assertEqual(entry["SmtjRecipeTemplateS3Uri"], sm_jobs_yaml_uri)
            self.assertEqual(entry["SmtjOverrideParamsS3Uri"], sm_jobs_json_uri)
            self.assertEqual(entry["SmtjImageUri"], image_uri)

    def test_entry_shape_gpt_oss(self):
        self._run_entry_shape_test(
            display_name="MTRL Eval GPT-OSS 20B",
            versions=["1.0.0"],
            model_id="openai-reasoning-gpt-oss-20b",
            sm_jobs_yaml_uri="s3://bucket/recipes/mtrl-eval-gpt-oss-20b_payload.yaml",
            sm_jobs_json_uri="s3://bucket/recipes/mtrl-eval-gpt-oss-20b_override.json",
            image_uri="920498770698.dkr.ecr.us-west-2.amazonaws.com/mtrl-recipes:rft-trainer-v1.0.0",
        )

    def test_entry_shape_qwen(self):
        self._run_entry_shape_test(
            display_name="MTRL Eval Qwen 3 32B",
            versions=["1.2.3", "2.0.0"],
            model_id="huggingface-vlm-qwen3-6-27b",
            sm_jobs_yaml_uri="s3://my-bucket/path/to/recipe.yaml",
            sm_jobs_json_uri="s3://my-bucket/path/to/override.json",
            image_uri="839249767557.dkr.ecr.us-west-2.amazonaws.com/mtrl-recipes:rft-trainer-v1.0.0",
        )

    def test_entry_shape_nova_lite(self):
        self._run_entry_shape_test(
            display_name="MTRL Eval Nova Lite",
            versions=["3.0.0"],
            model_id="nova-textgeneration-lite-v2",
            sm_jobs_yaml_uri="s3://recipes-bucket/nova/eval.yaml",
            sm_jobs_json_uri="s3://recipes-bucket/nova/eval-override.json",
            image_uri="300869608763.dkr.ecr.us-west-2.amazonaws.com/mtrl-recipes:rft-trainer-v1.0.0",
        )


class TestDisplayNameFallback(unittest.TestCase):
    """If launch.json metadata has no DisplayName, helper falls back to a stable string."""

    def test_display_name_fallback_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            launch_json_path = _write_launch_json(tmp_path, {})
            s3_uris = {
                "sm_jobs_yaml": "s3://b/x.yaml",
                "sm_jobs_json": "s3://b/x.json",
            }
            with patch(
                "scripts.model_hub.update_private_hub.get_regional_ecr_uri",
                return_value="dummy.ecr.uri",
            ):
                entry = process_mtrl_eval_recipe_metadata(
                    launch_json_path,
                    s3_uris,
                    model_id="openai-reasoning-gpt-oss-20b",
                    region="us-west-2",
                    endpoint="prod",
                )
        self.assertEqual(entry["DisplayName"], "MTRL Evaluation")


class TestSharedRecipePath(unittest.TestCase):
    """The helper points at the single shared mtrl_eval.yaml."""

    def test_get_mtrl_eval_recipe_path_matches_module_constant(self):
        self.assertEqual(get_mtrl_eval_recipe_path(), MTRL_EVAL_RECIPE_PATH)

    def test_get_mtrl_eval_recipe_path_points_to_shared_yaml(self):
        p = get_mtrl_eval_recipe_path()
        self.assertTrue(p.endswith("evaluation/mtrl/mtrl_eval.yaml"))

    def test_shared_yaml_exists_on_disk(self):
        self.assertTrue(Path(get_mtrl_eval_recipe_path()).is_file())


class TestUniqueNamesPerModel(unittest.TestCase):
    """N distinct model_ids produce N entries with distinct Name values."""

    def test_unique_names_per_model(self):
        model_ids = [
            "openai-reasoning-gpt-oss-20b",
            "nova-textgeneration-lite-v2",
            "huggingface-vlm-gemma-4-31b-it",
            "huggingface-vlm-qwen3-6-27b",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            launch_json_path = _write_launch_json(
                tmp_path,
                {
                    "DisplayName": "MTRL Evaluation",
                    "Name": "mtrl-eval",
                    "Versions": ["1.0.0"],
                },
            )

            entries = []
            for model_id in model_ids:
                s3_uris = {
                    "sm_jobs_yaml": f"s3://bucket/recipes/mtrl-eval-{model_id}_payload_sm_jobs_1.0.0.yaml",
                    "sm_jobs_json": f"s3://bucket/recipes/mtrl-eval-{model_id}_override_sm_jobs_1.0.0.json",
                }
                with patch(
                    "scripts.model_hub.update_private_hub.get_regional_ecr_uri",
                    return_value="dummy.ecr.uri",
                ):
                    entry = process_mtrl_eval_recipe_metadata(
                        launch_json_path,
                        s3_uris,
                        model_id=model_id,
                        region="us-west-2",
                        endpoint="prod",
                    )
                entries.append(entry)

            self.assertEqual(len(entries), len(model_ids))

            expected_names = [f"mtrl-eval-{m}" for m in model_ids]
            self.assertEqual([e["Name"] for e in entries], expected_names)

            names = [e["Name"] for e in entries]
            self.assertEqual(len(set(names)), len(names))


if __name__ == "__main__":
    unittest.main()
