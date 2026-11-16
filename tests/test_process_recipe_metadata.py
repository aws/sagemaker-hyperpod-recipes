"""Tests for .github/scripts/process_recipe_metadata.py

process_recipe_metadata.py owns the full pipeline that turns changed recipe
files into the per-model SQS messages: launch.json enrichment (version +
Model_ID), exclusion filtering, grouping by model, and message construction.
Its GITHUB_OUTPUT is exactly what will be sent over SQS (minus the
hyperpodBranch the workflow injects into each message afterward).

Also carries regression coverage for the eval version-drift bug: eval recipes
fan out per model, so their launch.json is generated under "<stem>_<model_name>";
the version lookup previously used the bare "<stem>" path and silently dropped
the eval entry (leaving JumpStart pinned to a stale version).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from process_recipe_metadata import (
    build_message,
    build_sqs_messages,
    filter_excluded,
    get_recipe_metadata,
    get_version_from_launch_json,
    group_by_model,
    load_exclusion_patterns,
    main,
    parse_changed_file_entry,
)

EVAL_RECIPE_REL = "evaluation/open-source/open_source_llmaj_eval.yaml"
EVAL_STEM = "open_source_llmaj_eval"
MODEL_NAME = "deepseek-llm-r1-distill-llama-8b"
LLMFT_RECIPE_REL = "fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.yaml"
LLMFT_STEM = "llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora"
MODEL_ID = "deepseek-llm-r1-distill-llama-8b"


def _write_launch_json(launch_dir, dir_name, version, model_id=MODEL_ID):
    """Create <launch_dir>/<dir_name>/k8s/launch.json with the given metadata."""
    path = Path(launch_dir) / dir_name / "k8s"
    path.mkdir(parents=True, exist_ok=True)
    metadata = {"Versions": [version]}
    if model_id is not None:
        metadata["Model_ID"] = model_id
    (path / "launch.json").write_text(json.dumps({"metadata": metadata}))


def _recipe(recipe, version, model_id):
    return {"recipe": recipe, "version": version, "modelId": model_id}


# ── get_version_from_launch_json ─────────────────────────────────────────────


class TestGetVersionFromLaunchJson:
    def test_reads_version_at_expected_path(self, tmp_path):
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        assert get_version_from_launch_json(f"{EVAL_STEM}_{MODEL_NAME}", str(tmp_path)) == "2.0.3"

    def test_missing_file_returns_none_without_raising(self, tmp_path):
        assert get_version_from_launch_json("does_not_exist", str(tmp_path)) is None

    def test_empty_launch_json_dir_returns_none(self):
        assert get_version_from_launch_json(EVAL_STEM, "") is None

    def test_no_versions_in_metadata_returns_none(self, tmp_path):
        path = tmp_path / EVAL_STEM / "k8s"
        path.mkdir(parents=True)
        (path / "launch.json").write_text(json.dumps({"metadata": {}}))
        assert get_version_from_launch_json(EVAL_STEM, str(tmp_path)) is None


# ── parse_changed_file_entry ─────────────────────────────────────────────────


class TestParseChangedFileEntry:
    def test_eval_composite_key(self):
        entry = f"recipes_collection/recipes/{EVAL_RECIPE_REL}:{MODEL_NAME}"
        file_path, model_name = parse_changed_file_entry(entry)
        assert file_path == f"recipes_collection/recipes/{EVAL_RECIPE_REL}"
        assert model_name == MODEL_NAME

    def test_non_eval_entry_has_no_model(self):
        entry = f"recipes_collection/recipes/{LLMFT_RECIPE_REL}"
        file_path, model_name = parse_changed_file_entry(entry)
        assert file_path == entry
        assert model_name is None


# ── get_recipe_metadata (enrichment) ─────────────────────────────────────────


class TestGetRecipeMetadata:
    def test_eval_recipe_uses_per_model_dir(self, tmp_path):
        """Regression: eval version must be read from '<stem>_<model>', not '<stem>'."""
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        file_path = f"recipes_collection/recipes/{EVAL_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path), model_name=MODEL_NAME)

        assert result == {
            "recipe": EVAL_RECIPE_REL.replace(".yaml", ""),
            "version": "2.0.3",
            "modelId": MODEL_ID,
        }

    def test_eval_recipe_without_model_name_misses(self, tmp_path):
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        file_path = f"recipes_collection/recipes/{EVAL_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path))  # no model_name

        assert result["version"] is None

    def test_non_eval_recipe_uses_stem_dir(self, tmp_path):
        _write_launch_json(tmp_path, LLMFT_STEM, "2.2.3")
        file_path = f"recipes_collection/recipes/{LLMFT_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path))

        assert result == {
            "recipe": LLMFT_RECIPE_REL.replace(".yaml", ""),
            "version": "2.2.3",
            "modelId": MODEL_ID,
        }

    def test_missing_model_id_yields_none(self, tmp_path):
        _write_launch_json(tmp_path, LLMFT_STEM, "2.2.3", model_id=None)
        file_path = f"recipes_collection/recipes/{LLMFT_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path))

        assert result["modelId"] is None


# ── filter_excluded ──────────────────────────────────────────────────────────


class TestFilterExcluded:
    def test_drops_matching_recipes(self):
        recipes = [
            _recipe("evaluation/nova/nova_lite_rubric_llm_judge_eval", "1.0.20", MODEL_ID),
            _recipe("fine-tuning/nova/nova_lite_rft", "1.0.19", MODEL_ID),
        ]
        kept, dropped = filter_excluded(recipes, ["rubric_llm_judge_eval$"])
        assert [r["recipe"] for r in kept] == ["fine-tuning/nova/nova_lite_rft"]
        assert dropped == ["evaluation/nova/nova_lite_rubric_llm_judge_eval"]

    def test_no_patterns_keeps_all(self):
        recipes = [_recipe("fine-tuning/x", "1.0.0", MODEL_ID)]
        kept, dropped = filter_excluded(recipes, [])
        assert kept == recipes
        assert dropped == []


# ── group_by_model ───────────────────────────────────────────────────────────


class TestGroupByModel:
    def test_two_models_two_groups(self):
        recipes = [
            _recipe("fine-tuning/nova/rft", "1.0.19", "model-a"),
            _recipe("fine-tuning/llama/sft", "2.0.0", "model-b"),
        ]
        groups, unresolved = group_by_model(recipes)
        assert list(groups.keys()) == ["model-a", "model-b"]
        assert unresolved == []

    def test_eval_and_finetuning_same_model_one_group(self):
        recipes = [
            _recipe("fine-tuning/nova/rft", "1.0.19", "model-a"),
            _recipe("evaluation/nova/rubric", "1.0.20", "model-a"),
        ]
        groups, _ = group_by_model(recipes)
        assert list(groups.keys()) == ["model-a"]
        assert groups["model-a"] == [
            {"recipe": "fine-tuning/nova/rft", "version": "1.0.19"},
            {"recipe": "evaluation/nova/rubric", "version": "1.0.20"},
        ]

    def test_missing_model_id_is_unresolved(self):
        recipes = [_recipe("a", "1", None)]
        groups, unresolved = group_by_model(recipes)
        assert groups == {}
        assert unresolved == recipes

    def test_missing_version_is_unresolved(self):
        recipes = [_recipe("a", None, "model-a")]
        groups, unresolved = group_by_model(recipes)
        assert groups == {}
        assert unresolved == recipes


# ── build_message / build_sqs_messages ───────────────────────────────────────


class TestBuildMessage:
    def test_exact_shape(self):
        msg = build_message("model-a", [{"recipe": "fine-tuning/nova/rft", "version": "1.0.19"}])
        assert msg == {
            "modelId": "model-a",
            "recipes": [{"recipe": "fine-tuning/nova/rft", "version": "1.0.19"}],
            "options": {"subscription_model": False},
        }


class TestBuildSqsMessages:
    def test_end_to_end_grouping(self):
        recipes = [
            _recipe("fine-tuning/nova/rft", "1.0.19", "model-a"),
            _recipe("evaluation/nova/rubric", "1.0.20", "model-a"),
            _recipe("fine-tuning/llama/sft", "2.0.0", "model-b"),
            _recipe("evaluation/nova/rubric_llm_judge_eval", "1.0.0", "model-a"),  # excluded
            _recipe("fine-tuning/x/broken", None, "model-c"),  # unresolved
        ]
        messages, dropped, unresolved = build_sqs_messages(recipes, ["rubric_llm_judge_eval$"])

        assert dropped == ["evaluation/nova/rubric_llm_judge_eval"]
        assert [r["recipe"] for r in unresolved] == ["fine-tuning/x/broken"]
        assert messages == [
            build_message(
                "model-a",
                [
                    {"recipe": "fine-tuning/nova/rft", "version": "1.0.19"},
                    {"recipe": "evaluation/nova/rubric", "version": "1.0.20"},
                ],
            ),
            build_message("model-b", [{"recipe": "fine-tuning/llama/sft", "version": "2.0.0"}]),
        ]


# ── load_exclusion_patterns ──────────────────────────────────────────────────


class TestLoadExclusionPatterns:
    def test_override_short_circuits(self):
        assert load_exclusion_patterns(["only-this"]) == ["only-this"]

    def test_default_file_is_a_list(self):
        # No override -> reads recipes_collection/jumpstart_exclusions.yaml.
        patterns = load_exclusion_patterns()
        assert isinstance(patterns, list)
        assert "checkpointless" in patterns


# ── main() end-to-end ────────────────────────────────────────────────────────


class TestMain:
    def _run_main(self, tmp_path, monkeypatch, changed_files, recipes_root, exclusions="__no_match__"):
        launch_dir = tmp_path / "launch"
        github_output = tmp_path / "github_output.txt"
        github_output.touch()

        monkeypatch.setenv("CHANGED_FILES", "\n".join(changed_files))
        monkeypatch.setenv("LAUNCH_JSON_DIR", str(launch_dir))
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
        # Isolate from the real exclusions yaml: a non-matching override pattern
        # keeps every recipe while exercising the override path.
        monkeypatch.setenv("EXCLUSIONS", exclusions)
        monkeypatch.chdir(recipes_root)

        return launch_dir, github_output

    def _parse_output(self, github_output):
        text = github_output.read_text()
        assert "recipe_metadata<<EOF" in text
        body = text.split("recipe_metadata<<EOF\n", 1)[1].rsplit("\nEOF", 1)[0]
        return json.loads(body)

    def test_eval_recipe_produces_message(self, tmp_path, monkeypatch):
        """The core regression: an eval change must publish its version in a message."""
        repo_root = Path(__file__).resolve().parent.parent
        eval_yaml = repo_root / "recipes_collection" / "recipes" / EVAL_RECIPE_REL
        assert eval_yaml.exists(), "eval recipe fixture moved; update EVAL_RECIPE_REL"

        launch_dir, github_output = self._run_main(
            tmp_path,
            monkeypatch,
            changed_files=[f"recipes_collection/recipes/{EVAL_RECIPE_REL}:{MODEL_NAME}"],
            recipes_root=repo_root,
        )
        _write_launch_json(launch_dir, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")

        main()

        output = self._parse_output(github_output)
        assert output == [
            build_message(MODEL_ID, [{"recipe": EVAL_RECIPE_REL.replace(".yaml", ""), "version": "2.0.3"}])
        ]

    def test_missing_launch_json_yields_no_messages(self, tmp_path, monkeypatch):
        """A missing launch.json yields an unresolved recipe and no message (no crash)."""
        repo_root = Path(__file__).resolve().parent.parent

        launch_dir, github_output = self._run_main(
            tmp_path,
            monkeypatch,
            changed_files=[f"recipes_collection/recipes/{EVAL_RECIPE_REL}:{MODEL_NAME}"],
            recipes_root=repo_root,
        )
        # Intentionally do NOT create the launch.json.

        main()

        output = self._parse_output(github_output)
        assert output == []

    def test_no_changed_files_writes_empty_array(self, tmp_path, monkeypatch):
        repo_root = Path(__file__).resolve().parent.parent
        _, github_output = self._run_main(tmp_path, monkeypatch, changed_files=[], recipes_root=repo_root)

        main()

        assert self._parse_output(github_output) == []
