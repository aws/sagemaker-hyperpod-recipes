"""Tests for .github/scripts/process_recipe_metadata.py

Regression coverage for the eval version-drift bug: eval recipes fan out per
model, so their launch.json is generated under "<stem>_<model_name>", but the
version lookup previously used the bare "<stem>" path and silently dropped the
eval entry (leaving JumpStart pinned to a stale version).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from process_recipe_metadata import (
    get_recipe_metadata,
    get_version_from_launch_json,
    main,
    parse_changed_file_entry,
)

EVAL_RECIPE_REL = "evaluation/open-source/open_source_llmaj_eval.yaml"
EVAL_STEM = "open_source_llmaj_eval"
MODEL_NAME = "deepseek-llm-r1-distill-llama-8b"
LLMFT_RECIPE_REL = "fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.yaml"
LLMFT_STEM = "llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora"


def _write_launch_json(launch_dir, dir_name, version):
    """Create <launch_dir>/<dir_name>/k8s/launch.json with the given version."""
    path = Path(launch_dir) / dir_name / "k8s"
    path.mkdir(parents=True, exist_ok=True)
    (path / "launch.json").write_text(json.dumps({"metadata": {"Versions": [version]}}))


# ── get_version_from_launch_json ─────────────────────────────────────────────


class TestGetVersionFromLaunchJson:
    def test_reads_version_at_expected_path(self, tmp_path):
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        assert get_version_from_launch_json(f"{EVAL_STEM}_{MODEL_NAME}", str(tmp_path)) == "2.0.3"

    def test_missing_file_returns_none_without_raising(self, tmp_path):
        # Regression: a missing launch.json must not raise (it previously bubbled
        # up and dropped the whole recipe entry).
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


# ── get_recipe_metadata (path selection) ─────────────────────────────────────


class TestGetRecipeMetadata:
    def test_eval_recipe_uses_per_model_dir(self, tmp_path):
        """Regression: eval version must be read from '<stem>_<model>', not '<stem>'."""
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        file_path = f"recipes_collection/recipes/{EVAL_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path), model_name=MODEL_NAME)

        assert result == {"recipe": EVAL_RECIPE_REL.replace(".yaml", ""), "version": "2.0.3"}

    def test_eval_recipe_without_model_name_misses(self, tmp_path):
        """The old (buggy) stem-only lookup finds nothing for eval recipes."""
        _write_launch_json(tmp_path, f"{EVAL_STEM}_{MODEL_NAME}", "2.0.3")
        file_path = f"recipes_collection/recipes/{EVAL_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path))  # no model_name

        assert result["version"] is None

    def test_non_eval_recipe_uses_stem_dir(self, tmp_path):
        _write_launch_json(tmp_path, LLMFT_STEM, "2.2.3")
        file_path = f"recipes_collection/recipes/{LLMFT_RECIPE_REL}"

        result = get_recipe_metadata(file_path, str(tmp_path))

        assert result == {"recipe": LLMFT_RECIPE_REL.replace(".yaml", ""), "version": "2.2.3"}


# ── main() end-to-end ────────────────────────────────────────────────────────


class TestMain:
    def _run_main(self, tmp_path, monkeypatch, changed_files, recipes_root):
        """Invoke main() with env wired up; return the parsed recipe_metadata output."""
        launch_dir = tmp_path / "launch"
        github_output = tmp_path / "github_output.txt"
        github_output.touch()

        monkeypatch.setenv("CHANGED_FILES", "\n".join(changed_files))
        monkeypatch.setenv("LAUNCH_JSON_DIR", str(launch_dir))
        monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
        # main() gates on os.path.exists(file_path); run from a root where the
        # recipe YAMLs actually live.
        monkeypatch.chdir(recipes_root)

        return launch_dir, github_output

    def _parse_output(self, github_output):
        text = github_output.read_text()
        assert "recipe_metadata<<EOF" in text
        body = text.split("recipe_metadata<<EOF\n", 1)[1].rsplit("\nEOF", 1)[0]
        return json.loads(body)

    def test_eval_recipe_version_propagates(self, tmp_path, monkeypatch):
        """The core regression: an eval change must publish its version, not drop it."""
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
        assert output["updatedRecipes"] == [
            {
                "recipe": EVAL_RECIPE_REL.replace(".yaml", ""),
                "version": "2.0.3",
                "model_name": MODEL_NAME,
            }
        ]

    def test_missing_launch_json_still_emits_entry(self, tmp_path, monkeypatch):
        """A missing launch.json yields version=None instead of crashing/dropping."""
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
        assert len(output["updatedRecipes"]) == 1
        assert output["updatedRecipes"][0]["version"] is None
