"""
Tests for .github/scripts/check_eval_coverage.py

Tests the eval coverage checker that scans fine-tuning recipes and reports
which ones are missing eval instance mapping data.
"""

import json
import os

# Add the .github/scripts directory to the path so we can import the module
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

from check_eval_coverage import (
    EXCLUDED_SUBDIRS,
    check_eval_coverage,
    check_eval_coverage_for_recipes,
    deduplicate_by_model,
    extract_run_name,
    filter_recipe_files,
    find_recipes,
    load_eval_instance_mapping,
    load_model_id_map,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


def _write_yaml(path, data):
    """Helper to write a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_json(path, data):
    """Helper to write a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestLoadModelIdMap:
    def test_loads_valid_json(self, temp_dir):
        path = os.path.join(temp_dir, "map.json")
        expected = {"llama-3-1-8b-instruct": "meta-textgeneration-llama-3-1-8b-instruct"}
        _write_json(path, expected)

        result = load_model_id_map(path)
        assert result == expected

    def test_raises_on_missing_file(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            load_model_id_map(os.path.join(temp_dir, "nonexistent.json"))


class TestLoadEvalInstanceMapping:
    def test_loads_model_ids(self, temp_dir):
        path = os.path.join(temp_dir, "eval.json")
        data = {
            "js_model_name_instance_mapping": {
                "meta-textgeneration-llama-3-1-8b-instruct": ["ml.g5.12xlarge"],
                "deepseek-llm-r1-distill-llama-8b": ["ml.g5.12xlarge"],
            }
        }
        _write_json(path, data)

        result = load_eval_instance_mapping(path)
        assert result == {"meta-textgeneration-llama-3-1-8b-instruct", "deepseek-llm-r1-distill-llama-8b"}

    def test_returns_empty_set_for_missing_key(self, temp_dir):
        path = os.path.join(temp_dir, "eval.json")
        _write_json(path, {"other_key": {}})

        result = load_eval_instance_mapping(path)
        assert result == set()


class TestFindRecipes:
    def test_finds_yaml_files(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        _write_yaml(os.path.join(recipes_dir, "llama", "recipe1.yaml"), {"run": {"name": "test"}})
        _write_yaml(os.path.join(recipes_dir, "qwen", "recipe2.yaml"), {"run": {"name": "test2"}})

        result = find_recipes(recipes_dir)
        assert len(result) == 2

    def test_excludes_nova_directory(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        _write_yaml(os.path.join(recipes_dir, "llama", "recipe1.yaml"), {"run": {"name": "test"}})
        _write_yaml(os.path.join(recipes_dir, "nova", "nova_recipe.yaml"), {"run": {"name": "my-run"}})

        result = find_recipes(recipes_dir)
        assert len(result) == 1
        assert not any("nova" in r for r in result)

    def test_excludes_all_excluded_subdirs(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        _write_yaml(os.path.join(recipes_dir, "llama", "recipe1.yaml"), {"run": {"name": "test"}})

        for subdir in EXCLUDED_SUBDIRS:
            _write_yaml(os.path.join(recipes_dir, subdir, "excluded.yaml"), {"run": {"name": "x"}})

        result = find_recipes(recipes_dir)
        assert len(result) == 1

    def test_returns_sorted_results(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        _write_yaml(os.path.join(recipes_dir, "qwen", "b_recipe.yaml"), {"run": {"name": "b"}})
        _write_yaml(os.path.join(recipes_dir, "llama", "a_recipe.yaml"), {"run": {"name": "a"}})

        result = find_recipes(recipes_dir)
        assert "a_recipe" in result[0]
        assert "b_recipe" in result[1]

    def test_empty_directory(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        os.makedirs(recipes_dir, exist_ok=True)

        result = find_recipes(recipes_dir)
        assert result == []


class TestFilterRecipeFiles:
    """Tests for filter_recipe_files which filters PR-changed files."""

    def test_filters_valid_recipe_files(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        path1 = os.path.join(recipes_dir, "llama", "sft.yaml")
        path2 = os.path.join(recipes_dir, "qwen", "dpo.yaml")
        _write_yaml(path1, {"run": {"name": "test"}})
        _write_yaml(path2, {"run": {"name": "test2"}})

        changed = [path1, path2]
        result = filter_recipe_files(changed, recipes_dir)
        assert len(result) == 2

    def test_excludes_nova_files(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        path_llama = os.path.join(recipes_dir, "llama", "sft.yaml")
        path_nova = os.path.join(recipes_dir, "nova", "sft.yaml")
        _write_yaml(path_llama, {"run": {"name": "test"}})
        _write_yaml(path_nova, {"run": {"name": "nova-run"}})

        changed = [path_llama, path_nova]
        result = filter_recipe_files(changed, recipes_dir)
        assert len(result) == 1
        assert "llama" in result[0]

    def test_excludes_non_yaml_files(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        yaml_path = os.path.join(recipes_dir, "llama", "sft.yaml")
        py_path = os.path.join(recipes_dir, "llama", "script.py")
        _write_yaml(yaml_path, {"run": {"name": "test"}})
        os.makedirs(os.path.dirname(py_path), exist_ok=True)
        with open(py_path, "w") as f:
            f.write("print('hello')")

        changed = [yaml_path, py_path]
        result = filter_recipe_files(changed, recipes_dir)
        assert len(result) == 1
        assert result[0].endswith(".yaml")

    def test_excludes_files_outside_recipes_dir(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        inside = os.path.join(recipes_dir, "llama", "sft.yaml")
        outside = os.path.join(temp_dir, "other", "file.yaml")
        _write_yaml(inside, {"run": {"name": "test"}})
        _write_yaml(outside, {"run": {"name": "other"}})

        changed = [inside, outside]
        result = filter_recipe_files(changed, recipes_dir)
        assert len(result) == 1
        assert "llama" in result[0]

    def test_excludes_nonexistent_files(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        os.makedirs(recipes_dir, exist_ok=True)

        changed = [os.path.join(recipes_dir, "llama", "nonexistent.yaml")]
        result = filter_recipe_files(changed, recipes_dir)
        assert result == []

    def test_handles_empty_and_whitespace(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        path = os.path.join(recipes_dir, "llama", "sft.yaml")
        _write_yaml(path, {"run": {"name": "test"}})

        changed = ["", "  ", "\n", path]
        result = filter_recipe_files(changed, recipes_dir)
        assert len(result) == 1

    def test_returns_sorted(self, temp_dir):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        path_b = os.path.join(recipes_dir, "qwen", "b.yaml")
        path_a = os.path.join(recipes_dir, "llama", "a.yaml")
        _write_yaml(path_b, {"run": {"name": "b"}})
        _write_yaml(path_a, {"run": {"name": "a"}})

        changed = [path_b, path_a]
        result = filter_recipe_files(changed, recipes_dir)
        assert "a.yaml" in result[0]
        assert "b.yaml" in result[1]

    def test_empty_input(self, temp_dir):
        result = filter_recipe_files([], temp_dir)
        assert result == []


class TestExtractRunName:
    def test_extracts_run_name(self, temp_dir):
        path = os.path.join(temp_dir, "recipe.yaml")
        _write_yaml(path, {"run": {"name": "llama-3-1-8b-instruct"}})

        assert extract_run_name(path) == "llama-3-1-8b-instruct"

    def test_returns_none_for_missing_run(self, temp_dir):
        path = os.path.join(temp_dir, "recipe.yaml")
        _write_yaml(path, {"other_key": "value"})

        assert extract_run_name(path) is None

    def test_returns_none_for_missing_name(self, temp_dir):
        path = os.path.join(temp_dir, "recipe.yaml")
        _write_yaml(path, {"run": {"other_key": "value"}})

        assert extract_run_name(path) is None

    def test_returns_none_for_nonexistent_file(self):
        assert extract_run_name("/nonexistent/path.yaml") is None

    def test_returns_none_for_invalid_yaml(self, temp_dir):
        path = os.path.join(temp_dir, "bad.yaml")
        with open(path, "w") as f:
            f.write(": : : invalid yaml [[[")

        assert extract_run_name(path) is None

    def test_logs_debug_on_nonexistent_file(self, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="check_eval_coverage"):
            result = extract_run_name("/nonexistent/path.yaml")
        assert result is None
        assert "Failed to extract run.name from /nonexistent/path.yaml" in caplog.text

    def test_logs_debug_on_invalid_yaml(self, temp_dir, caplog):
        import logging

        path = os.path.join(temp_dir, "bad.yaml")
        with open(path, "w") as f:
            f.write(": : : invalid yaml [[[")

        with caplog.at_level(logging.DEBUG, logger="check_eval_coverage"):
            result = extract_run_name(path)
        assert result is None
        assert f"Failed to extract run.name from {path}" in caplog.text


class TestCheckEvalCoverageForRecipes:
    """Tests for check_eval_coverage_for_recipes (checks specific file list)."""

    def _setup_env(self, temp_dir, recipes, model_id_map, eval_mapping):
        """Helper to set up test environment."""
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        eval_params_path = os.path.join(temp_dir, "eval_params.json")

        paths = []
        for subpath, data in recipes.items():
            path = os.path.join(recipes_dir, subpath)
            _write_yaml(path, data)
            paths.append(path)

        _write_json(model_map_path, model_id_map)
        _write_json(eval_params_path, {"js_model_name_instance_mapping": eval_mapping})

        return paths, model_map_path, eval_params_path

    def test_all_covered(self, temp_dir):
        paths, model_map, eval_params = self._setup_env(
            temp_dir,
            recipes={"llama/sft.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            eval_mapping={"meta-llama-8b": ["ml.g5.12xlarge"]},
        )
        result = check_eval_coverage_for_recipes(paths, model_map, eval_params)
        assert result == []

    def test_missing_model_id(self, temp_dir):
        paths, model_map, eval_params = self._setup_env(
            temp_dir,
            recipes={"llama/sft.yaml": {"run": {"name": "new-model"}}},
            model_id_map={},
            eval_mapping={},
        )
        result = check_eval_coverage_for_recipes(paths, model_map, eval_params)
        assert len(result) == 1
        assert result[0]["run_name"] == "new-model"
        assert "jumpstart_model-id_map" in result[0]["reason"]

    def test_missing_eval_mapping(self, temp_dir):
        paths, model_map, eval_params = self._setup_env(
            temp_dir,
            recipes={"llama/sft.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            eval_mapping={},
        )
        result = check_eval_coverage_for_recipes(paths, model_map, eval_params)
        assert len(result) == 1
        assert result[0]["js_model_id"] == "meta-llama-8b"

    def test_only_checks_provided_files(self, temp_dir):
        """When given specific files, only those are checked."""
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        eval_params_path = os.path.join(temp_dir, "eval_params.json")

        # Create two recipes: one covered, one not
        covered = os.path.join(recipes_dir, "llama", "covered.yaml")
        uncovered = os.path.join(recipes_dir, "llama", "uncovered.yaml")
        _write_yaml(covered, {"run": {"name": "llama-8b"}})
        _write_yaml(uncovered, {"run": {"name": "new-model"}})
        _write_json(model_map_path, {"llama-8b": "meta-llama-8b"})
        _write_json(eval_params_path, {"js_model_name_instance_mapping": {"meta-llama-8b": ["ml.g5.12xlarge"]}})

        # Only check the covered file
        result = check_eval_coverage_for_recipes([covered], model_map_path, eval_params_path)
        assert result == []

        # Only check the uncovered file
        result = check_eval_coverage_for_recipes([uncovered], model_map_path, eval_params_path)
        assert len(result) == 1

    def test_skips_files_without_run_name(self, temp_dir):
        paths, model_map, eval_params = self._setup_env(
            temp_dir,
            recipes={"llama/config.yaml": {"some_key": "value"}},
            model_id_map={},
            eval_mapping={},
        )
        result = check_eval_coverage_for_recipes(paths, model_map, eval_params)
        assert result == []


class TestCheckEvalCoverage:
    """Tests for check_eval_coverage (scans entire directory)."""

    def _setup_test_env(self, temp_dir, recipes, model_id_map, eval_mapping):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        eval_params_path = os.path.join(temp_dir, "eval_params.json")

        for subpath, data in recipes.items():
            _write_yaml(os.path.join(recipes_dir, subpath), data)

        _write_json(model_map_path, model_id_map)
        _write_json(eval_params_path, {"js_model_name_instance_mapping": eval_mapping})

        return recipes_dir, model_map_path, eval_params_path

    def test_all_covered(self, temp_dir):
        recipes_dir, model_map, eval_params = self._setup_test_env(
            temp_dir,
            recipes={"llama/sft.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            eval_mapping={"meta-llama-8b": ["ml.g5.12xlarge"]},
        )
        result = check_eval_coverage(recipes_dir, model_map, eval_params)
        assert result == []

    def test_excludes_nova_recipes(self, temp_dir):
        recipes_dir, model_map, eval_params = self._setup_test_env(
            temp_dir,
            recipes={
                "nova/sft.yaml": {"run": {"name": "my-lora-run"}},
                "llama/sft.yaml": {"run": {"name": "llama-8b"}},
            },
            model_id_map={"llama-8b": "meta-llama-8b"},
            eval_mapping={"meta-llama-8b": ["ml.g5.12xlarge"]},
        )
        result = check_eval_coverage(recipes_dir, model_map, eval_params)
        assert result == []

    def test_multiple_recipes_same_model(self, temp_dir):
        recipes_dir, model_map, eval_params = self._setup_test_env(
            temp_dir,
            recipes={
                "llama/sft_lora.yaml": {"run": {"name": "llama-8b"}},
                "llama/sft_fft.yaml": {"run": {"name": "llama-8b"}},
                "llama/dpo.yaml": {"run": {"name": "llama-8b"}},
            },
            model_id_map={},
            eval_mapping={},
        )
        result = check_eval_coverage(recipes_dir, model_map, eval_params)
        assert len(result) == 3

    def test_mix_of_covered_and_missing(self, temp_dir):
        recipes_dir, model_map, eval_params = self._setup_test_env(
            temp_dir,
            recipes={
                "llama/sft.yaml": {"run": {"name": "llama-8b"}},
                "deepseek/sft.yaml": {"run": {"name": "new-deepseek"}},
            },
            model_id_map={"llama-8b": "meta-llama-8b"},
            eval_mapping={"meta-llama-8b": ["ml.g5.12xlarge"]},
        )
        result = check_eval_coverage(recipes_dir, model_map, eval_params)
        assert len(result) == 1
        assert result[0]["run_name"] == "new-deepseek"


class TestDeduplicateByModel:
    def test_deduplicates_same_run_name_no_js_id(self):
        """Entries with the same run_name and no js_model_id are grouped together."""
        missing = [
            {"recipe_path": "llama/sft.yaml", "run_name": "llama-8b", "js_model_id": None, "reason": "not mapped"},
            {"recipe_path": "llama/dpo.yaml", "run_name": "llama-8b", "js_model_id": None, "reason": "not mapped"},
            {"recipe_path": "llama/fft.yaml", "run_name": "llama-8b", "js_model_id": None, "reason": "not mapped"},
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 1
        assert result[0]["run_names"] == ["llama-8b"]
        assert len(result[0]["recipe_paths"]) == 3

    def test_groups_by_js_model_id(self):
        """Different run_names mapping to the same js_model_id are grouped together."""
        missing = [
            {
                "recipe_path": "llama/sft.yaml",
                "run_name": "llama-8b",
                "js_model_id": "meta-llama-8b",
                "reason": "missing eval",
            },
            {
                "recipe_path": "llama/verl_lora.yaml",
                "run_name": "verl-grpo-llama-8b-lora",
                "js_model_id": "meta-llama-8b",
                "reason": "missing eval",
            },
            {
                "recipe_path": "llama/verl_fft.yaml",
                "run_name": "verl-grpo-llama-8b-fft",
                "js_model_id": "meta-llama-8b",
                "reason": "missing eval",
            },
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 1
        assert set(result[0]["run_names"]) == {"llama-8b", "verl-grpo-llama-8b-lora", "verl-grpo-llama-8b-fft"}
        assert result[0]["js_model_id"] == "meta-llama-8b"
        assert len(result[0]["recipe_paths"]) == 3

    def test_keeps_different_models_separate(self):
        missing = [
            {"recipe_path": "llama/sft.yaml", "run_name": "llama-8b", "js_model_id": None, "reason": "not mapped"},
            {"recipe_path": "qwen/sft.yaml", "run_name": "qwen-7b", "js_model_id": None, "reason": "not mapped"},
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 2

    def test_keeps_different_js_model_ids_separate(self):
        """Different js_model_ids are not grouped even if reasons match."""
        missing = [
            {"recipe_path": "a.yaml", "run_name": "llama-8b", "js_model_id": "meta-llama-8b", "reason": "missing eval"},
            {"recipe_path": "b.yaml", "run_name": "qwen-7b", "js_model_id": "hf-qwen-7b", "reason": "missing eval"},
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 2

    def test_keeps_different_reasons_separate(self):
        missing = [
            {"recipe_path": "a.yaml", "run_name": "llama-8b", "js_model_id": None, "reason": "reason A"},
            {"recipe_path": "b.yaml", "run_name": "llama-8b", "js_model_id": "meta-llama", "reason": "reason B"},
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 2

    def test_no_duplicate_run_names(self):
        """Same run_name appearing multiple times should only be listed once in run_names."""
        missing = [
            {"recipe_path": "a.yaml", "run_name": "llama-8b", "js_model_id": "meta-llama-8b", "reason": "missing"},
            {"recipe_path": "b.yaml", "run_name": "llama-8b", "js_model_id": "meta-llama-8b", "reason": "missing"},
        ]

        result = deduplicate_by_model(missing)
        assert len(result) == 1
        assert result[0]["run_names"] == ["llama-8b"]
        assert len(result[0]["recipe_paths"]) == 2

    def test_empty_input(self):
        assert deduplicate_by_model([]) == []


class TestIntegrationWithRealData:
    """Integration tests using real repo data (skipped if files don't exist)."""

    RECIPES_DIR = "recipes_collection/recipes/fine-tuning"
    MODEL_ID_MAP = "launcher/recipe_templatization/jumpstart_model-id_map.json"
    EVAL_PARAMS = "launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json"

    @pytest.fixture(autouse=True)
    def _skip_if_no_repo_data(self):
        if not all(os.path.exists(p) for p in [self.RECIPES_DIR, self.MODEL_ID_MAP, self.EVAL_PARAMS]):
            pytest.skip("Real repo data not available")

    def test_real_data_runs_without_error(self):
        result = check_eval_coverage(self.RECIPES_DIR, self.MODEL_ID_MAP, self.EVAL_PARAMS)
        assert isinstance(result, list)

    def test_real_data_dedup_runs(self):
        missing = check_eval_coverage(self.RECIPES_DIR, self.MODEL_ID_MAP, self.EVAL_PARAMS)
        deduped = deduplicate_by_model(missing)
        assert isinstance(deduped, list)
        for entry in deduped:
            assert "recipe_paths" in entry
            assert len(entry["recipe_paths"]) >= 1

    def test_real_find_recipes_excludes_nova(self):
        recipes = find_recipes(self.RECIPES_DIR)
        assert not any("nova" in r for r in recipes)

    def test_real_filter_recipe_files(self):
        """filter_recipe_files works on real paths."""
        recipes = find_recipes(self.RECIPES_DIR)
        # Filtering the full list back through filter should return the same files
        filtered = filter_recipe_files(recipes, self.RECIPES_DIR)
        assert len(filtered) == len(recipes)

    def test_real_check_for_specific_recipes(self):
        """check_eval_coverage_for_recipes on a subset works."""
        recipes = find_recipes(self.RECIPES_DIR)[:3]
        result = check_eval_coverage_for_recipes(recipes, self.MODEL_ID_MAP, self.EVAL_PARAMS)
        assert isinstance(result, list)
