"""
Tests for .github/scripts/check_inference_coverage.py

Tests the inference (hosting) coverage checker that scans fine-tuning recipes
and reports which ones are missing hosting configuration files in
``utils/inference_configs/``.
"""

import json
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))

# Also import shared helpers to verify they work through the inference module
from check_eval_coverage import find_recipes
from check_inference_coverage import (
    check_inference_coverage,
    check_inference_coverage_for_recipes,
    get_hosting_config_path,
    load_hosting_config_stems,
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


def _write_hosting_config(configs_dir, recipe_stem, data=None):
    """Helper to write a hosting config JSON file."""
    if data is None:
        data = [{"InstanceType": "ml.g5.12xlarge", "Environment": {}}]
    path = os.path.join(configs_dir, f"hosting-{recipe_stem}.json")
    os.makedirs(configs_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ── get_hosting_config_path ──────────────────────────────────────────────────


class TestGetHostingConfigPath:
    def test_llmft_recipe(self):
        result = get_hosting_config_path(
            "recipes_collection/recipes/fine-tuning/qwen/llmft_qwen3_4b_seq4k_gpu_dpo.yaml"
        )
        assert result == "utils/inference_configs/hosting-llmft_qwen3_4b_seq4k_gpu_dpo.json"

    def test_verl_recipe(self):
        result = get_hosting_config_path(
            "recipes_collection/recipes/fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft.yaml"
        )
        assert result == "utils/inference_configs/hosting-verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft.json"

    def test_custom_configs_dir(self):
        result = get_hosting_config_path(
            "recipes_collection/recipes/fine-tuning/llama/sft_lora.yaml",
            inference_configs_dir="/custom/path",
        )
        assert result == "/custom/path/hosting-sft_lora.json"

    def test_nested_path(self):
        result = get_hosting_config_path(
            "a/b/c/d/recipe_name.yaml",
            inference_configs_dir="configs",
        )
        assert result == "configs/hosting-recipe_name.json"


# ── load_hosting_config_stems ────────────────────────────────────────────────


class TestLoadHostingConfigStems:
    def test_loads_stems(self, temp_dir):
        configs_dir = os.path.join(temp_dir, "inference_configs")
        _write_hosting_config(configs_dir, "llmft_qwen3_4b_seq4k_gpu_dpo")
        _write_hosting_config(configs_dir, "verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft")

        stems = load_hosting_config_stems(configs_dir)
        assert stems == {
            "llmft_qwen3_4b_seq4k_gpu_dpo",
            "verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft",
        }

    def test_ignores_non_hosting_files(self, temp_dir):
        configs_dir = os.path.join(temp_dir, "inference_configs")
        _write_hosting_config(configs_dir, "recipe_a")
        # Write a non-hosting file
        os.makedirs(configs_dir, exist_ok=True)
        with open(os.path.join(configs_dir, "other_file.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(configs_dir, "README.md"), "w") as f:
            f.write("# readme")

        stems = load_hosting_config_stems(configs_dir)
        assert stems == {"recipe_a"}

    def test_empty_directory(self, temp_dir):
        configs_dir = os.path.join(temp_dir, "inference_configs")
        os.makedirs(configs_dir, exist_ok=True)

        stems = load_hosting_config_stems(configs_dir)
        assert stems == set()

    def test_nonexistent_directory(self, temp_dir):
        stems = load_hosting_config_stems(os.path.join(temp_dir, "nonexistent"))
        assert stems == set()


# ── check_inference_coverage_for_recipes ─────────────────────────────────────


class TestCheckInferenceCoverageForRecipes:
    def _setup_env(self, temp_dir, recipes, model_id_map, hosting_stems):
        """Helper to set up test environment with recipes, model map, and hosting configs."""
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        configs_dir = os.path.join(temp_dir, "inference_configs")

        paths = []
        for subpath, data in recipes.items():
            path = os.path.join(recipes_dir, subpath)
            _write_yaml(path, data)
            paths.append(path)

        _write_json(model_map_path, model_id_map)

        # Create hosting config files for the given stems
        for stem in hosting_stems:
            _write_hosting_config(configs_dir, stem)

        return paths, model_map_path, configs_dir

    def test_all_covered(self, temp_dir):
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={"llama/llmft_llama3_1_8b_sft_lora.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=["llmft_llama3_1_8b_sft_lora"],
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert result == []

    def test_missing_hosting_config(self, temp_dir):
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={"llama/llmft_llama3_1_8b_sft_lora.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=[],  # No hosting config
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert len(result) == 1
        assert result[0]["js_model_id"] == "meta-llama-8b"
        assert "Hosting configuration not found" in result[0]["reason"]
        assert "hosting-llmft_llama3_1_8b_sft_lora.json" in result[0]["reason"]

    def test_missing_model_id(self, temp_dir):
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={"llama/llmft_new_model_sft_lora.yaml": {"run": {"name": "new-model"}}},
            model_id_map={},  # Model not in map
            hosting_stems=[],
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert len(result) == 1
        assert result[0]["js_model_id"] is None
        assert "jumpstart_model-id_map" in result[0]["reason"]

    def test_skips_files_without_run_name(self, temp_dir):
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={"llama/config.yaml": {"some_key": "value"}},
            model_id_map={},
            hosting_stems=[],
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert result == []

    def test_mix_of_covered_and_missing(self, temp_dir):
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={
                "llama/llmft_llama3_1_8b_sft_lora.yaml": {"run": {"name": "llama-8b"}},
                "qwen/llmft_qwen3_4b_sft_lora.yaml": {"run": {"name": "qwen-4b"}},
            },
            model_id_map={
                "llama-8b": "meta-llama-8b",
                "qwen-4b": "hf-qwen-4b",
            },
            hosting_stems=["llmft_llama3_1_8b_sft_lora"],  # Only llama covered
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert len(result) == 1
        assert result[0]["run_name"] == "qwen-4b"

    def test_only_checks_provided_files(self, temp_dir):
        """When given specific files, only those are checked."""
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        configs_dir = os.path.join(temp_dir, "inference_configs")

        covered = os.path.join(recipes_dir, "llama", "llmft_covered.yaml")
        uncovered = os.path.join(recipes_dir, "llama", "llmft_uncovered.yaml")
        _write_yaml(covered, {"run": {"name": "llama-8b"}})
        _write_yaml(uncovered, {"run": {"name": "new-model"}})
        _write_json(model_map_path, {"llama-8b": "meta-llama-8b"})
        _write_hosting_config(configs_dir, "llmft_covered")

        # Only check the covered file
        result = check_inference_coverage_for_recipes([covered], model_map_path, configs_dir)
        assert result == []

        # Only check the uncovered file — model not in map
        result = check_inference_coverage_for_recipes([uncovered], model_map_path, configs_dir)
        assert len(result) == 1

    def test_verl_recipe_covered(self, temp_dir):
        """VERL recipes with dashes in the name are correctly matched."""
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={
                "llama/verl-grpo-rlvr-llama-3-dot-1-8b-instruct-lora.yaml": {
                    "run": {"name": "verl-grpo-llama-3-dot-1-8b-instruct-lora"}
                }
            },
            model_id_map={"verl-grpo-llama-3-dot-1-8b-instruct-lora": "meta-textgeneration-llama-3-1-8b-instruct"},
            hosting_stems=["verl-grpo-rlvr-llama-3-dot-1-8b-instruct-lora"],
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert result == []

    def test_multiple_recipes_same_model_different_coverage(self, temp_dir):
        """Multiple recipes for the same model — one covered, one not."""
        paths, model_map, configs_dir = self._setup_env(
            temp_dir,
            recipes={
                "llama/llmft_llama3_1_8b_sft_lora.yaml": {"run": {"name": "llama-8b"}},
                "llama/llmft_llama3_1_8b_dpo.yaml": {"run": {"name": "llama-8b"}},
            },
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=["llmft_llama3_1_8b_sft_lora"],  # Only sft_lora covered
        )
        result = check_inference_coverage_for_recipes(paths, model_map, configs_dir)
        assert len(result) == 1
        assert "llmft_llama3_1_8b_dpo" in result[0]["reason"]


# ── check_inference_coverage (directory scan) ────────────────────────────────


class TestCheckInferenceCoverage:
    def _setup_test_env(self, temp_dir, recipes, model_id_map, hosting_stems):
        recipes_dir = os.path.join(temp_dir, "fine-tuning")
        model_map_path = os.path.join(temp_dir, "model_map.json")
        configs_dir = os.path.join(temp_dir, "inference_configs")

        for subpath, data in recipes.items():
            _write_yaml(os.path.join(recipes_dir, subpath), data)

        _write_json(model_map_path, model_id_map)

        for stem in hosting_stems:
            _write_hosting_config(configs_dir, stem)

        return recipes_dir, model_map_path, configs_dir

    def test_all_covered(self, temp_dir):
        recipes_dir, model_map, configs_dir = self._setup_test_env(
            temp_dir,
            recipes={"llama/llmft_sft.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=["llmft_sft"],
        )
        result = check_inference_coverage(recipes_dir, model_map, configs_dir)
        assert result == []

    def test_excludes_nova_recipes(self, temp_dir):
        recipes_dir, model_map, configs_dir = self._setup_test_env(
            temp_dir,
            recipes={
                "nova/sft.yaml": {"run": {"name": "my-lora-run"}},
                "llama/llmft_sft.yaml": {"run": {"name": "llama-8b"}},
            },
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=["llmft_sft"],
        )
        result = check_inference_coverage(recipes_dir, model_map, configs_dir)
        assert result == []

    def test_missing_hosting_config(self, temp_dir):
        recipes_dir, model_map, configs_dir = self._setup_test_env(
            temp_dir,
            recipes={"llama/llmft_sft.yaml": {"run": {"name": "llama-8b"}}},
            model_id_map={"llama-8b": "meta-llama-8b"},
            hosting_stems=[],  # No hosting configs
        )
        result = check_inference_coverage(recipes_dir, model_map, configs_dir)
        assert len(result) == 1
        assert result[0]["js_model_id"] == "meta-llama-8b"


# ── Integration tests with real data ────────────────────────────────────────


class TestIntegrationWithRealData:
    """Integration tests using real repo data (skipped if files don't exist)."""

    RECIPES_DIR = "recipes_collection/recipes/fine-tuning"
    MODEL_ID_MAP = "launcher/recipe_templatization/jumpstart_model-id_map.json"
    INFERENCE_CONFIGS_DIR = "utils/inference_configs"

    @pytest.fixture(autouse=True)
    def _skip_if_no_repo_data(self):
        if not all(os.path.exists(p) for p in [self.RECIPES_DIR, self.MODEL_ID_MAP, self.INFERENCE_CONFIGS_DIR]):
            pytest.skip("Real repo data not available")

    def test_real_data_runs_without_error(self):
        result = check_inference_coverage(self.RECIPES_DIR, self.MODEL_ID_MAP, self.INFERENCE_CONFIGS_DIR)
        assert isinstance(result, list)

    def test_real_data_output_format(self):
        """Inference outputs eval-compatible schema (run_names list, recipe_paths list)."""
        # check_inference_coverage returns flat dicts; main() normalizes them.
        # Here we test the raw output from the checker function.
        missing = check_inference_coverage(self.RECIPES_DIR, self.MODEL_ID_MAP, self.INFERENCE_CONFIGS_DIR)
        assert isinstance(missing, list)
        for entry in missing:
            assert "recipe_path" in entry
            assert "run_name" in entry
            assert "reason" in entry
            assert isinstance(entry["recipe_path"], str)

    def test_real_hosting_config_stems_loaded(self):
        stems = load_hosting_config_stems(self.INFERENCE_CONFIGS_DIR)
        assert len(stems) > 0
        # Verify at least one known config exists
        assert any("llmft" in s for s in stems)

    def test_real_check_for_specific_recipes(self):
        """check_inference_coverage_for_recipes on a subset works."""
        recipes = find_recipes(self.RECIPES_DIR)[:3]
        result = check_inference_coverage_for_recipes(recipes, self.MODEL_ID_MAP, self.INFERENCE_CONFIGS_DIR)
        assert isinstance(result, list)

    def test_hosting_config_path_matches_convention(self):
        """Verify the naming convention matches existing files."""
        stems = load_hosting_config_stems(self.INFERENCE_CONFIGS_DIR)
        # All stems should be non-empty
        for stem in stems:
            assert len(stem) > 0
            assert " " not in stem  # No spaces in filenames
