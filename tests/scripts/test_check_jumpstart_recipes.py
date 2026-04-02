import re
import unittest
from unittest.mock import mock_open, patch

from scripts.check_jumpstart_recipes import (
    get_available_recipe_paths_for_region,
    get_highest_version_spec_keys,
    is_excluded,
    load_exclusion_patterns,
    main,
    parse_version,
    print_table,
)


class TestLoadExclusionPatterns(unittest.TestCase):
    @patch(
        "builtins.open",
        mock_open(read_data='exclusion_patterns:\n  - "^training/"\n  - "checkpointless"\n'),
    )
    @patch("os.path.exists", return_value=True)
    def test_loads_patterns(self, _):
        patterns = load_exclusion_patterns()
        self.assertEqual(len(patterns), 2)
        self.assertIsInstance(patterns[0], re.Pattern)

    @patch("os.path.exists", return_value=False)
    def test_missing_file_returns_empty(self, _):
        patterns = load_exclusion_patterns()
        self.assertEqual(patterns, [])

    @patch(
        "builtins.open",
        mock_open(read_data="exclusion_patterns:\n"),
    )
    @patch("os.path.exists", return_value=True)
    def test_empty_patterns_returns_empty(self, _):
        patterns = load_exclusion_patterns()
        self.assertEqual(patterns, [])

    @patch(
        "builtins.open",
        mock_open(read_data='exclusion_patterns:\n  - "[invalid"\n  - "valid"\n'),
    )
    @patch("os.path.exists", return_value=True)
    def test_invalid_regex_skipped(self, _):
        patterns = load_exclusion_patterns()
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].pattern, "valid")


class TestIsExcluded(unittest.TestCase):
    def setUp(self):
        self.patterns = [re.compile(p) for p in [r"^training/(?!nova/)", r"checkpointless", r"llama4_scout"]]

    def test_excluded_training(self):
        self.assertTrue(is_excluded("training/custom_model/falcon", self.patterns))

    def test_not_excluded_training_nova(self):
        self.assertFalse(is_excluded("training/nova/nova_1_0/nova_lite/CPT/pretrain", self.patterns))

    def test_excluded_checkpointless(self):
        self.assertTrue(is_excluded("fine-tuning/llama/checkpointless_llama3_70b_lora", self.patterns))

    def test_excluded_llama4_scout(self):
        self.assertTrue(
            is_excluded("fine-tuning/llama/llmft_llama4_scout_17b_16e_instruct_seq4k_gpu_dpo", self.patterns)
        )

    def test_not_excluded_normal_recipe(self):
        self.assertFalse(is_excluded("fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora", self.patterns))

    def test_empty_patterns_never_excludes(self):
        self.assertFalse(is_excluded("training/anything", []))


class TestPrintTableErrors(unittest.TestCase):
    def _run_table(self, recipe_ids, region_paths, exclusion_patterns):
        return print_table(
            all_recipe_ids=recipe_ids,
            region_recipe_paths=region_paths,
            regions=["us-west-2"],
            recipe_to_js_model={},
            exclusion_patterns=exclusion_patterns,
        )

    def test_present_and_excluded_is_error(self):
        patterns = [re.compile("excluded_recipe")]
        errors = self._run_table(
            ["excluded_recipe"],
            {"us-west-2": {"recipes/excluded_recipe.yaml"}},
            patterns,
        )
        self.assertEqual(errors, ["excluded_recipe"])

    def test_present_not_excluded_no_error(self):
        errors = self._run_table(
            ["good_recipe"],
            {"us-west-2": {"recipes/good_recipe.yaml"}},
            [],
        )
        self.assertEqual(errors, [])

    def test_missing_excluded_no_error(self):
        patterns = [re.compile("excluded_recipe")]
        errors = self._run_table(
            ["excluded_recipe"],
            {"us-west-2": set()},
            patterns,
        )
        self.assertEqual(errors, [])

    def test_missing_not_excluded_no_error(self):
        errors = self._run_table(
            ["missing_recipe"],
            {"us-west-2": set()},
            [],
        )
        self.assertEqual(errors, [])

    def test_multiple_model_groups(self):
        """Covers the group separator branch when JS model changes between rows."""
        errors = print_table(
            all_recipe_ids=["recipe_a", "recipe_b"],
            region_recipe_paths={"us-west-2": {"recipes/recipe_a.yaml", "recipes/recipe_b.yaml"}},
            regions=["us-west-2"],
            recipe_to_js_model={"recipe_a": "model-1", "recipe_b": "model-2"},
            exclusion_patterns=[],
        )
        self.assertEqual(errors, [])


class TestGetHighestVersionSpecKeys(unittest.TestCase):
    def test_picks_highest_version(self):
        manifest = [
            {"model_id": "model-a", "version": "1.0.0", "spec_key": "old"},
            {"model_id": "model-a", "version": "2.0.0", "spec_key": "new"},
        ]
        result = get_highest_version_spec_keys(manifest, {"model-a"})
        self.assertEqual(result, {"model-a": "new"})

    def test_skips_deprecated(self):
        manifest = [
            {"model_id": "model-a", "version": "2.0.0", "spec_key": "deprecated_spec", "deprecated": True},
            {"model_id": "model-a", "version": "1.0.0", "spec_key": "good_spec"},
        ]
        result = get_highest_version_spec_keys(manifest, {"model-a"})
        self.assertEqual(result, {"model-a": "good_spec"})

    def test_ignores_unrelated_models(self):
        manifest = [
            {"model_id": "model-a", "version": "1.0.0", "spec_key": "spec-a"},
            {"model_id": "model-b", "version": "1.0.0", "spec_key": "spec-b"},
        ]
        result = get_highest_version_spec_keys(manifest, {"model-a"})
        self.assertEqual(result, {"model-a": "spec-a"})

    def test_empty_manifest(self):
        result = get_highest_version_spec_keys([], {"model-a"})
        self.assertEqual(result, {})

    def test_no_matching_models(self):
        manifest = [{"model_id": "model-b", "version": "1.0.0", "spec_key": "spec-b"}]
        result = get_highest_version_spec_keys(manifest, {"model-a"})
        self.assertEqual(result, {})

    def test_unparseable_version_falls_back(self):
        manifest = [
            {"model_id": "model-a", "version": "bad", "spec_key": "fallback"},
            {"model_id": "model-a", "version": "1.0.0", "spec_key": "good"},
        ]
        result = get_highest_version_spec_keys(manifest, {"model-a"})
        self.assertEqual(result, {"model-a": "good"})


class TestParseVersion(unittest.TestCase):
    def test_valid_version(self):
        self.assertEqual(parse_version("1.2.3"), (1, 2, 3))

    def test_invalid_version(self):
        self.assertEqual(parse_version("bad"), (0,))

    def test_none_version(self):
        self.assertEqual(parse_version(None), (0,))


class TestGetAvailableRecipePathsForRegion(unittest.TestCase):
    MODULE = "scripts.check_jumpstart_recipes"

    @patch(f"{MODULE}.fetch_json")
    def test_manifest_404_returns_empty(self, mock_fetch):
        mock_fetch.return_value = None
        paths, mapping = get_available_recipe_paths_for_region("us-west-2", {"model-a"})
        self.assertEqual(paths, set())
        self.assertEqual(mapping, {})

    @patch(f"{MODULE}.fetch_json")
    def test_collects_recipe_paths(self, mock_fetch):
        manifest = [{"model_id": "model-a", "version": "1.0.0", "spec_key": "specs/model-a.json"}]
        spec = {
            "recipe_collection": [
                {"recipe_file_path": "recipes/fine-tuning/llama/recipe_a.yaml"},
                {"recipe_file_path": "recipes/fine-tuning/llama/recipe_b.yaml"},
            ]
        }
        mock_fetch.side_effect = [manifest, spec]
        paths, mapping = get_available_recipe_paths_for_region("us-west-2", {"model-a"})
        self.assertEqual(paths, {"recipes/fine-tuning/llama/recipe_a.yaml", "recipes/fine-tuning/llama/recipe_b.yaml"})
        self.assertEqual(mapping["recipes/fine-tuning/llama/recipe_a.yaml"], "model-a")

    @patch(f"{MODULE}.fetch_json")
    def test_spec_404_skipped(self, mock_fetch):
        manifest = [{"model_id": "model-a", "version": "1.0.0", "spec_key": "specs/model-a.json"}]
        mock_fetch.side_effect = [manifest, None]
        paths, mapping = get_available_recipe_paths_for_region("us-west-2", {"model-a"})
        self.assertEqual(paths, set())

    @patch(f"{MODULE}.fetch_json")
    def test_spec_fetch_error_skipped(self, mock_fetch):
        manifest = [{"model_id": "model-a", "version": "1.0.0", "spec_key": "specs/model-a.json"}]
        mock_fetch.side_effect = [manifest, Exception("network error")]
        paths, mapping = get_available_recipe_paths_for_region("us-west-2", {"model-a"})
        self.assertEqual(paths, set())

    @patch(f"{MODULE}.fetch_json")
    def test_warns_on_missing_model_ids(self, mock_fetch):
        manifest = [{"model_id": "model-a", "version": "1.0.0", "spec_key": "specs/model-a.json"}]
        spec = {"recipe_collection": []}
        mock_fetch.side_effect = [manifest, spec]
        # model-b is requested but not in manifest
        paths, _ = get_available_recipe_paths_for_region("us-west-2", {"model-a", "model-b"})
        self.assertEqual(paths, set())


class TestMain(unittest.TestCase):
    MODULE = "scripts.check_jumpstart_recipes"

    def _make_recipe(self, recipe_id, run_name):
        r = unittest.mock.MagicMock()
        r.recipe_id = recipe_id
        r.config.run.name = run_name
        return r

    @patch(f"{MODULE}.get_available_recipe_paths_for_region")
    @patch(f"{MODULE}.load_exclusion_patterns")
    @patch("builtins.open", mock_open(read_data='{"model-a": "js-model-a"}'))
    @patch(f"{MODULE}.list_recipes")
    def test_no_errors_exits_cleanly(self, mock_recipes, mock_excl, mock_region):
        mock_recipes.return_value = [self._make_recipe("fine-tuning/llama/recipe_a", "model-a")]
        mock_excl.return_value = []
        mock_region.return_value = (
            {"recipes/fine-tuning/llama/recipe_a.yaml"},
            {"recipes/fine-tuning/llama/recipe_a.yaml": "js-model-a"},
        )

        # Should not raise SystemExit
        main()

    @patch(f"{MODULE}.get_available_recipe_paths_for_region")
    @patch(f"{MODULE}.load_exclusion_patterns")
    @patch("builtins.open", mock_open(read_data='{"model-a": "js-model-a"}'))
    @patch(f"{MODULE}.list_recipes")
    def test_present_but_excluded_exits_with_error(self, mock_recipes, mock_excl, mock_region):
        mock_recipes.return_value = [self._make_recipe("fine-tuning/llama/recipe_a", "model-a")]
        mock_excl.return_value = [re.compile("recipe_a")]
        mock_region.return_value = (
            {"recipes/fine-tuning/llama/recipe_a.yaml"},
            {"recipes/fine-tuning/llama/recipe_a.yaml": "js-model-a"},
        )

        with self.assertRaises(SystemExit) as ctx:
            main()
        self.assertEqual(ctx.exception.code, 1)

    @patch(f"{MODULE}.get_available_recipe_paths_for_region")
    @patch(f"{MODULE}.load_exclusion_patterns")
    @patch("builtins.open", mock_open(read_data='{"model-a": "js-model-a"}'))
    @patch(f"{MODULE}.list_recipes")
    def test_missing_excluded_no_error(self, mock_recipes, mock_excl, mock_region):
        mock_recipes.return_value = [self._make_recipe("fine-tuning/llama/recipe_a", "model-a")]
        mock_excl.return_value = [re.compile("recipe_a")]
        mock_region.return_value = (set(), {})

        # Should not raise SystemExit
        main()

    @patch(f"{MODULE}.get_available_recipe_paths_for_region")
    @patch(f"{MODULE}.load_exclusion_patterns")
    @patch("builtins.open", mock_open(read_data='{"model-a": "js-model-a"}'))
    @patch(f"{MODULE}.list_recipes")
    def test_missing_not_excluded_no_error(self, mock_recipes, mock_excl, mock_region):
        mock_recipes.return_value = [self._make_recipe("fine-tuning/llama/recipe_a", "model-a")]
        mock_excl.return_value = []
        mock_region.return_value = (set(), {})

        # Should not raise SystemExit
        main()

    @patch(f"{MODULE}.get_available_recipe_paths_for_region")
    @patch(f"{MODULE}.load_exclusion_patterns")
    @patch("builtins.open", mock_open(read_data='{"model-a": "js-model-a"}'))
    @patch(f"{MODULE}.list_recipes")
    def test_gap_filling_from_js_paths(self, mock_recipes, mock_excl, mock_region):
        """Covers the recipe_path_to_js_model gap-filling branch in main()."""
        recipe = self._make_recipe("fine-tuning/llama/recipe_a", "unknown-run-name")
        mock_recipes.return_value = [recipe]
        mock_excl.return_value = []
        mock_region.return_value = (
            {"recipes/fine-tuning/llama/recipe_a.yaml"},
            {"recipes/fine-tuning/llama/recipe_a.yaml": "js-model-a"},
        )

        main()


if __name__ == "__main__":
    unittest.main()
