import os
import unittest
from unittest.mock import patch

from hyperpod_recipes import get_recipe, list_recipes
from hyperpod_recipes.recipe import RECIPES_DIR, Recipe


class TestHyperpodRecipes(unittest.TestCase):
    def test_list_recipes_returns_recipes(self):
        recipes = list_recipes()

        # Assertions
        self.assertGreaterEqual(len(recipes), 100)
        # A sanity net against runaway generation, not a pin, so it keeps room
        # to grow as the per-instance verl ladders expand. 607 with all four family
        # ladders stacked; raised to 750 for ~20% headroom. Only the tip of the PR
        # chain sees the combined total -- gpt_oss alone is 416.
        self.assertLessEqual(len(recipes), 750)
        self.assertTrue(all(isinstance(r, Recipe) for r in recipes))
        self.assertTrue(
            "training/nova/nova_1_0/nova_micro/CPT/nova_micro_1_0_p5x8_gpu_pretrain"
            in [recipe.name for recipe in recipes]
        )

    @patch("os.path.exists")
    def test_list_recipes_raises_if_no_directory(self, mock_exists):
        # list_recipes is lru_cached, so a warm cache left by another test on the same
        # xdist worker would short-circuit the patched os.path.exists and no error would
        # be raised. Clearing it first makes the assertion independent of test order.
        list_recipes.cache_clear()
        mock_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            list_recipes()

    @patch("os.path.exists")
    def test_get_recipe_found(self, mock_exists):
        mock_exists.return_value = True

        recipe = get_recipe("file1")

        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(recipe.name, "file1")
        self.assertEqual(recipe.recipe_id, "file1")
        self.assertEqual(recipe.path, os.path.join(RECIPES_DIR, "file1.yaml"))

    @patch("os.path.exists")
    def test_get_recipe_not_found(self, mock_exists):
        mock_exists.return_value = False

        with self.assertRaises(KeyError) as cm:
            get_recipe("nonexistent")

        self.assertIn("Recipe not found", str(cm.exception))
