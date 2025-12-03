import os
import unittest

from hyperpod_recipes.recipe import RECIPES_DIR, Recipe


class TestRecipe(unittest.TestCase):
    def setUp(self):
        self.name = "path/to/recipe"
        self.recipe_id = "path/to/recipe"
        self.path = os.path.join(RECIPES_DIR, "path/to/recipe.yaml")
        self.recipe = Recipe(self.path)

    def test_init(self):
        # Test that the object initializes with correct attributes
        self.assertEqual(self.recipe.name, self.name)
        self.assertEqual(self.recipe.recipe_id, self.recipe_id)
        self.assertEqual(self.recipe.path, self.path)

    def test_repr(self):
        # Test that __repr__ returns a string containing all fields
        repr_str = repr(self.recipe)
        self.assertIn(self.name, repr_str)
        self.assertIn(self.recipe_id, repr_str)
        self.assertIn(self.path, repr_str)
        self.assertTrue(repr_str.startswith("Recipe("))
        self.assertTrue(repr_str.endswith(")"))

    def test_methods_exist(self):
        # Ensure the methods exist and are callable
        self.assertTrue(callable(getattr(self.recipe, "validate", None)))
        self.assertTrue(callable(getattr(self.recipe, "launch", None)))
        self.assertTrue(callable(getattr(self.recipe, "launch_json", None)))
        self.assertTrue(callable(getattr(self.recipe, "write_recipe_dir", None)))
        self.assertTrue(callable(getattr(Recipe, "read_recipe_dir", None)))

    # Placeholder tests for future implementation
    def test_validate(self):
        # Currently validate does nothing, but the test ensures it can be called
        self.recipe.validate()

    def test_launch(self):
        self.recipe.launch()

    def test_launch_json(self):
        result = self.recipe.launch_json()
        self.assertIsNone(result)  # current stub returns None

    def test_write_recipe_dir(self):
        result = self.recipe.write_recipe_dir("/tmp", format="nemo_launcher")
        self.assertIsNone(result)  # stub returns None

    def test_read_recipe_dir(self):
        result = Recipe.read_recipe_dir("/tmp")
        self.assertIsNone(result)  # stub returns None
