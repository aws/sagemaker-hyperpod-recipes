"""Tests for scripts/detect_recipes.py."""

import subprocess
from unittest.mock import MagicMock, patch

import yaml

from scripts.detect_recipes import _get_file_at_ref, detect_recipes


class TestDetectRecipes:
    @patch("scripts.detect_recipes.subprocess.run")
    def test_new_recipes_only(self, mock_run):
        """include_modified=False only detects added files."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/new-recipe.yaml\n",
            returncode=0,
        )

        with patch("scripts.detect_recipes._get_file_at_ref", return_value=None):
            result = detect_recipes("main", include_modified=False)

        assert result == ["recipes_collection/recipes/fine-tuning/qwen/new-recipe.yaml"]
        call_args = mock_run.call_args[0][0]
        assert "--diff-filter=A" in call_args

    @patch("scripts.detect_recipes.subprocess.run")
    def test_include_modified_default(self, mock_run):
        """Default includes both added and modified."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/recipe.yaml\n",
            returncode=0,
        )

        with patch("scripts.detect_recipes._get_file_at_ref", return_value=None):
            result = detect_recipes("main")

        call_args = mock_run.call_args[0][0]
        assert "--diff-filter=AM" in call_args
        assert len(result) == 1

    @patch("scripts.detect_recipes.subprocess.run")
    def test_functional_change_included(self, mock_run):
        """Modified recipe with functional change is included."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/recipe.yaml\n",
            returncode=0,
        )

        old_yaml = yaml.dump({"training_config": {"trainer": {"total_training_steps": 100}}})
        new_yaml = yaml.dump({"training_config": {"trainer": {"total_training_steps": 200}}})

        with patch("scripts.detect_recipes._get_file_at_ref", side_effect=[old_yaml, new_yaml]):
            result = detect_recipes("main")

        assert len(result) == 1

    @patch("scripts.detect_recipes.subprocess.run")
    def test_non_functional_change_filtered(self, mock_run):
        """Modified recipe with only non-functional change is filtered out."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/recipe.yaml\n",
            returncode=0,
        )

        old_yaml = yaml.dump({"data": {"train_files": "/old/path"}, "training_config": {"trainer": {"steps": 100}}})
        new_yaml = yaml.dump({"data": {"train_files": "/new/path"}, "training_config": {"trainer": {"steps": 100}}})

        with patch("scripts.detect_recipes._get_file_at_ref", side_effect=[old_yaml, new_yaml]):
            result = detect_recipes("main")

        assert result == []

    @patch("scripts.detect_recipes.subprocess.run")
    def test_no_changes(self, mock_run):
        """No changed files returns empty list."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = detect_recipes("main")
        assert result == []

    @patch("scripts.detect_recipes.subprocess.run")
    def test_unparseable_yaml_included(self, mock_run):
        """Recipe that can't be parsed is included (safe default)."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/recipe.yaml\n",
            returncode=0,
        )

        with patch("scripts.detect_recipes._get_file_at_ref", side_effect=["valid: true", "{{invalid"]):
            result = detect_recipes("main")

        assert len(result) == 1

    @patch("scripts.detect_recipes.subprocess.run")
    def test_new_content_none_included(self, mock_run):
        """Recipe where new content can't be read is included."""
        mock_run.return_value = MagicMock(
            stdout="recipes_collection/recipes/fine-tuning/qwen/recipe.yaml\n",
            returncode=0,
        )

        with patch("scripts.detect_recipes._get_file_at_ref", side_effect=["old: true", None]):
            result = detect_recipes("main")

        assert len(result) == 1


class TestGetFileAtRef:
    def test_success(self):
        with patch("scripts.detect_recipes.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="file content", returncode=0)
            result = _get_file_at_ref("path/to/file.yaml", "main")

        assert result == "file content"
        mock_run.assert_called_once()

    def test_not_found(self):
        with patch("scripts.detect_recipes.subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
            result = _get_file_at_ref("nonexistent.yaml", "main")

        assert result is None
