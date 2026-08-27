"""Unit tests for BaseRecipeTemplateProcessor coverage gaps."""

from unittest.mock import patch

import pytest

from launcher.recipe_templatization.base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
)


class TestExtractPlaceholderNames:
    """Tests for the static extract_placeholder_names delegate."""

    def test_extracts_simple_placeholders(self):
        template = {"key": "{{foo}}", "nested": {"inner": "{{bar}}"}}
        result = BaseRecipeTemplateProcessor.extract_placeholder_names(template)
        assert result == {"foo", "bar"}

    def test_extracts_from_lists(self):
        template = {"items": ["{{a}}", "{{b}}"]}
        result = BaseRecipeTemplateProcessor.extract_placeholder_names(template)
        assert result == {"a", "b"}

    def test_empty_template(self):
        result = BaseRecipeTemplateProcessor.extract_placeholder_names({})
        assert result == set()

    def test_no_placeholders(self):
        template = {"key": "plain_value", "num": 42}
        result = BaseRecipeTemplateProcessor.extract_placeholder_names(template)
        assert result == set()


class TestLoadBaseOverrideParameters:
    """Tests for _load_base_override_parameters error handling."""

    def test_raises_when_file_missing(self):
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Base override parameters file not found"):
                # Create a minimal subclass to test the base class method
                class DummyProcessor(BaseRecipeTemplateProcessor):
                    def _load_template(self):
                        pass

                    def get_recipe_template(self, *args, **kwargs):
                        pass

                    def get_recipe_metadata(self, *args, **kwargs):
                        pass

                DummyProcessor({})


class TestGetTemplateCategory:
    """Tests for the default _get_template_category implementation."""

    def test_default_returns_fine_tuning(self):
        class DummyProcessor(BaseRecipeTemplateProcessor):
            def _load_template(self):
                pass

            def get_recipe_template(self, *args, **kwargs):
                pass

            def get_recipe_metadata(self, *args, **kwargs):
                pass

        # Bypass __init__ to avoid loading files
        proc = object.__new__(DummyProcessor)
        assert proc._get_template_category() == "fine_tuning"
