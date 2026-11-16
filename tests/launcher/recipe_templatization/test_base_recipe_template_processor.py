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


class _PinnedDummyProcessor(BaseRecipeTemplateProcessor):
    def _load_template(self):
        pass

    def get_recipe_template(self, *args, **kwargs):
        pass

    def get_recipe_metadata(self, *args, **kwargs):
        pass


def _make_pinned_processor(override_params: dict, pinned: dict):
    proc = object.__new__(_PinnedDummyProcessor)
    proc.recipe_override_parameters = override_params
    proc.pinned_parameters = pinned
    proc.avoid_default_val_update_attributes = set()
    return proc


class TestGetPinnedParameters:
    """Tests for detecting params whose resolved range collapses to a single value."""

    def test_min_equals_max_is_pinned(self):
        params = {"lora_dropout": {"type": "float", "min": 0.0, "max": 0.0, "default": 0.0}}
        assert BaseRecipeTemplateProcessor._get_pinned_parameters(params) == {"lora_dropout": 0.0}

    def test_open_range_is_not_pinned(self):
        params = {"lora_dropout": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.05}}
        assert BaseRecipeTemplateProcessor._get_pinned_parameters(params) == {}

    def test_missing_bounds_or_non_numeric_not_pinned(self):
        params = {
            "only_min": {"type": "integer", "min": 1},
            "name": {"type": "string", "min": "a", "max": "a"},
        }
        assert BaseRecipeTemplateProcessor._get_pinned_parameters(params) == {}

    def test_handles_empty(self):
        assert BaseRecipeTemplateProcessor._get_pinned_parameters(None) == {}


class TestApplyTemplatePinnedParameters:
    """Pinned params keep their literal recipe value rather than a placeholder."""

    TEMPLATE = {"peft": {"lora_dropout": "{{lora_dropout}}", "lora_rank": "{{lora_rank}}"}}

    def test_pinned_param_keeps_literal_value(self):
        proc = _make_pinned_processor(
            {"lora_dropout": {"default": 0.0}, "lora_rank": {"default": 8}},
            {"lora_dropout": 0.0},
        )
        result = proc.apply_template({"peft": {"lora_dropout": 0.0, "lora_rank": 16}}, self.TEMPLATE)
        assert result == {"peft": {"lora_dropout": 0.0, "lora_rank": "{{lora_rank}}"}}
        assert proc.recipe_override_parameters["lora_rank"]["default"] == 16

    def test_pinned_param_mismatch_raises(self):
        proc = _make_pinned_processor({"lora_dropout": {"default": 0.0}}, {"lora_dropout": 0.0})
        with pytest.raises(ValueError, match="pinned to 0.0"):
            proc.apply_template({"peft": {"lora_dropout": 0.05, "lora_rank": 16}}, self.TEMPLATE)
