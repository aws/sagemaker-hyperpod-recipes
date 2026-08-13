"""Unit tests for NovaRecipeTemplateProcessor.get_recipe_template coverage gaps."""

import pytest

from launcher.recipe_templatization.nova.nova_recipe_template_processor import (
    NovaRecipeTemplateProcessor,
)


@pytest.fixture
def nova_processor():
    """Create a NovaRecipeTemplateProcessor with minimal setup, bypassing full __init__."""
    proc = object.__new__(NovaRecipeTemplateProcessor)
    proc.nova_metadata = {
        "nova_distill_recipe.yaml": {"display_name": "Nova Data Distillation"},
        "nova_fallback_recipe.yaml": {"display_name": "Nova Custom Recipe"},
    }
    return proc


class TestNovaDistillationPath:
    """Test that distillation recipes match the nova_distill template."""

    def test_distillation_recipe_matches_nova_distill(self, nova_processor):
        templates = {"nova_distill": {"recipe_template": {}}}

        result = nova_processor.get_recipe_template(
            yaml_data={},
            template=templates,
            recipe_file_path="fine-tuning/nova/nova_distill_recipe.yaml",
        )

        assert result == {"recipe_template": {}}
        assert nova_processor._matched_template_key == "nova_distill"


class TestNovaFilenameFallback:
    """Test the filename-based fallback template matching."""

    def test_filename_fallback_matches_template_key(self, nova_processor):
        # Template key "nova_fallback" is a substring of the recipe filename "nova_fallback_recipe.yaml"
        templates = {"nova_fallback": {"recipe_template": {"some": "data"}}}

        # Mock _determine_recipe_type to return something that won't match any template
        nova_processor._determine_recipe_type = lambda *args: "nonexistent_type"

        result = nova_processor.get_recipe_template(
            yaml_data={},
            template=templates,
            recipe_file_path="fine-tuning/nova/nova_fallback_recipe.yaml",
        )

        assert result == {"recipe_template": {"some": "data"}}
        assert nova_processor._matched_template_key == "nova_fallback"
