"""Unit tests for pruning placeholder-derived override params that land on no recipe path.

`resolve_override_parameters` advertises an override for every {{placeholder}} in the
matched template, but `apply_template` only substitutes a placeholder whose dotted path
exists in the recipe. `BaseRecipeTemplateProcessor._prune_unlanded_placeholder_params`
drops any placeholder-derived override whose placeholder landed on NO recipe path (a
phantom override), while leaving alone:
  - params that come only from the template's `override_parameters` block (not placeholders)
  - placeholders placed at two alternative paths where exactly one exists (the
    `lr_scheduler`/`min_lr` FSDP-vs-Megatron pattern).
"""

import logging

import pytest

from launcher.recipe_templatization.base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
)


class _DummyProcessor(BaseRecipeTemplateProcessor):
    """Minimal concrete processor that skips file loading in __init__."""

    def _load_template(self):  # pragma: no cover - not exercised
        pass

    def get_recipe_template(self, *args, **kwargs):  # pragma: no cover - not exercised
        pass

    def get_recipe_metadata(self, *args, **kwargs):  # pragma: no cover - not exercised
        pass


@pytest.fixture
def proc():
    """A processor instance built without touching the filesystem."""
    p = object.__new__(_DummyProcessor)
    p.avoid_default_val_update_attributes = {"results_directory"}
    return p


class TestPruneUnlandedPlaceholderParams:
    def test_placeholder_on_missing_path_is_pruned_and_recipe_untouched(self, proc):
        """A placeholder at a path the recipe lacks -> param dropped, recipe unchanged."""
        # Template advertises `logging_steps` at trainer.logging_steps, which the recipe lacks.
        template = {
            "training_config": {
                "trainer": {"logging_steps": "{{logging_steps}}"},
                "optim": {"lr": "{{lr}}"},
            }
        }
        recipe = {"training_config": {"optim": {"lr": 0.001}}}

        proc.matched_template = template
        proc.recipe_override_parameters = {
            "logging_steps": {"default": 10},
            "lr": {"default": 0.0},
        }

        result = proc.apply_template(recipe, template)
        proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")

        # logging_steps did not land -> pruned; lr landed -> kept and updated.
        assert "logging_steps" not in proc.recipe_override_parameters
        assert "lr" in proc.recipe_override_parameters
        assert proc.recipe_override_parameters["lr"]["default"] == 0.001
        # The recipe itself gained no new key: the missing trainer path is not added.
        assert "trainer" not in result["training_config"]

    def test_placeholder_at_two_alt_paths_one_exists_is_kept(self, proc):
        """The lr_scheduler pattern: placeholder at two paths, one present -> kept."""
        template = {
            "training_config": {
                # Same placeholder at two alternative paths (FSDP vs Megatron style).
                "fsdp": {"lr_scheduler": "{{lr_scheduler}}"},
                "megatron": {"lr_scheduler": "{{lr_scheduler}}"},
            }
        }
        # Recipe carries only the FSDP path.
        recipe = {"training_config": {"fsdp": {"lr_scheduler": "cosine"}}}

        proc.matched_template = template
        proc.recipe_override_parameters = {"lr_scheduler": {"default": "constant"}}

        proc.apply_template(recipe, template)
        proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")

        assert "lr_scheduler" in proc.recipe_override_parameters
        assert proc.recipe_override_parameters["lr_scheduler"]["default"] == "cosine"

    def test_override_only_param_not_a_placeholder_is_untouched(self, proc):
        """A param present only via template override_parameters (no placeholder) is never pruned."""
        template = {"training_config": {"trainer": {"logging_steps": "{{logging_steps}}"}}}
        recipe = {"training_config": {}}  # logging_steps does not land

        proc.matched_template = template
        proc.recipe_override_parameters = {
            "logging_steps": {"default": 10},  # placeholder-derived, unlanded
            "instance_type": {"default": "ml.p5.48xlarge"},  # override-only, not a placeholder
        }

        proc.apply_template(recipe, template)
        proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")

        assert "logging_steps" not in proc.recipe_override_parameters
        assert "instance_type" in proc.recipe_override_parameters

    def test_logs_pruned_name_with_recipe_and_paths(self, proc, caplog):
        template = {"training_config": {"trainer": {"logging_steps": "{{logging_steps}}"}}}
        recipe = {"training_config": {}}

        proc.matched_template = template
        proc.recipe_override_parameters = {"logging_steps": {"default": 10}}

        proc.apply_template(recipe, template)
        with caplog.at_level(logging.INFO):
            proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "logging_steps" in messages
        assert "verl-sft-some-recipe" in messages
        assert "training_config.trainer.logging_steps" in messages

    def test_all_placeholders_land_no_pruning(self, proc):
        template = {"training_config": {"optim": {"lr": "{{lr}}"}}}
        recipe = {"training_config": {"optim": {"lr": 0.001}}}

        proc.matched_template = template
        proc.recipe_override_parameters = {"lr": {"default": 0.0}}

        proc.apply_template(recipe, template)
        proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")

        assert set(proc.recipe_override_parameters) == {"lr"}

    def test_noop_when_no_override_params(self, proc):
        proc.matched_template = {"a": "{{x}}"}
        proc.recipe_override_parameters = None
        proc._landed_placeholder_names = set()
        # Should not raise.
        proc._prune_unlanded_placeholder_params("fine-tuning/x/verl-sft-some-recipe")
        assert proc.recipe_override_parameters is None
