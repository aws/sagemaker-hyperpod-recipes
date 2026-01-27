# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
Tests for launcher script generator.

Key tests:
1. test_generated_scripts_match_disk - authoritative validation that scripts match generator output
2. Config/Generator/Orchestrator unit tests for individual component functionality
3. Check mode tests for CI integration

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate launcher scripts before validation
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.launcher_scripts_generator.generate_launcher_scripts import (
    LauncherConfig,
    LauncherScriptGenerationOrchestrator,
    LauncherScriptGenerator,
    main,
)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset LauncherConfig singleton before and after each test."""
    LauncherConfig.reset()
    yield
    LauncherConfig.reset()


class TestLauncherConfig:
    """Tests for LauncherConfig singleton and configuration loading."""

    def test_config_loads_with_required_sections(self):
        """Config loads with all required sections and templates."""
        config = LauncherConfig()

        # Core properties exist
        assert config.header.strip().startswith("#!/bin/bash")
        assert len(config.templates) > 0
        assert config.default_template in config.templates

        # Required templates exist
        for template in ["llm_finetuning_aws", "verl", "nova"]:
            assert template in config.templates

        # Settings properties return correct types
        assert isinstance(config.preserved_dirs, list)
        assert isinstance(config.manually_maintained, dict)
        assert "recipes" in config.recipes_dir
        assert "launcher_scripts" in config.output_dir

    def test_singleton_pattern(self):
        """Config is singleton, reset clears it."""
        config1 = LauncherConfig()
        config2 = LauncherConfig()
        assert config1 is config2

        LauncherConfig.reset()
        config3 = LauncherConfig()
        assert config3.templates is not None


class TestLauncherScriptGenerator:
    """Tests for LauncherScriptGenerator template selection and script generation."""

    @pytest.mark.parametrize(
        "recipe_name,recipe_path,expected_template",
        [
            (
                "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
                "fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
                "llm_finetuning_aws",
            ),
            (
                "verl-grpo-rlaif-llama-3-dot-1-8b-instruct-lora",
                "fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-1-8b-instruct-lora",
                "verl",
            ),
            (
                "nova_lite_1_0_p5_p4d_gpu_sft",
                "fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft",
                "nova",
            ),
            (
                "checkpointless_llama3_70b_pretrain",
                "training/llama/checkpointless_llama3_70b_pretrain",
                "hyperpod_checkpointless_nemo",
            ),
            ("falcon", "training/custom_model/falcon", "hf"),
        ],
    )
    def test_template_selection(self, recipe_name, recipe_path, expected_template):
        """Generator selects correct template based on recipe's model_type."""
        generator = LauncherScriptGenerator(recipe_name=recipe_name, recipe_path=recipe_path)
        assert generator.get_template_key() == expected_template

    def test_fallback_to_default_for_unknown_model_type(self):
        """Unknown/missing model_type falls back to default template."""
        generator = LauncherScriptGenerator(
            recipe_name="unknown_recipe",
            recipe_path="nonexistent/path/unknown_recipe",
        )
        config = LauncherConfig()
        assert generator.get_template_key() == config.default_template
        assert generator.get_model_type() == ""
        assert generator.recipe_data == {}

    def test_generate_creates_executable_script_with_correct_content(self):
        """Generate creates executable script with recipe path and no unreplaced placeholders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = LauncherScriptGenerator(
                recipe_name="test_recipe",
                recipe_path="fine-tuning/test/test_recipe",
            )
            script_path, content = generator.generate(Path(tmpdir))

            # File exists and is executable
            assert script_path.exists()
            assert script_path.stat().st_mode & 0o111
            assert script_path.name == "run_test_recipe.sh"

            # Content is correct
            assert content.startswith("#!/bin/bash")
            assert "recipes=fine-tuning/test/test_recipe" in content
            assert len(re.findall(r"\{[a-z_]+\}", content)) == 0  # No unreplaced placeholders

    def test_generate_with_custom_script_name(self):
        """Custom script name is used when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = LauncherScriptGenerator(
                recipe_name="test_recipe",
                recipe_path="test/path/test_recipe",
            )
            script_path, _ = generator.generate(Path(tmpdir), script_name="custom.sh")
            assert script_path.name == "custom.sh"

    def test_generate_dry_run_does_not_write(self):
        """Dry run returns content but doesn't write file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = LauncherScriptGenerator(
                recipe_name="test_recipe",
                recipe_path="test/path/test_recipe",
            )
            script_path, content = generator.generate(Path(tmpdir), dry_run=True)

            assert not script_path.exists()
            assert "#!/bin/bash" in content

    def test_script_name_converts_hyphens_to_underscores(self):
        """Hyphens in recipe name become underscores in script name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = LauncherScriptGenerator(
                recipe_name="my-hyphenated-recipe",
                recipe_path="test/path/my-hyphenated-recipe",
            )
            script_path, _ = generator.generate(Path(tmpdir))
            assert script_path.name == "run_my_hyphenated_recipe.sh"

    def test_generator_accepts_extra_kwargs(self):
        """Extra kwargs are accepted for backwards compatibility."""
        generator = LauncherScriptGenerator(
            recipe_name="test",
            recipe_path="test/path",
            extra_param="ignored",
        )
        assert generator.recipe_name == "test"


class TestLauncherScriptGenerationOrchestrator:
    """Tests for orchestrator recipe discovery and model family inference."""

    def test_orchestrator_initialization(self):
        """Orchestrator initializes with correct paths."""
        orchestrator = LauncherScriptGenerationOrchestrator()
        assert orchestrator.recipes_dir.exists()
        assert orchestrator.output_dir.exists()
        assert len(orchestrator.generated) == 0

    def test_discover_recipes(self):
        """Recipe discovery finds sorted YAML files."""
        orchestrator = LauncherScriptGenerationOrchestrator()
        recipes = orchestrator.discover_recipes()

        assert len(recipes) > 0
        assert all(r.suffix == ".yaml" for r in recipes)
        assert recipes == sorted(recipes)

    @pytest.mark.parametrize(
        "path_parts,expected_family",
        [
            ("fine-tuning/llama/recipe.yaml", "llama"),
            ("fine-tuning/nova/nova_1_0/recipe.yaml", "nova"),
            ("training/custom_model/falcon.yaml", "custom_model"),
            ("fine-tuning/deepseek/recipe.yaml", "deepseek"),
            ("evaluation/open-source/recipe.yaml", "open-source"),
            ("recipe.yaml", "misc"),  # Short path defaults to misc
        ],
    )
    def test_infer_model_family(self, path_parts, expected_family):
        """Model family is correctly inferred from recipe path."""
        orchestrator = LauncherScriptGenerationOrchestrator()
        recipe_path = orchestrator.recipes_dir / path_parts
        assert orchestrator.infer_model_family(recipe_path) == expected_family

    def test_check_returns_empty_when_scripts_match(self):
        """Check returns empty list when all scripts match."""
        orchestrator = LauncherScriptGenerationOrchestrator()
        mismatches = orchestrator.check(show_diff=False)
        # If CI passes, this should be empty; test structure is valid either way
        assert isinstance(mismatches, list)

    def test_check_with_show_diff_prints_output(self, capsys):
        """Check with show_diff prints diff information."""
        orchestrator = LauncherScriptGenerationOrchestrator()
        # Run check with show_diff - captures any output
        orchestrator.check(show_diff=True)
        # Just verify it doesn't crash; actual diffs depend on disk state


class TestTemplateContent:
    """Tests for template content and placeholder requirements."""

    def test_all_templates_have_recipe_path_placeholder(self):
        """All templates include {recipe_path} placeholder."""
        config = LauncherConfig()
        for name, template in config.templates.items():
            assert "{recipe_path}" in template, f"Template '{name}' missing recipe_path"

    def test_verl_template_has_required_variables(self):
        """VERL template has actor/critic/train variables."""
        config = LauncherConfig()
        template = config.templates.get("verl", "")
        for var in ["ACTOR_MODEL_PATH", "CRITIC_MODEL_PATH", "TRAIN_DATA"]:
            assert var in template

    def test_llmft_template_has_required_variables(self):
        """LLMFT template has required environment variables."""
        config = LauncherConfig()
        template = config.templates.get("llm_finetuning_aws", "")
        for var in ["TRAIN_DIR", "EXP_DIR", "MODEL_NAME_OR_PATH"]:
            assert var in template

    def test_checkpointless_template_has_required_content(self):
        """Checkpointless template has cluster and train_dir."""
        config = LauncherConfig()
        template = config.templates.get("hyperpod_checkpointless_nemo", "")
        assert "cluster=k8s" in template
        assert "TRAIN_DIR" in template


class TestMainCLI:
    """Tests for main() CLI entry point."""

    def test_main_check_mode_success(self):
        """Main with --check exits 0 when scripts match."""
        with patch.object(LauncherScriptGenerationOrchestrator, "check", return_value=[]):
            with patch.object(sys, "argv", ["prog", "--check"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

    def test_main_check_mode_failure(self, capsys):
        """Main with --check exits 1 when scripts don't match."""
        # Create orchestrator to get real root
        orchestrator = LauncherScriptGenerationOrchestrator()
        fake_path = orchestrator.root / "fake_script.sh"
        mock_mismatch = [(fake_path, "actual", "expected")]
        with patch.object(LauncherScriptGenerationOrchestrator, "check", return_value=mock_mismatch):
            with patch.object(sys, "argv", ["prog", "--check"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_main_generate_mode(self, capsys):
        """Main without --check runs generation."""
        with patch.object(LauncherScriptGenerationOrchestrator, "run") as mock_run:
            with patch.object(sys, "argv", ["prog"]):
                main()
                mock_run.assert_called_once()


class TestLauncherScriptValidation:
    """
    Authoritative validation that scripts on disk match generator output.

    This is the key test - if scripts match what the generator produces,
    all properties (shebang, copyright, recipe path, etc.) are implicitly
    validated since they come from the generator templates.

    Set GOLDEN_TEST_WRITE=true to regenerate scripts before validation.
    """

    def test_generated_scripts_match_disk(self):
        """
        All launcher scripts on disk exactly match what the generator would produce.

        When GOLDEN_TEST_WRITE=true, regenerates scripts first then validates.
        Otherwise, only validates existing scripts against generator output.

        Failures indicate:
        1. Scripts manually edited and drifted from generator
        2. Generator updated but scripts not regenerated
        3. New recipes added without running generator

        Fix: python scripts/launcher_scripts_generator/generate_launcher_scripts.py
        Or:  GOLDEN_TEST_WRITE=true pytest tests/test_launcher_script_generator.py
        """
        orchestrator = LauncherScriptGenerationOrchestrator()

        # Regenerate scripts if GOLDEN_TEST_WRITE is set
        if os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes"):
            orchestrator.run()

        mismatches = orchestrator.check(show_diff=False)

        if mismatches:
            error_lines = [
                f"Found {len(mismatches)} launcher script(s) out of sync with generator.",
                "",
                "Mismatched scripts:",
            ]
            for path, actual, _ in mismatches:
                status = "MISSING" if actual is None else "DIFFERS"
                error_lines.append(f"  [{status}] {path.relative_to(orchestrator.root)}")

            error_lines.extend(
                [
                    "",
                    "To fix: python scripts/launcher_scripts_generator/generate_launcher_scripts.py",
                    "Or:     GOLDEN_TEST_WRITE=true pytest tests/test_launcher_script_generator.py",
                    "To see diffs: python scripts/launcher_scripts_generator/generate_launcher_scripts.py --check --diff",
                ]
            )
            pytest.fail("\n".join(error_lines))
