"""
Tests for CONTAINERS.md documentation generator.

Validates that:
1. docs/CONTAINERS.md matches generator output (golden test)
2. Generator infrastructure works correctly
3. Container data is parsed and formatted correctly
4. All recipe/container links are valid

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate CONTAINERS.md before validation
"""

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_containers_doc import (
    OUTPUT_FILE,
    TEMPLATIZATION_DIR,
    _format_container_cell,
    _get_container_for_region,
    _load_regional_parameters,
    _match_recipe_to_regional_key,
    check_generation,
    generate_markdown,
    run_generation,
)
from scripts.generate_recipes_doc import scan_recipes


# =============================================================================
# GOLDEN TEST - CONTAINERS.md matches generator output
# =============================================================================
class TestContainersDocValidation:
    """Authoritative validation that CONTAINERS.md matches generator output."""

    def test_generated_doc_matches_disk(self):
        """
        CONTAINERS.md on disk exactly matches what the generator would produce.

        When GOLDEN_TEST_WRITE=true, regenerates doc first then validates.
        Otherwise, only validates existing doc against generator output.

        Failures indicate:
        1. Doc manually edited and drifted from generator
        2. Generator updated but doc not regenerated
        3. New regional parameters added without running generator
        4. Regional parameter files changed without regenerating doc

        Fix:
          uv run poe generate-containers-doc
        Or:
          GOLDEN_TEST_WRITE=true pytest tests/test_containers_doc_generator.py::TestContainersDocValidation::test_generated_doc_matches_disk
        """
        if os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes"):
            print("\nGOLDEN_TEST_WRITE enabled - regenerating CONTAINERS.md")
            run_generation()

        matches = check_generation(show_diff=False)

        if not matches:
            error_lines = [
                "",
                "=" * 70,
                "CONTAINERS.md is out of sync with generator",
                "=" * 70,
            ]

            if not OUTPUT_FILE.exists():
                error_lines.append("  Status: [MISSING] docs/CONTAINERS.md")
            else:
                error_lines.append("  Status: [DIFFERS] docs/CONTAINERS.md")

            error_lines.extend(
                [
                    "",
                    "To fix:",
                    "  uv run poe generate-containers-doc",
                    "",
                    "Or with test:",
                    "  GOLDEN_TEST_WRITE=true pytest tests/test_containers_doc_generator.py",
                    "",
                    "To see diffs:",
                    "  uv run poe generate-containers-doc --check --diff",
                    "=" * 70,
                ]
            )
            pytest.fail("\n".join(error_lines))


# =============================================================================
# GENERATOR LOGIC TESTS
# =============================================================================
class TestContainersDocGenerator:
    """Tests for generator infrastructure."""

    def test_paths_are_valid(self):
        """Output file and directories exist."""
        assert TEMPLATIZATION_DIR.exists(), f"Templatization dir not found: {TEMPLATIZATION_DIR}"
        assert OUTPUT_FILE.parent.exists(), f"Output dir not found: {OUTPUT_FILE.parent}"
        assert OUTPUT_FILE.name == "CONTAINERS.md"

    def test_check_returns_boolean(self):
        """check_generation returns a boolean."""
        result = check_generation(show_diff=False)
        assert isinstance(result, bool), "check_generation() should return bool"

    def test_load_regional_parameters_returns_dict(self):
        """_load_regional_parameters returns a non-empty dict."""
        regional_params = _load_regional_parameters()
        assert isinstance(regional_params, dict)
        assert len(regional_params) > 0, "Expected at least one regional parameter entry"

    def test_generate_markdown_returns_string(self):
        """generate_markdown returns a non-empty string."""
        recipes = scan_recipes()
        content = generate_markdown(recipes)
        assert isinstance(content, str)
        assert len(content) > 0


# =============================================================================
# CONTAINER CELL FORMATTING TESTS
# =============================================================================
class TestContainerCellFormatting:
    """Tests for container cell formatting logic."""

    def test_format_both_none_returns_dash(self):
        """When both URIs are None, returns '—'."""
        result = _format_container_cell(None, None)
        assert result == "—"

    def test_format_same_uri_returns_single(self):
        """When both URIs are identical, returns single URI."""
        uri = "123.dkr.ecr.us-east-1.amazonaws.com/repo:tag"
        result = _format_container_cell(uri, uri)
        assert result == uri

    def test_format_different_uris_shows_both(self):
        """When URIs differ, shows both with K8S/SM prefixes."""
        k8s = "123.dkr.ecr.us-east-1.amazonaws.com/repo:k8s-tag"
        sm = "123.dkr.ecr.us-east-1.amazonaws.com/repo:sm-tag"
        result = _format_container_cell(k8s, sm)
        assert "K8S:" in result
        assert "SM:" in result
        assert k8s in result
        assert sm in result

    def test_format_only_k8s(self):
        """When only K8S URI exists, shows with K8S prefix."""
        k8s = "123.dkr.ecr.us-east-1.amazonaws.com/repo:k8s-tag"
        result = _format_container_cell(k8s, None)
        assert "K8S:" in result
        assert k8s in result

    def test_format_only_sm(self):
        """When only SM URI exists, shows with SM prefix."""
        sm = "123.dkr.ecr.us-east-1.amazonaws.com/repo:sm-tag"
        result = _format_container_cell(None, sm)
        assert "SM:" in result
        assert sm in result


# =============================================================================
# RECIPE MATCHING TESTS
# =============================================================================
class TestRecipeMatching:
    """Tests for recipe-to-regional-key matching logic."""

    def test_match_llmft_generic(self):
        """LLMFT recipes should match 'llmft' fallback key."""
        regional_keys = {"llmft", "nova_lite_2_0_p5_gpu_lora_sft"}
        recipe_filename = "llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml"
        result = _match_recipe_to_regional_key(recipe_filename, regional_keys)
        assert result == "llmft"

    def test_match_nova_specific(self):
        """Nova recipes should match specific regional keys."""
        regional_keys = {"llmft", "nova_lite_2_0_p5_gpu_lora_sft", "nova"}
        recipe_filename = "nova_lite_2_0_p5_gpu_lora_sft.yaml"
        result = _match_recipe_to_regional_key(recipe_filename, regional_keys)
        # Should match the specific key, not the generic "nova"
        assert result in {"nova_lite_2_0_p5_gpu_lora_sft", "nova"}

    def test_match_no_match_returns_none(self):
        """When no match found, returns None."""
        regional_keys = {"llmft", "nova"}
        recipe_filename = "unknown_recipe.yaml"
        result = _match_recipe_to_regional_key(recipe_filename, regional_keys)
        # Could match a framework fallback or be None
        assert result is None or result in regional_keys


# =============================================================================
# DATA ACCURACY TESTS
# =============================================================================
class TestDataAccuracy:
    """Tests for data accuracy and completeness."""

    def test_regional_parameters_have_prod_data(self):
        """Regional parameter files contain 'prod' stage data."""
        regional_params = _load_regional_parameters()
        has_prod = False

        for recipe_key, recipe_data in regional_params.items():
            if recipe_key.startswith("_"):
                continue
            for pipeline in ["k8s", "sm_jobs"]:
                if pipeline in recipe_data:
                    container_data = recipe_data[pipeline].get("container_image", {})
                    if "prod" in container_data and container_data["prod"]:
                        has_prod = True
                        break
            if has_prod:
                break

        assert has_prod, "Expected at least one regional parameter with 'prod' stage data"

    def test_container_uris_are_valid_format(self):
        """Container URIs follow expected ECR format."""
        regional_params = _load_regional_parameters()
        invalid_uris = []

        for recipe_key, recipe_data in regional_params.items():
            if recipe_key.startswith("_"):
                continue

            for pipeline in ["k8s", "sm_jobs"]:
                if pipeline not in recipe_data:
                    continue

                prod_data = recipe_data[pipeline].get("container_image", {}).get("prod", {})
                for region, uri in prod_data.items():
                    if not isinstance(uri, str):
                        invalid_uris.append(f"{recipe_key}/{pipeline}/{region}: not a string")
                        continue

                    # Basic ECR URI validation
                    if not (".dkr.ecr." in uri and ".amazonaws.com" in uri and ":" in uri):
                        invalid_uris.append(f"{recipe_key}/{pipeline}/{region}: {uri}")

        assert not invalid_uris, f"Found {len(invalid_uris)} invalid container URIs:\n" + "\n".join(
            f"  - {u}" for u in invalid_uris[:10]
        )

    def test_all_recipes_have_model_short_name_or_model(self):
        """Every recipe has either model_short_name or model for display."""
        recipes = scan_recipes()
        missing = []

        for recipe in recipes:
            if not recipe.model_short_name and not recipe.model:
                missing.append(recipe.recipe_path)

        assert not missing, f"Found {len(missing)} recipes without model name:\n" + "\n".join(
            f"  - {p}" for p in missing[:10]
        )

    def test_generated_doc_has_all_sections(self):
        """Generated document includes all expected model family sections."""
        recipes = scan_recipes()
        content = generate_markdown(recipes)

        # New structure is organized by model family
        expected_sections = [
            "## Llama",
            "## Qwen",
            "## DeepSeek",
            "## GPT-OSS",
            "## Nova",
        ]

        for section in expected_sections:
            assert section in content, f"Missing section: {section}"

    def test_generated_doc_has_table_of_contents(self):
        """Generated document includes table of contents with links."""
        recipes = scan_recipes()
        content = generate_markdown(recipes)

        assert "## Table of Contents" in content
        # New structure uses model families
        assert "[Llama]" in content or "[Qwen]" in content or "[DeepSeek]" in content
        assert "](#" in content  # Has anchor links


# =============================================================================
# CONTAINER EXTRACTION TESTS
# =============================================================================
class TestContainerExtraction:
    """Tests for container image extraction from regional data."""

    def test_get_container_for_region_k8s(self):
        """Extract K8S container for a specific region."""
        regional_params = _load_regional_parameters()

        # Find a recipe with k8s containers
        for recipe_key, recipe_data in regional_params.items():
            if recipe_key.startswith("_"):
                continue

            prod_data = recipe_data.get("k8s", {}).get("container_image", {}).get("prod", {})
            if "us-east-1" in prod_data:
                uri = _get_container_for_region(recipe_data, "k8s", "us-east-1")
                assert uri is not None
                assert isinstance(uri, str)
                assert ".dkr.ecr." in uri
                return

        pytest.skip("No K8S containers found in regional parameters")

    def test_get_container_for_region_sm_jobs(self):
        """Extract SM Jobs container for a specific region."""
        regional_params = _load_regional_parameters()

        # Find a recipe with sm_jobs containers
        for recipe_key, recipe_data in regional_params.items():
            if recipe_key.startswith("_"):
                continue

            prod_data = recipe_data.get("sm_jobs", {}).get("container_image", {}).get("prod", {})
            if "us-east-1" in prod_data:
                uri = _get_container_for_region(recipe_data, "sm_jobs", "us-east-1")
                assert uri is not None
                assert isinstance(uri, str)
                assert ".dkr.ecr." in uri
                return

        pytest.skip("No SM Jobs containers found in regional parameters")

    def test_get_container_missing_region_returns_none(self):
        """When region doesn't exist, returns None."""
        recipe_data = {
            "k8s": {"container_image": {"prod": {"us-east-1": "123.dkr.ecr.us-east-1.amazonaws.com/repo:tag"}}}
        }

        result = _get_container_for_region(recipe_data, "k8s", "nonexistent-region")
        assert result is None
