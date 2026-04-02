"""
Tests for RECIPES.md documentation generator.

Validates that:
1. docs/RECIPES.md matches generator output (golden test)
2. Every recipe file on disk appears exactly once in the doc (no missing, no duplicates)
3. All recipe/script links point to existing files
4. All metadata columns are populated where expected

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate RECIPES.md before validation
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_recipes_doc import (
    OUTPUT_FILE,
    RECIPES_DIR,
    Recipe,
    check_generation,
    generate_markdown,
    run_generation,
    scan_recipes,
)


# =============================================================================
# GOLDEN TEST - RECIPES.md matches generator output
# =============================================================================
class TestRecipesDocValidation:
    """Authoritative validation that RECIPES.md matches generator output."""

    def test_generated_doc_matches_disk(self):
        """
        RECIPES.md on disk exactly matches what the generator would produce.

        When GOLDEN_TEST_WRITE=true, regenerates doc first then validates.
        Otherwise, only validates existing doc against generator output.

        Failures indicate:
        1. Doc manually edited and drifted from generator
        2. Generator updated but doc not regenerated
        3. New recipes added without running generator
        4. Recipe YAML files changed without regenerating doc

        Fix:
          python scripts/generate_recipes_doc.py
        Or:
          GOLDEN_TEST_WRITE=true pytest tests/test_recipes_doc_generator.py::TestRecipesDocValidation::test_generated_doc_matches_disk
        """
        if os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes"):
            print("\nGOLDEN_TEST_WRITE enabled - regenerating RECIPES.md")
            run_generation()

        matches = check_generation(show_diff=False)

        if not matches:
            error_lines = [
                "",
                "=" * 70,
                "RECIPES.md is out of sync with generator",
                "=" * 70,
            ]

            if not OUTPUT_FILE.exists():
                error_lines.append("  Status: [MISSING] docs/RECIPES.md")
            else:
                error_lines.append("  Status: [DIFFERS] docs/RECIPES.md")

            error_lines.extend(
                [
                    "",
                    "To fix:",
                    "  python scripts/generate_recipes_doc.py",
                    "",
                    "Or with test:",
                    "  GOLDEN_TEST_WRITE=true pytest tests/test_recipes_doc_generator.py",
                    "",
                    "To see diffs:",
                    "  python scripts/generate_recipes_doc.py --check --diff",
                    "=" * 70,
                ]
            )
            pytest.fail("\n".join(error_lines))


# =============================================================================
# GENERATOR LOGIC TESTS
# =============================================================================
class TestRecipesDocGenerator:
    """Tests for generator infrastructure."""

    def test_paths_are_valid(self):
        """Output file and directories exist."""
        assert RECIPES_DIR.exists(), f"Recipes dir not found: {RECIPES_DIR}"
        assert OUTPUT_FILE.parent.exists(), f"Output dir not found: {OUTPUT_FILE.parent}"
        assert OUTPUT_FILE.name == "RECIPES.md"

    def test_check_returns_boolean(self):
        """check_generation returns a boolean."""
        result = check_generation(show_diff=False)
        assert isinstance(result, bool), "check_generation() should return bool"

    def test_scan_recipes_returns_list(self):
        """scan_recipes returns a flat list of Recipe objects."""
        recipes = scan_recipes()
        assert isinstance(recipes, list)
        assert len(recipes) > 0, "Expected at least one recipe"
        assert all(isinstance(r, Recipe) for r in recipes)

    def test_generate_markdown_returns_string(self):
        """generate_markdown returns a non-empty string."""
        recipes = scan_recipes()
        content = generate_markdown(recipes)
        assert isinstance(content, str)
        assert len(content) > 0


# =============================================================================
# DATA ACCURACY & COMPLETENESS TESTS
# =============================================================================
class TestDataValidation:
    """Tests for data accuracy, completeness, and link validity."""

    def test_all_recipe_links_are_valid(self):
        """All recipe file paths in the doc point to existing files."""
        repo_root = Path(__file__).parent.parent
        recipes = scan_recipes()

        broken = []
        for r in recipes:
            path = repo_root / r.recipe_path
            if not path.exists():
                broken.append(f"Recipe: {r.recipe_path}")

        assert not broken, f"Found {len(broken)} broken recipe links:\n" + "\n".join(broken)

    def test_all_launch_script_links_are_valid(self):
        """All launch script paths in the doc point to existing files."""
        repo_root = Path(__file__).parent.parent
        recipes = scan_recipes()

        broken = []
        for r in recipes:
            if r.script_path:
                path = repo_root / r.script_path
                if not path.exists():
                    broken.append(f"Script: {r.script_path} (for {r.recipe_path})")

        assert not broken, f"Found {len(broken)} broken script links:\n" + "\n".join(broken)

    def test_no_duplicate_recipes_in_doc(self):
        """Each recipe file path appears exactly once in the generated documentation.

        This verifies that the section filter functions in DOC_SECTIONS don't
        cause any recipe to appear in multiple sections. Checks by full path
        (not just filename) since versioned directories may share filenames.
        """
        recipes = scan_recipes()
        content = generate_markdown(recipes)

        # Extract all recipe paths from markdown links: [name](../path/to/recipe.yaml)
        recipe_paths = re.findall(r"\]\(\.\./([^)]+\.ya?ml)\)", content)
        duplicates = [path for path, count in Counter(recipe_paths).items() if count > 1]

        assert not duplicates, f"Found {len(duplicates)} duplicate recipe path(s) in RECIPES.md:\n" + "\n".join(
            f"  - {path}" for path in duplicates
        )

    def test_every_recipe_file_has_single_entry(self):
        """Every recipe YAML file on disk has exactly one entry in scan_recipes().

        This ensures 1:1 mapping between files and parsed recipes.
        """
        recipes = scan_recipes()
        recipe_paths = [r.recipe_path for r in recipes]
        duplicates = [p for p, count in Counter(recipe_paths).items() if count > 1]

        assert not duplicates, f"Found {len(duplicates)} recipe path(s) with multiple entries:\n" + "\n".join(
            f"  - {p}" for p in duplicates
        )

    def test_all_existing_recipes_are_documented(self):
        """All recipe YAML files in the repo are included in the documentation.

        This ensures no recipe is silently excluded due to parsing errors or
        filtering logic. Excludes hydra_config directories (composition fragments).

        Failures indicate:
        1. A recipe file exists but failed to parse
        2. A recipe is in an unexpected location
        3. A recipe was filtered out by the documentation logic

        Fix:
        - Check if the recipe file is valid YAML
        - Ensure recipe is in recipes_collection/recipes/
        - Review parse_recipe() for parsing issues
        """
        repo_root = Path(__file__).parent.parent

        # Find all recipe files on disk (excluding Hydra config fragments)
        existing = set()
        for root, dirs, files in os.walk(RECIPES_DIR):
            if "hydra_config" in root:
                continue
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    rel = str((Path(root) / f).relative_to(repo_root))
                    existing.add(rel)

        # Get documented recipe paths
        recipes = scan_recipes()
        documented = set(r.recipe_path for r in recipes)

        missing = existing - documented
        assert not missing, f"Found {len(missing)} recipe(s) on disk but not in documentation:\n" + "\n".join(
            f"  - {p}" for p in sorted(missing)
        )

    def test_every_recipe_has_model_name(self):
        """Every parsed recipe has a non-empty model name."""
        recipes = scan_recipes()
        missing = [r.recipe_path for r in recipes if not r.model or r.model == "-"]

        assert not missing, f"Found {len(missing)} recipe(s) with missing model name:\n" + "\n".join(
            f"  - {p}" for p in missing
        )

    def test_every_recipe_has_category_and_family(self):
        """Every parsed recipe has category and family derived from path."""
        recipes = scan_recipes()
        missing = [r.recipe_path for r in recipes if not r.category or not r.family]

        assert not missing, f"Found {len(missing)} recipe(s) with missing category/family:\n" + "\n".join(
            f"  - {p}" for p in missing
        )

    def test_sequence_lengths_are_valid(self):
        """Extracted sequence lengths are positive integers where present."""
        recipes = scan_recipes()
        invalid = []
        for r in recipes:
            if r.seq_len is not None:
                if not isinstance(r.seq_len, int) or r.seq_len <= 0:
                    invalid.append(f"{r.recipe_path}: seq_len={r.seq_len}")

        assert not invalid, f"Found {len(invalid)} recipe(s) with invalid sequence length:\n" + "\n".join(
            f"  - {p}" for p in invalid
        )

    def test_node_counts_are_valid(self):
        """Extracted node counts are positive integers where present."""
        recipes = scan_recipes()
        invalid = []
        for r in recipes:
            if r.nodes is not None:
                if not isinstance(r.nodes, int) or r.nodes <= 0:
                    invalid.append(f"{r.recipe_path}: nodes={r.nodes}")

        assert not invalid, f"Found {len(invalid)} recipe(s) with invalid node count:\n" + "\n".join(
            f"  - {p}" for p in invalid
        )
