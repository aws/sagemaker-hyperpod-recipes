"""
Tests for HYPERPARAMETERS.md documentation generator.

Validates that:
1. docs/HYPERPARAMETERS.md matches generator output (golden test)
2. Every template JSON file is loadable and has at least one template
3. Every template has a display_name
4. All framework JSON paths resolve to existing files

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate HYPERPARAMETERS.md before validation
"""

import os
import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_hyperparameters_doc import OUTPUT_FILE, check_generation, run_generation


# =============================================================================
# GOLDEN TEST - HYPERPARAMETERS.md matches generator output
# =============================================================================
class TestHyperparametersDocValidation:
    """Authoritative validation that HYPERPARAMETERS.md matches generator output."""

    def test_generated_doc_matches_disk(self):
        """
        HYPERPARAMETERS.md on disk exactly matches what the generator would produce.

        When GOLDEN_TEST_WRITE=true, regenerates doc first then validates.
        Otherwise, only validates existing doc against generator output.

        Failures indicate:
        1. Doc manually edited and drifted from generator
        2. Generator updated but doc not regenerated
        3. Template JSON files changed without regenerating doc

        Fix:
          python scripts/generate_hyperparameters_doc.py
        Or:
          GOLDEN_TEST_WRITE=true pytest tests/test_hyperparameters_doc_generator.py::TestHyperparametersDocValidation::test_generated_doc_matches_disk
        """
        if os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes"):
            print("\nGOLDEN_TEST_WRITE enabled - regenerating HYPERPARAMETERS.md")
            run_generation()

        matches = check_generation(show_diff=False)

        if not matches:
            error_lines = [
                "",
                "=" * 70,
                "HYPERPARAMETERS.md is out of sync with generator",
                "=" * 70,
            ]

            if not OUTPUT_FILE.exists():
                error_lines.append("  Status: [MISSING] docs/HYPERPARAMETERS.md")
            else:
                error_lines.append("  Status: [DIFFERS] docs/HYPERPARAMETERS.md")

            error_lines.extend(
                [
                    "",
                    "To fix:",
                    "  python scripts/generate_hyperparameters_doc.py",
                    "",
                    "Or with test:",
                    "  GOLDEN_TEST_WRITE=true pytest tests/test_hyperparameters_doc_generator.py",
                    "",
                    "To see diffs:",
                    "  python scripts/generate_hyperparameters_doc.py --check --diff",
                    "=" * 70,
                ]
            )
            pytest.fail("\n".join(error_lines))
