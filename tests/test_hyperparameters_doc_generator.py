"""
Tests for HYPERPARAMETERS.md documentation generator.

Validates that:
1. docs/HYPERPARAMETERS.md matches generator output (golden test)
2. Every resolved override parameter appears in the doc, regardless of category
3. Every template JSON file is loadable and has at least one template
4. Every template has a display_name
5. All framework JSON paths resolve to existing files

Environment variables:
- GOLDEN_TEST_WRITE=true: Regenerate HYPERPARAMETERS.md before validation
"""

import json
import os
import re
import sys
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from generate_hyperparameters_doc import (
    FRAMEWORKS,
    OUTPUT_FILE,
    TEMPLATIZATION_DIR,
    check_generation,
    run_generation,
)

from utils.resolve_override_params import resolve_params


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

    def test_all_resolved_override_params_appear_in_doc(self):
        """
        Every resolved override parameter must appear in HYPERPARAMETERS.md under
        its corresponding template section, regardless of `category`.

        `category` drives UI rendering only; every resolved param is overridable
        through the recipes repo, so the doc lists all of them. Checking all
        categories means a param silently dropping out of the doc fails here.
        """
        assert OUTPUT_FILE.exists(), (
            f"HYPERPARAMETERS.md not found at {OUTPUT_FILE}. " f"Run: python scripts/generate_hyperparameters_doc.py"
        )

        doc_content = OUTPUT_FILE.read_text(encoding="utf-8")

        # Load base params
        base_path = TEMPLATIZATION_DIR / "base_override_parameters.json"
        with open(base_path) as f:
            base_all = json.load(f)

        combined_base = {}
        combined_base.update(base_all.get("evaluation", {}))
        combined_base.update(base_all.get("fine_tuning", {}))

        missing = []

        for fw_title, rel_path in FRAMEWORKS:
            template_path = TEMPLATIZATION_DIR / rel_path
            if not template_path.exists():
                continue

            with open(template_path) as f:
                data = json.load(f)

            templates = data.get("templates", {})
            for template_key, template_data in templates.items():
                display_name = template_data.get("display_name", template_key)
                template_overrides = template_data.get("override_parameters", {})
                recipe_template = template_data.get("recipe_template", {})

                resolved = resolve_params(combined_base, template_overrides, recipe_template)

                for param_name in resolved:
                    # Check that this param appears in the doc as a backtick-quoted name
                    # within a table row (| `param_name` | ...)
                    pattern = rf"\| `{re.escape(param_name)}` \|"
                    if not re.search(pattern, doc_content):
                        missing.append(f"{fw_title} > {display_name} > {param_name}")

        assert not missing, (
            f"{len(missing)} resolved override parameter(s) missing from HYPERPARAMETERS.md:\n"
            + "\n".join(f"  - {m}" for m in missing[:20])
            + (f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else "")
            + "\n\nFix: python scripts/generate_hyperparameters_doc.py"
        )
