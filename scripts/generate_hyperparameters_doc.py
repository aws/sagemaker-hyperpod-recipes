#!/usr/bin/env python3
"""Generate HYPERPARAMETERS.md documentation from recipe template parameter JSON files.

All parameter metadata is sourced from the *_recipe_template_parameters.json files
under launcher/recipe_templatization/.

A parameter is included if:
  - It has a 'category' field (any value, e.g. 'hyperparameter'), OR
  - It has min/max/enum constraints and is not a purely infrastructure param

Usage:
    python scripts/generate_hyperparameters_doc.py           # Generate docs/HYPERPARAMETERS.md
    python scripts/generate_hyperparameters_doc.py --check   # Validate docs/HYPERPARAMETERS.md is up-to-date
    python scripts/generate_hyperparameters_doc.py --check --diff  # Show what changed
"""

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATIZATION_DIR = REPO_ROOT / "launcher" / "recipe_templatization"
OUTPUT_FILE = REPO_ROOT / "docs" / "HYPERPARAMETERS.md"

# =============================================================================
# FRAMEWORK REGISTRY — (display title, relative path to JSON)
# =============================================================================

FRAMEWORKS: List[Tuple[str, str]] = [
    (
        "LLMFT (LLM Fine-Tuning Framework)",
        "llmft/llmft_recipe_template_parameters.json",
    ),
    (
        "VERL (Versatile Reinforcement Learning)",
        "verl/verl_recipe_template_parameters.json",
    ),
    (
        "Amazon Nova",
        "nova/nova_recipe_template_parameters.json",
    ),
    (
        "Checkpointless",
        "checkpointless/checkpointless_recipe_template_parameters.json",
    ),
    (
        "Evaluation",
        "evaluation/evaluation_recipe_template_parameters.json",
    ),
]


def _template_display_name(key: str, template_data: Dict) -> str:
    if "display_name" not in template_data:
        raise KeyError(f"Template '{key}' is missing a 'display_name' field in its JSON file.")
    return template_data["display_name"]


# =============================================================================
# PARAMETER SELECTION
# =============================================================================


def _should_include(spec: Any) -> bool:
    """Include a parameter if it carries a 'category' annotation (any value),
    or has min/max/enum constraints and is not a pure infrastructure param."""
    if not isinstance(spec, dict):
        return False
    if "category" in spec:
        return True
    has_constraints = any(k in spec for k in ("min", "max", "enum"))
    return has_constraints


# =============================================================================
# VALUE FORMATTING
# =============================================================================

_MAX_ENUM_INLINE = 10  # Show at most this many enum values inline


def _fmt_number(v) -> str:
    """Format a numeric value, using scientific notation for small floats (abs < 0.01)."""
    if not isinstance(v, float):
        return str(v)
    if v == 0.0:
        return "0.0"
    if 0 < abs(v) < 0.01:
        s = f"{v:.2e}"  # e.g. "1.00e-04"
        m, e = s.split("e")
        m = m.rstrip("0").rstrip(".")
        return f"{m}e{int(e):+03d}"  # e.g. "1e-04", "5e-07"
    return str(v)


def _format_range(spec: dict) -> str:
    if "enum" in spec:
        values = [str(v) for v in spec["enum"]]
        if len(values) <= _MAX_ENUM_INLINE:
            return ", ".join(values)
        shown = ", ".join(values[:4])
        return f"{shown}, … ({len(values)} values)"
    parts = []
    if "min" in spec:
        parts.append(_fmt_number(spec["min"]))
    if "max" in spec:
        parts.append(_fmt_number(spec["max"]))
    if len(parts) == 2:
        return f"{parts[0]}–{parts[1]}"
    if parts:
        prefix = "≥" if "min" in spec else "≤"
        return f"{prefix} {parts[0]}"
    return "—"


# =============================================================================
# TABLE GENERATION
# =============================================================================


def _make_param_table(params: Dict[str, Any]) -> List[str]:
    included = [(name, spec) for name, spec in params.items() if _should_include(spec)]
    if not included:
        return []

    cols = [
        ("Parameter", lambda n, s: f"`{n}`"),
        ("Type", lambda n, s: s.get("type") or "—"),
        ("Required", lambda n, s: "Yes" if s.get("required") else "No"),
        ("Range / Values", lambda n, s: _format_range(s)),
        ("Description", lambda n, s: (s.get("description") or "—").replace("|", "\\|")),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("-----" for _ in cols) + "|"
    rows = ["| " + " | ".join(fn(n, s) for _, fn in cols) + " |" for n, s in included]
    return [header, sep] + rows


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================


def _load_templates(rel_path: str) -> Dict[str, Any]:
    path = TEMPLATIZATION_DIR / rel_path
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("templates", {})
    except Exception:
        return {}


def _fw_anchor(title: str) -> str:
    return title.lower().replace(" ", "-").replace("(", "").replace(")", "")


def _generate_framework_section(fw_title: str, rel_path: str) -> List[str]:
    templates = _load_templates(rel_path)
    if not templates:
        return []

    lines = [f"## {fw_title}", ""]

    for template_key, template_data in templates.items():
        params = template_data.get("recipe_override_parameters", {})
        display = _template_display_name(template_key, template_data)
        lines.append(f"### {display}")
        lines.append("")
        table = _make_param_table(params)
        if table:
            lines.extend(table)
        else:
            lines.append("_No configurable parameters._")
        lines.append("")

    return lines


def generate_markdown() -> str:
    lines = [
        "# HyperPod Recipe Hyperparameter Reference",
        "",
        "This document contains the list of hyperparameters available when using the recipes repo through "
        "SMTJ Serverless Model Customization. All parameters are available in serverful usage "
        "but these are the ranges we recommend using for successful results.",
        "",
        "## Table of Contents",
        "",
    ]
    for fw_title, _ in FRAMEWORKS:
        anchor = _fw_anchor(fw_title)
        lines.append(f"- [{fw_title}](#{anchor})")
    lines.append("")

    for fw_title, rel_path in FRAMEWORKS:
        lines.extend(_generate_framework_section(fw_title, rel_path))

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def run_generation():
    print(f"Reading templates from: {TEMPLATIZATION_DIR}")
    content = generate_markdown()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"Generated: {OUTPUT_FILE}")


def check_generation(show_diff: bool = False) -> bool:
    print("Generating expected content...")
    expected = generate_markdown()
    if not OUTPUT_FILE.exists():
        if show_diff:
            print(f"MISSING: {OUTPUT_FILE}")
        return False
    actual = OUTPUT_FILE.read_text(encoding="utf-8")
    if actual == expected:
        return True
    if show_diff:
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile="a/docs/HYPERPARAMETERS.md",
            tofile="b/docs/HYPERPARAMETERS.md",
            lineterm="",
        )
        print("".join(diff))
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate HYPERPARAMETERS.md from recipe template parameter JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s          Generate HYPERPARAMETERS.md\n  %(prog)s --check  Check if up-to-date",
    )
    parser.add_argument("--check", action="store_true", help="Check if generated content matches disk")
    parser.add_argument("--diff", action="store_true", help="Show unified diff (requires --check)")
    args = parser.parse_args()

    if args.check:
        if check_generation(show_diff=args.diff):
            print("✓ HYPERPARAMETERS.md is up-to-date.")
            sys.exit(0)
        print(f"\n{'=' * 70}\nERROR: HYPERPARAMETERS.md is out of sync\n{'=' * 70}\nTo fix: python {sys.argv[0]}\n")
        sys.exit(1)
    run_generation()


if __name__ == "__main__":
    main()
