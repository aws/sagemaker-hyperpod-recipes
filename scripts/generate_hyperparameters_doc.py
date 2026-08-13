#!/usr/bin/env python3
"""Generate HYPERPARAMETERS.md documentation from recipe template parameter JSON files.

All parameter metadata is sourced from the *_recipe_template_parameters.json files
under launcher/recipe_templatization/, merged with base_override_parameters.json.

Every resolved override parameter is documented. The `category` field
(hyperparameter vs system) drives UI rendering only — it does not indicate
whether a value can be overridden, and every parameter listed here can be.

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
    """Include every resolved override parameter.

    `category` (hyperparameter vs system) exists to tell the UI which fields to
    render as tunable knobs; it says nothing about whether a value can be
    overridden. As far as this repo is concerned every resolved override
    parameter is overridable, so all of them are documented.
    """
    return isinstance(spec, dict)


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


# verl clamps these length params' max to each recipe's supported sequence
# length at recipe-processing time (see VerlRecipeTemplateProcessor). The static
# max in the base config is only an upper bound, so for verl we render the max as
# the per-recipe ceiling rather than the misleading static number.
_VERL_SEQUENCE_LENGTH_CLAMPED_PARAMS = frozenset({"max_prompt_length", "max_response_length", "dataset_max_len"})
_VERL_SEQUENCE_LENGTH_CEILING = "recipe sequence length"


def _format_range(spec: dict, *, clamp_max_to_sequence_length: bool = False) -> str:
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
        # For verl length params the effective max is the recipe's computed
        # sequence-length ceiling, not the static base max.
        parts.append(_VERL_SEQUENCE_LENGTH_CEILING if clamp_max_to_sequence_length else _fmt_number(spec["max"]))
    if len(parts) == 2:
        return f"{parts[0]}–{parts[1]}"
    if parts:
        prefix = "≥" if "min" in spec else "≤"
        return f"{prefix} {parts[0]}"
    return "—"


# =============================================================================
# TABLE GENERATION
# =============================================================================


def _make_param_table(params: Dict[str, Any], *, is_verl: bool = False) -> List[str]:
    included = [(name, spec) for name, spec in params.items() if _should_include(spec)]
    if not included:
        return []

    def _range(n: str, s: dict) -> str:
        clamp = is_verl and n in _VERL_SEQUENCE_LENGTH_CLAMPED_PARAMS
        return _format_range(s, clamp_max_to_sequence_length=clamp)

    cols = [
        ("Parameter", lambda n, s: f"`{n}`"),
        ("Type", lambda n, s: s.get("type") or "—"),
        ("Required", lambda n, s: "Yes" if s.get("required") else "No"),
        ("Range / Values", _range),
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


sys.path.insert(0, str(REPO_ROOT))
from utils.resolve_override_params import resolve_params


def _resolve_override_parameters(template_data: Dict) -> Dict:
    """Resolve override parameters using the shared resolve_params function."""
    base_path = TEMPLATIZATION_DIR / "base_override_parameters.json"
    if not base_path.exists():
        return template_data.get("override_parameters", {})

    with open(base_path) as f:
        base_all = json.load(f)

    combined_base = {}
    combined_base.update(base_all.get("evaluation", {}))
    combined_base.update(base_all.get("fine_tuning", {}))

    template_overrides = template_data.get("override_parameters", {})
    recipe_template = template_data.get("recipe_template", {})

    return resolve_params(combined_base, template_overrides, recipe_template)


def _generate_framework_section(fw_title: str, rel_path: str) -> List[str]:
    templates = _load_templates(rel_path)
    if not templates:
        return []

    is_verl = rel_path.startswith("verl/")

    lines = [f"## {fw_title}", ""]

    if is_verl:
        # The length params' max is not the static base value for verl — the
        # recipe processor clamps it to each recipe's supported sequence length.
        lines.append(
            "> **Note:** For verl recipes, `max_prompt_length`, "
            "`max_response_length`, and `dataset_max_len` are capped at the "
            "recipe's supported sequence length (derived from the recipe's "
            "per-GPU token budget), not the static upper bound. The effective "
            "maximum is per-recipe; the ranges below show it as "
            f'"{_VERL_SEQUENCE_LENGTH_CEILING}".'
        )
        lines.append("")

    for template_key, template_data in templates.items():
        params = _resolve_override_parameters(template_data)
        display = _template_display_name(template_key, template_data)
        lines.append(f"### {display}")
        lines.append("")
        table = _make_param_table(params, is_verl=is_verl)
        if table:
            lines.extend(table)
        else:
            lines.append("_No configurable parameters._")
        lines.append("")

    return lines


def generate_markdown() -> str:
    lines = [
        "# HyperPod Recipe Overridable Parameter Reference",
        "",
        "This document contains the list of parameters that can be overridden when using the recipes repo "
        "through SMTJ Serverless Model Customization. All parameters are available in serverful usage "
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
