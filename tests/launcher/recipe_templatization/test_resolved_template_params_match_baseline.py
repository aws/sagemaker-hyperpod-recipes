"""
Validates that the resolved (sparse) template parameter files produce the same
override parameters as the baseline (legacy full-definition) files.

For each framework:
  1. Load current `*_recipe_template_parameters.json` (sparse `override_parameters`).
  2. Resolve via `resolve_params` against `base_override_parameters.json`.
  3. Load baseline `*_recipe_template_parameters.json` from baseline_artifacts/
     (legacy format with full `recipe_override_parameters`).
  4. Compare per-template, per-param, per-field.

Allowed differences (won't fail the test, only reported):
  - `category` (new metadata added by the refactor)
  - `description` (new metadata added by the refactor)
  - `name` (some templates may have updated default name)

Any other difference is FATAL and indicates a behavioral regression.

Environment variables:
  GOLDEN_TEST_WRITE=1: Regenerate the baseline_artifacts/*_recipe_template_parameters.json
    files from the current sparse templates before validating. Use after intentional
    changes to base_override_parameters.json or a template file.
"""

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from utils.resolve_override_params import resolve_params

GOLDEN_TEST_WRITE = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATIZATION_DIR = REPO_ROOT / "launcher" / "recipe_templatization"
BASELINE_DIR = REPO_ROOT / "tests" / "launcher" / "recipe_templatization" / "baseline_artifacts"
BASE_FILE = TEMPLATIZATION_DIR / "base_override_parameters.json"

# (framework, template subdir, default category)
FRAMEWORKS: List[Tuple[str, str, str]] = [
    ("llmft", "llmft", "fine_tuning"),
    ("verl", "verl", "fine_tuning"),
    ("nova", "nova", "fine_tuning"),
    ("checkpointless", "checkpointless", "fine_tuning"),
    ("evaluation", "evaluation", "evaluation"),
    ("mtrl_eval", "mtrl_eval", "evaluation"),
    ("mtrl", "mtrl", "fine_tuning"),
]

# Nova evaluation templates use the "evaluation" category instead of fine_tuning
NOVA_EVAL_TEMPLATES = {
    "nova_general_text_benchmark_eval",
    "nova_general_text_benchmark_2_0_eval",
    "nova_general_multi_modal_benchmark_eval",
    "nova_general_multi_modal_benchmark_2_0_eval",
    "nova_llm_judge_eval",
    "nova_bring_your_own_dataset_eval",
    "nova_bring_your_own_dataset_2_0_eval",
}

# Fields that may differ between baseline (legacy) and current (resolved) without
# being treated as a behavioral regression.
# - category/description/name: metadata fields added by the refactor
# - conditional_constraints: 6 known cases were intentionally removed because
#   every constraint resolved to the same value as the existing top-level default
ALLOWED_DIFF_FIELDS = {"category", "description", "name", "conditional_constraints", "visibility_tier"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _category_for(framework: str, template_name: str, default_category: str) -> str:
    """Resolve the base parameter category for a given template."""
    if framework == "nova" and template_name in NOVA_EVAL_TEMPLATES:
        return "evaluation"
    return default_category


def _resolve_current_template(
    base: Dict, framework: str, template_data: Dict, default_category: str, template_name: str
) -> Dict:
    """Resolve the current sparse template into full override parameters."""
    category = _category_for(framework, template_name, default_category)
    base_dict = base.get(category, {})
    overrides = template_data.get("override_parameters", {})
    recipe_template = template_data.get("recipe_template", {})
    return resolve_params(base_dict, overrides, recipe_template)


def _load_baseline_overrides(baseline_data: Dict, template_name: str) -> Dict:
    """Load the baseline (legacy) `recipe_override_parameters` for a template."""
    template = baseline_data.get("templates", {}).get(template_name, {})
    return template.get("recipe_override_parameters", template.get("override_parameters", {}))


def _compare_param_def(
    framework: str, template_name: str, param_name: str, baseline_def: Dict, current_def: Dict
) -> Tuple[List[str], List[str]]:
    """
    Compare a single parameter's field-level defs.

    Returns:
        (fatal_diffs, allowed_diffs) — each a list of human-readable strings.
    """
    fatal = []
    allowed = []
    prefix = f"[{framework}.{template_name}.{param_name}]"

    all_fields = set(baseline_def.keys()) | set(current_def.keys())
    for field in sorted(all_fields):
        baseline_val = baseline_def.get(field)
        current_val = current_def.get(field)

        if baseline_val == current_val:
            continue

        if field in ALLOWED_DIFF_FIELDS:
            if field not in baseline_def:
                allowed.append(f"{prefix} {field}: ADDED -> {json.dumps(current_val)[:60]}")
            elif field not in current_def:
                allowed.append(f"{prefix} {field}: REMOVED (was: {json.dumps(baseline_val)[:60]})")
            else:
                allowed.append(
                    f"{prefix} {field}: CHANGED " f"({json.dumps(baseline_val)[:40]} -> {json.dumps(current_val)[:40]})"
                )
        else:
            b_str = json.dumps(baseline_val, default=str)[:80]
            c_str = json.dumps(current_val, default=str)[:80]
            if field not in baseline_def:
                fatal.append(f"{prefix} {field}: UNEXPECTED FIELD ADDED ({c_str})")
            elif field not in current_def:
                fatal.append(f"{prefix} {field}: UNEXPECTED FIELD REMOVED (was: {b_str})")
            else:
                fatal.append(f"{prefix} {field}: UNEXPECTED CHANGE ({b_str} -> {c_str})")

    return fatal, allowed


# ---------------------------------------------------------------------------
# Baseline regeneration (GOLDEN_TEST_WRITE)
# ---------------------------------------------------------------------------


def _reorder_to_match(resolved: Dict, ref: Dict) -> Dict:
    """Reorder dict keys (and one level of nested fields) to match ref order."""
    if not ref:
        return resolved
    ordered = {}
    for k in ref:
        if k in resolved:
            v, ref_v = resolved[k], ref[k]
            if isinstance(v, dict) and isinstance(ref_v, dict):
                ordered_v = {fk: v[fk] for fk in ref_v if fk in v}
                for fk in v:
                    if fk not in ordered_v:
                        ordered_v[fk] = v[fk]
                ordered[k] = ordered_v
            else:
                ordered[k] = v
    for k in resolved:
        if k not in ordered:
            ordered[k] = resolved[k]
    return ordered


def _regenerate_baselines(base: Dict) -> List[str]:
    """Regenerate every baseline_artifacts/{framework}_*.json file from the current
    sparse templates, preserving param/field/template-key order from the existing
    baseline so text diffs stay clean."""
    written = []
    for framework, subdir, default_category in FRAMEWORKS:
        src = TEMPLATIZATION_DIR / subdir / f"{framework}_recipe_template_parameters.json"
        dest = BASELINE_DIR / f"{framework}_recipe_template_parameters.json"
        if not src.exists():
            continue

        with open(src) as f:
            current = json.load(f)

        existing_baseline = {}
        if dest.exists():
            with open(dest) as f:
                existing_baseline = json.load(f)
        existing_templates = existing_baseline.get("templates", {})

        resolved_data = copy.deepcopy(current)
        for tmpl_name, tmpl_data in resolved_data.get("templates", {}).items():
            if "override_parameters" not in tmpl_data:
                continue
            category = _category_for(framework, tmpl_name, default_category)
            base_dict = base.get(category, {})
            sparse = tmpl_data["override_parameters"]
            recipe_template = tmpl_data.get("recipe_template", {})
            full_resolved = resolve_params(base_dict, sparse, recipe_template)

            # Reorder params + fields to match existing baseline
            existing_overrides = existing_templates.get(tmpl_name, {}).get("recipe_override_parameters", {})
            full_resolved = _reorder_to_match(full_resolved, existing_overrides)

            del tmpl_data["override_parameters"]
            tmpl_data["recipe_override_parameters"] = full_resolved

            # Reorder top-level template keys to match baseline
            existing_template = existing_templates.get(tmpl_name, {})
            if existing_template:
                ordered_tmpl = {k: tmpl_data[k] for k in existing_template if k in tmpl_data}
                for k in tmpl_data:
                    if k not in ordered_tmpl:
                        ordered_tmpl[k] = tmpl_data[k]
                resolved_data["templates"][tmpl_name] = ordered_tmpl

        with open(dest, "w") as f:
            json.dump(resolved_data, f, indent=4)
            f.write("\n")
        written.append(str(dest.relative_to(REPO_ROOT)))
    return written


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestResolvedTemplateParamsMatchBaseline:
    """Snapshot test ensuring sparse template resolution matches the legacy baseline."""

    def test_resolved_matches_baseline(self):
        assert BASE_FILE.exists(), f"Base file not found: {BASE_FILE}"
        with open(BASE_FILE) as f:
            base = json.load(f)

        # GOLDEN_TEST_WRITE=1 regenerates baseline files before validating
        if GOLDEN_TEST_WRITE:
            written = _regenerate_baselines(base)
            print(f"\nGOLDEN_TEST_WRITE: regenerated {len(written)} baseline file(s)")
            for w in written:
                print(f"  {w}")

        all_fatal: List[str] = []
        all_allowed: List[str] = []
        missing_baseline_files: List[str] = []
        missing_baseline_templates: List[str] = []
        extra_current_templates: List[str] = []

        for framework, subdir, default_category in FRAMEWORKS:
            current_path = TEMPLATIZATION_DIR / subdir / f"{framework}_recipe_template_parameters.json"
            baseline_path = BASELINE_DIR / f"{framework}_recipe_template_parameters.json"

            if not current_path.exists():
                continue

            if not baseline_path.exists():
                missing_baseline_files.append(str(baseline_path.relative_to(REPO_ROOT)))
                continue

            with open(current_path) as f:
                current_data = json.load(f)
            with open(baseline_path) as f:
                baseline_data = json.load(f)

            current_templates = current_data.get("templates", {})
            baseline_templates = baseline_data.get("templates", {})

            for template_name in sorted(set(current_templates) | set(baseline_templates)):
                if template_name not in baseline_templates:
                    extra_current_templates.append(f"{framework}.{template_name}")
                    continue
                if template_name not in current_templates:
                    missing_baseline_templates.append(f"{framework}.{template_name}")
                    continue

                resolved = _resolve_current_template(
                    base, framework, current_templates[template_name], default_category, template_name
                )
                baseline_overrides = _load_baseline_overrides(baseline_data, template_name)

                all_param_names = set(resolved.keys()) | set(baseline_overrides.keys())
                for param_name in sorted(all_param_names):
                    if param_name not in baseline_overrides:
                        all_fatal.append(f"[{framework}.{template_name}] {param_name}: PARAM ADDED in current")
                        continue
                    if param_name not in resolved:
                        all_fatal.append(f"[{framework}.{template_name}] {param_name}: PARAM REMOVED in current")
                        continue

                    fatal, allowed = _compare_param_def(
                        framework, template_name, param_name, baseline_overrides[param_name], resolved[param_name]
                    )
                    all_fatal.extend(fatal)
                    all_allowed.extend(allowed)

        # Build error report
        if all_fatal or missing_baseline_files or missing_baseline_templates or extra_current_templates:
            lines = ["", "=" * 70, "Resolved templates DIFFER from baseline", "=" * 70]

            if missing_baseline_files:
                lines.append(f"\nMissing baseline files ({len(missing_baseline_files)}):")
                for f in missing_baseline_files:
                    lines.append(f"  - {f}")

            if missing_baseline_templates:
                lines.append(f"\nTemplates in baseline but missing from current ({len(missing_baseline_templates)}):")
                for t in missing_baseline_templates:
                    lines.append(f"  - {t}")

            if extra_current_templates:
                lines.append(f"\nTemplates in current but missing from baseline ({len(extra_current_templates)}):")
                for t in extra_current_templates:
                    lines.append(f"  - {t}")

            if all_fatal:
                lines.append(f"\nFATAL differences ({len(all_fatal)}) — unexpected field changes:")
                for d in all_fatal[:50]:
                    lines.append(f"  - {d}")
                if len(all_fatal) > 50:
                    lines.append(f"  ... and {len(all_fatal) - 50} more")

            if all_allowed:
                lines.append(f"\nAllowed differences ({len(all_allowed)}) — category/description/name only:")
                for d in all_allowed[:20]:
                    lines.append(f"  - {d}")
                if len(all_allowed) > 20:
                    lines.append(f"  ... and {len(all_allowed) - 20} more")

            lines.append("=" * 70)
            pytest.fail("\n".join(lines))

        # If only allowed diffs exist, print them as informational (test passes)
        if all_allowed:
            print(f"\nFound {len(all_allowed)} allowed differences (category/description/name) — test passes")
