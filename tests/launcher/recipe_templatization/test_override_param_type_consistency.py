"""
Test that override parameter type families are consistent within each template
category (fine_tuning, evaluation).

Type families:
  - "numeric": integer, float, number (all compatible)
  - "string": string
  - "boolean": boolean

The test validates:
  1. The base_override_parameters.json file has internally consistent type/type_family
  2. Template-level overrides that specify a `type` field do not conflict with the
     base type_family for that parameter
  3. No cross-framework type family conflicts exist within each category

The base_override_parameters.json file IS the source of truth for type information.
No separate baseline file is needed.

Usage:
  pytest tests/launcher/recipe_templatization/test_override_param_type_consistency.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from utils.resolve_override_params import resolve_bound_placeholders, resolve_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path("launcher/recipe_templatization")
BASE_PARAMS_PATH = TEMPLATE_DIR / "base_override_parameters.json"
GOLDEN_WRITE = os.environ.get("GOLDEN_TEST_WRITE", "").lower() in ("true", "1", "yes")

# Nova eval template names — everything else in nova is train
NOVA_EVAL_TEMPLATES = {
    "nova_general_text_benchmark_eval",
    "nova_general_text_benchmark_2_0_eval",
    "nova_general_multi_modal_benchmark_eval",
    "nova_general_multi_modal_benchmark_2_0_eval",
    "nova_llm_judge_eval",
    "nova_bring_your_own_dataset_eval",
    "nova_bring_your_own_dataset_2_0_eval",
}

TYPE_FAMILIES = {
    "integer": "numeric",
    "float": "numeric",
    "number": "numeric",
    "string": "string",
    "boolean": "boolean",
}

# Map each framework to its category
FRAMEWORK_CATEGORIES = {
    "llmft": ("fine_tuning", None),
    "verl": ("fine_tuning", None),
    "mtrl": ("fine_tuning", None),
    "mtrl_eval": ("evaluation", None),
    "nova": None,  # Special: split into train and eval
    "evaluation": ("evaluation", None),
    "checkpointless": ("fine_tuning", None),
}


def _type_family(raw_type: str) -> str:
    return TYPE_FAMILIES.get(raw_type, raw_type)


def _get_category_for_template(framework: str, template_name: str) -> str:
    """Determine which category a (framework, template) belongs to."""
    if framework == "nova":
        return "evaluation" if template_name in NOVA_EVAL_TEMPLATES else "fine_tuning"
    mapping = FRAMEWORK_CATEGORIES.get(framework)
    if mapping is None:
        return "unknown"
    return mapping[0]


# ---------------------------------------------------------------------------
# Load base override parameters
# ---------------------------------------------------------------------------


def _load_base_params() -> Dict:
    with open(BASE_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Auto-discover all template parameter files
# ---------------------------------------------------------------------------


def _discover_template_files() -> Dict[str, Path]:
    """
    Find all *_recipe_template_parameters.json under launcher/recipe_templatization/.
    Returns {framework_name: path}.
    """
    files = {}
    for subdir in TEMPLATE_DIR.iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob("*_recipe_template_parameters.json"):
            framework = subdir.name
            files[framework] = f
    return files


# ---------------------------------------------------------------------------
# Collect template-level type overrides
# ---------------------------------------------------------------------------


def _collect_template_type_overrides() -> List[Tuple[str, str, str, str, str]]:
    """
    Scan template override_parameters for any `type` field overrides.
    Returns list of (category, framework, template_name, param_name, type_family).
    """
    instances = []
    template_files = _discover_template_files()

    for framework, fpath in sorted(template_files.items()):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tname, tdata in data.get("templates", {}).items():
            category = _get_category_for_template(framework, tname)
            overrides = tdata.get("override_parameters", {})
            for pname, pdef in overrides.items():
                if "type" in pdef:
                    raw_type = pdef["type"]
                    family = _type_family(raw_type)
                    instances.append((category, framework, tname, pname, family))

    return instances


# ---------------------------------------------------------------------------
# Module-level data collection
# ---------------------------------------------------------------------------

BASE_PARAMS = _load_base_params()
TEMPLATE_TYPE_OVERRIDES = _collect_template_type_overrides()


# ---------------------------------------------------------------------------
# Test 1: Base file internal consistency — type matches type_family
# ---------------------------------------------------------------------------


def _get_base_param_instances() -> List[Tuple[str, str, str, str]]:
    """
    Returns list of (category, param_name, type, type_family) from the base file.
    """
    instances = []
    for category in ("fine_tuning", "evaluation"):
        cat_params = BASE_PARAMS.get(category, {})
        for pname, pdef in cat_params.items():
            raw_type = pdef.get("type", "MISSING")
            type_family = pdef.get("type_family", "MISSING")
            instances.append((category, pname, raw_type, type_family))
    return instances


BASE_PARAM_INSTANCES = _get_base_param_instances()


@pytest.mark.parametrize(
    "category,param_name,raw_type,type_family",
    BASE_PARAM_INSTANCES,
    ids=[f"{i[0]}|{i[1]}" for i in BASE_PARAM_INSTANCES],
)
def test_base_type_family_matches_type(category, param_name, raw_type, type_family):
    """Each parameter in base_override_parameters.json must have type_family consistent with type."""
    expected_family = _type_family(raw_type)
    assert type_family == expected_family, (
        f"Type family mismatch in base file for '{param_name}' ({category}): "
        f"type='{raw_type}' should map to type_family='{expected_family}', "
        f"but got type_family='{type_family}'."
    )


# ---------------------------------------------------------------------------
# Test 2: Template type overrides don't conflict with base type_family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category,framework,template_name,param_name,override_family",
    TEMPLATE_TYPE_OVERRIDES,
    ids=[f"{i[0]}|{i[1]}::{i[2]}::{i[3]}" for i in TEMPLATE_TYPE_OVERRIDES],
)
def test_template_type_override_consistent_with_base(category, framework, template_name, param_name, override_family):
    """
    Template-level type overrides must have the same type_family as the base definition.
    A template can override 'integer' to 'number' (both numeric), but not 'string' to 'integer'.
    """
    cat_params = BASE_PARAMS.get(category, {})
    if param_name not in cat_params:
        # Parameter only exists in template overrides, no base to conflict with
        return

    base_family = cat_params[param_name].get("type_family", "MISSING")
    assert override_family == base_family, (
        f"Type family conflict for '{param_name}' in {framework}::{template_name} ({category}): "
        f"base type_family='{base_family}', but template override type maps to '{override_family}'. "
        f"This means the template override introduces a type incompatibility."
    )


# ---------------------------------------------------------------------------
# Test 3: No cross-framework type family conflicts within each category
#          (combining base definitions + template overrides)
# ---------------------------------------------------------------------------


def _get_categories() -> List[str]:
    """Get all categories from the base file."""
    return sorted(k for k in BASE_PARAMS.keys() if k != "_comment")


@pytest.mark.parametrize("category", _get_categories())
def test_no_type_family_conflicts_within_category(category):
    """
    For every param within a category, all type references (base + template overrides)
    must share the same type family.
    """
    # Start with base definitions
    param_families = defaultdict(lambda: defaultdict(list))
    cat_params = BASE_PARAMS.get(category, {})

    for pname, pdef in cat_params.items():
        family = pdef.get("type_family", "MISSING")
        param_families[pname][family].append("base_override_parameters.json")

    # Add template-level type overrides
    for cat, framework, tname, pname, family in TEMPLATE_TYPE_OVERRIDES:
        if cat == category:
            param_families[pname][family].append(f"{framework}::{tname}")

    conflicts = []
    for pname, families in sorted(param_families.items()):
        if len(families) > 1:
            detail = {f: sorted(srcs) for f, srcs in families.items()}
            conflicts.append(f"  {pname}: {detail}")

    assert not conflicts, (
        f"Type family conflicts in '{category}':\n"
        + "\n".join(conflicts)
        + f"\n\nThis means the same parameter has incompatible types across sources."
    )


# ---------------------------------------------------------------------------
# Test 4: Every parameter in base file has required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _get_categories())
def test_base_params_have_required_fields(category):
    """Every parameter in the base file must have type, type_family, required,
    category, and visibility_tier fields.

    category and visibility_tier are required because the MFE reads both to
    decide whether a field renders and which section it renders in. A param
    missing either has undefined rendering behavior post-cutover.
    """
    cat_params = BASE_PARAMS.get(category, {})
    missing = []

    for pname, pdef in cat_params.items():
        for field in ("type", "type_family", "required", "category", "visibility_tier"):
            if field not in pdef:
                missing.append(f"  {pname}: missing '{field}'")

    assert not missing, f"Parameters in '{category}' missing required fields:\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# Test 5: No empty string fields in base_override_parameters.json (except `default`)
# ---------------------------------------------------------------------------


# Fields that are allowed to be empty (e.g., `default: ""` for optional string params)
_FIELDS_ALLOWED_TO_BE_EMPTY = {"default"}


@pytest.mark.parametrize("category", _get_categories())
def test_base_params_have_no_empty_fields(category):
    """Every field in every param def must be non-empty, except `default`.

    Catches accidental `"description": ""`, `"category": ""`, `"display_name": ""`, etc.
    A field is considered empty if it's an empty string, empty list, or empty dict.
    """
    cat_params = BASE_PARAMS.get(category, {})
    empties = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict):
            continue
        for field, value in pdef.items():
            if field in _FIELDS_ALLOWED_TO_BE_EMPTY:
                continue
            # Empty if string with no content, or empty container
            is_empty = (isinstance(value, str) and value.strip() == "") or (
                isinstance(value, (list, dict)) and len(value) == 0
            )
            if is_empty:
                empties.append(f"  {pname}.{field}: empty value {value!r}")

    assert not empties, f"Empty fields in '{category}' (only `default` may be empty):\n" + "\n".join(empties)


# ---------------------------------------------------------------------------
# Test 6: primary/advanced params must have a non-empty display_name; hint
#         if present must be non-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", _get_categories())
def test_primary_advanced_params_have_display_name_and_valid_hint(category):
    """Params with visibility_tier in {primary, advanced} must have a non-empty
    display_name. If a `hint` field is present, it must also be non-empty.

    Tertiary and untagged params are not checked.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict):
            continue
        tier = pdef.get("visibility_tier")
        if tier not in ("primary", "advanced"):
            continue

        display_name = pdef.get("display_name")
        if display_name is None:
            issues.append(f"  {pname} (tier={tier}): missing 'display_name'")
        elif not isinstance(display_name, str) or display_name.strip() == "":
            issues.append(f"  {pname} (tier={tier}): 'display_name' is empty")

        if "hint" in pdef:
            hint = pdef["hint"]
            if not isinstance(hint, str) or hint.strip() == "":
                issues.append(f"  {pname} (tier={tier}): 'hint' is present but empty")

    assert not issues, f"Primary/advanced params in '{category}' missing required UI metadata:\n" + "\n".join(issues)


# ---------------------------------------------------------------------------
# Test 7: category field, if present, must be one of the allowed values
# ---------------------------------------------------------------------------


ALLOWED_CATEGORY_VALUES = {"hyperparameter", "system"}


@pytest.mark.parametrize("category", _get_categories())
def test_category_field_uses_allowed_values(category):
    """If `category` is present on a param, it must be one of the allowed enum
    values: `hyperparameter` (tunable) or `system` (infrastructure/admin).

    Empty strings are caught separately by test_base_params_have_no_empty_fields.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict) or "category" not in pdef:
            continue
        value = pdef["category"]
        if value not in ALLOWED_CATEGORY_VALUES:
            issues.append(f"  {pname}: category={value!r} (allowed: {sorted(ALLOWED_CATEGORY_VALUES)})")

    assert not issues, f"Invalid category values in '{category}':\n" + "\n".join(issues)


# ---------------------------------------------------------------------------
# Test 7b: visibility_tier field must be one of the allowed values
# ---------------------------------------------------------------------------


ALLOWED_VISIBILITY_TIER_VALUES = {"primary", "advanced", "tertiary"}


@pytest.mark.parametrize("category", _get_categories())
def test_visibility_tier_field_uses_allowed_values(category):
    """`visibility_tier` must be one of `primary` (always-visible grid),
    `advanced` (collapsed section), or `tertiary` (not rendered in the generic
    form). A typo would silently change where the MFE renders the field.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict) or "visibility_tier" not in pdef:
            continue
        value = pdef["visibility_tier"]
        if value not in ALLOWED_VISIBILITY_TIER_VALUES:
            issues.append(f"  {pname}: visibility_tier={value!r} (allowed: {sorted(ALLOWED_VISIBILITY_TIER_VALUES)})")

    assert not issues, f"Invalid visibility_tier values in '{category}':\n" + "\n".join(issues)


# ---------------------------------------------------------------------------
# Test 7c–7g: self-describing UI metadata fields (control_type, step, and the
#             S3/name/namespace regex family)
#
# These fields let the MFE render each param from the published artifact instead
# of its hardcoded UI_CONTRACT_CONFIG table.
#
# pass_k_values HAZARD (flagged for engineers, not fixed here):
#   pass_k_values is type:"array" (type_family:"array") in BOTH sections. Today
#   the MFE's isValidOverrideParam drops it, so it never reaches a render branch
#   and is harmless. But the fine_tuning copy is visibility_tier:"tertiary", and
#   the day the MFE honors visibility_tier it will route pass_k_values into the
#   tertiary branch of parseUIContract.ts (~:173-183), which THROWS because
#   "array" is not a valid tertiary type — a whole-form outage. We deliberately
#   do NOT add a test that trips on pass_k_values (it must stay green); its
#   control_type is set like every other param. This comment is the flag.
# ---------------------------------------------------------------------------


# checkbox and toggle are intentionally excluded: neither has a render branch in
# the MFE's UIContractField, so booleans use "dropdown" (see gradient_clipping,
# merge_weights, reasoning_enabled, use_kl_loss, postprocessing).
ALLOWED_CONTROL_TYPE_VALUES = {
    "number_input",
    "text_input",
    "dropdown",
    "s3_uri_input",
    "kms_key_input",
}


@pytest.mark.parametrize("category", _get_categories())
def test_control_type_uses_allowed_values(category):
    """If `control_type` is present on a param, it must be one of the allowed
    render types the MFE understands.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict) or "control_type" not in pdef:
            continue
        value = pdef["control_type"]
        if value not in ALLOWED_CONTROL_TYPE_VALUES:
            issues.append(f"  {pname}: control_type={value!r} (allowed: {sorted(ALLOWED_CONTROL_TYPE_VALUES)})")

    assert not issues, f"Invalid control_type values in '{category}':\n" + "\n".join(issues)


@pytest.mark.parametrize("category", _get_categories())
def test_number_inputs_have_step(category):
    """Any param with control_type=='number_input' and no `enum` must define a
    numeric `step`, so the MFE's number spinner has an increment.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict):
            continue
        if pdef.get("control_type") != "number_input" or "enum" in pdef:
            continue
        step = pdef.get("step")
        if not isinstance(step, (int, float)) or isinstance(step, bool):
            issues.append(f"  {pname}: number_input missing numeric 'step' (got {step!r})")

    assert not issues, f"number_input params missing 'step' in '{category}':\n" + "\n".join(issues)


@pytest.mark.parametrize("category", _get_categories())
def test_step_within_range(category):
    """Where a param declares numeric `min` and `max` and a `step`, the step must
    satisfy 0 < step <= (max - min); a step larger than the range is unusable.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict) or "step" not in pdef:
            continue
        step, mn, mx = pdef.get("step"), pdef.get("min"), pdef.get("max")
        for label, val in (("min", mn), ("max", mx), ("step", step)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                break
        else:
            span = mx - mn
            if not (0 < step <= span):
                issues.append(f"  {pname}: step={step!r} not in (0, max-min={span!r}]")

    assert not issues, f"Invalid step relative to range in '{category}':\n" + "\n".join(issues)


@pytest.mark.parametrize("category", _get_categories())
def test_regex_pattern_compiles(category):
    """Every `regex_pattern` must compile under Python `re`, and (mirroring the
    MFE's `new RegExp(pattern).test(value)` semantics via re.search) must accept
    the param's own `default` when that default is a non-empty string.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict) or "regex_pattern" not in pdef:
            continue
        pattern = pdef["regex_pattern"]
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            issues.append(f"  {pname}: regex_pattern {pattern!r} does not compile ({e})")
            continue
        default = pdef.get("default")
        if isinstance(default, str) and default != "":
            if not compiled.search(default):
                issues.append(f"  {pname}: default {default!r} rejected by regex_pattern {pattern!r}")

    assert not issues, f"regex_pattern issues in '{category}':\n" + "\n".join(issues)


@pytest.mark.parametrize("category", _get_categories())
def test_invalid_format_error_requires_regex(category):
    """`invalid_format_error` is the message shown when `regex_pattern` fails, so
    it must never appear without a `regex_pattern` to trigger it.
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict):
            continue
        if "invalid_format_error" in pdef and "regex_pattern" not in pdef:
            issues.append(f"  {pname}: has 'invalid_format_error' but no 'regex_pattern'")

    assert not issues, f"invalid_format_error without regex_pattern in '{category}':\n" + "\n".join(issues)


# ---------------------------------------------------------------------------
# Test 8: data-type conformance — default / min / max / enum values must
#         match the declared type_family
# ---------------------------------------------------------------------------


# Python types allowed per type_family for value-bearing fields.
# - `bool` is intentionally excluded from numeric (Python booleans are subclasses of int,
#   but treating True/False as numeric here would mask real bugs).
TYPE_FAMILY_PYTHON_TYPES = {
    "numeric": (int, float),
    "string": (str,),
    "boolean": (bool,),
    "array": (list,),
}


def _matches_type_family(value, type_family: str) -> bool:
    """Return True if the Python value is compatible with the declared type_family."""
    allowed = TYPE_FAMILY_PYTHON_TYPES.get(type_family)
    if allowed is None:
        return False
    # Exclude Python booleans from being treated as numeric
    if type_family == "numeric" and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


@pytest.mark.parametrize("category", _get_categories())
def test_param_values_match_type_family(category):
    """Values in `default`, `min`, `max`, and entries of `enum` must conform
    to the param's declared `type_family`.

    For type_family == 'numeric', values must be int or float (not bool/string).
    For type_family == 'string', values must be str.
    For type_family == 'boolean', values must be bool.
    For type_family == 'array', `default` must be a list and its entries are
    not type-checked here (lists may hold mixed types).
    """
    cat_params = BASE_PARAMS.get(category, {})
    issues = []

    for pname, pdef in cat_params.items():
        if not isinstance(pdef, dict):
            continue
        type_family = pdef.get("type_family")
        if type_family is None or type_family not in TYPE_FAMILY_PYTHON_TYPES:
            continue

        # Scalar fields whose value type must match
        for field in ("default", "min", "max"):
            if field not in pdef:
                continue
            value = pdef[field]
            if value is None:
                # null is a valid sentinel (e.g. unbounded) — skip
                continue
            if not _matches_type_family(value, type_family):
                issues.append(
                    f"  {pname}.{field}={value!r} (type {type(value).__name__}) "
                    f"does not match type_family={type_family!r}"
                )

        # enum entries must each match
        if "enum" in pdef:
            enum_val = pdef["enum"]
            if not isinstance(enum_val, list):
                issues.append(f"  {pname}.enum is not a list: got {type(enum_val).__name__}")
                continue
            for i, item in enumerate(enum_val):
                if item is None:
                    continue
                if not _matches_type_family(item, type_family):
                    issues.append(
                        f"  {pname}.enum[{i}]={item!r} (type {type(item).__name__}) "
                        f"does not match type_family={type_family!r}"
                    )

    assert not issues, f"Type conformance violations in '{category}':\n" + "\n".join(issues)


# ---------------------------------------------------------------------------
# Test 9: {min}/{max} placeholders are resolved in the RESOLVED output, and the
#         substituted values agree with the param's actual bounds
# ---------------------------------------------------------------------------


# Display-text fields that may carry {min}/{max} placeholders in the base file.
_PLACEHOLDER_TEXT_FIELDS = ("description", "display_name", "hint")

# Matches the substituted form, e.g. "Must be a value between 0.001 and 0.1".
_BETWEEN_RE = re.compile(r"between\s+(\S+)\s+and\s+(\S+)")


def _resolved_params_by_template() -> List[Tuple[str, str, Dict]]:
    """Resolve every template's override params. Returns (framework, template, resolved).

    Applies `resolve_bound_placeholders` after `resolve_params`, mirroring the order
    the processors use: bounds are finalised first (a subclass may set `max` from the
    recipe), then display text is substituted. Calling only `resolve_params` here
    would test a state no published artifact is ever in.
    """
    out = []
    for framework, fpath in sorted(_discover_template_files().items()):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tname, tdata in data.get("templates", {}).items():
            category = _get_category_for_template(framework, tname)
            resolved = resolve_params(
                BASE_PARAMS.get(category, {}),
                tdata.get("override_parameters", {}),
                tdata.get("recipe_template", {}),
            )
            out.append((framework, tname, resolve_bound_placeholders(resolved)))
    return out


RESOLVED_BY_TEMPLATE = _resolved_params_by_template()


@pytest.mark.parametrize(
    "framework,template_name,resolved",
    RESOLVED_BY_TEMPLATE,
    ids=[f"{fw}::{t}" for fw, t, _ in RESOLVED_BY_TEMPLATE],
)
def test_no_unresolved_bound_placeholders(framework, template_name, resolved):
    """No {min}/{max} placeholder may survive into the resolved override params.

    The base file stores display text as a template ("Must be a value between {min}
    and {max}"); resolve_params substitutes the param's real bounds. A surviving
    token would be rendered literally to the customer.

    A param that references a bound it does not define has the field dropped
    instead, so an unresolved token here means the substitution was skipped.
    """
    issues = []
    for pname, pdef in resolved.items():
        if not isinstance(pdef, dict):
            continue
        for field in _PLACEHOLDER_TEXT_FIELDS:
            value = pdef.get(field)
            if isinstance(value, str) and ("{min}" in value or "{max}" in value):
                issues.append(f"  {pname}.{field} still contains a placeholder: {value!r}")

    assert not issues, (
        f"Unresolved {{min}}/{{max}} placeholders in resolved params for "
        f"'{framework}::{template_name}':\n" + "\n".join(issues)
    )


@pytest.mark.parametrize(
    "framework,template_name,resolved",
    RESOLVED_BY_TEMPLATE,
    ids=[f"{fw}::{t}" for fw, t, _ in RESOLVED_BY_TEMPLATE],
)
def test_resolved_bounds_in_text_match_min_max(framework, template_name, resolved):
    """A "between X and Y" phrase must quote the param's own min and max.

    Guards against the substituted text drifting from the bounds it describes — for
    example a template overriding `max` without the hint following, which would show
    the customer a limit the validator does not enforce.
    """
    issues = []
    for pname, pdef in resolved.items():
        if not isinstance(pdef, dict):
            continue
        for field in _PLACEHOLDER_TEXT_FIELDS:
            value = pdef.get(field)
            if not isinstance(value, str):
                continue
            match = _BETWEEN_RE.search(value)
            if not match:
                continue
            shown_min, shown_max = match.group(1), match.group(2).rstrip(".,;:")
            for label, shown in (("min", shown_min), ("max", shown_max)):
                if label not in pdef:
                    issues.append(
                        f"  {pname}.{field} states a {label} of {shown!r} but the param " f"declares no '{label}'"
                    )
                elif shown != str(pdef[label]):
                    issues.append(
                        f"  {pname}.{field} states {label}={shown!r} but the param " f"declares {label}={pdef[label]!r}"
                    )

    assert not issues, f"Text bounds disagree with min/max for '{framework}::{template_name}':\n" + "\n".join(issues)


# NOTE: Tests previously here that compared baseline_artifacts/*_recipe_template_parameters.json
# byte-for-byte against live sparse template files have been removed. The baseline_artifacts
# copies are now fully-resolved snapshots (with metadata) and are validated by
# test_resolved_template_params_match_baseline.py instead.
