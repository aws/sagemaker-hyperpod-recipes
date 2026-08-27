"""
Standalone override parameter resolution logic.

This module contains the pure resolution algorithm with no external dependencies
(stdlib only). It is imported by:
  - BaseRecipeTemplateProcessor (runtime resolution)
  - generate_hyperparameters_doc.py (doc generation)
  - test_golden_override_params.py (golden test)
  - test_launch_json_generation_and_validation.py (validation)
"""

import copy
import re
from typing import Dict, Set

# Fields that are internal metadata and should not appear in the resolved output
PRIVATE_FIELDS = frozenset({"type_family", "platform"})


def extract_placeholder_names(recipe_template: dict) -> Set[str]:
    """Recursively extract all {{placeholder}} names from a recipe template.

    Args:
        recipe_template: A dict (possibly nested) containing {{name}} placeholder strings.

    Returns:
        A set of unique placeholder names without the {{ }} delimiters.
    """
    placeholder_pattern = re.compile(r"\{\{(\w+)\}\}")
    names = set()

    def _recurse(obj):
        if isinstance(obj, dict):
            for value in obj.values():
                _recurse(value)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)
        elif isinstance(obj, str):
            for match in placeholder_pattern.finditer(obj):
                names.add(match.group(1))
        elif obj is not None and not isinstance(obj, (int, float, bool)):
            raise TypeError(f"Unexpected type {type(obj).__name__} in recipe template: {obj!r}")

    _recurse(recipe_template)
    return names


def resolve_params(base_params: Dict, template_overrides: Dict, recipe_template: Dict) -> Dict:
    """Resolve full override parameters by merging base definitions with template-level overrides.

    1. Extract {{placeholder}} names from recipe_template
    2. For each placeholder, deep-copy the base definition and merge template overrides on top
    3. Include extra override params from template that aren't in placeholders
    4. Skip placeholders with no definition (e.g., instance_type — infra params without overrides)
    5. Strip private fields from the resolved output

    Args:
        base_params: The base parameter definitions dict for the relevant category.
        template_overrides: The sparse override_parameters dict from the template.
        recipe_template: The recipe_template dict containing {{placeholder}} strings.

    Returns:
        Complete recipe_override_parameters dict with private fields stripped.
    """
    placeholder_names = extract_placeholder_names(recipe_template)

    resolved = {}

    for param_name in placeholder_names:
        if param_name in template_overrides:
            if param_name in base_params:
                resolved[param_name] = copy.deepcopy(base_params[param_name])
            else:
                resolved[param_name] = {}
            for field, value in template_overrides[param_name].items():
                if value is None:
                    resolved[param_name].pop(field, None)
                else:
                    resolved[param_name][field] = copy.deepcopy(value)
        elif param_name in base_params:
            resolved[param_name] = copy.deepcopy(base_params[param_name])
        else:
            continue

    for param_name, override_def in template_overrides.items():
        if param_name not in resolved:
            if param_name in base_params:
                resolved[param_name] = copy.deepcopy(base_params[param_name])
            else:
                resolved[param_name] = {}
            for field, value in override_def.items():
                if value is None:
                    resolved[param_name].pop(field, None)
                else:
                    resolved[param_name][field] = copy.deepcopy(value)

    for param_def in resolved.values():
        for field in PRIVATE_FIELDS:
            param_def.pop(field, None)

    return dict(sorted(resolved.items()))


# Display-text fields that may carry {min}/{max} placeholders.
_TEXT_FIELDS = ("description", "display_name", "hint")


def resolve_bound_placeholders(override_params: Dict) -> Dict:
    """Substitute {min}/{max} in display text with each param's actual bounds.

    Call this only after every per-recipe adjustment to `min`/`max` has been applied.
    Some processors set bounds after the base+template merge -- the verl length clamp
    computes `max` from the recipe's own sequence length -- so substituting earlier
    would either bake in a bound that is about to change or drop the field before it
    becomes resolvable. Processors declare such params via _deferred_bound_params so
    their text is substituted after the bound is set, making the text agree with the
    final bounds by construction rather than by convention.

    The MFE also resolves these at render time (parseUIContract.ts); doing it here
    means the published contract needs no further processing.

    A field referencing a bound the param does not define is dropped entirely -- a
    hint like "Must be a value between {min} and {max}" is meaningless without
    bounds, so it should not survive with a dangling token (e.g. rollout_n, which
    uses `enum` rather than min/max). Other placeholders such as {name} are left
    untouched.

    Mutates and returns `override_params` for convenient chaining.
    """
    for param_def in override_params.values():
        if not isinstance(param_def, dict):
            continue
        for field in _TEXT_FIELDS:
            value = param_def.get(field)
            if not isinstance(value, str):
                continue
            unresolvable = False
            for token, key in (("{min}", "min"), ("{max}", "max")):
                if token in value:
                    if key in param_def:
                        value = value.replace(token, str(param_def[key]))
                    else:
                        unresolvable = True
            if unresolvable:
                param_def.pop(field, None)
            else:
                param_def[field] = value
    return override_params
