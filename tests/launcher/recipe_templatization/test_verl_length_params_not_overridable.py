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

"""Regression guard: the VERL sequence-length params are NOT overridable.

max_prompt_length, max_response_length, and dataset_max_len are baked to each
recipe's own value -- not overridable, absent from the UI, never resolved into
``recipe_override_parameters``.

max_prompt_length is the subtle one: it is still defined in
``base_override_parameters.json``, so it would reappear in the resolved contract
if *any* ``{{max_prompt_length}}`` placeholder survived in the VERL template
(base definition + placeholder is enough to expose it; see
``utils.resolve_override_params.resolve_params``). This test therefore checks the
resolved output, not just the template's ``override_parameters`` block.
"""

import json
from pathlib import Path

import pytest

from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)
from utils.resolve_override_params import extract_placeholder_names, resolve_params

TEMPLATE_DIR = Path("launcher/recipe_templatization")
VERL_TEMPLATE_FILE = TEMPLATE_DIR / "verl" / "verl_recipe_template_parameters.json"
BASE_FILE = TEMPLATE_DIR / "base_override_parameters.json"

# Sequence-length params that #1189 made overridable and that must now be baked.
NON_OVERRIDABLE_LENGTH_PARAMS = ("max_prompt_length", "max_response_length", "dataset_max_len")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verl_templates() -> dict:
    return _load(VERL_TEMPLATE_FILE)["templates"]


ALL_TEMPLATE_NAMES = sorted(_verl_templates().keys())


@pytest.mark.parametrize("template_name", ALL_TEMPLATE_NAMES)
@pytest.mark.parametrize("param", NON_OVERRIDABLE_LENGTH_PARAMS)
def test_param_absent_from_template_override_parameters(template_name, param):
    template = _verl_templates()[template_name]
    override_parameters = template.get("override_parameters", {})
    assert param not in override_parameters, (
        f"'{param}' must not be an override_parameters entry in VERL template "
        f"'{template_name}' -- it is baked, not overridable."
    )


@pytest.mark.parametrize("template_name", ALL_TEMPLATE_NAMES)
@pytest.mark.parametrize("param", NON_OVERRIDABLE_LENGTH_PARAMS)
def test_param_placeholder_absent_from_recipe_template(template_name, param):
    template = _verl_templates()[template_name]
    placeholders = extract_placeholder_names(template["recipe_template"])
    assert param not in placeholders, (
        f"'{{{{{param}}}}}' placeholder still bound in VERL template "
        f"'{template_name}'. Any surviving placeholder re-exposes the param via "
        f"its base definition -- the rollout prompt_length binding was the last one."
    )


@pytest.mark.parametrize("template_name", ALL_TEMPLATE_NAMES)
@pytest.mark.parametrize("param", NON_OVERRIDABLE_LENGTH_PARAMS)
def test_param_absent_from_resolved_override_contract(template_name, param):
    """The resolved (base + template) contract is what the UI/PySDK consume."""
    base_params = _load(BASE_FILE).get("fine_tuning", {})
    template = _verl_templates()[template_name]
    resolved = resolve_params(
        base_params,
        template.get("override_parameters", {}),
        template["recipe_template"],
    )
    assert param not in resolved, (
        f"'{param}' is still exposed in the resolved override contract for VERL "
        f"template '{template_name}'; it must be baked, not overridable."
    )


@pytest.mark.parametrize("attr", ("_extract_context_length", "get_additional_data", "_CONTEXT_LENGTH_BOUND_PARAMS"))
def test_processor_length_clamp_internals_removed(attr):
    """#1189's per-recipe length clamp was reverted; its internals must be gone.

    Checks the class's own ``__dict__`` rather than ``hasattr`` because
    ``get_additional_data`` is defined on the base class -- #1189 added a VERL
    override, and the revert must remove that override, not the inherited method.
    """
    assert attr not in vars(
        VerlRecipeTemplateProcessor
    ), f"VerlRecipeTemplateProcessor defines '{attr}' -- the #1189 length-clamp revert has regressed."
