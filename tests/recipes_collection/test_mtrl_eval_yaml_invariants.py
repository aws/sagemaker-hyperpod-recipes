"""
Structural invariant tests for the single shared MTRL eval recipe YAML.

Validates Property 11 from the design (Shared YAML Structural Invariants)
against Requirements 8.1–8.8. Asserts exactly one shared, model-agnostic
``mtrl_eval.yaml`` exists at the expected path with the required sections,
no forbidden training-only sections, and the deterministic eval-mode
scheduling invariants.

Tag: Feature: mtrl-eval-recipe, Property 11: Shared YAML Structural Invariants
"""

from pathlib import Path

import pytest
import yaml

# Locate repo root from this file's location so the tests work regardless
# of pytest invocation cwd. File layout:
#   <repo_root>/tests/recipes_collection/test_mtrl_eval_yaml_invariants.py
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_YAML_REL = Path("recipes_collection/recipes/evaluation/mtrl/mtrl_eval.yaml")
EXPECTED_YAML_PATH = REPO_ROOT / EXPECTED_YAML_REL
EXPECTED_PARENT_DIR = REPO_ROOT / "recipes_collection/recipes/evaluation/mtrl"


@pytest.fixture(scope="module")
def yaml_data() -> dict:
    """Load the shared MTRL eval YAML and sanity-check it parses."""
    assert EXPECTED_YAML_PATH.exists(), f"Shared MTRL eval recipe not found at {EXPECTED_YAML_PATH}"
    with open(EXPECTED_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "Top-level YAML must parse to a mapping"
    return data


def test_yaml_parses_without_error():
    """Sanity check: the file exists and ``yaml.safe_load`` succeeds."""
    assert EXPECTED_YAML_PATH.exists(), f"Shared MTRL eval recipe not found at {EXPECTED_YAML_PATH}"
    with open(EXPECTED_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None
    assert isinstance(data, dict)


def test_shared_yaml_exists_at_expected_path():
    """Requirement 8.1: exactly one shared YAML at the expected path.

    The file must be the single model-agnostic ``mtrl_eval.yaml`` — not
    per-model ``mtrl-eval-<model>-<peft>.yaml`` variants.
    """
    assert EXPECTED_YAML_PATH.is_file()

    # There must be no per-model mtrl-eval-*.yaml files in the same dir.
    per_model_matches = list(EXPECTED_PARENT_DIR.glob("mtrl-eval-*.yaml"))
    assert per_model_matches == [], (
        f"Unexpected per-model eval YAMLs found: {per_model_matches}. "
        "The design mandates a single shared mtrl_eval.yaml."
    )


def test_required_top_level_sections_present(yaml_data):
    """Requirement 8.2: ``run``, ``batch``, ``eval_sampling_params``, ``rollout`` present."""
    required = ("run", "batch", "eval_sampling_params", "rollout")
    for section in required:
        assert section in yaml_data, f"Missing required top-level section: {section!r}"


def test_forbidden_top_level_sections_absent(yaml_data):
    """Requirement 8.3: training-only sections must not appear.

    - ``rl`` must be absent at top level.
    - Top-level ``sampling_params`` must be absent (only ``eval_sampling_params``
      is allowed for eval).
    - If a ``model`` section exists, it must not contain LoRA or
      ``learning_rate`` fields.
    """
    assert "rl" not in yaml_data, "Forbidden top-level section 'rl' present"
    assert (
        "sampling_params" not in yaml_data
    ), "Forbidden top-level 'sampling_params' present; use 'eval_sampling_params'"

    model = yaml_data.get("model")
    if model is not None:
        assert isinstance(model, dict)
        for forbidden in ("lora_rank", "lora_alpha", "learning_rate"):
            assert forbidden not in model, f"model.{forbidden} is training-only and must not appear in eval recipe"


def test_run_model_type_is_mtrl(yaml_data):
    """Requirement 8.4: ``run.model_type == "mtrl"``."""
    run = yaml_data["run"]
    assert isinstance(run, dict)
    assert run.get("model_type") == "mtrl"


def test_display_name_and_version_non_empty(yaml_data):
    """Requirement 8.5: ``display_name`` and ``recipe_version``/``version`` non-empty strings."""
    display_name = yaml_data.get("display_name")
    assert isinstance(display_name, str) and display_name.strip(), "display_name must be a non-empty string"

    version = yaml_data.get("recipe_version") or yaml_data.get("version")
    assert isinstance(version, str) and version.strip(), "recipe_version (or version) must be a non-empty string"


def test_filename_and_parent_directory(yaml_data):
    """Requirement 8.8: filename is exactly ``mtrl_eval.yaml`` under the expected dir."""
    assert EXPECTED_YAML_PATH.name == "mtrl_eval.yaml"
    # Explicitly guard against the per-model naming pattern.
    assert not EXPECTED_YAML_PATH.name.startswith("mtrl-eval-")

    # Parent directory check (relative to repo root for stability).
    assert EXPECTED_YAML_PATH.parent == EXPECTED_PARENT_DIR
    assert EXPECTED_YAML_PATH.parent.is_dir()
    rel_parent = EXPECTED_YAML_PATH.parent.relative_to(REPO_ROOT).as_posix()
    assert rel_parent == "recipes_collection/recipes/evaluation/mtrl"


def test_rollout_max_concurrency_matches_override_default(yaml_data):
    """Shared eval YAML ``rollout.max_concurrency`` matches the override default (96).

    The customer-overridable ``rollout_max_concurrency`` has default=96 and
    min=32 in ``mtrl_eval_recipe_template_parameters.json``. The YAML default
    must equal that override default so the two sources of truth agree.
    """
    rollout = yaml_data["rollout"]
    assert isinstance(rollout, dict)
    assert rollout.get("max_concurrency") == 96


def test_batch_eval_group_size_is_concrete_default(yaml_data):
    """``batch.eval_group_size`` must be a concrete integer default.

    This value is the default for the customer-overridable ``eval_group_size``
    parameter. Customers can override it at launch time. The value in the
    shared YAML must NOT be a placeholder.
    """
    batch = yaml_data["batch"]
    assert isinstance(batch, dict)
    eval_group_size = batch.get("eval_group_size")
    assert isinstance(eval_group_size, int), (
        f"batch.eval_group_size must be a concrete int (Hub uploader injects the training value); "
        f"got {eval_group_size!r}"
    )
    assert eval_group_size >= 1


def test_no_placeholder_strings_in_shared_yaml(yaml_data):
    """Shared YAML must not contain any ``{{...}}`` placeholder strings.

    Placeholders belong in the template-parameters JSON. The shared recipe
    YAML is the source-of-truth for concrete defaults; if a placeholder leaks
    in, ``yaml.safe_load`` might still parse it but downstream rendering will
    fail.
    """
    import re

    def _walk(obj):
        if isinstance(obj, str):
            assert not re.search(r"\{\{\s*\w+\s*\}\}", obj), f"Placeholder leaked into shared YAML: {obj!r}"
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(yaml_data)
