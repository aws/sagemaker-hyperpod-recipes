"""
Regression guard that the shared MTRL eval YAML is excluded from the
generic ``.sh`` launcher scripts generator.

Validates Requirements 9.1, 9.2 of the MTRL eval recipe spec: MTRL eval
recipes must NOT produce ``.sh`` wrappers via
``scripts/launcher_scripts_generator``. This is currently achieved by the
``mtrl`` substring entry under ``settings.excluded_recipe_patterns`` in
``launcher_scripts_config.yaml``. If a future change narrows that pattern
(e.g. to ``mtrl-`` or ``mtrl_training``) these tests will catch it.
"""

from pathlib import Path

import pytest
import yaml

from scripts.launcher_scripts_generator.generate_launcher_scripts import (
    LauncherConfig,
    LauncherScriptGenerationOrchestrator,
)

SHARED_EVAL_YAML = Path("recipes_collection/recipes/evaluation/mtrl/mtrl_eval.yaml")
CONFIG_YAML = Path("scripts/launcher_scripts_generator/launcher_scripts_config.yaml")


@pytest.fixture(autouse=True)
def _reset_launcher_config():
    """Ensure a clean ``LauncherConfig`` singleton per test."""
    LauncherConfig.reset()
    yield
    LauncherConfig.reset()


def test_mtrl_pattern_still_in_excluded_patterns():
    """Req 9.1: ``mtrl`` remains under ``settings.excluded_recipe_patterns``.

    This is the regression guard for task 8.1 — if someone removes or
    narrows the ``mtrl`` pattern entry, the shared ``mtrl_eval.yaml``
    would stop being filtered and produce an unwanted ``.sh`` wrapper.
    """
    assert CONFIG_YAML.exists(), f"Config file missing at {CONFIG_YAML}"
    with open(CONFIG_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    excluded = cfg.get("settings", {}).get("excluded_recipe_patterns", [])
    assert "mtrl" in excluded, (
        "'mtrl' must remain in settings.excluded_recipe_patterns so that "
        "mtrl_eval.yaml (stem contains 'mtrl') is filtered out of the "
        "launcher scripts generator."
    )


def test_shared_mtrl_eval_yaml_is_excluded_by_discover_recipes():
    """Req 9.2: ``discover_recipes()`` must not surface the shared eval YAML."""
    assert SHARED_EVAL_YAML.exists(), f"Shared eval YAML missing at {SHARED_EVAL_YAML}"

    orchestrator = LauncherScriptGenerationOrchestrator()
    discovered = orchestrator.discover_recipes()
    discovered_resolved = {p.resolve() for p in discovered}

    assert SHARED_EVAL_YAML.resolve() not in discovered_resolved, (
        "Shared MTRL eval YAML was surfaced by discover_recipes(); the "
        "'mtrl' excluded_recipe_patterns entry is no longer filtering it."
    )


def test_no_sh_file_produced_for_shared_mtrl_eval_recipe():
    """Req 9.1 + 9.2: no discovered recipe resolves to an ``mtrl_eval`` stem.

    Strongest form: walk every recipe ``discover_recipes()`` would hand
    off to the script generator and assert none of them would generate a
    script named after ``mtrl_eval``.
    """
    orchestrator = LauncherScriptGenerationOrchestrator()
    discovered = orchestrator.discover_recipes()

    offenders = [p for p in discovered if "mtrl_eval" in p.stem.lower()]
    assert not offenders, (
        f"Found recipes with 'mtrl_eval' in their stem that would produce "
        f".sh wrappers: {[str(p) for p in offenders]}"
    )

    # Also double-check via the same resolve() comparison used above —
    # belt-and-suspenders against symlink / relative-path surprises.
    discovered_resolved = {p.resolve() for p in discovered}
    assert SHARED_EVAL_YAML.resolve() not in discovered_resolved
