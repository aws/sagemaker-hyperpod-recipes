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

"""
Tests for MTRL eval wiring in ``scripts/generate_launch_jsons.py``.

Covers tasks 5.5 (discovery priority) and 5.6 (get_mtrl_eval_params)
of the mtrl-eval-recipe spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_launch_jsons import LaunchJsonGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus(
    working_dir: Path,
    training_stem: str,
    eval_stem: str,
) -> tuple[Path, Path]:
    """Create a minimal recipe corpus with one MTRL training and one MTRL eval YAML.

    Returns the absolute paths to (training_yaml, eval_yaml).
    """
    train_dir = working_dir / "recipes_collection" / "recipes" / "fine-tuning" / "gpt-oss"
    eval_dir = working_dir / "recipes_collection" / "recipes" / "evaluation" / "mtrl"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    training_yaml = train_dir / f"{training_stem}.yaml"
    eval_yaml = eval_dir / f"{eval_stem}.yaml"

    # The discovery path only cares about filenames, not contents.
    training_yaml.write_text("run:\n  model_type: mtrl\n")
    eval_yaml.write_text("run:\n  model_type: mtrl\n")

    return training_yaml, eval_yaml


# ---------------------------------------------------------------------------
# Task 5.5  Discovery Priority — Eval Before Training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "eval_stem",
    [
        "mtrl-eval-gpt-oss-20b-lora",
        "mtrl-eval-qwen-3-32b-lora",
        "mtrl-eval-nova-lite-2-0-lora",
    ],
)
def test_discovery_priority_eval_before_training(tmp_path: Path, eval_stem: str) -> None:
    """MTRL eval YAMLs must be discovered under the ``mtrl_eval`` prefix, never ``mtrl``.

    Both prefixes' recursive globs would match an ``mtrl-eval-*.yaml`` filename;
    the priority sort (``mtrl_eval`` before ``mtrl``) plus ``seen_recipes``
    deduplication must route every eval YAML to the eval prefix.
    """
    corpus_dir = tmp_path / eval_stem
    training_yaml, eval_yaml = _make_corpus(
        corpus_dir,
        training_stem="mtrl-gpt-oss-20b-lora",
        eval_stem=eval_stem,
    )

    generator = LaunchJsonGenerator(working_dir=str(corpus_dir))
    recipes = generator.discover_recipes(prefixes=["mtrl_eval", "mtrl"])

    # Index by (path, prefix) for easy assertions.
    by_path: dict[Path, list[str]] = {}
    for path, prefix, _model in recipes:
        by_path.setdefault(path, []).append(prefix)

    # The eval YAML must appear exactly once, under "mtrl_eval".
    assert eval_yaml in by_path, f"eval YAML not discovered: {eval_yaml}"
    assert by_path[eval_yaml] == ["mtrl_eval"], f"eval YAML prefix tags = {by_path[eval_yaml]}, expected ['mtrl_eval']"

    # The training YAML must appear exactly once, under "mtrl".
    assert training_yaml in by_path, f"training YAML not discovered: {training_yaml}"
    assert by_path[training_yaml] == [
        "mtrl"
    ], f"training YAML prefix tags = {by_path[training_yaml]}, expected ['mtrl']"


# ---------------------------------------------------------------------------
# Task 5.6  get_mtrl_eval_params(sm_jobs, ...) override tokens
# ---------------------------------------------------------------------------


def test_get_mtrl_eval_params_sm_jobs_tokens(tmp_path: Path) -> None:
    """The sm_jobs override list must contain the six expected override tokens."""
    generator = LaunchJsonGenerator(working_dir=str(tmp_path))
    params = generator.get_mtrl_eval_params(
        Path("evaluation/mtrl/mtrl-eval-gpt-oss-20b-lora"),
        "sm_jobs",
        "/tmp/out",
    )

    expected_substrings = [
        "recipes=",
        "base_results_dir=",
        "cluster=sm_jobs",
        "cluster_type=sm_jobs",
        "container=test_container",
        "launch_json=true",
    ]

    for token in expected_substrings:
        assert any(token in p for p in params), f"missing override token {token!r} in {params}"
