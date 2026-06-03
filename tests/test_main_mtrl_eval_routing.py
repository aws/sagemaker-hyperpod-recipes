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
Tests for MTRL routing in ``main.py``.

``MtrlSMTJLauncher`` is a single unified launcher that handles both MTRL
training and MTRL eval recipes. Dispatch between the two happens internally
based on recipe file path. These tests verify ``main.py`` routes any
``model_type == "mtrl"`` config to ``MtrlSMTJLauncher`` exactly once, and
that non-MTRL configs are not affected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

import main as main_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The ``main`` symbol exported from ``main.py`` is wrapped by ``@hydra.main``
# and ``@validate_config``. We invoke the innermost (raw) function directly so
# tests can pass pre-built configs without the Hydra / validator machinery.
_raw_main = main_module.main.__wrapped__.__wrapped__


def _build_cfg(**overrides) -> OmegaConf:
    """Minimal cfg with ``cluster``/``cluster_type`` so downstream routing doesn't trip."""
    base = {
        "cluster_type": "sm_jobs",
        "cluster": {"cluster_type": "sm_jobs"},
        "dry_run": True,
        "launch_json": True,
        "base_results_dir": "/tmp/test-results",
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _mtrl_eval_cfg() -> OmegaConf:
    return _build_cfg(
        recipes={
            "run": {"name": "mtrl-eval-gpt-oss-20b-lora", "model_type": "mtrl"},
        },
    )


def _mtrl_training_cfg() -> OmegaConf:
    return _build_cfg(
        recipes={
            "run": {"name": "mtrl-gpt-oss-20b-lora", "model_type": "mtrl"},
        },
    )


def _nova_cfg() -> OmegaConf:
    return _build_cfg(
        recipes={
            "run": {"name": "nova-sft", "model_type": "amazon.nova.lite-v1:0"},
        },
    )


def _open_source_eval_cfg() -> OmegaConf:
    return _build_cfg(
        recipes={
            "run": {"name": "eval-job", "model_type": "hf"},
            "evaluation": {"task": "mmlu"},
        },
    )


def _nemo_cfg() -> OmegaConf:
    return _build_cfg(
        recipes={
            "run": {"name": "nemo-training", "model_type": "gpt3"},
        },
    )


class _RoutingMocks:
    """Container for patches covering every dispatch target in ``main``."""

    def __init__(self):
        self.patches = {}
        self.mocks = {}

    def __enter__(self):
        # download_model -> no-op
        self.patches["download_model"] = patch.object(main_module, "download_model")
        self.mocks["download_model"] = self.patches["download_model"].__enter__()

        # Launcher classes — all return an instance whose .run() we can assert on.
        for name in (
            "SMEvaluationJobsLauncher",
            "SMEvaluationK8SLauncher",
            "MtrlSMTJLauncher",
        ):
            p = patch.object(main_module, name)
            m = p.__enter__()
            m.return_value = MagicMock(name=f"{name}_instance")
            self.patches[name] = p
            self.mocks[name] = m

        # get_nova_launcher is a function, not a class.
        p = patch.object(main_module, "get_nova_launcher")
        m = p.__enter__()
        m.return_value = MagicMock(name="nova_launcher_instance")
        self.patches["get_nova_launcher"] = p
        self.mocks["get_nova_launcher"] = m

        # Stages / preprocess for the NeMo/default path.
        p = patch.object(main_module, "preprocess_config", return_value=(False, False))
        self.patches["preprocess_config"] = p
        self.mocks["preprocess_config"] = p.__enter__()

        p = patch.object(main_module, "get_training_stage")
        m = p.__enter__()
        stage_class_mock = MagicMock(name="stage_class")
        stage_instance = MagicMock(name="stage_instance")
        stage_instance.run.return_value = None
        stage_class_mock.return_value = stage_instance
        m.return_value = stage_class_mock
        self.patches["get_training_stage"] = p
        self.mocks["get_training_stage"] = m

        # STR2STAGECLASS default training entry.
        self._saved_str2stage = main_module.STR2STAGECLASS.copy()
        stub_stage_cls = MagicMock(name="stub_stage_cls")
        stub_stage_instance = MagicMock(name="stub_stage_instance")
        stub_stage_instance.run.return_value = None
        import pathlib as _pl

        stub_stage_instance.get_job_path.return_value.folder = _pl.Path("/tmp")
        stub_stage_cls.return_value = stub_stage_instance
        main_module.STR2STAGECLASS["training"] = stub_stage_cls
        self.mocks["default_training_stage"] = stub_stage_cls

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        main_module.STR2STAGECLASS.clear()
        main_module.STR2STAGECLASS.update(self._saved_str2stage)
        for p in self.patches.values():
            p.__exit__(exc_type, exc_val, exc_tb)

    def dispatches(self) -> dict:
        """Map of dispatch-target name -> number of *construction* calls."""
        targets = [
            "SMEvaluationJobsLauncher",
            "SMEvaluationK8SLauncher",
            "get_nova_launcher",
            "MtrlSMTJLauncher",
            "default_training_stage",
        ]
        return {name: self.mocks[name].call_count for name in targets}


# ---------------------------------------------------------------------------
# Routing exclusivity — one dispatch per cfg
# ---------------------------------------------------------------------------

_CFG_FACTORIES = {
    "mtrl_eval": _mtrl_eval_cfg,
    "mtrl_training": _mtrl_training_cfg,
    "nova": _nova_cfg,
    "open_source_eval": _open_source_eval_cfg,
    "nemo": _nemo_cfg,
}


@pytest.mark.parametrize("kind", list(_CFG_FACTORIES.keys()))
def test_routing_exclusivity(kind):
    """For any cfg shape, exactly one dispatch branch is taken.

    Both MTRL training and MTRL eval cfgs route to ``MtrlSMTJLauncher`` (the
    launcher internally picks the eval vs training processor based on recipe
    path).
    """
    cfg = _CFG_FACTORIES[kind]()

    with _RoutingMocks() as mocks:
        _raw_main(cfg)

        dispatched = {k: v for k, v in mocks.dispatches().items() if v > 0}

        # Exactly one dispatch branch fired.
        assert sum(dispatched.values()) == 1, f"For kind={kind}, expected exactly one dispatch, got {dispatched}"

        # Both MTRL variants route through the unified launcher.
        if kind in ("mtrl_eval", "mtrl_training"):
            assert mocks.mocks["MtrlSMTJLauncher"].call_count == 1


# ---------------------------------------------------------------------------
# Non-interference — every non-MTRL kind dispatches to the same target it did
# before the launcher consolidation.
# ---------------------------------------------------------------------------

_EXPECTED_DISPATCH = {
    "nova": "get_nova_launcher",
    "open_source_eval": "SMEvaluationJobsLauncher",  # cluster_type == sm_jobs
    "nemo": "default_training_stage",
}


@pytest.mark.parametrize("kind", list(_EXPECTED_DISPATCH.keys()))
def test_non_interference_routing(kind):
    """Non-MTRL configs must dispatch the same launcher/stage as before the
    MTRL consolidation, and MTRL launcher is never constructed for them.
    """
    cfg = _CFG_FACTORIES[kind]()

    with _RoutingMocks() as mocks:
        _raw_main(cfg)

        # Non-MTRL configs never construct MtrlSMTJLauncher.
        assert (
            mocks.mocks["MtrlSMTJLauncher"].call_count == 0
        ), f"MtrlSMTJLauncher was unexpectedly constructed for kind={kind}"

        # The expected dispatch target fired exactly once.
        expected_target = _EXPECTED_DISPATCH[kind]
        dispatched = mocks.dispatches()
        assert dispatched[expected_target] == 1, (
            f"For kind={kind}, expected {expected_target} to be dispatched once, " f"got {dispatched}"
        )
