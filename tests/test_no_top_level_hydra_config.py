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
Guard test: no Hydra config groups (e.g. ``model_config``) may live at the
repository ROOT.

Recipe resolution composes Hydra config groups relative to the resolver's
``--config-dir`` (``scripts/hydra_resolver.py`` ->
``hyperpod_recipes/recipes_src/fine-tuning``). Config groups referenced by
recipes as ``/hydra_config/verl-0_7_0/...`` therefore resolve ONLY from
``hyperpod_recipes/recipes_src/fine-tuning/hydra_config/`` — never from a
``hydra_config/`` folder at the repository root.

A root-level ``hydra_config/`` is dead weight: it is not on any search path,
it duplicates the canonical files, and — because this repo mirrors to the
public ``aws/sagemaker-hyperpod-recipes`` OSS repo — it leaks a stray
top-level folder to customers. Such a folder was accidentally introduced by
the Ministral-3-2512 recipe PR (#1202) and removed here; this test keeps it
from coming back.
"""

from launcher.nemo.constants import ROOT_DIR

# The one legitimate home for Hydra config groups consumed by recipe
# resolution (the resolver's --config-dir).
CANONICAL_HYDRA_CONFIG_DIR = ROOT_DIR / "hyperpod_recipes" / "recipes_src" / "fine-tuning" / "hydra_config"


def test_no_hydra_config_at_repo_root():
    """A ``hydra_config/`` directory must NOT exist at the repository root."""
    stray = ROOT_DIR / "hydra_config"
    assert not stray.exists(), (
        f"Found a stray Hydra config tree at the repo root: {stray}. "
        "Hydra config groups (model_config, engine, optim, ...) must live under "
        f"{CANONICAL_HYDRA_CONFIG_DIR.relative_to(ROOT_DIR)} — the resolver's "
        "--config-dir (see scripts/hydra_resolver.py) — not at the repo root. "
        "A root-level copy is unreferenced and leaks to the public OSS mirror. "
        "Move the files under hyperpod_recipes/recipes_src/fine-tuning/hydra_config/ "
        "and delete the root-level folder."
    )


def test_canonical_hydra_config_dir_exists():
    """Positive control: the canonical Hydra config-group tree is present."""
    assert CANONICAL_HYDRA_CONFIG_DIR.is_dir(), (
        f"Expected the canonical Hydra config-group tree at "
        f"{CANONICAL_HYDRA_CONFIG_DIR}; recipe resolution depends on it."
    )
