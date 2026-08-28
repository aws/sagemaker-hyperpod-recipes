#!/usr/bin/env python3
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
Prune redundant fields from Hydra recipe source files.

Recipes under ``hyperpod_recipes/recipes_src/fine-tuning/`` inherit shared values
from base configs (via their ``defaults:`` list) under
``.../fine-tuning/hydra_config/``. Over time recipes accumulate fields that
re-declare a value identical to what they already inherit. This script removes
every such redundant field WITHOUT changing the fully-resolved output.

The resolved output (``recipes_collection/recipes/``, produced by
generate_resolved_recipes.py) is guaranteed byte-identical: for each recipe the
pruned source is re-resolved and compared against the original resolved YAML; a
recipe is only rewritten if that comparison matches exactly.

Comments, key order, and formatting in the source files are preserved via
ruamel.yaml (only the redundant keys are surgically removed).

Usage:
    # Dry-run (default): report what would be pruned, change nothing.
    uv run poe hydra-prune

    # Apply the pruning to the source files.
    uv run poe hydra-prune --write
"""

import argparse
import io
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from ruamel.yaml import YAML

from hyperpod_recipes import list_recipes
from hyperpod_recipes.recipe import RECIPES_DIR, Recipe

# Only fine-tuning recipes use Hydra ``defaults:`` inheritance. Recipes under
# fine-tuning/nova are self-contained direct-copies and have no defaults list,
# so they are excluded both by path and by the has-defaults guard below.
PRUNE_ROOT = Path(RECIPES_DIR) / "fine-tuning"
EXCLUDE_DIRS = [PRUNE_ROOT / "nova"]


def _in_scope(recipe_path: str) -> bool:
    p = Path(recipe_path)
    if not str(p).startswith(str(PRUNE_ROOT)):
        return False
    return not any(str(p).startswith(str(d)) for d in EXCLUDE_DIRS)


def _resolved_yaml(recipe_path: str) -> str:
    """Fully-resolved YAML for a recipe (matches generate_resolved_recipes.py)."""
    return OmegaConf.to_yaml(Recipe(recipe_path).config, sort_keys=True)


def _baseline_container(recipe_path: str, defaults) -> dict:
    """Resolve the recipe with an *empty body* (only its ``defaults:`` block).

    Written to a temp file in the recipe's own directory so ``_self_`` ordering,
    ``@package`` group overrides, and relative default paths all still apply.
    Returned unresolved (interpolations kept as raw strings) to match how the
    recipe body compares.
    """
    recipe_dir = os.path.dirname(recipe_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", dir=recipe_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            yaml.safe_dump({"defaults": defaults}, f, sort_keys=False)
        cfg = Recipe(tmp_path).config
        return OmegaConf.to_container(cfg, resolve=False)
    finally:
        os.remove(tmp_path)


def _flatten(node, prefix=""):
    """Flatten a nested dict to {dotted_path: leaf_value}."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = node
    return out


def _lookup(container, dotted):
    """Return (found, value) for a dotted path in a nested dict."""
    cur = container
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _represent_none(representer, data):
    """Emit ``None`` as the explicit ``null`` token (matches source style)."""
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


def _make_yaml() -> YAML:
    """ruamel YAML configured to round-trip recipe sources with minimal churn."""
    ruamel = YAML()
    ruamel.preserve_quotes = True
    # Match the recipes' 2-space mapping / block-sequence indentation.
    ruamel.indent(mapping=2, sequence=4, offset=2)
    ruamel.representer.add_representer(type(None), _represent_none)
    return ruamel


def _delete_path(node, parts):
    """Delete a leaf at ``parts`` from a (ruamel) mapping, pruning emptied maps."""
    key = parts[0]
    if len(parts) == 1:
        if isinstance(node, dict) and key in node:
            del node[key]
        return
    child = node.get(key) if isinstance(node, dict) else None
    if isinstance(child, dict):
        _delete_path(child, parts[1:])
        # Remove the parent map if pruning emptied it.
        if len(child) == 0:
            del node[key]


def _prune_one(recipe_path: str):
    """Compute the pruning for a single recipe. Runs in a worker process.

    Returns a result dict; never writes to the source file (the parent applies
    approved writes).
    """
    name = Recipe(recipe_path).name
    result = {
        "name": name,
        "path": recipe_path,
        "pruned_paths": [],
        "new_text": None,
        "status": "ok",
        "detail": "",
    }

    raw = yaml.safe_load(Path(recipe_path).read_text())
    if not isinstance(raw, dict) or "defaults" not in raw:
        result["status"] = "skipped"
        result["detail"] = "no defaults list"
        return result

    golden = _resolved_yaml(recipe_path)
    baseline = _baseline_container(recipe_path, raw["defaults"])

    body = {k: v for k, v in raw.items() if k != "defaults"}
    candidates = []
    for dotted, value in _flatten(body).items():
        found, base_value = _lookup(baseline, dotted)
        if found and base_value == value:
            candidates.append(dotted)

    if not candidates:
        result["detail"] = "nothing redundant"
        return result

    # Surgically delete the redundant keys with ruamel to preserve comments,
    # key order, and formatting.
    ruamel = _make_yaml()
    doc = ruamel.load(Path(recipe_path).read_text())
    for dotted in candidates:
        _delete_path(doc, dotted.split("."))
    buf = io.StringIO()
    ruamel.dump(doc, buf)
    new_text = buf.getvalue()

    # Validation gate: re-resolve the pruned source and require byte-identical
    # output. Only then is the recipe safe to rewrite.
    recipe_dir = os.path.dirname(recipe_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", dir=recipe_dir)
    os.close(fd)
    try:
        Path(tmp_path).write_text(new_text)
        pruned_resolved = _resolved_yaml(tmp_path)
    finally:
        os.remove(tmp_path)

    if pruned_resolved != golden:
        result["status"] = "gate_failed"
        result["detail"] = "resolved output would change; left untouched"
        return result

    result["pruned_paths"] = candidates
    result["new_text"] = new_text
    return result


def prune_recipes(write=False):
    """Prune redundant inherited fields from all in-scope source recipes."""
    recipes = [r for r in list_recipes() if _in_scope(r.path)]
    recipe_paths = [r.path for r in recipes]

    mode = "Pruning" if write else "Checking (dry-run)"
    print(f"{mode} {len(recipe_paths)} fine-tuning recipes...")

    max_workers = min(len(recipe_paths), os.cpu_count() or 1)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_prune_one, recipe_paths))
    elapsed = time.perf_counter() - t0
    print(f"Analyzed {len(results)} recipes in {elapsed:.1f}s ({max_workers} workers)\n")

    total_pruned = 0
    changed = 0
    gate_failures = []

    for res in sorted(results, key=lambda r: r["name"]):
        if res["status"] == "gate_failed":
            gate_failures.append(res)
            print(f"[GATE FAILED] {res['name']}: {res['detail']}")
            continue
        if not res["pruned_paths"]:
            continue

        changed += 1
        total_pruned += len(res["pruned_paths"])
        verb = "Pruning" if write else "Would prune"
        print(f"{verb} {len(res['pruned_paths'])} field(s) from {res['name']}:")
        for dotted in res["pruned_paths"]:
            print(f"    - {dotted}")

        if write:
            Path(res["path"]).write_text(res["new_text"])

    print()
    print(f"Recipes with redundant fields: {changed}")
    print(f"Total fields {'pruned' if write else 'prunable'}: {total_pruned}")
    if gate_failures:
        print(f"Recipes skipped by validation gate: {len(gate_failures)}")
    if write:
        print(
            "\nDone. Run `uv run poe generate-resolved-recipes --check` to confirm " "the resolved output is unchanged."
        )
    else:
        print("\nDry-run only. Re-run with --write to apply.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prune redundant inherited fields from Hydra recipe source files.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the pruning to source files (default: dry-run report only).",
    )
    args = parser.parse_args()
    prune_recipes(write=args.write)


if __name__ == "__main__":
    main()
