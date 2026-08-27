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

"""Bump the patch version (X.Y.Z -> X.Y.Z+1) of Hydra source recipes in place.

Each source recipe carries an inline top-level `version: X.Y.Z` line (bare or
quoted). This bumps only the patch component, preserving major/minor and the
original quoting/whitespace. Files without a version line are skipped, as are
hydra_config fragments and checkpointless recipes (versioned independently).

After bumping, regenerate recipes_collection so the resolved YAMLs match:

    python scripts/bump_recipe_versions.py --dir hyperpod_recipes/recipes_src/fine-tuning
    python scripts/generate_resolved_recipes.py

After applying, a contamination guard diffs each changed file against git HEAD and fails
if anything other than the `version:` line changed (catches an unrelated edit — e.g. a
stash-pop that merged an instance_types change — leaking into the bump). Use --no-verify
to skip it (e.g. when intentionally bumping on top of other staged edits).

Usage:
    python scripts/bump_recipe_versions.py                 # bump fine-tuning source, apply + verify
    python scripts/bump_recipe_versions.py --dry-run       # preview only, no writes
    python scripts/bump_recipe_versions.py --dir <path>    # bump a different source subtree
    python scripts/bump_recipe_versions.py --no-verify     # skip the post-bump contamination guard
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import subprocess

# Top-level `version:` line with an X.Y.Z value, optionally single/double quoted.
# Groups: 1=prefix ("version: "), 2=open quote, 3=major, 4=minor, 5=patch, 6=close quote.
_VERSION_RE = re.compile(r'^(version:\s*)(["\']?)(\d+)\.(\d+)\.(\d+)(["\']?)\s*$')

# Recipes under hydra_config/ are composition fragments, not standalone recipes — skip.
_EXCLUDE_SUBSTR = "/hydra_config/"

# Recipe families intentionally left out of the patch bump (matched as a filename substring).
#   checkpointless: versioned independently, should not ride the bulk bump.
#   scout (llama4 scout): held back from the bump as a family.
_EXCLUDE_NAME_SUBSTRINGS = ("checkpointless", "scout")

# Specific recipes excluded by exact filename (not a whole family).
_EXCLUDE_NAMES: frozenset[str] = frozenset()

DEFAULT_DIR = "hyperpod_recipes/recipes_src/fine-tuning"


def _bump_line(line: str) -> str | None:
    """Return the version-bumped line, or None if this line isn't a version line."""
    m = _VERSION_RE.match(line.rstrip("\n"))
    if not m:
        return None
    prefix, q_open, major, minor, patch, q_close = m.groups()
    return f"{prefix}{q_open}{major}.{minor}.{int(patch) + 1}{q_close}"


def bump_file(path: str, dry_run: bool) -> tuple[str, str] | None:
    """Bump the first version line in `path`. Returns (old, new) or None if no version."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines(keepends=True)

    old_new = None
    out = []
    for line in lines:
        if old_new is None:
            m = _VERSION_RE.match(line.rstrip("\n"))
            if m:
                old = f"{m.group(3)}.{m.group(4)}.{m.group(5)}"
                bumped = _bump_line(line)
                old_new = (old, bumped.split("version:")[1].strip().strip("\"'"))
                out.append(bumped + "\n")
                continue
        out.append(line if line.endswith("\n") else line + "\n")

    if old_new and not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("".join(out))
    return old_new


def verify_only_version_changed(paths: list[str]) -> list[str]:
    """Contamination guard: for each changed source file, diff against git HEAD and confirm
    the ONLY changed content is a `version:` line. Any other added/removed line means the
    working tree carried in an unrelated change (e.g. a stash-pop merge leaked an
    instance_types edit) — return the list of offending files. Empty list = clean."""
    offenders = []
    for p in paths:
        diff = subprocess.run(
            ["git", "diff", "--unified=0", "--", p],
            capture_output=True,
            text=True,
        ).stdout
        # content changes are +/- lines that aren't the hunk/file headers (+++/---)
        changed = [
            ln
            for ln in diff.splitlines()
            if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---"))
        ]
        # every changed line must be a version: line
        if any(not ln[1:].lstrip().startswith("version:") for ln in changed):
            offenders.append(p)
    return offenders


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help=f"source recipe root to bump (default: {DEFAULT_DIR})")
    ap.add_argument("--dry-run", action="store_true", help="preview changes without writing")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the post-bump guard that checks only version: lines changed vs git HEAD",
    )
    args = ap.parse_args()

    files = [
        f
        for f in glob.glob(os.path.join(args.dir, "**", "*.yaml"), recursive=True)
        if _EXCLUDE_SUBSTR not in f
        and os.path.basename(f) not in _EXCLUDE_NAMES
        and not any(s in os.path.basename(f) for s in _EXCLUDE_NAME_SUBSTRINGS)
    ]
    bumped = 0
    no_version = 0
    changed_paths = []
    tally: collections.Counter = collections.Counter()
    for f in sorted(files):
        result = bump_file(f, args.dry_run)
        if result is None:
            no_version += 1
            continue
        old, new = result
        tally[(old, new)] += 1
        bumped += 1
        changed_paths.append(f)

    verb = "would bump" if args.dry_run else "bumped"
    print(f"{verb} {bumped} recipes ({no_version} without a version line) under {args.dir}")
    for (old, new), count in sorted(tally.items()):
        print(f"  {old} -> {new}: {count}")

    # Contamination guard: confirm the ONLY change vs git HEAD is the version line. This
    # catches an unrelated edit (e.g. a stash-pop that merged an instance_types change into
    # a recipe) sneaking into the version-bump commit. Skipped on --dry-run (nothing written).
    if not args.dry_run and bumped and not args.no_verify:
        offenders = verify_only_version_changed(changed_paths)
        if offenders:
            print(f"\n✗ VERIFY FAILED: {len(offenders)} file(s) changed beyond the version line:")
            for o in offenders:
                print(f"    {o}")
            print("  These carry a non-version change (contamination). Fix before committing.")
            return 1
        print(f"\n✓ verify: all {bumped} files changed only their version line")

    if not args.dry_run and bumped:
        print("\nNext: python scripts/generate_resolved_recipes.py  # regenerate recipes_collection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
