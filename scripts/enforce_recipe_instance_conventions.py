#!/usr/bin/env python3
"""Enforce the two instance-type conventions on the per-length verl recipes.

These recipes come in "ladders": one file per measured `max_token_len_per_gpu` ceiling for
a (model, technique, tuning) slot, named after the instance the sweep measured
(`…-g6-48xl-15k-tt-fft`). Two rules govern which instances a rung may advertise.

RULE 1 — UNION UPWARD. An instance that reached length L can obviously also run every
shorter length, so a rung at L must list every instance in its class whose measured
ceiling is >= L. Before this, each rung listed roughly the one instance it was measured
on, so `ml.p5.48xlarge` was missing from 56 rungs it can trivially run.

RULE 2 — GPU COUNT MUST MATCH THE INSTANCES. `trainer.devices` and
`training_config.trainer.n_gpus_per_node` are inherited from the base recipe, which says
8. That is wrong for a rung whose instances are `*.12xlarge` (4 GPUs), and it is not
cosmetic: launcher/nemo/stages.py copies n_gpus_per_node into
`rayCluster.workerNodes.gpu`, so on EKS the pod requests 8 GPUs on a 4-GPU node and never
schedules. A rung therefore holds exactly ONE gpu-count class, and 4-GPU rungs carry the
override.

MEASURED CEILINGS ONLY. An instance earns a place on a ladder only by having a sweep of
its own at that ceiling. Instances that were only ever co-listed with another (e.g.
`ml.p4de.24xlarge` riding along on the p5 rung) are dropped rather than assumed
equivalent; `--report-drops` lists them so they can be swept later. There are exactly two
ways to hold a measured ceiling without a rung file of your own, and they are different
kinds of claim:

  * scripts/measured_ceilings.json — a real sweep whose rung file no longer exists,
    recorded per (slot, instance, ceiling). See load_manifest().
  * INHERITED_INSTANCES — a hardware-dominance claim with no length dimension, used for
    an instance the LLMFT predecessor advertised. See that constant.

Only the authored overlays under hyperpod_recipes/recipes_src/ are edited. Regenerate
recipes_collection/recipes/ afterwards with scripts/generate_resolved_recipes.py.

Usage:
    python scripts/enforce_recipe_instance_conventions.py --check
    python scripts/enforce_recipe_instance_conventions.py --apply
    python scripts/enforce_recipe_instance_conventions.py --check --report-drops

Before deleting rung files, capture what they measured; after deleting, drop the entries
the surviving filenames still supply:

    python scripts/enforce_recipe_instance_conventions.py --write-manifest   # pre-delete
    python scripts/enforce_recipe_instance_conventions.py --prune-manifest   # post-delete
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOLVED = REPO_ROOT / "recipes_collection" / "recipes" / "fine-tuning"
AUTHORED = REPO_ROOT / "hyperpod_recipes" / "recipes_src" / "fine-tuning"
MANIFEST_PATH = Path(__file__).resolve().parent / "measured_ceilings.json"

# GPUs per instance from the size suffix. Every instance these recipes use is covered;
# an unknown suffix is reported rather than guessed, because guessing here is exactly
# the mistake rule 2 exists to fix.
GPUS_BY_SIZE = {"48xlarge": 8, "24xlarge": 8, "12xlarge": 4}
LADDER_RE = re.compile(r"verl-(sft|dpo).*-tt-")
# The instance the rung was measured on, from its filename: `-g6e-12xl-` -> g6e.12xlarge.
SEG_RE = re.compile(r"-(g5|g6|g6e|p4d|p4de|p5|p5e|p5en)-(\d+)xl-")


def gpus_for(instance):
    for suffix, n in GPUS_BY_SIZE.items():
        if instance.endswith(suffix):
            return n
    return None


def named_instance(stem):
    m = SEG_RE.search(stem)
    return f"ml.{m.group(1)}.{m.group(2)}xlarge" if m else None


def load_ladders():
    """{(run_name, task, tuning): [rung]} from the resolved recipes.

    Read from the resolved tree because that is where run.name and the fully composed
    max_length live; the edits then land on the matching authored overlay by stem.
    """
    rungs = []
    for path in sorted(RESOLVED.rglob("*.yaml")):
        if not LADDER_RE.search(path.stem):
            continue
        cfg = yaml.safe_load(path.read_text()) or {}
        tc = cfg.get("training_config") or {}
        instance = named_instance(path.stem)
        rungs.append(
            {
                "stem": path.stem,
                "family": path.parent.name,
                "instance": instance,
                "gpus": gpus_for(instance) if instance else None,
                "max_length": (tc.get("data") or {}).get("max_length"),
                "instance_types": list(cfg.get("instance_types") or []),
                "n_gpus_per_node": ((tc.get("trainer") or {}).get("n_gpus_per_node")),
                "devices": (cfg.get("trainer") or {}).get("devices"),
                "slot": (
                    (cfg.get("run") or {}).get("name"),
                    "dpo" if "-dpo-" in path.stem else "sft",
                    "fft" if path.stem.endswith("-fft") else "lora",
                ),
            }
        )
    ladders = {}
    for r in rungs:
        ladders.setdefault(r["slot"], []).append(r)
    return ladders


# MEASURED-CEILING MANIFEST — sweeps whose rung file no longer exists.
#
# The filename is the *presentation* of a rung; the sweep it records is a separate fact.
# The two were the same thing only for as long as every sweep kept its own file. They stop
# being the same thing the moment two instances of equal HBM measure the same ceiling: the
# two rung files come out byte-identical, one is deleted as a duplicate, and the deleted
# filename was the only record that the other instance had been swept at all. Rule 1 would
# then strip that instance from every rung in the slot — deleting genuinely measured,
# customer-visible coverage on the strength of a filename change.
#
# So the manifest holds exactly the measurements no surviving filename supplies:
#
#     {"<run.name>": {"ml.g5.48xlarge": 6144}}
#
# keyed on run.name (unique per slot; the loader reports it if that ever stops being true)
# and read as "a sweep of this instance in this slot reached this max_length" — the same
# claim a rung file makes, so it feeds `measured` in target_state() unchanged.
#
# Per-(slot, instance) with a length is the only shape that is true. A blanket
# "g5.48xlarge == g6.48xlarge" table would be simpler but is false: of the 20 slots that
# swept both, 3 measured them differently, and for p4de/p5 it is 20 of 46 — in both
# directions (verl-dpo-qwen-3-14b-lora has p4de below p5, verl-dpo-deepseek-r1-distilled-
# llama-8b-lora has it above). Nothing dominates, so nothing may be assumed.
def load_manifest():
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def save_manifest(manifest):
    ordered = {k: dict(sorted(v.items())) for k, v in sorted(manifest.items()) if v}
    MANIFEST_PATH.write_text(json.dumps(ordered, indent=2) + "\n")


def manifest_from_filenames(ladders):
    """The manifest the current rung filenames would produce, for --write/--prune."""
    out = {}
    for slot, rungs in ladders.items():
        for r in rungs:
            if not r["instance"] or r["max_length"] is None:
                continue
            per_slot = out.setdefault(slot[0], {})
            per_slot[r["instance"]] = max(per_slot.get(r["instance"], 0), r["max_length"])
    return out


# LLMFT-INHERITED INSTANCES — the one exception to MEASURED CEILINGS ONLY.
#
# Unlike the manifest above, this is not a record of a sweep: it admits an instance that
# was never swept at all, on a hardware-dominance argument, and therefore carries no
# length. Rule 1 admits an instance only on the strength of its own swept rung. That is
# right for a ladder built by sweeping, but wrong for an instance the LLMFT recipe this one
# replaces already offered: dropping it is a customer-visible regression, not tidiness.
# Every LLMFT gpt-oss SFT/DPO recipe lists H200 (`ml.p5e.48xlarge`, `ml.p5en.48xlarge`) —
# see recipes_collection/recipes/fine-tuning/gpt_oss/llmft_gpt_oss_*_seq4k_gpu_{sft,dpo}*.yaml
# — so the verl replacements keep them by inheritance until H200 is swept.
#
# H200 is 141 GB against H100's 80 GB, so a p5-measured ceiling is a floor for it: admitting
# it cannot cause an OOM the p5 rung would not already cause. Keyed on the family so a new
# gpt-oss rung inherits automatically; drop an entry here once that instance has a real rung.
INHERITED_INSTANCES = {
    "gpt_oss-0_7_0": ("ml.p5e.48xlarge", "ml.p5en.48xlarge"),
}


def inherited_for(rung):
    """Instances this rung may list without a measured ceiling of their own."""
    return tuple(i for i in INHERITED_INSTANCES.get(rung["family"], ()) if gpus_for(i) == rung["gpus"])


def target_state(ladders, manifest):
    """[(rung, want_instance_types, want_gpus)] for every rung, plus problems."""
    plan, problems = [], []
    slots_by_name = {}
    for slot in ladders:
        slots_by_name.setdefault(slot[0], []).append(slot)
    for name in sorted(set(manifest) - set(slots_by_name)):
        problems.append(f"measured_ceilings.json: '{name}' matches no ladder — stale entry or typo")
    for name in sorted(n for n, ss in slots_by_name.items() if len(ss) > 1 and n in manifest):
        problems.append(f"measured_ceilings.json: '{name}' is ambiguous — {len(slots_by_name[name])} slots share it")

    for slot, rungs in sorted(ladders.items()):
        swept = {}
        for r in rungs:
            if not r["instance"]:
                problems.append(f"{r['stem']}: filename has no instance segment")
                continue
            if r["gpus"] is None:
                problems.append(f"{r['stem']}: unknown GPU count for {r['instance']}")
                continue
            if r["max_length"] is None:
                problems.append(f"{r['stem']}: no training_config.data.max_length")
                continue
            # Two rungs measured on the same instance in one slot would make the ladder
            # ambiguous — keep the longer, report it.
            if r["instance"] in swept and swept[r["instance"]] != r["max_length"]:
                problems.append(
                    f"{slot[0]} {slot[1]}/{slot[2]}: two rungs on {r['instance']} "
                    f"({swept[r['instance']]} and {r['max_length']})"
                )
            swept[r["instance"]] = max(swept.get(r["instance"], 0), r["max_length"])

        # Sweeps recorded in the manifest because their rung file was deleted. A filename
        # that contradicts one is the re-sweep case: the manifest entry is stale and a
        # human has to retire it, so refuse to average the two silently.
        recorded = {}
        for instance, ceiling in sorted((manifest.get(slot[0]) or {}).items()):
            if gpus_for(instance) is None:
                problems.append(f"measured_ceilings.json: unknown GPU count for {instance} in '{slot[0]}'")
                continue
            if instance in swept and swept[instance] != ceiling:
                problems.append(
                    f"measured_ceilings.json: '{slot[0]}' says {instance} reached {ceiling} but its "
                    f"rung says {swept[instance]} — retire the stale entry after the re-sweep"
                )
                continue
            recorded[instance] = ceiling
        measured = {**recorded, **swept}

        for r in rungs:
            if not r["instance"] or r["gpus"] is None or r["max_length"] is None:
                continue
            want = sorted(
                {i for i, ceiling in measured.items() if gpus_for(i) == r["gpus"] and ceiling >= r["max_length"]}
                | set(inherited_for(r))
            )
            plan.append((r, want, r["gpus"]))
    return plan, problems


INSTANCE_LINE_RE = re.compile(r"^instance_types:.*$", re.M)
TRAINING_CONFIG_RE = re.compile(r"^training_config:\s*$", re.M)


def rewrite_overlay(text, want_instances, want_gpus, base_gpus=8):
    """The overlay with instance_types replaced and, when needed, the GPU override added.

    Text surgery rather than a YAML round trip: all 149 overlays are hand-authored with
    the same four keys, quoted inline lists and blank-line grouping, and re-emitting them
    through PyYAML would reformat every line and bury the real change in the diff.
    """
    rendered = ", ".join(f'"{i}"' for i in want_instances)
    new, n = INSTANCE_LINE_RE.subn(f"instance_types: [{rendered}]", text, count=1)
    if n != 1:
        raise ValueError("expected exactly one instance_types line")

    already_overridden = re.search(r"^\s+n_gpus_per_node:", new, re.M)
    if want_gpus == base_gpus:
        return new
    if already_overridden:
        return re.sub(r"(^\s+n_gpus_per_node:).*$", rf"\g<1> {want_gpus}", new, count=1, flags=re.M)

    # Top-level trainer.devices — what the launcher provisions — plus verl's own
    # world-size view under training_config. Both, because they are different fields and
    # a job that provisions 4 GPUs while verl plans for 8 hangs at rendezvous.
    devices_block = (
        f"\n# {want_gpus} GPUs per node, not the {base_gpus} the base recipe assumes:\n"
        f"# every instance listed above is a {want_gpus}-GPU type.\n"
        f"trainer:\n  devices: {want_gpus}"
    )
    new = INSTANCE_LINE_RE.sub(lambda m: m.group(0) + devices_block, new, count=1)
    new = TRAINING_CONFIG_RE.sub(f"training_config:\n  trainer:\n    n_gpus_per_node: {want_gpus}", new, count=1)
    return new


def run_write_manifest(ladders, manifest):
    """Fold every filename-derived sweep into the manifest. Never removes."""
    from_files = manifest_from_filenames(ladders)
    added, updated = 0, 0
    for name, per_slot in sorted(from_files.items()):
        cur = manifest.setdefault(name, {})
        for instance, ceiling in sorted(per_slot.items()):
            if instance not in cur:
                cur[instance] = ceiling
                added += 1
            elif cur[instance] != ceiling:
                print(f"    {name}: {instance} {cur[instance]} -> {ceiling} (filename wins)")
                cur[instance] = ceiling
                updated += 1
    save_manifest(manifest)
    print(f"  wrote {MANIFEST_PATH}: {added} added, {updated} updated")


def run_prune_manifest(ladders, manifest):
    """Drop entries a surviving rung filename already supplies, so the file stays minimal."""
    from_files = manifest_from_filenames(ladders)
    removed = 0
    for name in sorted(manifest):
        per_slot = from_files.get(name) or {}
        for instance in sorted(manifest[name]):
            if per_slot.get(instance) == manifest[name][instance]:
                del manifest[name][instance]
                removed += 1
    save_manifest(manifest)
    kept = sum(len(v) for v in manifest.values())
    print(f"  pruned {MANIFEST_PATH}: {removed} removed, {kept} kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report drift, change nothing")
    ap.add_argument("--apply", action="store_true", help="rewrite the authored overlays")
    ap.add_argument(
        "--report-drops",
        action="store_true",
        help="list every (rung, instance) pair dropped for want of a " "measured ceiling, so those cells can be swept",
    )
    ap.add_argument("--family", action="append", default=[], help="restrict to a family directory (repeatable)")
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="fold today's filename-derived sweeps into scripts/measured_ceilings.json "
        "(run BEFORE deleting rung files)",
    )
    ap.add_argument(
        "--prune-manifest",
        action="store_true",
        help="drop manifest entries a surviving rung filename already supplies " "(run AFTER deleting rung files)",
    )
    args = ap.parse_args()
    if not (args.check or args.apply or args.write_manifest or args.prune_manifest):
        ap.error("pass --check or --apply")

    ladders = load_ladders()
    manifest = load_manifest()
    if args.write_manifest:
        run_write_manifest(ladders, manifest)
    if args.prune_manifest:
        run_prune_manifest(ladders, manifest)
    if not (args.check or args.apply):
        return 0

    plan, problems = target_state(ladders, manifest)
    if args.family:
        plan = [p for p in plan if p[0]["family"] in args.family]

    drift, drops = [], []
    for rung, want, want_gpus in plan:
        cur = sorted(rung["instance_types"])
        gpu_wrong = int(rung["n_gpus_per_node"] or 0) != want_gpus
        if cur != want or gpu_wrong:
            drift.append((rung, want, want_gpus, cur))
        drops += [(rung["stem"], i) for i in cur if i not in want]

    print(f"  {len(plan)} ladder rungs, {len(drift)} need a change")
    by_family = {}
    for rung, *_ in drift:
        by_family[rung["family"]] = by_family.get(rung["family"], 0) + 1
    print(f"  by family: {by_family}")
    if problems:
        print(f"\n  {len(problems)} problem(s) that need a human:")
        for p in problems:
            print(f"      {p}")

    if args.report_drops and drops:
        print(
            f"\n  {len(drops)} (rung, instance) pair(s) dropped — no measured ceiling "
            f"for that instance in the slot:"
        )
        for stem, inst in sorted(drops):
            print(f"      {stem[:58]:<60}{inst}")

    if args.check:
        for rung, want, want_gpus, cur in drift[:200]:
            print(
                f"    {rung['stem'][:56]:<58}len={rung['max_length']:<7}" f"gpus {rung['n_gpus_per_node']}->{want_gpus}"
            )
            print(f"        {[i.replace('ml.', '') for i in cur]}")
            print(f"     -> {[i.replace('ml.', '') for i in want]}")
        return 1 if drift or problems else 0

    written = 0
    for rung, want, want_gpus, _cur in drift:
        matches = list(AUTHORED.rglob(f"{rung['stem']}.yaml"))
        if len(matches) != 1:
            print(f"    SKIP {rung['stem']}: {len(matches)} authored overlay(s) found")
            continue
        path = matches[0]
        try:
            new = rewrite_overlay(path.read_text(), want, want_gpus)
        except ValueError as e:
            print(f"    SKIP {rung['stem']}: {e}")
            continue
        path.write_text(new)
        written += 1
    print(f"\n  rewrote {written} authored overlay(s) under {AUTHORED}")
    print("  now regenerate: python scripts/generate_resolved_recipes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
