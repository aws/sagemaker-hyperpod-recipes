#!/usr/bin/env python3
"""
Resolve S3 URIs for the latest trained model artifacts.

Given the list of missing eval coverage entries (from check_eval_coverage.py),
this script searches S3 for the latest trained model artifact for each
(js_model_id, peft_type, training_type) combination.

The artifact path convention is:
  s3://<bucket>/validation_results/<run_folder>/<js_model_id>/<PEFT>/<TRAINING_TYPE>/<instance_type>/<job>/output/model/checkpoints/hf_merged/

Usage (from GitHub Actions):
    python .github/scripts/resolve_model_artifacts.py \
        --missing-recipes "$MISSING_RECIPES" \
        --bucket hyperpod-recipes-validation-artifacts \
        --prefix validation_results/

Outputs (via GITHUB_OUTPUT):
    artifact_data: JSON array of {model-id, recipe_path, s3Uri}
    has_unresolved: "true" or "false"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import boto3

logger = logging.getLogger(__name__)

# ── Filename-based extraction helpers ────────────────────────────────────────


def get_peft_type_from_filename(filename: str) -> str:
    """Extract PEFT type from recipe filename.

    Returns "FFT" if 'fft' appears in the filename, otherwise "LORA".
    """
    return "FFT" if "fft" in filename.lower() else "LORA"


def get_training_type_from_filename(filename: str) -> str:
    """Extract training type from recipe filename.

    Checks in order of specificity: RLVR, RLAIF, DPO, then defaults to SFT.
    """
    lower = filename.lower()
    if "rlvr" in lower:
        return "RLVR"
    if "rlaif" in lower:
        return "RLAIF"
    if "dpo" in lower:
        return "DPO"
    return "SFT"


# ── S3 artifact resolution ──────────────────────────────────────────────────

# Default sub-path suffix inside each job folder.
# Eval uses ``hf_merged/`` (merged model weights); inference uses ``hf/``
# (base weights for hosting).  Override via ``--checkpoint-suffix`` CLI flag.
DEFAULT_CHECKPOINT_SUFFIX = "output/model/checkpoints/hf_merged/"


def find_latest_artifact(
    s3_client,
    bucket: str,
    base_prefix: str,
    js_model_id: str,
    peft_type: str,
    training_type: str,
    checkpoint_suffix: str = DEFAULT_CHECKPOINT_SUFFIX,
) -> str:
    """Search S3 for the latest trained model artifact.

    Strategy:
    1. List run folders under ``<base_prefix>`` (e.g.
       ``validation_results/2026-02-03_run_0/``), keeping the newest via
       ``max_prefixes``.
    2. Iterate newest-first.  For each run folder, check whether
       ``<run>/<js_model_id>/<peft_type>/<training_type>/`` exists.
    3. On the **first** run folder that contains the target model path,
       enumerate instance-type / job-name sub-folders within that run only.
    4. Among those that contain the expected ``hf_merged/`` path, pick the
       one whose S3 objects have the latest ``LastModified``.
    5. Return immediately — older run folders are not searched.

    Returns the full ``s3://`` URI of the winning checkpoint folder,
    or an empty string if nothing is found.
    """
    # Step 1 — discover run folders (e.g. "2026-02-03_run_0/")
    run_folders = _list_common_prefixes(s3_client, bucket, base_prefix)
    if not run_folders:
        return ""

    # Run folders are date-prefixed (YYYY-MM-DD_run_N) — check newest first.
    run_folders.reverse()

    for run_folder in run_folders:
        # Step 2 — does this run contain our model / peft / training?
        target_prefix = f"{run_folder}{js_model_id}/{peft_type}/{training_type}/"

        instance_prefixes = _list_common_prefixes(s3_client, bucket, target_prefix)
        if not instance_prefixes:
            continue  # This run doesn't have our model — try next

        # Step 3 — first matching (newest) run found; search within it only.
        best_uri = ""
        best_time = None

        for inst_prefix in instance_prefixes:
            job_prefixes = _list_common_prefixes(s3_client, bucket, inst_prefix)
            if not job_prefixes:
                continue

            for job_prefix in job_prefixes:
                candidate = f"{job_prefix}{checkpoint_suffix}"
                last_mod = _get_latest_modified(s3_client, bucket, candidate)
                if last_mod is not None:
                    if best_time is None or last_mod > best_time:
                        best_time = last_mod
                        best_uri = f"s3://{bucket}/{candidate}"

        if best_uri:
            return best_uri
        # Else: run has the prefix structure but no hf_merged/ objects.
        # Fall through to the next-newest run.

    return ""


def _list_common_prefixes(s3_client, bucket: str, prefix: str, *, max_prefixes: int = 1000) -> list[str]:
    """Return common prefixes (sub-folders) under *prefix* using ``/``
    delimiter.

    S3 returns prefixes in ascending lexicographic order.  When run folders
    are date-prefixed (``YYYY-MM-DD_run_N``), lex order equals chronological
    order.  If more than *max_prefixes* are found, only the **newest**
    (trailing) entries are kept so that the caller always sees the most
    recent runs.
    """
    prefixes: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])

    if len(prefixes) > max_prefixes:
        logger.warning(
            "Found %d prefixes listing s3://%s/%s — keeping newest %d",
            len(prefixes),
            bucket,
            prefix,
            max_prefixes,
        )
        # Keep the tail (newest) entries since S3 lists lexicographically.
        prefixes = prefixes[-max_prefixes:]

    return prefixes


def _get_latest_modified(s3_client, bucket: str, prefix: str, *, sample_size: int = 5):
    """Return the latest ``LastModified`` datetime among a sample of objects
    under *prefix*, or ``None`` if no objects exist.

    Only the first *sample_size* objects are inspected — this is enough to
    confirm the folder has content and obtain a representative timestamp
    without listing potentially hundreds of model shard files.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    latest = None
    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            mod = obj["LastModified"]
            if latest is None or mod > latest:
                latest = mod
            count += 1
            if count >= sample_size:
                return latest
    return latest


# ── Orchestration ────────────────────────────────────────────────────────────


def resolve_artifacts(
    missing_recipes: list[dict],
    bucket: str,
    base_prefix: str,
    s3_client=None,
    checkpoint_suffix: str = DEFAULT_CHECKPOINT_SUFFIX,
) -> list[dict]:
    """Resolve S3 artifact URIs for all missing-recipe entries.

    For each entry in *missing_recipes* (output of ``check_eval_coverage.py``
    or ``check_inference_coverage.py``), this function:
    1. Extracts peft_type / training_type from each recipe path.
    2. Searches S3 for the latest matching artifact (cached per model variant
       so identical model/peft/training combos don't re-query S3).
    3. Returns one entry **per recipe_path** with ``model-id``, ``s3Uri``,
       and ``recipe_path`` so that downstream consumers (e.g. SNS) can
       track each recipe individually.

    Args:
        missing_recipes: List of dicts from coverage checker:
            ``{run_names, js_model_id, reason, recipe_paths}``.
        bucket: S3 bucket name.
        base_prefix: S3 key prefix (e.g. ``"validation_results/"``).
        s3_client: Optional pre-built boto3 S3 client (for testing).
        checkpoint_suffix: Sub-path suffix inside each job folder.
            Defaults to ``output/model/checkpoints/hf_merged/`` (eval).
            Use ``output/model/checkpoints/hf/`` for inference.

    Returns:
        List of dicts::

            {
                "model-id": "<js_model_id>",
                "recipe_path": "<path to recipe YAML>",
                "s3Uri": "<resolved uri or empty string>"
            }
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Cache S3 lookups by (js_model_id, peft, training_type) so we don't
    # re-query for recipes that share the same model variant.
    s3_cache: dict[tuple, str] = {}

    # One output entry per recipe_path
    results: list[dict] = []

    for entry in missing_recipes:
        js_model_id = entry.get("js_model_id")
        if not js_model_id:
            # Entries without a JumpStart model ID should be caught earlier
            # by the workflow (has_unmapped_models gate). Skip them here.
            continue

        for rp in entry.get("recipe_paths", []):
            peft = get_peft_type_from_filename(rp)
            training = get_training_type_from_filename(rp)
            cache_key = (js_model_id, peft, training)

            if cache_key not in s3_cache:
                s3_uri = find_latest_artifact(
                    s3_client,
                    bucket,
                    base_prefix,
                    js_model_id,
                    peft,
                    training,
                    checkpoint_suffix=checkpoint_suffix,
                )
                s3_cache[cache_key] = s3_uri

                if s3_uri:
                    print(f"  ✓ {js_model_id}/{peft}/{training} → {s3_uri}")
                else:
                    print(f"  ✗ {js_model_id}/{peft}/{training} → no artifact found")

            results.append(
                {
                    "model-id": js_model_id,
                    "recipe_path": rp,
                    "s3Uri": s3_cache[cache_key],
                }
            )

    return results


# ── GitHub Actions helpers ───────────────────────────────────────────────────


def _set_github_output(name: str, value: str) -> None:
    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if gh_output:
        with open(gh_output, "a") as f:
            if "\n" in value:
                f.write(f"{name}<<GHEOF\n{value}\nGHEOF\n")
            else:
                f.write(f"{name}={value}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Resolve S3 URIs for latest trained model artifacts")
    parser.add_argument(
        "--missing-recipes",
        required=True,
        help="JSON string of missing recipes (from check_eval_coverage.py)",
    )
    parser.add_argument(
        "--bucket",
        default="hyperpod-recipes-validation-artifacts",
        help="S3 bucket containing training artifacts",
    )
    parser.add_argument(
        "--prefix",
        default="validation_results/",
        help="S3 key prefix under the bucket",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default=DEFAULT_CHECKPOINT_SUFFIX,
        help="Sub-path suffix inside each job folder. "
        "Default: 'output/model/checkpoints/hf_merged/' (eval). "
        "Use 'output/model/checkpoints/hf/' for inference.",
    )
    args = parser.parse_args()

    missing = json.loads(args.missing_recipes)

    print("=== Resolving Model Artifact S3 URIs ===")
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {args.prefix}")
    print(f"Models to resolve: {len(missing)}")
    print()

    checkpoint_suffix = args.checkpoint_suffix
    print(f"Checkpoint suffix: {checkpoint_suffix}")

    events = resolve_artifacts(
        missing_recipes=missing,
        bucket=args.bucket,
        base_prefix=args.prefix,
        checkpoint_suffix=checkpoint_suffix,
    )

    has_unresolved = any(e["s3Uri"] == "" for e in events)

    print()
    print(f"Resolved {sum(1 for e in events if e['s3Uri'])} / {len(events)} artifact(s)")
    if has_unresolved:
        print("⚠ Some models have no trained artifacts available")

    # Set GitHub outputs
    _set_github_output("artifact_data", json.dumps(events, indent=2))
    _set_github_output("has_unresolved", str(has_unresolved).lower())

    return 0


if __name__ == "__main__":
    sys.exit(main())
