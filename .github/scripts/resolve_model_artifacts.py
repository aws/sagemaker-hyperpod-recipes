#!/usr/bin/env python3
"""
Resolve S3 URIs for the latest trained model artifacts.

Given the list of missing coverage entries (from check_eval_coverage.py or
check_inference_coverage.py), this script searches S3 for the latest trained
model artifact for each (js_model_id, peft_type, training_type) combination.

Returns all available checkpoint paths so that downstream workflows (eval,
inference) can pick the ones they need:
  - hf_s3_uri: LoRA adapter checkpoint (output/model/checkpoints/hf/)
  - hf_merged_s3_uri: Merged full model (output/model/checkpoints/hf_merged/)
  - base_model_s3_uri: Pre-staged base model (validation_results/base-models/<js_model_id>/)

Usage (from GitHub Actions):
    python .github/scripts/resolve_model_artifacts.py \
        --missing-recipes "$MISSING_RECIPES" \
        --bucket hyperpod-recipes-validation-artifacts \
        --prefix validation_results/

Outputs (via GITHUB_OUTPUT):
    artifact_data: JSON array of resolved entries
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

# ── Constants ────────────────────────────────────────────────────────────────

# S3 prefix where pre-trained base model weights are stored, keyed by JumpStart
# model ID.  Path convention: s3://<bucket>/<prefix>/<js_model_id>/
DEFAULT_BASE_MODEL_PREFIX = "validation_results/base-models"

# Checkpoint sub-paths inside each training job output folder.
CHECKPOINT_SUFFIX_HF_MERGED = "output/model/checkpoints/hf_merged/"
CHECKPOINT_SUFFIX_HF = "output/model/checkpoints/hf/"


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


def find_latest_artifact(
    s3_client,
    bucket: str,
    base_prefix: str,
    js_model_id: str,
    peft_type: str,
    training_type: str,
    checkpoint_suffix: str,
) -> str:
    """Search S3 for the latest trained model artifact.

    Returns the full ``s3://`` URI of the winning checkpoint folder,
    or an empty string if nothing is found.
    """
    run_folders = _list_common_prefixes(s3_client, bucket, base_prefix)
    if not run_folders:
        return ""

    # Run folders are date-prefixed — check newest first.
    run_folders.reverse()

    for run_folder in run_folders:
        target_prefix = f"{run_folder}{js_model_id}/{peft_type}/{training_type}/"
        instance_prefixes = _list_common_prefixes(s3_client, bucket, target_prefix)
        if not instance_prefixes:
            continue

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

    return ""


def _list_common_prefixes(s3_client, bucket: str, prefix: str, *, max_prefixes: int = 1000) -> list[str]:
    """Return common prefixes (sub-folders) under *prefix*."""
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
        prefixes = prefixes[-max_prefixes:]

    return prefixes


def _get_latest_modified(s3_client, bucket: str, prefix: str, *, sample_size: int = 5):
    """Return the latest LastModified datetime among objects under *prefix*."""
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


# ── Base model resolution ────────────────────────────────────────────────────


def resolve_base_model_s3_uri(
    bucket: str,
    js_model_id: str,
    base_model_prefix: str = DEFAULT_BASE_MODEL_PREFIX,
) -> str:
    """Construct the S3 URI for a pre-trained base model.

    Returns full s3:// URI or empty string if js_model_id is empty.
    """
    if not js_model_id:
        return ""
    return f"s3://{bucket}/{base_model_prefix}/{js_model_id}/"


# ── Orchestration ────────────────────────────────────────────────────────────


def resolve_artifacts(
    missing_recipes: list[dict],
    bucket: str,
    base_prefix: str,
    s3_client=None,
    base_model_prefix: str = DEFAULT_BASE_MODEL_PREFIX,
) -> list[dict]:
    """Resolve S3 artifact URIs for all missing-recipe entries.

    Always resolves both hf/ and hf_merged/ checkpoint paths plus the
    base-models/ path.  Downstream workflows pick what they need.

    Args:
        missing_recipes: List of per-recipe dicts. Each must include
            ``recipe_path`` and ``js_model_id``; ``run_name``/``reason`` are
            ignored here. Entries without ``js_model_id`` are skipped.

    Returns:
        List of dicts with keys: model-id, recipe_path, peft_type,
        hf_s3_uri, hf_merged_s3_uri, base_model_s3_uri
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    # Cache training-artifact lookups by (js_model_id, peft, training_type) so
    # recipes that resolve to the same S3 prefix share one S3 listing.
    s3_cache: dict[tuple, tuple[str, str]] = {}
    base_model_cache: dict[str, str] = {}
    results: list[dict] = []

    for entry in missing_recipes:
        js_model_id = entry.get("js_model_id")
        if not js_model_id:
            continue

        rp = entry.get("recipe_path", "")
        if not rp:
            continue

        peft = get_peft_type_from_filename(rp)
        training = get_training_type_from_filename(rp)

        # Resolve both hf/ and hf_merged/ from the SAME job output.
        # First find the latest job using hf_merged/ (always present),
        # then derive the hf/ path from the same job prefix.
        cache_key = (js_model_id, peft, training)
        if cache_key not in s3_cache:
            # Find the latest hf_merged/ artifact (guaranteed to exist for both FFT and LoRA)
            hf_merged_uri = find_latest_artifact(
                s3_client,
                bucket,
                base_prefix,
                js_model_id,
                peft,
                training,
                checkpoint_suffix=CHECKPOINT_SUFFIX_HF_MERGED,
            )
            # Derive hf/ path from the same job prefix (replace suffix)
            if hf_merged_uri:
                hf_uri = hf_merged_uri.replace(CHECKPOINT_SUFFIX_HF_MERGED, CHECKPOINT_SUFFIX_HF)
                # Verify hf/ actually exists (only present for LoRA recipes)
                hf_prefix = hf_uri.replace(f"s3://{bucket}/", "")
                if _get_latest_modified(s3_client, bucket, hf_prefix) is None:
                    hf_uri = ""  # hf/ doesn't exist for this job (FFT recipe)
            else:
                hf_uri = ""
            s3_cache[cache_key] = (hf_merged_uri, hf_uri)

            # Log first time we resolve this combination
            if hf_merged_uri:
                print(f"  ✓ {js_model_id}/{peft}/{training}/hf_merged → {hf_merged_uri}")
            if hf_uri:
                print(f"  ✓ {js_model_id}/{peft}/{training}/hf → {hf_uri}")
            if not hf_merged_uri and not hf_uri:
                print(f"  ✗ {js_model_id}/{peft}/{training} → no artifacts found")

        # Resolve base model path and verify it exists in S3 (cached per js_model_id)
        if js_model_id not in base_model_cache:
            candidate_uri = resolve_base_model_s3_uri(bucket, js_model_id, base_model_prefix)
            if candidate_uri:
                # Verify the base model actually has objects in S3
                base_prefix_key = candidate_uri.replace(f"s3://{bucket}/", "")
                if _get_latest_modified(s3_client, bucket, base_prefix_key) is not None:
                    base_model_cache[js_model_id] = candidate_uri
                    print(f"  ✓ base model verified for {js_model_id} → {candidate_uri}")
                else:
                    base_model_cache[js_model_id] = ""
                    print(f"  ✗ base model NOT FOUND for {js_model_id} at {candidate_uri}")
            else:
                base_model_cache[js_model_id] = ""

        hf_merged_uri, hf_uri = s3_cache[cache_key]
        base_uri = base_model_cache[js_model_id]

        results.append(
            {
                "model-id": js_model_id,
                "recipe_path": rp,
                "peft_type": peft,
                "hf_s3_uri": hf_uri,
                "hf_merged_s3_uri": hf_merged_uri,
                "base_model_s3_uri": base_uri,
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
        help="JSON string of missing recipes",
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
        "--base-model-prefix",
        default=DEFAULT_BASE_MODEL_PREFIX,
        help=f"S3 prefix for pre-trained base models. Default: '{DEFAULT_BASE_MODEL_PREFIX}'.",
    )
    parser.add_argument(
        "--mode",
        choices=["evaluation", "inference"],
        default="evaluation",
        help="Validation mode. 'evaluation': unresolved if hf_merged missing. "
        "'inference': also unresolved if LoRA recipe missing base_model_s3_uri.",
    )
    args = parser.parse_args()

    missing = json.loads(args.missing_recipes)

    print("=== Resolving Model Artifact S3 URIs ===")
    print(f"Bucket: {args.bucket}")
    print(f"Prefix: {args.prefix}")
    print(f"Base model prefix: {args.base_model_prefix}")
    print(f"Models to resolve: {len(missing)}")
    print()

    events = resolve_artifacts(
        missing_recipes=missing,
        bucket=args.bucket,
        base_prefix=args.prefix,
        base_model_prefix=args.base_model_prefix,
    )

    # Determine has_unresolved based on mode:
    # - evaluation: unresolved if hf_merged is missing
    # - inference: check what's actually needed per PEFT type for SNS:
    #     FFT  → needs hf_merged_s3_uri (sent as base_model_s3_uri)
    #     LORA → needs hf_s3_uri (adapter) AND base_model_s3_uri
    if args.mode == "inference":
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
    else:
        has_unresolved = any(not e["hf_merged_s3_uri"] for e in events)

    print()
    resolved_count = sum(1 for e in events if e["hf_s3_uri"] or e["hf_merged_s3_uri"])
    print(f"Resolved {resolved_count} / {len(events)} artifact(s)")
    if has_unresolved:
        if args.mode == "inference":
            print("⚠ Some models have missing artifacts (training checkpoints or base model)")
        else:
            print("⚠ Some models have no hf_merged artifacts available")

    # Set GitHub outputs
    _set_github_output("artifact_data", json.dumps(events, indent=2))
    _set_github_output("has_unresolved", str(has_unresolved).lower())

    return 0


if __name__ == "__main__":
    sys.exit(main())
