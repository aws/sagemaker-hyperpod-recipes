"""Pricing benchmark workflow: start benchmark executions with capacity checks."""

import json
import logging
import os
import sys
import threading
from pathlib import Path

import boto3
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from lib.capacity import setup_kubeconfig
from lib.execution import report_results, run_work_item

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from utils.recipe_naming_util import instance_short

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(threadName)s]: %(message)s")
logger = logging.getLogger()

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
ASSUME_ROLE_ARN = os.environ.get("ASSUME_ROLE_ARN", "")
CLUSTER_MAP = json.loads(os.environ.get("CLUSTER_MAP", "{}"))
BRANCH = os.environ.get("BRANCH", "")
DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"
MAX_WAIT = int(os.environ.get("MAX_WAIT", "604800"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))
RECIPES_PREFIX = "recipes_collection/recipes"

sfn = boto3.client("stepfunctions", region_name="us-west-2")

results_lock = threading.Lock()
results = []


def _get_recipe_instance_types(recipe_path):
    """Read instance_types from a recipe YAML."""
    try:
        with open(recipe_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("instance_types", [])
    except Exception as e:
        logger.warning("Failed to read instance_types from %s: %s", recipe_path, e)
        return []


def _build_benchmark_payload(recipe_stem, instance_type, cluster_name, cluster_region, run_id):
    """Build the benchmark state machine input."""
    args = {
        "recipe": recipe_stem,
        "instance-type": instance_type,
        "source": "pricing",
        "tag": f"{recipe_stem}_{run_id}_pricing",
        "platform": "hyperpod-eks",
        "run-mode": "benchmark",
        "cluster-name": cluster_name,
        "cluster-region": cluster_region,
        "stream-logs": True,
    }
    if ASSUME_ROLE_ARN:
        args["role-arn"] = ASSUME_ROLE_ARN
        args["assume-role-arn"] = ASSUME_ROLE_ARN
    if BRANCH:
        args["use-github-branch"] = BRANCH
    return {"benchmarks": [{"args": args, "wait_for_completion": True}]}


def main():
    # Parse execution matrix
    entries = json.loads(os.environ.get("EXECUTION_MATRIX", "[]"))
    items = []
    for entry in entries if isinstance(entries, list) else []:
        for recipe in entry.get("recipes", []):
            items.append({"recipe": recipe, "recipe_path": f"{RECIPES_PREFIX}/{recipe}"})

    # Fallback: detect recipes directly
    if not items:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
        from scripts.detect_recipes import detect_recipes

        recipe_paths = detect_recipes(os.environ.get("BASE_BRANCH", "main"), include_modified=False)
        items = [{"recipe": r.replace(RECIPES_PREFIX + "/", ""), "recipe_path": r} for r in recipe_paths]

    if not items:
        logger.info("No work items — nothing to do")
        return

    logger.info("Starting %d pricing benchmark(s)", len(items))
    setup_kubeconfig(CLUSTER_MAP, ASSUME_ROLE_ARN)

    run_id = f"gh_{os.environ.get('GITHUB_SHA', 'manual')[:7]}"
    threads = []

    for item in items:
        recipe = item["recipe"]
        recipe_path = item["recipe_path"]
        instance_types = _get_recipe_instance_types(recipe_path)

        if not instance_types:
            logger.warning("No instance_types in %s, skipping", recipe)
            continue

        for it in instance_types:
            cluster_arn = CLUSTER_MAP.get(it, "")
            if not cluster_arn:
                logger.warning("No cluster mapping for %s, skipping", it)
                continue

            cluster_name = cluster_arn.rsplit("/", 1)[-1]
            cluster_region = cluster_arn.split(":")[3]
            recipe_stem = Path(recipe).stem
            inst = instance_short(it)
            exec_name = f"{recipe_stem}_{inst}_{run_id}_pricing"[:80]
            payload = _build_benchmark_payload(recipe_stem, it, cluster_name, cluster_region, run_id)

            t = threading.Thread(
                target=run_work_item,
                args=(sfn, STATE_MACHINE_ARN, exec_name, payload, it, recipe, CLUSTER_MAP, results, results_lock),
                kwargs={"dry_run": DRY_RUN, "poll_timeout": MAX_WAIT, "poll_interval": POLL_INTERVAL},
                name=f"{inst}-pricing",
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    report_results(results, title="💰 Pricing Benchmark Results", dry_run=DRY_RUN)
    print("✅ All pricing benchmarks completed")


if __name__ == "__main__":
    main()
