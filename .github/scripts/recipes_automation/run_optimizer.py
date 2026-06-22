"""Auto-configurator workflow: start optimization executions with capacity checks."""

import json
import logging
import os
import sys
import threading
import uuid

import boto3

sys.path.insert(0, os.path.dirname(__file__))
from lib.capacity import setup_kubeconfig
from lib.execution import report_results, run_work_item

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from utils.recipe_naming_util import instance_short

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(threadName)s]: %(message)s")
logger = logging.getLogger()

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
ASSUME_ROLE_ARN = os.environ["ASSUME_ROLE_ARN"]
CLUSTER_MAP = json.loads(os.environ.get("CLUSTER_MAP", "{}"))
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
MAX_CAPACITY_WAIT = 432000
RETRY_INTERVAL = 300
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))
MAX_POLL_WAIT = int(os.environ.get("MAX_WAIT", "604800"))
MIN_SUBMISSION_INTERVAL = 600  # 10 minutes between submissions per instance type

sfn = boto3.client("stepfunctions", region_name=os.environ.get("AWS_REGION", "us-west-2"))

results_lock = threading.Lock()
results = []

# Track last submission time per instance type
submission_times_lock = threading.Lock()  # protects access to submission_times and instance_locks
submission_times = {}  # instance_type -> timestamp
instance_locks = {}  # instance_type -> Lock


def main():
    messages = json.loads(os.environ["EXECUTION_MATRIX"])

    setup_kubeconfig(CLUSTER_MAP, ASSUME_ROLE_ARN)

    threads = []
    for msg in messages:
        run_id = msg["run_id"]
        for it in msg["instance_type_list"]:
            recipe = msg["recipes"][0]
            recipe_stem = recipe.rsplit("/", 1)[-1].replace(".yaml", "")
            inst = instance_short(it)
            unique_id = uuid.uuid4().hex[:8]
            suffix_part = f"_{inst}_{run_id}_{unique_id}"
            max_stem = 80 - len(suffix_part)
            exec_name = f"{recipe_stem[:max_stem]}{suffix_part}"

            # Get or create per-instance-type lock
            with submission_times_lock:
                if it not in instance_locks:
                    instance_locks[it] = threading.Lock()
                inst_lock = instance_locks[it]

            payload = {**msg, "instance_type_list": [it], "run_id": run_id}

            t = threading.Thread(
                target=run_work_item,
                args=(sfn, STATE_MACHINE_ARN, exec_name, payload, it, recipe, CLUSTER_MAP, results, results_lock),
                kwargs={
                    "dry_run": DRY_RUN,
                    "capacity_timeout": MAX_CAPACITY_WAIT,
                    "capacity_interval": RETRY_INTERVAL,
                    "poll_timeout": MAX_POLL_WAIT,
                    "poll_interval": POLL_INTERVAL,
                    "submission_times": submission_times,
                    "instance_lock": inst_lock,
                    "min_submission_interval": MIN_SUBMISSION_INTERVAL,
                },
                name=inst,
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    report_results(results, title="🚀 Auto-Configurator Results", dry_run=DRY_RUN)
    print("✅ All work items completed")


if __name__ == "__main__":
    main()
