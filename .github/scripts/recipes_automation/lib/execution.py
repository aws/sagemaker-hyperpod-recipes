"""Shared Step Functions execution utilities — start, poll, and manage executions."""

import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


def start_execution(sfn_client, state_machine_arn, exec_name, payload, dry_run=False):
    """Start a Step Functions execution. Returns exec_arn or None on failure.

    Handles ExecutionAlreadyExists gracefully (returns ARN for polling).
    """
    if dry_run:
        logger.info("DRY RUN: would start %s", exec_name)
        return None

    try:
        resp = sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=exec_name,
            input=json.dumps(payload),
        )
        logger.info("Started: %s → %s", exec_name, resp["executionArn"])
        return resp["executionArn"]
    except sfn_client.exceptions.ExecutionAlreadyExists:
        exec_arn = f"{state_machine_arn.replace(':stateMachine:', ':execution:')}:{exec_name}"
        logger.info("Already exists: %s — will poll", exec_name)
        return exec_arn
    except Exception as e:
        logger.error("Failed to start %s: %s", exec_name, e)
        return None


def poll_execution(sfn_client, exec_arn, timeout=604800, interval=300):
    """Poll until execution reaches terminal state. Returns status string."""
    elapsed = 0
    while elapsed < timeout:
        try:
            resp = sfn_client.describe_execution(executionArn=exec_arn)
            status = resp["status"]
        except Exception:
            status = "UNKNOWN"

        if status not in ("RUNNING",):
            return status

        time.sleep(interval)
        elapsed += interval

    return "TIMED_OUT_POLL"


def run_work_item(
    sfn_client,
    state_machine_arn,
    exec_name,
    payload,
    instance_type,
    recipe,
    cluster_map,
    results,
    results_lock,
    dry_run=False,
    capacity_timeout=432000,
    capacity_interval=300,
    poll_timeout=604800,
    poll_interval=300,
    submission_times=None,
    instance_lock=None,
    min_submission_interval=600,
):
    """Full lifecycle: enforce cooldown → wait for capacity → start → poll. Thread-safe."""
    sys.path.insert(0, os.path.dirname(__file__))
    from capacity import wait_for_capacity

    # Phase 1: Serialize submissions per instance type — cooldown + capacity + start
    if not dry_run and submission_times is not None and instance_lock is not None:
        with instance_lock:
            last_time = submission_times.get(instance_type, 0)
            wait_needed = min_submission_interval - (time.time() - last_time)
            if wait_needed > 0:
                logger.info("Cooldown for %s: waiting %.0fs before next submission", instance_type, wait_needed)
                time.sleep(wait_needed)

            if not wait_for_capacity(instance_type, cluster_map, timeout=capacity_timeout, interval=capacity_interval):
                with results_lock:
                    results.append(
                        {
                            "name": exec_name,
                            "instance_type": instance_type,
                            "recipe": recipe,
                            "status": "TIMED_OUT_CAPACITY",
                        }
                    )
                return

            exec_arn = start_execution(sfn_client, state_machine_arn, exec_name, payload)
            if not exec_arn:
                with results_lock:
                    results.append(
                        {"name": exec_name, "instance_type": instance_type, "recipe": recipe, "status": "START_FAILED"}
                    )
                return

            submission_times[instance_type] = time.time()
    else:
        # Dry run or no cooldown tracking
        if not wait_for_capacity(instance_type, cluster_map, timeout=capacity_timeout, interval=capacity_interval):
            with results_lock:
                results.append(
                    {
                        "name": exec_name,
                        "instance_type": instance_type,
                        "recipe": recipe,
                        "status": "TIMED_OUT_CAPACITY",
                    }
                )
            return

        if dry_run:
            logger.info("DRY RUN: would start %s", exec_name)
            with results_lock:
                results.append(
                    {"name": exec_name, "instance_type": instance_type, "recipe": recipe, "status": "DRY_RUN"}
                )
            return

        exec_arn = start_execution(sfn_client, state_machine_arn, exec_name, payload)
        if not exec_arn:
            with results_lock:
                results.append(
                    {"name": exec_name, "instance_type": instance_type, "recipe": recipe, "status": "START_FAILED"}
                )
            return

    # Phase 2: Poll
    status = poll_execution(sfn_client, exec_arn, timeout=poll_timeout, interval=poll_interval)
    logger.info("Completed: %s → %s", exec_name, status)
    with results_lock:
        results.append({"name": exec_name, "instance_type": instance_type, "recipe": recipe, "status": status})


def report_results(results, title="Results", dry_run=False):
    """Write GitHub step summary and output, exit 1 if failures."""
    failed = [r for r in results if r["status"] not in ("SUCCEEDED", "DRY_RUN")]
    exec_names = [r["name"] for r in results if r["status"] not in ("TIMED_OUT_CAPACITY", "START_FAILED")]

    summary_lines = [f"## {title}", "", "| Recipe | Instance Type | Status |", "|--------|---------------|--------|"]
    for r in results:
        icon = "✅" if r["status"] in ("SUCCEEDED", "DRY_RUN") else "❌"
        summary_lines.append(f"| `{r['recipe']}` | `{r['instance_type']}` | {icon} {r['status']} |")
    summary_lines.append(f"\n**Total:** {len(results)} work items, {len(failed)} failed")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write("\n".join(summary_lines) + "\n")

    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"exec_names={json.dumps(exec_names)}\n")

    if failed and not dry_run:
        sys.exit(1)
