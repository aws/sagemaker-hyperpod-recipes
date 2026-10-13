"""Plumbing for the litmus (Custom Scorer) eval notebook — dataset assembly, the quota-aware job scheduler,
and results reading — kept out of the notebook so the cells focus on the eval *content* (the reward function
and the gate decision).
"""
import json
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path


# ---------------------------------------------------------------- datasets
def build_eval_datasets(*, val_file, eval_tasks, sb_data, tokenizer_id, max_query_tokens,
                        n_examples, s3_prefix, bucket, region, seed=42):
    """Turn the local held-out val file into one Custom Scorer dataset per task and upload to S3.

    Recovers each row's task (product/shop/voucher) by matching its <user>…</user> query against
    ShoppingBench's per-task test files, drops prompts >= max_query_tokens (they exceed the serving context
    window and fail the managed job), and writes one {query, response, system?} JSONL per task.

    Returns (datasets, counts): {task: s3_uri} and {task: row_count}. Prints a per-task summary.
    """
    import random
    import boto3
    from collections import Counter
    from transformers import AutoTokenizer

    sb_data = Path(sb_data)
    assert sb_data.exists(), f"ShoppingBench data not found at {sb_data}; run setup_shopbench.sh first."

    # query -> task label, from the per-task test files
    q2task = {}
    for t in eval_tasks:
        for l in open(sb_data / f"synthesize_{t}_test.jsonl"):
            if l.strip():
                q2task.setdefault(json.loads(l)["query"].strip(), t)

    def row_task(user_content):
        m = re.search(r"<user>(.*?)</user>", user_content, flags=re.S)
        return q2task.get(m.group(1).strip()) if m else None

    rows = [json.loads(l) for l in open(val_file) if l.strip()]
    random.seed(seed); random.shuffle(rows)
    if n_examples is not None:
        rows = rows[:n_examples]

    def to_eval_row(r):
        msgs = r["messages"]
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        target = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
        row = {"query": user, "response": target}
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        if system:
            row["system"] = system
        return row, (row_task(user) or "unknown")

    eval_rows, tasks = zip(*(to_eval_row(r) for r in rows)) if rows else ([], [])
    eval_rows, tasks = list(eval_rows), list(tasks)

    # length filter
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    def prompt_text(er):
        return (er["system"] + "\n\n" + er["query"]) if "system" in er else er["query"]
    n_before, kept, chunk = len(eval_rows), [], 500
    for i in range(0, len(eval_rows), chunk):
        batch = eval_rows[i:i + chunk]
        lens = [len(ids) for ids in tok([prompt_text(er) for er in batch])["input_ids"]]
        for er, tk, n in zip(batch, tasks[i:i + chunk], lens):
            if n < max_query_tokens:
                kept.append((er, tk))
    eval_rows = [er for er, _ in kept]
    tasks = [tk for _, tk in kept]
    print(f"length filter: kept {len(eval_rows)} / {n_before} rows "
          f"(dropped {n_before - len(eval_rows)} with >= {max_query_tokens} tokens)")

    dist = Counter(tasks)
    print("task label distribution (after filter):", dict(dist))
    if dist.get("unknown"):
        print(f"  note: {dist['unknown']} rows could not be task-labeled and are excluded from per-task evals.")

    s3 = boto3.client("s3", region_name=region)
    out_dir = Path("eval_inputs"); out_dir.mkdir(exist_ok=True)   # local scratch for the per-task datasets
    datasets, counts = {}, {}
    for t in eval_tasks:
        grp = [er for er, tk in zip(eval_rows, tasks) if tk == t]
        if not grp:
            print(f"  (skip '{t}': 0 rows)"); continue
        local = out_dir / f"litmus_eval_{t}.jsonl"
        with open(local, "w") as f:
            for r in grp:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        key = f"{s3_prefix}/data/litmus_eval_{t}.jsonl"
        s3.upload_file(str(local), bucket, key)
        datasets[t] = f"s3://{bucket}/{key}"
        counts[t] = len(grp)
        print(f"  {t}: {len(grp)} rows -> {datasets[t]}")
    return datasets, counts


# ---------------------------------------------------------------- job scheduling
_QUIET_LOGGERS = ("sagemaker.train", "sagemaker.core", "sagemaker")
_EVAL_QUOTA_CODE = "L-619D690E"   # "Maximum number of concurrent model evaluation serverless jobs per Region"


@contextmanager
def _quiet_sdk_logs(enabled=True, level=logging.WARNING):
    """Temporarily raise sagemaker.* log levels, then restore them (no permanent global change)."""
    if not enabled:
        yield
        return
    saved = {n: logging.getLogger(n).level for n in _QUIET_LOGGERS}
    try:
        for n in _QUIET_LOGGERS:
            logging.getLogger(n).setLevel(level)
        yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)


def _eval_job_quota(region):
    """Applied concurrent model-evaluation serverless-job quota, or None if undeterminable (exact code first,
    then a name scan, with backoff on throttling)."""
    import boto3
    from botocore.exceptions import ClientError
    try:
        sq = boto3.client("service-quotas", region_name=region)
    except Exception:
        return None
    for attempt in range(4):
        try:
            return int(sq.get_service_quota(ServiceCode="sagemaker", QuotaCode=_EVAL_QUOTA_CODE)["Quota"]["Value"])
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") in ("TooManyRequestsException", "ThrottlingException"):
                time.sleep(2 ** attempt * 3); continue
            break
        except Exception:
            break
    try:
        for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page["Quotas"]:
                if "concurrent model evaluation serverless" in q["QuotaName"].lower():
                    return int(q["Value"])
    except Exception:
        pass
    return None


def run_eval_jobs(*, datasets, model_pkg_arn, evaluator_arn, s3_output, region,
                  mlflow_experiment, evaluate_base=True, quiet=True):
    """Run one Custom Scorer job per task, scheduling to stay under the concurrent-eval-job quota.

    Each task launches up to 2 serverless jobs (student + base). Reads the quota (falls back to 1 task in
    flight if undeterminable) and keeps that many tasks running, launching the next as each finishes.

    Returns (executions, results): {task: Execution} and {task: final_status_string}. Prints progress; a
    trailing warning lists any failed tasks.
    """
    import boto3
    from sagemaker.train.evaluate import CustomScorerEvaluator
    from sagemaker.core import Attribution, set_attribution
    set_attribution(Attribution.SAGEMAKER_AGENT_PLUGIN)

    jobs_per_task = 2 if evaluate_base else 1
    quota = _eval_job_quota(region)
    if quota is None:
        max_in_flight = 1
        print(f"eval-job quota: UNKNOWN (lookup failed/throttled) -> fallback: 1 task "
              f"({jobs_per_task} jobs) at a time, {len(datasets)} sequential group(s)\n")
    else:
        max_in_flight = max(1, quota // jobs_per_task)
        print(f"eval-job quota = {quota}; {jobs_per_task} jobs/task -> scheduling <= {max_in_flight} task(s)\n")

    terminal = ("succeeded", "stopped", "completed")

    def submit(name):
        job = CustomScorerEvaluator(
            model=model_pkg_arn, evaluator=evaluator_arn, dataset=datasets[name],
            s3_output_path=f"{s3_output}{name}/", evaluate_base_model=evaluate_base,
            region=region, mlflow_experiment_name=mlflow_experiment)
        with _quiet_sdk_logs(quiet):
            ex = job.evaluate()
        print(f"started '{name}': {ex.name}  ({ex.status.overall_status})")
        return ex

    def live_status(ex):
        # overall_status is cached at submit time ('Executing') and doesn't auto-update — refresh() each poll.
        try:
            ex.refresh()
        except Exception:
            pass
        return (ex.status.overall_status or "").lower()

    executions, pending, running, results = {}, list(datasets.keys()), {}, {}
    while pending or running:
        while pending and len(running) < max_in_flight:
            name = pending.pop(0)
            running[name] = executions[name] = submit(name)
            time.sleep(5)   # small stagger so step submissions don't race the quota check
        done = []
        for name, ex in running.items():
            status = live_status(ex)                       # one refresh per execution per poll
            if "fail" in status or status in terminal:
                done.append((name, status))
        for name, status in done:
            results[name] = status; running.pop(name)
            print(f"'{name}' -> {status}")
        if pending or running:
            time.sleep(30)

    print("\nfinal per-task status:", results)
    failed = [t for t, s in results.items() if "fail" in s.lower()]
    print(f"⚠️  failed tasks (re-run these): {failed}" if failed else "✅ all per-task jobs succeeded")
    return executions, results


# ---------------------------------------------------------------- results
def read_task_metrics(*, datasets, s3_output, counts, region, evaluate_base=True):
    """Read per-task student (and base) custom metrics from each job's output.

    Returns per_task[t] = {"student": {metric: val}, "base": {metric: val}, "n": count}.
    """
    import boto3

    def _read(output_prefix, step_substring):
        s3 = boto3.client("s3", region_name=region)
        bkt, prefix = output_prefix.replace("s3://", "").split("/", 1)
        keys = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=bkt, Prefix=prefix)
                for o in pg.get("Contents", [])
                if step_substring in o["Key"] and "/eval_results/results_" in o["Key"] and o["Key"].endswith(".json")]
        if not keys:
            return {}
        doc = json.loads(s3.get_object(Bucket=bkt, Key=sorted(keys)[-1])["Body"].read())
        res = doc.get("results", {})
        task_key = next((k for k in res if k.startswith("custom|")), None)
        return res.get(task_key, {}) if task_key else {}

    per_task = {}
    for t in datasets:
        out = f"{s3_output}{t}/"
        per_task[t] = {
            "student": _read(out, "EvaluateCustomModel"),
            "base":    _read(out, "EvaluateBaseModel") if evaluate_base else {},
            "n":       counts[t],
        }
    return per_task


def macro_mean(per_task, role, metric):
    """Simple mean of a metric across tasks that reported it for the given role (each task weighted equally)."""
    vals = [per_task[t][role][metric] for t in per_task if per_task[t][role].get(metric) is not None]
    return sum(vals) / len(vals) if vals else None


def _pct(x):
    return f"{x:.1%}" if x is not None else "  n/a"


def print_results_tables(per_task, eval_tasks, metrics):
    """Print a per-task student/base/Δ table for each metric, with a macro-average aggregate row.

    metrics: list of (metric_key, title). Returns {metric_key: {"student": agg, "base": agg}} so the caller
    can reuse the aggregates (e.g. for the gate) without recomputing them.
    """
    hdr = f"{'task':<10} {'n':>5} {'student':>9} {'base':>9} {'Δ pts':>8}"
    n_total = sum(per_task[t]["n"] for t in per_task)
    aggregates = {}
    for metric, title in metrics:
        print(("\n" if metric != metrics[0][0] else "") + title)
        print(hdr); print("-" * len(hdr))
        for t in eval_tasks:
            if t not in per_task:
                continue
            s = per_task[t]["student"].get(metric)
            b = per_task[t]["base"].get(metric)
            d = f"{100*(s-b):+.1f}" if (s is not None and b is not None) else "   -"
            print(f"{t:<10} {per_task[t]['n']:>5} {_pct(s):>9} {_pct(b):>9} {d:>8}")
        print("-" * len(hdr))
        sa, ba = macro_mean(per_task, "student", metric), macro_mean(per_task, "base", metric)
        da = f"{100*(sa-ba):+.1f}" if (sa is not None and ba is not None) else "   -"
        print(f"{'Aggregate':<10} {n_total:>5} {_pct(sa):>9} {_pct(ba):>9} {da:>8}   (mean over tasks)")
        aggregates[metric] = {"student": sa, "base": ba}
    return aggregates


def mlflow_experiment_link(experiment_name, session, resource_arn=None):
    """Resolve the MLflow app ARN the eval jobs logged to and render a compact clickable presigned link to the
    experiment. Best-effort — a tracking/permissions hiccup prints a note instead of raising.

    resource_arn: an explicit mlflow-app/tracking-server ARN to use; None resolves it from `session`.
    """
    print("\n--- MLflow tracking ---")
    print(f"experiment name : {experiment_name}")
    arn = resource_arn
    if not arn:
        try:
            from sagemaker.train.common_utils.finetune_utils import _resolve_mlflow_resource_arn
            # Use the notebook's region-bound session, NOT a bare sagemaker.Session(): the resolver lists MLflow
            # apps via this session's SageMaker client, so a default session can resolve the WRONG app.
            # dry_run=True keeps it read-only (never creates an app).
            arn = _resolve_mlflow_resource_arn(session, None, dry_run=True)
        except Exception as e:
            print(f"(MLflow ARN could not be resolved automatically: {e})")
            print("resource ARN    : <unresolved>")
            return
    if not arn:
        print("resource ARN    : <unresolved>")
        return
    print(f"resource ARN    : {arn}")
    try:
        from sagemaker.train.common_utils.job_wait import _get_mlflow_presigned_url
        url = _get_mlflow_presigned_url(arn, experiment_name=experiment_name)
        if url:
            from IPython.display import display, Markdown   # compact clickable link, not the huge raw URL
            display(Markdown(f"**MLflow:** [▶ Open experiment '{experiment_name}']({url}) "
                             f"&nbsp;·&nbsp; _presigned link, short-lived — re-run this cell for a fresh one_"))
        else:
            print("open experiment : <could not mint presigned URL; open MLflow from the Studio left nav>")
    except Exception as e:
        print(f"open experiment : <presigned URL unavailable: {e}>")
