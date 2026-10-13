"""Evaluation helpers for the ShoppingBench eval notebook — the mechanical scoring/table-assembly plumbing,
kept out of the notebook so the cells can focus on the eval *decisions* (which metric, how to aggregate).
"""


def score_all(harness, urls, eval_tasks, jobs, clean):
    """Score every (model, task) cell and assemble the results table.

    Isolates per-cell failures: if one task's scoring errors (e.g. a broken/empty rollout), it's recorded and
    the rest keep going, so a long run's worth of results isn't lost to one bad cell.

    Args:
        harness: the ShoppingBench Harness (provides .score and .health).
        urls:    the per-model relays dict (its keys are the model tags: base/student/frontier_*/reference_*).
        eval_tasks, jobs, clean: the notebook's EVAL_TASKS, JOBS (tag_task -> (cfg, rollout, n)), CLEAN.

    Returns (results, df, errors):
        results: {(task, model): {metric: value, "health:<k>": value, ...}}
        df:      a pandas DataFrame of every scored cell, indexed by (task, model).
        errors:  {(task, model): short_reason} for cells that failed to score (empty if all succeeded).

    Raises RuntimeError if NOTHING scored (all cells failed) — there's no table to return.
    """
    import pandas as pd

    results, errors = {}, {}
    for tag in urls:
        for task in eval_tasks:
            rollout = jobs[f"{tag}_{task}"][1]
            try:
                score = harness.score(task, rollout, clean[task])
                health = {f"health:{k}": v for k, v in harness.health(rollout, clean[task]).items()}
                results[(task, tag)] = {**score, **health}
            except Exception as e:
                errors[(task, tag)] = str(e)[-300:]
                print(f"⚠️  scoring failed for {tag}/{task}: {str(e)[-150:]}")
    if not results:
        raise RuntimeError(
            "No tasks scored — all scoring calls failed. Inspect the warnings above / the " "returned errors dict."
        )
    df = (
        pd.DataFrame([{"task": task, "model": tag, **r} for (task, tag), r in results.items()])
        .set_index(["task", "model"])
        .sort_index()
    )
    return results, df, errors


def plot_results(results, eval_tasks, metrics, avg=None, enable_thinking=False, labels=None):
    """Grouped bar chart (one panel per metric) comparing models across intents + an aggregate column.

    results:   {(task, model): {metric: value, ...}} from score_all.
    eval_tasks: ordered intent names (the per-group x positions, plus a trailing "Avg." when >1).
    metrics:   list of (metric_key, panel_title) — one subplot each (e.g. ("asr", "ASR ...")).
    avg:       {model: aggregate_asr} precomputed by the caller (reused so the chart's Avg. column matches
               the printed aggregate exactly). The Avg. column is only drawn for the "asr" metric; for other
               metrics it's the simple mean over intents computed here. Pass None to compute asr's avg here too.
    labels:    optional {model: display_name} (default: the model tag itself).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = labels or {}
    # base + student first, then any additional student_* tags (e.g. a side-by-side new student that
    # hasn't replaced "student" yet), then the reference/frontier comparison tags.
    # reference_qwen3-32b (Qwen3 32B dense, same family as base/student) is shown before the frontier
    # comparison model (frontier_medium).
    preferred = ["base", "student", "reference_qwen3-32b", "frontier_medium"]
    present = {m for (t, m) in results}
    extra_students = sorted(m for m in present if m.startswith("student") and m not in preferred)
    extra_refs = sorted(m for m in present if m.startswith(("reference", "frontier")) and m not in preferred)
    ordered = ["base", "student"] + extra_students + ["reference_qwen3-32b", "frontier_medium"] + extra_refs
    models = [m for m in ordered if any((t, m) in results for t in eval_tasks)]
    groups = list(eval_tasks) + (["Avg."] if len(eval_tasks) > 1 else [])

    def value(metric, group, model):
        if group != "Avg.":
            return results.get((group, model), {}).get(metric, np.nan)
        if metric == "asr" and avg is not None:
            return avg.get(model, np.nan)  # reuse the caller's aggregate for ASR
        vals = [results.get((t, model), {}).get(metric) for t in eval_tasks]
        return sum(vals) / len(vals) if all(v is not None for v in vals) else np.nan

    # Stack the metric panels vertically (one per row) so each chart is on its own line.
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9.0, 4.2 * len(metrics)), squeeze=False)
    panels = [row[0] for row in axes]  # flatten the single-column grid to a list of axes
    x, w = np.arange(len(groups)), 0.8 / max(len(models), 1)
    for ax, (metric, title) in zip(panels, metrics):
        for i, m in enumerate(models):
            vals = [value(metric, g, m) for g in groups]
            bars = ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, label=labels.get(m, m))
            ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.2f}" for v in vals], fontsize=8, padding=2)
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        # Legend placed OUTSIDE the axes (to the right) so it never overlaps bars/labels.
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)
    fig.suptitle("Held-out ShoppingBench problems" + (" (thinking on)" if enable_thinking else " (thinking off)"))
    fig.tight_layout()
    plt.show()
