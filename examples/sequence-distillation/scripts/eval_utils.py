"""Evaluation helpers for the ShoppingBench eval notebook — the mechanical scoring/table-assembly plumbing,
kept out of the notebook so the cells can focus on the eval *decisions* (which metric, how to aggregate).
"""


def score_all(harness, urls, eval_tasks, jobs, clean):
    """Score every (model, task) cell and assemble the results table.

    Isolates per-cell failures: if one task's scoring errors (e.g. a broken/empty rollout), it's recorded and
    the rest keep going, so a long run's worth of results isn't lost to one bad cell.

    Args:
        harness: the ShoppingBench Harness (provides .score and .health).
        urls:    the per-model relays dict (its keys are the model tags: base/student/frontier/reference_*).
        eval_tasks, jobs, clean: the notebook's EVAL_TASKS, JOBS (tag_task -> (cfg, rollout, n)), CLEAN.

    Returns (results, df, errors):
        results: {(task, model): {metric: value, "health:<k>": value, ...}}
        df:      a pandas DataFrame of every scored cell, indexed by (task, model).
        errors:  {(task, model): short_reason} for cells that failed to score (empty if all succeeded).

    Raises RuntimeError if NOTHING scored (all cells failed) — there's no table to return.
    """
    import time

    import pandas as pd

    results, errors = {}, {}
    cells = [(tag, task) for tag in urls for task in eval_tasks]
    for i, (tag, task) in enumerate(cells, 1):
        rollout = jobs[f"{tag}_{task}"][1]
        t0 = time.time()
        try:
            score = harness.score(task, rollout, clean[task])
            # ASR over ALL held-out problems: a problem with no trajectory (stopped by the stall guard, endpoint
            # errors) counts as a failure instead of silently dropping out of the denominator.
            total = sum(1 for line in open(clean[task]) if line.strip())
            if score.get("n") and total and score["n"] < total:
                score = {**score, "asr_scored": score["asr"], "asr": round(score["asr"] * score["n"] / total, 3)}
            score["missing"] = max(total - (score.get("n") or 0), 0)
            health = {f"health:{k}": v for k, v in harness.health(rollout, clean[task]).items()}
            results[(task, tag)] = {**score, **health}
            print(f"[{i}/{len(cells)}] scored {tag}/{task}: asr={score.get('asr')} ({time.time() - t0:.1f}s)")
        except Exception as e:
            errors[(task, tag)] = str(e)[-300:]
            print(f"[{i}/{len(cells)}] ⚠️  scoring failed for {tag}/{task}: {str(e)[-150:]}")
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
    # reference_qwen3-32b (Qwen3 32B dense, same family as base/student) and the teacher (reference_glm-5.3) are
    # shown before the frontier comparison model.
    preferred = ["base", "student", "reference_qwen3-32b", "reference_glm-5.3", "frontier"]
    present = {m for (t, m) in results}
    extra_students = sorted(m for m in present if m.startswith("student") and m not in preferred)
    extra_refs = sorted(m for m in present if m.startswith(("reference", "frontier")) and m not in preferred)
    ordered = (
        ["base", "student"] + extra_students + ["reference_qwen3-32b", "reference_glm-5.3", "frontier"] + extra_refs
    )
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
    fig.suptitle(
        "Held-out ShoppingBench problems"
        + (" (base/student thinking on" if enable_thinking else " (base/student thinking off")
        + "; cached references keep their own settings)"
    )
    fig.tight_layout()
    plt.show()


def plot_ifeval(ifeval, labels=None):
    """Horizontal grouped bar chart for IFEval: prompt_strict vs inst_strict per model, with a base->student
    delta callout and a visible flag on any model whose reasoning_leaks > 0 (score invalid for that model —
    see the "## 8. IFEval" markdown cell for why a leak invalidates the number).

    ifeval: {model: {"prompt_strict": float, "inst_strict": float, "reasoning_leaks": int}} (the notebook's
            IFEVAL dict, in whatever state it's in — models not yet scored are simply absent).
    labels: optional {model: display_name} (default: the model tag itself).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = labels or {}
    if not ifeval:
        print("IFEVAL is empty — nothing to plot yet (run/finish the IFEval cell above first).")
        return

    # Same model ordering convention as plot_results: base/student first, then reference/frontier comparisons.
    preferred = ["base", "student", "reference_qwen3-32b", "reference_glm-5.3", "frontier"]
    extra_students = sorted(m for m in ifeval if m.startswith("student") and m not in preferred)
    extra_refs = sorted(m for m in ifeval if m.startswith(("reference", "frontier")) and m not in preferred)
    ordered = (
        ["base", "student"] + extra_students + ["reference_qwen3-32b", "reference_glm-5.3", "frontier"] + extra_refs
    )
    models = [m for m in ordered if m in ifeval]

    metrics = [("prompt_strict", "Prompt-level strict acc."), ("inst_strict", "Instruction-level strict acc.")]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(models) + 1.5))
    h = 0.8 / len(metrics)
    colors = {"prompt_strict": "#4C72B0", "inst_strict": "#55A868"}
    for i, (metric, title) in enumerate(metrics):
        vals = [ifeval[m][metric] for m in models]
        offset = (i - (len(metrics) - 1) / 2) * h
        bars = ax.barh(y + offset, vals, h, label=title, color=colors.get(metric))
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=8, padding=3)

    # Flag any model with leaked reasoning: its score is invalid (see markdown above the IFEval cell).
    row_labels = []
    for m in models:
        leaks = ifeval[m].get("reasoning_leaks", 0)
        name = labels.get(m, m)
        row_labels.append(f"{name}  ⚠ INVALID ({leaks} leak{'s' if leaks != 1 else ''})" if leaks else name)
    ax.set_yticks(y, row_labels)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()  # first model (base) on top
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.9)
    ax.set_xlabel("accuracy")

    # Regression-check callout: base -> student delta, mirroring the ASR gap-closed framing used for ASR.
    title = "IFEval — general instruction following (held out of distillation)"
    if "base" in ifeval and "student" in ifeval:
        for metric, short in [("prompt_strict", "prompt"), ("inst_strict", "inst")]:
            b, s = ifeval["base"][metric], ifeval["student"][metric]
            title += f"\n{short}: base {b:.3f} -> student {s:.3f} ({100 * (s - b):+.1f} pts)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    plt.show()
