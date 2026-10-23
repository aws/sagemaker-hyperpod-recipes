"""Per-step tool-call reward function for the litmus Custom Scorer eval.

This file is NOT imported by the notebook — it is the source that `Evaluator.create(source=...)` ships to
SageMaker, where it runs as the managed eval job's scoring Lambda. The managed job calls `reward_function`
once per sample with the student's generated action (`model_response`), the query, and the teacher's
ground-truth action (`reference_answer` / `response`).

Design (three graded levels; see the notebook's "custom reward function" section):
  * kind_match     — right *kind* of turn (tool_call vs response)?   weakest signal, reported as a Metric.
  * toolset_match  — right *tool names, in order*, ignoring params?  THE GATE (aggregate_reward_score). Parsed
                     leniently so it measures tool SELECTION, not formatting.
  * format_match   — right tools AND the exact harness `<tool_call>[{...}]</tool_call>` array form?  reported as
                     a Metric; this is what the base model usually misses.

Required contract for the Custom Scorer runtime: keep `lambda_handler` and `reward_function` with these names.
"""
import json
import re
from typing import Any, Dict


# ---- 1. parse a model/reference turn into (kind, tool-names) --------------------------------------
def _tool_names_lenient(body):
    """Parse a <tool_call> body into a list of tool names, tolerant of object-vs-array form."""
    body = body.strip()
    for cand in (body, f"[{body}]"):  # try as-is, then wrap a bare object in an array
        try:
            arr = json.loads(cand)
            if isinstance(arr, dict):
                arr = [arr]
            if isinstance(arr, list):
                return [c.get("name") for c in arr if isinstance(c, dict)]
        except Exception:
            pass
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', body)  # last resort: pull names by regex
    return names or None


def parse_turn(text, strict):
    """Return ('tool_call', [names]) | ('response', text) | ('none', None).
    strict=True requires the exact harness array form `<tool_call>[...]</tool_call>`."""
    if not text:
        return ("none", None)
    # Drop reasoning. With thinking on, the chat template opens <think> in the prompt, so a reply carries only
    # the closing tag; keep the text after the LAST </think> so a tool call mentioned while reasoning isn't scored.
    text = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    if strict:
        m = re.search(r"<tool_call>\s*(\[.*?\])\s*</tool_call>", text, flags=re.S)
        if m:
            try:
                return ("tool_call", [c.get("name") for c in json.loads(m.group(1))])
            except Exception:
                return ("tool_call", None)
    else:
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S)
        if m:
            return ("tool_call", _tool_names_lenient(m.group(1)))
    if re.search(r"<response>.*?</response>", text, flags=re.S):
        return ("response", text)
    return ("none", None)


# ---- 2. score a prediction against the reference tool names ---------------------------------------
def _toolset(ref_names, pred):
    pk, pnames = pred
    if ref_names is None:  # ref is a response/none turn: match on kind only
        return 1.0 if pk != "tool_call" else 0.0
    return 1.0 if (pk == "tool_call" and pnames == ref_names and len(ref_names) > 0) else 0.0


# ---- 3. the reward function the managed job calls per sample --------------------------------------
def reward_function(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    pred = sample.get("model_response", "")
    ref = sample.get("reference_answer", "")
    if isinstance(ref, dict):
        ref = ref.get("text", "") or sample.get("response", "")
    else:
        ref = ref or sample.get("response", "")

    # Reference tool names (ground truth is always the strict harness array form).
    rk, rnames = parse_turn(ref, strict=True)
    ref_names = rnames if rk == "tool_call" else None

    # Lenient (tool selection) and strict (exact format) predictions.
    toolset = _toolset(ref_names, parse_turn(pred, strict=False))  # right tools, any format
    fmt = _toolset(ref_names, parse_turn(pred, strict=True))  # right tools AND exact harness format
    kind = 1.0 if (parse_turn(pred, strict=False)[0] == rk) else 0.0  # same turn type as the reference?

    return {
        "id": str(sample.get("id", f"sample-{index:03d}")),
        "aggregate_reward_score": float(toolset),  # gate metric: tool selection (format-agnostic)
        "metrics_list": [
            {"name": "toolset_match", "value": float(toolset), "type": "Reward"},
            {"name": "format_match", "value": float(fmt), "type": "Metric"},
            {"name": "kind_match", "value": float(kind), "type": "Metric"},
        ],
    }


# ---- 4. Lambda entry point ------------------------------------------------------------------------
def lambda_handler(event, context):
    try:
        samples = event if isinstance(event, list) else [event]
        result = reward_function(samples[0], 0)
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps([result])}
    except Exception as e:
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
