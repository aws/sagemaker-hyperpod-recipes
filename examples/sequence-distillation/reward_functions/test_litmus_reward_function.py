"""Offline unit tests for the litmus reward function — run before registering it as an Evaluator, so a scoring
bug is caught here (fast) instead of inside a managed eval job (slow). Run: python reward_functions/test_litmus_reward_function.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litmus_reward_function as rf

REF = '<tool_call>[{"name": "find_product", "parameters": {"q": "jeans"}}]</tool_call>'  # ground truth


def score(pred, ref=REF):
    out = rf.reward_function({"model_response": pred, "reference_answer": ref}, 0)
    m = {d["name"]: d["value"] for d in out["metrics_list"]}
    return out["aggregate_reward_score"], m


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    assert cond, label


print("reward_function behavior:")

# 1. Student emits the exact harness array form -> right tools AND right format.
agg, m = score('<tool_call>[{"name": "find_product", "parameters": {"q": "baggy jeans"}}]</tool_call>')
check("exact array form: toolset=1, format=1", agg == 1.0 and m["format_match"] == 1.0)

# 2. Base-style: correct tool but a BARE OBJECT inside <think> -> tool selection right, exact format wrong.
#    This is the central distinction the two metrics are designed to separate.
agg, m = score('<think>I should search</think><tool_call>{"name": "find_product", "parameters": {}}</tool_call>')
check("bare object (base style): toolset=1, format=0", agg == 1.0 and m["format_match"] == 0.0)

# 3. Wrong tool entirely -> both zero.
agg, m = score('<tool_call>[{"name": "terminate", "parameters": {}}]</tool_call>')
check("wrong tool: toolset=0, format=0", agg == 0.0 and m["format_match"] == 0.0)

# 4. A plain response when the reference wanted a tool call -> toolset 0, kind 0.
agg, m = score("<response>Here you go</response>")
check("response vs tool-call ref: toolset=0, kind=0", agg == 0.0 and m["kind_match"] == 0.0)

# 5. Reference itself is a response (no tool call) -> a predicted response matches on kind.
agg, m = score("<response>ok</response>", ref="<response>sure</response>")
check("response-vs-response ref: toolset=1", agg == 1.0)

# 6. Empty / garbage prediction -> no crash, zero.
agg, m = score("")
check("empty prediction: toolset=0 (no crash)", agg == 0.0)

# 7. lambda_handler envelope: 200 + a JSON body list with the metrics.
resp = rf.lambda_handler({"model_response": REF, "reference_answer": REF}, None)
body = json.loads(resp["body"])
check("lambda_handler: 200 + well-formed body", resp["statusCode"] == 200 and body[0]["aggregate_reward_score"] == 1.0)

print("\nAll reward-function tests passed. ✅  Safe to register as an Evaluator.")
