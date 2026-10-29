"""Turn harness rollouts (one JSON list of steps per line) into SFT rows for the serverless SFTTrainer.

Each rollout step already records the exact prompt the agent saw ([system, user], where the user message is
ShoppingBench's flattened "Dialogue Records History") and the teacher's reply. One step -> one row:
  user:      "{system}\n\n{user}"   (system folded into the user turn, as in notebook 1)
  assistant: "<think>\n{reasoning}\n</think>\n\n{tool_call / response}"
The relay returns the model's native reasoning as a leading <think>; ShoppingBench's prompt also asks for a
visible <think>, so a reply can carry two blocks. Both are merged into the one think block the Qwen3.5 chat
template expects. Steps whose reply has no parseable action are dropped.

Usage: python encode_rollouts.py OUT.jsonl ROLLOUT.jsonl [ROLLOUT.jsonl ...]
"""
import json
import re
import sys

THINK = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
RESPONSE = re.compile(r"<response>.*?</response>", re.S)


def target(content):
    """Normalize a teacher reply to one think block + its actions; None if it has no valid action."""
    thoughts = [t.strip() for t in THINK.findall(content) if t.strip()]
    rest = THINK.sub("", content).strip()
    calls = TOOL_CALL.findall(rest)
    actions = []
    for c in calls:
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            return None
        parsed = parsed if isinstance(parsed, list) else [parsed]
        actions.append(f"<tool_call>{json.dumps(parsed, ensure_ascii=False)}</tool_call>")
    actions += RESPONSE.findall(rest)
    if not actions:
        return None
    # dedupe identical thoughts (the visible block sometimes repeats the native reasoning)
    merged = "\n\n".join(dict.fromkeys(thoughts))
    return f"<think>\n{merged}\n</think>\n\n" + "\n".join(actions)


THINK_IN_HISTORY = re.compile(r"<think>.*?</think>\n?", re.S)


def sft_rows(rollout_path, keep_queries, include_reasoning=True):
    """SFT rows from one rollout file, keeping only trajectories whose query is in keep_queries (e.g. the solved
    ones). Returns [(query, {"messages": [user, assistant]}), ...]. Without reasoning, the <think> blocks are
    removed from the target and from the transcript of earlier steps, which is what a non-thinking student sees."""
    out = []
    for line in open(rollout_path):
        if not line.strip():
            continue
        traj = json.loads(line)
        q = traj[0]["extra_info"]["query"]
        if q not in keep_queries:
            continue
        for step in traj:
            system = next(m["content"] for m in step["prompt"] if m["role"] == "system")
            user = next(m["content"] for m in step["prompt"] if m["role"] == "user")
            t = target(step["completion"].get("content") or "")
            if t is None:
                continue
            if not include_reasoning:
                user, t = THINK_IN_HISTORY.sub("", user), t.split("</think>\n\n", 1)[1]
            out.append(
                (
                    q,
                    {
                        "messages": [
                            {"role": "user", "content": f"{system}\n\n{user}"},
                            {"role": "assistant", "content": t},
                        ]
                    },
                )
            )
    return out


def rows_from(rollout_path):
    kept = dropped = 0
    for line in open(rollout_path):
        for step in json.loads(line):
            system = next(m["content"] for m in step["prompt"] if m["role"] == "system")
            user = next(m["content"] for m in step["prompt"] if m["role"] == "user")
            t = target(step["completion"].get("content") or "")
            if t is None:
                dropped += 1
                continue
            kept += 1
            yield {
                "messages": [
                    {"role": "user", "content": f"{system}\n\n{user}"},
                    {"role": "assistant", "content": t},
                ]
            }
    print(f"{rollout_path}: kept {kept} steps, dropped {dropped}", file=sys.stderr)


if __name__ == "__main__":
    out, *paths = sys.argv[1:]
    with open(out, "w") as f:
        for p in paths:
            for row in rows_from(p):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
