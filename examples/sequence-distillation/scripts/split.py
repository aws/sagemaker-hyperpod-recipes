"""The fixed ShoppingBench split: which test problems train the student (the Bedrock teacher's prompts, notebook 1)
and which are held out for evaluation (notebook 3).

split/<task>.json stores 0-based line indices into ShoppingBench's data/synthesize_<task>_test.jsonl (at the commit
pinned in setup_shopbench.sh), not the problems themselves, plus a short SHA-256 of that file so a different
ShoppingBench version fails loudly instead of silently changing the split. One voucher problem (line 49) is in
neither set ("unused"), exactly as in the published runs.

Usage: python split.py <ShoppingBench>/data   (checks every task and prints the counts)
"""
import hashlib
import json
import sys
from pathlib import Path

SPLIT_DIR = Path(__file__).resolve().parent.parent / "split"
COUNTS = {"product": (147, 103), "shop": (145, 105), "voucher": (133, 116)}  # (train, heldout) problems


def load_split(sb_data_dir, task):
    """(train_rows, heldout_rows): the parsed test problems of one task, in file order."""
    spec = json.loads((SPLIT_DIR / f"{task}.json").read_text())
    path = Path(sb_data_dir) / spec["file"]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest.startswith(spec["sha256"]), (
        f"{path} has SHA-256 {digest[:16]}, split/{task}.json expects {spec['sha256']}: "
        "use the ShoppingBench commit pinned in setup_shopbench.sh"
    )
    lines = raw.decode("utf-8").split("\n")  # not splitlines(): queries may contain U+2028 and similar
    train = [json.loads(lines[i]) for i in spec["train"]]
    heldout = [json.loads(lines[i]) for i in spec["heldout"]]
    assert not set(spec["train"]) & set(spec["heldout"]), f"split/{task}.json: train and heldout overlap"
    assert (len(train), len(heldout)) == COUNTS[task], f"{task}: {len(train)} / {len(heldout)}, expected {COUNTS[task]}"
    return train, heldout


def write_jsonl(rows, path):
    """Write problems in the format the harness reads (one JSON object per line)."""
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


if __name__ == "__main__":
    for t in COUNTS:
        train, heldout = load_split(sys.argv[1], t)
        print(f"{t}: {len(train)} train / {len(heldout)} held-out problems")
