"""Build execution matrix for the auto-configurator workflow."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation_helpers import _detect_new_recipes

RECIPES_PREFIX = "recipes_collection/recipes/"


def detect_new_recipes(base_branch="main"):
    """Detect newly added recipes using shared helper."""
    subprocess.run(["git", "fetch", "origin", f"{base_branch}:{base_branch}"], check=True, capture_output=True)
    new = _detect_new_recipes(base_branch, RECIPES_PREFIX)
    return [r[len(RECIPES_PREFIX) :] for r in new if r.startswith(RECIPES_PREFIX)]


def main():
    recipes_str = os.environ.get("RECIPES", "")
    branch = os.environ.get("BRANCH", "")
    run_id = os.environ.get("RUN_ID", "")[:16]
    instance_types_str = os.environ.get("INSTANCE_TYPES", "")
    seq_lengths_str = os.environ.get("SEQUENCE_LENGTHS", "")
    base_branch = os.environ.get("BASE_BRANCH", "main")
    output_file = os.environ.get("GITHUB_OUTPUT", "")

    # Detect recipes if not provided
    if not recipes_str:
        recipes = detect_new_recipes(base_branch)
        if not recipes:
            if output_file:
                with open(output_file, "a") as f:
                    f.write("skip=true\n")
            print("No new recipes detected.")
            return
    else:
        recipes = [r.strip() for r in recipes_str.split(",") if r.strip()]

    instance_types = (
        [i.strip() for i in instance_types_str.split(",") if i.strip()]
        if instance_types_str
        else [
            "ml.p5.48xlarge",
            "ml.p4d.24xlarge",
            "ml.p4de.24xlarge",
            "ml.g5.12xlarge",
            "ml.g5.48xlarge",
            "ml.g6.48xlarge",
        ]
    )
    sequence_lengths = [int(s.strip()) for s in seq_lengths_str.split(",") if s.strip()] if seq_lengths_str else [4096]

    entries = []
    for recipe in recipes:
        entries.append(
            {
                "recipes": [recipe],
                "instance_type_list": instance_types,
                "sequence_lengths": sequence_lengths,
                "use_github_branch": branch,
                "run_id": run_id,
                "source": "github_pr",
            }
        )

    if output_file:
        with open(output_file, "a") as f:
            f.write(f"execution_matrix={json.dumps(entries)}\n")
            f.write(f"run_id={run_id}\n")
            f.write("skip=false\n")
            f.write(f"branch={branch}\n")

    print(f"Generated {len(entries)} execution(s) for {len(recipes)} recipe(s)")


if __name__ == "__main__":
    main()
