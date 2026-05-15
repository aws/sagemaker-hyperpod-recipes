#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
Resolve a single Hydra recipe and write the fully-resolved YAML to a file.

Companion to generate_resolved_recipes.py (which batch-resolves all recipes).
This script is useful for debugging individual recipes and comparing against
training job output.

Usage:
    python scripts/hydra_resolver.py \\
        --config-dir hyperpod_recipes/recipes_src/fine-tuning \\
        --recipe-name qwen-0_7_0/verl-grpo-rlvr-qwen-3-5-9b-lora \\
        --output /tmp/qwen-3-5-9b-resolved.yaml

    # With overrides:
    python scripts/hydra_resolver.py \\
        --config-dir hyperpod_recipes/recipes_src/fine-tuning \\
        --recipe-name qwen-0_7_0/verl-grpo-rlvr-qwen-3-5-9b-lora \\
        --output /tmp/qwen-3-5-9b-resolved.yaml \\
        --overrides actor_rollout_ref.actor.optim.lr=5e-6
"""

import argparse
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def resolve_config(config_dir, recipe_name, overrides=None):
    """Resolve a single Hydra recipe config.

    Handles the two-key layout where Hydra produces a top-level config with
    a ``recipes`` key and a per-model key that need to be merged, then
    extracts ``training_config``.
    """
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=recipe_name)

    # Merge the "recipes" defaults with the per-model overrides
    if "recipes" in cfg and len(cfg) == 2:
        keys = list(cfg.keys())
        per_model_key = keys[0] if keys[1] == "recipes" else keys[1]
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg.recipes, cfg[per_model_key])

    cfg = cfg.training_config

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Resolve a single Hydra recipe and write sorted YAML output.")
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Path to Hydra config directory (e.g. hyperpod_recipes/recipes_src/fine-tuning)",
    )
    parser.add_argument(
        "--recipe-name",
        required=True,
        help="Recipe name relative to config-dir (e.g. qwen-0_7_0/verl-grpo-rlvr-qwen-3-5-9b-lora)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file path for the resolved YAML",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides (e.g. actor_rollout_ref.actor.optim.lr=5e-6)",
    )
    args = parser.parse_args()

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        print(f"Error: Config directory '{args.config_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = resolve_config(config_dir, args.recipe_name, args.overrides or None)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        OmegaConf.resolve(cfg)

        with open(args.output, "w") as f:
            f.write(OmegaConf.to_yaml(cfg, sort_keys=True))

        print(f"Resolved config written to: {args.output}")
    except Exception as e:
        print(f"Error resolving config: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
