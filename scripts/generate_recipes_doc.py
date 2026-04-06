#!/usr/bin/env python3
"""Generate RECIPES.md documentation from recipe YAML files.

All metadata is derived from two sources:
  1. YAML content (authoritative) — run.model_type, display_name, instance_types,
     training_config fields, version, trainer.num_nodes, etc.
  2. Directory path (structural) — category, family, Nova technique subdirectory

No Hydra loading needed — all recipes under recipes_collection/recipes are
fully hydrated YAML files. Just yaml.safe_load().

Usage:
    python scripts/generate_recipes_doc.py           # Generate docs/RECIPES.md
    python scripts/generate_recipes_doc.py --check   # Validate docs/RECIPES.md is up-to-date
    python scripts/generate_recipes_doc.py --check --diff  # Show what changed
"""

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import yaml

# =============================================================================
# PATHS
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "recipes_collection" / "recipes"
LAUNCHER_DIR = REPO_ROOT / "launcher_scripts"
OUTPUT_FILE = REPO_ROOT / "docs" / "RECIPES.md"

# =============================================================================
# DOC SECTIONS — filter functions reference Recipe.framework (from run.model_type)
# =============================================================================
DOC_SECTIONS = [
    (
        "LLMFT (LLM Fine-Tuning Framework) Recipes",
        lambda r: r.framework == "llmft",
        "family",
        "",
    ),
    (
        "Checkpointless Training Recipes",
        lambda r: r.framework == "checkpointless",
        None,
        "",
    ),
    (
        "Evaluation Recipes",
        lambda r: r.category == "evaluation",
        "family",
        "",
    ),
    (
        "VERL (Versatile Reinforcement Learning) Recipes",
        lambda r: r.framework == "verl",
        "family",
        "VERL recipes support reinforcement learning from AI feedback (RLAIF) "
        "and verifiable rewards (RLVR) using GRPO.",
    ),
    (
        "Amazon Nova Recipes",
        lambda r: r.framework == "nova" and r.category != "evaluation",
        "family",
        "",
    ),
    (
        "Other Training Recipes",
        lambda r: r.framework not in ("llmft", "verl", "checkpointless", "nova") and r.category not in ("evaluation",),
        None,
        "",
    ),
]

# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class Recipe:
    model: str = ""
    framework: str = ""
    technique: str = ""
    adapter: str = ""
    seq_len: Optional[int] = None
    nodes: Optional[int] = None
    instance_type: str = ""
    version: str = ""
    category: str = ""
    family: str = ""
    recipe_path: str = ""
    script_path: str = ""


# =============================================================================
# YAML HELPERS
# =============================================================================


def _get(data: dict, *keys) -> Any:
    """Safely navigate nested dict."""
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
    return data


def _load_yaml(path: Path) -> dict:
    """Load fully-hydrated recipe YAML. No Hydra needed."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# =============================================================================
# PATH-BASED METADATA (category, family, nova technique dir)
# =============================================================================


def _path_metadata(recipe_path: Path) -> tuple:
    """Extract (category, family_dir, all_parts) from directory structure.

    Structure: recipes_collection/recipes/{category}/{family}/[...]/file.yaml
    """
    try:
        rel = recipe_path.relative_to(RECIPES_DIR)
        parts = rel.parts
        return (
            parts[0] if len(parts) > 0 else "other",
            parts[1] if len(parts) > 1 else "other",
            parts,
        )
    except ValueError:
        return "other", "other", ()


def _display_family(family_dir: str) -> str:
    """Titlecase family directory name. Only two non-obvious cases."""
    lower = family_dir.lower()
    if "gpt_oss" in lower:
        return "GPT-OSS"
    if lower == "deepseek":
        return "DeepSeek"
    if lower == "open-source":
        return "Open Source"
    if lower == "custom_model":
        return "Custom Model"
    # Strip version suffixes like "qwen-0_7_0" → "Qwen"
    base = lower.split("-")[0]
    return base.capitalize()


# =============================================================================
# FRAMEWORK — derived from run.model_type YAML field
# =============================================================================

# Maps run.model_type values to canonical framework names
_FRAMEWORK_MAP = {
    "llm_finetuning_aws": "llmft",
    "verl": "verl",
    "hyperpod_checkpointless_nemo": "checkpointless",
    "hf": "hf",
}


def _detect_framework(data: dict, category: str) -> str:
    """Derive framework from run.model_type YAML field."""
    model_type = _get(data, "run", "model_type") or ""

    # Direct lookup
    if model_type in _FRAMEWORK_MAP:
        return _FRAMEWORK_MAP[model_type]

    # Nova model types: "amazon.nova-lite-v1:0:300k", etc.
    if "amazon.nova" in model_type:
        return "nova"

    # Evaluation templates without run.model_type
    if category == "evaluation":
        return "evaluation"

    return model_type or "other"


# =============================================================================
# TECHNIQUE — derived from YAML fields per framework
# =============================================================================


def _detect_technique(data: dict, framework: str, path_parts: tuple) -> str:
    """Derive technique from authoritative YAML fields.

    - LLMFT: training_config.training_args.trainer_type (sft/dpo)
    - VERL: training_config.algorithm.adv_estimator (grpo) + reward type
    - Nova: parent directory name (SFT/, DPO/, PPO/, RFT/, distill/, CPT/)
    - Checkpointless: last meaningful filename token
    - Evaluation: eval type from filename
    """
    if framework == "llmft":
        tt = _get(data, "training_config", "training_args", "trainer_type")
        if isinstance(tt, str):
            return tt.upper()  # "sft" → "SFT", "dpo" → "DPO"
        return ""

    if framework == "verl":
        algo = _get(data, "training_config", "algorithm", "adv_estimator") or ""
        reward_fn = (
            _get(data, "training_config", "reward_model", "reward_kwargs", "kwargs", "reward_fn_name")
            or _get(data, "training_config", "reward_model", "reward_kwargs", "reward_fn_name")
            or ""
        )
        custom = _get(data, "training_config", "custom_reward_function", "name") or ""

        # Determine reward type
        reward_type = ""
        if reward_fn == "llmj":
            reward_type = "RLAIF"
        elif custom or reward_fn:
            reward_type = "RLVR"

        # Check if it's a pure SFT recipe (verl-sft-*)
        tt = _get(data, "training_config", "training_args", "trainer_type")
        if isinstance(tt, str) and tt.lower() == "sft":
            return "SFT"

        algo_upper = algo.upper() if algo else ""
        if reward_type:
            return f"{algo_upper} + {reward_type}".strip(" +")
        return algo_upper

    if framework == "nova":
        # Nova recipes organize by technique directory: SFT/, DPO/, PPO/, RFT/, distill/, CPT/
        technique_dirs = {
            "sft": "SFT",
            "dpo": "DPO",
            "ppo": "PPO",
            "rft": "RFT",
            "cpt": "CPT",
            "distill": "Distillation",
        }
        for part in path_parts:
            lower_part = part.lower()
            if lower_part in technique_dirs:
                return technique_dirs[lower_part]
        return ""

    if framework == "checkpointless":
        # Derive from display_name or filename: pretrain, lora, full_fine_tuning
        display = (data.get("display_name") or "").lower()
        if "pretrain" in display:
            return "Pretraining"
        if "full fine" in display:
            return "Full Fine-Tuning"
        if "lora" in display:
            return "LoRA"
        return ""

    if framework == "evaluation":
        stem = path_parts[-1] if path_parts else ""
        lower = stem.lower()
        if "deterministic" in lower:
            return "Deterministic Eval"
        if "llmaj" in lower or "llm_judge" in lower or "rubric_llm_judge" in lower:
            return "LLM-as-Judge Eval"
        if "bring_your_own" in lower:
            return "Custom Dataset Eval"
        if "general_text_benchmark" in lower:
            return "Text Benchmark Eval"
        if "general_multi_modal_benchmark" in lower:
            return "Multi-Modal Benchmark Eval"
        if "rft_eval" in lower:
            return "RFT Eval"
        return "Evaluation"

    return ""


# =============================================================================
# ADAPTER — derived from YAML peft fields per framework
# =============================================================================


def _detect_adapter(data: dict, framework: str) -> str:
    """Derive adapter type (LoRA/FFT) from YAML peft configuration fields.

    - LLMFT: training_config.model_config.peft_config.peft_type
    - VERL: training_config.actor_rollout_ref.model.lora_rank (0=FFT, >0=LoRA)
    - Nova: training_config.model.peft.peft_scheme
    - Checkpointless: model.peft.peft_type
    """
    if framework == "llmft":
        peft = _get(data, "training_config", "model_config", "peft_config")
        if isinstance(peft, dict):
            pt = peft.get("peft_type")
            if pt and str(pt).lower() == "lora":
                return "LoRA"
            return "FFT"
        return "FFT"  # No peft_config → full fine-tuning

    if framework == "verl":
        rank = _get(data, "training_config", "actor_rollout_ref", "model", "lora_rank")
        if rank is not None:
            return "LoRA" if int(rank) > 0 else "FFT"
        return ""

    if framework == "nova":
        scheme = _get(data, "training_config", "model", "peft", "peft_scheme")
        if scheme and str(scheme).lower() == "lora":
            return "LoRA"
        # No peft section → FFT (full fine-tuning)
        peft = _get(data, "training_config", "model", "peft")
        if isinstance(peft, dict) and not scheme:
            return ""  # Empty peft dict (Nova DPO without lora)
        return "FFT" if peft is None else ""

    if framework == "checkpointless":
        pt = _get(data, "model", "peft", "peft_type")
        if pt and str(pt).lower() != "none":
            return str(pt).upper()
        return ""

    return ""


# =============================================================================
# INSTANCE TYPE — from instance_types YAML field, with filename fallback for Nova
# =============================================================================


def _extract_instance_type(data: dict, recipe_path: Path) -> str:
    """Extract from YAML instance_types field. Fallback: parse filename tokens for Nova."""
    inst = data.get("instance_types")
    if isinstance(inst, list) and inst and isinstance(inst[0], str):
        return inst[0].replace("ml.", "")

    # Nova/training recipes embed instance info in filename tokens
    stem = recipe_path.stem.lower()
    parts = stem.split("_")

    # Check for specific instance tokens in filename
    # Ordered most-specific first to avoid p5 matching before p5en
    _INST_TOKENS = {
        "p5en": "p5en.48xlarge",
        "p5e": "p5en.48xlarge",
        "p5x32": "p5.48xlarge",
        "p5x24": "p5.48xlarge",
        "p5x16": "p5.48xlarge",
        "p5x8": "p5.48xlarge",
        "p5": "p5.48xlarge",
        "p4de": "p4de.24xlarge",
        "p4d": "p4d.24xlarge",
        "g6e": "g6e.12xlarge",
        "r5": "r5.2xlarge",
    }
    for part in parts:
        if part in _INST_TOKENS:
            return _INST_TOKENS[part]

    # Compound patterns: g5_g6_48x, g5_g6_12x
    joined = "_".join(parts)
    if "48x" in joined:
        for p in ("g6", "g5"):
            if p in parts:
                return f"{p}.48xlarge"
    if "12x" in joined:
        for p in ("g6", "g5"):
            if p in parts:
                return f"{p}.12xlarge"

    return ""


# =============================================================================
# SEQUENCE LENGTH — from YAML fields per framework
# =============================================================================


def _extract_seq_len(data: dict, framework: str) -> Optional[int]:
    """Extract sequence length from the authoritative YAML field per framework.

    - LLMFT: training_config.training_args.max_len
    - VERL: recursive max of ppo_max_token_len_per_gpu etc.
    - Nova FT: training_config.max_length
    - Nova Eval: inference.max_new_tokens
    - Checkpointless: data.seq_length
    - HF/custom: model.max_context_width
    """
    # Ordered by framework prevalence
    paths = [
        ("training_config", "training_args", "max_len"),  # LLMFT (87)
        ("training_config", "max_length"),  # Nova FT (30)
        ("inference", "max_new_tokens"),  # Nova eval (17)
        ("data", "seq_length"),  # Checkpointless (4)
        ("model", "max_context_width"),  # Falcon (1)
        ("evaluation", "max_length"),  # Eval templates
        ("training_config", "global_batch_size"),  # Skip — not seq len
    ]

    for path in paths:
        val = _get(data, *path)
        if val is not None and path != ("training_config", "global_batch_size"):
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # VERL: recursive search for max token length
    if framework == "verl":
        verl_keys = ("ppo_max_token_len_per_gpu", "forward_max_token_len_per_gpu", "log_prob_max_token_len_per_gpu")
        return _find_max_recursive(data, verl_keys)

    # Nova model_type context window: "amazon.nova-lite-v1:0:300k"
    model_type = _get(data, "run", "model_type") or ""
    m = re.search(r":0:(\d+)k", model_type)
    if m:
        return int(m.group(1)) * 1000

    return None


def _find_max_recursive(data: Any, target_keys: tuple) -> Optional[int]:
    """Recursively find maximum value for given keys in nested dict."""
    values = []

    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in target_keys and isinstance(v, (int, float)):
                    values.append(int(v))
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return max(values) if values else None


# =============================================================================
# NODE COUNT — from YAML fields
# =============================================================================


def _extract_nodes(data: dict) -> Optional[int]:
    """Extract node count from YAML. Priority: trainer.num_nodes > run.replicas > scale_config."""
    for path in [("trainer", "num_nodes"), ("run", "replicas"), ("training_config", "trainer", "num_nodes")]:
        val = _get(data, *path)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass

    # Scale config: minimum key
    sc = data.get("scale_config")
    if isinstance(sc, dict):
        int_keys = [int(k) for k in sc if str(k).isdigit()]
        return min(int_keys) if int_keys else None

    return None


# =============================================================================
# VERSION — from YAML, fallback to path for Nova
# =============================================================================


def _extract_version(data: dict, path_parts: tuple) -> str:
    """Version from YAML field. Nova fallback: path segment nova_1_0 → '1.0'."""
    v = data.get("version")
    if v is not None:
        return str(v)

    # Nova path fallback: look for nova_N_M pattern in path parts
    for part in path_parts:
        m = re.match(r"nova_(\d+)_(\d+)", part)
        if m:
            return f"{m.group(1)}.{m.group(2)}"

    return ""


# =============================================================================
# MODEL NAME — from YAML display_name, fallback to humanized filename
# =============================================================================


def _extract_model_name(data: dict, stem: str) -> str:
    """Prefer YAML display_name. Fallback: humanize filename stem."""
    dn = data.get("display_name")
    if dn:
        return str(dn)
    return stem.replace("_", " ").replace("-", " ")


# =============================================================================
# LAUNCHER SCRIPT FINDER
# =============================================================================


def _find_launcher_script(stem: str) -> str:
    """Find launcher script by convention: run_{stem}.sh (normalizing hyphens)."""
    target = f"run_{stem.replace('-', '_')}.sh"
    matches = []
    for root, _, files in os.walk(LAUNCHER_DIR):
        if target in files:
            matches.append(os.path.join(root, target))
    if not matches:
        return ""
    # Return the shortest path to prefer base directory over versioned variants
    return min(matches, key=len)


# =============================================================================
# RECIPE PARSING — pure YAML + path, no Hydra
# =============================================================================


def parse_recipe(recipe_path: Path) -> Optional[Recipe]:
    """Parse a fully-hydrated recipe YAML and return a Recipe with all metadata."""
    data = _load_yaml(recipe_path)
    if not data:
        return None

    stem = recipe_path.stem
    category, family_dir, path_parts = _path_metadata(recipe_path)
    family = _display_family(family_dir)
    framework = _detect_framework(data, category)
    technique = _detect_technique(data, framework, path_parts)
    adapter = _detect_adapter(data, framework)

    # For Nova eval recipes under evaluation/nova/, override family
    if category == "evaluation" and "nova" in str(recipe_path).lower():
        family = "Nova"
        framework = "nova"  # Use nova-specific technique detection
        technique = _detect_technique(data, "evaluation", path_parts)

    return Recipe(
        model=_extract_model_name(data, stem),
        framework=framework,
        technique=technique,
        adapter=adapter,
        seq_len=_extract_seq_len(data, framework),
        nodes=_extract_nodes(data),
        instance_type=_extract_instance_type(data, recipe_path),
        version=_extract_version(data, path_parts),
        category=category,
        family=family,
        recipe_path=str(recipe_path.relative_to(REPO_ROOT)),
        script_path=(os.path.relpath(_find_launcher_script(stem), REPO_ROOT) if _find_launcher_script(stem) else ""),
    )


def scan_recipes() -> List[Recipe]:
    """Scan all recipe YAML files. Sorted for deterministic output.
    Skips hydra_config directories (composition fragments).
    """
    recipes = []
    for root, dirs, files in os.walk(RECIPES_DIR):
        dirs[:] = [d for d in sorted(dirs) if d != "hydra_config"]
        for f in sorted(files):
            if f.endswith((".yaml", ".yml")):
                r = parse_recipe(Path(root) / f)
                if r:
                    recipes.append(r)
    return recipes


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================


def _make_table(recipes: List[Recipe]) -> List[str]:
    cols = [
        ("Model", lambda r: r.model or "-"),
        ("Framework", lambda r: r.framework or "-"),
        ("Technique", lambda r: r.technique or "-"),
        ("Adapter", lambda r: r.adapter or "-"),
        ("Seq Length", lambda r: f"{r.seq_len:,}" if r.seq_len else "-"),
        ("Nodes", lambda r: str(r.nodes) if r.nodes else "-"),
        ("Instance Type", lambda r: r.instance_type or "-"),
        ("Version", lambda r: r.version or "-"),
        ("Recipe", lambda r: f"[{Path(r.recipe_path).name}](../{r.recipe_path})"),
        ("Launcher Script", lambda r: f"[{Path(r.script_path).name}](../{r.script_path})" if r.script_path else "-"),
    ]
    if not recipes:
        return []
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("-----" for _ in cols) + "|"
    rows = ["| " + " | ".join(fn(r) for _, fn in cols) + " |" for r in recipes]
    return [header, sep] + rows


def _generate_section(title, recipes, filter_fn, group_by, intro):
    matched = [r for r in recipes if filter_fn(r)]
    if not matched:
        return []
    lines = [f"## {title}", ""]
    if intro:
        lines.extend([intro, ""])
    if group_by == "family":
        for fam in sorted(set(r.family for r in matched)):
            fam_recipes = sorted([r for r in matched if r.family == fam], key=lambda r: (r.model, r.technique))
            lines.extend([f"### {fam} Models", ""])
            lines.extend(_make_table(fam_recipes))
            lines.append("")
    else:
        lines.extend(_make_table(sorted(matched, key=lambda r: (r.model, r.technique))))
        lines.append("")
    return lines


def generate_markdown(recipes: List[Recipe]) -> str:
    lines = [
        "# Amazon SageMaker HyperPod Recipes",
        "",
        "This document provides a comprehensive catalog of all available recipes organized by category.",
        "",
        "## Table of Contents",
        "",
    ]
    for title, *_ in DOC_SECTIONS:
        anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
        lines.append(f"- [{title}](#{anchor})")
    lines.append("")
    for title, filter_fn, group_by, intro in DOC_SECTIONS:
        lines.extend(_generate_section(title, recipes, filter_fn, group_by, intro))
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================


def run_generation():
    print(f"Scanning recipes in: {RECIPES_DIR}")
    recipes = scan_recipes()
    print(f"Found {len(recipes)} recipes")
    content = generate_markdown(recipes)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(content, encoding="utf-8")
    print(f"Generated: {OUTPUT_FILE}")


def check_generation(show_diff: bool = False) -> bool:
    print("Generating expected content...")
    recipes = scan_recipes()
    expected = generate_markdown(recipes)
    if not OUTPUT_FILE.exists():
        if show_diff:
            print(f"MISSING: {OUTPUT_FILE}")
        return False
    actual = OUTPUT_FILE.read_text(encoding="utf-8")
    if actual == expected:
        return True
    if show_diff:
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile="a/docs/RECIPES.md",
            tofile="b/docs/RECIPES.md",
            lineterm="",
        )
        print("".join(diff))
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate RECIPES.md documentation from recipe YAML files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  %(prog)s          Generate RECIPES.md\n  %(prog)s --check  Check if up-to-date",
    )
    parser.add_argument("--check", action="store_true", help="Check if generated content matches disk")
    parser.add_argument("--diff", action="store_true", help="Show unified diff (requires --check)")
    args = parser.parse_args()

    if args.check:
        if check_generation(show_diff=args.diff):
            print("✓ RECIPES.md is up-to-date.")
            sys.exit(0)
        print(f"\n{'=' * 70}\nERROR: RECIPES.md is out of sync\n{'=' * 70}\nTo fix: python {sys.argv[0]}\n")
        sys.exit(1)
    run_generation()


if __name__ == "__main__":
    main()
