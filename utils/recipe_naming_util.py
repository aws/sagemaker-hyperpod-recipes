"""Naming utilities for recipe config files.

Aligns with hyperpod_recipes naming convention:
{repo}-{technique}-{model}-{instance}-{context}-{fft|lora}.yaml
"""

import re
from pathlib import Path

_KNOWN_INSTANCE_FAMILIES = r"(?:p4de|p4d|p5en|p5e|p5|g6e|g6|g5|trn2|trn1|inf2|inf1)"
_INSTANCE_PATTERN = re.compile(rf"[-_]{_KNOWN_INSTANCE_FAMILIES}(?:[-_]\d+){{0,2}}")
_SEQ_PATTERN = re.compile(r"[-_]seq\d+k?")


def instance_short(instance_type: str) -> str:
    """ml.p4de.24xlarge → p4de, ml.g5.12xlarge → g5-12, ml.g5.48xlarge → g5-48"""
    parts = instance_type.replace("ml.", "").split(".")
    family = parts[0]
    if len(parts) == 2:
        size = parts[1].replace("xlarge", "")
        if size:
            return f"{family}-{size}"
    return family


def seq_short(seq_len: int) -> str:
    """4096 → seq4k, 1536 → seq1536"""
    if seq_len >= 1024 and seq_len % 1024 == 0:
        return f"seq{seq_len // 1024}k"
    return f"seq{seq_len}"


def config_name(recipe: str, instance_type: str, seq_len: int) -> str:
    """Build config name: {stem_without_suffix}-{instance}-{seq}-{suffix}.

    Strips any existing instance type or sequence length patterns from the
    recipe stem before inserting the new ones.

    Examples:
        verl-sft-qwen-3-dot-5-4b-fft.yaml + ml.p4de.24xlarge + 4096
        → verl-sft-qwen-3-dot-5-4b-p4de-seq4k-fft
    """
    stem = Path(recipe).stem
    inst = instance_short(instance_type)
    seq = seq_short(seq_len)

    # Extract fft/lora suffix
    suffix = ""
    for sfx in ("fft", "lora"):
        for sep in ("-", "_"):
            if stem.endswith(f"{sep}{sfx}"):
                suffix = sfx
                stem = stem[: -(len(sfx) + 1)]
                break
        if suffix:
            break

    # Strip existing instance/seq patterns
    stem = _INSTANCE_PATTERN.sub("", stem)
    stem = _SEQ_PATTERN.sub("", stem)

    parts = [stem, inst, seq]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)
