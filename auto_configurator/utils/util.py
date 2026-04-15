"""Model architecture configurations for supported models"""

import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path

import boto3
from pydantic import BaseModel
from transformers import AutoConfig

MAX_STEPS = 15
BYTE_TO_GB = 1e-9
AUTO = "auto"


class AutoConfiguratorLogger:
    def __init__(self):
        # Suppress verbose logs before any imports that use boto3
        logging.getLogger("botocore").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

        _logger = logging.getLogger("AutoConfigurator")
        _logger.setLevel(logging.DEBUG)

        self.logger: logging.Logger = _logger

    def get_logger(self):
        return self.logger


class OptimizerType(Enum):
    LLMFT = "llmft"
    VERL = "verl"


def get_optimizer_type(recipe):
    _recipe = recipe.lower()

    if OptimizerType.LLMFT.value in _recipe:
        return OptimizerType.LLMFT
    elif OptimizerType.VERL.value in _recipe:
        return OptimizerType.VERL
    else:
        raise ValueError(f"Unable to determine optimizer type for recipe: {recipe}")


class ModelParams(BaseModel):
    vocab_size: int
    hidden_width: int
    num_heads: int
    num_key_value_heads: int
    num_layers: int
    intermediate_size: int
    moe: bool
    max_context_width: int
    num_local_experts: int | None = None
    num_experts_per_tok: int | None = None


# Model architecture parameters sourced from model configs per model in fsx
# These are hardcoded as a baseline implementation to avoid:
# - Network dependencies (HuggingFace Hub API calls)
# - Authentication complexity (HF tokens for gated models)
# - Field name mapping across different architectures (hidden_size vs hidden_width, etc.)
# - Runtime latency during optimizer initialization
#
# Locate the model's config.json in FSx or download from HuggingFace
#  - vocab_size: Size of the vocabulary
#  - hidden_width: Hidden dimension size (may be "hidden_size" in config)
#  - num_heads: Number of attention heads (may be "num_attention_heads")
#  - num_key_value_heads: Number of KV heads for GQA (may be "num_key_value_heads")
#  - num_layers: Number of transformer layers (may be "num_hidden_layers")
#  - intermediate_size: FFN intermediate dimension
#  - max_context_width: Maximum sequence length (may be "max_position_embeddings")
#  - moe: Whether model uses Mixture of Experts (infer from architecture)
#  - num_local_experts: Number of expert networks (MoE models only)
#  - num_experts_per_tok: Experts activated per token (MoE models only)
# These are gated models which require hf token
MODEL_ARCHITECTURES: dict[str, ModelParams] = {
    "meta-llama/Llama-3.1-8B-Instruct": ModelParams(
        vocab_size=128256,
        hidden_width=4096,
        num_heads=32,
        num_key_value_heads=8,
        num_layers=32,
        intermediate_size=14336,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.1-8B": ModelParams(
        vocab_size=128256,
        hidden_width=4096,
        num_heads=32,
        num_key_value_heads=8,
        num_layers=32,
        intermediate_size=14336,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct": ModelParams(
        vocab_size=128256,
        hidden_width=4096,
        num_heads=32,
        num_key_value_heads=8,
        num_layers=40,
        intermediate_size=14336,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.2-11B-Vision": ModelParams(
        vocab_size=128256,
        hidden_width=4096,
        num_heads=32,
        num_key_value_heads=8,
        num_layers=40,
        intermediate_size=14336,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.2-1B-Instruct": ModelParams(
        vocab_size=128256,
        hidden_width=2048,
        num_heads=32,
        num_key_value_heads=8,
        num_layers=16,
        intermediate_size=8192,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelParams(
        vocab_size=128256,
        hidden_width=3072,
        num_heads=24,
        num_key_value_heads=8,
        num_layers=28,
        intermediate_size=8192,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.2-90B-Vision": ModelParams(
        vocab_size=128256,
        hidden_width=8192,
        num_heads=64,
        num_key_value_heads=8,
        num_layers=100,
        intermediate_size=28672,
        max_context_width=131072,
        moe=False,
    ),
    "meta-llama/Llama-3.3-70B-Instruct": ModelParams(
        vocab_size=128256,
        hidden_width=8192,
        num_heads=64,
        num_key_value_heads=8,
        num_layers=80,
        intermediate_size=28672,
        max_context_width=131072,
        moe=False,
    ),
    "openai/gpt-oss-120b-bf16": ModelParams(
        vocab_size=201088,
        hidden_width=2880,
        num_heads=64,
        num_key_value_heads=8,
        num_layers=36,
        intermediate_size=2880,
        max_context_width=131072,
        moe=True,
        num_local_experts=128,
        num_experts_per_tok=4,
    ),
    "openai/gpt-oss-20b-bf16": ModelParams(
        vocab_size=201088,
        hidden_width=2880,
        num_heads=64,
        num_key_value_heads=8,
        num_layers=24,
        intermediate_size=2880,
        max_context_width=131072,
        moe=True,
        num_local_experts=32,
        num_experts_per_tok=4,
    ),
    "openai/gpt-oss-20b": ModelParams(
        vocab_size=201088,
        hidden_width=2880,
        num_heads=64,
        num_key_value_heads=8,
        num_layers=24,
        intermediate_size=2880,
        max_context_width=131072,
        moe=True,
        num_local_experts=32,
        num_experts_per_tok=4,
    ),
}


_model_params_cache: dict[str, ModelParams] = {}


def load_model_params(model_path: str, hf_token=None) -> ModelParams:
    """Load model params from HuggingFace AutoConfig, falling back to static map."""
    if model_path in _model_params_cache:
        return _model_params_cache[model_path].model_copy()

    try:
        logger = logging.getLogger("AutoConfigurator")
        logger.info(f"Fetching model config HuggingFace: {model_path}")
        cfg = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, token=hf_token if hf_token else os.environ.get("HF_TOKEN", None)
        )

        # Multimodal models (e.g. Llama Vision) nest params under text_config
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config

        num_local_experts = getattr(cfg, "num_local_experts", None)
        num_experts_per_tok = getattr(cfg, "num_experts_per_tok", None)
        is_moe = num_local_experts is not None and num_local_experts > 1

        params = ModelParams(
            vocab_size=cfg.vocab_size,
            hidden_width=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            num_layers=cfg.num_hidden_layers,
            intermediate_size=cfg.intermediate_size,
            max_context_width=cfg.max_position_embeddings,
            moe=is_moe,
            num_local_experts=num_local_experts if is_moe else None,
            num_experts_per_tok=num_experts_per_tok if is_moe else None,
        )
        _model_params_cache[model_path] = params
        return params.model_copy()
    except Exception as e:
        logger.warning(
            f"Failed to load model config from HuggingFace for {model_path}: {e}. Falling back to static map."
        )
        for key in MODEL_ARCHITECTURES:
            if key in model_path:
                return MODEL_ARCHITECTURES[key].model_copy()

        raise ValueError(
            f"Unknown model: {model_path}. Not in MODEL_ARCHITECTURES and failed to load from HuggingFace: {e}"
        )


def get_gpu_memory_gb(instance_type: str) -> float:
    """Get GPU memory in GB for an instance type

    Args:
        instance_type: EC2 or SageMaker instance type (e.g., 'p5.48xlarge' or 'ml.p5.48xlarge')

    Returns:
        GPU memory per GPU in GB
    """
    memory_gb, _ = get_gpu_info(instance_type)
    return memory_gb


def get_gpu_info(instance_type: str) -> tuple[float, int]:
    """Get GPU memory and count for an instance type.

    Args:
        instance_type: EC2 or SageMaker instance type (e.g., 'p5.48xlarge' or 'ml.p5.48xlarge')

    Returns:
        (gpu_memory_gb, gpu_count) tuple
    """
    ec2_type = instance_type.replace("ml.", "")

    ec2 = boto3.client("ec2", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    response = ec2.describe_instance_types(InstanceTypes=[ec2_type])
    gpu_info = response["InstanceTypes"][0]["GpuInfo"]
    memory_mib = gpu_info["Gpus"][0]["MemoryInfo"]["SizeInMiB"]
    gpu_count = gpu_info["TotalGpuMemoryInMiB"] // memory_mib

    return memory_mib / 1024, gpu_count


def prettify(obj):
    return json.dumps(obj, indent=2)


def copy_file(source_path: str, destination_path: str):
    """copy file to destination"""

    if source_path and Path(source_path).exists():
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(source_path), destination_path)
        return destination_path

    return ""


def get_sequence_length_range(auto_config):
    seq_lengths = auto_config.autotune_config.get("sequence_lengths", AUTO)

    if seq_lengths == AUTO:
        return [2**i for i in range(12, 18)]  # 4K to 128K: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    elif not isinstance(seq_lengths, list):
        return [seq_lengths]
    else:
        return seq_lengths


def format_params(params: dict):
    return tuple(sorted((k, str(v)) for k, v in params.items()))
