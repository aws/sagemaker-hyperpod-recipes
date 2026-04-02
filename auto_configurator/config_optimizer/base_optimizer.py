import logging
from abc import ABC, abstractmethod
from itertools import product

from auto_configurator.utils.util import (
    MODEL_ARCHITECTURES,
    ModelParams,
    get_gpu_memory_gb,
    prettify,
)


class BaseOptimizer(ABC):
    """Abstract base class for framework-specific config optimizer"""

    def __init__(self, autotune_config, recipe_cfg, instance_type):
        self._autotune_cfg = autotune_config
        self._gpu_memory: float = get_gpu_memory_gb(instance_type)
        self.cfg = recipe_cfg

        self.logger = logging.getLogger(self.__class__.__name__)

        self._is_valid_autoconfig()

        self.num_params = self._compute_num_params()

    @abstractmethod
    def get_recipe_overrides(self, candidate_params) -> list[str]:
        """
        Returns a list of hydra overrides with the adjusted parameters
        """

    @abstractmethod
    def tune_candidate(self, candidate: dict, error_code, tried_configs: set | None = None) -> tuple[dict, bool]:
        """
        Adjust candidate based on error code.
        Returns (adjusted_candidate, should_retry)
        """

    @abstractmethod
    def _tunable_params(self) -> list[str]:
        """Return list of tunable parameter names for framework"""

    @abstractmethod
    def _generate_parameter_ranges(self) -> dict:
        """Set up ranges for tunable parameters"""

    @abstractmethod
    def _estimate_memory_per_gpu(self, train_batch_size: int, candidate_params: dict) -> tuple[float, float]:
        """Estimate memory usage per GPU for given batch size and candidate_params

        Returns:
            (lower_bound, upper_bound) memory in GB
        """

    @abstractmethod
    def _is_valid_candidate(self, candidate) -> bool:
        """Validate if a recipe configuration is valid for the model"""

    def _load_model_params(self) -> ModelParams:
        """Load model architecture params from MODEL_ARCHITECTURES mapping"""
        model_path = self.cfg.training_config.model_config.model_name_or_path

        for key in MODEL_ARCHITECTURES:
            if key in model_path:
                return MODEL_ARCHITECTURES[key].model_copy()

        raise ValueError(f"Unknown model: {model_path}. Add to MODEL_ARCHITECTURES in constants.py")

    def _compute_num_params(self) -> int:
        """Compute number of parameters from model config"""
        p = self._load_model_params()

        num_heads = p.num_key_value_heads
        vocab_size = p.vocab_size
        hidden_width = p.hidden_width
        intermediate_size = p.intermediate_size
        num_layers = p.num_layers
        moe = p.moe

        embedding_params = vocab_size * hidden_width

        q_proj_params = hidden_width * hidden_width
        k_proj_params = hidden_width * (hidden_width // num_heads)
        v_proj_params = hidden_width * (hidden_width // num_heads)
        o_proj_params = hidden_width * hidden_width
        attn_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params

        mlp_params = 3 * hidden_width * intermediate_size
        if moe:
            mlp_params *= 8

        layer_norm_params = 2 * hidden_width
        params_per_layer = attn_params + mlp_params + layer_norm_params
        total_layer_params = params_per_layer * num_layers
        final_layer_norm_params = vocab_size * hidden_width

        total_params = embedding_params + total_layer_params + final_layer_norm_params + embedding_params
        return total_params

    def _is_valid_autoconfig(self):
        """Ensure only tunable params are set to auto or range"""
        tunable_params = self._tunable_params()
        for key in self._autotune_cfg.keys():
            if key not in tunable_params:
                raise KeyError(f"{key} is not a tunable parameter")

    def generate_candidate_configurations(self, max_len: int = 4096):
        """
        Create set of concrete recipes from range of parameters with help of memory_estimator
        1. Dynamically iterates over tunable params from _set_tunable_params()
        2. Uses itertools.product to generate all combinations
        3. Skips train_batch_size since it's computed via find_batch_size()
        4. Works for any subclass - NeMo with its params, LLMFT with its params, etc.
        """

        param_ranges = self._generate_parameter_ranges()

        tunable_params = self._tunable_params()

        # Get all parameter values to iterate over
        param_values = []
        for param in tunable_params:
            if param == "micro_train_batch_size":
                continue  # batch size is computed, not iterated
            values = param_ranges.get(param, [])
            if not isinstance(values, list):
                values = [values]
            param_values.append((param, values))

        candidates = []
        # Generate all combinations of tunable parameters
        for combo in product(*[vals for _, vals in param_values]):
            candidate_params = dict(zip([p for p, _ in param_values], combo))
            candidate_params["max_len"] = max_len
            train_batch_size = self._find_batch_size(candidate_params)
            if train_batch_size == 0:
                continue
            candidate_params["micro_train_batch_size"] = train_batch_size

            # Validate after batch size is set
            if not self._is_valid_candidate(candidate_params):
                self.logger.info(f"Skipping candidate - failed validation: {candidate_params}")
                continue

            candidates.append(candidate_params)

        self.logger.info(f"Generated candidates for max_len {max_len}: {prettify(candidates)}")
        return candidates

    def _find_batch_size(self, candidate: dict) -> int:
        """Find largest micro batch size that fits in GPU memory.

        Starts from the max valid value (train_batch_size // num_gpus) and halves
        until estimated memory fits. Returns 1 for borderline fits, 0 if impossible.
        """
        num_gpus = self.cfg.trainer.num_nodes * self.cfg.trainer.devices
        train_batch_size = candidate.get("train_batch_size", self.cfg.training_config.training_args.train_batch_size)
        micro_train_batch_size = max(1, train_batch_size // num_gpus)

        while micro_train_batch_size > 1:
            _, upper_bound = self._estimate_memory_per_gpu(micro_train_batch_size, candidate)
            if upper_bound < self._gpu_memory:
                return micro_train_batch_size
            micro_train_batch_size //= 2

        _, estimated_mem = self._estimate_memory_per_gpu(1, candidate)

        if estimated_mem < 5 / 4 * self._gpu_memory:
            return 1
        return 0
