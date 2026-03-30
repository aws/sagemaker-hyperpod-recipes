from auto_configurator.config_optimizer.base_optimizer import BaseOptimizer
from auto_configurator.utils.util import MAX_STEPS


class LlmftOptimizer(BaseOptimizer):
    def _tunable_params(self) -> list[str]:
        return [
            "train_batch_size",
            "sharding_strategy",
            "gradient_checkpointing",
            "cpu_offload",
        ]

    def _generate_parameter_ranges(self) -> dict:
        """Set up ranges for LLMFT tunable parameters"""
        param_ranges = {}
        self.__set_sharding_strategy(param_ranges)
        self.__set_gradient_checkpointing(param_ranges)
        self.__set_cpu_offload(param_ranges)
        return param_ranges

    def get_recipe_overrides(self, candidate_params) -> list[str]:
        recipe_overrides = []
        for key, value in candidate_params.items():
            match key:
                case "train_batch_size":
                    recipe_overrides.append(f"++recipes.training_config.training_args.train_batch_size={value}")

                    # limit max steps
                    recipe_overrides.append(f"++recipes.training_config.datasets.train_data.limit={value * MAX_STEPS}")
                case "micro_train_batch_size":
                    recipe_overrides.append(f"++recipes.training_config.training_args.micro_train_batch_size={value}")
                case "sharding_strategy":
                    recipe_overrides.append(
                        f"++recipes.training_config.training_args.strategy.fsdp_config.sharding_strategy={value}"
                    )
                case "gradient_checkpointing":
                    recipe_overrides.append(f"++recipes.training_config.training_args.gradient_checkpointing={value}")
                case "cpu_offload":
                    recipe_overrides.append(
                        f"++recipes.training_config.training_args.strategy.fsdp_config.cpu_offload={value}"
                    )
                case "max_len":
                    recipe_overrides.append(f"++recipes.training_config.training_args.max_len={value}")
                case _:
                    continue

        return recipe_overrides

    def __set_sharding_strategy(self, params):
        """Set list of FSDP sharding strategies to explore

        Tests FULL_SHARD and HYBRID_SHARD (NO_SHARD excluded due to memory constraints and FSDP state issues)
        """
        cfg_value = getattr(self._autotune_cfg, "sharding_strategy", "auto")

        if cfg_value == "auto":
            params["sharding_strategy"] = ["FULL_SHARD", "HYBRID_SHARD"]
        elif not isinstance(cfg_value, list):
            params["sharding_strategy"] = [cfg_value]
        else:
            params["sharding_strategy"] = cfg_value

    def __set_gradient_checkpointing(self, params):
        """Set gradient checkpointing options

        Heuristic:
        - Models > 7B: Always enable (memory constrained)
        - Models ≤ 7B: Test both True and False
        """
        cfg_value = getattr(self._autotune_cfg, "gradient_checkpointing", "auto")

        if cfg_value == "auto":
            # For models > 7B, always enable. For smaller, test both
            if self.num_params > 7e9:
                params["gradient_checkpointing"] = [True]
            else:
                params["gradient_checkpointing"] = [True, False]
        elif not isinstance(cfg_value, list):
            params["gradient_checkpointing"] = [cfg_value]
        else:
            params["gradient_checkpointing"] = cfg_value

    def __set_cpu_offload(self, params):
        """Set CPU offload options

        Heuristic based on model size relative to GPU memory:
        - If model params alone would use >50% of GPU memory per GPU: Test both True/False
        - Otherwise: Default to False (sufficient GPU memory)

        Calculation: params_per_gpu = (num_params * 2 bytes) / num_gpus / 1e9 GB
        """
        cfg_value = getattr(self._autotune_cfg, "cpu_offload", "auto")

        if cfg_value == "auto":
            # Estimate parameter memory per GPU (bf16 = 2 bytes per param)
            num_gpus = self.cfg.trainer.num_nodes * self.cfg.trainer.devices
            params_memory_per_gpu = (self.num_params * 2) / num_gpus / 1e9

            # If params alone use >50% of GPU memory, test with offload
            if params_memory_per_gpu > 0.5 * self._gpu_memory:
                params["cpu_offload"] = [True, False]
            else:
                params["cpu_offload"] = [False]
        elif not isinstance(cfg_value, list):
            params["cpu_offload"] = [cfg_value]
        else:
            params["cpu_offload"] = cfg_value

    def _is_valid_candidate(self, candidate) -> bool:
        """Validate if a recipe configuration is valid (Nemo-equivalent logic)"""
        valid_strategies = ["FULL_SHARD", "HYBRID_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]

        # Check sharding strategy is valid
        if candidate.get("sharding_strategy") not in valid_strategies:
            return False

        # Check batch size is at least 2 (minimum for stable training)
        train_batch_size = candidate.get("train_batch_size", 0)

        # Check gradient accumulation is valid (>= 1)
        num_gpus = self.cfg.trainer.num_nodes * self.cfg.trainer.devices

        micro_train_batch_size = self.cfg.training_config.training_args.micro_train_batch_size
        if train_batch_size < 2 or train_batch_size < micro_train_batch_size * num_gpus:
            return False

        # Check train_batch_size doesn't exceed validation dataset size
        val_limit = getattr(self.cfg.training_config.datasets.val_data, "limit", None)
        if val_limit and train_batch_size > val_limit:
            return False

        # CPU offload requires gradient checkpointing (equivalent to Nemo's offload_activations check)
        if candidate.get("cpu_offload") and not candidate.get("gradient_checkpointing"):
            return False

        return True

    def tune_candidate(self, candidate: dict, error_code) -> tuple[dict, bool]:
        """Adjust LLMFT candidate based on error code (Nemo-equivalent logic)"""
        from auto_configurator.evaluation.base_evaluator import ErrorCode

        adjusted = candidate.copy()

        if error_code in [ErrorCode.OOM, ErrorCode.CACHE_FLUSH]:
            # Try increasing sharding first
            if adjusted["sharding_strategy"] == "NO_SHARD":
                adjusted["sharding_strategy"] = "HYBRID_SHARD"
                return (adjusted, True)
            elif adjusted["sharding_strategy"] == "HYBRID_SHARD":
                adjusted["sharding_strategy"] = "FULL_SHARD"
                return (adjusted, True)
            elif adjusted["train_batch_size"] > 2:
                adjusted["train_batch_size"] //= 2
                if self._is_valid_candidate(adjusted):
                    return (adjusted, True)

        elif error_code == ErrorCode.LOW_MEMORY:
            adjusted["train_batch_size"] *= 2
            adjusted["sharding_strategy"] = "FULL_SHARD"
            if self._is_valid_candidate(adjusted):
                return (adjusted, True)

        return (candidate, False)

    def _estimate_memory_per_gpu(self, train_batch_size: int, candidate_params: dict) -> tuple[float, float]:
        """Estimate memory usage per GPU for LLMFT

        Args:
            train_batch_size: Micro batch size per GPU (for memory estimation)
            candidate_params: Recipe parameters including sharding_strategy

        FSDP sharding strategies (maps to ZeRO stages):
        - FULL_SHARD (ZeRO-3): Shards optimizer + gradients + parameters
        - HYBRID_SHARD (ZeRO-2): Shards optimizer + gradients, replicates parameters
        - SHARD_GRAD_OP (ZeRO-2): Shards optimizer + gradients, replicates parameters
        - NO_SHARD (DDP): Replicates everything
        """
        from auto_configurator.utils.util import BYTE_TO_GB

        model_cfg = self._load_model_params()
        num_bytes_per_param = 2  # bf16
        num_gpus = self.cfg.trainer.num_nodes * self.cfg.trainer.devices
        sharding_strategy = candidate_params.get("sharding_strategy")

        # Use actual sequence length from training config, not model's max capability
        seq_length = candidate_params["max_len"]

        # FIXME: these values numbers are empirically derived approximations for transformer memory usage, need to be reevaluated
        # Activation memory (replicated per GPU)
        # Non-MoE: base attention (18) + MLP activations (4 * intermediate/hidden ratio)
        # MoE: reduced base (15) + per-expert cost (3 + 4 * ratio) * num_experts_per_tok
        activation_factor = (
            18 + 4 * model_cfg.intermediate_size / model_cfg.hidden_width
            if not model_cfg.moe
            else 15 + model_cfg.num_experts_per_tok * (3 + 4 * model_cfg.intermediate_size / model_cfg.hidden_width)  # type: ignore
        )

        activations_memory = (
            model_cfg.num_layers
            * seq_length  # Use actual training sequence length
            * train_batch_size  # train_batch_size parameter is micro batch per GPU
            * model_cfg.hidden_width
            * activation_factor
            * BYTE_TO_GB
        )

        """
        Without gradient checkpointing:
        - During forward pass, all intermediate activations are stored in memory
        - During backward pass, these stored activations are used to compute gradients
        - Memory usage: HIGH (stores all activations)

        With gradient checkpointing:
        - During forward pass, only checkpoint certain activations (e.g., at layer boundaries)
        - During backward pass, recompute the missing activations on-the-fly from checkpoints
        - Memory usage: LOW (only stores checkpoints, recomputes the rest)
        - Trade-off: ~33% slower (due to recomputation) but uses ~90% less activation memory
        """
        if candidate_params.get("gradient_checkpointing"):
            activations_memory /= 10

        # Adjust what gets sharded based on strategy
        if sharding_strategy == "FULL_SHARD":
            # ZeRO-3: Shard everything
            parameters_memory = num_bytes_per_param * self.num_params / num_gpus * BYTE_TO_GB
            gradients_memory = 2 * num_bytes_per_param * self.num_params / num_gpus * BYTE_TO_GB
            optimizer_memory = 3 * 4 * self.num_params / num_gpus * BYTE_TO_GB
        elif sharding_strategy in ["HYBRID_SHARD", "SHARD_GRAD_OP"]:
            # ZeRO-2: Shard optimizer + gradients, replicate parameters
            parameters_memory = num_bytes_per_param * self.num_params * BYTE_TO_GB
            gradients_memory = 2 * num_bytes_per_param * self.num_params / num_gpus * BYTE_TO_GB
            optimizer_memory = 3 * 4 * self.num_params / num_gpus * BYTE_TO_GB
        else:  # NO_SHARD
            # DDP: Replicate everything
            parameters_memory = num_bytes_per_param * self.num_params * BYTE_TO_GB
            gradients_memory = 2 * num_bytes_per_param * self.num_params * BYTE_TO_GB
            optimizer_memory = 3 * 4 * self.num_params * BYTE_TO_GB

        # Total allocated memory
        max_memory_alloc = activations_memory + parameters_memory + gradients_memory + optimizer_memory

        # Apply CUDA memory allocator overhead based on gradient checkpointing from candidate params
        gradient_checkpointing = candidate_params.get("gradient_checkpointing", True)
        lower_bound = max_memory_alloc * (1.3 if gradient_checkpointing else 1.1)
        upper_bound = max_memory_alloc * (1.7 if gradient_checkpointing else 1.3)

        return (lower_bound, upper_bound)
