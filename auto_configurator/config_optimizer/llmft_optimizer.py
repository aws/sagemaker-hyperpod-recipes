from auto_configurator.config_optimizer.base_optimizer import BaseOptimizer
from auto_configurator.evaluation.base_evaluator import ErrorCode
from auto_configurator.utils.util import BYTE_TO_GB, MAX_STEPS, format_params


class LlmftOptimizer(BaseOptimizer):
    SHARDING_INCREASE = {"NO_SHARD": "HYBRID_SHARD", "HYBRID_SHARD": "FULL_SHARD", "FULL_SHARD": "FULL_SHARD"}
    SHARDING_DECREASE = {"FULL_SHARD": "HYBRID_SHARD", "HYBRID_SHARD": "NO_SHARD", "NO_SHARD": "NO_SHARD"}

    def _tunable_params(self) -> list[str]:
        return [
            "micro_train_batch_size",
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

        # limit max steps and ensure validation dataset is large enough
        train_batch_size = candidate_params.get(
            "train_batch_size", self.cfg.training_config.training_args.train_batch_size
        )
        recipe_overrides.append(f"++recipes.training_config.datasets.train_data.limit={train_batch_size * MAX_STEPS}")
        recipe_overrides.append(f"++recipes.training_config.datasets.val_data.limit={train_batch_size * 2}")

        # Override GPU count: recipes.trainer.devices is read by the recipe config,
        # trainer.devices is read by the launcher for the Helm template k8s resource requests.
        recipe_overrides.append(f"++recipes.trainer.devices={self._gpu_count}")
        recipe_overrides.append(f"++trainer.devices={self._gpu_count}")

        return recipe_overrides

    def __set_sharding_strategy(self, params):
        """Set list of FSDP sharding strategies to explore

        Tests FULL_SHARD and HYBRID_SHARD
        NO_SHARD excluded due to memory constraints and FSDP state issues, e.g:
            RuntimeError: Cannot writeback when the parameter shape changes
            Expects torch.Size([525336576]) but got torch.Size([128256, 4096])
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

        Always generates both True and False for auto mode. Candidates where
        GC=False doesn't fit in memory are filtered out by _find_batch_size
        returning 0 in generate_candidate_configurations.
        """
        cfg_value = getattr(self._autotune_cfg, "gradient_checkpointing", "auto")

        if cfg_value == "auto":
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
            num_gpus = self.cfg.trainer.num_nodes * self._gpu_count
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
        """Validate if a recipe configuration is valid"""
        valid_strategies = ["FULL_SHARD", "HYBRID_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]

        # Check sharding strategy is valid
        if candidate.get("sharding_strategy") not in valid_strategies:
            self.logger.debug(f"Invalid sharding strategy: {candidate.get('sharding_strategy')}")
            return False

        # Check gradient accumulation is valid (>= 1)
        num_gpus = self.cfg.trainer.num_nodes * self._gpu_count

        train_batch_size = candidate.get("train_batch_size", self.cfg.training_config.training_args.train_batch_size)
        micro_train_batch_size = candidate.get("micro_train_batch_size", 0)

        if micro_train_batch_size < 1:
            self.logger.debug(f"Invalid: micro_train_batch_size ({micro_train_batch_size}) < 1")
            return False

        if train_batch_size < micro_train_batch_size * num_gpus:
            self.logger.debug(
                f"Invalid: train_batch_size ({train_batch_size}) < micro_train_batch_size ({micro_train_batch_size}) * num_gpus ({num_gpus})"
            )
            return False

        # CPU offload requires gradient checkpointing
        if candidate.get("cpu_offload") and not candidate.get("gradient_checkpointing"):
            self.logger.debug("CPU offload requires gradient checkpointing")
            return False

        return True

    def tune_candidate(self, candidate: dict, error_code, tried_configs: set | None = None) -> tuple[dict, bool]:
        """Adjust LLMFT candidate based on error code"""
        tried_configs = tried_configs if tried_configs is not None else set()

        sharding_strategy = candidate["sharding_strategy"]

        # Priority order by throughput impact (least → most negative)
        if error_code in [ErrorCode.OOM, ErrorCode.CACHE_FLUSH]:
            # Try increasing sharding first
            adjusted = {**candidate, "sharding_strategy": self.__get_sharding_progression(sharding_strategy)}
            if format_params(adjusted) not in tried_configs:
                return (adjusted, True)

            # Reduce batch size
            adjusted = {**candidate, "micro_train_batch_size": candidate["micro_train_batch_size"] // 2}
            if format_params(adjusted) not in tried_configs and self._is_valid_candidate(adjusted):
                return (adjusted, True)

        elif error_code == ErrorCode.LOW_MEMORY:
            # Try increase batch size
            adjusted = {**candidate, "micro_train_batch_size": candidate["micro_train_batch_size"] * 2}
            if format_params(adjusted) not in tried_configs and self._is_valid_candidate(adjusted):
                return (adjusted, True)

            # Try decreasing sharding (more memory available)
            adjusted = {
                **candidate,
                "sharding_strategy": self.__get_sharding_progression(sharding_strategy, increase=False),
            }
            if format_params(adjusted) not in tried_configs:
                return (adjusted, True)

        return (candidate, False)

    def _estimate_memory_per_gpu(self, train_batch_size: int, candidate_params: dict) -> tuple[float, float]:
        """Estimate memory usage per GPU for LLMFT

        Memory formulas based on: https://arxiv.org/pdf/1910.02054 & https://arxiv.org/pdf/2411.06465
        - ZeRO (Rajbhandari et al., 2020, arXiv:1910.02054): Memory partitioning across ZeRO stages
        - Fujii et al. (2024, arXiv:2411.06465): Precise per-layer activation formulas for
          decoder-only transformers with GQA, gated FFN (SwiGLU), RMSNorm, and FlashAttention.
          Applicable to all supported architectures (Llama, Qwen, DeepSeek, GPT-OSS).

        Model states per parameter (Fujii et al. Eq. 4):
        - Weights: 2 bytes (bf16)
        - Gradients: 4 bytes (fp32 accumulation for numerical stability)
        - Optimizer (Adam): 12 bytes (master weight + momentum + variance, all fp32)
        - Total: 18 bytes per parameter

        Activation memory per layer (Fujii et al. Eq. 12):
        - sbh * (12 + 4*k/a + 8*h_ffn/h)
        - Assumes FlashAttention (no QK^T/softmax storage), GQA, gated FFN, RMSNorm

        FSDP sharding strategies (maps to ZeRO stages):
        - FULL_SHARD (ZeRO-3): Shards optimizer + gradients + parameters
        - HYBRID_SHARD: Shards optimizer + gradients within a node, replicates parameters
        - SHARD_GRAD_OP (ZeRO-2): Shards optimizer + gradients across all ranks, replicates parameters
        - NO_SHARD (DDP): Replicates everything
        """

        model_cfg = self._load_model_params()
        num_gpus = self.cfg.trainer.num_nodes * self._gpu_count
        sharding_strategy = candidate_params.get("sharding_strategy")
        seq_length = candidate_params["max_len"]

        # --- Activation memory per layer (Fujii et al. Eq. 12) ---
        # For decoder-only transformers with FlashAttention, GQA, gated FFN, RMSNorm:
        #   Attention: 6*sbh + 4*sbh*(k/a)  [input, Q, attn_out + K,V scaled by GQA ratio]
        #   Gated FFN (SwiGLU): 2*sb*(h + 4*h_ffn)  [input + up/gate/activation/down]
        #   RMSNorm: 4*sbh  [2 norms per layer]
        #   Total factor: 12 + 4*(k/a) + 8*(h_ffn/h)
        k_over_a = model_cfg.num_key_value_heads / model_cfg.num_heads
        h_ffn_over_h = model_cfg.intermediate_size / model_cfg.hidden_width

        if not model_cfg.moe:
            activation_factor = 12 + 4 * k_over_a + 8 * h_ffn_over_h
        else:
            # MoE: attention unchanged, FFN cost scaled by active experts per token
            activation_factor = 12 + 4 * k_over_a + model_cfg.num_experts_per_tok * 8 * h_ffn_over_h

        activations_memory = (
            model_cfg.num_layers
            * seq_length
            * train_batch_size
            * model_cfg.hidden_width
            * activation_factor
            * BYTE_TO_GB
        )

        # With gradient checkpointing (per-layer via apply_activation_checkpointing):
        # Only layer-boundary inputs are stored; during backward, one layer's activations
        # are recomputed at a time. Memory reduces from L layers to ~1 layer's worth.
        if candidate_params.get("gradient_checkpointing"):
            activations_memory /= model_cfg.num_layers

        # --- Model states memory (ZeRO + Fujii et al.) ---
        bytes_weight = 2  # bf16 parameters
        bytes_gradient = 4  # fp32 gradient accumulation (Fujii et al. §2.2)
        bytes_optimizer = 12  # fp32 master weight(4) + momentum(4) + variance(4)

        if sharding_strategy == "FULL_SHARD":
            # ZeRO-3: shard parameters, gradients, and optimizer states
            parameters_memory = bytes_weight * self.num_params / num_gpus * BYTE_TO_GB
            gradients_memory = bytes_gradient * self.num_params / num_gpus * BYTE_TO_GB
            optimizer_memory = bytes_optimizer * self.num_params / num_gpus * BYTE_TO_GB
        elif sharding_strategy == "HYBRID_SHARD":
            # HYBRID_SHARD: shards optimizer + gradients within a node, replicates parameters
            gpus_per_node = self._gpu_count
            parameters_memory = bytes_weight * self.num_params * BYTE_TO_GB
            gradients_memory = bytes_gradient * self.num_params / gpus_per_node * BYTE_TO_GB
            optimizer_memory = bytes_optimizer * self.num_params / gpus_per_node * BYTE_TO_GB
        elif sharding_strategy == "SHARD_GRAD_OP":
            # SHARD_GRAD_OP (ZeRO-2): shards optimizer + gradients across all ranks
            parameters_memory = bytes_weight * self.num_params * BYTE_TO_GB
            gradients_memory = bytes_gradient * self.num_params / num_gpus * BYTE_TO_GB
            optimizer_memory = bytes_optimizer * self.num_params / num_gpus * BYTE_TO_GB
        else:  # NO_SHARD
            parameters_memory = bytes_weight * self.num_params * BYTE_TO_GB
            gradients_memory = bytes_gradient * self.num_params * BYTE_TO_GB
            optimizer_memory = bytes_optimizer * self.num_params * BYTE_TO_GB

        max_memory_alloc = activations_memory + parameters_memory + gradients_memory + optimizer_memory

        # Overhead for temporary buffers and memory fragmentation.
        # Fujii et al. §4.1: estimated memory <= 80% of GPU capacity never OOMs (1.25x factor).
        # Gradient checkpointing increases fragmentation due to repeated recomputation.
        gradient_checkpointing = candidate_params.get("gradient_checkpointing")
        lower_bound = max_memory_alloc * (1.15 if gradient_checkpointing else 1.1)
        upper_bound = max_memory_alloc * (1.35 if gradient_checkpointing else 1.25)

        return (lower_bound, upper_bound)

    def __get_sharding_progression(self, sharding_strategy: str, increase: bool = True) -> str:
        """Get next sharding strategy in progression.

        Args:
            sharding_strategy: Current strategy (NO_SHARD, HYBRID_SHARD, or FULL_SHARD)
            increase: If True, move toward more sharding (less memory). If False, move toward less sharding (more memory)

        Returns:
            Next sharding strategy in progression, or same strategy if at boundary
        """
        if increase:
            return self.SHARDING_INCREASE[sharding_strategy]
        else:
            return self.SHARDING_DECREASE[sharding_strategy]
