# VERL (Versatile Reinforcement Learning) Optimizer
# Supports both FSDP and Megatron strategies for RLAIF/RLVR recipes

from auto_configurator.config_optimizer.base_optimizer import BaseOptimizer


class VerlOptimizer(BaseOptimizer):
    """Optimizer for VERL recipes (RLAIF/RLVR with GRPO algorithm)"""

    def _tunable_params(self) -> list[str]:
        """Return tunable parameters for VERL recipes

        FIXME:
        tensor_model_parallel_degree: actor_rollout_ref.rollout.tensor_model_parallel_size
        shard_degree: we only support FULL_SHARD (actor_rollout_ref.actor.fsdp_config.fsdp_size=-1) and HYBRID_SHARD (actor_rollout_ref.actor.fsdp_config.fsdp_size<world_size).
        expert_model_parallel_degree: fsdp does not support expert parallel
        train_batch_size: data.train_batch_size for each RL update and actor_rollout_ref.actor.ppo_mini_batch_size for each model update
        fp8: actor_rollout_ref.actor.fsdp_config.model_dtype, actor_rollout_ref.ref.fsdp_config.model_dtype, actor_rollout_ref.rollout.dtype. set to bfloat16 by default.
        activation_checkpointing: actor_rollout_ref.model.enable_gradient_checkpointing (default to True)
        offload_activations: actor_rollout_ref.model.enable_activation_offload (default to False)
        sharding_strategy: actor_rollout_ref.actor.fsdp_config.fsdp_size
        limit_all_gathers: not for fsdp2, we use fsdp2 by default actor_rollout_ref.actor.strategy=fsdp2
        """
        raise NotImplementedError("_tunable_params must be implemented")

    def _generate_parameter_ranges(self) -> dict:
        raise NotImplementedError("_generate_parameter_ranges must be implemented")

    def get_recipe_overrides(self, candidate_params) -> list[str]:
        raise NotImplementedError("get_recipe_overrides must be implemented")
        # recipe_overrides = []
        # strategy = self.cfg.training_config.actor.strategy

        # for key, value in candidate_params.items():
        #     match key:
        #         # Common parameters
        #         case "train_batch_size":
        #             recipe_overrides.append(f"++recipes.training_config.data.train_batch_size={value}")
        #         case "ppo_mini_batch_size":
        #             recipe_overrides.append(f"++recipes.training_config.actor.ppo_mini_batch_size={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.ppo_mini_batch_size={value}")
        #         case "ppo_micro_batch_size_per_gpu":
        #             recipe_overrides.append(f"++recipes.training_config.actor.ppo_micro_batch_size_per_gpu={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.ppo_micro_batch_size_per_gpu={value}")

        #         # Gradient checkpointing
        #         case "actor_gradient_checkpointing":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.actor_rollout_ref.enable_gradient_checkpointing={value}"
        #             )
        #         case "critic_gradient_checkpointing":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.critic.model.enable_gradient_checkpointing={value}"
        #             )

        #         # Activation offload
        #         case "actor_activation_offload":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.actor_rollout_ref.enable_activation_offload={value}"
        #             )
        #         case "critic_activation_offload":
        #             recipe_overrides.append(f"++recipes.training_config.critic.model.enable_activation_offload={value}")

        #         # FSDP-specific
        #         case "fsdp_size":
        #             recipe_overrides.append(f"++recipes.training_config.actor.fsdp_config.fsdp_size={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.fsdp_config.fsdp_size={value}")
        #         case "ulysses_sequence_parallel_size":
        #             recipe_overrides.append(f"++recipes.training_config.actor.ulysses_sequence_parallel_size={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.ulysses_sequence_parallel_size={value}")
        #         case "param_offload":
        #             recipe_overrides.append(f"++recipes.training_config.actor.fsdp_config.param_offload={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.fsdp_config.param_offload={value}")
        #         case "optimizer_offload":
        #             recipe_overrides.append(f"++recipes.training_config.actor.fsdp_config.optimizer_offload={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.fsdp_config.optimizer_offload={value}")

        #         # Megatron-specific
        #         case "tensor_model_parallel_size":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.actor.megatron.tensor_model_parallel_size={value}"
        #             )
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.critic.megatron.tensor_model_parallel_size={value}"
        #             )
        #         case "pipeline_model_parallel_size":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.actor.megatron.pipeline_model_parallel_size={value}"
        #             )
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.critic.megatron.pipeline_model_parallel_size={value}"
        #             )
        #         case "expert_model_parallel_size":
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.actor.megatron.expert_model_parallel_size={value}"
        #             )
        #             recipe_overrides.append(
        #                 f"++recipes.training_config.critic.megatron.expert_model_parallel_size={value}"
        #             )
        #         case "context_parallel_size":
        #             recipe_overrides.append(f"++recipes.training_config.actor.megatron.context_parallel_size={value}")
        #             recipe_overrides.append(f"++recipes.training_config.critic.megatron.context_parallel_size={value}")
        #         case _:
        #             continue

        # return recipe_overrides

    def tune_candidate(self, candidate: dict, error_code) -> tuple[dict, bool]:
        """Adjust candidate based on error code

        Returns:
            (adjusted_candidate, should_retry)
        """
        raise NotImplementedError("tune_candidate must be implemented")

    def _estimate_memory_per_gpu(self, train_batch_size: int, candidate_params: dict) -> tuple[float, float]:
        """Estimate memory usage per GPU for VERL training

        VERL has 3 models: actor, critic, ref
        Memory = actor_mem + critic_mem + ref_mem + optimizer + activations

        Returns:
            (lower_bound, upper_bound) memory in GB
        """
        # TODO: Implement VERL-specific memory estimation
        # Consider:
        # - Actor model (trainable)
        # - Critic model (trainable)
        # - Reference model (frozen, can be offloaded)
        # - Rollout buffer
        # - PPO mini-batch vs micro-batch
        raise NotImplementedError("VERL memory estimation not yet implemented")

    def _is_valid_candidate(self, candidate):
        """Validate if a VERL configuration is valid"""
        # TODO: Implement VERL-specific validation
        # Check:
        # - ppo_mini_batch_size divisible by ppo_micro_batch_size_per_gpu
        # - Parallelism constraints (TP * PP * EP <= num_gpus)
        # - Sequence parallel requires use_remove_padding
        raise NotImplementedError("_is_valid_candidate must be implemented")

    # Helper methods for parameter range generation

    # def _set_batch_sizes(self, params):
    #     """Set batch size ranges for VERL"""
    #     # ppo_mini_batch_size: typically 32-256
    #     cfg_value = getattr(self._autotune_cfg, "ppo_mini_batch_size", "auto")
    #     if cfg_value == "auto":
    #         params["ppo_mini_batch_size"] = [32, 64, 128, 256]
    #     elif not isinstance(cfg_value, list):
    #         params["ppo_mini_batch_size"] = [cfg_value]
    #     else:
    #         params["ppo_mini_batch_size"] = cfg_value

    #     # ppo_micro_batch_size_per_gpu: typically 1-8
    #     cfg_value = getattr(self._autotune_cfg, "ppo_micro_batch_size_per_gpu", "auto")
    #     if cfg_value == "auto":
    #         params["ppo_micro_batch_size_per_gpu"] = [1, 2, 4]
    #     elif not isinstance(cfg_value, list):
    #         params["ppo_micro_batch_size_per_gpu"] = [cfg_value]
    #     else:
    #         params["ppo_micro_batch_size_per_gpu"] = cfg_value

    # def _set_gradient_checkpointing(self, params):
    #     """Set gradient checkpointing for actor and critic"""
    #     # Actor gradient checkpointing
    #     cfg_value = getattr(self._autotune_cfg, "actor_gradient_checkpointing", "auto")
    #     if cfg_value == "auto":
    #         params["actor_gradient_checkpointing"] = [True, False]
    #     elif not isinstance(cfg_value, list):
    #         params["actor_gradient_checkpointing"] = [cfg_value]
    #     else:
    #         params["actor_gradient_checkpointing"] = cfg_value

    #     # Critic gradient checkpointing
    #     cfg_value = getattr(self._autotune_cfg, "critic_gradient_checkpointing", "auto")
    #     if cfg_value == "auto":
    #         params["critic_gradient_checkpointing"] = [True, False]
    #     elif not isinstance(cfg_value, list):
    #         params["critic_gradient_checkpointing"] = [cfg_value]
    #     else:
    #         params["critic_gradient_checkpointing"] = cfg_value

    # def _set_activation_offload(self, params):
    #     """Set activation offload for actor and critic"""
    #     # Actor activation offload
    #     cfg_value = getattr(self._autotune_cfg, "actor_activation_offload", "auto")
    #     if cfg_value == "auto":
    #         params["actor_activation_offload"] = [False]  # Usually False for performance
    #     elif not isinstance(cfg_value, list):
    #         params["actor_activation_offload"] = [cfg_value]
    #     else:
    #         params["actor_activation_offload"] = cfg_value

    #     # Critic activation offload
    #     cfg_value = getattr(self._autotune_cfg, "critic_activation_offload", "auto")
    #     if cfg_value == "auto":
    #         params["critic_activation_offload"] = [False]
    #     elif not isinstance(cfg_value, list):
    #         params["critic_activation_offload"] = [cfg_value]
    #     else:
    #         params["critic_activation_offload"] = cfg_value

    # def _set_fsdp_params(self, params):
    #     """Set FSDP-specific parameters"""
    #     # fsdp_size: -1 means auto (use all GPUs)
    #     cfg_value = getattr(self._autotune_cfg, "fsdp_size", "auto")
    #     if cfg_value == "auto":
    #         params["fsdp_size"] = [-1]
    #     elif not isinstance(cfg_value, list):
    #         params["fsdp_size"] = [cfg_value]
    #     else:
    #         params["fsdp_size"] = cfg_value

    #     # Ulysses sequence parallelism
    #     cfg_value = getattr(self._autotune_cfg, "ulysses_sequence_parallel_size", "auto")
    #     if cfg_value == "auto":
    #         params["ulysses_sequence_parallel_size"] = [1]  # Default: no SP
    #     elif not isinstance(cfg_value, list):
    #         params["ulysses_sequence_parallel_size"] = [cfg_value]
    #     else:
    #         params["ulysses_sequence_parallel_size"] = cfg_value

    #     # Parameter offload
    #     cfg_value = getattr(self._autotune_cfg, "param_offload", "auto")
    #     if cfg_value == "auto":
    #         params["param_offload"] = [False]
    #     elif not isinstance(cfg_value, list):
    #         params["param_offload"] = [cfg_value]
    #     else:
    #         params["param_offload"] = cfg_value

    #     # Optimizer offload
    #     cfg_value = getattr(self._autotune_cfg, "optimizer_offload", "auto")
    #     if cfg_value == "auto":
    #         params["optimizer_offload"] = [False]
    #     elif not isinstance(cfg_value, list):
    #         params["optimizer_offload"] = [cfg_value]
    #     else:
    #         params["optimizer_offload"] = cfg_value

    # def _set_megatron_params(self, params):
    #     """Set Megatron-specific parameters"""
    #     # Tensor parallelism
    #     cfg_value = getattr(self._autotune_cfg, "tensor_model_parallel_size", "auto")
    #     if cfg_value == "auto":
    #         params["tensor_model_parallel_size"] = [1, 2, 4, 8]
    #     elif not isinstance(cfg_value, list):
    #         params["tensor_model_parallel_size"] = [cfg_value]
    #     else:
    #         params["tensor_model_parallel_size"] = cfg_value

    #     # Pipeline parallelism
    #     cfg_value = getattr(self._autotune_cfg, "pipeline_model_parallel_size", "auto")
    #     if cfg_value == "auto":
    #         params["pipeline_model_parallel_size"] = [1]  # Default: no PP
    #     elif not isinstance(cfg_value, list):
    #         params["pipeline_model_parallel_size"] = [cfg_value]
    #     else:
    #         params["pipeline_model_parallel_size"] = cfg_value

    #     # Expert parallelism (for MoE models)
    #     cfg_value = getattr(self._autotune_cfg, "expert_model_parallel_size", "auto")
    #     if cfg_value == "auto":
    #         params["expert_model_parallel_size"] = [1]  # Default: no EP
    #     elif not isinstance(cfg_value, list):
    #         params["expert_model_parallel_size"] = [cfg_value]
    #     else:
    #         params["expert_model_parallel_size"] = cfg_value

    #     # Context parallelism (for long sequences)
    #     cfg_value = getattr(self._autotune_cfg, "context_parallel_size", "auto")
    #     if cfg_value == "auto":
    #         params["context_parallel_size"] = [1]  # Default: no CP
    #     elif not isinstance(cfg_value, list):
    #         params["context_parallel_size"] = [cfg_value]
    #     else:
    #         params["context_parallel_size"] = cfg_value
