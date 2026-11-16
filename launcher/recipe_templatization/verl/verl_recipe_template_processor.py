#!/usr/bin/env python3
import copy
import json
import logging
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)

logger = logging.getLogger(__name__)


class VerlRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """VERL-specific recipe template processor for reinforcement learning fine-tuning."""

    framework_type = "verl"

    def __init__(
        self,
        staging_cfg: dict,
        template_path: str = "./launcher/recipe_templatization/verl/verl_recipe_template_parameters.json",
        platform: str = "k8s",
    ):
        self.template_path = template_path
        self.platform = platform
        self._verl_regional_key: Optional[str] = None
        # Set by get_additional_data for the recipe it is processing; see _pow2_length_cap.
        self._pow2_cap: Optional[int] = None
        super().__init__(staging_cfg)

    def _get_template_category(self) -> str:
        """Return the category key for base parameter lookup."""
        return "fine_tuning"

    def _load_template(self):
        """Load VERL template files."""
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open("./launcher/recipe_templatization/jumpstart_model-id_map.json", "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open("./launcher/recipe_templatization/verl/verl_regional_parameters.json", "r") as f:
            self._original_regional_parameters = json.load(f)

    def process_recipe(self, recipe_file_path: str = None, output_path: str = None, dont_override_list=None) -> dict:
        """Process recipe file with template matching, with lora_rank backfill for Mcore recipes."""
        result = super().process_recipe(recipe_file_path, output_path, dont_override_list)

        # Backfill lora_rank default from model.lora.rank for Mcore recipes
        # (Super uses model.lora.rank, not model.lora_rank, to avoid vLLM enabling LoRA for MoE)
        if "lora_rank" in self.recipe_override_parameters:
            lora_rank_default = self.recipe_override_parameters["lora_rank"].get("default", 0)
            if lora_rank_default == 0:
                recipe_cfg = self._load_recipe_config(recipe_file_path)
                training_config = recipe_cfg.get("training_config", {})
                actor_rollout_ref = training_config.get("actor_rollout_ref")
                model = actor_rollout_ref.get("model") if actor_rollout_ref else training_config.get("model")
                if model:
                    lora_cfg = model.get("lora")
                    if lora_cfg and lora_cfg.get("rank") and int(lora_cfg.get("rank")) > 0:
                        self.recipe_override_parameters["lora_rank"]["default"] = int(lora_cfg["rank"])

        return result

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        """Get matching template for VERL recipes based on algorithm type and reward mechanism."""
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        self.algorithm_type = self._extract_algorithm_type(recipe_cfg)

        logger.debug(f"Detected verl algorithm type: {self.algorithm_type}")

        # Determine template key based on algorithm and reward mechanism
        template_key = self.algorithm_type
        if self.algorithm_type == "sft":
            if "fft" in recipe_file_name.lower():
                template_key = "sft_fft"
            else:
                template_key = "sft"
        elif self.algorithm_type == "dpo":
            if "fft" in recipe_file_name.lower():
                template_key = "dpo_fft"
            else:
                template_key = "dpo"
        elif self.algorithm_type == "grpo":
            display_name = recipe_cfg.get("display_name", "")
            customization_technique = self._training_technique(self.algorithm_type, display_name)
            if customization_technique == "RLAIF":
                if "fft" in recipe_file_name.lower():
                    template_key = "grpo_rlaif_fft"
                else:
                    template_key = "grpo_rlaif"
            elif customization_technique == "RLVR":
                if "fft" in recipe_file_name.lower():
                    template_key = "grpo_rlvr_fft"
                else:
                    template_key = "grpo_rlvr"

        if self.algorithm_type in ("sft", "dpo") and self._uses_nemotron_templates(recipe_file_path, recipe_cfg):
            template_key += self.NEMOTRON_TEMPLATE_SUFFIX

        if template_key in template:
            if self._is_verl_0_9_0_recipe(recipe_file_path):
                return self._relocate_reward_kwargs_for_verl_0_9_0(template[template_key])
            return template[template_key]

        raise ValueError(f"Invalid VERL template key: {template_key}")

    @staticmethod
    def _is_verl_0_9_0_recipe(recipe_file_path: str) -> bool:
        """True for verl 0.9 recipes, by path (the rule get_recipe_metadata uses for the regional key)."""
        return "-0_9_0/" in recipe_file_path.lower()

    @staticmethod
    def _relocate_reward_kwargs_for_verl_0_9_0(template_entry: dict) -> dict:
        """Return a copy of the template that writes reward_kwargs where verl 0.9 reads them.

        The shared templates write the RLAIF judge settings (judge_model_id, judge_prompt_template) to
        training_config.reward_model.reward_kwargs, the verl 0.7 path. verl 0.9 recipes keep them under
        training_config.reward.reward_kwargs, so on the 0.7 path the placeholders land nowhere and
        _prune_unlanded_placeholder_params drops them.
        """
        training_config = template_entry.get("recipe_template", {}).get("training_config", {})
        reward_kwargs = training_config.get("reward_model", {}).get("reward_kwargs")
        if reward_kwargs is None:
            return template_entry

        relocated = copy.deepcopy(template_entry)
        training_config = relocated["recipe_template"]["training_config"]
        del training_config["reward_model"]["reward_kwargs"]
        if not training_config["reward_model"]:
            del training_config["reward_model"]
        training_config.setdefault("reward", {})["reward_kwargs"] = copy.deepcopy(reward_kwargs)
        return relocated

    # The sft/sft_fft/dpo/dpo_fft templates expose the llmft override names (logging_steps,
    # generation_cfg, gradient_clipping, seed, ...), so a customer job spec written for an llmft
    # recipe is still accepted by the verl recipe replacing it. A recipe that lacks one of those
    # keys does not advertise it (_prune_unlanded_placeholder_params drops the override).
    # The Nemotron recipes run in their own container (verl_regional_parameters.json keys
    # verl-0.7.0-vllm012 / verl-0.7.0-megatron), which does not support the llmft keys yet,
    # so they use the "<key>_nemotron" copies of the templates without them. The test is the
    # one get_recipe_metadata uses to pick the Nemotron container.
    # Recipes on the Megatron engine (the verl 0.9 GLM recipes) use them too: Megatron's optimizer
    # takes an absolute optim.min_lr, not the min_lr_ratio fraction the default templates describe.
    # test_verl_sft_dpo_templates.py pins which recipes are routed to the Nemotron templates.
    NEMOTRON_TEMPLATE_SUFFIX = "_nemotron"

    @staticmethod
    def _is_nemotron_recipe(recipe_file_path: str) -> bool:
        """True for a Nemotron recipe, which uses the Nemotron container and SFT/DPO templates."""
        return "nemotron" in recipe_file_path.lower()

    @staticmethod
    def _is_megatron_engine_recipe(recipe_cfg) -> bool:
        """True for a recipe whose training_config.engine.strategy is megatron."""
        engine = (recipe_cfg.get("training_config") or {}).get("engine") or {}
        return engine.get("strategy") == "megatron"

    @classmethod
    def _uses_nemotron_templates(cls, recipe_file_path: str, recipe_cfg) -> bool:
        """True for the SFT/DPO recipes routed to the "<key>_nemotron" templates."""
        return cls._is_nemotron_recipe(recipe_file_path) or cls._is_megatron_engine_recipe(recipe_cfg)

    # verl SFT/DPO recipes that replace an llmft recipe on JumpStart publish one power-of-2
    # length (_pow2_length_cap) instead of their raw token budget. The prefixes name each
    # model whose recipes do, so any other model, including a new Llama 3.x one, keeps the
    # raw budget until it is added here. test_verl_sft_dpo_templates.py pins the recipes
    # these prefixes select.
    POW2_LENGTH_CAP_RECIPE_PREFIXES = (
        "verl-sft-llama-3-dot-2-1b-instruct-",
        "verl-sft-llama-3-dot-2-3b-instruct-",
        "verl-sft-llama-3-dot-1-8b-instruct-",
        "verl-sft-llama-3-dot-3-70b-instruct-",
        "verl-dpo-llama-3-dot-2-1b-instruct-",
        "verl-dpo-llama-3-dot-2-3b-instruct-",
        "verl-dpo-llama-3-dot-1-8b-instruct-",
        "verl-dpo-llama-3-dot-3-70b-instruct-",
        "verl-sft-deepseek-r1-distilled-llama-8b-",
        "verl-sft-deepseek-r1-distilled-llama-70b-",
        "verl-sft-deepseek-r1-distilled-qwen-1-dot-5b-",
        "verl-sft-deepseek-r1-distilled-qwen-7b-",
        "verl-sft-deepseek-r1-distilled-qwen-14b-",
        "verl-sft-deepseek-r1-distilled-qwen-32b-",
        "verl-dpo-deepseek-r1-distilled-llama-8b-",
        "verl-dpo-deepseek-r1-distilled-llama-70b-",
        "verl-dpo-deepseek-r1-distilled-qwen-1-dot-5b-",
        "verl-dpo-deepseek-r1-distilled-qwen-7b-",
        "verl-dpo-deepseek-r1-distilled-qwen-14b-",
        "verl-dpo-deepseek-r1-distilled-qwen-32b-",
        "verl-sft-qwen-2-5-7b-instruct-",
        "verl-sft-qwen-2-5-14b-instruct-",
        "verl-sft-qwen-2-5-32b-instruct-",
        "verl-sft-qwen-2-5-72b-instruct-",
        "verl-sft-qwen-3-0-dot-6b-",
        "verl-sft-qwen-3-1-dot-7b-",
        "verl-sft-qwen-3-4b-",
        "verl-sft-qwen-3-8b-",
        "verl-sft-qwen-3-14b-",
        "verl-sft-qwen-3-32b-",
        "verl-dpo-qwen-2-5-7b-instruct-",
        "verl-dpo-qwen-2-5-14b-instruct-",
        "verl-dpo-qwen-2-5-32b-instruct-",
        "verl-dpo-qwen-2-5-72b-instruct-",
        "verl-dpo-qwen-3-0-dot-6b-",
        "verl-dpo-qwen-3-1-dot-7b-",
        "verl-dpo-qwen-3-4b-",
        "verl-dpo-qwen-3-8b-",
        "verl-dpo-qwen-3-14b-",
        "verl-dpo-qwen-3-32b-",
    )

    @classmethod
    def _uses_pow2_length_cap(cls, recipe_file_name: str) -> bool:
        """True if the recipe publishes a power-of-2 length (_pow2_length_cap)."""
        return recipe_file_name.lower().startswith(cls.POW2_LENGTH_CAP_RECIPE_PREFIXES)

    def _pow2_length_cap(self, recipe_file_name: str, recipe_cfg) -> Optional[int]:
        """The power-of-2 length a POW2_LENGTH_CAP_RECIPE_PREFIXES recipe publishes, or None for any other recipe.

        JumpStart offers one of these recipes per (model, technique, PEFT) in place of an
        llmft recipe, and the length it offers must be a power of 2. So instead of the raw
        token budget every other verl recipe publishes, the length params' default and max
        and the SequenceLength label are all the largest power of 2 <= data.max_length.

        For DPO, data.max_length is per sequence: verl packs the chosen and the rejected
        sequence into one sample, so the pair must fit max_token_len_per_gpu. The cap
        therefore also stays within half the token budget for DPO, and within the whole
        budget for SFT. Every Llama, DeepSeek and Qwen 2.5/3 recipe's budget already covers its max_length (SFT
        budget = max_length, DPO budget = 2 x max_length), so this bound only guards a
        recipe sized differently.
        """
        if not self._uses_pow2_length_cap(recipe_file_name):
            return None
        training_config = recipe_cfg.get("training_config")
        if training_config is None:
            return None
        algorithm_type = self._extract_algorithm_type(recipe_cfg)
        if algorithm_type not in ("sft", "dpo"):
            return None
        data = training_config.get("data") or {}
        max_length = data.get("max_length")
        if max_length is None:
            return None
        longest = int(max_length)
        budget = self._extract_context_length(recipe_cfg)
        if budget is not None:
            longest = min(longest, budget // 2 if algorithm_type == "dpo" else budget)
        if longest <= 0:
            return None
        return 1 << (longest.bit_length() - 1)

    def _get_regional_parameters(self, recipe_name: str, recipe_metadata: dict) -> dict:
        """Get regional parameters using the detected verl version key.

        Overrides the base class implementation because detecting verl version requires having the recipe contents.
        It uses the cached self._verl_regional_key from the last run of self.get_recipe_metadata()
        """
        if self._verl_regional_key is None:
            raise ValueError("Failed to get verl regional key")

        if self._verl_regional_key not in self._original_regional_parameters:
            raise ValueError("Invalid verl regional key")

        # Temporarily patch regional_parameters so the base implementation only has the matching version
        patched = {"verl": self._original_regional_parameters[self._verl_regional_key]}
        self.regional_parameters = patched
        return super()._get_regional_parameters(recipe_name, recipe_metadata)

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        """Generate metadata for VERL recipes."""
        metadata = OrderedDict()
        recipe_cfg = self._load_recipe_config(recipe_file_path)
        recipe_metadata_helpers = self.matched_template_group["recipe_metadata_helpers"]

        # Check for verl-0.9.0 first, by path: 0.9 recipes also carry "profiler" / "global_profiler",
        # so the 0.7 config-shape check below cannot tell the two generations apart.
        is_verl_0_9_0 = "-0_9_0/" in recipe_file_path.lower()
        # Check for verl-0.7.0
        # SFT 0.7.0 recipes have "profiler" directly under training_config,
        # while RL 0.7.0 recipes (GRPO/PPO) have "global_profiler" instead.
        is_verl_0_7_0 = "profiler" in recipe_cfg.training_config or "global_profiler" in recipe_cfg.training_config
        if is_verl_0_9_0:
            self._verl_regional_key = "verl-0.9.0"
        elif is_verl_0_7_0:
            if "nemotron-3-5-lightning" in recipe_file_path.lower():
                self._verl_regional_key = "verl-0.7.0-megatron"
            elif self._is_nemotron_recipe(recipe_file_path):
                self._verl_regional_key = "verl-0.7.0-vllm012"
            elif "gemma4" in recipe_file_path.lower():
                self._verl_regional_key = "verl-0.7.0-tf58"
            else:
                self._verl_regional_key = "verl-0.7.0"
        else:
            self._verl_regional_key = "verl"
        logger.debug(f"Detected verl recipe version: {self._verl_regional_key}")

        # Basic metadata
        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        assert recipe_file_name is not None, "Recipe file name not found in recipe config"
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        # Get Display Name
        assert recipe_cfg["display_name"] is not None, "Recipe display name not found in recipe config"
        metadata["DisplayName"] = recipe_cfg["display_name"]

        # Get Job Type - All Verl recipes are FineTuning
        metadata["Type"] = "FineTuning"

        # Extract algorithm type from config
        self.algorithm_type = self._extract_algorithm_type(recipe_cfg)

        # Extract VERL-specific technique using helper function
        is_sft = (
            "sft_algorithms" in recipe_metadata_helpers
            and self.algorithm_type in recipe_metadata_helpers["sft_algorithms"]
        )
        if is_sft:
            metadata["CustomizationTechnique"] = recipe_metadata_helpers["sft_algorithms"][self.algorithm_type]
        elif (
            "rl_algorithms" in recipe_metadata_helpers
            and self.algorithm_type in recipe_metadata_helpers["rl_algorithms"]
        ):
            if self.algorithm_type == "grpo":
                metadata["CustomizationTechnique"] = self._training_technique(
                    self.algorithm_type, metadata["DisplayName"]
                )
            else:
                metadata["CustomizationTechnique"] = recipe_metadata_helpers["rl_algorithms"][self.algorithm_type]
        else:
            raise ValueError(f"Customization technique not found: {self.algorithm_type}")

        # Extract model info from the composed configuration
        model_name = self._extract_model_name(recipe_cfg)
        if model_name in self.recipe_jumpstart_model_id_mapping:
            metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[model_name]
        else:
            raise ValueError(f"Model ID not found: {model_name}")

        # Hardware and instance types
        hardware_type = self._extract_hardware_type(recipe_file_name)
        if hardware_type in recipe_metadata_helpers["hardware_types"]:
            metadata["Hardware"] = hardware_type.upper()

            # Get Instance Types directly from the recipe file
            assert recipe_cfg["instance_types"] is not None, "Recipe InstanceTypes not found in recipe config"
            metadata["InstanceTypes"] = OmegaConf.to_container(recipe_cfg["instance_types"], resolve=True)

        # Extract reward function if present (only for templates that support it)
        if "preset_reward_function" in recipe_metadata_helpers:
            reward_function_name = self._extract_reward_function_name(recipe_cfg)
            if reward_function_name and reward_function_name in recipe_metadata_helpers["preset_reward_function"]:
                metadata["PresetRewardFunction"] = recipe_metadata_helpers["preset_reward_function"][
                    reward_function_name
                ]

        # Set PEFT technique based on recipe name or config
        peft_technique = self._extract_peft_type_from_config(recipe_cfg)
        if peft_technique is None:
            for peft_key, peft_value in recipe_metadata_helpers.get("peft_techniques", {}).items():
                if peft_key in recipe_file_name.lower():
                    peft_technique = peft_value
                    break
        if peft_technique:
            metadata["Peft"] = peft_technique

        # Get Recipe Version
        assert recipe_cfg["version"] is not None, "Recipe version not found in recipe file"
        metadata["Versions"] = [recipe_cfg["version"]]

        # Default OutputConfig
        metadata["OutputConfig"] = {"SageMakerInferenceRecipeName": "default"}

        # Get hosting configs
        hosting_configs = self.load_hosting_config(recipe_file_name)
        if hosting_configs:
            metadata["HostingConfigs"] = hosting_configs

        # Add Serverless Metering
        if is_sft:
            metadata["ServerlessMeteringType"] = ServerlessMeteringType.TOKEN_BASED.value
        else:
            metadata["ServerlessMeteringType"] = ServerlessMeteringType.HOURLY.value

        # Get Instance Count (number of nodes)
        num_nodes = self._extract_num_nodes(recipe_cfg)
        assert num_nodes is not None, "Number of nodes not found in recipe config"
        metadata["InstanceCount"] = num_nodes

        # SequenceLength is the recipe's supported context: the cap the customer
        # dials up to via max_prompt_length + max_response_length (RL) or
        # dataset_max_len (SFT/DPO). It is the DISPLAY/selection label -- the token
        # budget (max_token_len_per_gpu * ulysses_sequence_parallel_size) bucketed
        # DOWN to the backend power-of-2 enum string ("<n>K") by format_sequence_length.
        # The hub content schema (v2_4_0) enforces this power-of-2 enum, so the label
        # MUST floor to it. The numeric ceiling the UI actually enforces is the length
        # params' override `max` (set to the exact budget in get_additional_data), NOT
        # this bucketed label -- so the enforced cap is the true budget even when the
        # label floors below it (e.g. budget 24576 -> label "16K", override max 24576).
        # The POW2_LENGTH_CAP_RECIPE_PREFIXES recipes are the exception: their label,
        # default and max are one power-of-2 length (_pow2_length_cap).
        context_length = self._pow2_length_cap(recipe_file_name, recipe_cfg)
        if context_length is None:
            context_length = self._extract_context_length(recipe_cfg)
        if context_length is None:
            # Fallback for recipes without a computable context length: derive from
            # the configured sequence length.
            context_length = self._extract_sequence_length(recipe_cfg)
        assert context_length is not None, "Sequence length not found in recipe config"
        metadata["SequenceLength"] = self.format_sequence_length(context_length)

        return metadata

    def _extract_algorithm_type(self, recipe_cfg) -> Optional[str]:
        """Extract algorithm type from recipe configuration.

        For RL recipes (e.g. GRPO), the type is found at
        training_config.algorithm.adv_estimator. For DPO recipes, the trainer
        section contains a 'beta' field. For SFT recipes there is no algorithm
        section and no beta field, so we return "sft" in that case.
        """
        training_config = recipe_cfg.get("training_config")
        algorithm = training_config.get("algorithm")
        if algorithm is None:
            trainer = training_config.get("trainer", {})
            if "beta" in trainer:
                return "dpo"
            return "sft"
        adv_estimator = algorithm.get("adv_estimator")
        if adv_estimator is None:
            return "sft"
        return adv_estimator

    def _extract_model_name(self, recipe_cfg) -> Optional[str]:
        """Extract model name for jumpstart mapping lookup."""
        run_config = recipe_cfg.get("run")
        if run_config and "name" in run_config:
            return run_config.get("name")
        return None

    def _extract_hardware_type(self, recipe_name: str) -> str:
        """Extract hardware type from recipe name."""
        return "gpu"

    def _extract_reward_function_name(self, recipe_cfg) -> Optional[str]:
        """Extract reward function name from recipe configuration.

        Returns None for SFT recipes which have no custom_reward_function section.
        """
        training_config = recipe_cfg.get("training_config")
        custom_reward_function = training_config.get("custom_reward_function")
        if custom_reward_function is None:
            return None
        return custom_reward_function.get("name")

    def _extract_peft_type_from_config(self, recipe_cfg) -> Optional[str]:
        """Extract PEFT type from recipe configuration.

        Handles both RL recipes (lora_rank under actor_rollout_ref.model) and
        SFT recipes (lora_rank directly under model).
        """
        training_config = recipe_cfg.get("training_config")

        # SFT recipes have the model config directly under training_config
        actor_rollout_ref = training_config.get("actor_rollout_ref")
        if actor_rollout_ref is not None:
            model = actor_rollout_ref.get("model")
        else:
            model = training_config.get("model")

        if model is None:
            raise ValueError("Could not determine PEFT type")

        lora_rank = model.get("lora_rank")

        # If lora_rank > 0, it's LoRA; if 0 or absent, check model.lora.rank
        # (Mcore/Nemotron recipes use model.lora.rank instead of model.lora_rank)
        if lora_rank and int(lora_rank) > 0:
            return "LoRA"

        # Check nested lora.rank for Mcore recipes
        lora_config = model.get("lora")
        if lora_config and lora_config.get("rank") and int(lora_config.get("rank")) > 0:
            return "LoRA"

        return None

    def _extract_num_nodes(self, recipe_cfg) -> Optional[int]:
        """Extract number of nodes from recipe configuration.

        SFT recipes use trainer.num_nodes; RL recipes use
        ray_cluster.worker_nodes.replicas (with trainer.num_nodes as fallback).
        """
        trainer = recipe_cfg.get("trainer")
        if trainer and "num_nodes" in trainer:
            return trainer.get("num_nodes")

        ray_cluster = recipe_cfg.get("ray_cluster")
        if ray_cluster is None:
            return None
        worker_nodes = ray_cluster.get("worker_nodes")
        if worker_nodes is None:
            return None
        return worker_nodes.get("replicas")

    def _extract_sequence_length(self, recipe_cfg) -> Optional[int]:
        """Extract sequence length from recipe configuration.

        RL recipes use max_prompt_length + max_response_length as the full
        sequence the GPU must handle. SFT recipes use data.max_length.
        Falls back to max_length if max_prompt_length is not present.
        """
        training_config = recipe_cfg.get("training_config")
        data = training_config.get("data")
        max_prompt_length = data.get("max_prompt_length")
        max_response_length = data.get("max_response_length")
        if max_prompt_length is not None and max_response_length is not None:
            return max_prompt_length + max_response_length
        return data.get("max_length")

    # Length override params (RL prompt/response; SFT/DPO dataset_max_len) are
    # dialable up to the recipe's supported sequence length = its effective per-GPU
    # token budget under sequence parallelism:
    #   budget = max_token_len_per_gpu * ulysses_sequence_parallel_size
    # (RL reads actor_rollout_ref.actor.ppo_max_token_len_per_gpu and
    # actor_rollout_ref.actor.ulysses_sequence_parallel_size; SFT/DPO read
    # data.max_token_len_per_gpu and engine.ulysses_sequence_parallel_size). This is
    # published as each length param's override `max` -- the ceiling the UI enforces
    # prompt+response (RL) / dataset_max_len (SFT/DPO) against -- and is NOT floored:
    # the override max must equal the true budget so the UI admits everything the
    # recipe supports (flooring below it would reject the recipe's own baked lengths).
    #
    # Why * sp: under Ulysses sequence parallelism a single sequence is sharded across
    # sp ranks, so verl's packing assert compares max_token_len_per_gpu * sp against
    # the sequence (dp_actor scales the gate by sp before the assert). A recipe
    # shipping sp=2 (e.g. qwen-3-14b) holds 2x its per-GPU gate in one sequence; not
    # crediting sp would under-state its true ceiling and wrongly reject its own baked
    # lengths. Recipes ship sp=1 by default (the multiplier is then a no-op).
    #
    # The published SequenceLength *label* is separately bucketed to the power-of-2
    # enum by format_sequence_length(); that label is display / recipe-selection only
    # and is never the enforced ceiling.
    #
    # The POW2_LENGTH_CAP_RECIPE_PREFIXES recipes are the one exception to all of the
    # above: they publish a single power-of-2 length as default, max and label alike
    # (_pow2_length_cap), so their default is lowered to it rather than rejected.
    _CONTEXT_LENGTH_BOUND_PARAMS = ("max_prompt_length", "max_response_length", "dataset_max_len")

    def _extract_context_length(self, recipe_cfg) -> Optional[int]:
        """The recipe's supported sequence length = max_token_len_per_gpu * sp.

        RL reads actor_rollout_ref.actor.ppo_max_token_len_per_gpu and its
        ulysses_sequence_parallel_size; SFT/DPO read data.max_token_len_per_gpu and
        engine.ulysses_sequence_parallel_size. sp defaults to 1 when absent. Returns
        None when the token-budget field is absent (non-verl or malformed config),
        making the clamp a no-op.
        """
        training_config = recipe_cfg.get("training_config")
        if training_config is None:
            return None

        # RL recipes carry the budget under actor_rollout_ref.actor (its sp gates
        # ppo_max_token_len_per_gpu); SFT/DPO carry it under data (with sp on engine).
        actor_rollout_ref = training_config.get("actor_rollout_ref")
        if actor_rollout_ref is not None:
            actor = actor_rollout_ref.get("actor") or {}
            token_budget = actor.get("ppo_max_token_len_per_gpu")
            sp = actor.get("ulysses_sequence_parallel_size")
        else:
            data = training_config.get("data") or {}
            token_budget = data.get("max_token_len_per_gpu")
            engine = training_config.get("engine") or {}
            sp = engine.get("ulysses_sequence_parallel_size")

        if token_budget is None:
            return None
        budget = int(token_budget)
        if budget <= 0:
            return None
        sp = int(sp) if sp else 1
        return budget * max(sp, 1)

    def get_additional_data(self, recipe_file_path: str) -> list:
        """Set each length param's `max` to the recipe's sequence length.

        The length params carry no static `max` in base_override_parameters.json;
        for verl recipes we set it here to the recipe's effective token budget
        (max_token_len_per_gpu * ulysses_sequence_parallel_size; not floored), so the
        enforced max equals the true sequence length the recipe can serve. No-op when
        the budget can't be computed (the param then stays uncapped) or no length
        params are exposed.

        Power-of-2-capped recipes skip this clamp: _apply_recipe_specific_bounds has already
        capped them at their power-of-2 length, before the base class wrote the
        {min}/{max} hint text, so their hints quote the final range.
        """
        self._pow2_cap = None
        if recipe_file_path is not None:
            recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
            if self._uses_pow2_length_cap(recipe_file_name):
                self._pow2_cap = self._pow2_length_cap(recipe_file_name, self._load_recipe_config(recipe_file_path))
        try:
            additional_data = super().get_additional_data(recipe_file_path)
        finally:
            # Per-recipe state: never let one recipe's cap reach the next recipe.
            pow2_cap, self._pow2_cap = self._pow2_cap, None
        if not additional_data:
            return additional_data

        recipe_metadata, override_params, regional_parameters = additional_data
        if not override_params or pow2_cap is not None:
            return additional_data

        context_length = self._extract_context_length(self._load_recipe_config(recipe_file_path))
        if context_length is not None:
            for param in self._CONTEXT_LENGTH_BOUND_PARAMS:
                if param in override_params:
                    # Never let the ceiling drop below the param's floor (min, and
                    # the default which must stay dialable); otherwise a recipe with
                    # a tiny sequence length would produce an invalid min > max range.
                    param_spec = override_params[param]
                    floor = max(param_spec.get("min", 0), param_spec.get("default", 0))
                    param_spec["max"] = max(context_length, floor)

        return [recipe_metadata, override_params, regional_parameters]

    def _deferred_bound_params(self) -> tuple:
        """A power-of-2-capped recipe's length params, whose bounds _apply_recipe_specific_bounds sets."""
        return self._CONTEXT_LENGTH_BOUND_PARAMS if self._pow2_cap is not None else ()

    def _apply_recipe_specific_bounds(self, override_params: dict, recipe_file_path: str) -> bool:
        """Cap a POW2_LENGTH_CAP_RECIPE_PREFIXES recipe's length params at its power-of-2 length.

        dataset_max_len's default becomes the cap too, so a job that leaves it unset trains
        at that length. max_response_length keeps its default unless it is above the cap.
        Does nothing for other recipes, which get_additional_data clamps to the raw budget.
        """
        cap = self._pow2_cap
        if cap is None:
            return False
        changed = False
        for param in self._CONTEXT_LENGTH_BOUND_PARAMS:
            spec = override_params.get(param)
            if not isinstance(spec, dict):
                continue
            # Never below the param's min, which would invert the range.
            ceiling = max(cap, spec.get("min", 0))
            spec["max"] = ceiling
            default = spec.get("default")
            if param == "dataset_max_len" or (isinstance(default, int) and default > ceiling):
                spec["default"] = ceiling
            changed = True
        return changed

    def _training_technique(self, algorithm_type: str, display_name: str) -> str:
        """Determine training technique based on algorithm type and display name."""
        if algorithm_type == "sft":
            return "SFT"
        elif algorithm_type == "grpo":
            if "RLAIF" in display_name:
                return "RLAIF"
            elif "RLVR" in display_name:
                return "RLVR"
            else:
                raise ValueError(
                    f"Customization technique not found for GRPO technique with display name: {display_name}"
                )
        else:
            # For other algorithms like GAE, return the algorithm type as technique
            return algorithm_type.upper()
