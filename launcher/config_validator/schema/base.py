from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

### LLMFT hydra schema


@dataclass
class LLMFTTrainerConfig:
    devices: int = 1
    num_nodes: int = 1


@dataclass
class LLMFTTrainingArgsConfig:
    micro_train_batch_size: int = 1
    train_batch_size: int = 16
    learning_rate: float = 0.0001
    lr_warmup_ratio: float = 0.1
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    max_epochs: int = 3
    logging_steps: int = 1
    save_steps: int = 0
    eval_steps: int = -1
    beta: float = 0.01
    nll_loss_coef: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class LLMFTrainingConfig:
    training_args: LLMFTTrainingArgsConfig = field(default_factory=LLMFTTrainingArgsConfig)


@dataclass
class LLMFTRecipeConfig:
    trainer: LLMFTTrainerConfig = field(default_factory=LLMFTTrainerConfig)
    training_config: Optional[LLMFTrainingConfig] = field(default=None)


### NOVA hydra schema
@dataclass
class NovaModelConfig:
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0


@dataclass
class NovaTrainerConfig:
    max_epochs: int = 1


@dataclass
class NovaTrainingConfig:
    max_length: int = 8192
    global_batch_size: int = 256
    trainer: NovaTrainerConfig = field(default_factory=NovaTrainerConfig)
    model: NovaModelConfig = field(default_factory=NovaModelConfig)


@dataclass
class NovaRecipeConfig:
    training_config: NovaTrainingConfig = field(default_factory=NovaTrainingConfig)


### Verl hydra schema
@dataclass
class VerlModelOptimConfig:
    """Model optimizer configuration for VERL."""

    lr: float = 1e-5


@dataclass
class VerlModelConfig:
    """Model configuration for VERL."""

    path: Optional[str] = None


@dataclass
class VerlCriticConfig:
    """Critic configuration for VERL."""

    optim: VerlModelOptimConfig = field(default_factory=VerlModelOptimConfig)
    model: VerlModelConfig = field(default_factory=VerlModelConfig)
    ppo_micro_batch_size_per_gpu: int = 4


@dataclass
class VerlKlCtrlConfig:
    """KL control configuration for VERL."""

    kl_coef: float = 0.001
    target_kl: float = 0.1


@dataclass
class VerlAlgorithmConfig:
    """Algorithm configuration for VERL."""

    kl_ctrl: VerlKlCtrlConfig = field(default_factory=VerlKlCtrlConfig)
    adv_estimator: Optional[str] = None  # "gae" or "grpo"


@dataclass
class VerlRayInitConfig:
    """Ray initialization configuration for VERL."""

    num_cpus: Optional[int] = None
    timeline_json_file: Optional[str] = None


@dataclass
class VerlRecipeConfig:
    """Top-level configuration for VERL recipes."""

    critic: Optional[VerlCriticConfig] = field(default_factory=VerlCriticConfig)
    algorithm: Optional[VerlAlgorithmConfig] = field(default_factory=VerlAlgorithmConfig)
    ray_init: Optional[VerlRayInitConfig] = field(default_factory=VerlRayInitConfig)


# Register with Hydra
cs = ConfigStore.instance()
cs.store(name="recipe_schema", node=LLMFTRecipeConfig())
cs.store(group="trainer", name="base_trainer", node=LLMFTTrainerConfig())
cs.store(group="training_config", name="base_training", node=LLMFTrainingConfig())

# Register Nova configs with Hydra
cs.store(name="nova_recipe_schema", node=NovaRecipeConfig())
cs.store(group="training_config", name="nova_training", node=NovaTrainingConfig())

# Register Verl configs with Hydra
cs.store(name="verl_recipe_schema", node=VerlRecipeConfig())
cs.store(group="algorithm", name="verl_algorithm", node=VerlAlgorithmConfig())
cs.store(group="critic", name="verl_critic", node=VerlCriticConfig())
