from enum import Enum

RFT_REGION_ACCOUNT_MAP = {"us-east-1": "708977205387"}


class RFTJobType(Enum):
    TRAINING = "training"
    REWARD_FUNCTION = "reward-function"
    VLLM_GENERATION = "vllm-generation"
    HUB = "hub"
    PROMPT_RBS = "prompt-rbs"
    NATS_SERVER = "nats-server"
    NATS_BOOTSTRAP = "nats-bootstrap"
    REDIS = "redis"


RFT_JOB_TYPE_DICT = {
    RFTJobType.TRAINING: "rft_training",
    RFTJobType.REWARD_FUNCTION: "rft_training",  # Uses nested config
    RFTJobType.VLLM_GENERATION: "rft_training",  # Uses nested config
    RFTJobType.HUB: "rft_training",  # Uses nested config
    RFTJobType.PROMPT_RBS: "rft_training",  # Uses nested config
    RFTJobType.NATS_SERVER: "rft_training",  # Uses nested config
    RFTJobType.NATS_BOOTSTRAP: "rft_training",  # Uses nested config
    RFTJobType.REDIS: "rft_training",  # Uses nested config
}

RFT_JOB_TASK_TYPE_DICT = {
    RFTJobType.TRAINING: "omega_rl_train",
    RFTJobType.REWARD_FUNCTION: "storm_reward_func",
    RFTJobType.VLLM_GENERATION: "storm_gen_vllm",
    RFTJobType.HUB: "storm_hub",
    RFTJobType.PROMPT_RBS: "storm_prompt_loader",  # For prompter
    RFTJobType.NATS_BOOTSTRAP: "storm_nats_bootstrap",
    RFTJobType.REDIS: "redis_cache",
}

# Additional task types for prompt-rbs service (has two components)
RFT_PROMPT_RBS_TASK_TYPES = {
    "prompter": "storm_prompt_loader",
    "rbs": "storm_rbs",
}

# Keys to remove from run config to avoid duplication in service configs
RFT_KEYS_TO_REMOVE = [
    "training_replicas",
    "reward_function_replicas",
    "vllm_generation_replicas",
    "hub_replicas",
    "prompt_rbs_replicas",
    "nats_server_replicas",
    "redis_replicas",
]

# Service-specific config paths within rft_training
RFT_SERVICE_CONFIG_PATHS = {
    RFTJobType.TRAINING: None,  # Uses root rft_training config
    RFTJobType.REWARD_FUNCTION: "reward_function",
    RFTJobType.VLLM_GENERATION: "vllm_generation",
    RFTJobType.HUB: "hub",
    RFTJobType.PROMPT_RBS: "prompt_rbs",
    RFTJobType.NATS_SERVER: "nats_server",
    RFTJobType.REDIS: "redis",
}

RFT_TRAIN_CONTAINER_IMAGE = "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-TRAIN-V2-latest"
RFT_GENERATION_CONTAINER_IMAGE = (
    "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-GENERATION-V2-latest"
)
RFT_STORM_CONTAINER_IMAGE = "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-STORM-V2-latest"
RFT_NATS_SERVER_CONTAINER_IMAGE = (
    "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-NATS-server-V2-latest"
)
RFT_NATS_RELOADER_CONTAINER_IMAGE = (
    "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-NATS-reloader-V2-latest"
)
RFT_REDIS_CONTAINER_IMAGE = "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SM-HP-RFT-REDIS-V2-latest"
