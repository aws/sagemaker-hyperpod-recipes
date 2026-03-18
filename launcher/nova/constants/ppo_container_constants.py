from enum import Enum

ACTOR_GENERATION_REGION_ACCOUNT_MAP = {"us-east-1": "708977205387", "ap-northeast-1": "878185805882"}


class JobType(Enum):
    REWARD_MODEL = "rm"
    CRITIC_MODEL = "cm"
    ANCHOR_MODEL = "am"
    ACTOR_GENERATION = "ag"
    ACTOR_TRAIN = "at"


JOB_TYPE_DICT = {
    JobType.REWARD_MODEL: "ppo_reward",
    JobType.CRITIC_MODEL: "ppo_critic",
    JobType.ANCHOR_MODEL: "ppo_anchor",
    JobType.ACTOR_GENERATION: "ppo_actor_generation",
    JobType.ACTOR_TRAIN: "ppo_actor_train",
}
JOB_TASK_TYPE_DICT = {
    JobType.REWARD_MODEL: "ppo_rm",
    JobType.CRITIC_MODEL: "ppo_cm",
    JobType.ANCHOR_MODEL: "ppo_anchor",
    JobType.ACTOR_GENERATION: "ppo_actor_gen",
    JobType.ACTOR_TRAIN: "ppo_actor_train",
}
KEYS_TO_REMOVE = ["actor_train_replicas", "rm_replicas", "cm_replicas", "am_replicas"]
ACTOR_GENERATION_CONTAINER_IMAGE = "{account_id}.dkr.ecr.{region}.amazonaws.com/nova-fine-tune-repo:SMHP-PPO-TRT-latest"
