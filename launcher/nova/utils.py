import os

import boto3

from .constants.init_container_constants import (
    INIT_CONTAINER_IMAGE_URI,
    INIT_CONTAINER_REGION_ACCOUNT_MAP,
)
from .constants.ppo_container_constants import (
    ACTOR_GENERATION_CONTAINER_IMAGE,
    ACTOR_GENERATION_REGION_ACCOUNT_MAP,
)
from .constants.rft_container_constants import (
    RFT_GENERATION_CONTAINER_IMAGE,
    RFT_NATS_RELOADER_CONTAINER_IMAGE,
    RFT_NATS_SERVER_CONTAINER_IMAGE,
    RFT_REDIS_CONTAINER_IMAGE,
    RFT_REGION_ACCOUNT_MAP,
    RFT_STORM_CONTAINER_IMAGE,
    RFT_TRAIN_CONTAINER_IMAGE,
)


def get_current_region():
    region = os.environ.get("AWS_REGION") or boto3.session.Session().region_name

    if not region:
        raise ValueError("AWS region could not be determined during initialization.")
    return region


def get_actor_generation_container_uri() -> str:
    region = get_current_region()

    account_id = ACTOR_GENERATION_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova recipes are not supported for region '{region}'.")

    return ACTOR_GENERATION_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_init_container_uri() -> str:
    region = get_current_region()

    account_id = INIT_CONTAINER_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova recipes are not supported for region '{region}'.")

    return INIT_CONTAINER_IMAGE_URI.format(account_id=account_id, region=region)


def get_rft_generation_container_uri() -> str:
    """Get the RFT VLLM generation container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_GENERATION_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_rft_storm_container_uri() -> str:
    """Get the RFT Storm container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_STORM_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_rft_nats_server_container_uri() -> str:
    """Get the RFT NATS server container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_NATS_SERVER_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_rft_nats_reloader_container_uri() -> str:
    """Get the RFT NATS reloader container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_NATS_RELOADER_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_rft_redis_container_uri() -> str:
    """Get the RFT Redis container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_REDIS_CONTAINER_IMAGE.format(account_id=account_id, region=region)


def get_rft_train_container_uri() -> str:
    """Get the RFT train container URI."""
    region = get_current_region()
    account_id = RFT_REGION_ACCOUNT_MAP.get(region)
    if not account_id:
        raise ValueError(f"Nova RFT recipes are not supported for region '{region}'.")

    return RFT_TRAIN_CONTAINER_IMAGE.format(account_id=account_id, region=region)
