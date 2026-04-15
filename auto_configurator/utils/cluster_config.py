"""Cluster and instance type configuration for auto-configurator."""

import logging
import subprocess

from omegaconf import DictConfig, ListConfig

AUTO = "auto"

DEFAULT_INSTANCE_TYPES = [
    "ml.p4d.24xlarge",
    "ml.p4de.24xlarge",
    "ml.p5.48xlarge",
    "ml.g5.12xlarge",
    "ml.g5.48xlarge",
    "ml.g6.48xlarge",
]


def get_instance_type_list(auto_config):
    """Resolve instance_type_list from auto config. Supports 'auto', a list, or a single value."""
    instance_types = auto_config.autotune_config.get("instance_type_list", AUTO)

    if instance_types == AUTO:
        return DEFAULT_INSTANCE_TYPES
    elif not isinstance(instance_types, (list, ListConfig)):
        return [instance_types]
    else:
        return list(instance_types)


def validate_cluster(instance_type: str, cfg: DictConfig) -> None:
    """Verify the cluster has nodes with the expected instance type.

    Args:
        instance_type: The instance type to validate.
        cfg: OmegaConf config with k8.cluster_context_map.

    Raises:
        ValueError: If k8.cluster_context_map is missing or has no entry for instance_type.
        RuntimeError: If kubectl fails or no matching nodes are found.
    """
    logger = logging.getLogger("AutoConfigurator")

    context = cfg.get("k8", {}).get("cluster_context_map", {}).get(instance_type)
    if not context:
        raise ValueError(f"No kubectl context mapped for instance type {instance_type}.")

    cmd = [
        "kubectl",
        "--context",
        context,
        "get",
        "nodes",
        "-l",
        f"node.kubernetes.io/instance-type={instance_type}",
        "--no-headers",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to query cluster nodes: {result.stderr}")

    nodes = [line for line in result.stdout.strip().splitlines() if line]

    if not nodes:
        raise RuntimeError(f"Instance type {instance_type} not found in cluster.")

    logger.info(f"Found {len(nodes)} node(s) with instance type {instance_type}")
