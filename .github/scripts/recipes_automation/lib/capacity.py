"""Shared K8s capacity check utilities for GitHub workflow scripts."""

import json
import logging
import os
import subprocess

import boto3

logger = logging.getLogger(__name__)

CAPACITY_THRESHOLD = 0.8


def setup_kubeconfig(cluster_map, assume_role_arn):
    """Configure kubectl for all clusters in cluster_map."""
    if not cluster_map or not assume_role_arn:
        return

    sts = boto3.client("sts")
    creds = sts.assume_role(RoleArn=assume_role_arn, RoleSessionName="capacity-check")["Credentials"]
    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"],
        "AWS_SESSION_TOKEN": creds["SessionToken"],
    }

    for cluster_arn in set(cluster_map.values()):
        cluster_name = cluster_arn.split("/")[-1]
        region = cluster_arn.split(":")[3]
        subprocess.run(
            [
                "aws",
                "eks",
                "update-kubeconfig",
                "--name",
                cluster_name,
                "--region",
                region,
                "--role-arn",
                assume_role_arn,
            ],
            check=True,
            capture_output=True,
            env=env,
        )


def has_capacity(instance_type, cluster_map):
    """Check if <80% of GPU nodes for this instance type are busy."""
    cluster_arn = cluster_map.get(instance_type)
    if not cluster_arn:
        logger.warning("No cluster mapping for %s, assuming capacity available", instance_type)
        return True

    try:
        context = cluster_arn
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                context,
                "get",
                "nodes",
                "-l",
                f"node.kubernetes.io/instance-type={instance_type}",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes = json.loads(result.stdout).get("items", [])
        total_nodes = len(nodes)
        if total_nodes == 0:
            logger.info("No nodes found for %s, assuming capacity available", instance_type)
            return True

        node_names = {n["metadata"]["name"] for n in nodes}
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                context,
                "get",
                "pods",
                "--all-namespaces",
                "--field-selector=status.phase!=Succeeded,status.phase!=Failed",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pods = json.loads(result.stdout).get("items", [])

        busy_nodes = set()
        for pod in pods:
            node_name = pod.get("spec", {}).get("nodeName")
            if node_name not in node_names:
                continue
            for container in pod.get("spec", {}).get("containers", []):
                requests = container.get("resources", {}).get("requests", {})
                limits = container.get("resources", {}).get("limits", {})
                if "nvidia.com/gpu" in requests or "nvidia.com/gpu" in limits:
                    busy_nodes.add(node_name)
                    break

        utilization = len(busy_nodes) / total_nodes
        has_cap = utilization < CAPACITY_THRESHOLD
        logger.info(
            "Capacity %s: %d/%d busy (%.0f%%) — %s",
            instance_type,
            len(busy_nodes),
            total_nodes,
            utilization * 100,
            "available" if has_cap else "full",
        )
        return has_cap
    except Exception as e:
        logger.warning("Capacity check failed for %s: %s — assuming available", instance_type, e)
        return True


def wait_for_capacity(instance_type, cluster_map, timeout=432000, interval=300):
    """Block until capacity is available. Returns True if available, False if timed out."""
    import time

    elapsed = 0
    while elapsed < timeout:
        if has_capacity(instance_type, cluster_map):
            return True
        logger.info("Waiting for capacity on %s, retrying in %ds...", instance_type, interval)
        time.sleep(interval)
        elapsed += interval
    return False
