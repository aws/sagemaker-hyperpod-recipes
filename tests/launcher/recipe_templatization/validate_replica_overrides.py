"""
Shared validation functions for Nova K8s launch.json replica overrides.

These validators ensure:
1. Master replicas are NEVER modified (always = 1)
2. Worker replicas are correctly templatized per recipe type
3. Proper recipe-specific override behavior
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_launch_json(launch_json_path: Path) -> Dict:
    """Load and parse launch.json file."""
    with open(launch_json_path, "r") as f:
        return json.load(f)


def detect_recipe_type(metadata: Dict) -> str:
    """
    Detect recipe type from metadata.

    Returns: 'ppo', 'rft', 'sft', 'dpo', 'lora', 'cpt', or 'unknown'
    """
    recipe_name = metadata.get("Name", "").lower()

    if "ppo" in recipe_name:
        return "ppo"
    elif "rft" in recipe_name or "grpo" in recipe_name:
        return "rft"
    elif "lora" in recipe_name:
        return "lora"
    elif "dpo" in recipe_name:
        return "dpo"
    elif "sft" in recipe_name:
        return "sft"
    elif "pretrain" in recipe_name or "cpt" in recipe_name:
        return "cpt"

    return "unknown"


def validate_master_replicas_never_modified(training_yaml: str) -> Tuple[bool, List[str]]:
    """
    Validate that ALL Master replicas in training.yaml are exactly 1.

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Find all Master replica specifications
    master_pattern = r"Master:\s+replicas:\s+(\d+|{{.*?}})"
    matches = re.findall(master_pattern, training_yaml)

    if not matches:
        errors.append("No Master replica specifications found in training.yaml")
        return False, errors

    for i, replica_value in enumerate(matches):
        if replica_value != "1":
            errors.append(f"Master replicas #{i+1} is '{replica_value}', expected '1'")

    if errors:
        return False, errors

    logger.info(f"✓ PASS: All {len(matches)} Master replica specs are correctly set to 1")
    return True, []


def validate_worker_replicas_templatized(
    training_yaml: str, recipe_type: str, expected_jobs_with_workers: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that Worker replicas are correctly templatized based on recipe type.

    Args:
        training_yaml: Content of training.yaml from launch.json
        recipe_type: Type of recipe ('ppo', 'rft', 'sft', etc.)
        expected_jobs_with_workers: Optional list of job suffixes that should have Workers
                                   (e.g., ['-at'] for PPO, ['-train'] for RFT)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Find all job names and their Worker replica values
    # Use regex to find each PyTorchJob and check if it has Workers
    # Pattern: Find metadata name, then look for Worker section within same PyTorchJob
    matches = []

    # Split into documents using the actual YAML document separator
    # Handle both stored as JSON string (literal \n) and actual newlines
    if "\\n---\\n" in training_yaml:
        yaml_docs = training_yaml.split("\\n---\\n")
    else:
        yaml_docs = training_yaml.split("\n---\n")

    for doc in yaml_docs:
        # Extract job name from this document
        name_match = re.search(r"name:\s+({{name}}-\w+)", doc)
        if not name_match:
            continue
        job_name = name_match.group(1)

        # Check if this specific document has a Worker section
        worker_match = re.search(r"Worker:\s+replicas:\s+({{replicas}}|\d+|\'{{replicas}}\')", doc)
        if worker_match:
            replica_value = worker_match.group(1)
            matches.append((job_name, replica_value))

    if not matches:
        # No Workers found - valid for some recipes (e.g., single-node jobs)
        if recipe_type in ["sft", "lora", "dpo", "cpt"]:
            # These might have or not have Workers depending on node count
            logger.info(f"No Worker specs found for {recipe_type} recipe (may be single-node)")
            return True, []
        else:
            errors.append(f"No Worker replica specifications found for {recipe_type} recipe")
            return False, errors

    # Validate based on recipe type
    for job_name, replica_value in matches:
        # Clean up replica value (remove quotes if present)
        clean_value = replica_value.strip("'\"")

        if recipe_type == "ppo":
            # For PPO: only actor-train (-at) should have Worker replicas templatized
            if "-at" in job_name:
                if clean_value != "{{replicas}}":
                    errors.append(
                        f"PPO actor-train job '{job_name}' Worker replicas is '{replica_value}', "
                        f"expected '{{{{replicas}}}}'"
                    )
            else:
                # Other PPO jobs (rm, cm, am) should NOT have Workers or should be numeric
                if clean_value == "{{replicas}}":
                    errors.append(f"PPO non-actor-train job '{job_name}' should NOT have templatized Worker replicas")

        elif recipe_type == "rft":
            # For RFT: only training (-train) job should have Worker replicas templatized
            if "-train" in job_name or "train" in job_name:
                if clean_value != "{{replicas}}":
                    errors.append(
                        f"RFT training job '{job_name}' Worker replicas is '{replica_value}', "
                        f"expected '{{{{replicas}}}}'"
                    )
            else:
                # Other RFT services should NOT have Workers in PyTorchJob
                errors.append(f"RFT non-training job '{job_name}' has unexpected Worker spec")

        elif recipe_type in ["sft", "lora", "dpo", "cpt"]:
            # For SFT/LoRA/DPO/CPT: ALL PyTorchJob Workers should be templatized
            if clean_value != "{{replicas}}":
                errors.append(
                    f"{recipe_type.upper()} job '{job_name}' Worker replicas is '{replica_value}', "
                    f"expected '{{{{replicas}}}}'"
                )

    if errors:
        return False, errors

    logger.info(f"✓ PASS: All Worker replicas correctly templatized for {recipe_type} recipe")
    return True, []


def validate_config_num_nodes_templatized(training_config_yaml: str, recipe_type: str) -> Tuple[bool, List[str]]:
    """
    Validate that training config num_nodes is correctly templatized.

    For PPO: actor-train config should have num_nodes: '{{replicas}}'
    For others: Managed via values.yaml, not in config

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if recipe_type == "ppo":
        # Find actor-train config and check num_nodes
        pattern = r"name: training-config-{{name}}-at.*?num_nodes:\s+(\'{{replicas}}\'|{{replicas}}|\d+)"
        matches = re.findall(pattern, training_config_yaml, re.DOTALL)

        if not matches:
            errors.append("PPO actor-train config not found or missing num_nodes")
            return False, errors

        num_nodes_value = matches[0].strip("'\"")
        if num_nodes_value != "{{replicas}}":
            errors.append(f"PPO actor-train config num_nodes is '{matches[0]}', expected '{{{{replicas}}}}'")
            return False, errors

        logger.info("✓ PASS: PPO actor-train config num_nodes correctly templatized")

    return True, []


def validate_launch_json_replica_overrides(launch_json_path: Path) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation of replica overrides in launch.json.

    Validates:
    1. All Master replicas = 1
    2. Worker replicas correctly templatized per recipe type
    3. Config num_nodes correctly templatized (PPO only)

    Returns:
        Tuple of (is_valid, list_of_all_errors)
    """
    all_errors = []

    try:
        launch_data = load_launch_json(launch_json_path)

        # Get recipe type from metadata
        metadata = launch_data.get("metadata", {})
        recipe_type = detect_recipe_type(metadata)
        recipe_name = metadata.get("Name", "unknown")

        logger.info(f"\n{'='*80}")
        logger.info(f"Validating replica overrides for: {recipe_name} ({recipe_type})")
        logger.info(f"{'='*80}")

        # Get YAML contents
        training_yaml = launch_data.get("training.yaml", "")
        training_config_yaml = launch_data.get("training-config.yaml", "")

        if not training_yaml:
            all_errors.append("training.yaml not found in launch.json")
            return False, all_errors

        # Validation 1: Master replicas must always be 1
        is_valid, errors = validate_master_replicas_never_modified(training_yaml)
        if not is_valid:
            all_errors.extend([f"[Master Validation] {e}" for e in errors])

        # Validation 2: Worker replicas correctly templatized
        is_valid, errors = validate_worker_replicas_templatized(training_yaml, recipe_type)
        if not is_valid:
            all_errors.extend([f"[Worker Validation] {e}" for e in errors])

        # Validation 3: Config num_nodes templatized (PPO only)
        if training_config_yaml:
            is_valid, errors = validate_config_num_nodes_templatized(training_config_yaml, recipe_type)
            if not is_valid:
                all_errors.extend([f"[Config Validation] {e}" for e in errors])

        if all_errors:
            logger.error(f"\n✗ FAILED: {len(all_errors)} validation error(s) found")
            for error in all_errors:
                logger.error(f"  - {error}")
            return False, all_errors

        logger.info(f"\n✓ SUCCESS: All replica override validations passed for {recipe_name}")
        return True, []

    except Exception as e:
        error_msg = f"Exception during validation: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]


def validate_all_nova_k8s_launch_jsons(results_dir: Path) -> Tuple[int, int, List[str]]:
    """
    Validate all Nova K8s launch.json files in results directory.

    Returns:
        Tuple of (passed_count, failed_count, list_of_failures)
    """
    passed = 0
    failed = 0
    failures = []

    # Find all launch.json files in results directory
    launch_json_files = list(results_dir.glob("*/launch.json"))

    logger.info(f"\nFound {len(launch_json_files)} launch.json files to validate")

    for launch_json_path in launch_json_files:
        is_valid, errors = validate_launch_json_replica_overrides(launch_json_path)

        if is_valid:
            passed += 1
        else:
            failed += 1
            failures.append({"path": str(launch_json_path), "errors": errors})

    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total: {len(launch_json_files)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")

    if failures:
        logger.error(f"\nFailed validations:")
        for failure in failures:
            logger.error(f"\n{failure['path']}:")
            for error in failure["errors"]:
                logger.error(f"  - {error}")

    return passed, failed, failures
