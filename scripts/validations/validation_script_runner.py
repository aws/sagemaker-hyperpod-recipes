# How to run this script:-
# Script should be run from inside its current folder
# The default venv created as a part of the recipes package should contain all the required
# packages.
# Update common_config.yaml file with details specific to your run
# In the run command the path to the recipe file or folder should follow a similar
# format as the ones in the launcher scripts i.e start the path directly from inside the
# recipes_collection/recipes/ folder like in the example below :-
# Example run command :- python scripts/validations/run_validation.py --fileList fine-tuning/llama/filename

import argparse
import copy
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from omegaconf import OmegaConf

from scripts.validations.job_recorder import JobRecorder
from scripts.validations.validation_launchers import (
    K8sValidationLauncher,
    SageMakerJobsValidationLauncher,
    ServerlessValidationLauncher,
    SlurmValidationLauncher,
)
from scripts.validations.validation_launchers.launcher_utils import (
    COMMON_CONFIG_PATH,
    build_argument_parser,
    cleanup,
    get_input_recipes,
    get_instance_types_from_recipe,
    get_recipes_by_regex,
    group_recipes_by_model,
    parse_instance_types,
    start_execution,
    validate_platform_auth,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_validation(cfg, fileList=None, save_model_files=True, instance_type=None):
    """
    Run validation for a specific instance type.

    Args:
        cfg: Configuration object
        fileList: List of recipe files to validate
        save_model_files: Whether to keep model files after validation
        instance_type: Optional instance type override (if None, uses cfg.instance_type)
    """
    # Set instance_type in config if provided
    if instance_type:
        cfg.instance_type = instance_type
        logging.info(f"Running validation with instance type: {instance_type}")

    # Create JobRecorder with instance_type and platform set
    current_instance_type = instance_type or getattr(cfg, "instance_type", None)
    current_platform = getattr(cfg, "platform", None)
    jobRecorder = JobRecorder(instance_type=current_instance_type, platform=current_platform)

    # Verify environment
    if not validate_platform_auth(cfg):
        logging.error("Unsupported platform. Currently supported platforms are K8, SLURM, SMJOBS")
        # Return jobRecorder even on failure so JSON can be written
        return jobRecorder
    fileList = fileList if fileList != None else cfg.recipe_list
    input_file_list = get_input_recipes(fileList, jobRecorder)
    if input_file_list == None:
        logging.error(f"Folder/File '{fileList}' not valid")
        jobRecorder.print_results()
        return None

    # Group recipes by model
    model_groups = group_recipes_by_model(input_file_list)
    logging.info(f"Found {len(model_groups)} model groups")

    # Initiate launcher object and start execution
    match cfg.platform:
        case "K8":
            launcher = K8sValidationLauncher(jobRecorder, cfg)
        case "SLURM":
            launcher = SlurmValidationLauncher(jobRecorder, cfg)
        case "SMJOBS":
            launcher = SageMakerJobsValidationLauncher(jobRecorder, cfg)
        case "SERVERLESS":
            launcher = ServerlessValidationLauncher(jobRecorder, cfg)
    start_execution(model_groups, launcher)

    # Cleans resources no longer needed like model files etc
    if not save_model_files:
        cleanup(cfg.models.model_parent_folder)

    return jobRecorder


# Default mock instance type for serverless platform (instance type is not used but required for config)
SERVERLESS_MOCK_INSTANCE_TYPE = "serverless_mock_instance_type"


def split_recipes_into_batches(recipe_list, batch_size):
    """
    Split a list of recipe files into batches of specified size.

    Args:
        recipe_list: List of recipe file paths to batch
        batch_size: Number of recipes per batch. If None or <= 0, returns all recipes as single batch.

    Returns:
        List of recipe batches (each batch is a list of recipe file paths)
    """
    if not recipe_list:
        return []

    # If batch_size is None or <= 0, return all recipes as a single batch
    if batch_size is None or batch_size <= 0:
        return [recipe_list]

    # Calculate number of batches
    num_batches = math.ceil(len(recipe_list) / batch_size)

    # Split recipes into batches
    recipe_batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(recipe_list))
        recipe_batches.append(recipe_list[start_idx:end_idx])

    return recipe_batches


def run_recipe_batch_validation(
    cfg,
    recipe_batch,
    batch_num,
    total_batches,
    save_model_files=True,
    cli_instance_types=None,
    default_instance_types=None,
):
    """
    Run validation for a batch of recipes in parallel.

    All recipes within a batch are scheduled concurrently using ThreadPoolExecutor.
    This function blocks until all recipes in the batch complete execution.

    Args:
        cfg: Configuration object
        recipe_batch: List of recipe file paths to validate in this batch
        batch_num: Current batch number (1-indexed for logging)
        total_batches: Total number of batches
        save_model_files: Whether to keep model files after validation
        cli_instance_types: Instance types from CLI (takes priority)
        default_instance_types: Default instance types from config

    Returns:
        dict: Dictionary mapping instance_type -> JobRecorder with merged results from all recipes in batch
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"BATCH {batch_num}/{total_batches} - Processing {len(recipe_batch)} recipe(s) in parallel")
    logging.info(f"{'='*60}")
    for recipe in recipe_batch:
        logging.info(f"  - {recipe}")

    batch_results = {}

    # Use ThreadPoolExecutor to process all recipes in the batch in parallel
    with ThreadPoolExecutor() as executor:
        recipe_futures = {}

        for recipe_file in recipe_batch:
            # Determine instance types for this recipe
            if cli_instance_types:
                recipe_instance_types = cli_instance_types
            elif default_instance_types:
                recipe_instance_types = default_instance_types
            else:
                # Try to get instance types from the recipe file itself
                recipe_instance_types = get_instance_types_from_recipe(recipe_file)
                if not recipe_instance_types:
                    logging.error(
                        f"Recipe '{recipe_file}': No instance types available. "
                        "Specify via --instance_types, in the recipe's instance_types field, "
                        "or in config's instance_type_list."
                    )
                    continue

            # Create config copy for this recipe
            cfg_copy = copy.deepcopy(cfg)
            cfg_copy.instance_type_list = recipe_instance_types

            # Submit validation for this recipe
            future = executor.submit(run_validation_for_all_instance_types, cfg_copy, [recipe_file], save_model_files)
            recipe_futures[future] = recipe_file

        # Collect results as recipes complete
        for future in as_completed(recipe_futures):
            recipe_file = recipe_futures[future]
            try:
                recipe_results = future.result()

                # Merge recipe results into batch_results
                for instance_type, job_recorder in recipe_results.items():
                    if instance_type not in batch_results:
                        batch_results[instance_type] = job_recorder
                    elif job_recorder:
                        if batch_results[instance_type]:
                            batch_results[instance_type].jobs.update(job_recorder.jobs)
                        else:
                            batch_results[instance_type] = job_recorder

                logging.info(f"Completed validation for recipe: {recipe_file}")
            except Exception as e:
                logging.error(f"Validation failed for recipe '{recipe_file}': {e}")

    logging.info(f"Batch {batch_num}/{total_batches} completed - all {len(recipe_batch)} recipes finished")
    return batch_results


def run_validation_for_all_instance_types(cfg, fileList=None, save_model_files=True):
    """
    Run validation for each instance type in instance_type_list concurrently.

    Uses ThreadPoolExecutor to submit jobs to different instance types in parallel.
    Each instance type gets its own copy of the config to avoid conflicts.

    For SERVERLESS platform, instance types are not used, so we run validation only once
    with a mock instance type to avoid triggering multiple duplicate jobs.

    Args:
        cfg: Configuration object
        fileList: List of recipe files to validate
        save_model_files: Whether to keep model files after validation

    Returns:
        dict: Dictionary mapping instance_type -> JobRecorder
    """
    # For SERVERLESS platform, instance types don't matter
    # Run validation only once with a mock instance type
    if cfg.platform == "SERVERLESS":
        logging.info(
            f"SERVERLESS platform detected - instance types are not used. "
            f"Running validation once with mock instance type: {SERVERLESS_MOCK_INSTANCE_TYPE}"
        )
        cfg_copy = copy.deepcopy(cfg)
        job_recorder = run_validation(cfg_copy, fileList, save_model_files, SERVERLESS_MOCK_INSTANCE_TYPE)
        return {SERVERLESS_MOCK_INSTANCE_TYPE: job_recorder}

    # Get instance type list from config
    instance_type_list = list(cfg.instance_type_list) if hasattr(cfg, "instance_type_list") else []

    if not instance_type_list:
        logging.error("No instance types configured in instance_type_list")
        return {}

    logging.info(f"Running validation for {len(instance_type_list)} instance type(s) in parallel: {instance_type_list}")

    all_results = {}

    # Use ThreadPoolExecutor to run validations concurrently
    with ThreadPoolExecutor(max_workers=len(instance_type_list)) as executor:
        # Submit all instance type validations
        future_to_instance = {}
        for instance_type in instance_type_list:
            # Create a deep copy of config for each thread to avoid conflicts
            cfg_copy = copy.deepcopy(cfg)
            logging.info(f"Submitting validation job for instance type: {instance_type}")
            future = executor.submit(run_validation, cfg_copy, fileList, save_model_files, instance_type)
            future_to_instance[future] = instance_type

        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance_type = future_to_instance[future]
            try:
                job_recorder = future.result()
                all_results[instance_type] = job_recorder
                logging.info(f"Completed validation for instance type: {instance_type}")
            except Exception as e:
                logging.error(f"Validation failed for instance type {instance_type}: {e}")
                all_results[instance_type] = None

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for validating hyperpod recipes")
    parser = build_argument_parser(parser)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config) if args.config != None else OmegaConf.load(COMMON_CONFIG_PATH)

    # Determine file list - CLI args take priority over config
    # Priority: fileList CLI arg > regex CLI arg > config recipe_list
    file_list = None
    if args.fileList:
        # Handle comma-separated values (from GitHub Actions) or space-separated (from CLI)
        # nargs="+" gives us a list, but each element might contain comma-separated values
        file_list = []
        for item in args.fileList:
            # Split by comma and strip whitespace
            file_list.extend([f.strip() for f in item.split(",") if f.strip()])
        logging.info(f"Using file list from CLI --fileList: {file_list}")
    elif args.regex:
        file_list = get_recipes_by_regex(args.regex)
        if not file_list:
            logging.error(f"No recipes found matching regex '{args.regex}'")
            sys.exit(1)
        logging.info(f"Using file list from CLI --regex: {len(file_list)} recipes matched")
    elif hasattr(cfg, "recipe_list") and cfg.recipe_list:
        file_list = list(cfg.recipe_list)
        logging.info(f"Using file list from config recipe_list: {file_list}")
    else:
        logging.error("No recipes specified. Use --fileList, --regex, or set recipe_list in config.")
        sys.exit(1)

    # Determine instance types
    # Priority: CLI args >  config instance_type_list > recipe's instance_types field
    cli_instance_types = None
    logging.debug(f" args.instance_types raw value: '{args.instance_types}' (type: {type(args.instance_types)})")
    if args.instance_types:
        cli_instance_types = parse_instance_types(args.instance_types)
        logging.debug(f" Parsed CLI instance types: {cli_instance_types}")
        if cli_instance_types:
            logging.info(f"Using instance types from CLI --instance_types: {cli_instance_types}")
        else:
            logging.warning(f"Could not parse instance_types from CLI: '{args.instance_types}'.")
            cli_instance_types = None

    # Get default instance types from config
    default_instance_types = (
        list(cfg.instance_type_list) if hasattr(cfg, "instance_type_list") and cfg.instance_type_list else []
    )

    # Determine recipe batch size
    # Priority: CLI arg > config recipe_batch_size > None (all recipes in single batch)
    batch_size = None
    if hasattr(args, "recipe_batch_size") and args.recipe_batch_size is not None:
        batch_size = args.recipe_batch_size
        logging.info(f"Using recipe batch size from CLI --recipe_batch_size: {batch_size}")
    elif hasattr(cfg, "recipe_batch_size") and cfg.recipe_batch_size is not None:
        batch_size = cfg.recipe_batch_size
        logging.info(f"Using recipe batch size from config: {batch_size}")

    # Validate batch_size
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            logging.error(f"Invalid batch_size: {batch_size}. Must be a positive integer or None.")
            sys.exit(1)

    # Split recipes into batches
    recipe_batches = split_recipes_into_batches(file_list, batch_size)
    total_batches = len(recipe_batches)

    if batch_size is None:
        logging.info(f"Batch size not set - running all {len(file_list)} recipe(s) in a single batch (parallel)")
    else:
        logging.info(
            f"Splitting {len(file_list)} recipe(s) into {total_batches} batch(es) of up to {batch_size} recipe(s) each"
        )
        logging.info("Batches will execute sequentially, recipes within each batch will execute in parallel")

    # Run validation for each batch sequentially
    # Within each batch, recipes are processed in parallel
    all_results = {}
    for batch_num, recipe_batch in enumerate(recipe_batches, start=1):
        batch_results = run_recipe_batch_validation(
            cfg=cfg,
            recipe_batch=recipe_batch,
            batch_num=batch_num,
            total_batches=total_batches,
            save_model_files=args.save_model_files,
            cli_instance_types=cli_instance_types,
            default_instance_types=default_instance_types,
        )

        # Merge batch results into all_results
        for instance_type, job_recorder in batch_results.items():
            if instance_type not in all_results:
                all_results[instance_type] = job_recorder
            elif job_recorder:
                if all_results[instance_type]:
                    all_results[instance_type].jobs.update(job_recorder.jobs)
                else:
                    all_results[instance_type] = job_recorder

        logging.info(f"Batch {batch_num}/{total_batches} results merged into overall results")

    # Print summary of all runs
    logging.info(f"\n{'='*60}")
    logging.info("VALIDATION SUMMARY")
    logging.info(f"{'='*60}")
    for instance_type, job_recorder in all_results.items():
        if job_recorder:
            logging.info(f"\nResults for instance type: {instance_type}")
            job_recorder.print_results()

            # Write JSON results
            json_file = job_recorder.write_results_json()
            if json_file:
                logging.info(f"JSON results written to {json_file}")

    # Cleans resources no longer needed like model files etc
    if not args.save_model_files:
        cleanup(cfg.models.model_parent_folder)
