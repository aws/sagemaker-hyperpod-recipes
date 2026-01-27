import json
import logging
import os
import re
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from omegaconf import OmegaConf

from .path_utils import get_common_config, get_recipes_folder

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# # Constants
RECIPES_FOLDER = get_recipes_folder()
COMMON_CONFIG_PATH = get_common_config()
COMMON_CONFIG = OmegaConf.load(COMMON_CONFIG_PATH)

# Path to jumpstart model ID map
JUMPSTART_MODEL_ID_MAP_PATH = os.path.join(
    Path(__file__).resolve().parent.parent.parent.parent,
    "launcher",
    "recipe_templatization",
    "jumpstart_model-id_map.json",
)


# Load jumpstart model ID map
def _load_jumpstart_model_id_map():
    """Load the jumpstart model ID mapping from JSON file."""
    try:
        with open(JUMPSTART_MODEL_ID_MAP_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Jumpstart model ID map not found at {JUMPSTART_MODEL_ID_MAP_PATH}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing jumpstart model ID map: {e}")
        return {}


JUMPSTART_MODEL_ID_MAP = _load_jumpstart_model_id_map()


def _get_training_type_from_filename(filename):
    """
    Extract training type from recipe filename.

    Training types detected (in order of priority):
    - RLVR: if 'rlvr' in filename
    - RLAIF: if 'rlaif' in filename
    - DPO: if 'dpo' in filename
    - SFT: if 'sft', 'lora', or 'fft' in filename (default for fine-tuning)

    Args:
        filename: Recipe filename

    Returns:
        str: Training type (SFT, DPO, RLVR, or RLAIF)
    """
    filename_lower = filename.lower()

    # Check in order of specificity
    if "rlvr" in filename_lower:
        return "RLVR"
    elif "rlaif" in filename_lower:
        return "RLAIF"
    elif "dpo" in filename_lower:
        return "DPO"
    else:
        # Default to SFT for sft, lora, fft, or other fine-tuning recipes
        return "SFT"


def _get_run_name_from_recipe(recipe_cfg):
    """
    Extract run name from recipe configuration.

    Args:
        recipe_cfg: Loaded recipe configuration (OmegaConf)

    Returns:
        str: Run name from recipe, or None if not found
    """
    if hasattr(recipe_cfg, "run") and hasattr(recipe_cfg.run, "name"):
        return recipe_cfg.run.name
    return None


def _get_jumpstart_model_id(run_name):
    """
    Get jumpstart model ID from run name using the mapping.

    Args:
        run_name: Run name from recipe (e.g., 'llama-3-2-1b-instruct')

    Returns:
        str: Jumpstart model ID, or None if not found
    """
    if not run_name:
        return None
    return JUMPSTART_MODEL_ID_MAP.get(run_name)


def get_instance_types_from_recipe(recipe_file_path):
    """
    Extract instance_types field from a recipe file.

    Args:
        recipe_file_path: Path to the recipe file (relative to RECIPES_FOLDER)

    Returns:
        list: List of instance types from the recipe, or None if not found
    """
    try:
        full_path = os.path.join(RECIPES_FOLDER, recipe_file_path)
        recipe_cfg = OmegaConf.load(full_path)
        if hasattr(recipe_cfg, "instance_types") and recipe_cfg.instance_types:
            instance_types = list(recipe_cfg.instance_types)
            logging.info(f"Found instance_types in recipe '{recipe_file_path}': {instance_types}")
            return instance_types
    except Exception as e:
        logging.warning(f"Error reading instance_types from recipe '{recipe_file_path}': {e}")
    return None


def construct_dynamic_output_path(base_output_path, recipe_cfg, input_file_path):
    """
    Construct dynamic S3 output path based on jumpstart model ID and training type.

    Output path structure:
    <base_output_path>/<jumpstart_model_id>/<training_type>/

    Example:
    s3://bucket/validation_run/meta-textgeneration-llama-3-2-1b-instruct/SFT/

    Args:
        base_output_path: Base S3 output path from config
        recipe_cfg: Loaded recipe configuration
        input_file_path: Recipe file path (for training type detection)

    Returns:
        str: Dynamic output path, or base path if unable to construct
    """
    # Get run name from recipe
    run_name = _get_run_name_from_recipe(recipe_cfg)
    if not run_name:
        logging.warning(f"Could not extract run name from recipe {input_file_path}. Using base output path.")
        return base_output_path

    # Get jumpstart model ID from mapping
    jumpstart_model_id = _get_jumpstart_model_id(run_name)
    if not jumpstart_model_id:
        logging.warning(f"Run name '{run_name}' not found in jumpstart model ID map. Using base output path.")
        return base_output_path

    # Get training type from filename
    training_type = _get_training_type_from_filename(input_file_path)

    # Construct dynamic output path
    dynamic_path = os.path.join(base_output_path, jumpstart_model_id, training_type)

    logging.info(f"Constructed dynamic output path: {dynamic_path}")
    logging.info(f"  - Run name: {run_name}")
    logging.info(f"  - Jumpstart model ID: {jumpstart_model_id}")
    logging.info(f"  - Training type: {training_type}")

    return dynamic_path


def to_hydra_override(value):
    """
    Convert a Python value to Hydra override syntax for command-line arguments.
    Hydra's structured value grammar uses YAML-like syntax.
    Strings containing special characters (brackets, braces) must be quoted with double quotes
    since the entire override is wrapped in single quotes for the shell.
    """
    if isinstance(value, dict):
        # Hydra structured grammar: {key: value, key2: value2}
        items = [f"{k}: {to_hydra_override(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(value, list):
        items = [to_hydra_override(item) for item in value]
        return "[" + ", ".join(items) + "]"
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        return value
    elif value is None:
        return "null"
    else:
        return str(value)


def validate_platform_auth(cfg) -> bool:
    # This method verifies the auth credentials for the platforms (slurm, k8, smjobs, etc)
    if cfg.platform == None:
        logging.info(
            "Platform not specified. Please specify platform by setting the env var PLATFORM = SLURM, K8, or SMJOBS"
        )
        return False

    match cfg.platform:
        case "SLURM":
            try:
                subprocess.run(["squeue"], check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e:
                logging.info(f"SLURM environment ran into an issue:- '{e.stderr}'")

        case "K8":
            try:
                subprocess.run(["kubectl", "get", "pods"], check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e:
                logging.info(f"K8 environment ran into an issue:- '{e.stderr}'")

        case "SMJOBS":
            return True

        case "SERVERLESS":  # Add this case
            # For serverless, we just need to check if AWS CLI is available
            try:
                subprocess.run(["aws", "--version"], check=True, capture_output=True, text=True)
                return True
            except subprocess.CalledProcessError as e:
                logging.info(f"SERVERLESS environment ran into an issue - AWS CLI not available: '{e.stderr}'")
                return False

        case _:
            logging.info("Valid environment (SLURM, K8, SMJOBS) not found")
            return False


def get_input_recipes(recipe_list, jobRecorder):
    fileList = recipe_list
    if fileList == None:
        return None

    # Input sanity check
    input_file_list = []
    for inputRecipe in fileList:
        complete_path = os.path.join(RECIPES_FOLDER, inputRecipe)
        li = os.listdir(complete_path) if os.path.isdir(complete_path) else [inputRecipe]
        for file in li:
            # Handles both kinds of inputs:- Folders and individual files
            filepath = os.path.join(inputRecipe, file) if os.path.isdir(complete_path) else file
            if os.path.exists(os.path.join(RECIPES_FOLDER, filepath)):
                jobRecorder.add_job(input_filename=filepath)
                input_file_list.append(filepath)
            else:
                jobRecorder.update_job(input_filename=filepath, status="Failed", output_log="File does not exist")
                return None
    return input_file_list if len(input_file_list) > 0 else None


def build_argument_parser(parser):
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file for the validation run. Should follow the same template as the commong_validation_config.yaml",
    )
    parser.add_argument(
        "--fileList",
        "-f",
        type=str,
        nargs="+",
        help="list of hyperpod recipes that needs to be executed. Usage: '-f filepath1 filepath2 etc'",
    )
    parser.add_argument(
        "--inputFolder",
        type=str,
        help=" Folder with hyperpod recipes that needs to be executed. Usage: '--inputFolder folder_path'",
    )
    parser.add_argument(
        "--regex",
        "-r",
        type=str,
        default=None,
        help="Regex pattern to match recipe filenames within recipes_collection/recipes/ subfolders. Usage: '--regex verl.*llama'",
    )
    parser.add_argument(
        "--instance_types",
        "-i",
        type=str,
        default=None,
        help="Comma-separated list of instance types to run validation on. Usage: '--instance_types ml.p4d.24xlarge,ml.p5.48xlarge'",
    )
    parser.add_argument(
        "--save_model_files",
        type=bool,
        default=True,
        help="Flag to decide whether to save the model files or delete it after the jobs",
    )
    return parser


def parse_instance_types(instance_types_str):
    """
    Parse comma-separated instance types string into a list.

    Args:
        instance_types_str: Comma-separated string of instance types

    Returns:
        list: List of instance type strings
    """
    if not instance_types_str:
        return []

    # Strip surrounding quotes (single or double) that might come from shell
    cleaned_str = instance_types_str.strip().strip("'\"")

    # Split by comma and strip whitespace
    result = [it.strip() for it in cleaned_str.split(",") if it.strip()]
    logging.info(f"Parsed instance types from '{instance_types_str}' -> {result}")
    return result


def get_recipes_by_regex(regex_pattern):
    """
    Find all recipe files matching a regex pattern within recipes_collection/recipes/ subfolders.

    Args:
        regex_pattern: A regex pattern string to match against recipe filenames

    Returns:
        list: List of matching recipe file paths relative to recipes_collection/recipes/
    """
    matching_recipes = []
    recipes_folder = Path(RECIPES_FOLDER)

    if not recipes_folder.exists():
        logging.error(f"Recipes folder not found: {recipes_folder}")
        return []

    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        logging.error(f"Invalid regex pattern '{regex_pattern}': {e}")
        raise RuntimeError(f"Invalid regex pattern '{regex_pattern}': {e}")

    # Walk through all subfolders in recipes_collection/recipes/
    for root, dirs, files in os.walk(recipes_folder):
        for filename in files:
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                # Get relative path from recipes folder
                full_path = Path(root) / filename
                relative_path = full_path.relative_to(recipes_folder)

                # Match against the full relative path or just the filename
                if pattern.search(str(relative_path)) or pattern.search(filename):
                    matching_recipes.append(str(relative_path))

    if matching_recipes:
        logging.info(f"Found {len(matching_recipes)} recipes matching regex '{regex_pattern}':")
        for recipe in matching_recipes:
            logging.info(f"  - {recipe}")
    else:
        logging.warning(f"No recipes found matching regex '{regex_pattern}'")

    return matching_recipes


def group_recipes_by_model(recipe_list):
    """
    Group recipes by their model name for batched processing.
    Uses config-driven model path extraction.
    """
    model_groups = {}
    for recipe in recipe_list:
        recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, recipe))

        # Get model name using config-driven approach
        model_name = _extract_model_name_from_recipe(COMMON_CONFIG, recipe, recipe_cfg)

        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(recipe)
    return model_groups


def start_execution(model_groups, launcher):
    try:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(launcher.launch_model_group, model_name, recipes)
                for model_name, recipes in model_groups.items()
            ]
            # Wait for all model groups to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Model group execution failed with exception: {e}")
                    logging.error(f"Full traceback:\n{traceback.format_exc()}")

        # Write consolidated throughput CSV after all jobs complete
        try:
            csv_filename = launcher.job_recorder.write_throughput_csv()
            if csv_filename:
                logging.info(f"All jobs completed. Consolidated throughput results available in: {csv_filename}")
        except Exception as e:
            logging.error(f"Error writing consolidated throughput CSV: {e}")

    except Exception as e:
        logging.error(f"Script failed with exception: {e}")
        logging.error(f"Full traceback:\n{traceback.format_exc()}")


# Launcher utils
def construct_slurm_launch_command(cfg, run_info):
    launch_command = get_launch_command(cfg, run_info)
    launch_command.append(f"+cluster.container_mounts.0=/fsx:/fsx")
    return launch_command


def construct_k8_launch_command(cfg, run_info):
    """
    Construct K8-specific launch command.
    Adds platform-common K8 configurations. Recipe-specific commands
    are added via get_launch_command() from config.
    """
    launch_command = get_launch_command(cfg, run_info)

    # Platform-common K8 configurations
    launch_command.extend(
        [
            "cluster=k8s",
            "cluster_type=k8s",
            "+cluster.persistent_volume_claims.0.mountPath=/data",
            f"+cluster.general_pod={run_info['k8_general_pod']}",
        ]
    )

    return launch_command


def construct_smjobs_launch_command(cfg, run_info):
    """
    Construct SM Jobs-specific launch command using config-driven approach.
    Adds platform-common SM Jobs configurations and applies recipe-structure-specific
    overrides from config (for /opt/ml/input/data path mappings).

    Supports two modes:
    - S3 mode (default): Uses S3 paths for inputs
    - FSx mode: Uses FSx Lustre file system for inputs (when smjobs.use_fsx is true)
    """
    # Determine if FSx mode is enabled
    use_fsx = hasattr(cfg, "smjobs") and hasattr(cfg.smjobs, "use_fsx") and cfg.smjobs.use_fsx

    # Update run_info with FSx-specific data if FSx mode is enabled
    if use_fsx:
        run_info = _add_fsx_run_info(cfg, run_info)

    launch_command = get_launch_command(cfg, run_info, use_fsx=use_fsx)

    # Platform-common SM Jobs setup
    launch_command.extend(["cluster=sm_jobs", "cluster_type=sm_jobs", f"instance_type={run_info['instance_type']}"])

    # Get recipe structure config for SM Jobs overrides
    recipe_type_info = _get_recipe_type_info(cfg, run_info["input_file_path"])
    recipe_structure = recipe_type_info["recipe_structure"]

    if not hasattr(cfg, "recipe_structure_config"):
        raise ValueError("recipe_structure_config not found in configuration")

    if recipe_structure not in cfg.recipe_structure_config:
        raise ValueError(f"Recipe structure '{recipe_structure}' not found in recipe_structure_config")

    structure_config = cfg.recipe_structure_config[recipe_structure]

    if use_fsx:
        # Apply FSx-specific overrides
        _apply_fsx_overrides(cfg, launch_command, structure_config, run_info)
    else:
        # Apply S3-specific overrides (original behavior)
        _apply_s3_overrides(launch_command, structure_config, run_info)

    launch_command.append("'cluster.sm_jobs_config.wait=False'")
    print("launch command", launch_command)
    return launch_command


def _add_fsx_run_info(cfg, run_info):
    """
    Add FSx-specific information to run_info dictionary.
    When FSx mode is enabled, this function:
    - Reads model path from smjobs_fsx.model_parent_folder + hf_model_name
    - Reads dataset paths from smjobs_fsx section
    - Adds model_parent_folder for use in template substitution
    """
    run_info = run_info.copy()

    # Get recipe type info to look up correct config keys
    recipe_type_info = _get_recipe_type_info(cfg, run_info["input_file_path"])
    model_config_key = recipe_type_info["model_config_key"]
    recipe_type = recipe_type_info["recipe_type"]

    # Get model_parent_folder from smjobs_fsx config
    if hasattr(cfg.models, model_config_key):
        model_cfg = cfg.models[model_config_key]
        if "smjobs_fsx" in model_cfg:
            smjobs_fsx_cfg = model_cfg["smjobs_fsx"]
            if hasattr(smjobs_fsx_cfg, "model_parent_folder"):
                model_parent_folder = smjobs_fsx_cfg.model_parent_folder
                run_info["model_parent_folder"] = model_parent_folder
                # Update local_model_name_or_path with FSx path
                run_info["local_model_name_or_path"] = os.path.join(model_parent_folder, run_info["hf_model_name"])

    # Get dataset paths from smjobs_fsx section
    if hasattr(cfg, "recipe_dataset_mapping") and recipe_type in cfg.recipe_dataset_mapping:
        recipe_mapping = cfg.recipe_dataset_mapping[recipe_type]
        if "smjobs" in recipe_mapping:
            dataset_names = recipe_mapping["smjobs"]
            if dataset_names and len(dataset_names) > 0:
                dataset_name = dataset_names[0]
                if dataset_name in cfg.datasets:
                    dataset_cfg = cfg.datasets[dataset_name]
                    if "smjobs_fsx" in dataset_cfg:
                        fsx_dataset = dataset_cfg["smjobs_fsx"]
                        run_info["train_data_dir"] = fsx_dataset.train_data_dir
                        run_info["val_data_dir"] = fsx_dataset.val_data_dir
                        if hasattr(fsx_dataset, "train_data_name"):
                            run_info["train_data_name"] = fsx_dataset.train_data_name
                        if hasattr(fsx_dataset, "val_data_name"):
                            run_info["val_data_name"] = fsx_dataset.val_data_name

    return run_info


def _apply_fsx_overrides(cfg, launch_command, structure_config, run_info):
    """
    Apply FSx-specific overrides to the launch command.
    Configures file_system input and VPC settings for FSx Lustre.
    """
    # Prepare variables for substitution
    model_name = run_info["hf_model_name"].split("/")[1]
    substitution_vars = {**run_info, "model_name": model_name}

    # Apply FSx-specific overrides from config
    if hasattr(structure_config, "smjobs_fsx_overrides"):
        for key_template, value_template in structure_config.smjobs_fsx_overrides.items():
            try:
                actual_key = key_template.format(**substitution_vars)
                actual_value = str(value_template).format(**substitution_vars)
                launch_command.append(f"+cluster.sm_jobs_config.{actual_key}={actual_value}")
            except KeyError as e:
                logging.warning(f"Missing variable {e} for FSx override template '{key_template}'. Skipping.")

    # Add file_system configuration - pass each key separately for Hydra compatibility
    if hasattr(cfg.smjobs, "file_system"):
        fs_config = cfg.smjobs.file_system
        fs_id = fs_config.id if hasattr(fs_config, "id") else None
        fs_type = fs_config.type if hasattr(fs_config, "type") else "FSxLustre"
        fs_dir_path = fs_config.directory_path if hasattr(fs_config, "directory_path") else None

        if fs_id and fs_dir_path:
            launch_command.append(f"+cluster.sm_jobs_config.inputs.file_system.id={fs_id}")
            launch_command.append(f"+cluster.sm_jobs_config.inputs.file_system.type={fs_type}")
            launch_command.append(f"+cluster.sm_jobs_config.inputs.file_system.directory_path={fs_dir_path}")

    # Add VPC configuration (subnets and security groups)
    if hasattr(cfg.smjobs, "vpc"):
        vpc_config = cfg.smjobs.vpc
        if hasattr(vpc_config, "subnets"):
            subnets = list(vpc_config.subnets)
            subnets_str = ", ".join([f'"{s}"' for s in subnets])
            launch_command.append(f"'+cluster.sm_jobs_config.additional_estimator_kwargs.subnets=[{subnets_str}]'")

        if hasattr(vpc_config, "security_group_ids"):
            sg_ids = list(vpc_config.security_group_ids)
            sg_str = ", ".join([f'"{s}"' for s in sg_ids])
            launch_command.append(
                f"'+cluster.sm_jobs_config.additional_estimator_kwargs.security_group_ids=[{sg_str}]'"
            )


def _apply_s3_overrides(launch_command, structure_config, run_info):
    """
    Apply S3-specific overrides to the launch command (original behavior).
    """
    if hasattr(structure_config, "smjobs_overrides"):
        # Prepare variables for substitution
        model_name = run_info["hf_model_name"].split("/")[1]
        model_s3_path = run_info["local_model_name_or_path"]
        substitution_vars = {**run_info, "model_name": model_name, "model_s3_path": model_s3_path}

        # Collect inputs.s3 entries to pass as a single dict (avoids Hydra key parsing issues)
        s3_inputs = {}
        for key_template, value_template in structure_config.smjobs_overrides.items():
            actual_key = key_template.format(**substitution_vars)
            actual_value = str(value_template).format(**substitution_vars)

            # Collect inputs.s3.* entries into dict
            if actual_key.startswith("inputs.s3."):
                channel_name = actual_key.replace("inputs.s3.", "")
                s3_inputs[channel_name] = actual_value
            else:
                launch_command.append(f"+cluster.sm_jobs_config.{actual_key}={actual_value}")

        # Pass inputs.s3 as a complete dict override
        if s3_inputs:
            s3_dict_str = ", ".join([f"'{k}': '{v}'" for k, v in s3_inputs.items()])
            launch_command.append(f"'+cluster.sm_jobs_config.inputs.s3={{{s3_dict_str}}}'")


def get_launch_command(cfg, run_info, use_fsx=False):
    """
    Build launch command using config-driven approach.
    Platform-common base + recipe-structure-specific commands from config.

    Args:
        cfg: Configuration object
        run_info: Dictionary with runtime information
        use_fsx: Boolean indicating if FSx mode is enabled (for smjobs platform)
    """
    # Base command (common to all)
    command = ["python3", os.path.join(run_info["training_launcher_dir"], "main.py")]
    command.append(f"recipes='{run_info['input_file_path']}'")
    command.append(f"base_results_dir='{run_info['base_results_dir']}'")

    # Add recipe-structure-specific commands from config
    platform = cfg.platform.lower()

    # For smjobs with FSx mode, use smjobs_fsx commands instead of smjobs
    if use_fsx and platform == "smjobs":
        platform_for_commands = "smjobs_fsx"
    else:
        platform_for_commands = platform

    recipe_commands = _get_recipe_launch_commands(cfg, run_info, platform_for_commands)
    command.extend(recipe_commands)

    # Add common-to-all commands (platform and recipe agnostic)
    command.append(f"git.use_default={run_info['use_default_repo']}")
    command.append(f"+local_model_name_or_path='{run_info['local_model_name_or_path']}'")
    command.append(f"instance_type={run_info['instance_type']}")

    # Handle special cases that need filename-based logic
    # TODO: Consider moving these to config as well
    if "rlaif" in run_info["input_file_path"]:
        command.append("+cluster.service_account_name=bedrock-service-account")

    if "rlvr" in run_info["input_file_path"]:
        command.append(
            "+recipe_overrides.training_config.custom_reward_function.lambda_arn='arn:aws:lambda:us-west-2:855988369404:function:customLambdaRewardGSM8k'"
        )

    if "ppo" in run_info["input_file_path"]:
        command.append("recipes.training_config.algorithm.adv_estimator=gae")
    elif "grpo" in run_info["input_file_path"]:
        command.append("recipes.training_config.algorithm.adv_estimator=grpo")

    # Handle additional_launch_config if present
    # Keys can contain '+' for AND logic (e.g., 'sft+lora' means filename must contain BOTH keywords)
    if cfg.additional_launch_config:
        recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, run_info["input_file_path"]))
        input_file_lower = run_info["input_file_path"].lower()
        for keyword_pattern, _ in cfg.additional_launch_config.items():
            # Split by '+' for AND logic - all keywords must match
            keywords = [kw.strip().lower() for kw in keyword_pattern.split("+")]
            all_keywords_match = all(kw in input_file_lower for kw in keywords)
            if all_keywords_match:
                if cfg.additional_launch_config[keyword_pattern]:
                    for expected_data_type, _ in cfg.additional_launch_config[keyword_pattern].items():
                        if expected_data_type == "ListConfig":
                            for key, value in cfg.additional_launch_config[keyword_pattern][expected_data_type].items():
                                current_cfg = recipe_cfg
                                for sub_cfg in key.split(".")[1:]:
                                    if sub_cfg in current_cfg:
                                        current_cfg = current_cfg[sub_cfg]
                                    else:
                                        logging.info(
                                            f"Unable to find additional_config {sub_cfg} from {key} in {run_info['input_file_path']}"
                                        )
                                value = current_cfg + [value] if current_cfg else [value]
                                override_li = [
                                    to_hydra_override(OmegaConf.to_container(override)) for override in value
                                ]
                                # Wrap entire argument in single quotes so shell passes it as one string to Hydra
                                # Use comma without space to avoid Hydra parsing issues
                                command.append(f"'+{key}=[{','.join(override_li)}]'")
                        else:
                            raise ValueError(
                                f"Additional configs for expected data type: '{expected_data_type}' not supported"
                            )
                break

    return command


def _extract_model_name_from_recipe(cfg, input_file_path, recipe_cfg):
    """
    Extract model name from recipe config using config-driven path.

    Supports both legacy inline recipes and hydra-based recipes where the model path
    is specified in the recipe file as an override.

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path
        recipe_cfg: Loaded recipe configuration

    Returns:
        str: Model name extracted from recipe

    Raises:
        ValueError: If recipe_structure_config is missing or path invalid
    """
    if not hasattr(cfg, "recipe_structure_config"):
        raise ValueError("recipe_structure_config not found in configuration")

    recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
    recipe_structure = recipe_type_info["recipe_structure"]

    if recipe_structure not in cfg.recipe_structure_config:
        raise ValueError(
            f"Recipe structure '{recipe_structure}' not found in recipe_structure_config. "
            f"Available structures: {list(cfg.recipe_structure_config.keys())}"
        )

    structure_config = cfg.recipe_structure_config[recipe_structure]
    if not hasattr(structure_config, "model_path_in_recipe"):
        raise ValueError(f"model_path_in_recipe not defined for recipe structure '{recipe_structure}'")

    # Navigate the recipe config using dot notation path
    model_path = structure_config.model_path_in_recipe
    current = recipe_cfg

    # Traverse the path to find the model name
    for key in model_path.split("."):
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Path not found - this can happen with hydra-based recipes where defaults
            # are not resolved. Fall back to filename-based detection.
            logging.debug(
                f"Key '{key}' not found in recipe config path '{model_path}' for '{input_file_path}'. "
                f"Falling back to filename-based detection."
            )
            current = None
            break

    # Handle empty or None model path - use filename-based fallback
    if current is None or current == "":
        # Filename-based fallback for known model patterns
        filename_lower = input_file_path.lower()

        if "gpt_oss_120b" in filename_lower or "gpt-oss-120b" in filename_lower:
            return "openai/gpt-oss-120b-bf16"
        elif "gpt_oss_20b" in filename_lower or "gpt-oss-20b" in filename_lower:
            return "openai/gpt-oss-20b-bf16"
        else:
            raise ValueError(
                f"Unable to extract model name from recipe config for '{input_file_path}'. "
                f"Path '{model_path}' returned empty or None. "
                f"Ensure the recipe file has model_name_or_path defined or add a filename pattern fallback."
            )

    return current


def _get_recipe_launch_commands(cfg, run_info, platform):
    """
    Get recipe-structure-specific launch commands from config.

    Uses recipe_structure_config to load command templates and applies
    variable substitution using run_info values.

    Args:
        cfg: Configuration object
        run_info: Dictionary with runtime information for variable substitution
        platform: Platform name (k8, smjobs, slurm)

    Returns:
        list: List of formatted command strings

    Raises:
        ValueError: If recipe_structure_config is missing or invalid
    """
    if not hasattr(cfg, "recipe_structure_config"):
        raise ValueError("recipe_structure_config not found in configuration")

    recipe_type_info = _get_recipe_type_info(cfg, run_info["input_file_path"])
    recipe_structure = recipe_type_info["recipe_structure"]

    if recipe_structure not in cfg.recipe_structure_config:
        raise ValueError(
            f"Recipe structure '{recipe_structure}' not found in recipe_structure_config. "
            f"Available structures: {list(cfg.recipe_structure_config.keys())}"
        )

    structure_config = cfg.recipe_structure_config[recipe_structure]
    if not hasattr(structure_config, "launch_commands"):
        logging.warning(
            f"No launch_commands defined for recipe structure '{recipe_structure}'. " f"Returning empty command list."
        )
        return []

    commands = []

    # Add common commands (platform-independent)
    if hasattr(structure_config.launch_commands, "common"):
        for cmd_template in structure_config.launch_commands.common:
            try:
                commands.append(cmd_template.format(**run_info))
            except KeyError as e:
                raise RuntimeError(f"Missing variable {e} for command template '{cmd_template}'.")

    # Add platform-specific commands
    if hasattr(structure_config.launch_commands, platform):
        platform_commands = getattr(structure_config.launch_commands, platform)
        for cmd_template in platform_commands:
            try:
                commands.append(cmd_template.format(**run_info))
            except KeyError as e:
                raise RuntimeError(
                    f"Missing variable {e} for command template '{cmd_template}' in recipe structure '{recipe_structure}'"
                )

    return commands


# This function gets the values for all the relevant experiment
# variables similar to the once used in launcher scripts
def pre_launch_setup(cfg, input_file_path):
    """
    Set up run parameters for a recipe using config-driven approach.

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path

    Returns:
        dict: Run information including model name and all job parameters
    """
    recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, input_file_path))

    # Extract model name using config-driven approach
    hf_model_name = _extract_model_name_from_recipe(cfg, input_file_path, recipe_cfg)

    run_info = get_job_parameters(cfg, input_file_path, hf_model_name)
    run_info["input_file_path"] = input_file_path
    return run_info


# Fetch all parameters used in a Hyperpod recipe launcher_scripts
def get_job_parameters(cfg, input_file_path, hf_model_name):
    """
    Extract job parameters from config based on recipe type and platform.

    Args:
        cfg: OmegaConf configuration object
        input_file_path: Path to the recipe file
        hf_model_name: HuggingFace model name

    Returns:
        dict: Job parameters including paths, model info, and dataset info

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if not hasattr(cfg, "platform") or not cfg.platform:
        raise ValueError("Platform not specified in configuration")

    # Normalize platform name to lowercase
    platform = cfg.platform.lower()

    # Check if FSx mode is enabled for smjobs
    use_fsx = platform == "smjobs" and hasattr(cfg, "smjobs") and hasattr(cfg.smjobs, "use_fsx") and cfg.smjobs.use_fsx

    # Get model path based on recipe type and platform
    local_model_path = _get_model_path(cfg, input_file_path, hf_model_name, platform)

    # Get dataset information
    dataset_info = _get_dataset_info(cfg, input_file_path, platform)

    # Get container path
    container_path = _get_container_path(cfg, input_file_path, platform)

    # Get directory paths
    training_launcher_dir = Path.cwd()
    base_results_dir = os.path.join(training_launcher_dir, "results")

    # Get experiment configuration
    # exp_dir = cfg.experiment_dir
    entry_module = cfg.entry_module
    use_default_repo = cfg.git.use_default

    # Get SM Jobs S3 paths if platform is smjobs
    s3_models_path = ""
    s3_output_path = ""
    if platform == "smjobs":
        # Get S3 models path from models config
        recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
        model_config_key = recipe_type_info["model_config_key"]
        if hasattr(cfg.models, model_config_key) and "smjobs" in cfg.models[model_config_key]:
            s3_models_path = cfg.models[model_config_key]["smjobs"]

        # Get S3 output path - construct dynamic path based on jumpstart model ID and training type
        if hasattr(cfg, "smjobs") and hasattr(cfg.smjobs, "output_path"):
            base_output_path = cfg.smjobs.output_path
            # Load recipe config to extract run name for dynamic path construction
            recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, input_file_path))
            s3_output_path = construct_dynamic_output_path(base_output_path, recipe_cfg, input_file_path)

    # Build job parameters dictionary
    job_params = {
        "training_launcher_dir": training_launcher_dir,
        "base_results_dir": base_results_dir,
        "hf_model_name": hf_model_name,
        "local_model_name_or_path": local_model_path,
        "hf_access_token": cfg.hf.access_token,
        "train_data_name": dataset_info["train_data_name"],
        "train_data_dir": dataset_info["train_data_dir"],
        "val_data_name": dataset_info["val_data_name"],
        "val_data_dir": dataset_info["val_data_dir"],
        "container_path": container_path,
        "training_dir": "",
        "entry_module": entry_module,
        "use_default_repo": use_default_repo,
        "k8_general_pod": cfg.k8.general_pod,
        "instance_type": cfg.instance_type,
        "s3_models_path": s3_models_path,
        "s3_output_path": s3_output_path,
    }

    # Add FSx-specific paths if FSx mode is enabled
    if use_fsx:
        fsx_paths = _get_fsx_paths(cfg, input_file_path, hf_model_name)
        job_params.update(fsx_paths)

    return job_params


def _get_fsx_paths(cfg, input_file_path, hf_model_name):
    """
    Get FSx-specific paths for model and datasets.
    Note: This function is kept for backward compatibility but the main FSx path
    handling is now done in _add_fsx_run_info().

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path
        hf_model_name: HuggingFace model name

    Returns:
        dict: FSx paths for model and datasets
    """
    recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
    model_config_key = recipe_type_info["model_config_key"]
    recipe_type = recipe_type_info["recipe_type"]

    fsx_paths = {}

    # Get FSx model path from smjobs_fsx.model_parent_folder
    if hasattr(cfg.models, model_config_key):
        model_cfg = cfg.models[model_config_key]
        if "smjobs_fsx" in model_cfg:
            smjobs_fsx_cfg = model_cfg["smjobs_fsx"]
            if hasattr(smjobs_fsx_cfg, "model_parent_folder"):
                model_parent_folder = smjobs_fsx_cfg.model_parent_folder
                fsx_paths["model_parent_folder"] = model_parent_folder
                fsx_paths["local_model_name_or_path_fsx"] = os.path.join(model_parent_folder, hf_model_name)

    # Get FSx dataset paths
    if hasattr(cfg, "recipe_dataset_mapping") and recipe_type in cfg.recipe_dataset_mapping:
        recipe_mapping = cfg.recipe_dataset_mapping[recipe_type]
        if "smjobs" in recipe_mapping:
            dataset_names = recipe_mapping["smjobs"]
            if dataset_names and len(dataset_names) > 0:
                dataset_name = dataset_names[0]
                if dataset_name in cfg.datasets:
                    dataset_cfg = cfg.datasets[dataset_name]
                    if "smjobs_fsx" in dataset_cfg:
                        fsx_dataset = dataset_cfg["smjobs_fsx"]
                        fsx_paths["train_data_dir_fsx"] = fsx_dataset.train_data_dir
                        fsx_paths["val_data_dir_fsx"] = fsx_dataset.val_data_dir

    return fsx_paths


def _get_model_path(cfg, input_file_path, hf_model_name, platform):
    """
    Get the local model path based on recipe type and platform.
    Uses recipe_type_config to determine the appropriate model configuration.

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path
        hf_model_name: HuggingFace model name
        platform: Platform (k8 or smjobs)

    Returns:
        str: Local model path

    Raises:
        ValueError: If model configuration is missing for the platform
    """
    recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
    model_config_key = recipe_type_info["model_config_key"]

    if not hasattr(cfg.models, model_config_key):
        raise ValueError(
            f"Model configuration '{model_config_key}' not found in config. "
            f"Available keys: {list(cfg.models.keys())}"
        )

    model_cfg = cfg.models[model_config_key]
    if platform not in model_cfg:
        raise ValueError(
            f"Platform '{platform}' not configured for model config '{model_config_key}'. "
            f"Available platforms: {list(model_cfg.keys())}"
        )

    model_parent_folder = model_cfg[platform]
    return os.path.join(model_parent_folder, hf_model_name)


def _get_recipe_type_info(cfg, input_file_path):
    """
    Get recipe type information from config based on filename keywords.

    This is the core function that makes the system config-driven.
    It reads recipe_type_config from the config file and matches
    detection_keywords against the filename.

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path

    Returns:
        dict: Recipe type info containing:
            - recipe_type: The matched recipe type name
            - model_config_key: Key to use in models section
            - container_key: Key to use in container_info section
            - recipe_structure: Recipe structure type (verl/llmft)

    Raises:
        ValueError: If recipe_type_config is missing or no match found
    """
    if not hasattr(cfg, "recipe_type_config"):
        raise ValueError(
            "recipe_type_config not found in configuration. " "Please add recipe type definitions to your config file."
        )

    filename_lower = input_file_path.lower()

    # Iterate through recipe types in order (order matters!)
    # The first match wins
    for recipe_type, config in cfg.recipe_type_config.items():
        if not hasattr(config, "detection_keywords"):
            logging.warning(f"Recipe type '{recipe_type}' missing detection_keywords. Skipping.")
            continue

        # Check if any keyword matches
        for keyword in config.detection_keywords:
            if keyword.lower() in filename_lower:
                return {
                    "recipe_type": recipe_type,
                    "model_config_key": config.model_config_key,
                    "container_key": config.container_key,
                    "recipe_structure": config.recipe_structure if hasattr(config, "recipe_structure") else "llmft",
                }

    # No match found
    available_types = list(cfg.recipe_type_config.keys())
    raise ValueError(
        f"Could not determine recipe type for '{input_file_path}'. "
        f"No detection keywords matched. Available recipe types: {available_types}. "
        f"Consider adding this recipe type to recipe_type_config in your config file."
    )


def _get_dataset_info(cfg, input_file_path, platform):
    """
    Get dataset information based on recipe type and platform.

    Uses the new two-tier structure:
    1. Determine recipe type from filename using config
    2. Look up dataset names in recipe_dataset_mapping
    3. Get dataset details from datasets section

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path
        platform: Platform (k8 or smjobs)

    Returns:
        dict: Dataset info with train/val names and directories

    Raises:
        ValueError: If dataset configuration is missing or invalid
    """
    if not hasattr(cfg, "recipe_dataset_mapping"):
        raise ValueError("recipe_dataset_mapping not found in configuration")
    if not hasattr(cfg, "datasets"):
        raise ValueError("datasets section not found in configuration")

    # Determine recipe type from filename using config
    recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
    recipe_type = recipe_type_info["recipe_type"]

    if recipe_type not in cfg.recipe_dataset_mapping:
        raise ValueError(
            f"Recipe type '{recipe_type}' not found in recipe_dataset_mapping. "
            f"Available types: {list(cfg.recipe_dataset_mapping.keys())}"
        )

    # Get platform-specific dataset mapping
    recipe_mapping = cfg.recipe_dataset_mapping[recipe_type]
    if platform not in recipe_mapping:
        raise ValueError(
            f"Platform '{platform}' not configured for recipe type '{recipe_type}'. "
            f"Available platforms: {list(recipe_mapping.keys())}"
        )

    # Get list of dataset names for this recipe type and platform
    dataset_names = recipe_mapping[platform]
    if not dataset_names or len(dataset_names) == 0:
        raise ValueError(f"No datasets configured for recipe type '{recipe_type}' on platform '{platform}'")

    # Use the first dataset in the list (or implement more sophisticated selection logic)
    dataset_name = dataset_names[0]

    if dataset_name not in cfg.datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in datasets configuration. "
            f"Available datasets: {list(cfg.datasets.keys())}"
        )

    # Get dataset configuration
    dataset_cfg = cfg.datasets[dataset_name]
    if platform not in dataset_cfg:
        raise ValueError(
            f"Platform '{platform}' not configured for dataset '{dataset_name}'. "
            f"Available platforms: {list(dataset_cfg.keys())}"
        )

    platform_dataset = dataset_cfg[platform]

    # Validate required fields
    required_fields = ["train_data_name", "train_data_dir", "val_data_name", "val_data_dir"]
    for field in required_fields:
        if field not in platform_dataset:
            raise ValueError(
                f"Required field '{field}' missing from dataset '{dataset_name}' "
                f"configuration for platform '{platform}'"
            )

    return {
        "train_data_name": platform_dataset.train_data_name,
        "train_data_dir": platform_dataset.train_data_dir,
        "val_data_name": platform_dataset.val_data_name,
        "val_data_dir": platform_dataset.val_data_dir,
    }


def _get_container_path(cfg, input_file_path, platform):
    """
    Get container path based on recipe type and platform.
    Uses recipe_type_config to determine the appropriate container.

    Args:
        cfg: Configuration object
        input_file_path: Recipe file path
        platform: Platform (k8 or smjobs)

    Returns:
        str: Container image path

    Raises:
        ValueError: If container configuration is missing
    """
    if not hasattr(cfg, "container_info"):
        raise ValueError("container_info not found in configuration")

    recipe_type_info = _get_recipe_type_info(cfg, input_file_path)
    container_key = recipe_type_info["container_key"]

    # Check if container type exists in configuration
    if not hasattr(cfg.container_info, container_key):
        raise ValueError(
            f"Container configuration for '{container_key}' not found. "
            f"Available types: {list(cfg.container_info.keys())}"
        )

    container_cfg = cfg.container_info[container_key]

    # Check if platform is configured
    if platform not in container_cfg:
        raise ValueError(
            f"Platform '{platform}' not configured for container type '{container_key}'. "
            f"Available platforms: {list(container_cfg.keys())}"
        )

    return container_cfg[platform]


# Cleanup utils
def cleanup(path):
    # cleanup model files
    delete_dir(path)


def delete_dir(path: str | Path) -> None:
    p = Path(path)
    logging.info("Deleting directory: '{path}'")
    if p.exists():
        shutil.rmtree(p)
