import logging
import os
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
COMMON_CONFIG = get_common_config()


def to_hydra_override(value):
    if isinstance(value, dict):
        items = [f"{k}: {to_hydra_override(v)}" for k, v in value.items()]
        return "{" + ", ".join(items) + "}"
    elif isinstance(value, list):
        items = [to_hydra_override(item) for item in value]
        return "[" + ", ".join(items) + "]"
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, str):
        return value
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
        "--save_model_files",
        type=bool,
        default=True,
        help="Flag to decide whether to save the model files or delete it after the jobs",
    )
    return parser


def group_recipes_by_model(recipe_list):
    model_groups = {}
    for recipe in recipe_list:
        recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, recipe))
        if "llmft" in recipe:
            model_name = recipe_cfg.training_config.model_config.model_name_or_path
        elif "verl" in recipe:
            model_name = recipe_cfg.training_config.actor_rollout_ref.model.path
        else:
            raise ValueError(f"Unknown recipe structure in file: {recipe}")
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
    launch_command = get_launch_command(cfg, run_info)
    launch_command.append("cluster=k8s")
    launch_command.append("cluster_type=k8s")
    launch_command.append("+cluster.persistent_volume_claims.0.claimName=fsx-claim")
    launch_command.append("+cluster.persistent_volume_claims.0.mountPath=/data")
    launch_command.append(f"+cluster.general_pod={run_info['k8_general_pod']}")

    # For VERL/Ray jobs, add Ray-specific configurations
    if "verl" in run_info["input_file_path"]:
        launch_command.append("+cluster.ray.enabled=true")
        launch_command.append("cluster.namespace=ray-training")
        launch_command.append("++cluster.persistent_volume_claims.0.claimName=fsx-pvc-aps1")
        # launch_command.append("cluster.persistent_volume_claims=[{claimName:fsx-pvc-aps1,mountPath:data}]")
    return launch_command


def construct_smjobs_launch_command(cfg, run_info):
    launch_command = get_default_launch_command(cfg, run_info)
    launch_command.append("cluster=sm_jobs")
    launch_command.append("cluster_type=sm_jobs")
    launch_command.append(f"instance_type={run_info['instance_type']}")

    model_owner, model_name = run_info["hf_model_name"].split("/")

    sm_jobs_updates = {}
    if "verl" in run_info["input_file_path"]:
        sm_jobs_updates = {
            "recipe_overrides.training_config.data.train_files": f"{run_info['train_data_dir']}",
            "recipe_overrides.training_config.data.val_files": f"{run_info['val_data_dir']}",
            "recipe_overrides.training_config.actor_rollout_ref.model.path": f"/opt/ml/input/data/models/{run_info['hf_model_name']}",
            "recipe_overrides.training_config.critic.model.path": "/opt/ml/input/data/models/deepseek-ai/deepseek-llm-7b-chat",
            "recipe_overrides.training_config.critic.model.tokenizer_path": f"/opt/ml/input/data/models/{run_info['hf_model_name']}",
            "recipe_overrides.training_config.reward_model.model.path": "/opt/ml/input/data/models/sfairXC/FsfairX-LLaMA3-RM-v0.1",
            "recipe_overrides.training_config.reward_model.model.input_tokenizer": f"/opt/ml/input/data/models/{run_info['hf_model_name']}",
        }
        launch_command += [
            f"+cluster.sm_jobs_config.additional_estimator_kwargs.image_uri={run_info['container_path']}",
            "+cluster.sm_jobs_config.additional_estimator_kwargs.use_training_recipe=true",
            "++cluster.sm_jobs_config.inputs.file_system=null",
            f"+cluster.sm_jobs_config.inputs.s3.models={run_info['s3'].models}",
            "cluster.sm_jobs_config.tensorboard_config=''",
            f"recipes.training_config.trainer.default_local_dir='{run_info['base_results_dir']}/verl_outputs'",
        ]

    else:
        sm_jobs_updates = {
            "inputs.s3.train": run_info["train_data_dir"],
            "inputs.s3.validation": run_info["val_data_dir"],
            f"inputs.s3.{model_owner}": run_info["s3"]["models_parent_folder"] + model_owner,
            "recipe_overrides.run.results_dir": "/opt/ml/model",
            "recipe_overrides.training_config.datasets.train_data.name": run_info["train_data_name"],
            "recipe_overrides.training_config.datasets.train_data.file_path": "/opt/ml/input/data/train",
            "recipe_overrides.training_config.datasets.train_data.limit": 150,
            "recipe_overrides.training_config.datasets.val_data.name": run_info["val_data_name"],
            "recipe_overrides.training_config.datasets.val_data.file_path": "/opt/ml/input/data/validation",
            "recipe_overrides.training_config.datasets.val_data.limit": 50,
            "recipe_overrides.training_config.training_args.max_epochs": 1,
            "recipe_overrides.training_config.model_config.model_name_or_path": os.path.join(
                "/opt/ml/input/data", run_info["hf_model_name"]
            ),
            "additional_estimator_kwargs.image_uri": run_info["container_path"],
        }
        launch_command.append("+model.model_type=llm_finetuning_aws")
        launch_command.append("cluster.sm_jobs_config.inputs.file_system=''")
        launch_command.append("cluster.sm_jobs_config.tensorboard_config=''")
        launch_command.append("+model.model_type=llm_finetuning_aws")

    for key, value in sm_jobs_updates.items():
        launch_command.append(f"+cluster.sm_jobs_config.{key}={value}")

    # TODO: Add sm_jobs_updates for filesystem input"
    launch_command += [
        f"cluster.sm_jobs_config.output_path={run_info['s3']['validation_output_folder']}",
        "cluster.sm_jobs_config.wait=False",
    ]

    return launch_command


def get_default_launch_command(cfg, run_info):
    command = ["python3", os.path.join(run_info["training_launcher_dir"], "main.py")]
    command.append(f"recipes='{run_info['input_file_path']}'")
    command.append(f"base_results_dir='{run_info['base_results_dir']}'")

    # Standard LLMFT structure
    if not "verl" in run_info["input_file_path"]:
        command.append(f"recipes.training_config.training_args.training_dir='{run_info['training_dir']}'")
        command.append(f"recipes.run.hf_access_token='{run_info['hf_access_token']}'")
        if cfg.additional_launch_config:
            recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, run_info["input_file_path"]))
            for recipe_type, _ in cfg.additional_launch_config.items():
                if recipe_type in run_info["input_file_path"]:
                    if cfg.additional_launch_config[recipe_type]:
                        for expected_data_type, _ in cfg.additional_launch_config[recipe_type].items():
                            if expected_data_type == "ListConfig":
                                for key, value in cfg.additional_launch_config[recipe_type][expected_data_type].items():
                                    current_cfg = recipe_cfg
                                    for sub_cfg in key.split(".")[
                                        1:
                                    ]:  # Ignore the custom wrappers like recipes. and recipe_overrides. which are not directly present in the recipe file
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
                                    command.append(f"+{key}=[{','.join(override_li)}]")
                                    print(command)
                            else:
                                raise ValueError(
                                    f"Additional configs for expected data type: '{expected_data_type}' not supported"
                                )
                    break
        command.append("recipes.training_config.datasets.save_intermediate_processing_steps=true")
    return command


def get_launch_command(cfg, run_info):
    command = get_default_launch_command(cfg, run_info)

    if "verl" in run_info["input_file_path"]:
        # For VERL recipes
        command.append(f"recipes.training_config.data.train_files='{run_info['train_data_dir']}'")
        command.append(f"recipes.training_config.data.val_files='{run_info['val_data_dir']}'")
        command.append(f"recipes.training_config.actor_rollout_ref.model.path='{run_info['local_model_name_or_path']}'")
        command.append(
            f"recipes.training_config.trainer.default_local_dir='{run_info['base_results_dir']}/verl_outputs'"
        )
        command.append("++recipes.training_config.critic.model.path=/data/hf_pretrained_models/deepseek-llm-7b-chat")
        command.append(
            f"++recipes.training_config.critic.model.tokenizer_path='{run_info['local_model_name_or_path']}'"
        )
        command.append(
            "++recipes.training_config.reward_model.model.path=/data/hf_pretrained_models/FsfairX-LLaMA3-RM-v0.1"
        )
        command.append(
            f"++recipes.training_config.reward_model.model.input_tokenizer='{run_info['local_model_name_or_path']}'"
        )
        command.append(f"container='{run_info['container_path']}'")
        if "rlaif" in run_info["input_file_path"]:
            command.append("cluster.service_account_name=bedrock-service-account")

        # Set algorithm.adv_estimator based on recipe type
        if "ppo" in run_info["input_file_path"]:
            command.append("recipes.training_config.algorithm.adv_estimator=gae")
        elif "grpo" in run_info["input_file_path"]:
            command.append("recipes.training_config.algorithm.adv_estimator=grpo")

    else:
        # LLMFT structure
        command.append(
            f"recipes.training_config.model_config.model_name_or_path='{run_info['local_model_name_or_path']}'"
        )
        command.append(f"recipes.training_config.datasets.train_data.name='{run_info['train_data_name']}'")
        command.append(f"recipes.training_config.datasets.train_data.file_path='{run_info['train_data_dir']}'")
        command.append(f"recipes.training_config.datasets.train_data.limit={150}")
        command.append(f"recipes.training_config.datasets.val_data.name='{run_info['val_data_name']}'")
        command.append(f"recipes.training_config.datasets.val_data.file_path='{run_info['val_data_dir']}'")
        command.append(f"recipes.training_config.datasets.val_data.limit={50}")
        command.append(f"recipes.training_config.training_args.max_epochs={1}")
        command.append(f"container='{run_info['container_path']}'")
        command.append(f"+entry_module='{run_info['entry_module']}'")
    command.append(f"git.use_default={run_info['use_default_repo']}")
    command.append(f"+local_model_name_or_path='{run_info['local_model_name_or_path']}'")
    command.append(f"instance_type={run_info['instance_type']}")
    return command


# This function gets the values for all the relevant experiment
# variables similar to the once used in launcher scripts
def pre_launch_setup(cfg, input_file_path):
    recipe_cfg = OmegaConf.load(os.path.join(RECIPES_FOLDER, input_file_path))

    # Handle different recipe structures for model name extraction
    if "verl" in input_file_path:
        # VERL uses actor_rollout_ref.model.path
        hf_model_name = recipe_cfg.training_config.actor_rollout_ref.model.path
    else:
        # Standard LLMFT structure
        hf_model_name = recipe_cfg.training_config.model_config.model_name_or_path

    run_info = get_job_parameters(cfg, input_file_path, hf_model_name)
    run_info["input_file_path"] = input_file_path
    return run_info


# Fetch all parameters used in a Hyperpod recipe launcher_scripts
def get_job_parameters(cfg, input_file_path, hf_model_name):
    if "verl" in input_file_path:
        local_model_path = os.path.join(cfg.models.verl.model_parent_folder, hf_model_name)
        if "ppo" in input_file_path:
            dataset_key = "ppo"
        else:
            dataset_key = "grpo"
        if cfg.platform == "SMJOBS":
            if "rlaif" in input_file_path:
                dataset_key = "verl-smjobs-panda"
            else:
                dataset_key = "verl-smjobs"
    else:
        local_model_path = os.path.join(cfg.models.default.model_parent_folder, hf_model_name)
        dataset_key = ""

    training_launcher_dir = Path.cwd()
    base_results_dir = cfg.base_results_dir or os.path.join(training_launcher_dir, "results")

    # get dataset info
    if dataset_key == "":
        for key in cfg.dataset_keys:
            if key in input_file_path:
                dataset_key = key
                break
    if dataset_key == "":
        raise ValueError(f"Supported dataset not found for {input_file_path}")
    train_data_name = cfg.dataset_keys[dataset_key].train_data_name
    train_data_dir = cfg.dataset_keys[dataset_key].train_data_dir
    val_data_name = cfg.dataset_keys[dataset_key].val_data_name
    val_data_dir = cfg.dataset_keys[dataset_key].val_data_dir

    # get container info
    container_path = cfg.container_info.default.container_path
    if input_file_path in cfg.container_info:
        container_path = cfg.container_info[input_file_path].container_path
    elif "verl" in input_file_path and hasattr(cfg.container_info, "verl"):
        if cfg.platform == "SMJOBS":
            container_path = cfg.container_info.verl_smjobs.container_path
        else:
            container_path = cfg.container_info.verl.container_path

    # get experiment info
    exp_dir = cfg.experiment_dir
    entry_module = cfg.entry_module
    use_default_repo = cfg.git.use_default

    # get s3 info
    instance_type = getattr(cfg, "instance_type", None)

    job_params = {
        "training_launcher_dir": training_launcher_dir,
        "base_results_dir": base_results_dir,
        "hf_model_name": hf_model_name,
        "local_model_name_or_path": local_model_path,
        "hf_access_token": cfg.hf.access_token,
        "train_data_name": train_data_name,
        "train_data_dir": train_data_dir,
        "val_data_name": val_data_name,
        "val_data_dir": val_data_dir,
        "container_path": container_path,
        "training_dir": exp_dir,
        "entry_module": entry_module,
        "use_default_repo": use_default_repo,
        "k8_general_pod": cfg.k8.general_pod,
        "s3": cfg.s3,
        "instance_type": cfg.instance_type,
    }
    return job_params


# Cleanup utils
def cleanup(path):
    # cleanup model files
    delete_dir(path)


def delete_dir(path: str | Path) -> None:
    p = Path(path)
    logging.info("Deleting directory: '{path}'")
    if p.exists():
        shutil.rmtree(p)
