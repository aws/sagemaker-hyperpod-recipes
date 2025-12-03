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
import logging
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from omegaconf import OmegaConf

from scripts.validations.job_recorder import JobRecorder
from scripts.validations.validation_launchers import (
    K8sValidationLauncher,
    SageMakerJobsValidationLauncher,
    SlurmValidationLauncher,
)
from scripts.validations.validation_launchers.launcher_utils import (
    COMMON_CONFIG,
    build_argument_parser,
    cleanup,
    get_input_recipes,
    group_recipes_by_model,
    start_execution,
    validate_platform_auth,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_validation(cfg, fileList=None, save_model_files=True):
    jobRecorder = JobRecorder()

    # Verify environment
    if not validate_platform_auth(cfg):
        logging.error("Unsupported platform. Currently supported platforms are K8, SLURM, SMJOBS")
        return None
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
    start_execution(model_groups, launcher)

    # Cleans resources no longer needed like model files etc
    if not save_model_files:
        cleanup(cfg.models.model_parent_folder)

    return jobRecorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for validating hyperpod recipes")
    parser = build_argument_parser(parser)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config) if args.config != None else OmegaConf.load(COMMON_CONFIG)
    jobRecorder = run_validation(cfg, args.fileList)
    # Output job results
    jobRecorder.print_results()
    # Cleans resources no longer needed like model files etc
    if not args.save_model_files:
        cleanup(cfg.models.model_parent_folder)
