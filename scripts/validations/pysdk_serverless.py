# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""
PySDK Serverless Training Job Launcher

This script launches serverless fine-tuning jobs using SageMaker PySDK Trainers.
Supports SFT, DPO, RLAIF, and RLVR training types.

"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

from sagemaker.train import DPOTrainer, RLAIFTrainer, RLVRTrainer, SFTTrainer
from sagemaker.train.common import TrainingType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

JUMPSTART_MODEL_ID_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "launcher",
    "recipe_templatization",
    "jumpstart_model-id_map.json",
)

TRAINER_MAPPING = {
    "sft": SFTTrainer,
    "dpo": DPOTrainer,
    "rlaif": RLAIFTrainer,
    "rlvr": RLVRTrainer,
}


def load_jumpstart_model_id_map() -> dict:
    try:
        with open(JUMPSTART_MODEL_ID_MAP_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"JumpStart model ID map not found at {JUMPSTART_MODEL_ID_MAP_PATH}")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JumpStart model ID map: {e}")
        return {}


def convert_to_jumpstart_model_id(model_name: str) -> str:
    jumpstart_map = load_jumpstart_model_id_map()

    if model_name in jumpstart_map:
        jumpstart_id = jumpstart_map[model_name]
        logger.info(f"Converted model name '{model_name}' to JumpStart ID '{jumpstart_id}'")
        return jumpstart_id

    logger.warning(
        f"No JumpStart model ID mapping found for '{model_name}'. "
        "Using original name. Ensure run.name from recipe is being passed."
    )
    return model_name


def get_trainer_class(training_type: str):
    training_type_lower = training_type.lower()

    if training_type_lower not in TRAINER_MAPPING:
        raise ValueError(f"Unknown training type: {training_type}. " f"Supported types: {list(TRAINER_MAPPING.keys())}")

    return TRAINER_MAPPING[training_type_lower]


def get_training_type_enum(training_type_str: str):
    if training_type_str.lower() == "full":
        return TrainingType.FULL
    return TrainingType.LORA


def detect_training_type_from_filename(recipe_path: Optional[str]) -> str:
    if not recipe_path:
        return "sft"

    recipe_lower = recipe_path.lower()

    if "rlvr" in recipe_lower:
        return "rlvr"
    elif "rlaif" in recipe_lower:
        return "rlaif"
    elif "dpo" in recipe_lower:
        return "dpo"
    elif "sft" in recipe_lower:
        return "sft"

    return "sft"


def detect_lora_or_full_from_filename(recipe_path: Optional[str]) -> str:
    if not recipe_path:
        return "lora"

    recipe_lower = recipe_path.lower()

    if "_fft" in recipe_lower or "full_fine_tuning" in recipe_lower:
        return "full"

    return "lora"


def create_trainer(
    model: str,
    training_type: str,
    training_dataset: str,
    model_package_group: str,
    validation_dataset: Optional[str] = None,
    lora_or_full: str = "lora",
    mlflow_resource_arn: Optional[str] = None,
    mlflow_experiment_name: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    s3_output_path: Optional[str] = None,
    kms_key_id: Optional[str] = None,
    accept_eula: bool = False,
    custom_reward_function: Optional[str] = None,
    **kwargs,
):
    """
    Create a serverless trainer instance.

    Args:
        model: Model name from catalog (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
        training_type: Training type ('sft', 'dpo', 'rlaif', 'rlvr')
        training_dataset: S3 URI or dataset ARN for training data
        model_package_group: Model package group name or ARN
        validation_dataset: S3 URI or dataset ARN for validation data
        lora_or_full: Fine-tuning approach ('lora' or 'full')
        mlflow_resource_arn: MLflow tracking server ARN
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        s3_output_path: S3 path for outputs
        kms_key_id: KMS key ID for encryption
        accept_eula: Whether to accept EULA for gated models
        custom_reward_function: Evaluator ARN for RLAIF/RLVR

    Returns:
        Trainer instance
    """
    TrainerClass = get_trainer_class(training_type)
    training_type_enum = get_training_type_enum(lora_or_full)

    trainer_kwargs = {
        "model": model,
        "training_type": training_type_enum,
        "model_package_group": model_package_group,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "mlflow_resource_arn": mlflow_resource_arn,
        "mlflow_experiment_name": mlflow_experiment_name,
        "mlflow_run_name": mlflow_run_name,
        "s3_output_path": s3_output_path,
        "kms_key_id": kms_key_id,
        "accept_eula": accept_eula,
    }

    # RLAIF and RLVR require custom_reward_function
    if training_type.lower() in ["rlaif", "rlvr"]:
        if custom_reward_function:
            trainer_kwargs["custom_reward_function"] = custom_reward_function
        else:
            logger.warning(
                f"{training_type.upper()} training typically requires custom_reward_function. " "Proceeding without it."
            )

    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    jumpstart_model_id = convert_to_jumpstart_model_id(model)
    trainer_kwargs["model"] = jumpstart_model_id

    logger.info(f"Creating {TrainerClass.__name__} with model: {jumpstart_model_id}")
    logger.info(f"Training type: {training_type_enum}")
    logger.info(f"Model package group: {model_package_group}")

    return TrainerClass(**trainer_kwargs)


def launch_training(
    trainer,
    training_dataset: Optional[str] = None,
    validation_dataset: Optional[str] = None,
    wait: bool = True,
):
    """
    Launch the training job.

    Args:
        trainer: Trainer instance
        training_dataset: Override training dataset (optional)
        validation_dataset: Override validation dataset (optional)
        wait: Whether to wait for job completion

    Returns:
        TrainingJob object
    """
    logger.info("Launching serverless training job...")

    train_kwargs = {"wait": wait}
    if training_dataset:
        train_kwargs["training_dataset"] = training_dataset
    if validation_dataset:
        train_kwargs["validation_dataset"] = validation_dataset

    training_job = trainer.train(**train_kwargs)

    print(f"TRAINING_JOB_NAME={training_job.training_job_name}")

    logger.info(f"Training job created: {training_job.training_job_name}")

    if wait:
        logger.info("Waiting for training job to complete...")
    else:
        logger.info("Training job submitted (wait=False). Check status separately.")

    return training_job


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch serverless fine-tuning jobs using SageMaker PySDK Trainers",
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Model name from catalog (e.g., 'meta-llama/Llama-3.2-1B-Instruct')"
    )
    parser.add_argument("--training_dataset", type=str, required=True, help="S3 URI or dataset ARN for training data")
    parser.add_argument(
        "--model_package_group",
        type=str,
        required=True,
        help="Model package group name or ARN for storing fine-tuned model",
    )

    parser.add_argument(
        "--training_type",
        type=str,
        choices=["sft", "dpo", "rlaif", "rlvr"],
        default="sft",
        help="Training type (default: sft)",
    )
    parser.add_argument(
        "--lora_or_full",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Fine-tuning approach: LoRA or full fine-tuning (default: lora)",
    )

    parser.add_argument("--validation_dataset", type=str, help="S3 URI or dataset ARN for validation data")

    parser.add_argument("--mlflow_resource_arn", type=str, help="MLflow tracking server ARN")
    parser.add_argument("--mlflow_experiment_name", type=str, help="MLflow experiment name")
    parser.add_argument("--mlflow_run_name", type=str, help="MLflow run name")

    parser.add_argument("--s3_output_path", type=str, help="S3 path for training outputs")
    parser.add_argument("--kms_key_id", type=str, help="KMS key ID for encryption")

    # RLAIF/RLVR specific
    parser.add_argument("--custom_reward_function", type=str, help="Evaluator ARN for RLAIF/RLVR training")

    parser.add_argument("--accept_eula", action="store_true", help="Accept EULA for gated models")
    parser.add_argument(
        "--wait", action="store_true", default=True, help="Wait for training job to complete (default: True)"
    )
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Don't wait for training job to complete")

    parser.add_argument("--recipe_path", type=str, help="Recipe path for auto-detecting training type and lora/full")

    return parser.parse_args()


def main():
    args = parse_args()

    training_type = args.training_type
    lora_or_full = args.lora_or_full

    if args.recipe_path:
        training_type = detect_training_type_from_filename(args.recipe_path)
        lora_or_full = detect_lora_or_full_from_filename(args.recipe_path)
        logger.info(f"Auto-detected from recipe: training_type={training_type}, lora_or_full={lora_or_full}")

    try:
        trainer = create_trainer(
            model=args.model,
            training_type=training_type,
            training_dataset=args.training_dataset,
            model_package_group=args.model_package_group,
            validation_dataset=args.validation_dataset,
            lora_or_full=lora_or_full,
            mlflow_resource_arn=args.mlflow_resource_arn,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            s3_output_path=args.s3_output_path,
            kms_key_id=args.kms_key_id,
            accept_eula=args.accept_eula,
            custom_reward_function=args.custom_reward_function,
        )

        training_job = launch_training(
            trainer=trainer,
            wait=args.wait,
        )

        logger.info(f"Training job name: {training_job.training_job_name}")

        return 0

    except Exception as e:
        logger.error(f"Training job failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
