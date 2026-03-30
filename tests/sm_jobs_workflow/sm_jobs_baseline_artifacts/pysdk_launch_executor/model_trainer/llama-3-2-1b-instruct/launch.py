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
SageMaker Training Job launcher using ModelTrainer API.

This is an alternative to sm_jobs.py which uses the PyTorch Estimator.
"""

import argparse
import logging
import os

import omegaconf
import sagemaker
from omegaconf import OmegaConf
from sagemaker.modules import Session as ModulesSession
from sagemaker.modules.configs import (
    Compute,
    FileSystemDataSource,
    InputData,
    Networking,
    OutputDataConfig,
    TensorBoardOutputConfig,
)
from sagemaker.modules.train.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


def parse_args():
    script_dir = os.path.dirname(os.path.join(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="Launch training recipe using SM jobs ModelTrainer API")
    parser.add_argument(
        "--recipe", type=str, default=os.path.join(script_dir, "recipe.yaml"), help="Path to recipe config."
    )
    parser.add_argument(
        "--sm_jobs_config",
        type=str,
        default=os.path.join(script_dir, "sm_jobs_config.yaml"),
        help="Path to sm jobs config.",
    )
    parser.add_argument("--job_name", type=str, required=True, help="Job name for the SDK job.")
    parser.add_argument("--instance_type", type=str, required=True, help="Instance type to use for the training job.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    sagemaker_session = sagemaker.Session()
    modules_session = ModulesSession(
        boto_session=sagemaker_session.boto_session,
        default_bucket=sagemaker_session.default_bucket(),
    )
    role = sagemaker.get_execution_role()

    sm_jobs_config = OmegaConf.load(args.sm_jobs_config)
    recipe_overrides = sm_jobs_config.get("recipe_overrides", omegaconf.DictConfig(dict()))
    recipe = OmegaConf.load(args.recipe)
    recipe = OmegaConf.merge(recipe, recipe_overrides)
    recipe_overrides = OmegaConf.to_container(recipe_overrides)

    sm_inputs = sm_jobs_config.get("inputs")
    input_data_config = None

    if sm_inputs:
        s3 = sm_inputs.get("s3")
        file_system = sm_inputs.get("file_system")

        if s3 and file_system:
            raise ValueError("Must set only one of s3 or file_system in sm_jobs_config.inputs.")
        if s3 is None and file_system is None:
            raise ValueError("Must set either s3 or file_system in sm_jobs_config.inputs.")

        if file_system:
            file_system_id = file_system.get("id")
            file_system_type = file_system.get("type")
            directory_path = file_system.get("directory_path")

            if file_system_id is None or file_system_type is None or directory_path is None:
                raise ValueError("Must set id, type and directory_path for file_system input type in sm_jobs_config.")

            input_data_config = [
                InputData(
                    channel_name="training",
                    data_source=FileSystemDataSource(
                        file_system_id=file_system_id,
                        file_system_type=file_system_type,
                        directory_path=directory_path,
                        file_system_access_mode="ro",
                    ),
                )
            ]
        else:
            s3_dict = OmegaConf.to_container(s3)
            input_data_config = []
            for channel_name, s3_uri in s3_dict.items():
                if s3_uri:
                    input_data_config.append(InputData(channel_name=channel_name, data_source=s3_uri))

            if not input_data_config:
                input_data_config = None

    output_path = sm_jobs_config.get("output_path")
    if output_path is None:
        raise ValueError("Expected output_path to be set with sm_jobs cluster type")

    additional_estimator_kwargs = sm_jobs_config.get("additional_estimator_kwargs", omegaconf.DictConfig(dict()))
    additional_estimator_kwargs = OmegaConf.to_container(additional_estimator_kwargs)

    environment = sm_jobs_config.get("environment", omegaconf.DictConfig(dict()))
    environment = OmegaConf.to_container(environment)

    compute = Compute(instance_type=args.instance_type)
    output_config = OutputDataConfig(s3_output_path=output_path)

    networking = additional_estimator_kwargs.pop("networking", None)
    subnets = additional_estimator_kwargs.pop("subnets", None)
    security_group_ids = additional_estimator_kwargs.pop("security_group_ids", None)

    if not networking and (subnets or security_group_ids):
        networking = Networking(
            subnets=subnets,
            security_group_ids=security_group_ids,
        )

    base_job_name = args.job_name.replace(".", "-")
    base_job_name = base_job_name.replace("_", "-")

    training_image = additional_estimator_kwargs.pop("training_image", None) or additional_estimator_kwargs.pop(
        "image_uri", None
    )

    trainer = ModelTrainer.from_recipe(
        training_recipe=args.recipe,
        recipe_overrides=recipe_overrides,
        compute=compute,
        output_data_config=output_config,
        input_data_config=input_data_config,
        base_job_name=base_job_name,
        role=role,
        sagemaker_session=modules_session,
        networking=networking,
        stopping_condition=additional_estimator_kwargs.pop("stopping_condition", None),
        requirements=additional_estimator_kwargs.pop("requirements", None),
        training_image=training_image,
        training_image_config=additional_estimator_kwargs.pop("training_image_config", None),
        checkpoint_config=additional_estimator_kwargs.pop("checkpoint_config", None),
        training_input_mode=additional_estimator_kwargs.pop("training_input_mode", "File"),
        environment=environment if environment else None,
        hyperparameters=additional_estimator_kwargs.pop("hyperparameters", None),
        tags=additional_estimator_kwargs.pop("tags", None),
    )

    tensorboard_config = sm_jobs_config.get("tensorboard_config")
    if tensorboard_config:
        tb_output_path = tensorboard_config.get("output_path")
        tb_container_path = tensorboard_config.get("container_logs_path")
        if tb_output_path is None or tb_container_path is None:
            raise ValueError("Please set output path and container path when using tensorboard.")

        trainer.with_tensorboard_output_config(
            TensorBoardOutputConfig(
                s3_output_path=tb_output_path,
                local_path=tb_container_path,
            )
        )

        if recipe.get("exp_manager") is None or recipe.get("exp_manager", dict()).get("explicit_log_dir") is None:
            logger.warning("Using tensorboard but not set exp_manager -> explicit_log_dir for recipe.")

    trainer.train(wait=sm_jobs_config.get("wait", False))


if __name__ == "__main__":
    main()
