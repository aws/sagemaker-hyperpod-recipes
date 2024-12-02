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

import argparse
import logging
import os

import omegaconf
import sagemaker
from omegaconf import OmegaConf
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import FileSystemInput
from sagemaker.interactive_apps import SupportedInteractiveAppTypes
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)


def parse_args():
    script_dir = os.path.dirname(os.path.join(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="Launch training recipe using SM jobs")
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
    role = sagemaker.get_execution_role()

    sm_jobs_config = OmegaConf.load(args.sm_jobs_config)
    recipe_overrides = sm_jobs_config.get("recipe_overrides", omegaconf.DictConfig(dict()))
    recipe = OmegaConf.load(args.recipe)
    recipe = OmegaConf.merge(recipe, recipe_overrides)
    recipe_overrides = OmegaConf.to_container(recipe_overrides)

    sm_inputs = sm_jobs_config.get("inputs")
    inputs = None
    if sm_inputs:
        s3 = sm_inputs.get("s3")
        file_system = sm_inputs.get("file_system")
        if s3 and file_system:
            raise ValueError("Must set only one of s3 or file_system in sm_jobs_config.inputs.")
        if s3 is None and file_system is None:
            raise ValueError("Must set either s3 or file_system in sm_jobs_config.inputs.")
        if s3:
            inputs = OmegaConf.to_container(s3)
        else:
            file_system_id = file_system.get("id")
            file_system_type = file_system.get("type")
            directory_path = file_system.get("directory_path")
            if file_system_id is None or file_system_type is None or directory_path is None:
                raise ValueError("Must set id, type and directory_path for file_system input type in sm_jobs_config.")
            inputs = FileSystemInput(
                file_system_id=file_system_id,
                file_system_type=file_system_type,
                directory_path=directory_path,
                file_system_access_mode="ro",
            )

    output_path = sm_jobs_config.get("output_path")
    if output_path is None:
        raise ValueError("Expected output_path to be set with sm_jobs cluster type")

    additional_estimator_kwargs = sm_jobs_config.get("additional_estimator_kwargs", omegaconf.DictConfig(dict()))
    additional_estimator_kwargs = OmegaConf.to_container(additional_estimator_kwargs)

    tensorboard_config = sm_jobs_config.get("tensorboard_config")
    if tensorboard_config:
        tb_output_path = tensorboard_config.get("output_path")
        tb_container_path = tensorboard_config.get("container_logs_path")
        if tb_output_path is None or tb_container_path is None:
            raise ValueError("Please set output path and container path when using tensorboard.")
        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=tb_output_path, container_local_output_path=tb_container_path
        )
        additional_estimator_kwargs["tensorboard_output_config"] = tensorboard_output_config
        if recipe.get("exp_manager") is None or recipe.get("exp_manager", dict()).get("explicit_log_dir") is None:
            logger.warning("Using tensorboard but not set exp_manager -> explicit_log_dir for recipe.")

    base_job_name = args.job_name.replace(".", "-")
    base_job_name = base_job_name.replace("_", "-")
    estimator = PyTorch(
        base_job_name=base_job_name,
        instance_type=args.instance_type,
        training_recipe=args.recipe,
        recipe_overrides=recipe_overrides,
        output_path=output_path,
        role=role,
        sagemaker_session=sagemaker_session,
        **additional_estimator_kwargs,
    )

    if tensorboard_config:
        logger.info("Tensorboard url:")
        logger.info(
            estimator.get_app_url(
                app_type=SupportedInteractiveAppTypes.TENSORBOARD,
                open_in_default_web_browser=False,
            )
        )

    if not isinstance(inputs, FileSystemInput):
        keys_to_pop = []
        for item in inputs.keys():
            if not inputs[item]:
                print(f"poping input {inputs[item]}, {item}")
                keys_to_pop.append(item)
        for item in keys_to_pop:
            inputs.pop(item)
        if len(inputs) == 0:
            inputs = None

    estimator.fit(inputs=inputs, wait=sm_jobs_config.get("wait", False))


if __name__ == "__main__":
    main()
