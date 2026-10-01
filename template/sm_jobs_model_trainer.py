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

Two submission modes are supported, selected by the
``additional_estimator_kwargs.use_training_recipe`` flag (default: False):

* ``use_training_recipe`` unset / False  (NeMo / LLMFT recipes)
    The recipe is handed to ``ModelTrainer.from_recipe(training_recipe=...)``.
    The SageMaker SDK derives the entry script (``examples/<model>/<model>_pretrain.py``)
    from ``recipe.model.model_type`` and runs it inside the managed DLC.

* ``use_training_recipe: true``  (container-native recipes, e.g. VERL)
    The recipe is NOT processed by the SDK. Instead the recipe file is shipped
    as the ``recipe`` input channel and the trained container's OWN entrypoint
    (e.g. VERL's ``docker_entrypoint.py``) reads it via the ``sagemaker_recipe_local_path``
    hyperparameter -- exactly the contract the container already implements for the
    K8s / SLURM launch paths.

    This path deliberately does NOT call ``from_recipe`` and passes NO ``source_code``,
    so the SDK never runs its NeMo GPU-script derivation
    (``sagemaker.modules.train.sm_recipes.utils._configure_gpu_args``) which, on
    sagemaker>=2.250.0 (post 2026-09-01), hard-requires ``recipe.model.model_type`` --
    a field VERL recipes correctly omit (they carry ``run.model_type: verl`` instead).
    That requirement is what breaks VERL SMJOBS submissions on both the estimator and
    the ModelTrainer ``from_recipe`` executors; routing around it here is the fix.
"""

import argparse
import logging
import os
import tempfile

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
    StoppingCondition,
    TensorBoardOutputConfig,
)
from sagemaker.modules.constants import (
    SM_RECIPE,
    SM_RECIPE_CONTAINER_PATH,
    SM_RECIPE_YAML,
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


def _build_input_data_config(sm_inputs):
    """Translate sm_jobs_config.inputs (s3 or file_system) into a ModelTrainer input list."""
    if not sm_inputs:
        return None

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

        return [
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

    s3_dict = OmegaConf.to_container(s3)
    input_data_config = []
    for channel_name, s3_uri in s3_dict.items():
        if s3_uri:
            input_data_config.append(InputData(channel_name=channel_name, data_source=s3_uri))

    return input_data_config or None


def _resolve_stopping_condition(additional_estimator_kwargs):
    """Honor an explicit stopping_condition, else map max_run -> StoppingCondition."""
    stopping_condition = additional_estimator_kwargs.pop("stopping_condition", None)
    max_run = additional_estimator_kwargs.pop("max_run", None)
    if stopping_condition is None and max_run is not None:
        stopping_condition = StoppingCondition(max_runtime_in_seconds=int(max_run))
    return stopping_condition


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

    input_data_config = _build_input_data_config(sm_jobs_config.get("inputs"))

    output_path = sm_jobs_config.get("output_path")
    if output_path is None:
        raise ValueError("Expected output_path to be set with sm_jobs cluster type")

    additional_estimator_kwargs = sm_jobs_config.get("additional_estimator_kwargs", omegaconf.DictConfig(dict()))
    additional_estimator_kwargs = OmegaConf.to_container(additional_estimator_kwargs)

    environment = sm_jobs_config.get("environment", omegaconf.DictConfig(dict()))
    environment = OmegaConf.to_container(environment)

    compute = Compute(instance_type=args.instance_type)

    disable_output_compression = additional_estimator_kwargs.pop("disable_output_compression", False)
    output_kms_key = additional_estimator_kwargs.pop("output_kms_key", None)
    output_config = OutputDataConfig(
        s3_output_path=output_path,
        compression_type="NONE" if disable_output_compression else None,
        kms_key_id=output_kms_key,
    )

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

    # use_training_recipe: when true the *container* consumes the recipe via its own
    # entrypoint (see module docstring); we must NOT route through from_recipe (which
    # trips the SDK's model.model_type requirement on sagemaker>=2.250.0).
    use_training_recipe = bool(additional_estimator_kwargs.pop("use_training_recipe", False))

    # Keep a reference to the temp dir alive until train() has uploaded the recipe channel.
    recipe_channel_dir = None

    if use_training_recipe:
        if not training_image:
            raise ValueError(
                "use_training_recipe=true requires a training_image / image_uri "
                "(the container that consumes the recipe via its own entrypoint)."
            )

        # Ship the fully-merged recipe (recipe + recipe_overrides) as the `recipe` input
        # channel; the container reads it from SM_RECIPE_CONTAINER_PATH via the
        # `sagemaker_recipe_local_path` hyperparameter (the same contract used elsewhere).
        recipe_channel_dir = tempfile.TemporaryDirectory(prefix="smtj_recipe_")
        recipe_file = os.path.join(recipe_channel_dir.name, SM_RECIPE_YAML)
        OmegaConf.save(config=recipe, f=recipe_file)

        input_data_config = list(input_data_config) if input_data_config else []
        # Drop any pre-existing `recipe` channel to avoid a duplicate.
        input_data_config = [c for c in input_data_config if getattr(c, "channel_name", None) != SM_RECIPE]
        input_data_config.append(InputData(channel_name=SM_RECIPE, data_source=recipe_file))

        hyperparameters = additional_estimator_kwargs.pop("hyperparameters", None) or {}
        hyperparameters["sagemaker_recipe_local_path"] = SM_RECIPE_CONTAINER_PATH

        instance_count = additional_estimator_kwargs.pop("instance_count", None)
        if instance_count is None:
            instance_count = OmegaConf.select(recipe, "trainer.num_nodes", default=1)
        compute = Compute(instance_type=args.instance_type, instance_count=int(instance_count))

        stopping_condition = _resolve_stopping_condition(additional_estimator_kwargs)

        # NOTE: no source_code / entry_script is passed. The SDK only overrides the
        # container entrypoint when source_code is provided; omitting it lets the
        # container's own ENTRYPOINT (docker_entrypoint.py) run.
        trainer = ModelTrainer(
            training_image=training_image,
            compute=compute,
            output_data_config=output_config,
            input_data_config=input_data_config,
            base_job_name=base_job_name,
            role=role,
            sagemaker_session=modules_session,
            networking=networking,
            stopping_condition=stopping_condition,
            training_image_config=additional_estimator_kwargs.pop("training_image_config", None),
            checkpoint_config=additional_estimator_kwargs.pop("checkpoint_config", None),
            training_input_mode=additional_estimator_kwargs.pop("training_input_mode", "File"),
            environment=environment if environment else None,
            hyperparameters=hyperparameters,
            tags=additional_estimator_kwargs.pop("tags", None),
        )
    else:
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
            stopping_condition=_resolve_stopping_condition(additional_estimator_kwargs),
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

    if recipe_channel_dir is not None:
        recipe_channel_dir.cleanup()


if __name__ == "__main__":
    main()
