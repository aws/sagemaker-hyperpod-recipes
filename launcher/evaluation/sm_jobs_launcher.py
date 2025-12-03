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

import json
import logging
import os
import subprocess
from pathlib import Path

import omegaconf
from omegaconf import OmegaConf

from ..recipe_templatization.evaluation.evaluation_recipe_template_processor import (
    EvaluationRecipeTemplateProcessor,
)
from .launchers import EvaluationK8SLauncher, get_recipe_file_path

logger = logging.getLogger(__name__)


class SMEvaluationJobsLauncher(EvaluationK8SLauncher):
    """
    Launcher for Evaluation jobs on SageMaker Training Jobs that handles deployment of evaluation jobs.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._template_dir = Path(__file__).parent / "sm_jobs_templates"

    def run(self):
        self._prepare_output_dir()
        self._save_hydra_config()
        self._create_sm_jobs_template()
        self._process_sm_jobs_config()
        script_path = self._create_sm_jobs_script()

        if self._launch_json:
            self._create_launch_json_smtj()
        else:
            self._run_sm_jobs_script(script_path)
            logger.info(f"SageMaker Training Job launcher successfully generated")

    def _create_sm_jobs_template(self):
        """Create SageMaker Training Jobs template directory and files."""
        sm_jobs_dir = self._output_dir / "sm_jobs_template"
        sm_jobs_dir.mkdir(parents=True, exist_ok=True)

        # Create the SM Jobs Python script
        sm_jobs_script = sm_jobs_dir / "sm_jobs_launcher.py"
        template_content = self._get_sm_jobs_template_content()
        sm_jobs_script.write_text(template_content)

        logger.info(f"SageMaker Training Jobs template created at {sm_jobs_dir}")

    def _get_sm_jobs_template_content(self) -> str:
        """Generate the SageMaker Training Jobs launcher script content."""
        return """#!/usr/bin/env python3
import argparse
import logging
import os

import omegaconf
import sagemaker
from omegaconf import OmegaConf
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import FileSystemInput
from sagemaker.pytorch import PyTorch

logger = logging.getLogger(__name__)


def parse_args():
    script_dir = os.path.dirname(os.path.join(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(description="Launch evaluation recipe using SM jobs")
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
    parser.add_argument("--instance_type", type=str, required=True, help="Instance type to use for the evaluation job.")
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

    if not isinstance(inputs, FileSystemInput):
        keys_to_pop = []
        for item in inputs.keys():
            if not inputs[item]:
                print(f"popping input {inputs[item]}, {item}")
                keys_to_pop.append(item)
        for item in keys_to_pop:
            inputs.pop(item)
        if len(inputs) == 0:
            inputs = None

    estimator.fit(inputs=inputs, wait=sm_jobs_config.get("wait", False))


if __name__ == "__main__":
    main()
"""

    def _process_sm_jobs_config(self):
        """Generate SM Jobs configuration file."""
        sm_jobs_config = {
            "recipe_overrides": {},
            "inputs": {"s3": {}},
            "output_path": "s3://your-bucket/evaluation-results",
            "wait": False,
            "additional_estimator_kwargs": {
                "instance_count": 1,
                "volume_size": 100,
                "max_run": 86400,  # 24 hours
                "keep_alive_period_in_seconds": 0,
                "container_log_level": logging.INFO,
                "source_dir": None,
                "entry_point": None,
                "dependencies": None,
                "code_location": None,
                "git_config": None,
                "checkpoint_s3_uri": None,
                "checkpoint_local_path": None,
                "enable_cloudwatch_metrics": False,
                "enable_sagemaker_metrics": False,
                "enable_network_isolation": False,
                "use_spot_instances": False,
                "max_wait": None,
                "spot_instance_type": None,
                "tags": None,
                "subnets": None,
                "security_group_ids": None,
                "model_uri": None,
                "model_channel_name": "model",
                "metric_definitions": None,
                "encrypt_inter_container_traffic": False,
                "use_compiled_model": False,
                "train_use_spot_instances": False,
                "train_max_wait": None,
                "train_spot_instance_type": None,
                "train_volume_kms_key": None,
                "train_vpc_config": None,
                "train_max_run": None,
            },
        }

        # Add evaluation-specific configurations
        if hasattr(self.cfg, "recipes") and hasattr(self.cfg.recipes, "run"):
            if hasattr(self.cfg.recipes.run, "model_name_or_path"):
                sm_jobs_config["inputs"]["s3"]["model"] = self.cfg.recipes.run.model_name_or_path
            if hasattr(self.cfg.recipes.run, "data_s3_path"):
                sm_jobs_config["inputs"]["s3"]["data"] = self.cfg.recipes.run.data_s3_path

        # Write SM Jobs config
        sm_jobs_config_path = self._output_dir / "sm_jobs_template" / "sm_jobs_config.yaml"
        OmegaConf.save(OmegaConf.create(sm_jobs_config), sm_jobs_config_path)

    def _create_sm_jobs_script(self) -> Path:
        """Create the main launcher script for SM Jobs."""
        script_path = self._output_dir / f"{self._job_name}_sm_jobs_launch.sh"

        sm_jobs_dir = self._output_dir / "sm_jobs_template"
        recipe_path = sm_jobs_dir / f"{self._job_name}_hydra.yaml"
        config_path = sm_jobs_dir / "sm_jobs_config.yaml"
        launcher_path = sm_jobs_dir / "sm_jobs_launcher.py"

        instance_type = self.instance_type or "ml.p5.48xlarge"

        script_content = f"""#!/bin/bash

# SageMaker Training Jobs launcher for evaluation
echo "Launching evaluation job: {self._job_name}"
echo "Instance type: {instance_type}"
echo "Recipe path: {recipe_path}"
echo "Config path: {config_path}"

python3 {launcher_path} \\
    --recipe {recipe_path} \\
    --sm_jobs_config {config_path} \\
    --job_name {self._job_name} \\
    --instance_type {instance_type}

echo "Job {self._job_name} submission file created at '{script_path}'"
"""

        script_path.write_text(script_content)
        script_path.chmod(0o755)
        logger.info(f"SM Jobs script created at {script_path}")
        return script_path

    def _create_launch_json_smtj(self):
        """Create launch.json for SMTJ with templatization support."""
        launch_json = {}
        recipe_file_path = get_recipe_file_path()
        # Add the SM Jobs launcher script
        launcher_script_content = self._get_sm_jobs_template_content()
        launch_json["sm_jobs_launcher.py"] = launcher_script_content

        # Add SM Jobs config template
        sm_jobs_config_template = {
            "recipe_overrides": {},
            "output_path": "{{output_path}}",
            "wait": False,
            "additional_estimator_kwargs": {"instance_count": 1, "volume_size": 100, "max_run": 86400},
        }

        if recipe_file_path is not None and "open_source_deterministic_eval" in recipe_file_path:
            sm_jobs_config_template["inputs"] = {"s3": {"model": "{{model_name_or_path}}", "data": "{{data_s3_path}}"}}

        launch_json["sm_jobs_config.yaml"] = OmegaConf.to_yaml(OmegaConf.create(sm_jobs_config_template))

        # Add the main launcher script template
        launcher_script_template = f"""#!/bin/bash

# SageMaker Training Jobs launcher for evaluation
echo "Launching evaluation job: {{{{name}}}}"
echo "Instance type: {{{{instance_type}}}}"

python3 sm_jobs_launcher.py \\
    --recipe {{{{name}}}}_hydra.yaml \\
    --sm_jobs_config sm_jobs_config.yaml \\
    --job_name {{{{name}}}} \\
    --instance_type {{{{instance_type}}}}

echo "Job {{{{name}}}} submission file created"
"""
        launch_json[f"{self._job_name}_sm_jobs_launch.sh"] = launcher_script_template

        # Add recipe templatization metadata if available
        if hasattr(self, "_recipe_template_processor") and self._recipe_template_processor is not None:
            try:
                recipe_file_path = get_recipe_file_path()
                if recipe_file_path is not None:
                    additional_data = self._recipe_template_processor.get_additional_data(recipe_file_path, self.cfg)
                    if additional_data is not None:
                        (
                            launch_json["metadata"],
                            launch_json["recipe_override_parameters"],
                            launch_json["regional_parameters"],
                        ) = additional_data
                        logger.info("Added recipe templatization metadata to launch.json")
            except Exception as e:
                logger.warning(f"Failed to add recipe templatization metadata: {e}")
                # Add empty metadata for testing
                launch_json["metadata"] = {}
                launch_json["recipe_override_parameters"] = {}
                launch_json["regional_parameters"] = {}

        # Add untemplated recipe
        try:
            untemplated_recipe_file_path = get_recipe_file_path()
            if untemplated_recipe_file_path:
                full_recipe_path = os.path.join("./recipes_collection/recipes", untemplated_recipe_file_path + ".yaml")
                untemplated_recipe = OmegaConf.load(full_recipe_path)
                wrapped_recipe = OmegaConf.create({"recipes": untemplated_recipe})
                recipe_container = OmegaConf.to_container(wrapped_recipe, resolve=True)
                launch_json["training_recipe.json"] = recipe_container
            else:
                launch_json["training_recipe.json"] = {"recipes": {}}
        except Exception:
            # Hydra not initialized, use current config
            recipe_container = OmegaConf.to_container(self.cfg, resolve=True)
            launch_json["training_recipe.json"] = recipe_container

        # Add templated recipe YAML
        with open(self.recipe_file_path, "r") as f:
            recipe_yaml_content = f.read()
        launch_json["training_recipe.yaml"] = recipe_yaml_content

        # Write launch.json
        launch_path = self._output_dir / "launch.json"
        with open(launch_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=True)
            f.write("\n")

        logger.info(f"SMTJ launch.json created at {launch_path}")

    def _run_sm_jobs_script(self, script_path: Path):
        """Execute the SM Jobs launcher script."""
        logger.info(f"Running SM Jobs script: {script_path}")
        subprocess.Popen(str(script_path)).wait()

    def _save_hydra_config(self):
        """Save Hydra configuration with recipe templatization for SMTJ."""
        self._interpolate_hydra()
        config_path = Path(self._output_dir / "sm_jobs_template")
        config_path.mkdir(parents=True, exist_ok=True)
        config_file = Path(config_path / f"{self._job_name}_hydra.yaml")
        self.recipe_file_path = config_file

        # Apply recipe templatization if launch_json is enabled
        if self._launch_json:
            try:
                recipe_file_path = get_recipe_file_path()
                if recipe_file_path is None:
                    raise KeyError("Recipe file path not found in hydra")
                else:
                    # Create recipe template processor for evaluation recipes with sm_jobs platform
                    recipe_template_processor = EvaluationRecipeTemplateProcessor(self.cfg.recipes, platform="sm_jobs")
                    templatized_recipe = recipe_template_processor.process_recipe(recipe_file_path)
                    omegaconf.OmegaConf.save(templatized_recipe, config_file)
                    self._recipe_template_processor = recipe_template_processor
            except Exception as e:
                logger.warning(f"Recipe templatization failed, using original recipe: {e}")
                omegaconf.OmegaConf.save(self.cfg.recipes, config_file)
                self._recipe_template_processor = None
        else:
            omegaconf.OmegaConf.save(self.cfg.recipes, config_file)
            self._recipe_template_processor = None

    def _copy_k8s_template(self):
        """Not needed for SM Jobs launcher."""

    def _process_values_yaml(self):
        """Not needed for SM Jobs launcher."""
