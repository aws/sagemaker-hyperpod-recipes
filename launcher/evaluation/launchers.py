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

import copy
import json
import logging
import os
import re
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path

import boto3
import omegaconf
from botocore.exceptions import BotoCoreError, ClientError
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ..efa import (
    INSTANCE_TO_DEVICE_COUNT,
    efa_supported_instance,
    instanceWithMultipleEFAs,
)

logger = logging.getLogger(__name__)


def get_recipe_file_path():
    hydra_cfg = HydraConfig.get()

    # Look for recipes override in task overrides
    for override in hydra_cfg.overrides.task:
        if override.startswith("recipes="):
            return override.split("=", 1)[1]

    return None


def _is_efa_supported(instance_type):
    if instance_type is None:
        return False

    base_type = None
    if instance_type.startswith("ml."):
        base_type = instance_type[3:]
    else:
        base_type = instance_type
    return base_type in efa_supported_instance


def get_instance_type(cfg):
    instance_type = None
    with open("./launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json", "r") as f:
        regional_parameters = json.load(f)
        instance_map = regional_parameters.get("js_model_name_instance_mapping", {})
        base_model_name = cfg.recipes.run.get("base_model_name", "")
        instance_type = instance_map.get(base_model_name, ["ml.p5.48xlarge"])
        return instance_type[0]
    return "ml.p5.48xlarge"


def get_num_efa_devices(instance_type):
    if instance_type is None:
        return 0
    if instance_type.startswith("ml."):
        instance_type = instance_type[3:]

    if not _is_efa_supported(instance_type):
        return 0

    if instance_type in instanceWithMultipleEFAs:
        return instanceWithMultipleEFAs[instance_type]
    return 1


def get_device_count_for_instance(instance_type):
    if instance_type is None:
        return 8
    if instance_type.startswith("ml."):
        instance_type = instance_type[3:]

    return INSTANCE_TO_DEVICE_COUNT.get(instance_type, 8)


class EvaluationK8SLauncher:
    """
    Base class for Evaluation Kubernetes Launchers that provides common functionality for deploying evaluation jobs on K8s clusters.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._job_name = cfg.recipes.run["name"]
        self._output_dir = Path(cfg["base_results_dir"]) / self._job_name
        self._output_dir_k8s_folder = self._output_dir / "k8s_templates"
        self._launch_json = cfg["launch_json"]
        self.instance_type = get_instance_type(cfg)
        self.num_efa_devices = get_num_efa_devices(self.instance_type)
        self.recipe_file_path = None

    @staticmethod
    def _get_aws_account_id():
        """Returns the AWS account ID for the current credentials."""
        try:
            sts = boto3.client("sts")
            return sts.get_caller_identity()["Account"]
        except (BotoCoreError, ClientError) as e:
            print(f"Error retrieving AWS account ID: {e}")
            return None

    def _get_env_vars(self):
        """Returns a dictionary of environment variables to inject."""
        env_vars = {}

        if self._launch_json:
            env_vars["X_AMZ_SOURCE_ACCOUNT"] = "PLACEHOLDER_ACCOUNT_ID"
        else:
            account_id = self._get_aws_account_id()
            if account_id:
                env_vars["X_AMZ_SOURCE_ACCOUNT"] = account_id

        if self.instance_type:
            env_vars["INSTANCE_TYPE"] = self.instance_type

        return env_vars

    def _prepare_output_dir(self):
        if self._output_dir_k8s_folder.exists():
            shutil.rmtree(self._output_dir_k8s_folder)
        os.makedirs(self._output_dir_k8s_folder / "templates", exist_ok=True)
        logger.info(f"Prepared output directory at {self._output_dir_k8s_folder}")

    def _interpolate_hydra(self):
        def interpolate(cfg):
            if isinstance(cfg, DictConfig):
                for k, v in cfg.items():
                    cfg[k] = interpolate(v)
            elif isinstance(cfg, list):
                for i, v in enumerate(cfg):
                    cfg[i] = interpolate(v)
            return cfg

        interpolate(self.cfg.recipes)

    def _create_chart_file(self, template_dir):
        static_chart = template_dir / "Chart.yaml"
        if static_chart.exists():
            shutil.copyfile(static_chart, self._output_dir_k8s_folder / "Chart.yaml")

    def _write_value_template(self, values_template):
        """Write the value template into disk"""
        k8s_template_file = Path(self._output_dir_k8s_folder) / "values.yaml"
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to container to ensure proper serialization
        values_dict = OmegaConf.to_container(values_template, resolve=True)

        # Ensure numEFADevices is an integer in the final output
        if "evaluationConfig" in values_dict and "numEFADevices" in values_dict["evaluationConfig"]:
            efa_val = values_dict["evaluationConfig"]["numEFADevices"]
            if isinstance(efa_val, str):
                try:
                    values_dict["evaluationConfig"]["numEFADevices"] = int(efa_val)
                except (ValueError, TypeError):
                    values_dict["evaluationConfig"]["numEFADevices"] = 0

        conf = OmegaConf.create(values_dict)
        OmegaConf.save(conf, k8s_template_file)

    def _create_helm_script(self, chart_path: Path):
        script_path = self._output_dir / f"{self._job_name}_launch.sh"
        job_name = self._job_name.replace("_", "-")

        extra_helm_args = ""
        if self.cfg.cluster.get("namespace"):
            extra_helm_args += f" --namespace {self.cfg.cluster['namespace']}"

        helm_command = f"#!/bin/bash\n" f"helm install --timeout=15m {extra_helm_args} {job_name} {chart_path}\n"

        script_path.write_text(helm_command)
        script_path.chmod(0o755)
        logger.info(f"Helm script created at {script_path}")
        return script_path

    def _get_label_selectors(self):
        """Constructs and returns a dictionary of label selectors required for evaluation jobs."""

        if self._launch_json:
            instance_type_value = "PLACEHOLDER_INSTANCE_TYPE"
        else:
            instance_type_value = self.instance_type

        required_instances = {
            "node.kubernetes.io/instance-type": [instance_type_value],
        }

        label_selector = self.cfg.cluster.get("label_selector") or {}
        required_labels = label_selector.get("required") or {}
        return {
            **label_selector,
            "required": {**required_labels, **required_instances},
        }

    def _create_launch_json(self, chart_path):
        """Run `helm template [NAME] [CHART]` and dumps the output into a launch.json file"""
        job_name = self._job_name.replace("_", "-")

        render_dir = self._output_dir / "rendered_templates"
        render_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "helm",
            "template",
            job_name,
            str(chart_path),
            "--output-dir",
            str(render_dir),
        ]
        subprocess.run(cmd, check=True)

        launch_json = {}
        for path in sorted(render_dir.rglob("*.yaml"), key=lambda p: p.name):
            content = path.read_text()
            job_name_with_hyphens = self._job_name.replace("_", "-")
            content = content.replace(f"name: {self._job_name}", "name: {{name}}")
            content = content.replace(f"name: {job_name_with_hyphens}", "name: {{name}}")
            content = content.replace(f"evaluation-config-{self._job_name}", "evaluation-config-{{name}}")
            content = content.replace(f"evaluation-config-{job_name_with_hyphens}", "evaluation-config-{{name}}")
            content = content.replace(f"app: {self._job_name}", "app: {{name}}")
            content = content.replace(f"app: {job_name_with_hyphens}", "app: {{name}}")
            content = content.replace(f"value: {self._job_name}", "value: {{name}}")
            content = content.replace(f"value: {job_name_with_hyphens}", "value: {{name}}")
            content = re.sub(r"\breplicas:\s*\d+", "replicas: {{replicas}}", content)

            container_image = self.cfg.get("container", "")
            if container_image:
                content = content.replace(f"image: {container_image}", "image: {{container_image}}")

            if self.instance_type:
                content = content.replace(f'value: "{self.instance_type}"', 'value: "{{instance_type}}"')
                content = content.replace(f'- "{self.instance_type}"', '- "{{instance_type}}"')

            if self.cfg.cluster.get("namespace"):
                namespace_value = self.cfg.cluster["namespace"]
                content = content.replace(f"namespace: {namespace_value}", "namespace: {{namespace}}")

            if self._launch_json:
                content = content.replace('value: "PLACEHOLDER_INSTANCE_TYPE"', 'value: "{{instance_type}}"')
                content = content.replace('- "PLACEHOLDER_INSTANCE_TYPE"', '- "{{instance_type}}"')

            launch_json[path.name] = content

        # Add evaluation-specific metadata and regional parameters
        base_model_name = self.cfg.recipes.run.get("base_model_name", "")

        # Create metadata for evaluation recipe
        metadata = {
            "Name": base_model_name.replace("-", "_"),
            "Model_ID": base_model_name,
            "DisplayName": f"Open Source Evaluation - {base_model_name.replace('-', ' ').title()}",
            "Type": "Evaluation",
            "Hardware": "GPU",
            "InstanceTypes": [self.instance_type],
            "Versions": ["1.0"],
        }
        launch_json["metadata"] = metadata

        # Add regional parameters for evaluation
        with open("./launcher/recipe_templatization/evaluation/evaluation_regional_parameters.json", "r") as f:
            regional_params = json.load(f)

        recipe_container_mapping = regional_params.get("recipe_container_mapping", {})
        eval_recipe_config = recipe_container_mapping.get("open_source_deterministic_eval", {})

        regional_parameters = {}
        if self.cfg.cluster_type == "k8s":
            if "container_image" in eval_recipe_config:
                regional_parameters["container_image"] = eval_recipe_config["container_image"]
        elif self.cfg.cluster_type == "sm_jobs":
            if "smtj_container_image" in eval_recipe_config:
                regional_parameters["container_image"] = eval_recipe_config["smtj_container_image"]

        launch_json["regional_parameters"] = regional_parameters

        # Add recipe override parameters
        recipe_override_parameters = {
            "name": {"type": "string", "default": self._job_name, "required": True},
            "base_model_name": {"type": "string", "default": base_model_name, "required": True},
            "model_name_or_path": {"type": "string", "default": "", "required": True},
            "data_s3_path": {"type": "string", "default": "", "required": True},
            "eval_results_dir": {"type": "string", "default": "", "required": True},
            "task": {"type": "string", "default": "mmlu", "required": True},
            "strategy": {"type": "string", "default": "zs_cot", "required": False},
            "metric": {"type": "string", "default": "accuracy", "required": False},
            "subtask": {"type": "string", "default": "abstract_algebra", "required": False},
        }
        launch_json["recipe_override_parameters"] = recipe_override_parameters

        # Add training recipe
        untemplated_recipe_file_path = get_recipe_file_path()
        if untemplated_recipe_file_path:
            recipe_path = untemplated_recipe_file_path
            if recipe_path.endswith(".yaml"):
                recipe_path = recipe_path[:-5]
            full_recipe_path = os.path.join("./recipes_collection/recipes", recipe_path + ".yaml")
            untemplated_recipe = OmegaConf.load(full_recipe_path)
            wrapped_recipe = OmegaConf.create({"recipes": untemplated_recipe})
            recipe_container = OmegaConf.to_container(wrapped_recipe, resolve=True)
            launch_json["training_recipe.json"] = recipe_container
        else:
            launch_json["training_recipe.json"] = {"recipes": {}}

        launch_path = self._output_dir / "launch.json"
        with open(launch_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=True)
            f.write("\n")

        logger.info(f"Helm templates dumped into {launch_path}")

    @staticmethod
    def _run_helm_script(script_path: Path):
        logger.info(f"Running Helm script: {script_path}")
        subprocess.Popen(str(script_path)).wait()

    @abstractmethod
    def _save_hydra_config(self):
        pass

    @abstractmethod
    def run(self):
        """Generate a Helm-installable k8s_template directory."""

    @abstractmethod
    def _copy_k8s_template(self):
        """copy helm files in k8s_template directory."""

    @abstractmethod
    def _process_values_yaml(self):
        """Generate values based on evaluation type for values.yaml file."""


class SMEvaluationK8SLauncher(EvaluationK8SLauncher):
    """
    Launcher for Evaluation jobs on Kubernetes that handles deployment of evaluation jobs with proper configuration.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._template_dir = Path(__file__).parent / "k8s_templates/EVAL"

    def run(self):
        self._prepare_output_dir()
        self._save_hydra_config()
        self._create_chart_file(self._template_dir)
        self._copy_k8s_template()
        self._process_values_yaml()
        script_path = self._create_helm_script(self._output_dir_k8s_folder)

        if self._launch_json:
            self._create_launch_json(self._output_dir_k8s_folder)
            if self.cfg.cluster_type != "k8s":
                logger.info(
                    "Evaluation recipes are only available for EKS clusters. Please update the config file to use k8s."
                )
                return
        else:
            if self.cfg.cluster_type != "k8s":
                raise ValueError(
                    "Evaluation recipes are only available for EKS clusters. Please update the config file to use k8s."
                )

            self._run_helm_script(script_path)
            logger.info(f"Launcher successfully generated: {self._template_dir}")

    def _save_hydra_config(self):
        self._interpolate_hydra()
        config_path = Path(self._output_dir_k8s_folder / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        config_file = Path(config_path / f"{self._job_name}_hydra.yaml")
        self.recipe_file_path = config_file
        omegaconf.OmegaConf.save(self.cfg.recipes, config_file)

    def _copy_k8s_template(self):
        for fname in ["evaluation.yaml", "evaluation-config.yaml"]:
            src = self._template_dir / fname
            dst = self._output_dir_k8s_folder / "templates" / fname
            shutil.copyfile(src, dst)

    def _process_values_yaml(self):
        with open(self._template_dir / "values.yaml") as value_file:
            values_template = OmegaConf.load(value_file)

        cluster_cfg = copy.deepcopy(self.cfg.get("cluster") or {})
        k8s_cfg = {**cluster_cfg}

        if self._launch_json:
            k8s_cfg["custom_labels"] = {"placeholder": "custom_labels"}
            k8s_cfg["service_account_name"] = "placeholder_service_account_name"

        values_template.image.evaluationImage = self.cfg.get("container")
        values_template.evaluationConfig.jobName = self._job_name
        values_template.evaluationConfig.envVars = self._get_env_vars()
        # Ensure numEFADevices is always a valid integer
        efa_devices = self.num_efa_devices
        if efa_devices is None:
            efa_devices = 0
        elif not isinstance(efa_devices, int):
            try:
                efa_devices = int(efa_devices)
            except (ValueError, TypeError):
                efa_devices = 0

        values_template.evaluationConfig.numEFADevices = efa_devices

        values_template.evaluationConfig.devices = OmegaConf.select(
            self.cfg, "recipes.resource_config.devices"
        ) or get_device_count_for_instance(self.instance_type)

        values_template.evaluationConfig.replicas = OmegaConf.select(self.cfg, "recipes.run.replicas", default=1)

        optional_fields = {
            "namespace": "namespace",
            "annotations": "annotations",
            "priority_class_name": "priorityClassName",
            "service_account_name": "serviceAccountName",
            "custom_labels": "customLabels",
        }

        for src, dest in optional_fields.items():
            val = k8s_cfg.get(src)
            if val is not None:
                setattr(values_template.evaluationConfig, dest, val)

        values_template.evaluationConfig.labelSelector = self._get_label_selectors()
        self._write_value_template(values_template)
