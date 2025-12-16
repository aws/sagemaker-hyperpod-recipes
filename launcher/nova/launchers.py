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
from typing import Dict, List

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
from ..recipe_templatization.nova.nova_recipe_template_processor import (
    NovaRecipeTemplateProcessor,
)
from .constants.ppo_container_constants import (
    JOB_TASK_TYPE_DICT,
    JOB_TYPE_DICT,
    KEYS_TO_REMOVE,
    JobType,
)
from .constants.rft_container_constants import (
    RFT_JOB_TASK_TYPE_DICT,
    RFT_KEYS_TO_REMOVE,
    RFT_PROMPT_RBS_TASK_TYPES,
    RFTJobType,
)
from .utils import (
    get_actor_generation_container_uri,
    get_init_container_uri,
    get_rft_generation_container_uri,
    get_rft_nats_reloader_container_uri,
    get_rft_nats_server_container_uri,
    get_rft_redis_container_uri,
    get_rft_storm_container_uri,
    get_rft_train_container_uri,
)

logger = logging.getLogger(__name__)

DONT_OVERRIDE_IN_SMTJ = {"replicas", "mlflow_tracking_uri", "mlflow_experiment_name", "mlflow_run_name"}
EFA_RESOURCE_KEY = "vpc.amazonaws.com/efa"


# Get the original recipe path
def get_recipe_file_path():
    hydra_cfg = HydraConfig.get()
    recipe_path = None
    # Check if recipes override was used
    recipe_override_index = -1
    if hydra_cfg.overrides.task and len(hydra_cfg.overrides.task) > 0:
        for index in range(len(hydra_cfg.overrides.task)):
            if "recipes" in hydra_cfg.overrides.task[index]:
                recipe_override_index = index
                break

    if recipe_override_index != -1:
        recipe_path = hydra_cfg.overrides.task[recipe_override_index].split("=")[1]
        return recipe_path

    else:
        raise KeyError("Recipe file path not found in hydra config")


def get_recipe_name_from_path(recipe_file_path: str) -> str:
    """
    Extract recipe name from file path with null checks.

    :param recipe_file_path: Path to the recipe file
    :return: Recipe name without .yaml extension
    :raises ValueError: If recipe_file_path is None or empty
    """
    if not recipe_file_path:
        raise ValueError("Recipe file path cannot be None or empty")

    # Handle both full paths and just filenames
    recipe_name = recipe_file_path.split("/")[-1] if "/" in recipe_file_path else recipe_file_path

    # Remove .yaml extension if present
    if recipe_name.endswith(".yaml"):
        recipe_name = recipe_name[:-5]

    if not recipe_name:
        raise ValueError(f"Invalid recipe file path - no filename found for recipe_file_path: {recipe_file_path}")

    return recipe_name


def get_nova_metadata():
    recipe_name = get_recipe_name_from_path(get_recipe_file_path())
    nova_metadata_path = "./launcher/recipe_templatization/nova/nova_metadata.json"
    if not os.path.exists(nova_metadata_path):
        raise ValueError(f"Nova metadata not found at {nova_metadata_path}")
    nova_metadata = None
    with open(nova_metadata_path, "r") as f:
        nova_metadata = json.load(f)

    if not isinstance(nova_metadata, dict) or nova_metadata == None:
        raise ValueError(f"Invalid nova metadata at {nova_metadata_path}")

    if nova_metadata.get(recipe_name, None) == None:
        raise KeyError(f"Recipe {recipe_name} not found in nova metadata at {nova_metadata_path}")
    return nova_metadata[recipe_name]


def _is_efa_supported(instance_type):
    if instance_type is None:
        return False

    base_type = None
    # Remove ml. prefix if present to get base instance type
    if instance_type.startswith("ml."):
        base_type = instance_type[3:]
    else:
        base_type = instance_type
    return base_type in efa_supported_instance


def get_instance_type(cfg):
    instance_type = None

    if cfg.get("instance_type"):
        instance_type = cfg.instance_type
    else:
        # custom path - check cluster config
        if hasattr(cfg, "cluster") and cfg.cluster:
            instance_type = cfg.cluster.get("instance_type")

    if instance_type is None:
        instance_type = "p5.48xlarge"

    # Add ml. prefix if not present as it is not expected as per customer contract.
    if not instance_type.startswith("ml."):
        instance_type = f"ml.{instance_type}"

    if not _is_efa_supported(instance_type):
        return None

    return instance_type.lower()


def get_num_efa_devices(instance_type):
    if instance_type is None:
        return 0
    if instance_type.startswith("ml."):
        instance_type = instance_type[3:]

    # If not a EFA instance, return 0
    if not _is_efa_supported(instance_type):
        return 0

    # If multi-EFA, return from mapping
    if instance_type in instanceWithMultipleEFAs:
        return instanceWithMultipleEFAs[instance_type]
    # Only a single EFA device
    return 1


def get_device_count_for_instance(instance_type):
    if instance_type is None:
        return 8
    if instance_type.startswith("ml."):
        instance_type = instance_type[3:]

    return INSTANCE_TO_DEVICE_COUNT.get(instance_type, 8)


def templatize_K8_container_images(content, recipe_name):
    # Init container for nova sft/dpo recipes
    init_container_image = get_init_container_uri()
    content = content.replace(init_container_image, "{{init_container_image}}")

    # Actor generation container for Nova PPO recipes
    if "ppo" in recipe_name:
        actor_generation_container_uri = get_actor_generation_container_uri()
        content = content.replace(actor_generation_container_uri, "{{actor_generation_container_image}}")

    # # Containers for Nova RFT recipes
    if "rft" in recipe_name:
        rft_generation_container_image = get_rft_generation_container_uri()
        rft_storm_container_image = get_rft_storm_container_uri()
        rft_nats_server_container_image = get_rft_nats_server_container_uri()
        rft_nats_reloader_container_image = get_rft_nats_reloader_container_uri()
        rft_redis_container_image = get_rft_redis_container_uri()
        rft_train_container_image = get_rft_train_container_uri()
        rft_containers = {
            rft_generation_container_image: "{{rft_generation_container_image}}",
            rft_storm_container_image: "{{rft_storm_container_image}}",
            rft_nats_server_container_image: "{{rft_nats_server_container_image}}",
            rft_nats_reloader_container_image: "{{rft_nats_reloader_container_image}}",
            rft_redis_container_image: "{{rft_redis_container_image}}",
            rft_train_container_image: "{{container_image}}",
        }
        for container_placeholder in rft_containers:
            content = content.replace(container_placeholder, rft_containers[container_placeholder])

    return content


class NovaK8SLauncher:
    """
    Base class for Nova Kubernetes Launchers that provides common functionality for deploying Nova jobs on K8s clusters, handling AWS account integration, environment variables, and Helm chart generation.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._job_name = cfg.recipes.run["name"]
        self._output_dir = Path(cfg["base_results_dir"]) / self._job_name
        self._output_dir_k8s_folder = self._output_dir / "k8s_templates"
        # Try to get the region using boto3 session or env var
        self._init_container_uri = get_init_container_uri()
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

        # For launch_json generation we dont need X_AMZ_SOURCE_ACCOUNT
        if not self._launch_json:
            # For regular deployments, dynamically retrieve the account ID
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
        """
        Write the value template into disk
        """
        k8s_template_file = Path(self._output_dir_k8s_folder) / "values.yaml"
        k8s_template_file.parent.mkdir(parents=True, exist_ok=True)

        conf = OmegaConf.create(values_template)
        # Save using OmegaConf to maintain consistent formatting
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
        """
        Constructs and returns a dictionary of label selectors required for Nova jobs.

        This method ensures that the returned label selectors always include the required
        instance types and instance group types necessary for Nova jobs to run on the
        appropriate hardware. It merges any user-provided required label selectors from
        the configuration with the hardcoded required labels.

        Returns:
            dict: A dictionary containing the merged label selectors, with the "required"
            key including both user-specified and mandatory labels.
        """

        # Use placeholder for launch_json mode, actual instance type otherwise
        if self._launch_json:
            instance_type_value = "PLACEHOLDER_INSTANCE_TYPE"
        else:
            instance_type_value = self.instance_type

        # Default instance types for required labels
        # Nova jobs cannot be run on any other instance types apart from these
        # This is a hard requirement for the Nova jobs to run
        # on the required hardware.
        required_instances = {
            "node.kubernetes.io/instance-type": [instance_type_value],
            "sagemaker.amazonaws.com/instance-group-type": ["Restricted"],
        }

        # Handle labelSelector merging safely
        label_selector = self.cfg.cluster.get("label_selector") or {}
        required_labels = label_selector.get("required") or {}
        return {
            **label_selector,
            "required": {**required_labels, **required_instances},
        }

    def _create_launch_json(self, chart_path):
        """
        Run `helm template [NAME] [CHART]` and dumps the output into a launch.json file
        containing the content of the launch artifacts.

        Example structure of launch.json:
        {
            "template_file_1.yaml": "<CONTENT OF template_file_1.yaml>",
            "template_file_2.yaml": "<CONTENT OF template_file_2.yaml>",
            "metadata": {...},
            "recipe_override_parameters": {...},
            "regional_parameters": {...}
        }
        """
        job_name = self._job_name.replace("_", "-")

        # Use helm template for processing (works now that customLabels/serviceAccountName are null)
        render_dir = self._output_dir / "rendered_templates"
        render_dir.mkdir(parents=True, exist_ok=True)
        # Run helm template and capture output
        cmd = [
            "helm",
            "template",
            job_name,
            str(chart_path),
            "--output-dir",
            str(render_dir),
        ]
        subprocess.run(cmd, check=True)
        recipe_file_path = get_recipe_file_path()

        # Walk rendered files and dump into JSON
        # A lot of string replacement is happening here because of the recipe templatization process.
        # Templatizing some of the params before helm render causes the render process to fail. Therefore updating them
        # post the render process.
        launch_json = {}
        for path in sorted(render_dir.rglob("*.yaml"), key=lambda p: p.name):
            content = path.read_text()
            # Now do string replacements for names, namespaces, etc.
            # Replace following references in content. String replace is followed here instead of templatization because
            # for these values the helm rendering process fails if we have templates
            job_name_with_hyphens = self._job_name.replace("_", "-")
            content = content.replace(f"name: {self._job_name}", "name: {{name}}")
            content = content.replace(self._job_name, "{{name}}")
            content = content.replace(f"name: {job_name_with_hyphens}", "name: {{name}}")
            content = content.replace(f"training-config-{self._job_name}", "training-config-{{name}}")
            content = content.replace(f"training-config-{job_name_with_hyphens}", "training-config-{{name}}")
            content = content.replace(f"app: {self._job_name}", "app: {{name}}")
            content = content.replace(f"app: {job_name_with_hyphens}", "app: {{name}}")
            content = content.replace(f"value: {self._job_name}", "value: {{name}}")
            content = content.replace(f"value: {job_name_with_hyphens}", "value: {{name}}")
            if self.cfg.cluster.get("namespace"):
                namespace_value = self.cfg.cluster["namespace"]
                content = content.replace(f"namespace: {namespace_value}", "namespace: {{namespace}}")
            # Fix: Use re.sub() for regex pattern, not str.replace()
            content = re.sub(r"namespace:\s+default", "namespace: {{namespace}}", content)
            # Handle any other namespace values (not just "default"), including quoted values
            # Pattern ensures it stops at whitespace/newline and properly handles matching quotes
            content = re.sub(r'namespace:\s+(["\']?)([\w-]+)\1(?=\s|$)', "namespace: {{namespace}}", content)
            content = content.replace("namespace_placeholder", "{{namespace}}")
            # Replace container image references with parameterized values
            container_image = self.cfg.get("container", "")
            if container_image:
                content = content.replace(f"image: {container_image}", "image: {{container_image}}")

            # Replace instance type references with template variable
            if self.instance_type:
                content = content.replace(f'value: "{self.instance_type}"', 'value: "{{instance_type}}"')
                # Also replace instance type in node affinity values
                content = content.replace(f'- "{self.instance_type}"', '- "{{instance_type}}"')

            # Replicas should be overridable only in the non-eval nova recipes or HP platform. Only actor train replicas should be overridable in the ppo recipe and it is handled in SMNovaK8SLauncherPPO
            if "eval" not in recipe_file_path and self.cfg.cluster != "sm_jobs" and self.cfg.cluster_type != "sm_jobs":
                content = content.replace("replicas_placeholder", "{{replicas}}")

            # Replace placeholder values with template variables when launch_json is enabled
            if self._launch_json:
                content = content.replace('value: "PLACEHOLDER_INSTANCE_TYPE"', 'value: "{{instance_type}}"')
                content = content.replace('- "PLACEHOLDER_INSTANCE_TYPE"', '- "{{instance_type}}"')

            # update container image uris
            content = templatize_K8_container_images(content, get_recipe_name_from_path(recipe_file_path))
            launch_json[path.name] = content

        # # Add recipe templatization metadata if available
        if hasattr(self, "_recipe_template_processor") and self._recipe_template_processor is not None:
            try:
                # Generate recipe file path from job name for metadata generation
                # Nova recipes are typically in fine-tuning/nova/ directory
                recipe_file_path = get_recipe_file_path()

                # If recipe_file_path is None, we can't generate metadata
                if recipe_file_path is None:
                    logger.error("Could not get recipe file path from Hydra config. Metadata will not be added.")
                    raise KeyError("Metadata generation error. Recipe File Path not found")
                else:
                    # The Nova recipe template processor expects the path without .yaml extension
                    # and adds it internally, so we don't need to add it here
                    additional_data = self._recipe_template_processor.get_additional_data(recipe_file_path)

                    # Check if container is available for this platform
                    if additional_data is None:
                        logger.error("Recipe templatization failed because additional_data was returned as None")
                        return

                    (
                        launch_json["metadata"],
                        launch_json["recipe_override_parameters"],
                        launch_json["regional_parameters"],
                    ) = additional_data

                    logger.info("Added recipe templatization metadata to launch.json")
            except Exception as e:
                logger.error(f"Failed to add recipe templatization metadata: {e}")
                raise Exception

        untemplated_recipe_file_path = get_recipe_file_path()
        if untemplated_recipe_file_path:
            full_recipe_path = os.path.join("./recipes_collection/recipes", untemplated_recipe_file_path + ".yaml")
            untemplated_recipe = OmegaConf.load(full_recipe_path)
            wrapped_recipe = OmegaConf.create({"recipes": untemplated_recipe})
            recipe_container = OmegaConf.to_container(wrapped_recipe, resolve=True)
            launch_json["training_recipe.json"] = recipe_container
        else:
            launch_json["training_recipe.json"] = {"recipes": {}}

        # SMTJ specific changes:- Right now hyperpod-recipes does not support launching jobs for smtj so we do not
        # have a separate code path for it. Handling smtj launch json generation as a part of K8's. SMTJ update will
        # be tracked using this ticket:- https://taskei.amazon.dev/tasks/P310302663
        if self.cfg.cluster == "sm_jobs" or self.cfg.cluster_type == "sm_jobs":
            # Remove k8 related config files for smtj scenario
            if "training.yaml" in launch_json:
                del launch_json["training.yaml"]
            if "training-config.yaml" in launch_json:
                del launch_json["training-config.yaml"]

            with open(self.recipe_file_path, "r") as f:
                recipe_yaml_content = f.read()

            launch_json["training_recipe.yaml"] = recipe_yaml_content

        launch_path = self._output_dir / "launch.json"
        with open(launch_path, "w") as f:
            json.dump(launch_json, f, indent=2, sort_keys=False)
            f.write("\n")

        logger.info(f"Helm templates dumped into {launch_path}")

    @staticmethod
    def _run_helm_script(script_path: Path):
        logger.info(f"Running Helm script: {script_path}")
        subprocess.Popen(str(script_path)).wait()

    @abstractmethod
    def _save_hydra_config(self):
        pass  # pragma: no cover

    @abstractmethod
    def run(self):
        """Generate a Helm-installable k8s_template directory."""
        pass  # pragma: no cover

    @abstractmethod
    def _copy_k8s_template(self):
        """copy helm files in k8s_template directory."""
        pass  # pragma: no cover

    @abstractmethod
    def _process_values_yaml(self):
        """Generate values based on training type for values.yaml file."""
        pass  # pragma: no cover


class SMNovaK8SLauncherSFT(NovaK8SLauncher):
    """
    Launcher for Supervised Fine-Tuning (SFT) jobs on Kubernetes that handles deployment of SFT training jobs with proper configuration, container setup, and node allocation.
    This launcher also acts as the default Nova launcher and can handle other job types not in a dedicated class such as Eval.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._template_dir = Path(__file__).parent / "k8s_templates/SFT"

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
                    "Nova recipes are only available for EKS clusters. Please update the config file to use k8s."
                )
                return
        else:
            if self.cfg.cluster_type != "k8s":
                raise ValueError(
                    "Nova recipes are only available for EKS clusters. Please update the config file to use k8s."
                )

            self._run_helm_script(script_path)
            logger.info(f"Launcher successfully generated: {self._template_dir}")

    def _save_hydra_config(self):
        # IMPORTANT: resolve all interpolations in recipes before _interpolate_hydra,
        # because _interpolate_hydra mutates the tree and can break oc.select paths
        # like ${oc.select:training_config.max_steps}, causing them to become null.
        # For now, placing it under the not self._launch_json condition to be safe
        # and ensure it does not impact the JS or the user experience.
        if not self._launch_json:
            recipes_conf = OmegaConf.create(self.cfg.recipes)
            OmegaConf.resolve(recipes_conf)
            self.cfg.recipes = recipes_conf

        self._interpolate_hydra()
        config_path = Path(self._output_dir_k8s_folder / "config")
        config_path.mkdir(parents=True, exist_ok=True)
        config_file = Path(config_path / f"{self._job_name}_hydra.yaml")
        self.recipe_file_path = config_file

        # Apply recipe templatization if launch_json is enabled
        if self._launch_json:
            try:
                # Get the recipe file path from Hydra config
                recipe_file_path = get_recipe_file_path()
                if recipe_file_path is None:
                    raise KeyError("Recipe file path not found in hydra")
                else:
                    # Create recipe template processor for Nova recipes
                    recipe_template_processor = NovaRecipeTemplateProcessor(
                        self.cfg.recipes, platform=self.cfg.cluster_type
                    )
                    # Pass the recipe file path to process_recipe
                    templatized_recipe = None
                    if self.cfg.cluster_type == "sm_jobs" or self.cfg.cluster == "sm_jobs":
                        templatized_recipe = recipe_template_processor.process_recipe(
                            recipe_file_path, dont_override_list=DONT_OVERRIDE_IN_SMTJ
                        )
                    else:
                        templatized_recipe = recipe_template_processor.process_recipe(recipe_file_path)
                    omegaconf.OmegaConf.save(templatized_recipe, config_file)
                    # Store the processor for later use in launch_json creation
                    self._recipe_template_processor = recipe_template_processor
            except Exception as e:
                logger.warning(f"Recipe templatization failed, using original recipe: {e}")
                omegaconf.OmegaConf.save(self.cfg.recipes, config_file)
                self._recipe_template_processor = None
        else:
            omegaconf.OmegaConf.save(self.cfg.recipes, config_file)
            self._recipe_template_processor = None

    def _copy_k8s_template(self):
        # For SM Jobs, don't copy K8s-specific template files
        if self.cfg.cluster == "sm_jobs" or self.cfg.cluster_type == "sm_jobs":
            return

        for fname in ["training.yaml", "training-config.yaml"]:
            src = self._template_dir / fname
            dst = self._output_dir_k8s_folder / "templates" / fname
            shutil.copyfile(src, dst)

    def _process_values_yaml(self):
        # Load values.yaml template
        with open(self._template_dir / "values.yaml") as value_file:
            values_template = OmegaConf.load(value_file)

        cluster_cfg = copy.deepcopy(self.cfg.get("cluster") or {})
        k8s_cfg = {**cluster_cfg}

        # Basic assignments
        values_template.image.trainingImage = self.cfg.get("container")
        values_template.trainingConfig.jobName = self._job_name
        values_template.trainingConfig.initContainer.image = self._init_container_uri
        values_template.trainingConfig.envVars = self._get_env_vars()
        values_template.trainingConfig.numEFADevices = self.num_efa_devices
        if not self._launch_json:
            values_template.trainingConfig.namespace = "default"

        # Allow lambda access for eval jobs
        if OmegaConf.select(self.cfg, "recipes.evaluation", default=False):
            values_template.trainingConfig.lambdaAccess = True

        # Default is 8 if we dont find value in resource_config.devices or training_config.trainer.devices
        # resource_config is for eval recipes
        # training_config.trainer is for training recipes
        values_template.trainingConfig.devices = (
            OmegaConf.select(self.cfg, "recipes.resource_config.devices")
            or OmegaConf.select(self.cfg, "recipes.training_config.trainer.devices")
            or get_device_count_for_instance(self.instance_type)
        )
        # Replicas: always at least one node
        num_nodes = OmegaConf.select(self.cfg, "recipes.run.replicas", default=0)

        # Set worker_nodes based on launch_json mode
        # For launch_json: use template placeholder "{{replicas}}"
        # For actual runs: calculate as (num_nodes - 1)
        if self._launch_json:
            values_template.trainingConfig.processed_worker_nodes = "replicas_placeholder"
        else:
            values_template.trainingConfig.processed_worker_nodes = max(num_nodes - 1, 0)

        values_template.trainingConfig.worker_nodes = max(num_nodes - 1, 0)

        # Optional K8s fields
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
                setattr(values_template.trainingConfig, dest, val)

        values_template.trainingConfig.labelSelector = self._get_label_selectors()
        self._write_value_template(values_template)


class SMNovaK8SLauncherPPO(NovaK8SLauncher):
    """
    Launcher for Proximal Policy Optimization (PPO) jobs on Kubernetes that manages complex multi-job deployments including reward models, critic models, actor generation and training with appropriate resource allocation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._template_dir = Path(__file__).parent / "k8s_templates/PPO"
        self._reward_model_job_name = f"{self._job_name}_rm"
        self._critic_model_job_name = f"{self._job_name}_cm"
        self._anchor_model_job_name = f"{self._job_name}_am"
        self._actor_generation_job_name = f"{self._job_name}_ag"
        self._actor_train_job_name = f"{self._job_name}_at"
        self._actor_generation_container_uri = get_actor_generation_container_uri()

    def _save_hydra_config(self):
        """
        Saves the Hydra configuration for the current job.

        This method performs the following steps:
        1. Interpolates the Hydra configuration.
        2. Creates a 'config' directory inside the Kubernetes output folder if it does not exist.
        3. Converts the 'run' recipe configuration to a dictionary, resolving all interpolations.
        4. Removes specific keys defined in KEYS_TO_REMOVE from the run configuration.
        5. For each job type defined in JOB_TYPE_DICT:
            - Converts the corresponding recipe configuration to a dictionary.
            - Sets the 'task_type' field based on JOB_TASK_TYPE_DICT.
            - Builds a new configuration dictionary containing both 'run' and 'training_config'.
            - Saves the configuration as a YAML file named with the job name and job type.
        """
        self._interpolate_hydra()
        config_path = Path(self._output_dir_k8s_folder / "config")
        config_path.mkdir(parents=True, exist_ok=True)

        recipe_to_use = self.cfg

        # Initialize recipe template processor if launch_json is enabled
        if self._launch_json:
            try:
                # Get the recipe file path from Hydra config
                recipe_file_path = get_recipe_file_path()
                if recipe_file_path:
                    self._recipe_template_processor = NovaRecipeTemplateProcessor(
                        self.cfg.recipes, platform=self.cfg.cluster_type
                    )

                    # Call process_recipe() with recipe_file_path to set up matched_template_group
                    templated_recipe_dict = None
                    if self.cfg.cluster_type == "sm_jobs" or self.cfg.cluster == "sm_jobs":
                        templated_recipe_dict = self._recipe_template_processor.process_recipe(
                            recipe_file_path, dont_override_list=DONT_OVERRIDE_IN_SMTJ
                        )
                    else:
                        templated_recipe_dict = self._recipe_template_processor.process_recipe(recipe_file_path)

                    # Convert dict back to OmegaConf for compatibility
                    recipe_to_use = OmegaConf.create(templated_recipe_dict)

                    # Wrap the recipe in the expected "recipes" structure for proper interpolation resolution
                    recipe_to_use = OmegaConf.create({"recipes": recipe_to_use})
                else:
                    raise ValueError("Could not get recipe file path from Hydra config.")

            except Exception as e:
                raise ValueError(f"Failed to get templated recipe for config files.{e}")

        # Start building the new config.
        run_config = omegaconf.OmegaConf.to_container(recipe_to_use.recipes.run, resolve=True)
        for key in KEYS_TO_REMOVE:
            run_config.pop(key, None)

        # Build final config
        new_config_dict = {"run": run_config}
        for key, value in JOB_TYPE_DICT.items():
            config_file = Path(config_path / f"{self._job_name}-{key.value}_hydra.yaml")
            training_config = omegaconf.OmegaConf.to_container(recipe_to_use.recipes[value], resolve=True)
            training_config["task_type"] = JOB_TASK_TYPE_DICT[key]
            new_config_dict["training_config"] = training_config
            if training_config["task_type"] == "ppo_actor_train" and self._launch_json:
                recipe_file_path = get_recipe_file_path()
                if "eval" not in recipe_file_path:
                    if (
                        "trainer" not in new_config_dict["training_config"]
                        or "num_nodes" not in new_config_dict["training_config"]["trainer"]
                    ):
                        raise KeyError("Invalid PPO recipe. Actor train config not found")
                    new_config_dict["training_config"]["trainer"]["num_nodes"] = "{{replicas}}"
            # Save using OmegaConf to maintain consistent formatting
            OmegaConf.save(new_config_dict, config_file)

        # Set recipe_file_path to the first config file for SM Jobs compatibility
        self.recipe_file_path = Path(config_path / f"{self._job_name}-{list(JOB_TYPE_DICT.keys())[0].value}_hydra.yaml")

    def _build_job_list(self) -> List[Dict]:
        """
        Constructs a list of job configurations for different job types defined in `JOB_TYPE_DICT`.

        Each job dictionary contains:
        - `jobName`: A unique job name composed of the base job name and job type.
        - `master_nodes`: Number of master nodes. For `JobType.ACTOR_GENERATION`, it uses the full node count; otherwise, it's 1.
        - `worker_nodes`: Number of worker nodes, always calculated as (nodes - 1) as integer for Helm rendering.
        - `devices`: Number of devices (e.g., GPUs) per job, defaulting to 8 if unspecified.

        The values for `nodes` and `devices` are retrieved from a hierarchical config (Hydra OmegaConf),
        using paths from recipe `recipes.<task_name>.trainer.num_nodes`.

        Note: For launch_json mode, worker_nodes replicas will be replaced with "{{replicas}}"
        during post-processing in _create_launch_json().

        Returns:
            List[Dict]: A list of dictionaries, each representing a job configuration with keys:
                - "jobName": The name of the job (str)
                - "master_nodes": Number of master nodes (int)
                - "worker_nodes": Number of worker nodes (int)
                - "devices": Number of devices per node (int)
        """
        job_list = []
        recipe_file_path = get_recipe_file_path()
        for job_type in JOB_TYPE_DICT.keys():
            task_name = JOB_TYPE_DICT.get(job_type)
            if not task_name:
                continue

            nodes = OmegaConf.select(self.cfg, f"recipes.{task_name}.trainer.num_nodes", default=0)
            devices = OmegaConf.select(self.cfg, f"recipes.{task_name}.trainer.devices", default=8)
            master_nodes = nodes if job_type == JobType.ACTOR_GENERATION else 1
            worker_nodes = max(0, nodes - 1)

            if task_name != "ppo_actor_train":
                job_list.append(
                    {
                        "jobName": f"{self._job_name}-{job_type.value}",
                        "master_nodes": master_nodes,
                        "worker_nodes": worker_nodes,
                        "devices": devices,
                        "processed_worker_nodes": worker_nodes,
                    }
                )
            else:
                # For launch json generation, processed_worker_nodes will help us replace worker node replica values with {{replicas}}
                job_list.append(
                    {
                        "jobName": f"{self._job_name}-{job_type.value}",
                        "master_nodes": master_nodes,
                        "worker_nodes": worker_nodes,
                        "devices": devices,
                        "processed_worker_nodes": "replicas_placeholder"
                        if (self._launch_json and "eval" not in recipe_file_path)
                        else worker_nodes,
                    }
                )
        return job_list

    def _process_values_yaml(self):
        # Load values.yaml as an OmegaConf object
        with open(self._template_dir / "values.yaml") as value_file:
            values_template = OmegaConf.load(value_file)

        # Deep copy cluster config for isolation
        cluster_cfg = self.cfg.get("cluster") or {}
        k8s_cfg = copy.deepcopy(cluster_cfg)

        # Assign base container config
        values_template.image.trainingImage = self.cfg.get("container")
        values_template.image.actor_generation_image = self._actor_generation_container_uri

        # Assign job-specific values
        values_template.trainingConfig.jobName = self._job_name
        values_template.trainingConfig.initContainer.image = self._init_container_uri
        values_template.trainingConfig.envVars = self._get_env_vars()
        values_template.trainingConfig.numEFADevices = self.num_efa_devices
        if not self._launch_json:
            values_template.trainingConfig.namespace = "default"

        values_template["jobList"] = self._build_job_list()

        # Optional fields mapping
        field_mapping = {
            "namespace": "namespace",
            "annotations": "annotations",
            "priority_class_name": "priorityClassName",
            "service_account_name": "serviceAccountName",
            "custom_labels": "customLabels",
        }

        for key, attr in field_mapping.items():
            val = k8s_cfg.get(key)
            if val is not None:
                setattr(values_template.trainingConfig, attr, val)

        values_template.trainingConfig.labelSelector = self._get_label_selectors()

        self._write_value_template(values_template)

    def _copy_k8s_template(self):
        # For SM Jobs, don't copy K8s-specific template files
        if self.cfg.cluster == "sm_jobs" or self.cfg.cluster_type == "sm_jobs":
            return

        for fname in ["training.yaml", "training-ag.yaml", "training-config.yaml"]:
            src = self._template_dir / fname
            dst = self._output_dir_k8s_folder / "templates" / fname
            shutil.copyfile(src, dst)

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
                    "Nova recipes are only available for EKS clusters. Please update the config file to use k8s."
                )
                return
        else:
            self._run_helm_script(script_path)
            if self.cfg.cluster_type != "k8s":
                raise ValueError(
                    "Nova recipes are only available for EKS clusters. Please update the config file to use k8s."
                )


class SMNovaK8SLauncherRFT(NovaK8SLauncher):
    """
    Launcher for RFT (Reward Fine-Tuning) jobs on Kubernetes.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._template_dir = Path(__file__).parent / "k8s_templates" / "RFT"
        self._launch_json = self.cfg.get("launch_json", False)
        try:
            self.recipe_file_path = get_recipe_file_path()
        except ValueError:
            # Hydra not initialized (e.g., in tests)
            self.recipe_file_path = None

        # Service deployment order
        self._service_order = [
            RFTJobType.NATS_SERVER.value,
            RFTJobType.HUB.value,
            RFTJobType.PROMPT_RBS.value,
            RFTJobType.TRAINING.value,
            RFTJobType.VLLM_GENERATION.value,
        ]

        # Add Redis if delegate is enabled
        delegate = OmegaConf.select(cfg, "recipes.training_config.rollout.delegate")
        if delegate is not None and delegate is True:
            self._service_order.insert(0, RFTJobType.REDIS.value)

    def _save_hydra_config(self):
        """Save Hydra configuration files for each RFT service type."""

        # Revolve recipes before _interpolate_hydra which breaks refernces
        recipes_conf = OmegaConf.create(self.cfg.recipes)
        OmegaConf.resolve(recipes_conf)
        self.cfg.recipes = recipes_conf

        self._interpolate_hydra()
        config_path = Path(self._output_dir_k8s_folder / "config")
        config_path.mkdir(parents=True, exist_ok=True)

        recipe_to_use = self.cfg

        # Initialize recipe template processor if launch_json is enabled
        if self._launch_json:
            try:
                # Get the recipe file path from Hydra config
                recipe_file_path = get_recipe_file_path()
                if recipe_file_path:
                    self._recipe_template_processor = NovaRecipeTemplateProcessor(
                        self.cfg.recipes, platform=self.cfg.cluster_type
                    )

                    # Call process_recipe() with recipe_file_path to set up matched_template_group
                    templated_recipe_dict = None
                    if self.cfg.cluster_type == "sm_jobs" or self.cfg.cluster == "sm_jobs":
                        templated_recipe_dict = self._recipe_template_processor.process_recipe(
                            recipe_file_path, dont_override_list=DONT_OVERRIDE_IN_SMTJ
                        )
                    else:
                        templated_recipe_dict = self._recipe_template_processor.process_recipe(recipe_file_path)

                    # Convert dict back to OmegaConf for compatibility
                    recipe_to_use = OmegaConf.create(templated_recipe_dict)

                    # Wrap the recipe in the expected "recipes" structure for proper interpolation resolution
                    recipe_to_use = OmegaConf.create({"recipes": recipe_to_use})
                else:
                    raise ValueError("Could not get recipe file path from Hydra config.")

            except Exception as e:
                raise ValueError(f"Failed to get templated recipe for config files.{e}")

        # Start building the new config
        run_config = omegaconf.OmegaConf.to_container(recipe_to_use.recipes.run, resolve=True)
        for key in RFT_KEYS_TO_REMOVE:
            run_config.pop(key, None)

        # Create config files for each RFT service
        base_training_config = omegaconf.OmegaConf.to_container(recipe_to_use.recipes.training_config, resolve=True)

        # Create configs for main services (excluding NATS server which has its own config)
        for job_type in [RFTJobType.TRAINING, RFTJobType.VLLM_GENERATION, RFTJobType.HUB, RFTJobType.NATS_BOOTSTRAP]:
            service_config = copy.deepcopy(base_training_config)
            service_config["task_type"] = RFT_JOB_TASK_TYPE_DICT[job_type]

            config_file = Path(config_path / f"{self._job_name}-{job_type.value}_hydra.yaml")
            new_config_dict = {"run": run_config, "training_config": service_config}

            # Save using OmegaConf to maintain consistent formatting
            OmegaConf.save(new_config_dict, config_file)

        # Create separate configs for prompter and rbs (both part of prompt-rbs service)
        for component, task_type in RFT_PROMPT_RBS_TASK_TYPES.items():
            service_config = copy.deepcopy(base_training_config)
            service_config["task_type"] = task_type

            config_file = Path(config_path / f"{self._job_name}-{component}_hydra.yaml")
            new_config_dict = {"run": run_config, "training_config": service_config}

            # Save using OmegaConf to maintain consistent formatting
            OmegaConf.save(new_config_dict, config_file)

        # Set recipe_file_path to the first config file for SM Jobs compatibility
        self.recipe_file_path = Path(config_path / f"{self._job_name}-{RFTJobType.TRAINING.value}_hydra.yaml")

    def _build_job_list(self, values_template):
        """Build job list for RFT services based on replica counts from run config."""
        job_list = []
        run_config = self.cfg.recipes.run
        job_name = run_config.name

        # Get replica counts
        training_replicas = values_template.trainingConfig.training.replicas
        vllm_generation_replicas = values_template.trainingConfig.vllmGeneration.replicas
        rollout_worker_replicas = values_template.trainingConfig.hub.replicas

        # Build job list based on replica counts using RFT constants (excluding NATS server which manages its own config)
        services = [
            (RFTJobType.TRAINING.value, training_replicas),
            (RFTJobType.VLLM_GENERATION.value, vllm_generation_replicas),
            (RFTJobType.HUB.value, rollout_worker_replicas),
            (RFTJobType.NATS_BOOTSTRAP.value, 1),  # NATS bootstrap runs once
        ]

        for service_name, replicas in services:
            job_list.append(
                {"jobName": f"{job_name}-{service_name}", "replicas": replicas, "serviceType": service_name}
            )

        # Special case: prompt-rbs has two components (prompter and rbs) - get from values.yaml
        prompter_replicas = getattr(values_template.trainingConfig, "prompter", {}).get(
            "replicas", rollout_worker_replicas
        )
        rbs_replicas = getattr(values_template.trainingConfig, "rbs", {}).get("replicas", rollout_worker_replicas)
        job_list.append({"jobName": f"{job_name}-prompter", "replicas": prompter_replicas, "serviceType": "prompter"})
        job_list.append({"jobName": f"{job_name}-rbs", "replicas": rbs_replicas, "serviceType": "rbs"})

        # Add Redis for delegate
        delegate = OmegaConf.select(self.cfg, "recipes.training_config.rollout.delegate")
        if delegate is not None and delegate is True:
            redis_replicas = run_config.get("redis_replicas", 1)
            job_list.append(
                {
                    "jobName": f"{job_name}-{RFTJobType.REDIS.value}",
                    "replicas": redis_replicas,
                    "serviceType": RFTJobType.REDIS.value,
                }
            )

        return job_list

    def _process_values_yaml(self):
        """Process values.yaml template with RFT configuration."""
        with open(self._template_dir / "values.yaml") as value_file:
            values_template = OmegaConf.load(value_file)

        # Get configurations from cluster config
        cluster_cfg = self.cfg.get("cluster") or {}

        # Set basic configuration
        values_template.trainingConfig.jobName = self._job_name
        values_template.trainingConfig.initContainer.image = self._init_container_uri
        values_template.trainingConfig.envVars = self._get_env_vars()
        values_template.trainingConfig.numEFADevices = self.num_efa_devices
        if not self._launch_json:
            values_template.trainingConfig.namespace = "kubeflow"

        # Set EFA device resources programmatically
        self._set_efa_resources(values_template)

        # Set region, alias, nodeType from cluster config
        # Try deployment_metadata section first, then fall back to root level
        deployment_meta = cluster_cfg.get("deployment_metadata", {})
        values_template.trainingConfig.region = (
            deployment_meta.get("region")
            or cluster_cfg.get("region")
            or self.cfg.get("deployment_metadata", {}).get("region")
        )
        values_template.trainingConfig.alias = (
            deployment_meta.get("alias")
            or cluster_cfg.get("alias")
            or self.cfg.get("deployment_metadata", {}).get("alias")
        )
        values_template.trainingConfig.nodeType = (
            deployment_meta.get("node_type")
            or cluster_cfg.get("node_type")
            or self.cfg.get("deployment_metadata", {}).get("node_type")
        )

        # SET REQUIRED TOLERATIONS
        values_template.trainingConfig.requiredTolerations = [
            {
                "key": "sagemaker.amazonaws.com/RestrictedNode",
                "operator": "Equal",
                "value": "Worker",
                "effect": "NoSchedule",
            }
        ]

        # Map K8s metadata from cluster config
        field_mapping = {
            "namespace": "namespace",
            "annotations": "annotations",
            "priority_class_name": "priorityClassName",
            "service_account_name": "serviceAccountName",
            "custom_labels": "customLabels",
        }

        for key, attr in field_mapping.items():
            val = cluster_cfg.get(key)
            if val is not None:
                setattr(values_template.trainingConfig, attr, val)

        # Set replica counts and Redis config
        self._map_rft_replica_config(values_template)

        # Calculate worker_replicas based on launch_json mode
        # For launch_json: use template placeholder "{{replicas}}"
        # For actual runs: calculate as (replicas - 1)
        training_replicas = values_template.trainingConfig.training.replicas
        recipe_file_path = get_recipe_file_path()
        if self._launch_json and "eval" not in recipe_file_path:
            values_template.trainingConfig.training.worker_replicas = "replicas_placeholder"
        else:
            values_template.trainingConfig.training.worker_replicas = max(training_replicas - 1, 0)

        # Set resource configuration
        self._map_resource_config(values_template)

        # Set job list for ConfigMap generation
        values_template["jobList"] = self._build_job_list(values_template)

        # Set image configuration
        self._map_image_config(values_template)

        # Set label selectors
        values_template.trainingConfig.labelSelector = self._get_label_selectors()

        # Resolve values_template before writing
        OmegaConf.resolve(values_template)
        self._write_value_template(values_template)

    def _set_efa_resources(self, values_template):
        """Set EFA device resources for training and vllmGeneration services."""
        efa_count = self.num_efa_devices

        # Set EFA resources for training service
        if hasattr(values_template.trainingConfig, "training"):
            training_config = values_template.trainingConfig.training
            if hasattr(training_config, "resources"):
                if hasattr(training_config.resources, "master"):
                    training_config.resources.master.requests[EFA_RESOURCE_KEY] = efa_count
                    training_config.resources.master.limits[EFA_RESOURCE_KEY] = efa_count
                if hasattr(training_config.resources, "worker"):
                    training_config.resources.worker.requests[EFA_RESOURCE_KEY] = efa_count
                    training_config.resources.worker.limits[EFA_RESOURCE_KEY] = efa_count

        # Set EFA resources for vllmGeneration service
        if hasattr(values_template.trainingConfig, "vllmGeneration"):
            vllm_config = values_template.trainingConfig.vllmGeneration
            if hasattr(vllm_config, "resources"):
                vllm_config.resources.requests[EFA_RESOURCE_KEY] = efa_count
                vllm_config.resources.limits[EFA_RESOURCE_KEY] = efa_count

    def _map_rft_replica_config(self, values_template):
        """Map replica counts and Redis configuration."""
        run_config = self.cfg.recipes.run

        # Set replica counts
        if run_config.get("replicas") is not None:
            values_template.trainingConfig.training.replicas = run_config.get("replicas")
        if run_config.get("generation_replicas") is not None:
            values_template.trainingConfig.vllmGeneration.replicas = run_config.get("generation_replicas")
        if run_config.get("rollout_worker_replicas") is not None:
            values_template.trainingConfig.hub.replicas = run_config.get("rollout_worker_replicas")

        # REDIS CONFIG - Only for delegate, used by Helm templates
        delegate = OmegaConf.select(self.cfg, "recipes.training_config.rollout.delegate")
        is_delegate_enabled = delegate is not None and delegate is True
        values_template.trainingConfig.redis.enabled = is_delegate_enabled

        if is_delegate_enabled:
            values_template.trainingConfig.redis.replicas = run_config.get("redis_replicas", 1)
            # Get Redis config from training_config
            redis_config = OmegaConf.select(self.cfg, "recipes.training_config.redis")
            if redis_config:
                values_template.trainingConfig.redis.maxMemory = redis_config.get("max_memory", "8gb")
                values_template.trainingConfig.redis.maxMemoryPolicy = redis_config.get(
                    "max_memory_policy", "allkeys-lru"
                )

    def _map_image_config(self, values_template):
        """Map image configuration."""
        # Set training image from container field
        if self.cfg.get("container"):
            values_template.image.trainingImage = self.cfg.get("container")
        else:
            values_template.image.trainingImage = get_rft_train_container_uri()

        # Set RFT-specific images using utility functions
        values_template.image.generationImage = get_rft_generation_container_uri()
        values_template.image.stormImage = get_rft_storm_container_uri()
        values_template.image.natsServerImage = get_rft_nats_server_container_uri()
        values_template.image.natsReloaderImage = get_rft_nats_reloader_container_uri()
        values_template.image.redis = get_rft_redis_container_uri()

    def _map_resource_config(self, values_template):
        """Map resource configuration
        Set instance types and apply nodeAffinity to all services."""

        instance_type = self.instance_type or values_template.trainingConfig.defaultResources.instanceType

        # Set instance type and apply nodeAffinity to all services
        services = ["training", "vllmGeneration", "hub", "prompter", "rbs", "natsServer"]

        for service_name in services:
            service = getattr(values_template.trainingConfig, service_name, None)
            if service:
                # Set instance type
                service.instanceType = instance_type

        # Apply to redis if enabled
        if hasattr(values_template.trainingConfig, "redis") and values_template.trainingConfig.redis.enabled:
            values_template.trainingConfig.redis.instanceType = instance_type

    def _to_camel_case(self, snake_str):
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    def _copy_k8s_template(self):
        """Copy K8s template files."""
        if self.cfg.cluster == "sm_jobs" or self.cfg.cluster_type == "sm_jobs":
            return

        # Copy all RFT service templates
        template_files = [
            "training.yaml",
            "vllm-generation.yaml",
            "hub.yaml",
            "prompt-rbs.yaml",
            "nats-server.yaml",
            "training-config.yaml",
        ]

        # Add Redis template for delegate
        delegate = OmegaConf.select(self.cfg, "recipes.training_config.rollout.delegate")
        if delegate is not None and delegate is True:
            template_files.append("redis.yaml")

        templates_dir = self._output_dir_k8s_folder / "templates"
        templates_dir.mkdir(exist_ok=True)

        for fname in template_files:
            src = self._template_dir / "templates" / fname
            dst = templates_dir / fname
            if src.exists():
                shutil.copyfile(src, dst)

    def _validate_recipe_parameters(self):
        """Validate RFT recipe parameters."""
        training_config = self.cfg.recipes.training_config
        run_config = self.cfg.recipes.run

        # Validate model_config exists - check both new flattened structure and legacy structure
        model_config_found = False

        # Check for optimized RFT structure with rollout, data, and train sections
        if (
            hasattr(training_config, "rollout")
            and hasattr(training_config, "data")
            and hasattr(training_config, "trainer")
        ):
            model_config_found = True  # Optimized RFT structure is valid

        if not model_config_found:
            raise ValueError("Missing required model_config section in trainer configuration")

        # Validate dataset path - check multiple possible locations
        dataset_path = None

        # Check legacy location: training_config.trainer.dataset_config.path
        dataset_path = OmegaConf.select(self.cfg, "recipes.training_config.trainer.dataset_config.path")

        # Check flattened location: training_config.dataset_config.path
        if not dataset_path:
            dataset_path = OmegaConf.select(self.cfg, "recipes.training_config.dataset_config.path")

        # Check run config location: run.data_s3_path
        if not dataset_path:
            dataset_path = OmegaConf.select(self.cfg, "recipes.run.data_s3_path")

        # Check prompt source location: training_config.prompt_source.dataset.dataset_path
        if not dataset_path:
            dataset_path = OmegaConf.select(self.cfg, "recipes.training_config.prompt_source.dataset.dataset_path")

        if not dataset_path:
            raise ValueError(
                "Missing required dataset parameter: trainer.dataset_config.path, "
                "training_config.dataset_config.path, run.data_s3_path, or "
                "training_config.prompt_source.dataset.dataset_path"
            )

        # Validate replica counts are positive integers
        replica_fields = [
            "replicas",
            "generation_replicas",
            "rollout_worker_replicas",
        ]

        for field in replica_fields:
            value = getattr(run_config, field, 1)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Replica count '{field}' must be a positive integer, got: {value}")

    def _create_service_config_files(self):
        """Create individual config files for each RFT service."""
        config_path = Path(self._output_dir_k8s_folder / "config")
        config_path.mkdir(parents=True, exist_ok=True)

        job_list = self._build_job_list()

        # Create a config file for each service in the job list
        for job in job_list:
            service_type = job["serviceType"]
            job_name = job["jobName"]

            # Create a basic config for each service
            service_config = {"jobName": job_name, "serviceType": service_type, "replicas": job["replicas"]}

            config_file = config_path / f"{job_name}_service_config.yaml"
            OmegaConf.save(service_config, config_file)

    def run(self):
        """Run the RFT launcher."""
        self._validate_recipe_parameters()
        self._prepare_output_dir()
        self._save_hydra_config()
        self._create_chart_file(self._template_dir)
        self._copy_k8s_template()
        self._process_values_yaml()
        script_path = self._create_helm_script(self._output_dir_k8s_folder)

        if self._launch_json:
            self._create_launch_json(self._output_dir_k8s_folder)
        else:
            self._run_helm_script(script_path)
            logger.info(f"RFT Launcher successfully deployed: {self._template_dir}")


class SMNovaSMTJLauncherRFT(NovaK8SLauncher):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Only supports launch_json creation for now
        if self.cfg.cluster_type == "sm_jobs":
            self._prepare_output_dir()
            self.recipe_file_path = get_recipe_file_path()
            self.create_launch_json(self.cfg)

    def create_launch_json(self, cfg):
        if self.recipe_file_path is None:
            raise KeyError("Recipe file path not found in hydra")
        else:
            # Create recipe template processor for Nova recipes
            self.recipe_template_processor = NovaRecipeTemplateProcessor(
                self.cfg.recipes, platform=self.cfg.cluster_type
            )
            # Pass the recipe file path to process_recipe
            templatized_recipe = None
            if self.cfg.cluster_type == "sm_jobs" or self.cfg.cluster == "sm_jobs":
                templatized_recipe = self.recipe_template_processor.process_recipe(
                    self.recipe_file_path, dont_override_list=DONT_OVERRIDE_IN_SMTJ
                )
            else:
                templatized_recipe = self.recipe_template_processor.process_recipe(self.recipe_file_path)

            recipe_file_name = get_recipe_name_from_path(self.recipe_file_path)
            templatized_recipe_filename = "templatized_" + recipe_file_name + ".yaml"
            omegaconf.OmegaConf.save(templatized_recipe, templatized_recipe_filename)

            launch_json = {}
            try:
                # The Nova recipe template processor expects the path without .yaml extension
                # and adds it internally, so we don't need to add it here
                additional_data = self.recipe_template_processor.get_additional_data(self.recipe_file_path)

                # Check if container is available for this platform
                if additional_data is None:
                    logger.error("Recipe templatization failed because additional_data was returned as None")
                    return

                (
                    launch_json["metadata"],
                    launch_json["recipe_override_parameters"],
                    launch_json["regional_parameters"],
                ) = additional_data

                logger.info("Added recipe templatization metadata to launch.json")
            except Exception as e:
                logger.error(f"Failed to add recipe templatization metadata: {e}")
                raise Exception

            full_recipe_path = os.path.join("./recipes_collection/recipes", self.recipe_file_path + ".yaml")
            untemplated_recipe = OmegaConf.load(full_recipe_path)
            wrapped_recipe = OmegaConf.create({"recipes": untemplated_recipe})
            recipe_container = OmegaConf.to_container(wrapped_recipe, resolve=True)
            launch_json["training_recipe.json"] = recipe_container

            with open(templatized_recipe_filename, "r") as f:
                recipe_yaml_content = f.read()

            launch_json["training_recipe.yaml"] = recipe_yaml_content

            launch_path = self._output_dir / "launch.json"
            with open(launch_path, "w") as f:
                json.dump(launch_json, f, indent=2, sort_keys=False)
                f.write("\n")

    def run(self):
        logger.info("Launching RFT training job with SM Jobs has not been implemented yet")
        logger.info(f"Outputs are in: {self._output_dir}")
