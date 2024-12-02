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

import logging

from omegaconf import OmegaConf

from main import main

logger = logging.getLogger(__name__)


from tests.test_utils import (
    compare_artifacts,
    create_temp_directory,
    make_hydra_cfg_instance,
)


def compare_custom_slurm_artifacts(artifacts_dir):
    logger.info("Comparing custom slurm artifacts")

    artifacts_paths = [
        "/test_custom/launch_docker_container.sh",
        "/test_custom/testcustom_slurm_test_custom_submission.sh",
        "/test_custom/train_script.sh",
        "/test_custom/docker_exec_script.sh",
    ]
    slurm_baseline_artifacts_path = "/tests/slurm_workflow/slurm_baseline_artifacts"
    compare_artifacts(artifacts_paths, artifacts_dir, slurm_baseline_artifacts_path)


def test_custom_slurm_workflow():
    logger.info("Testing custom slurm workflow")

    artifacts_dir = create_temp_directory()
    overrides = [
        "training_cfg.entry_script=test.py",
        "cluster.instance_type=p5.48xlarge",
        "cluster.cluster_type=slurm",
        "cluster.cluster_config.slurm_create_submission_file_only=True",
        "git.repo_url_or_path=https://github.com/example",
        "base_results_dir={}".format(artifacts_dir),
        "container=test_container",
    ]

    sample_custom_slurm_config = make_hydra_cfg_instance("../launcher_scripts/custom_script", "config_slurm", overrides)

    logger.info("\nsample_custom_slurm_config\n")
    logger.info(OmegaConf.to_yaml(sample_custom_slurm_config))

    main(sample_custom_slurm_config)

    compare_custom_slurm_artifacts(artifacts_dir)
