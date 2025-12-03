import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.validations.validation_script_runner import run_validation

cfg = OmegaConf.load("./tests/hyperpod_recipe_validation_scripts/common_config.yaml")


@pytest.mark.skipif(
    os.environ.get("RUN_K8_VALIDATION_TEST", False) == False,
    reason="Please set env var RUN_K8_VALIDATION_TEST to True to run this pytests",
)
def test_k8s_job_submission():
    cfg.platform = "K8"
    jobRecorder = run_validation(cfg, cfg.recipe_list)

    if jobRecorder == None:
        assert False

    # Check test results
    final_job_statuses = jobRecorder.get_status()
    for job, status in final_job_statuses:
        assert status == "Complete"


@pytest.mark.skipif(
    os.environ.get("RUN_SLURM_VALIDATION_TEST", False) == False,
    reason="Please set env var RUN_SLURM_VALIDATION_TEST to True to run this pytests",
)
def test_slurm_job_submission():
    # cfg = OmegaConf.load("./tests/hyperpod_recipe_validation_scripts/common_config.yaml")
    cfg.platform = "SLURM"
    jobRecorder = run_validation(cfg, cfg.recipe_list)

    if jobRecorder == None:
        assert False

    # Check test results
    final_job_statuses = jobRecorder.get_status()
    for job, status in final_job_statuses:
        assert status == "Complete"


@pytest.mark.skipif(
    os.environ.get("RUN_SMJOBS_VALIDATION_TEST", False) == False,
    reason="Please set env var RUN_SMJOBS_VALIDATION_TEST to True to run this pytests",
)
def test_smjobs_job_submission():
    # cfg = OmegaConf.load("./tests/hyperpod_recipe_validation_scripts/common_config.yaml")
    cfg.platform = "SMJOBS"
    jobRecorder = run_validation(cfg, cfg.recipe_list)

    if jobRecorder == None:
        assert False

    # Check test results
    final_job_statuses = jobRecorder.get_status()
    for job, status in final_job_statuses:
        assert status == "Complete"
