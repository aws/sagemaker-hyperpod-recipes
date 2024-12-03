import difflib
import logging
import os
import shutil
import stat
import tempfile

from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


def create_temp_directory():
    """Create a temporary directory and Set full permissions for the directory"""
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return temp_dir


def replace_placeholder(file_path, placeholder, replacement):
    """Replace occurrences of placeholder in a file with the given replacement."""
    with open(file_path, "r") as file:
        content = file.read()

    content = content.replace(placeholder, replacement)

    with open(file_path, "w") as file:
        file.write(content)


def compare_artifacts(artifacts_paths, artifacts_dir, baseline_artifacts_path):
    for artifact_path in artifacts_paths:
        current_dir = os.getcwd()
        actual_artifact_path = artifacts_dir + artifact_path
        baseline_artifact_folder = current_dir + baseline_artifacts_path

        # Make a copy of baseline artifacts to replace placeholders
        baseline_artifact_copy_folder = create_temp_directory()
        shutil.copytree(baseline_artifact_folder, baseline_artifact_copy_folder, dirs_exist_ok=True)
        baseline_artifact_path = baseline_artifact_copy_folder + artifact_path

        results_dir_placeholder = "{$results_dir}"
        replace_placeholder(baseline_artifact_path, results_dir_placeholder, artifacts_dir)
        workspace_dir_placeholder = "{$workspace_dir}"
        replace_placeholder(baseline_artifact_path, workspace_dir_placeholder, current_dir)

        comparison_result = compare_files(baseline_artifact_path, actual_artifact_path)
        if comparison_result is False:
            assert False, baseline_artifact_path + " does not match " + actual_artifact_path


def compare_files(file1_path, file2_path):
    """Compare two files character by character."""
    with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
        file1_content = file1.readlines()
        file2_content = file2.readlines()

    # Using difflib to compare files
    diff = list(difflib.unified_diff(file1_content, file2_content, fromfile=file1_path, tofile=file2_path))

    if diff:
        diff_block = "\n" + "\n".join(line.strip() for line in diff)
        logger.info(f"Files differ:{diff_block}")
        return False

    logger.info("Files are identical.")
    return True


def compose_hydra_cfg(path, config_name, overrides=[]):
    """Init and compose a hydra config"""
    with initialize(version_base=None, config_path=path):
        return compose(config_name=config_name, overrides=overrides, return_hydra_config=True)


def make_hydra_cfg_instance(path, config_name, overrides):
    """Init hydra instance"""
    # Note: This is needed if using compose API and not hydra.main b/c we rely on hydra resolver
    # Open issue tracking fix https://github.com/facebookresearch/hydra/issues/2017
    config = compose_hydra_cfg(path, config_name, overrides)
    HydraConfig.instance().set_config(config)
    return config
