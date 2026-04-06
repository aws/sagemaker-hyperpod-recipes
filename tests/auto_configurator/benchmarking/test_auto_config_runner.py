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

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from auto_configurator.benchmarking.auto_config_runner import AutoConfigRunner


@pytest.fixture
def mock_auto_config():
    return OmegaConf.create(
        {
            "name": "test_run",
            "recipe": "training/llama/llmft_llama3_8b",
            "instance_type": "ml.p5.48xlarge",
            "platform": "k8s",
            "base_results_dir": "/test/results",
        }
    )


@pytest.fixture
def mock_base_config():
    return OmegaConf.create(
        {"cluster": {"type": "k8s"}, "recipes": {"model": {"data": {}}}, "base_results_dir": "/test/results"}
    )


def _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher):
    """Helper to create a runner with all init steps mocked"""
    mock_root.return_value = "/test/root"
    mock_load.return_value = mock_base_config
    mock_launcher_instance = MagicMock()
    mock_launcher_instance._build_command.return_value = ["python", "main.py"]
    mock_launcher.return_value.return_value = mock_launcher_instance

    with (
        patch.object(
            AutoConfigRunner, "_AutoConfigRunner__generate_base_launch_command", return_value=["python", "main.py"]
        ),
        patch.object(AutoConfigRunner, "_get_recipe", return_value=mock_base_config),
    ):
        return AutoConfigRunner(mock_auto_config)


class TestAutoConfigRunnerInit:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_init(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        assert runner.auto_config == mock_auto_config
        assert runner._AutoConfigRunner__run_identifier == "autoconfigurator-ml_p5_48xlarge"
        mock_recorder.assert_called_once_with(instance_type="ml.p5.48xlarge", platform="k8s")


class TestGenerateValidationConfig:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_generate_validation_config(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)
        result = runner._generate_validation_config()

        assert "name" in result
        assert result.name == "test_run"


class TestGetRecipe:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_recipe")
    def test_get_recipe_success(
        self,
        mock_get_recipe,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        mock_load.return_value = mock_base_config
        mock_launcher.return_value.return_value = MagicMock()

        mock_recipe = MagicMock()
        mock_recipe.config = mock_base_config
        mock_get_recipe.return_value = mock_recipe

        with patch.object(
            AutoConfigRunner, "_AutoConfigRunner__generate_base_launch_command", return_value=["python", "main.py"]
        ):
            runner = AutoConfigRunner(mock_auto_config)

        assert runner.base_recipe == mock_base_config
        mock_get_recipe.assert_called_with("training/llama/llmft_llama3_8b")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_recipe")
    def test_get_recipe_strips_yaml_extension(
        self, mock_get_recipe, mock_recorder, mock_launcher, mock_load, mock_makedirs, mock_base_config
    ):
        mock_load.return_value = mock_base_config
        mock_launcher.return_value.return_value = MagicMock()

        mock_recipe = MagicMock()
        mock_recipe.config = mock_base_config
        mock_get_recipe.return_value = mock_recipe

        auto_config = OmegaConf.create(
            {
                "name": "test_run",
                "recipe": "training/llama/llmft_llama3_8b.yaml",
                "instance_type": "ml.p5.48xlarge",
                "platform": "k8s",
                "base_results_dir": "/test/results",
            }
        )

        with patch.object(
            AutoConfigRunner, "_AutoConfigRunner__generate_base_launch_command", return_value=["python", "main.py"]
        ):
            runner = AutoConfigRunner(auto_config)

        mock_get_recipe.assert_called_with("training/llama/llmft_llama3_8b")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_recipe")
    def test_get_recipe_failure(
        self,
        mock_get_recipe,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        mock_load.return_value = mock_base_config
        mock_launcher.return_value.return_value = MagicMock()
        mock_get_recipe.side_effect = KeyError("Recipe not found")

        with (
            patch.object(
                AutoConfigRunner, "_AutoConfigRunner__generate_base_launch_command", return_value=["python", "main.py"]
            ),
            pytest.raises(KeyError, match="Recipe not found"),
        ):
            AutoConfigRunner(mock_auto_config)


class TestGenerateBaseLaunchCommand:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_generate_base_launch_command_success(
        self, mock_recorder, mock_launcher, mock_load, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher_instance._prepare_job.return_value = {"recipe": "test"}
        mock_launcher_instance._build_command.return_value = ["python", "main.py"]
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_get_recipe", return_value=mock_base_config):
            runner = AutoConfigRunner(mock_auto_config)

        assert runner._AutoConfigRunner__base_launch_command == ["python", "main.py"]
        mock_launcher_instance._prepare_job.assert_called_once()
        mock_launcher_instance._build_command.assert_called_once()

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_generate_base_launch_command_failure(
        self, mock_recorder, mock_launcher, mock_load, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher_instance._prepare_job.side_effect = Exception("Job preparation failed")
        mock_launcher.return_value.return_value = mock_launcher_instance

        with pytest.raises(Exception, match="Job preparation failed"):
            AutoConfigRunner(mock_auto_config)


class TestFindHydraConfig:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("glob.glob")
    def test_find_hydra_config_success(
        self,
        mock_glob,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_root,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)
        mock_glob.return_value = ["/test/artifact/test_hydra.yaml"]

        result = runner._AutoConfigRunner__find_hydra_config("/test/artifact")

        assert result == "/test/artifact/test_hydra.yaml"
        mock_glob.assert_called_once_with("/test/artifact/*_hydra.yaml")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("glob.glob")
    def test_find_hydra_config_not_found(
        self,
        mock_glob,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_root,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)
        mock_glob.return_value = []

        with pytest.raises(FileNotFoundError, match="No hydra config file found"):
            runner._AutoConfigRunner__find_hydra_config("/test/artifact")


class TestLaunch:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_launch_dryrun(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        runner._job_launcher._execute_command.return_value = mock_output

        with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
            mock_find.return_value = "/test/artifacts/test_hydra.yaml"
            result, success = runner.launch(dryrun=True)

            assert result["config_path"] == "/test/artifacts/test_hydra.yaml"
            assert success is True

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_launch_dryrun_missing_artifact_dir(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        mock_output.stdout = "No artifact directory in output"
        runner._job_launcher._execute_command.return_value = mock_output

        with pytest.raises(RuntimeError, match="Artifact directory not defined"):
            runner.launch(dryrun=True)

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_launch_with_overrides(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        runner._job_launcher._execute_command.return_value = mock_output

        with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
            mock_find.return_value = "/test/artifacts/test_hydra.yaml"
            overrides = ["++batch_size=32", "++learning_rate=0.001"]
            runner.launch(override_commands=overrides, dryrun=True)

            call_args = runner._job_launcher._execute_command.call_args[0][0]
            assert "++batch_size=32" in call_args
            assert "++learning_rate=0.001" in call_args

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.time.sleep")
    def test_launch_retry_and_fail(
        self,
        mock_sleep,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_root,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        with patch.object(runner, "_AutoConfigRunner__execute_launch") as mock_execute:
            mock_execute.side_effect = Exception("Launch execution failed")

            with pytest.raises(Exception, match="Launch execution failed"):
                runner.launch(override_commands=["++test=true"], dryrun=False, max_retry=2)

            assert mock_execute.call_count == 2
            mock_sleep.assert_called_once_with(1)

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.os.path.exists")
    def test_execute_launch_non_dryrun(
        self,
        mock_exists,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_root,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        mock_exists.return_value = True
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        mock_output.stdout = "Job launched successfully"
        runner._job_launcher._execute_command.return_value = mock_output

        job_details = {"job_name": "test-job", "pod_name": "test-pod", "log_path": "/test/logs/pod.log"}
        runner._job_launcher._parse_output.return_value = job_details
        runner._job_launcher._monitor_job.return_value = True

        result, success = runner.launch(override_commands=["++test=true"], dryrun=False)

        assert result == job_details
        assert success is True
        mock_exists.assert_called_once_with("/test/logs/pod.log")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    @patch("auto_configurator.benchmarking.auto_config_runner.os.path.exists")
    def test_execute_launch_non_dryrun_missing_log_file(
        self,
        mock_exists,
        mock_recorder,
        mock_launcher,
        mock_load,
        mock_root,
        mock_makedirs,
        mock_auto_config,
        mock_base_config,
    ):
        mock_exists.return_value = False
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        mock_output.stdout = "Job launched successfully"
        runner._job_launcher._execute_command.return_value = mock_output

        job_details = {"job_name": "test-job", "pod_name": "test-pod", "log_path": "/test/logs/pod.log"}
        runner._job_launcher._parse_output.return_value = job_details
        runner._job_launcher._monitor_job.return_value = True

        with pytest.raises(FileNotFoundError, match="Log file not found"):
            runner.launch(override_commands=["++test=true"], dryrun=False)


class TestGetJobDetails:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_get_job_details_with_output_folder_path(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        runner._job_launcher._parse_output.return_value = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "output_folder_path": "'/test/output/artifacts'",
        }

        with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
            mock_find.return_value = "/test/output/config.yaml"
            result = runner._AutoConfigRunner__get_job_details(mock_output)

            assert result["config_path"] == "/test/output/config.yaml"
            assert result["log_path"] == "/test/results/test-job/test-pod.logs"
            mock_find.assert_called_once_with("/test/output")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_get_job_details_without_output_folder_path(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        job_details_from_parse = {"job_name": "test-job", "pod_name": "test-pod"}
        runner._job_launcher._parse_output.return_value = job_details_from_parse

        result = runner._AutoConfigRunner__get_job_details(mock_output)

        assert "config_path" not in result
        assert result == job_details_from_parse

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_get_job_details_missing_job_name(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        runner = _create_runner(mock_auto_config, mock_base_config, mock_load, mock_root, mock_launcher)

        mock_output = MagicMock()
        runner._job_launcher._parse_output.return_value = {
            "pod_name": "test-pod",
            "output_folder_path": "'/test/output/artifacts'",
        }

        with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
            mock_find.return_value = "/test/output/config.yaml"

            with pytest.raises(ValueError, match="Unable to determine log_path for job"):
                runner._AutoConfigRunner__get_job_details(mock_output)
