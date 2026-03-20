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


class TestAutoConfigRunnerInit:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_init(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            assert runner.auto_config == mock_auto_config
            assert runner._AutoConfigRunner__run_identifier == "autoconfigurator-ml_p5_48xlarge"
            mock_recorder.assert_called_once_with(instance_type="ml.p5.48xlarge", platform="k8s")


class TestGenerateBaseRecipe:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_generate_base_recipe(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            result = runner._generate_base_recipe()

            assert "name" in result
            assert result.name == "test_run"


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
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance
        mock_glob.return_value = ["/test/artifact/test_hydra.yaml"]

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
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
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance
        mock_glob.return_value = []

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            with pytest.raises(FileNotFoundError, match="No hydra config file found"):
                runner._AutoConfigRunner__find_hydra_config("/test/artifact")


class TestPrepareBaseCfg:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_prepare_base_cfg_executes(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        """Test that __prepare_base_cfg actually executes its code"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_run_info = {"recipe": "test_recipe"}
        mock_launcher_instance._prepare_job.return_value = mock_run_info
        mock_launcher_instance._build_command.return_value = ["python", "main.py"]

        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        mock_launcher_instance._execute_command.return_value = mock_output
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__find_hydra_config") as mock_find:
            mock_find.return_value = "/test/config.yaml"

            with patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load") as mock_omega_load:
                mock_omega_load.return_value = mock_base_config

                runner = AutoConfigRunner(mock_auto_config)

                assert runner.base_recipe_cfg == mock_base_config
                mock_launcher_instance._prepare_job.assert_called_once()
                mock_launcher_instance._build_command.assert_called_once()
                mock_find.assert_called_once()

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_prepare_base_cfg_exception(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        """Test that __prepare_base_cfg handles exceptions correctly"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher_instance._prepare_job.side_effect = Exception("Job preparation failed")
        mock_launcher.return_value.return_value = mock_launcher_instance

        with pytest.raises(Exception, match="Job preparation failed"):
            AutoConfigRunner(mock_auto_config)

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_prepare_base_cfg_success(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_run_info = {"recipe": "test_recipe"}
        mock_launcher_instance._prepare_job.return_value = mock_run_info
        mock_launcher_instance._build_command.return_value = ["python", "main.py"]

        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        mock_launcher_instance._execute_command.return_value = mock_output
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "launch") as mock_launch:
            with patch.object(AutoConfigRunner, "_AutoConfigRunner__find_hydra_config") as mock_find:
                with patch.object(AutoConfigRunner, "_AutoConfigRunner__load_config") as mock_load_cfg:
                    mock_find.return_value = "/test/config.yaml"
                    mock_load_cfg.return_value = mock_base_config
                    mock_launch.return_value = ({"config_path": "/test/config.yaml"}, True)

                    runner = AutoConfigRunner(mock_auto_config)

                    assert runner.base_recipe_cfg == mock_base_config
                    mock_launcher_instance._prepare_job.assert_called_once()
                    mock_launcher_instance._build_command.assert_called_once()
                    mock_launch.assert_called_once_with(dryrun=True)
                    mock_load_cfg.assert_called_once_with("/test/config.yaml")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_prepare_base_cfg_failure(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.side_effect = Exception("Preparation failed")

            with pytest.raises(Exception, match="Preparation failed"):
                AutoConfigRunner(mock_auto_config)


class TestLoadConfig:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_load_config_success(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            with patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load") as mock_omega_load:
                mock_omega_load.return_value = mock_base_config
                result = runner._AutoConfigRunner__load_config("/test/config.yaml")

                assert result == mock_base_config
                mock_omega_load.assert_called_once_with("/test/config.yaml")

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_load_config_failure(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            with patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load") as mock_omega_load:
                mock_omega_load.side_effect = Exception("Failed to load")

                with pytest.raises(Exception, match="Failed to load"):
                    runner._AutoConfigRunner__load_config("/test/config.yaml")


class TestLaunch:
    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_launch_dryrun(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        mock_launcher_instance._execute_command.return_value = mock_output
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            runner._AutoConfigRunner__base_launch_command = ["python", "main.py"]

            with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
                mock_find.return_value = "/test/artifacts/test_hydra.yaml"
                result, success = runner.launch(dryrun=True)

                assert "config_path" in result
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
        """Test launch dryrun raises RuntimeError when artifact directory is not found"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.stdout = "No artifact directory in output"
        mock_launcher_instance._execute_command.return_value = mock_output
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            runner._AutoConfigRunner__base_launch_command = ["python", "main.py"]

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
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.stdout = "[DRY_RUN] Artifacts generated at: /test/artifacts"
        mock_launcher_instance._execute_command.return_value = mock_output
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            runner._AutoConfigRunner__base_launch_command = ["python", "main.py"]

            with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
                mock_find.return_value = "/test/artifacts/test_hydra.yaml"
                overrides = ["++batch_size=32", "++learning_rate=0.001"]
                runner.launch(override_commands=overrides, dryrun=True)

                call_args = mock_launcher_instance._execute_command.call_args[0][0]
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
        """Test that launch retries with exponential backoff and eventually fails"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            with patch.object(runner, "_AutoConfigRunner__execute_launch") as mock_execute:
                mock_execute.side_effect = Exception("Launch execution failed")

                with pytest.raises(Exception, match="Launch execution failed"):
                    runner.launch(override_commands=["++test=true"], dryrun=False, max_retry=2)

                assert mock_execute.call_count == 2
                # Verify exponential backoff: first retry sleeps for 2^0 = 1 second
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
        """Test __execute_launch for non-dryrun case"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_exists.return_value = True

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.stdout = "Job launched successfully"
        mock_launcher_instance._execute_command.return_value = mock_output

        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "log_path": "/test/logs/pod.log",
        }
        mock_launcher_instance._parse_output.return_value = job_details
        mock_launcher_instance._monitor_job.return_value = True
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            runner._AutoConfigRunner__base_launch_command = ["python", "main.py"]

            result, success = runner.launch(override_commands=["++test=true"], dryrun=False)

            assert result == job_details
            assert success is True
            mock_launcher_instance._parse_output.assert_called_once()
            mock_launcher_instance._monitor_job.assert_called_once()
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
        """Test __execute_launch raises FileNotFoundError when log file is missing"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config
        mock_exists.return_value = False

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.stdout = "Job launched successfully"
        mock_launcher_instance._execute_command.return_value = mock_output

        job_details = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "log_path": "/test/logs/pod.log",
        }
        mock_launcher_instance._parse_output.return_value = job_details
        mock_launcher_instance._monitor_job.return_value = True
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)
            runner._AutoConfigRunner__base_launch_command = ["python", "main.py"]

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
        """Test __get_job_details when output_folder_path is present"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()

        job_details_from_parse = {
            "job_name": "test-job",
            "pod_name": "test-pod",
            "output_folder_path": "'/test/output/artifacts'",
        }
        mock_launcher_instance._parse_output.return_value = job_details_from_parse
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

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
        """Test __get_job_details when output_folder_path is not present"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()

        job_details_from_parse = {
            "job_name": "test-job",
            "pod_name": "test-pod",
        }
        mock_launcher_instance._parse_output.return_value = job_details_from_parse
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            result = runner._AutoConfigRunner__get_job_details(mock_output)

            assert "config_path" not in result
            assert "log_path" not in result
            assert result == job_details_from_parse

    @patch("auto_configurator.benchmarking.auto_config_runner.os.makedirs")
    @patch("auto_configurator.benchmarking.auto_config_runner.get_common_config")
    @patch("auto_configurator.benchmarking.auto_config_runner.OmegaConf.load")
    @patch("auto_configurator.benchmarking.auto_config_runner.select_validation_launcher")
    @patch("auto_configurator.benchmarking.auto_config_runner.JobRecorder")
    def test_get_job_details_missing_job_name(
        self, mock_recorder, mock_launcher, mock_load, mock_root, mock_makedirs, mock_auto_config, mock_base_config
    ):
        """Test __get_job_details raises ValueError when job_name is missing"""
        mock_root.return_value = "/test/root"
        mock_load.return_value = mock_base_config

        mock_launcher_instance = MagicMock()
        mock_output = MagicMock()

        job_details_from_parse = {
            "pod_name": "test-pod",
            "output_folder_path": "'/test/output/artifacts'",
        }
        mock_launcher_instance._parse_output.return_value = job_details_from_parse
        mock_launcher.return_value.return_value = mock_launcher_instance

        with patch.object(AutoConfigRunner, "_AutoConfigRunner__prepare_base_cfg") as mock_prepare:
            mock_prepare.return_value = mock_base_config
            runner = AutoConfigRunner(mock_auto_config)

            with patch.object(runner, "_AutoConfigRunner__find_hydra_config") as mock_find:
                mock_find.return_value = "/test/output/config.yaml"

                with pytest.raises(ValueError, match="Unable to determine log_path for job"):
                    runner._AutoConfigRunner__get_job_details(mock_output)
