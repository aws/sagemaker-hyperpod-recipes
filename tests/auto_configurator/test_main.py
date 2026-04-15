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

from auto_configurator.evaluation.base_evaluator import ErrorCode
from auto_configurator.main import (
    FAILED,
    _write_results_csv,
    find_best_candidate,
    optimize_candidate,
    process,
    process_sequence_length,
    select_config_optimizer,
)
from auto_configurator.utils.util import OptimizerType


class TestSelectConfigOptimizer:
    def test_llmft_optimizer(self):
        from auto_configurator.config_optimizer.llmft_optimizer import LlmftOptimizer
        from auto_configurator.evaluation.llmft_evaluator import LlmftEvaluator

        optimizer_cls, evaluator_cls = select_config_optimizer(OptimizerType.LLMFT)
        assert optimizer_cls == LlmftOptimizer
        assert evaluator_cls == LlmftEvaluator

    def test_verl_optimizer(self):
        from auto_configurator.config_optimizer.verl_optimizer import VerlOptimizer
        from auto_configurator.evaluation.verl_evaluator import VerlEvaluator

        optimizer_cls, evaluator_cls = select_config_optimizer(OptimizerType.VERL)
        assert optimizer_cls == VerlOptimizer
        assert evaluator_cls == VerlEvaluator

    def test_unknown_optimizer(self):
        with pytest.raises(ValueError, match="Unknown config_optimizer_type"):
            select_config_optimizer("unknown")


class TestOptimizeCandidate:
    @pytest.fixture
    def mock_optimizer(self):
        optimizer = MagicMock()
        optimizer.get_recipe_overrides.return_value = ["++batch_size=32"]
        optimizer.tune_candidate.return_value = ({"batch_size": 16}, False)
        return optimizer

    @pytest.fixture
    def mock_evaluator(self):
        evaluator = MagicMock()
        evaluator.evaluate.return_value = (500.0, ErrorCode.NO_ISSUE)
        return evaluator

    @pytest.fixture
    def mock_job_runner(self):
        runner = MagicMock()
        runner.launch.return_value = ({"log_path": "/tmp/test.log", "config_path": "/tmp/config.yaml"}, True)
        return runner

    def test_successful_optimization(self, mock_optimizer, mock_evaluator, mock_job_runner):
        candidate = {"batch_size": 32}

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = ["log line 1"]

            best_candidate, best_metric, recipe_path, error_code = optimize_candidate(
                mock_optimizer, mock_evaluator, mock_job_runner, candidate, tried_configs=set()
            )

            assert best_metric == 500.0
            assert best_candidate == {"batch_size": 32}
            assert recipe_path == "/tmp/config.yaml"
            assert error_code == ErrorCode.NO_ISSUE

    def test_optimization_with_retry(self, mock_optimizer, mock_evaluator, mock_job_runner):
        candidate = {"batch_size": 32}
        mock_evaluator.evaluate.side_effect = [(0, ErrorCode.OOM), (450.0, ErrorCode.NO_ISSUE)]
        mock_optimizer.tune_candidate.side_effect = [({"batch_size": 16}, True), ({"batch_size": 16}, False)]

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = ["log line 1"]

            best_candidate, best_metric, recipe_path, error_code = optimize_candidate(
                mock_optimizer, mock_evaluator, mock_job_runner, candidate, tried_configs=set()
            )

            assert best_metric == 450.0
            assert mock_job_runner.launch.call_count == 2

    def test_optimization_stops_on_throughput_decrease(self, mock_optimizer, mock_evaluator, mock_job_runner):
        """Test that optimization stops when throughput decreases with LOW_MEMORY"""
        candidate = {"batch_size": 32}
        # First run: good throughput with LOW_MEMORY (will try to increase batch)
        # Second run: worse throughput with LOW_MEMORY (should stop)
        mock_evaluator.evaluate.side_effect = [(500.0, ErrorCode.LOW_MEMORY), (400.0, ErrorCode.LOW_MEMORY)]
        mock_optimizer.tune_candidate.return_value = ({"batch_size": 64}, True)

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = ["log line 1"]

            best_candidate, best_metric, recipe_path, error_code = optimize_candidate(
                mock_optimizer, mock_evaluator, mock_job_runner, candidate, tried_configs=set()
            )

            assert best_metric == 500.0
            assert best_candidate == {"batch_size": 32}
            # Should have run twice: first to get 500.0, second to get 400.0 and stop
            assert mock_job_runner.launch.call_count == 2

    def test_optimization_stops_on_run_error(self, mock_optimizer, mock_evaluator, mock_job_runner):
        """Test that optimization stops immediately on RUN_ERROR"""
        candidate = {"batch_size": 32}
        mock_evaluator.evaluate.return_value = (0, ErrorCode.RUN_ERROR)
        mock_optimizer.tune_candidate.return_value = ({"batch_size": 16}, False)  # should_retry=False

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = ["log line 1"]

            best_candidate, best_metric, recipe_path, error_code = optimize_candidate(
                mock_optimizer, mock_evaluator, mock_job_runner, candidate, tried_configs=set()
            )

            # Should stop after tune_candidate returns should_retry=False
            assert mock_job_runner.launch.call_count == 1
            assert best_metric == 0
            assert error_code == ErrorCode.RUN_ERROR

    def test_optimization_stops_on_duplicate_config(self, mock_optimizer, mock_evaluator, mock_job_runner):
        """Test that optimization stops when trying a duplicate configuration"""
        candidate = {"batch_size": 32}
        mock_evaluator.evaluate.side_effect = [(0, ErrorCode.OOM), (450.0, ErrorCode.NO_ISSUE)]
        # tune_candidate returns the same config, which should be detected as duplicate
        mock_optimizer.tune_candidate.return_value = ({"batch_size": 32}, True)

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.readlines.return_value = ["log line 1"]

            best_candidate, best_metric, recipe_path, error_code = optimize_candidate(
                mock_optimizer, mock_evaluator, mock_job_runner, candidate, tried_configs=set()
            )

            # Should run once, then detect duplicate and stop
            assert mock_job_runner.launch.call_count == 1
            assert best_metric == 0


class TestFindBestCandidate:
    def test_single_candidate(self):
        mock_optimizer = MagicMock()
        mock_evaluator = MagicMock()
        mock_job_runner = MagicMock()

        candidates = [{"batch_size": 32}]

        with patch("auto_configurator.main.optimize_candidate") as mock_optimize:
            mock_optimize.return_value = ({"batch_size": 32}, 500.0, "/tmp/config.yaml", ErrorCode.NO_ISSUE)

            result, error_code = find_best_candidate(mock_job_runner, mock_optimizer, mock_evaluator, candidates)

            assert result == "/tmp/config.yaml"
            assert error_code is None

    def test_multiple_candidates(self):
        mock_optimizer = MagicMock()
        mock_evaluator = MagicMock()
        mock_job_runner = MagicMock()

        candidates = [{"batch_size": 16}, {"batch_size": 32}, {"batch_size": 64}]

        with patch("auto_configurator.main.optimize_candidate") as mock_optimize:
            mock_optimize.side_effect = [
                ({"batch_size": 16}, 300.0, "/tmp/config1.yaml", ErrorCode.NO_ISSUE),
                ({"batch_size": 32}, 500.0, "/tmp/config2.yaml", ErrorCode.NO_ISSUE),
                ({"batch_size": 64}, 450.0, "/tmp/config3.yaml", ErrorCode.NO_ISSUE),
            ]

            result, error_code = find_best_candidate(mock_job_runner, mock_optimizer, mock_evaluator, candidates)

            assert result == "/tmp/config2.yaml"
            assert error_code is None

    def test_no_valid_candidates(self):
        mock_optimizer = MagicMock()
        mock_evaluator = MagicMock()
        mock_job_runner = MagicMock()

        candidates = [{"batch_size": 32}]

        with patch("auto_configurator.main.optimize_candidate") as mock_optimize:
            mock_optimize.return_value = ({"batch_size": 32}, 0, "", ErrorCode.OOM)

            result, error_code = find_best_candidate(mock_job_runner, mock_optimizer, mock_evaluator, candidates)

            assert result == ""
            assert error_code == ErrorCode.OOM


class TestProcessSequenceLength:
    @pytest.fixture
    def mock_job_runner(self):
        runner = MagicMock()
        runner.auto_config.instance_type = "ml.p5.48xlarge"
        runner.auto_config.base_results_dir = "/tmp/output"
        runner.auto_config.recipe = "fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_fft.yaml"
        return runner

    def test_successful_processing(self, mock_job_runner):
        mock_optimizer = MagicMock()
        mock_optimizer.generate_candidate_configurations.return_value = [{"batch_size": 32}]
        mock_evaluator = MagicMock()

        with patch("auto_configurator.main.find_best_candidate") as mock_find_best, patch(
            "auto_configurator.main.copy_file"
        ) as mock_copy:
            mock_find_best.return_value = ("/tmp/recipe.yaml", None)
            expected_path = "/tmp/output/results/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_fft/ml_p5_48xlarge/max_len_4096.yaml"
            mock_copy.return_value = expected_path

            result = process_sequence_length(mock_job_runner, mock_optimizer, mock_evaluator, 4096)

            mock_copy.assert_called_once_with("/tmp/recipe.yaml", expected_path)
            assert result == expected_path
            mock_optimizer.generate_candidate_configurations.assert_called_once_with(max_len=4096)

    def test_no_valid_config(self, mock_job_runner):
        mock_optimizer = MagicMock()
        mock_optimizer.generate_candidate_configurations.return_value = [{"batch_size": 32}]
        mock_evaluator = MagicMock()

        with patch("auto_configurator.main.find_best_candidate") as mock_find_best:
            mock_find_best.return_value = ("", ErrorCode.OOM)

            result = process_sequence_length(mock_job_runner, mock_optimizer, mock_evaluator, 4096)

            assert result == "OOM"

    def test_no_valid_config_no_error_code(self, mock_job_runner):
        mock_optimizer = MagicMock()
        mock_optimizer.generate_candidate_configurations.return_value = [{"batch_size": 32}]
        mock_evaluator = MagicMock()

        with patch("auto_configurator.main.find_best_candidate") as mock_find_best:
            mock_find_best.return_value = ("", None)

            result = process_sequence_length(mock_job_runner, mock_optimizer, mock_evaluator, 4096)

            assert result == FAILED

    def test_copy_file_fails(self, mock_job_runner):
        """Test when copy_file fails to save config"""
        mock_optimizer = MagicMock()
        mock_optimizer.generate_candidate_configurations.return_value = [{"batch_size": 32}]
        mock_evaluator = MagicMock()

        with patch("auto_configurator.main.find_best_candidate") as mock_find_best, patch(
            "auto_configurator.main.copy_file"
        ) as mock_copy:
            mock_find_best.return_value = ("/tmp/recipe.yaml", None)
            mock_copy.return_value = ""  # copy_file failed

            result = process_sequence_length(mock_job_runner, mock_optimizer, mock_evaluator, 4096)

            assert result == FAILED


@patch("auto_configurator.main.AutoConfigRunner")
@patch("auto_configurator.main.get_optimizer_type")
@patch("auto_configurator.main.select_config_optimizer")
@patch("auto_configurator.main.get_sequence_length_range")
@patch("auto_configurator.main.process_sequence_length")
class TestProcess:
    """Tests for process() which handles a single instance type."""

    def test_process_single_worker(
        self, mock_process_sl, mock_get_seq_lens, mock_select, mock_get_type, mock_runner_cls
    ):
        mock_get_type.return_value = OptimizerType.LLMFT
        mock_get_seq_lens.return_value = [2048, 4096]

        mock_optimizer_cls = MagicMock()
        mock_evaluator_cls = MagicMock()
        mock_select.return_value = (mock_optimizer_cls, mock_evaluator_cls)

        mock_runner = MagicMock()
        mock_runner.base_recipe = OmegaConf.create({})
        mock_runner_cls.return_value = mock_runner

        mock_process_sl.return_value = "/tmp/config.yaml"

        cfg = OmegaConf.create(
            {
                "name": "test",
                "recipe": "training/llama/llmft_llama3_8b",
                "platform": "k8s",
                "instance_type": "ml.p5.48xlarge",
                "autotune_config": {"llmft": {}},
                "max_workers": 1,
            }
        )

        results = process(cfg)

        assert mock_process_sl.call_count == 2
        assert 2048 in results
        assert 4096 in results

    def test_process_handles_exception(
        self, mock_process_sl, mock_get_seq_lens, mock_select, mock_get_type, mock_runner_cls
    ):
        mock_get_type.return_value = OptimizerType.LLMFT
        mock_get_seq_lens.return_value = [2048, 4096]

        mock_optimizer_cls = MagicMock()
        mock_evaluator_cls = MagicMock()
        mock_select.return_value = (mock_optimizer_cls, mock_evaluator_cls)

        mock_runner = MagicMock()
        mock_runner.base_recipe = OmegaConf.create({})
        mock_runner_cls.return_value = mock_runner

        mock_process_sl.side_effect = [Exception("Test error"), "/tmp/config.yaml"]

        cfg = OmegaConf.create(
            {
                "name": "test",
                "recipe": "training/llama/llmft_llama3_8b",
                "platform": "k8s",
                "instance_type": "ml.p5.48xlarge",
                "autotune_config": {"llmft": {}},
            }
        )

        results = process(cfg)

        assert mock_process_sl.call_count == 2
        # One should be the exception message, one should be the config path
        values = list(results.values())
        assert any("Test error" in str(v) for v in values)
        assert any(v == "/tmp/config.yaml" for v in values)


class TestWriteResultsCsv:
    def test_writes_csv(self, tmp_path):
        all_results = {
            "ml.p5.48xlarge": {4096: "/path/a.yaml", 8192: "OOM"},
            "ml.p4d.24xlarge": {4096: "/path/b.yaml", 8192: "/path/c.yaml"},
        }
        _write_results_csv(str(tmp_path), "fine-tuning/llama/llmft_llama3_8b.yaml", all_results)

        csv_path = tmp_path / "results" / "llmft_llama3_8b_summary.csv"
        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == "instance_type,4096,8192"
        assert len(lines) == 3

    def test_empty_results(self, tmp_path):
        _write_results_csv(str(tmp_path), "recipe.yaml", {})
        assert not (tmp_path / "results" / "recipe_summary.csv").exists()


@patch("auto_configurator.main._write_results_csv")
@patch("auto_configurator.main.process")
@patch("auto_configurator.main.validate_cluster")
@patch("auto_configurator.main.get_instance_type_list")
class TestMain:
    def _make_cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "name": "test",
                "recipe": "training/llama/llmft_llama3_8b",
                "platform": "k8s",
                "base_results_dir": str(tmp_path),
                "autotune_config": {"llmft": {}, "instance_type_list": ["ml.p5.48xlarge"]},
                "k8": {"cluster_context_map": {"ml.p5.48xlarge": "ctx"}},
            }
        )

    def test_runs_instance_types_and_writes_csv(
        self, mock_get_types, mock_validate, mock_process, mock_write_csv, tmp_path
    ):
        mock_get_types.return_value = ["ml.p5.48xlarge", "ml.p4d.24xlarge"]
        mock_process.return_value = {4096: "/path/a.yaml"}
        cfg = self._make_cfg(tmp_path)

        from auto_configurator.main import main

        main(cfg)

        assert mock_validate.call_count == 2
        assert mock_process.call_count == 2
        mock_write_csv.assert_called_once()
        all_results = mock_write_csv.call_args[0][2]
        assert "ml.p5.48xlarge" in all_results
        assert "ml.p4d.24xlarge" in all_results

    def test_skips_instance_type_on_validate_failure(
        self, mock_get_types, mock_validate, mock_process, mock_write_csv, tmp_path
    ):
        mock_get_types.return_value = ["ml.p5.48xlarge", "ml.p4d.24xlarge"]

        def _validate(instance_type, _cfg):
            if instance_type == "ml.p5.48xlarge":
                raise RuntimeError("no nodes")

        mock_validate.side_effect = _validate
        mock_process.return_value = {4096: "/path/a.yaml"}
        cfg = self._make_cfg(tmp_path)

        from auto_configurator.main import main

        main(cfg)

        assert mock_process.call_count == 1
        mock_write_csv.assert_called_once()
        all_results = mock_write_csv.call_args[0][2]
        assert "ml.p5.48xlarge" not in all_results
        assert "ml.p4d.24xlarge" in all_results

    def test_sets_default_base_results_dir(self, mock_get_types, mock_validate, mock_process, mock_write_csv, tmp_path):
        mock_get_types.return_value = []
        cfg = OmegaConf.create(
            {
                "name": "test",
                "recipe": "training/llama/llmft_llama3_8b",
                "platform": "k8s",
                "autotune_config": {"llmft": {}},
                "k8": {},
            }
        )

        from auto_configurator.main import main

        main(cfg)

        assert "base_results_dir" in cfg
        mock_write_csv.assert_called_once()
