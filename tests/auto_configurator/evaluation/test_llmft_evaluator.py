"""Tests for LlmftEvaluator"""

from unittest.mock import patch

import pytest

from auto_configurator.evaluation.base_evaluator import ErrorCode
from auto_configurator.evaluation.llmft_evaluator import LlmftEvaluator


@pytest.fixture
def mock_evaluator():
    with patch("auto_configurator.evaluation.base_evaluator.get_gpu_memory_gb") as mock_gpu_mem:
        mock_gpu_mem.return_value = 80.0
        evaluator = LlmftEvaluator("ml.p5.48xlarge")
        yield evaluator


class TestIsMaxMemory:
    def test_max_memory_present(self, mock_evaluator):
        line = "[2026-02-20 18:46:04,670][INFO] - [Train:step=14]: {'mem_peak_alloc_max': 20.892120361328125}"
        assert mock_evaluator._is_max_memory(line)

    def test_max_memory_absent(self, mock_evaluator):
        line = "Normal log line without memory info"
        assert not mock_evaluator._is_max_memory(line)


class TestIsPerformance:
    def test_performance_present(self, mock_evaluator):
        line = "'Throughput': '460.44 tokens/second'"
        assert mock_evaluator._is_performance(line)

    def test_performance_absent(self, mock_evaluator):
        line = "Normal log line"
        assert not mock_evaluator._is_performance(line)


class TestIsRunError:
    def test_srun_error(self, mock_evaluator):
        assert mock_evaluator._is_run_error("srun: error: node failed")

    def test_exception_during_training(self, mock_evaluator):
        assert mock_evaluator._is_run_error("Exception occurred during training")

    def test_cuda_unknown_error(self, mock_evaluator):
        assert mock_evaluator._is_run_error("CUDA error: unknown error")

    def test_error_exiting_training(self, mock_evaluator):
        assert mock_evaluator._is_run_error("Error: Something went wrong. Exiting the training")

    def test_error_exiting_training_mixed_case(self, mock_evaluator):
        """Test case-insensitive regex matching"""
        assert mock_evaluator._is_run_error("error: Something went wrong. exiting the training")
        assert mock_evaluator._is_run_error("ERROR: Something went wrong. EXITING THE TRAINING")

    def test_no_run_error(self, mock_evaluator):
        assert not mock_evaluator._is_run_error("Normal log line")


class TestGetPerformanceValue:
    def test_parse_throughput(self, mock_evaluator):
        line = "'Throughput': '460.44 tokens/second'"
        value = mock_evaluator._get_performance_value(line)
        assert value == 460.44

    def test_parse_throughput_with_context(self, mock_evaluator):
        line = "[INFO] - Metrics: 'Throughput': '123.45 tokens/second', 'Loss': 0.5"
        value = mock_evaluator._get_performance_value(line)
        assert value == 123.45

    def test_parse_throughput_case_insensitive(self, mock_evaluator):
        """Test case-insensitive throughput parsing"""
        line = "'throughput': '460.44 tokens/second'"
        value = mock_evaluator._get_performance_value(line)
        assert value == 460.44

    def test_parse_throughput_fails(self, mock_evaluator):
        line = "No throughput here"
        with pytest.raises(ValueError, match="Failed to parse performance metric"):
            mock_evaluator._get_performance_value(line)


class TestGetGpuMemoryUtilization:
    def test_parse_memory(self, mock_evaluator):
        line = "[2026-02-20 18:46:04,670][INFO] - [Train:step=14]: {'mem_peak_alloc_max': 20.892120361328125}"
        memory = mock_evaluator._get_gpu_memory_utilization(line)
        assert memory == 20.892120361328125

    def test_parse_memory_with_other_fields(self, mock_evaluator):
        line = "{'gpt_loss': 8.56, 'mem_peak_alloc_max': 35.5, 'mem_peak_res_max': 40.0}"
        memory = mock_evaluator._get_gpu_memory_utilization(line)
        assert memory == 35.5

    def test_parse_memory_case_insensitive(self, mock_evaluator):
        """Test case-insensitive memory parsing"""
        line = "{'MEM_PEAK_ALLOC_MAX': 20.5}"
        memory = mock_evaluator._get_gpu_memory_utilization(line)
        assert memory == 20.5

    def test_parse_memory_fails(self, mock_evaluator):
        line = "No memory info here"
        with pytest.raises(ValueError, match="Failed to parse max memory"):
            mock_evaluator._get_gpu_memory_utilization(line)


class TestEvaluate:
    def test_successful_run(self, mock_evaluator):
        logs = [
            "[Train:step=2]: {'mem_peak_alloc_max': 65.0}",
            "[Train:step=4]: {'mem_peak_alloc_max': 67.0}",
            "'Throughput': '450.5 tokens/second'",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 450.5
        assert error_code == ErrorCode.NO_ISSUE

    def test_oom_error(self, mock_evaluator):
        logs = [
            "[Train:step=2]: {'mem_peak_alloc_max': 65.0}",
            "torch.cuda.OutOfMemoryError: CUDA out of memory",
            "'Throughput': '450.5 tokens/second'",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 0
        assert error_code == ErrorCode.OOM

    def test_run_error(self, mock_evaluator):
        logs = [
            "[Train:step=2]: {'mem_peak_alloc_max': 65.0}",
            "Exception occurred during training",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 0
        assert error_code == ErrorCode.RUN_ERROR

    def test_low_memory(self, mock_evaluator):
        logs = [
            "[Train:step=2]: {'mem_peak_alloc_max': 40.0}",  # Below 60 (target)
            "'Throughput': '450.5 tokens/second'",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 450.5
        assert error_code == ErrorCode.LOW_MEMORY

    def test_cache_flush(self, mock_evaluator):
        logs = [
            "pytorch allocator cache flushes since last step: 1",
            "pytorch allocator cache flushes since last step: 2",
            "pytorch allocator cache flushes since last step: 3",
            "[Train:step=2]: {'mem_peak_alloc_max': 65.0}",
            "'Throughput': '450.5 tokens/second'",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 450.5
        assert error_code == ErrorCode.CACHE_FLUSH
