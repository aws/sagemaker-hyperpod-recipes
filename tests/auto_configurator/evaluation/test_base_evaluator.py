"""Tests for BaseEvaluator"""

from unittest.mock import patch

import pytest

from auto_configurator.evaluation.base_evaluator import BaseEvaluator, ErrorCode


class ConcreteEvaluator(BaseEvaluator):
    """Concrete implementation for testing"""

    def _get_performance_value(self, line: str) -> float:
        if "Model TFLOPS/GPU:" in line:
            return float(line.split("Model TFLOPS/GPU:")[1].strip())
        return -1

    def _get_gpu_memory_utilization(self, line: str) -> float:
        if "(alloc, max_alloc, cache, max_cache)" in line:
            # Extract max_alloc value - format: "text (alloc, max_alloc, cache, max_cache): (10, 50, 5, 15)"
            # Get the second parenthesis group (the values)
            parts = line.split("(")[2].split(")")[0].split(",")
            max_alloc = parts[1].strip()
            return float(max_alloc)
        return 0.0


@pytest.fixture
def mock_evaluator():
    with patch("auto_configurator.evaluation.base_evaluator.get_gpu_memory_gb") as mock_gpu_mem:
        mock_gpu_mem.return_value = 80.0
        evaluator = ConcreteEvaluator("ml.p5.48xlarge")
        yield evaluator


class TestInit:
    @patch("auto_configurator.evaluation.base_evaluator.get_gpu_memory_gb")
    def test_init(self, mock_gpu_mem):
        mock_gpu_mem.return_value = 80.0

        evaluator = ConcreteEvaluator("ml.p5.48xlarge")

        assert evaluator._gpu_memory == 80.0
        assert evaluator._target_utilization == 60.0  # 75% of 80


class TestIsOutOfMemory:
    def test_torch_oom(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("torch.cuda.OutOfMemoryError: CUDA out of memory")

    def test_cuda_oom(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("CUDA error: out of memory")

    def test_cuda_calloc_failed(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("Failed to CUDA calloc async")

    def test_cuda_failure(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("Cuda failure 2 'out of memory'")

    def test_cudnn_error(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("cuDNN Error: CUDNN_STATUS_BAD_PARAM")

    def test_cublas_error(self, mock_evaluator):
        assert mock_evaluator._is_out_of_memory("CUBLAS_STATUS_ALLOC_FAILED")

    def test_no_oom(self, mock_evaluator):
        assert not mock_evaluator._is_out_of_memory("Normal log line")


class TestIsCacheFlush:
    def test_cache_flush(self, mock_evaluator):
        assert mock_evaluator._is_cache_flush("pytorch allocator cache flushes since last step: 5")

    def test_no_cache_flush(self, mock_evaluator):
        assert not mock_evaluator._is_cache_flush("Normal log line")


class TestIsRunError:
    def test_srun_error(self, mock_evaluator):
        assert mock_evaluator._is_run_error("srun: error: node failed")

    def test_nccl_error(self, mock_evaluator):
        assert mock_evaluator._is_run_error("NCCL WARN NET/OFI Request completed with error")

    def test_no_run_error(self, mock_evaluator):
        assert not mock_evaluator._is_run_error("Normal log line")


class TestIsMaxMemory:
    def test_max_memory(self, mock_evaluator):
        assert mock_evaluator._is_max_memory("Memory usage (alloc, max_alloc, cache, max_cache): (10, 20, 5, 15)")

    def test_no_max_memory(self, mock_evaluator):
        assert not mock_evaluator._is_max_memory("Normal log line")


class TestIsPerformance:
    def test_performance(self, mock_evaluator):
        assert mock_evaluator._is_performance("Model TFLOPS/GPU: 123.45")

    def test_no_performance(self, mock_evaluator):
        assert not mock_evaluator._is_performance("Normal log line")


class TestEvaluate:
    def test_run_error(self, mock_evaluator):
        logs = ["Normal log", "srun: error: node failed", "More logs"]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 0
        assert error_code == ErrorCode.RUN_ERROR

    def test_oom_error(self, mock_evaluator):
        logs = ["Normal log", "torch.cuda.OutOfMemoryError: CUDA out of memory", "More logs"]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 0
        assert error_code == ErrorCode.OOM

    def test_cache_flush_high_count(self, mock_evaluator):
        logs = [
            "pytorch allocator cache flushes since last step: 1",
            "pytorch allocator cache flushes since last step: 2",
            "pytorch allocator cache flushes since last step: 3",
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 50, 5, 15)",
            "Model TFLOPS/GPU: 100.0",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 100.0
        assert error_code == ErrorCode.CACHE_FLUSH

    def test_high_memory_usage(self, mock_evaluator):
        logs = [
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 79, 5, 15)",  # 79 > 80 - 2
            "Model TFLOPS/GPU: 100.0",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 100.0
        assert error_code == ErrorCode.CACHE_FLUSH

    def test_low_memory_usage(self, mock_evaluator):
        logs = [
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 40, 5, 15)",  # 40 < 60 (target)
            "Model TFLOPS/GPU: 100.0",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 100.0
        assert error_code == ErrorCode.LOW_MEMORY

    def test_no_issue(self, mock_evaluator):
        logs = [
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 65, 5, 15)",  # 60 < 65 < 78
            "Model TFLOPS/GPU: 150.0",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 150.0
        assert error_code == ErrorCode.NO_ISSUE

    def test_max_memory_tracking(self, mock_evaluator):
        logs = [
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 50, 5, 15)",
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 70, 5, 15)",  # Higher
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 60, 5, 15)",  # Lower, ignored
            "Model TFLOPS/GPU: 100.0",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == 100.0
        assert error_code == ErrorCode.NO_ISSUE  # 70 is between 60 and 78

    def test_no_performance_metric(self, mock_evaluator):
        logs = [
            "Memory usage (alloc, max_alloc, cache, max_cache): (10, 65, 5, 15)",
            "Normal log without performance",
        ]

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == -1  # No performance metric found
        assert error_code == ErrorCode.NO_ISSUE

    def test_empty_logs(self, mock_evaluator):
        logs = []

        metric, error_code = mock_evaluator.evaluate(logs)

        assert metric == -1
        assert error_code == ErrorCode.LOW_MEMORY  # 0 < target_utilization
