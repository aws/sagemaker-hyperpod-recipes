import enum
import logging
from abc import ABC, abstractmethod
from typing import Tuple

from auto_configurator.utils.util import get_gpu_memory_gb


class ErrorCode(enum.Enum):
    NO_ISSUE = 1
    CACHE_FLUSH = -1
    OOM = -2
    RUN_ERROR = -3
    LOW_MEMORY = 2


class BaseEvaluator(ABC):
    """
    Base class for framework-specific log evaluation
    """

    def __init__(self, instance_type: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._gpu_memory: float = get_gpu_memory_gb(instance_type)
        self._target_utilization = 0.75 * self._gpu_memory  # target 75% utilization

    def _is_out_of_memory(self, line: str) -> bool:
        """Check if line indicates OOM error"""
        return (
            "torch.cuda.OutOfMemoryError" in line
            or "CUDA error: out of memory" in line
            or "Failed to CUDA calloc async" in line
            or "Cuda failure 2 'out of memory'" in line
            or "cuDNN Error: CUDNN_STATUS_BAD_PARAM" in line
            or "CUBLAS_STATUS_ALLOC_FAILED" in line
        )

    def _is_cache_flush(self, line: str) -> bool:
        """Check if line indicates cache flush"""
        return "pytorch allocator cache flushes" in line

    def _is_run_error(self, line: str) -> bool:
        """Check if line indicates run error"""
        return "srun: error" in line or "NCCL WARN NET/OFI Request completed with error" in line

    def _is_max_memory(self, line: str) -> bool:
        """Check if line contains max memory info"""
        return "(alloc, max_alloc, cache, max_cache)" in line

    def _is_performance(self, line: str) -> bool:
        """Check if line contains performance metrics"""
        return "Model TFLOPS/GPU:" in line

    @abstractmethod
    def _get_performance_value(self, line: str) -> float:
        """Extract performance metric from line"""

    @abstractmethod
    def _get_gpu_memory_utilization(self, line: str) -> float:
        """Extract max memory from line"""

    def evaluate(self, logs: list) -> Tuple[float, ErrorCode]:
        """
        Evaluate logs and return (metric, error_code)
        """
        cache_flush_count = 0
        max_memory = 0.0
        metric = -1

        for line in logs:
            if self._is_run_error(line):
                return 0, ErrorCode.RUN_ERROR
            if self._is_out_of_memory(line):
                return 0, ErrorCode.OOM
            if self._is_cache_flush(line):
                cache_flush_count += 1
            if self._is_max_memory(line):
                max_memory = max(max_memory, self._get_gpu_memory_utilization(line))
            if self._is_performance(line):
                metric = self._get_performance_value(line)

        self.logger.info(f"Max memory: {max_memory} GB")
        self.logger.info(f"Performance metric value: {metric}")

        if cache_flush_count > 2 or max_memory > self._gpu_memory - 2:
            return metric, ErrorCode.CACHE_FLUSH
        if max_memory < self._target_utilization:
            return metric, ErrorCode.LOW_MEMORY
        return metric, ErrorCode.NO_ISSUE
