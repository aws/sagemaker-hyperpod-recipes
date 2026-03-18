import re

from auto_configurator.evaluation.base_evaluator import BaseEvaluator


class LlmftEvaluator(BaseEvaluator):
    """Evaluation for LLMFT framework logs"""

    THROUGHPUT_REGEX = re.compile(r"'Throughput':\s*'([\d.]+)\s*tokens/second'", re.IGNORECASE)
    MEMORY_REGEX = re.compile(r"'mem_peak_alloc_max':\s*([\d.]+)", re.IGNORECASE)
    ERROR_EXITING_REGEX = re.compile(r"Error: .+?\. Exiting the training", re.IGNORECASE)

    def _is_max_memory(self, line: str) -> bool:
        """
        Example:
        [2026-02-20 18:46:04,670][amzn_awsllm_fine_tuning.trainer.base_trainer][INFO] - [Train:step=14 (epoch=0)]: {'gpt_loss': 8.561914443969727, 'total_loss': 8.561914443969727, 'lr': 5e-05, 'grad_norm': 258.68408203125, 'total_tokens_in_step': 8302.0, 'mem_peak_alloc_max': 20.892120361328125, 'mem_peak_res_max': 33.048828125}
        """
        return "mem_peak_alloc_max" in line

    def _is_performance(self, line: str) -> bool:
        """
        Example:
        'Throughput': '460.44 tokens/second'
        """
        return "'Throughput':" in line

    def _is_run_error(self, line: str) -> bool:
        """Check if line indicates run error"""
        return (
            "srun: error" in line.lower()
            or "exception occurred during training" in line.lower()
            or "cuda error: unknown error" in line.lower()
            or bool(self.ERROR_EXITING_REGEX.search(line))
        )

    def _get_performance_value(self, line: str) -> float:
        """
        Example:
        'Throughput': '460.44 tokens/second'
        """
        match = self.THROUGHPUT_REGEX.search(line)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Failed to parse performance metric for line: {line}")

    def _get_gpu_memory_utilization(self, line: str) -> float:
        """
        Example:
        [2026-02-20 18:46:04,670][amzn_awsllm_fine_tuning.trainer.base_trainer][INFO] - [Train:step=14 (epoch=0)]: {'gpt_loss': 8.561914443969727, 'total_loss': 8.561914443969727, 'lr': 5e-05, 'grad_norm': 258.68408203125, 'total_tokens_in_step': 8302.0, 'mem_peak_alloc_max': 20.892120361328125, 'mem_peak_res_max': 33.048828125}

        Note: Memory is in GB
        """
        match = self.MEMORY_REGEX.search(line)
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Failed to parse max memory for line: {line}")
