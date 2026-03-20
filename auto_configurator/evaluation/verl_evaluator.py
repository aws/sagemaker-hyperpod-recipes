from auto_configurator.evaluation.base_evaluator import BaseEvaluator


class VerlEvaluator(BaseEvaluator):
    """Evaluation for verl framework logs"""

    def _get_performance_value(self, line: str) -> float:
        """
        Example:
        FIXME
        """
        raise NotImplementedError("_get_performance_value must be implemented")

    def _get_gpu_memory_utilization(self, line: str) -> float:
        """
        Example:
        FIXME
        """
        raise NotImplementedError("_get_gpu_memory_utilization must be implemented")
