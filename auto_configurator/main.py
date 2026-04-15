"""
Auto-configurator main entry point.

Finds optimal training configuration by testing recipes and analyzing performance.
"""

import csv
import sys
from pathlib import Path

# Add workspace root to Python path FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

from concurrent.futures import ThreadPoolExecutor, as_completed

FAILED = "X"

import hydra
from omegaconf import OmegaConf

from auto_configurator.benchmarking.auto_config_runner import AutoConfigRunner
from auto_configurator.config_optimizer.base_optimizer import BaseOptimizer
from auto_configurator.config_optimizer.llmft_optimizer import LlmftOptimizer
from auto_configurator.config_optimizer.verl_optimizer import VerlOptimizer
from auto_configurator.evaluation.base_evaluator import BaseEvaluator, ErrorCode
from auto_configurator.evaluation.llmft_evaluator import LlmftEvaluator
from auto_configurator.evaluation.verl_evaluator import VerlEvaluator
from auto_configurator.utils.cluster_config import (
    get_instance_type_list,
    validate_cluster,
)
from auto_configurator.utils.util import (
    AutoConfiguratorLogger,
    OptimizerType,
    copy_file,
    format_params,
    get_optimizer_type,
    get_sequence_length_range,
    prettify,
)
from scripts.validations.validation_launchers.path_utils import get_project_root

logger = AutoConfiguratorLogger().get_logger()


def select_config_optimizer(optimizer_type: OptimizerType) -> tuple[type[BaseOptimizer], type[BaseEvaluator]]:
    match optimizer_type:
        case OptimizerType.LLMFT:
            return LlmftOptimizer, LlmftEvaluator
        case OptimizerType.VERL:
            return VerlOptimizer, VerlEvaluator
        case _:
            raise ValueError(f"Unknown config_optimizer_type: {optimizer_type}")


def optimize_candidate(
    optimizer: BaseOptimizer,
    evaluator: BaseEvaluator,
    job_runner: AutoConfigRunner,
    candidate: dict,
    tried_configs: set | None = None,
) -> tuple[dict, float, str]:
    """
    Optimize a single candidate configuration by iteratively running jobs and tuning based on errors.

    Launches training jobs with the candidate configuration, evaluates performance from logs,
    and adjusts parameters based on error codes (OOM, low memory, etc.) until optimal or no issues.

    Args:
        optimizer: Optimizer instance to generate overrides and tune candidates
        evaluator: Evaluator instance to assess job performance from logs
        job_runner: Runner to launch and monitor training jobs
        candidate: Candidate configuration dict with parameters to optimize
        tried_configs: Set tracking tested configurations to avoid duplicate run attempts
    Returns:
        Tuple of (best_candidate, best_metric, config_path):
            - best_candidate: Configuration dict that achieved best performance
            - best_metric: Best throughput metric (tokens/sec)
            - config_path: Path to the config file for the best candidate
    """
    tried_configs = tried_configs if tried_configs is not None else set()

    best_metric = 0
    best_candidate = candidate.copy()
    config_path = ""

    logger.info(f"Testing candidate: {prettify(candidate)}")

    metric, error_code = (-1, None)
    while error_code != ErrorCode.NO_ISSUE:
        # Check if we've already tried this configuration
        config_key = format_params(candidate)
        if config_key in tried_configs:
            logger.info(f"Already tried configuration {config_key}, stopping optimization")
            break
        tried_configs.add(config_key)

        recipe_overrides = optimizer.get_recipe_overrides(candidate)
        job_details, _ = job_runner.launch(recipe_overrides)

        with open(job_details["log_path"]) as f:
            metric, error_code = evaluator.evaluate(f.readlines())

        if error_code in [ErrorCode.NO_ISSUE, ErrorCode.CACHE_FLUSH, ErrorCode.LOW_MEMORY]:
            logger.info(f"Throughput: {metric} tokens/sec")
            if metric > best_metric:
                best_metric = metric
                best_candidate = candidate.copy()
                config_path = job_details["config_path"]
            elif error_code == ErrorCode.LOW_MEMORY and best_metric > 0:
                # Throughput decreased, stop increasing batch size
                logger.info(f"Throughput decreased ({metric} < {best_metric}), stopping optimization")
                break

        # Let optimizer tune the candidate (unhandled errors like RUN_ERROR
        # return should_retry=False, stopping optimization for this candidate)
        logger.info(f"Error code detected: {error_code}")
        candidate, should_retry = optimizer.tune_candidate(candidate, error_code, tried_configs)
        if not should_retry:
            logger.info(f"Adjusting candidate, no retry with candidate: {prettify(candidate)}")
            break
        else:
            logger.info(f"Adjusting candidate, retry with candidate: {prettify(candidate)}")

    return (best_candidate, best_metric, config_path, error_code)


def find_best_candidate(job_runner, optimizer, evaluator, candidates):
    """Find best candidate by optimizing each one"""
    tried_configs = set()
    results = [optimize_candidate(optimizer, evaluator, job_runner, c, tried_configs) for c in candidates]

    if not results or all(metric == 0 for _, metric, _, _ in results):
        last_error = results[-1][3] if results else None
        logger.warning("No valid configurations found")
        return "", last_error

    best_candidate, best_metric, config_path, _ = max(results, key=lambda x: x[1])
    logger.info(f"Best candidate: {best_candidate} with metric: {best_metric}")
    return config_path, None


def process_sequence_length(job_runner, optimizer, evaluator, sequence_length) -> str:
    """Process a single sequence length and return the config path or error code."""
    logger.info(f"Fetching candidates for sequence length: {sequence_length}")

    candidates = optimizer.generate_candidate_configurations(max_len=sequence_length)
    recipe, error_code = find_best_candidate(job_runner, optimizer, evaluator, candidates)

    instance_type = job_runner.auto_config.instance_type
    logger.info(f"Auto configurator completed for {instance_type} with sequence_length {sequence_length}.")

    if not recipe:
        logger.error(f"Auto-configurator failed to find valid configuration for sequence_length {sequence_length}")
        return error_code.name if error_code else FAILED

    recipe_name = Path(job_runner.auto_config.recipe).stem
    config_path = copy_file(
        recipe,
        f"{job_runner.auto_config.base_results_dir}/results/{recipe_name}/{instance_type.replace('.', '_')}/max_len_{sequence_length}.yaml",
    )

    if config_path:
        logger.info(f"Config saved at {config_path}")
    else:
        logger.error("Failed to save config.")

    return config_path or FAILED


def process(cfg):
    instance_type = cfg.instance_type
    logger.info(f"Auto-configurator started for {cfg.recipe} with instance_type: {instance_type}")

    # Create job runner
    job_runner = AutoConfigRunner(cfg)

    # Create optimizer and evaluator
    optimizer_type = get_optimizer_type(cfg.recipe)
    optimizer_cls, evaluator_cls = select_config_optimizer(optimizer_type)

    optimizer = optimizer_cls(cfg.autotune_config.get(optimizer_type.value), job_runner.base_recipe, instance_type)
    evaluator = evaluator_cls(instance_type)

    # Find best recipe for each sequence length
    sequence_lengths = get_sequence_length_range(cfg)
    max_workers = cfg.get("max_workers", 1)

    results = {}

    logger.info(f"Processing {len(sequence_lengths)} sequence lengths with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_sequence_length, job_runner, optimizer, evaluator, seq_len): seq_len
            for seq_len in sequence_lengths
        }

        for future in as_completed(futures):
            seq_len = futures[future]
            try:
                results[seq_len] = future.result()
            except Exception as e:
                logger.error(f"Auto configurator failed for {instance_type} with sequence_length {seq_len}: {e}")
                results[seq_len] = str(e)

    return results


def _write_results_csv(base_results_dir, recipe, all_results):
    """Write a pivot CSV: rows=instance_type, cols=sequence_length, cells=config_path."""
    if not all_results:
        return

    recipe_name = Path(recipe).stem
    seq_lens = sorted({sl for r in all_results.values() for sl in r})
    csv_path = Path(base_results_dir) / "results" / f"{recipe_name}_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_type"] + [str(sl) for sl in seq_lens])
        for instance_type, results in sorted(all_results.items()):
            writer.writerow([instance_type] + [results.get(sl, "") for sl in seq_lens])

    logger.info(f"Results summary written to {csv_path}")


@hydra.main(version_base="1.2", config_path="example", config_name="auto_config")
def main(cfg):
    """
    Run auto-configurator to find optimal training configuration.

    Usage:
        python auto_configurator/auto_config_runner.py
        python auto_configurator/auto_config_runner.py --config-path /path/to/configs --config-name my_auto_config

    Args:
        cfg: Auto-configurator config from auto_config.yaml
    """

    if "base_results_dir" not in cfg:
        OmegaConf.set_struct(cfg, False)
        cfg.base_results_dir = f"{get_project_root()}/auto_configurator/benchmarking/output"
        OmegaConf.set_struct(cfg, True)

    instance_type_list = get_instance_type_list(cfg)
    all_results = {}  # {instance_type: {seq_len: config_path}}

    def _run_instance_type(instance_type):
        cfg_instance = OmegaConf.create(OmegaConf.to_container(cfg))
        cfg_instance.instance_type = instance_type
        validate_cluster(instance_type, cfg)
        return process(cfg_instance)

    with ThreadPoolExecutor(max_workers=max(len(instance_type_list), 1)) as executor:
        futures = {executor.submit(_run_instance_type, it): it for it in instance_type_list}
        for future in as_completed(futures):
            instance_type = futures[future]
            try:
                all_results[instance_type] = future.result()
            except Exception as e:
                logger.warning(f"Unable to run auto configurator for instance type {instance_type}: {e}")

    _write_results_csv(cfg.base_results_dir, cfg.recipe, all_results)


if __name__ == "__main__":
    main()
