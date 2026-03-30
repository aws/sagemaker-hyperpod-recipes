# Auto Configurator

The Auto Configurator automatically optimizes training configurations for HyperPod Recipes by testing different parameter combinations and selecting the best performing configuration based on throughput metrics.

## Overview

The Auto Configurator helps you find optimal training configurations without manual trial-and-error. Instead of manually tuning parameters like batch size, sharding strategy, gradient checkpointing, and CPU offload, you specify which parameters to auto-tune and the system will:

1. Generate candidate configurations based on heuristics and parameter ranges
2. Run short benchmark jobs for each candidate (~15 steps)
3. Measure throughput and memory usage
4. Select the configuration with the highest throughput

## Current Support (Milestone 1)

**Frameworks:** LLMFT (LLM Fine-Tuning)
**Training Methods:** Full Fine-Tuning (FFT) for SFT and DPO
**Platform:** K8s

### Tunable Parameters

For LLMFT recipes, the following parameters can be auto-tuned:

- `train_batch_size` - Training batch size per GPU
- `gradient_checkpointing` - Enable/disable gradient checkpointing (boolean)
- `sharding_strategy` - FSDP sharding strategy (FULL_SHARD, HYBRID_SHARD, NO_SHARD)
- `cpu_offload` - Enable/disable CPU offloading (boolean)

**Parameter Value Options:**

Each parameter supports three configuration modes:

1. **Auto mode**: Set to `auto` to let the optimizer determine the best value, e.g.
   ```yaml
   train_batch_size: auto
   ```

2. **Fixed value**: Specify a single value to use that configuration, e.g.
   ```yaml
   train_batch_size: 16
   sharding_strategy: "FULL_SHARD"
   ```

3. **Range of values**: Provide a list to test multiple configurations, e.g.
   ```yaml
   train_batch_size: [16, 32, 64]
   sharding_strategy: ["FULL_SHARD", "HYBRID_SHARD"]
   gradient_checkpointing: [true, false]
   cpu_offload: [true, false]
   ```

## Usage

### 1. Create an Auto Configuration File

Create a YAML file specifying which parameters to auto-tune. See `auto_configurator/example/auto_config.yaml` for reference:

```yaml
name: llama_3_1_8b_autoconfig

platform: "K8"

# Path from inside recipes_collection/recipes/
recipe: "fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml"

autotune_config:
  sequence_lengths: auto  # or specify a list: [4096, 8192, 16384]
  instance_type_list:
    - "ml.p5.48xlarge"

  llmft:
    train_batch_size: auto
    gradient_checkpointing: auto
    sharding_strategy: auto
    cpu_offload: auto
```

Set parameters to `auto` to enable auto-tuning for that parameter.

**Sequence Length Configuration:**

The `sequence_lengths` parameter supports:
- `auto` (default): Tests all sequence lengths from 4K to 128K (4096, 8192, 16384, 32768, 65536, 131072)
- List of specific lengths: `[4096, 8192]` to test only those sequence lengths

**Parallel Processing:**

To process multiple sequence lengths in parallel, add the `max_workers` parameter:

```bash
python auto_configurator/main.py --config-path <path> --config-name <config> +max_workers=4
```

Default is `max_workers=1` (sequential processing). Increase based on cluster capacity.

**Configuration Inheritance:**

The Auto Configurator uses the validation script's common `scripts/validations/common_validation_config.yaml` as the base configuration. This file is not tracked by git — on first use it is automatically created by copying `scripts/validations/common_validation_config.example.yaml`. You can edit `common_validation_config.yaml` to customize your environment, or override any validation config settings in your `auto_config.yaml` file. For example, to override the container image:

```yaml
container_info:
  llmft:
    k8: "652744875666.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0"
```

### 2. Run the Auto Configurator

```bash
python auto_configurator/main.py --config-path <path-to-config> --config-name <config-file-name>
```

Example:
```bash
python auto_configurator/main.py --config-path auto_configurator/example --config-name auto_config
```

### 3. Review Results

The Auto Configurator will:
- Generate candidate configurations
- Run benchmark jobs for each candidate (limited to 15 training steps)
- Evaluate throughput and memory usage
- Output the optimal configuration

Results are stored in the output directory specified in your config.

## How It Works

1. **Configuration Generation**: Based on the parameters marked as `auto`, the optimizer generates a set of candidate configurations using heuristics (e.g., memory-based decisions for CPU offload, model size-based decisions for gradient checkpointing)

2. **Benchmarking**: Each candidate configuration is tested with a short training run (15 steps) to measure:
   - Throughput (samples/second or tokens/second)
   - GPU memory usage
   - Training stability

3. **Optimization**: If a configuration fails (e.g., OOM error), the optimizer automatically adjusts parameters and retries. The configuration with the highest throughput is selected as optimal.

4. **Output**: The optimal configuration is saved and can be used for full training runs.

## Limitations (Milestone 1)

- Single instance type per run
- LLMFT FFT only (no LoRA support yet)
- K8s platform only
- No support for sequence parallelism, pipeline parallelism, or tensor parallelism
- No fp8 precision support (bf16 only)

## Future Phases

- ~~**Phase 2**: LoRA support for LLMFT (SFT/DPO)~~
- **Phase 3**: VERL framework support (RLAIF, RLVR)


## References

- [Auto Configurator Revival Design Doc](https://quip-amazon.com/HubiAczbopbB/Auto-Configurator-Revival)
- [Original Auto Configurator Design Doc](https://quip-amazon.com/wZTdAm9yPtUc)
