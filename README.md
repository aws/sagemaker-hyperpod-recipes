# Amazon SageMaker HyperPod Recipes

## Overview

Amazon SageMaker HyperPod recipes help customers get started with training and fine-tuning popular publicly available foundation models in just minutes, with state-of-the-art performance. They provide a pre-configured training stack that is tested and validated on Amazon SageMaker.

Please see [Amazon SageMaker HyperPod recipes documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html) for full documentation.

The recipes support Amazon SageMaker HyperPod (with Amazon EKS for workload orchestration) and Amazon SageMaker training jobs.

This document provides recipes and guidance for training and fine-tuning models.

## Version History

This repository contains **v2.0.0** of Amazon SageMaker HyperPod recipes, which includes recipes built on the latest training frameworks.

**Looking for v1 recipes?** Please refer to the [v1 branch](../../tree/v1). We recommend using v2 recipes for new projects as they provide improved performance and additional features.

## Model Support

### Pre-Training
List of specific pre-training recipes used by the launch scripts.

| Source | Model | Sequence length | Nodes | Instance | Accelerator | Recipe | Script |
|--------|-------|-----------------|-------|----------|-------------|--------|--------|
| Amazon | Nova Micro | 8192 | 8 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/training/nova/nova_1_0/nova_micro/CPT/nova_micro_1_0_p5x8_gpu_pretrain.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5x8_gpu_pretrain.sh) |
| Amazon | Nova Lite | 8192 | 16 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/training/nova/nova_1_0/nova_lite/CPT/nova_lite_1_0_p5x16_gpu_pretrain.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5x16_gpu_pretrain.sh) |
| Amazon | Nova Pro | 8192 | 24 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/training/nova/nova_1_0/nova_pro/CPT/nova_pro_1_0_p5x24_gpu_pretrain.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5x24_gpu_pretrain.sh) |

### Fine-Tuning
List of specific fine-tuning recipes used by the launch scripts.

| Model | Method | Sequence length | Nodes | Instance | Accelerator | Recipe | Script |
|-------|--------|-----------------|-------|----------|-------------|--------|--------|
| Nova Micro | Supervised Fine-Tuning (LoRA) | 65536 | 2 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/SFT/nova_micro_1_0_p5_p4d_gpu_lora_sft.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_gpu_lora_sft.sh) |
| Nova Micro | Supervised Fine-Tuning (Full) | 65536 | 2 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/SFT/nova_micro_1_0_p5_p4d_gpu_sft.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_gpu_sft.sh) |
| Nova Micro | Direct Preference Optimization (Full) | 32768 | 2 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/DPO/nova_micro_1_0_p5_p4d_gpu_dpo.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_gpu_dpo.sh) |
| Nova Micro | Direct Preference Optimization (LoRA) | 32768 | 2 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/DPO/nova_micro_1_0_p5_p4d_gpu_lora_dpo.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_gpu_lora_dpo.sh) |
| Nova Micro | Rewards Based Reinforcement Learning (PPO) | 8192 | 5 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_micro/PPO/nova_micro_1_0_p5_gpu_ppo.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_gpu_ppo.sh) |
| Nova Lite | Supervised Fine-Tuning (LoRA) | 32768 | 4 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_lora_sft.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_gpu_lora_sft.sh) |
| Nova Lite | Supervised Fine-Tuning (Full) | 65536 | 4 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_gpu_sft.sh) |
| Nova Lite | Direct Preference Optimization (Full) | 32768 | 4 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_lite/DPO/nova_lite_1_0_p5_p4d_gpu_dpo.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_gpu_dpo.sh) |
| Nova Lite | Direct Preference Optimization (LoRA) | 32768 | 4 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_lite/DPO/nova_lite_1_0_p5_p4d_gpu_lora_dpo.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_gpu_lora_dpo.sh) |
| Nova Lite | Rewards Based Reinforcement Learning (PPO) | 8192 | 6 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_lite/PPO/nova_lite_1_0_p5_gpu_ppo.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_gpu_ppo.sh) |
| Nova Pro | Supervised Fine-Tuning (LoRA) | 65536 | 6 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/SFT/nova_pro_1_0_p5_p4d_gpu_lora_sft.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_gpu_lora_sft.sh) |
| Nova Pro | Supervised Fine-Tuning (Full) | 65536 | 6 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/SFT/nova_pro_1_0_p5_gpu_sft.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_gpu_sft.sh) |
| Nova Pro | Direct Preference Optimization (Full) | 32768 | 6 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/DPO/nova_pro_1_0_p5_gpu_dpo.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_gpu_dpo.sh) |
| Nova Pro | Direct Preference Optimization (LoRA) | 32768 | 6 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/DPO/nova_pro_1_0_p5_p4d_gpu_lora_dpo.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_gpu_lora_dpo.sh) |
| Nova Pro | Rewards Based Reinforcement Learning (PPO) | 8192 | 8 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/PPO/nova_pro_1_0_p5_gpu_ppo.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_gpu_ppo.sh) |
| Nova Pro | Model Distillation for Post-Training | - | 1 | ml.r5.24xlarge | - | [link](recipes_collection/recipes/fine-tuning/nova/nova_1_0/nova_pro/distill/nova_pro_1_0_r5_cpu_distill.yaml) | [link](launcher_scripts/nova/run_nova_pro_r5_cpu_distill.sh) |

### Evaluation
List of specific evaluation recipes used by the launch scripts.

| Model | Method | Sequence length | Nodes | Instance | Accelerator | Recipe | Script |
|-------|--------|-----------------|-------|----------|-------------|--------|--------|
| Nova Micro | General Text Benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_micro/nova_micro_1_0_p5_48xl_gpu_general_text_benchmark_eval.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_48xl_general_text_benchmark_eval.sh) |
| Nova Micro | Bring your own dataset (gen_qa) benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_micro/nova_micro_1_0_p5_48xl_gpu_bring_your_own_dataset_eval.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_48xl_bring_your_own_dataset_eval.sh) |
| Nova Micro | Nova LLM as a Judge Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_micro/nova_micro_1_0_p5_48xl_gpu_llm_judge_eval.yaml) | [link](launcher_scripts/nova/run_nova_micro_p5_48xl_llm_judge_eval.sh) |
| Nova Lite | General Text Benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_lite/nova_lite_1_0_p5_48xl_gpu_general_text_benchmark_eval.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_48xl_general_text_benchmark_eval.sh) |
| Nova Lite | Bring your own dataset (gen_qa) benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_lite/nova_lite_1_0_p5_48xl_gpu_bring_your_own_dataset_eval.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_48xl_bring_your_own_dataset_eval.sh) |
| Nova Lite | Nova LLM as a Judge Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_lite/nova_lite_1_0_p5_48xl_gpu_llm_judge_eval.yaml) | [link](launcher_scripts/nova/run_nova_lite_p5_48xl_llm_judge_eval.sh) |
| Nova Pro | General Text Benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_pro/nova_pro_1_0_p5_48xl_gpu_general_text_benchmark_eval.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_48xl_general_text_benchmark_eval.sh) |
| Nova Pro | Bring your own dataset (gen_qa) benchmark Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_pro/nova_pro_1_0_p5_48xl_gpu_bring_your_own_dataset_eval.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_48xl_bring_your_own_dataset_eval.sh) |
| Nova Pro | Nova LLM as a Judge Recipe | 8192 | 1 | ml.p5.48xlarge | GPU H100 | [link](recipes_collection/recipes/evaluation/nova/nova_1_0/nova_pro/nova_pro_1_0_p5_48xl_gpu_llm_judge_eval.yaml) | [link](launcher_scripts/nova/run_nova_pro_p5_48xl_llm_judge_eval.sh) |

## Supported Models

- **Nova Micro** - Smallest model, optimized for efficiency
- **Nova Lite** - Balanced performance and efficiency
- **Nova Pro** - High-performance model for complex tasks
- **Nova Premier** - Most capable model (distillation only)

## Training Techniques

### Supervised Fine-Tuning (SFT)
Fine-tune models on labeled datasets to adapt them to specific tasks or domains.
- **Full Fine-Tuning**: Update all model parameters
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning

### Reasoning Fine-Tuning (RFT)
Enhance model reasoning capabilities (v2.0 models).

### Direct Preference Optimization (DPO)
Align models with human preferences using preference pairs (v1.0 models).

### Proximal Policy Optimization (PPO)
Reinforcement learning-based fine-tuning (v1.0 models).

### Continued Pre-Training (CPT)
Continue pre-training on domain-specific data to adapt models to specialized knowledge areas.

### Distillation
Transfer knowledge from larger models to smaller ones (Pro and Premier).

### Evaluation
Assess model performance using standardized benchmarks and custom datasets.

## Available Recipes

### Nova Micro (v1.0)

**Fine-Tuning:**
- SFT (Full & LoRA)
- DPO (Full & LoRA)
- PPO

**Pre-Training:**
- CPT (Continued Pre-Training)

### Nova Lite (v1.0 & v2.0)

**Fine-Tuning:**
- SFT (Full & LoRA)
- DPO (Full & LoRA) - v1.0 only
- PPO - v1.0 only
- RFT (Full & LoRA) - v2.0 only

**Pre-Training:**
- CPT (Continued Pre-Training)

**Evaluation:**
- General Text Benchmarks
- LLM Judge
- Bring Your Own Dataset
- RFT Evaluation

### Nova Pro (v1.0)

**Fine-Tuning:**
- SFT (Full & LoRA)
- DPO (Full & LoRA)
- PPO

**Pre-Training:**
- CPT (Continued Pre-Training)

**Distillation:**
- CPU-based distillation

**Evaluation:**
- General Text Benchmarks
- LLM Judge
- Bring Your Own Dataset
- RFT Evaluation

### Nova Premier (v1.0)

**Distillation:**
- CPU-based distillation

## Quick Start

### Prerequisites

1. A SageMaker HyperPod cluster or SageMaker Training Jobs setup
2. Appropriate instance types (ml.p5.48xlarge, ml.p4d.24xlarge, or ml.g5/g6 instances)
3. Access to model weights
4. Training data prepared in the required format

**Note:** To run Amazon Nova recipes on SageMaker HyperPod clusters orchestrated by Amazon EKS, you will need to create a Restricted Instance Group in your cluster. Refer to the [SageMaker HyperPod documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-eks-restricted-instance-groups.html) to learn more.

### Running a Recipe

1. **Choose a recipe** based on your model and technique:
   ```bash
   # List available recipes
   ls recipes_collection/recipes/fine-tuning/nova/
   ls recipes_collection/recipes/training/nova/
   ```

2. **Configure the recipe** by editing the YAML file:
   ```bash
   # Example: Edit Nova Lite SFT LoRA recipe
   vim recipes_collection/recipes/fine-tuning/nova/nova_2_0/nova_lite/SFT/nova_lite_2_0_p5_gpu_lora_sft.yaml
   ```

3. **Launch the training job** using the corresponding launch script:
   ```bash
   # Example: Launch Nova Lite SFT LoRA training
   bash launcher_scripts/nova/run_nova_lite_2_0_p5_gpu_lora_sft.sh
   ```

### Example Recipes

**Nova Lite v2.0 SFT with LoRA:**
```bash
bash launcher_scripts/nova/run_nova_lite_2_0_p5_gpu_lora_sft.sh
```

**Nova Lite v2.0 RFT:**
```bash
bash launcher_scripts/nova/run_nova_lite_2_0_p5_lora_rft.sh
```

**Nova Micro v1.0 DPO:**
```bash
bash launcher_scripts/nova/run_nova_micro_p5_gpu_dpo.sh
```

**Nova Pro v1.0 PPO:**
```bash
bash launcher_scripts/nova/run_nova_pro_p5_gpu_ppo.sh
```

**Nova Lite Evaluation:**
```bash
bash launcher_scripts/nova/run_nova_lite_2_0_p5_48xl_general_text_benchmark_eval.sh
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Use LoRA instead of full fine-tuning

### Slow Training
- Increase batch size if memory allows
- Verify EFA (Elastic Fabric Adapter) is properly configured
- Check network bandwidth between nodes
- Ensure data loading is not a bottleneck


## Additional Resources

- [SageMaker HyperPod Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html)
- [Model Documentation](https://docs.aws.amazon.com/nova/)
- [Main Recipe Repository](README.md)

## Testing

Follow the instructions on the "Installing" then use the following command to install the dependencies for testing:

```bash
pip install pytest
pip install pytest-cov
```

### Unit Tests

To run the unit tests, navigate to the root directory and use the command `python -m pytest` plus any desired flags.

The `pyproject.toml` file defines additional options that are always appended to the pytest command:

```toml
[tool.pytest.ini_options]
...
addopts = [
    "--cache-clear",
    "--quiet",
    "--durations=0",
    "--cov=launcher/",
    # uncomment this line to see a detailed HTML test coverage report instead of the usual summary table output to stdout.
    # "--cov-report=html",
    "tests/",
]
```

For the golden tests including the launch JSON ones, the golden outputs can be updated by running `GOLDEN_TEST_WRITE=1 python -m pytest`.

## Contributing

We use pre-commit to unify our coding format, steps to setup are as follows:
- Install pre-commit which helps us run formatters before commit using `pip install pre-commit`
- Setup hooks from our pre-commit hook configs in `.pre-commit-config.yaml` using `pre-commit install`

When you commit, pre-commit hooks will be applied. If for some reason you need to skip the check, you can run `git commit ... --no-verify` but make sure to include the reason to skip pre-commit in the commit message.

For more information, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
