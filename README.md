# Amazon SageMaker HyperPod Recipes

## Overview

Amazon SageMaker HyperPod recipes help customers get started with training and fine-tuning popular publicly available foundation models in just minutes, with state-of-the-art performance. They provide a pre-configured training stack that is tested and validated on Amazon SageMaker.

Please see [Amazon SageMaker HyperPod recipes documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html) for full documentation.

The recipes support the following infrastructure (unless otherwise specified in documentation):
- **Amazon SageMaker HyperPod** with Amazon EKS for workload orchestration
- **Amazon SageMaker HyperPod** with Slurm for workload orchestration
- **Amazon SageMaker training jobs (SMTJ)**
- **Amazon SageMaker serverless model customization** (fully managed, no instance selection)


## Version History

This repository contains **v2.0.0** of Amazon SageMaker HyperPod recipes, which includes recipes built on the latest training frameworks.

**Looking for v1 recipes?** Please refer to the [v1 branch](../../tree/v1). We recommend using v2 recipes for new projects as they provide improved performance and additional features.

## Supported Models and Techniques

### Supported Models

- **Amazon Nova**: Micro, Lite, Pro
- **Gemma**: 4 (E4B, 26B, 31B)
- **Llama**: 3.1, 3.2, 3.3 (1B - 90B), 4 Scout (17B)
- **DeepSeek R1 Distilled**: Llama (8B, 70B), Qwen (1.5B, 7B, 14B, 32B)
- **GPT-OSS**: 20B, 120B
- **Nemotron**: 3 Nano (30B), 3 Super (120B)
- **Qwen**: 2.5 (0.5B - 72B), 3 (0.6B - 32B), 3.5 (4B - 27B), 3.6 (27b)

### Supported Techniques

| Technique | Description | Variants | Model Support |
|-----------|-------------|----------|---------------|
| **Supervised Fine-Tuning (SFT)** | Fine-tune models on supervised datasets | • Full Fine-Tuning (FFT): Complete model parameter updates<br>• LoRA: Low-rank adaptation for parameter efficiency<br>• QLoRA: Quantized LoRA for reduced memory | All models |
| **Direct Preference Optimization (DPO)** | Align models with human preferences without reward modeling | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Learning from AI Feedback (RLAIF)** | Train models using AI-generated feedback | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Learning with Verifiable Rewards (RLVR)** | RL training with verifiable reward signals | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Fine-Tuning (RFT)** | Reinforcement learning fine-tuning | • Full Fine-Tuning (FFT)<br>• LoRA | Nova models only |
| **Proximal Policy Optimization (PPO)** | Policy gradient RL algorithm | Standard | Nova models only |
| **Multi-Turn Reinforcement Learning (MTRL)** | RL training with multi-turn conversational interactions | LoRA | All models |
| **Pretraining** | Continued pre-training on domain-specific data | Full Fine-Tuning (FFT) | All models |

### Supported Accelerators

- NVIDIA H100 (ml.p5.48xlarge, ml.p5e.48xlarge, ml.p5en.48xlarge)
- NVIDIA A100 (ml.p4d.24xlarge, ml.p4de.24xlarge)
- NVIDIA A10G (ml.g5.48xlarge, ml.g5.12xlarge)

### Advanced Training Frameworks

#### LLMFT (LLM Fine-Tuning Framework)
Advanced fine-tuning framework with optimized implementations for:
- DeepSeek R1 Distilled models (Llama and Qwen variants)
- GPT-OSS models (20B, 120B)
- Llama models (3.1, 3.2, 3.3, 4)
- Qwen models (2.5, 3)
- Techniques: SFT (Full Fine-Tuning and LoRA), DPO (Full Fine-Tuning and LoRA)

#### VERL (Versatile Reinforcement Learning)
Reinforcement learning framework using the GRPO algorithm for:
- Gemma models (4) (LoRA only)
- Llama models (3.1, 3.2, 3.3)
- Nemotron models (3 Nano 30B, 3 Super 120B)
- Qwen models (2.5, 3, 3.5, 3.6)
- DeepSeek R1 Distilled models
- GPT-OSS models
- Techniques: RLAIF and RLVR, both available with Full Fine-Tuning or LoRA

#### Checkpointless Training

Memory-efficient training that eliminates traditional checkpoint storage during training, significantly reducing memory overhead and storage requirements. Particularly beneficial for large-scale models where checkpoint sizes can be substantial.

**Supported Models:**
- Llama 3 70B (LoRA, Pretraining)
- GPT-OSS 120B (Full Fine-Tuning, LoRA)

**Key Benefits:**
- Reduced memory footprint during training
- Lower storage costs
- Faster training iteration cycles
- Ideal for large-scale model training

**Available Recipes:**
- [Llama 3 70B LoRA](recipes_collection/recipes/training/llama/checkpointless_llama3_70b_lora.yaml)
- [Llama 3 70B Pretraining](recipes_collection/recipes/training/llama/checkpointless_llama3_70b_pretrain.yaml)
- [GPT-OSS 120B Full Fine-Tuning](recipes_collection/recipes/training/gpt_oss/checkpointless_gpt_oss_120b_full_fine_tuning.yaml)
- [GPT-OSS 120B LoRA](recipes_collection/recipes/training/gpt_oss/checkpointless_gpt_oss_120b_lora.yaml)

#### Elastic Training

Dynamic resource scaling that enables automatic adjustment of training resources based on cluster availability. Workloads can scale up or down to optimize resource utilization and reduce training costs.

**Supported Models:**
- SFT and DPO LLMFT models, such as [LLMFT Llama3.1 8B SFT](recipes_collection/recipes/fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml)

**Key Features:**
- Automatic scaling based on resource availability
- Optimized resource utilization
- Cost-effective training through dynamic capacity adjustment
- Seamless handling of node additions and removals
- Fault tolerance with automatic recovery

**Benefits:**
- Reduced training costs through better resource utilization
- Improved cluster efficiency
- Flexible training that adapts to available resources
- Minimized idle time during training

**How to use:**

With supported SFT/DPO recipes and [elastic training prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-elastic-training.html), just add the following line to your launching script:
```
HYDRA_FULL_ERROR=1 python3 ${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py \
...
recipes.elastic_policy.is_elastic=true \
cluster.use_hyperpod_pytorch_job=true \
cluster.queue_name=<queue_name> \
...
```

#### Multi-Turn Reinforcement Learning (MTRL)

Multi-turn reinforcement learning enables RL training with multi-turn conversational interactions, allowing models to learn from extended dialogue sequences rather than single-turn exchanges. One key feature of MTRL is tool calling.

> **Note:** MTRL recipes are not launched directly from this repository. Please refer to the [MTRL documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html) for full setup and usage instructions.

### Evaluation
- Open-source deterministic evaluation
- LLM-as-Judge evaluation
- Nova-specific evaluation benchmarks

### Logging Support
- [TensorBoard](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html)
- [MLflow](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html)

## Installation

Amazon SageMaker HyperPod recipes should be installed on the head node of your HyperPod cluster or on your local machine with a virtual python environment.

```bash
git clone --recursive git@github.com:aws/sagemaker-hyperpod-recipes.git
cd sagemaker-hyperpod-recipes
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage Guide

When using the SageMaker HyperPod recipes, you can either create your own training script or use the provided recipes which include popular publicly-available models. Based on your specific needs, you might need to modify the parameters defined in the recipes for pre-training or fine-tuning. Once your configurations are setup, you can run training on SageMaker HyperPod (with Amazon EKS for workload orchestration) or on SageMaker training jobs using the Amazon SageMaker Python SDK. Note that Amazon Nova model recipes are only compatible with SageMaker HyperPod with Amazon EKS and SageMaker training jobs.

### Container Images

The following container images are available for different recipe types:

- **For LLMFT recipes**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0`
- **For VERL 0.5.0 recipes (EKS)**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:verl-v1.0.0-eks`
- **For VERL 0.5.0 recipes (SageMaker Training Jobs)**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:verl-v1.0.0-smtj`
- **For VERL 0.7.0 recipes**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes-verl-0-7-0:verl-v1.1.0-smtj`
- **For VERL 0.7.0 Gemma 4 recipes**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes-verl-0-7-0:verl-v1.1.0-smtj-tf58`
- **For VERL 0.7.0 Nemotron recipes**: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes-verl-0-7-0:verl-v1.1.0-smtj-vllm012`

To use a container image for training, modify the `recipes_collection/config.yaml` file with your chosen container image:

```yaml
container: <your_container_image>
```

The launcher scripts have variables such as `TRAIN_DIR` which need to be set either by modifying the launcher script, or by setting environment variables. For example:

```bash
EXP_DIR=<your_exp_dir> TRAIN_DIR=<your_train_data_dir> VAL_DIR=<your_val_data_dir> bash ./launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.sh
```

### Running a recipe on a SageMaker HyperPod cluster orchestrated by Amazon EKS

Prior to commencing training on your cluster, you are required to configure your local environment by adhering to the installation instructions. Additionally, you will need to install Kubectl and Helm on your local machine. Refer to the following documentation for installation of [Kubectl](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html) and [Helm](https://helm.sh/docs/intro/install/).

Using the recipes involves updating `k8s.yaml`, `config.yaml`, and running the launch script.

- In k8s.yaml, update persistent_volume_claims. It mounts the Amazon FSx claim to the /data directory of each computing pod
    ```yaml
    persistent_volume_claims:
      - claimName: fsx-claim
        mountPath: data
    ```

- Update your launcher script (e.g., `launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.sh`)

    - `your_container`: Use the LLMFT container image: `327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0`

    - (Optional) You can provide the HuggingFace token if you need pre-trained weights from HuggingFace by setting the following key-value pair:
    ```bash
    recipes.model.hf_access_token=<your_hf_token>
    ```

```bash
#!/bin/bash
#Users should setup their cluster type in /recipes_collection/config.yaml
IMAGE="327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0"
SAGEMAKER_TRAINING_LAUNCHER_DIR=${SAGEMAKER_TRAINING_LAUNCHER_DIR:-"$(pwd)"}
EXP_DIR="<your_exp_dir>" # Location to save experiment info including logging, checkpoints, etc
TRAIN_DIR="<your_training_data_dir>" # Location of training dataset
VAL_DIR="<your_val_data_dir>" # Location of validation dataset

HYDRA_FULL_ERROR=1 python3 "${SAGEMAKER_TRAINING_LAUNCHER_DIR}/main.py" \
    recipes=training/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora \
    base_results_dir="${SAGEMAKER_TRAINING_LAUNCHER_DIR}/results" \
    recipes.run.name="llmft-deepseek-r1" \
    recipes.exp_manager.exp_dir="$EXP_DIR" \
    cluster=k8s \
    cluster_type=k8s \
    container="${IMAGE}" \
    recipes.model.data.train_dir=$TRAIN_DIR \
    recipes.model.data.val_dir=$VAL_DIR
```

- Launch the training job
    ```bash
    bash launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.sh
    ```

After you've submitted the training job, you can use the following command to verify if you submitted it successfully.
```bash
kubectl get pods
```
```
NAME                                      READY   STATUS    RESTARTS   AGE
llmft-deepseek-r1-<your-alias>-worker-0   0/1     Running   0          36s
```

If the `STATUS` is `PENDING` or `ContainerCreating`, run the following command to get more details.
```bash
kubectl describe pod <name-of-pod>
```

After the job `STATUS` changes to `Running`, you can examine the log by using the following command.
```bash
kubectl logs <name-of-pod>
```

The `STATUS` will turn to `Completed` when you run `kubectl get pods`.

For more information about the k8s cluster configuration, see [Running a training job on HyperPod k8s](https://docs.aws.amazon.com/sagemaker/latest/dg/cluster-specific-configurations-run-training-job-hyperpod-k8s.html).

To run Amazon Nova recipe on SageMaker HyperPod clusters orchestrated by Amazon EKS, you will need to create a Restricted Instance Group in your cluster. Refer to the following documentation to [learn more](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-hp-cluster.html).

### Running a recipe on a SageMaker HyperPod cluster orchestrated by Slurm

> **Note:** Only LLMFT recipes are supported on Slurm clusters. VERL recipes are not supported on Slurm but are available on EKS and SageMaker training jobs.

To run a recipe on a HyperPod cluster with Slurm, SSH into the head node and clone the HyperPod recipes repository onto a shared filesystem (FSx or NFS). Follow the installation instructions to set up a Python virtual environment with the required dependencies.

#### Configuring the Recipe

Update the `recipes_collection/config.yaml` file with the LLMFT container image:

```yaml
container: 327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0
```

#### Running the Training Job

Set the required environment variables and launch the training script. For example, to run an LLMFT recipe:

```bash
EXP_DIR=<your_exp_dir> TRAIN_DIR=<your_train_data_dir> VAL_DIR=<your_val_data_dir> bash ./launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_fine_tuning.sh
```

Or for a DeepSeek R1 Distilled model:

```bash
EXP_DIR=<your_exp_dir> TRAIN_DIR=<your_train_data_dir> VAL_DIR=<your_val_data_dir> bash ./launcher_scripts/deepseek/run_hf_deepseek_r1_llama_8b_seq8k_gpu_lora.sh
```

The launcher scripts will submit Slurm jobs to your cluster. You can monitor job status using standard Slurm commands:

```bash
squeue  # View job queue
scontrol show job <job_id>  # View job details
```

### Running a recipe on SageMaker training jobs

SageMaker training jobs automatically spin up a resilient distributed training cluster, monitors the infrastructure, and auto-recovers from faults to ensure a smooth training experience. You can leverage the SageMaker Python SDK to execute your recipes on SageMaker training jobs.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip setuptools

# install SageMaker SDK
pip install --upgrade sagemaker
```

The following Python code-snippet demonstrates how to submit a recipe to run on a SageMaker training job using the `ModelTrainer` class from the SageMaker Python SDK. `ModelTrainer` is the unified training interface introduced in v3 of the SDK that replaces the framework-specific estimators (such as the `PyTorch` estimator). It requires the v3 SageMaker Python SDK, which is installed by the `pip install --upgrade sagemaker` command above.

For example, to run the llama3.1-8b recipe on a SageMaker training job, pass the recipe name to `ModelTrainer.from_recipe` via the `training_recipe` argument: this can be one of the available recipe names (resolved from this repository), a url, or a local yaml file containing a modified recipe. Override any recipe field (for example, the HuggingFace access token or output paths) by providing `recipe_overrides`, or modify the recipe yaml file directly (the url or local file).

```python
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig, TensorBoardOutputConfig

# Override any recipe fields. Here we point the recipe's output at the standard
# SageMaker training-job container path so artifacts are uploaded to S3.
recipe_overrides = {
    "run": {
        "results_dir": "/opt/ml/model",
    },
}

# Compute for the training job. instance_type is required for recipes and must be
# a GPU or Trainium instance type.
compute = Compute(
    instance_type="ml.p5.48xlarge",
    keep_alive_period_in_seconds=3600,
)

model_trainer = ModelTrainer.from_recipe(
    training_recipe="fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
    recipe_overrides=recipe_overrides,
    compute=compute,
    # See the "Container Images" section above for the image that matches your recipe.
    training_image="327873000638.dkr.ecr.us-east-1.amazonaws.com/hyperpod-recipes:llmft-v1.0.0",
    base_job_name="llama-recipe",
    output_data_config=OutputDataConfig(s3_output_path="<s3 output url>"),
)

# (Optional) stream TensorBoard logs to S3.
model_trainer.with_tensorboard_output_config(
    TensorBoardOutputConfig(
        s3_output_path="<s3 tensorboard url>",
        local_path="/opt/ml/output/tensorboard",
    )
)

model_trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source="s3 or fsx input"),
        InputData(channel_name="val", data_source="s3 or fsx input"),
    ],
    wait=True,
)
```

`ModelTrainer.from_recipe` builds a trainer from the specified recipe and the `instance_type` in `Compute`, and `train()` launches the job. LLMFT, VERL, and Nova recipes require you to pass `training_image` explicitly (use the image that matches your recipe from the [Container Images](#container-images) section above). Each channel passed to `train()` is mounted under `/opt/ml/input/data/<channel_name>` inside the training container, where the recipe reads its inputs. If you do not pass `role` or `sagemaker_session`, `from_recipe` uses the default SageMaker execution role and creates a new session.

To learn more about running Amazon Nova recipe on SageMaker training job, refer to [this documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-training-job.html).

### Running a recipe with SageMaker serverless model customization

The sections above cover *serverful* usage, where you provision and manage the compute (a HyperPod cluster or a SageMaker training job with a chosen `instance_type`). Amazon SageMaker AI also offers **serverless model customization**: a fully managed path where you choose a base model and a customization technique, and SageMaker AI automatically provisions accelerators, applies the pre-optimized recipe, and cleans up compute when the job finishes. There is no instance type or cluster to configure.

For an overview of how serverless compares to SageMaker training jobs and HyperPod, see [Model customization](https://docs.aws.amazon.com/sagemaker/latest/dg/customizing-models.html) and [Serverless model customization](https://docs.aws.amazon.com/sagemaker/latest/dg/customize-model.html) in the AWS documentation.

#### Using the SageMaker Python SDK

Serverless jobs are launched with the model-customization trainers in the SageMaker Python SDK (`SFTTrainer`, `DPOTrainer`, `RLAIFTrainer`, and `RLVRTrainer`), rather than the `ModelTrainer` used for training jobs. Instead of a recipe path, you pass a base-model ID from the SageMaker JumpStart model catalog (for example, `meta-textgeneration-llama-3-1-8b-instruct`).

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip setuptools

# install SageMaker SDK
pip install --upgrade sagemaker
```

The following snippet launches a serverless SFT (LoRA) job. Swap in `DPOTrainer`, `RLAIFTrainer`, or `RLVRTrainer` for other techniques.

```python
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

trainer = SFTTrainer(
    model="meta-textgeneration-llama-3-1-8b-instruct",  # base model ID from the JumpStart catalog
    training_type=TrainingType.LORA,                    # or TrainingType.FULL for full fine-tuning
    model_package_group="my-custom-llama",              # group that tracks the fine-tuned model versions
    training_dataset="s3://<your-bucket>/train/",       # S3 URI or dataset ARN
    validation_dataset="s3://<your-bucket>/val/",       # optional
    s3_output_path="s3://<your-bucket>/output/",        # optional
    accept_eula=True,                                   # required for gated models
)

training_job = trainer.train(wait=True)
print(training_job.training_job_name)
```

Hyperparameters (learning rate, LoRA rank, batch size, sequence length, etc.) are passed to the trainer per technique. See [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) for the full list of supported hyperparameters and recommended ranges for serverless usage. For the complete trainer API, see the [SageMaker Python SDK model customization reference](https://sagemaker.readthedocs.io/en/stable/model_customization/index.html).

#### Using the Studio UI

You can also run serverless model customization visually from the guided interface in Amazon SageMaker Studio, without writing any code. The UI walks you through selecting a base model, technique, datasets, and evaluators, and provides live metrics and logs while the job runs. For step-by-step instructions, see [Serverless model customization](https://docs.aws.amazon.com/sagemaker/latest/dg/customize-model.html) in the AWS documentation.

## Troubleshooting

During training, if GPU memory usage approaches its limit, attempting to save sharded checkpoints to an S3 storage may result in a core dump. To address this issue, you may choose to:

* Reduce the overall memory consumption of the model training:
  * Increase the number of compute nodes for the training process
  * Decrease the batch size
  * Increase the sharding degrees
* Use FSx as the shared file system

By taking one of the above approaches, you can alleviate the memory pressure and prevent a core dump from occurring during checkpoint saving.

## Testing

Follow the instructions on the "Installing" section then use the following command to install the dependencies for testing:

```bash
pip install pytest
pip install pytest-cov
```

### Unit Tests
To run the unit tests, navigate to the root directory and use the command `python -m pytest` plus any desired flags.

The `pyproject.toml` file defines additional options that are always appended to the `pytest` command:
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

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
