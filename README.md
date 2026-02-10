# Amazon SageMaker HyperPod Recipes

## Overview

Amazon SageMaker HyperPod recipes help customers get started with training and fine-tuning popular publicly available foundation models in just minutes, with state-of-the-art performance. They provide a pre-configured training stack that is tested and validated on Amazon SageMaker.

Please see [Amazon SageMaker HyperPod recipes documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html) for full documentation.

The recipes support the following infrastructure (unless otherwise specified in documentation):
- **Amazon SageMaker HyperPod** with Amazon EKS for workload orchestration
- **Amazon SageMaker HyperPod** with Slurm for workload orchestration
- **Amazon SageMaker training jobs (SMTJ)**


## Version History

This repository contains **v2.0.0** of Amazon SageMaker HyperPod recipes, which includes recipes built on the latest training frameworks.

**Looking for v1 recipes?** Please refer to the [v1 branch](../../tree/v1). We recommend using v2 recipes for new projects as they provide improved performance and additional features.

## Supported Models and Techniques

### Supported Models

- **Amazon Nova**: Micro, Lite, Pro
- **Llama**: 3.1, 3.2, 3.3 (1B - 90B), 4 Scout (17B)
- **DeepSeek R1 Distilled**: Llama (8B, 70B), Qwen (1.5B, 7B, 14B, 32B)
- **GPT-OSS**: 20B, 120B
- **Qwen**: 2.5 (0.5B - 72B), 3 (0.6B - 32B)

### Supported Techniques

| Technique | Description | Variants | Model Support |
|-----------|-------------|----------|---------------|
| **Supervised Fine-Tuning (SFT)** | Fine-tune models on supervised datasets | • Full Fine-Tuning (FFT): Complete model parameter updates<br>• LoRA: Low-rank adaptation for parameter efficiency<br>• QLoRA: Quantized LoRA for reduced memory | All models |
| **Direct Preference Optimization (DPO)** | Align models with human preferences without reward modeling | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Learning from AI Feedback (RLAIF)** | Train models using AI-generated feedback | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Learning with Verifiable Rewards (RLVR)** | RL training with verifiable reward signals | • Full Fine-Tuning (FFT)<br>• LoRA | All models |
| **Reinforcement Fine-Tuning (RFT)** | Reinforcement learning fine-tuning | • Full Fine-Tuning (FFT)<br>• LoRA | Nova models only |
| **Proximal Policy Optimization (PPO)** | Policy gradient RL algorithm | Standard | Nova models only |
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
- Llama models (3.1, 3.2, 3.3)
- Qwen models (2.5, 3)
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

- **For LLMFT recipes**: `920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.0.0`, full list [here](https://github.com/aws/sagemaker-hyperpod-recipes/blob/main/launcher/recipe_templatization/llmft/llmft_regional_parameters.json)
- **For VERL recipes (EKS)**: `920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.0.0-eks`, full list [here](https://github.com/aws/sagemaker-hyperpod-recipes/blob/main/launcher/recipe_templatization/verl/verl_regional_parameters.json)
- **For VERL recipes (SageMaker Training Jobs)**: `920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:verl-v1.0.0-smtj`

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

    - `your_container`: Use the LLMFT container image: `920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.0.0`

    - (Optional) You can provide the HuggingFace token if you need pre-trained weights from HuggingFace by setting the following key-value pair:
    ```bash
    recipes.model.hf_access_token=<your_hf_token>
    ```

```bash
#!/bin/bash
#Users should setup their cluster type in /recipes_collection/config.yaml
IMAGE="920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.0.0"
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
container: 920498770698.dkr.ecr.us-west-2.amazonaws.com/hyperpod-recipes:llmft-v1.0.0
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

The following Python code-snippet demonstrates how to submit a recipe to run on a SageMaker training jobs by utilizing the `PyTorch` estimator from the SageMaker Python SDK.

For example, to run the llama3-8b recipe on a SageMaker training jobs, you need to set `training_recipe` arg to indicate which recipe: this can be a recipe from one of the available ones, or a url or a local yaml file containing a modified recipe. Please also modify the local directory paths and hf access token either by providing `recipe_overrides` or by modifying the recipe yaml file directly (the url or local file).

```python
import os
import sagemaker, boto3
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

bucket = sagemaker_session.default_bucket()
output = os.path.join(f"s3://{bucket}", "output")
output_path = "<s3 url>"

recipe_overrides = {
    "run": {
        "results_dir": "/opt/ml/model",
    },
    "exp_manager": {
        "exp_dir": "",
        "explicit_log_dir": "/opt/ml/output/tensorboard",
        "checkpoint_dir": "/opt/ml/checkpoints",
    },
    "model": {
        "data": {
            "train_dir": "/opt/ml/input/data/train",
            "val_dir": "/opt/ml/input/data/val",
        },
    },
}

tensorboard_output_config = TensorBoardOutputConfig(
    s3_output_path=os.path.join(output, 'tensorboard'),
    container_local_output_path=recipe_overrides["exp_manager"]["explicit_log_dir"]
)

estimator = PyTorch(
    output_path=output_path,
    base_job_name=f"llama-recipe",
    role=role,
    instance_type="ml.p5.48xlarge",
    training_recipe="training/llama/llmft_llama3_8b_seq4k_gpu_sft_lora",
    recipe_overrides=recipe_overrides,
    sagemaker_session=sagemaker_session,
    tensorboard_output_config=tensorboard_output_config,
)

estimator.fit(inputs={"train": "s3 or fsx input", "val": "s3 or fsx input"}, wait=True)
```

Running the above code creates a `PyTorch` estimator object with the specified training recipe and then trains the model using the `fit()` method. The new `training_recipe` parameter enables you to specify the recipe you want to use.

To learn more about running Amazon Nova recipe on SageMaker training job, refer to [this documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model-training-job.html).

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

