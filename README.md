# Amazon SageMaker HyperPod recipes

## Overview

Amazon SageMaker HyperPod recipes help customers get started with training and fine-tuning popular publicly available foundation models in just minutes, with state-of-the-art performance. The recipes provide a pre-configured training stack that is tested and validated on Amazon SageMaker.

Please see [Amazon SageMaker HyperPod recipes](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html) for documentation.

The recipes support Amazon SageMaker HyperPod (with Slurm or Amazon EKS for workload orchestration) and Amazon SageMaker training jobs.

Amazon SageMaker HyperPod recipes include built-in support for:

- Model parallelism - tensor parallelism and context parallel
- Automated distributed checkpointing
- Distributed optimizer
- Accelerators: NVIDIA H100 (ml.p5), NVIDIA A100 (ml.p4), and AWS Trainium (ml.trn1)
- Fine-tuning: Full, QLoRA, LoRA, DPO
- AWS Instances: ml.p5.48xlarge, ml.p4d.24xlarge, and ml.trn1.32xlarge instance families
- Supported Models: DeepSeek R1, DeepSeek R1 Distill Llama, DeepSeek R1 Distill Qwen, Llama, Mistral, Mixtral models
- Model Evaluation: [Tensorboard](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.tensorboard.html#module-lightning.pytorch.loggers.tensorboard), [MLflow](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html), [Wandb](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) - feel free to add any key word arguments to the Logger classes by using their associated kwargs config

###### ***Note: For DeepSeek R1 671b customers must ensure that their model repository contains weights of type bf16. DeepSeek's [HuggingFace repository](https://huggingface.co/deepseek-ai/DeepSeek-R1) contains the model in dtype fp8 by default. In order to convert a model repository from fp8 to bf16 we recommend using [this script](https://github.com/aws/sagemaker-hyperpod-training-adapter-for-nemo/blob/main/src/hyperpod_nemo_adapter/scripts/fp8_cast_bf16.py) and pointing your recipe to the output directory.

## Model Support

### Pre-Training

List of specific pre-training recipes used by the launch scripts.

| Source       | Model     | Size | Sequence length | Nodes | Instance      | Accelerator | Recipe | Script |
| ------------ | --------- | ---- | --------------- | ----- | ------------- | ----------- | ------ | ------ |
| Hugging Face | Llama 3.2 | 11b  | 8192            | 4     | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_2_11b_seq8k_gpu_p5x4_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_2_11b_seq8k_gpu_p5x4_pretrain.sh) |
| Hugging Face | Llama 3.2 | 90b  | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_2_90b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_2_90b_seq8k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Llama 3.2 | 1b   | 8192            | 1     | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_2_1b_seq8k_gpu_p5x1_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_2_1b_seq8k_gpu_p5x1_pretrain.sh) |
| Hugging Face | Llama 3.2 | 3b   | 8192            | 1     | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_2_3b_seq8k_gpu_p5x1_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_2_3b_seq8k_gpu_p5x1_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 16384           | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq16k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq16k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 16384           | 64    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq16k_gpu_p5x64_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq16k_gpu_p5x64_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 16384           | 128    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq16k_gpu_p5x128_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq16k_gpu_p5x128_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 8192            | 64    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq8k_gpu_p5x64_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_gpu_p5x64_pretrain.sh) |
| Hugging Face | Llama 3.1 | 70b  | 8192            | 128    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq8k_gpu_p5x128_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_gpu_p5x128_pretrain.sh) |
| Hugging Face | Llama 3   | 70b  | 8192            | 16    | ml.trn1.32xlarge | TRN         | [link](recipes_collection/recipes/training/llama/hf_llama3_70b_seq8k_trn1x16_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_trn1x16_pretrain.sh) |
| Hugging Face | Llama 3.1 | 8b   | 16384           | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_8b_seq16k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq16k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Llama 3.1 | 8b   | 16384           | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_8b_seq16k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq16k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Llama 3.1 | 8b   | 8192            | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_8b_seq8k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Llama 3.1 | 8b   | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/hf_llama3_8b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Llama 3   | 8b   | 8192            | 4     | ml.trn1.32xlarge | TRN         | [link](recipes_collection/recipes/training/llama/hf_llama3_8b_seq8k_trn1x4_pretrain.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_trn1x4_pretrain.sh) |
| Megatron     | Llama 3.1 | 8b   | 8192            | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/llama/megatron_llama3_1_8b_nemo.yaml) | - |
| Hugging Face | Mistral   | 7b   | 16384           | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mistral/hf_mistral_7b_seq16k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/mistral/run_hf_mistral_7b_seq16k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Mistral   | 7b   | 16384           | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mistral/hf_mistral_7b_seq16k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mistral/run_hf_mistral_7b_seq16k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Mistral   | 7b   | 8192            | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mistral/hf_mistral_7b_seq8k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/mistral/run_hf_mistral_7b_seq8k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Mistral   | 7b   | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mistral/hf_mistral_7b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mistral/run_hf_mistral_7b_seq8k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 16384           | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq16k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq16k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 16384           | 64    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq16k_gpu_p5x64_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq16k_gpu_p5x64_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 16384           | 128    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq16k_gpu_p5x128_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq16k_gpu_p5x128_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq8k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 8192            | 64    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq8k_gpu_p5x64_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq8k_gpu_p5x64_pretrain.sh) |
| Hugging Face | Mixtral   | 22b  | 8192            | 128    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x22b_seq8k_gpu_p5x128_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x22b_seq8k_gpu_p5x128_pretrain.sh) |
| Hugging Face | Mixtral   | 7b   | 16384           | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x7b_seq16k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x7b_seq16k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Mixtral   | 7b   | 16384           | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x7b_seq16k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x7b_seq16k_gpu_p5x32_pretrain.sh) |
| Hugging Face | Mixtral   | 7b   | 8192            | 16    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x7b_seq8k_gpu_p5x16_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x7b_seq8k_gpu_p5x16_pretrain.sh) |
| Hugging Face | Mixtral   | 7b   | 8192            | 32    | ml.p5.48xlarge   | GPU H100    | [link](recipes_collection/recipes/training/mixtral/hf_mixtral_8x7b_seq8k_gpu_p5x32_pretrain.yaml) | [link](launcher_scripts/mixtral/run_hf_mixtral_8x7b_seq8k_gpu_p5x32_pretrain.sh) |


### Fine-Tuning

List of specific fine-tuning recipes used by the launch scripts.
All model sources are from Hugging Face.

| Model     | Method | Size | Sequence length | Nodes | Instance       | Accelerator | Recipe | Script |
| --------- | ------ | ---- | ----------------| ----- | -------------- | ----------- | ------ | ------ |
| LLama 4 Scout | LoRA (multi-modal)  | 17B 16E (109B)   | 8192          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama4_17b_16e_seq8k_gpu_lora_multimodal_finetuning.yaml) | [link](launcher_scripts/llama/run_hf_llama4_17b_16e_seq8k_gpu_lora_multimodal_finetuning.sh) |
| LLama 4 Scout | LoRA (multi-modal)  | 17B 16E (109B)   | 4096          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama4_17b_16e_seq4k_gpu_lora_multimodal_finetuning.yaml) | [link](launcher_scripts/llama/run_hf_llama4_17b_16e_seq4k_gpu_lora_multimodal_finetuning.sh) |
| LLama 4 Scout | LoRA (text-only)   | 17B 16E (109B)   | 4096          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama4_17b_16e_seq4k_gpu_lora_text_to_text.yaml) | [link](launcher_scripts/llama/run_hf_llama4_17b_16e_seq4k_gpu_lora_text_to_text.sh) |
| DeepSeek R1 | QLoRA  | 671b   | 8192          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_671b_seq8k_gpu_qlora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_671b_seq8k_gpu_qlora.sh) |
| DeepSeek R1 | LoRA   | 671b   | 8192          | 5     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_671b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_671b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Llama 3 | SFT  | 8b   | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_8b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_8b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Llama 3 | LoRA | 8b   | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_8b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_8b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Llama 3 | SFT  | 8b   | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_8b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_8b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Llama 3 | LoRA | 8b   | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_8b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_8b_seq16k_gpu_lora.sh) |
| DeepSeek R1 Distill Llama 3 | SFT  | 70b  | 8192          | 10    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_70b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_70b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Llama 3 | LoRA | 70b  | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_70b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_70b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Llama 3 | SFT  | 70b  | 16384         | 16    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_70b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_70b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Llama 3 | LoRA | 70b  | 16384         | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_70b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_llama_70b_seq16k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 1.5b   | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_1_dot_5b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_1_dot_5b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 1.5b   | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_1_dot_5b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_1_dot_5b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 1.5b   | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_1_dot_5b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_1_dot_5b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 1.5b   | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_1_dot_5b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_1_dot_5b_seq16k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 7b   	 | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_7b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_7b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 7b     | 8192          | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_7b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_7b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 7b     | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_7b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_7b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 7b     | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_7b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_7b_seq16k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 14b    | 8192          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_14b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_14b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 14b    | 8192          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_14b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_14b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 14b    | 16384         | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_14b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_14b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 14b    | 16384         | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_14b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_14b_seq16k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 32b    | 8192          | 4     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_32b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_32b_seq8k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 32b    | 8192          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_32b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_32b_seq8k_gpu_lora.sh) |
| DeepSeek R1 Distill Qwen 2 | SFT  | 32b    | 16384         | 6     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_32b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_32b_seq16k_gpu_fine_tuning.sh) |
| DeepSeek R1 Distill Qwen 2 | LoRA | 32b    | 16384         | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/deepseek/hf_deepseek_r1_distilled_qwen_32b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/deepseek/run_hf_deepseek_r1_qwen_32b_seq16k_gpu_lora.sh) |
| Llama 3.1 | QLoRA  | 405b | 131072          | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq128k_gpu_qlora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq128k_gpu_qlora.sh) |
| Llama 3.1 | QLoRA  | 405b | 32768           | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq32k_gpu_qlora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq32k_gpu_qlora.sh) |
| Llama 3.1 | LoRA   | 405b | 16384           | 6     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq16k_gpu_lora.sh) |
| Llama 3.1 | QLoRA  | 405b | 16384           | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq16k_gpu_qlora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq16k_gpu_qlora.sh) |
| Llama 3.1 | LoRA   | 405b | 8192            | 6     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq8k_gpu_lora.sh) |
| Llama 3.1 | QLoRA  | 405b | 8192            | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_405b_seq8k_gpu_qlora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_405b_seq8k_gpu_qlora.sh) |
| Llama 3.3 | SFT  	 | 70b  | 16384           | 16    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_3_70b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_3_70b_seq16k_gpu_fine_tuning.sh) |
| Llama 3.3 | LoRA   | 70b  | 16384           | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_3_70b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_3_70b_seq16k_gpu_lora.sh) |
| Llama 3.3 | SFT    | 70b  | 8192            | 10    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_3_70b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_3_70b_seq8k_gpu_fine_tuning.sh) |
| Llama 3.3 | LoRA   | 70b  | 8192            | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_3_70b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_3_70b_seq8k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 70b  | 16384           | 16    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_70b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq16k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 70b  | 16384           | 2     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_70b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq16k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 70b  | 8192            | 10    | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_70b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 70b  | 8192            | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_70b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_70b_seq8k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 8b   | 16384           | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_8b_seq16k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq16k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 8b   | 16384           | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_8b_seq16k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq16k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 8b   | 8192            | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_8b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 8b   | 8192            | 1     | ml.p5.48xlarge    | GPU H100    | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_8b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 70b  | 8192            | 32    | ml.p4d.24xlarge   | GPU A100    | [link](recipes_collection/recipes/fine-tuning/llama/p4_hf_llama3_70b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/p4_run_hf_llama3_70b_seq8k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 70b  | 8192            | 20    | ml.p4d.24xlarge   | GPU A100    | [link](recipes_collection/recipes/fine-tuning/llama/p4_hf_llama3_70b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/p4_run_hf_llama3_70b_seq8k_gpu_lora.sh) |
| Llama 3.1 | SFT    | 8b   | 8192            | 4     | ml.p4d.24xlarge   | GPU A100    | [link](recipes_collection/recipes/fine-tuning/llama/p4_hf_llama3_8b_seq8k_gpu_fine_tuning.yaml) | [link](launcher_scripts/llama/p4_run_hf_llama3_8b_seq8k_gpu_fine_tuning.sh) |
| Llama 3.1 | LoRA   | 8b   | 8192            | 1     | ml.p4d.24xlarge   | GPU A100    | [link](recipes_collection/recipes/fine-tuning/llama/p4_hf_llama3_8b_seq8k_gpu_lora.yaml) | [link](launcher_scripts/llama/p4_run_hf_llama3_8b_seq8k_gpu_lora.sh) |
| Llama 3   | SFT    | 8b   | 8192            | 1     | ml.trn1.32xlarge  | TRN         | [link](recipes_collection/recipes/fine-tuning/llama/hf_llama3_8b_seq8k_trn1_fine_tuning.yaml) | [link](launcher_scripts/llama/run_hf_llama3_8b_seq8k_trn1_fine_tuning.sh) |


## Installation

Amazon SageMaker HyperPod recipes should be installed on the head node of your HyperPod cluster or on your local machine with a virtual python environment.

```
git clone --recursive git@github.com:aws/sagemaker-hyperpod-recipes.git
cd sagemaker-hyperpod-recipes
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage Guide

When using the SageMaker HyperPod recipes, you can either
create your own training script or leverage the SageMaker HyperPod adapter,
which includes popular publicly-available models like Llama or Mistral. Based on your specific
needs, you might need to modify the parameters defined in the recipes for
pre-training or fine-tuning. Once your configurations are setup, you can run training on SageMaker
HyperPod (with Slurm or Amazon EKS) for workload orchestration. Alternatively, you can run the recipe on
SageMaker training jobs using the Amazon SageMaker Python SDK.

### Running a recipe via a Slurm job on a SageMaker HyperPod cluster

To run a recipe via a Slurm job on a HyperPod cluster, you need to SSH into the head node
of the cluster and clone the HyperPod recipes repository onto a shared filesystem, such as
FSX or NFS. Next, follow the installation instructions to set up a Python
virtual environment with the required dependencies. Once the environment is
ready, you can launch a training job from the launcher\_scripts folder. For
example, you can modify the recipe launcher script [run_hf_llama3_8b_seq8k_gpu_p5x16_pretrain](launcher_scripts/llama/run_hf_llama3_8b_seq8k_gpu_p5x16_pretrain.sh)
with customized configurations such as your image or output directory. Once
setting all the necessary parameters in the recipe launcher, you can
start the training process by running the script.

We recommend that you utilize `enroot` to initiate a training process on
the Slurm cluster. You can get the latest docker image from [SMP release notes](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-model-parallel-support-v2.html). You can refer to the following example to generate a squash file
employing the `enroot` command. Please refer to the following documentation on building an [AWS-optimized Nemo-Launcher image](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/2.nemo-launcher#2-build-aws-optimized-nemo-launcher-image).

```bash
REGION="us-west-2"
IMAGE="658645717510.dkr.ecr.${REGION}.amazonaws.com/smdistributed-modelparallel:${TAG}"
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin 658645717510.dkr.ecr.${REGION}.amazonaws.com
enroot import -o $PWD/smdistributed-modelparallel.sqsh dockerd://${IMAGE}
mv $PWD/smdistributed-modelparallel.sqsh "/fsx/smdistributed-modelparallel.sqsh"
```

To use a prebuilt enroot:
```
wget https://sagemaker-distributed-model-parallel.s3.us-west-2.amazonaws.com/enroot/2.4.1-gpu-py311-cu121-ubuntu20.04-sagemaker-smpv2.7.0.sqsh
```

To use the Enroot squash file to start training, use the following example to
modify the `recipes_collection/config.yaml` file.

```
container: /fsx/smdistributed-modelparallel.sqsh
```

The launcher script has variables such as `TRAIN_DIR` which need to be set either by modifying the launcher script, or by setting environment variables. For example:

```bash
EXP_DIR=<your_exp_dir> TRAIN_DIR=<your_train_data_dir> VAL_DIR=<your_val_data_dir> bash ./launcher_scripts/llama/run_hf_llama3_8b_seq16k_gpu_p5x16_pretrain.sh
```

### Running a recipe on a SageMaker HyperPod clusters orchestrated by Amazon EKS

Prior to commencing training on your cluster, you are required to
configure your local environment by adhering to the installation instructions.
Additionally, you will need to install Kubectl and Helm on your local machine.
Refer to the following documentation for installation of [Kubectl](https://docs.aws.amazon.com/eks/latest/userguide/install-kubectl.html)
and [Helm](https://helm.sh/docs/intro/install/).

You can now proceed with submitting a training job by utilizing the same launcher script with the
following command:

```
aws eks update-kubeconfig --region "${CLUSTER_REGION}" --name "${CLUSTER_NAME}"
launcher_scripts/llama/run_hf_llama3_8b_seq8192.sh
```

We recommend that you utilize [HyperPod command-line tool](https://github.com/aws/sagemaker-hyperpod-cli/tree/release_v2)
to launch a training job.

```
hyperpod start-job --recipe training/llama/hf_llama3_8b_seq16k_gpu_p5x16_pretrain \
--persistent-volume-claims fsx-claim:data \
--override-parameters \
'{
 "recipes.run.name": "hf-llama3-8b",
 "recipes.exp_manager.exp_dir": "/data/<your_exp_dir>",
 "container": "658645717510.dkr.ecr.<region>.amazonaws.com/smdistributed-modelparallel:2.4.1-gpu-py311-cu121",
 "recipes.model.data.train_dir": "<your_train_data_dir>",
 "recipes.model.data.val_dir": "<your_val_data_dir>",
 "cluster": "k8s",
 "cluster_type": "k8s"
}'
```

### Running a recipe on SageMaker training jobs

SageMaker training jobs automatically spin up a resilient distributed training cluster,
monitors the infrastructure, and auto-recovers from faults to ensure a smooth training experience.
You can leverage the SageMaker Python SDK to execute your recipes on SageMaker training jobs.

```
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip setuptools

# install SageMaker SDK
pip install --upgrade sagemaker
```

The following Python code-snippet demonstrates how to submit a recipe to
run on a SageMaker training jobs by utilizing the `PyTorch`
estimator from the SageMaker Python SDK.

For example, to run the llama3-8b recipe on
a SageMaker training jobs, you need to set `training_recipe` arg to indicate which recipe: this
can be a recipe from one of the available ones, or a url or a local yaml file containing a modified
recipe. Please also modify the local directory paths and hf access token either by providing
`recipe_overrides` or by modifying the recipe yaml file directly (the url or local file).

```python
import os
import sagemaker,boto3
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
  training_recipe="training/llama/hf_llama3_8b_seq8k_gpu_p5x16_pretrain",
  recipe_overrides=recipe_overrides,
  sagemaker_session=sagemaker_session,
  tensorboard_output_config=tensorboard_output_config,
)

estimator.fit(inputs={"train": "s3 or fsx input", "val": "s3 or fsx input"}, wait=True)
```

Running the above code creates a `PyTorch` estimator object with the specified training recipe
and then trains the model using the `fit()` method. The new `training_recipe` parameter enables you
to specify the recipe you want to use.


## Troubleshooting

During training, if GPU memory usage approaches its limit, attempting to save sharded checkpoints to an S3 storage may result in a core dump.
To address this issue, you may choose to:

* Reduce the overall memory consumption of the model training:
  * Increase the number of compute nodes for the traninig process.
  * Decrease the batch size
  * Increase the sharding degrees, etc.
* Use FSx as the shared file system

By taking one of the above approaches, you can alleviate the memory pressure and prevent a core dump from occurring during checkpoint saving.

Llama 3.2 specifically requires transformers version 4.45.2 or above. They should be installed automatically in the container during job launch if using slurm or k8s.  If not, you can update your requirements.txt or container so that transformers==4.45.2 is installed.

## Testing

Follow the instructions on the "Installing" then use the following command to install the dependencies for testing:

```
pip install pytest
pip install pytest-cov
```

### Unit Tests
To run the unit tests, navigate to the root directory and use the command
```python -m pytest``` plus any desired flags.

The `pyproject.toml` file defines additional options that are always appended to the `pytest` command:
```
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

## Contributing
We use pre-commit to unify our coding format, steps to setup as as follows:
- Install pre-commit which helps us run formatters before commit using `pip install pre-commit`
- Setup hooks from our pre-commit hook configs in `.pre-commit-config.yaml` using `pre-commit install`
When you commit, pre-commit hooks will be applied. If for some reason you need to skip the check, you can run `git commit ... --no-verify` but make sure to include the reason to skip pre-commit in the commit message.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
