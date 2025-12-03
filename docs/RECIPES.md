# Amazon SageMaker HyperPod Recipes - Complete Recipe Catalog

This document provides a comprehensive catalog of all available recipes organized by category.

## LLMFT (LLM Fine-Tuning Framework) Recipes

### DeepSeek R1 Distilled Models

| Model | Technique | Sequence Length | Recipe and Launch Script |
|-------|-----------|-----------------|--------------------------|
| Llama 8B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_sft_lora.sh)) |
| Llama 8B | DPO | 4096 | [llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_8b_seq4k_gpu_dpo.sh)) |
| Llama 70B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_sft_lora.sh)) |
| Llama 70B | DPO | 4096 | [llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_llama_70b_seq4k_gpu_dpo.sh)) |
| Qwen 1.5B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_sft_lora.sh)) |
| Qwen 1.5B | DPO | 4096 | [llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_1_dot_5b_seq4k_gpu_dpo.sh)) |
| Qwen 7B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_sft_lora.sh)) |
| Qwen 7B | DPO | 4096 | [llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_dpo.sh)) |
| Qwen 14B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_sft_lora.sh)) |
| Qwen 14B | DPO | 4096 | [llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_14b_seq4k_gpu_dpo.sh)) |
| Qwen 32B | SFT LoRA | 4096 | [llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_sft_lora](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_sft_lora.sh)) |
| Qwen 32B | DPO | 4096 | [llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_dpo](fine-tuning/deepseek/llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/deepseek/run_llmft_deepseek_r1_distilled_qwen_32b_seq4k_gpu_dpo.sh)) |

### GPT-OSS Models

| Model | Technique | Sequence Length | Recipe and Launch Script |
|-------|-----------|-----------------|--------------------------|
| GPT-OSS 20B | SFT LoRA | 4096 | [llmft_gpt_oss_20b_seq4k_gpu_sft_lora](fine-tuning/gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_llmft_gpt_oss_20b_seq4k_gpu_sft_lora.sh)) |
| GPT-OSS 20B | DPO | 4096 | [llmft_gpt_oss_20b_seq4k_gpu_dpo](fine-tuning/gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_llmft_gpt_oss_20b_seq4k_gpu_dpo.sh)) |
| GPT-OSS 120B | SFT LoRA | 4096 | [llmft_gpt_oss_120b_seq4k_gpu_sft_lora](fine-tuning/gpt_oss/llmft_gpt_oss_120b_seq4k_gpu_sft_lora.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_llmft_gpt_oss_120b_seq4k_gpu_sft_lora.sh)) |
| GPT-OSS 120B | DPO | 4096 | [llmft_gpt_oss_120b_seq4k_gpu_dpo](fine-tuning/gpt_oss/llmft_gpt_oss_120b_seq4k_gpu_dpo.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_llmft_gpt_oss_120b_seq4k_gpu_dpo.sh)) |

## Checkpointless Training Recipes

| Model | Technique | Recipe and Launch Script |
|-------|-----------|--------------------------|
| Llama 3 70B | LoRA | [checkpointless_llama3_70b_lora](fine-tuning/llama/checkpointless_llama3_70b_lora.yaml) ([launch script](../../launcher_scripts/llama/run_checkpointless_llama3_70b_lora.sh)) |
| Llama 3 70B | Pretraining | [checkpointless_llama3_70b_pretrain](fine-tuning/llama/checkpointless_llama3_70b_pretrain.yaml) ([launch script](../../launcher_scripts/llama/run_checkpointless_llama3_70b_pretrain.sh)) |
| GPT-OSS 120B | Full Fine-Tuning | [checkpointless_gpt_oss_120b_full_fine_tuning](fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_full_fine_tuning.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_full_fine_tuning.sh)) |
| GPT-OSS 120B | LoRA | [checkpointless_gpt_oss_120b_lora](fine-tuning/gpt_oss/checkpointless_gpt_oss_120b_lora.yaml) ([launch script](../../launcher_scripts/gpt_oss/run_checkpointless_gpt_oss_120b_lora.sh)) |

## Amazon Nova Recipes

### Nova Micro

| Technique | Sequence Length | Nodes | Instance Type | Recipe | Launch Script |
|-----------|-----------------|-------|---------------|--------|---------------|
| SFT (LoRA) | 65536 | 2 | ml.p5.48xlarge | [nova_micro_1_0_p5_p4d_gpu_lora_sft.yaml](fine-tuning/nova/nova_1_0/nova_micro/SFT/nova_micro_1_0_p5_p4d_gpu_lora_sft.yaml) | [run_nova_micro_p5_gpu_lora_sft.sh](../../launcher_scripts/nova/run_nova_micro_p5_gpu_lora_sft.sh) |
| SFT (Full) | 65536 | 2 | ml.p5.48xlarge | [nova_micro_1_0_p5_p4d_gpu_sft.yaml](fine-tuning/nova/nova_1_0/nova_micro/SFT/nova_micro_1_0_p5_p4d_gpu_sft.yaml) | [run_nova_micro_p5_gpu_sft.sh](../../launcher_scripts/nova/run_nova_micro_p5_gpu_sft.sh) |
| RFT (DPO) | 65536 | 2 | ml.p5.48xlarge | [nova_micro_1_0_p5_p4d_gpu_dpo_rft.yaml](fine-tuning/nova/nova_1_0/nova_micro/RFT/nova_micro_1_0_p5_p4d_gpu_dpo_rft.yaml) | [run_nova_micro_p5_gpu_dpo_rft.sh](../../launcher_scripts/nova/run_nova_micro_p5_gpu_dpo_rft.sh) |
| RFT (PPO) | 65536 | 2 | ml.p5.48xlarge | [nova_micro_1_0_p5_p4d_gpu_ppo_rft.yaml](fine-tuning/nova/nova_1_0/nova_micro/RFT/nova_micro_1_0_p5_p4d_gpu_ppo_rft.yaml) | [run_nova_micro_p5_gpu_ppo_rft.sh](../../launcher_scripts/nova/run_nova_micro_p5_gpu_ppo_rft.sh) |

### Nova Lite

| Technique | Sequence Length | Nodes | Instance Type | Recipe | Launch Script |
|-----------|-----------------|-------|---------------|--------|---------------|
| SFT (LoRA) | 32768 | 4 | ml.p5.48xlarge | [nova_lite_1_0_p5_p4d_gpu_lora_sft.yaml](fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_lora_sft.yaml) | [run_nova_lite_p5_gpu_lora_sft.sh](../../launcher_scripts/nova/run_nova_lite_p5_gpu_lora_sft.sh) |
| SFT (Full) | 65536 | 4 | ml.p5.48xlarge | [nova_lite_1_0_p5_p4d_gpu_sft.yaml](fine-tuning/nova/nova_1_0/nova_lite/SFT/nova_lite_1_0_p5_p4d_gpu_sft.yaml) | [run_nova_lite_p5_gpu_sft.sh](../../launcher_scripts/nova/run_nova_lite_p5_gpu_sft.sh) |
| RFT (DPO) | 65536 | 4 | ml.p5.48xlarge | [nova_lite_1_0_p5_p4d_gpu_dpo_rft.yaml](fine-tuning/nova/nova_1_0/nova_lite/RFT/nova_lite_1_0_p5_p4d_gpu_dpo_rft.yaml) | [run_nova_lite_p5_gpu_dpo_rft.sh](../../launcher_scripts/nova/run_nova_lite_p5_gpu_dpo_rft.sh) |
| RFT (PPO) | 65536 | 4 | ml.p5.48xlarge | [nova_lite_1_0_p5_p4d_gpu_ppo_rft.yaml](fine-tuning/nova/nova_1_0/nova_lite/RFT/nova_lite_1_0_p5_p4d_gpu_ppo_rft.yaml) | [run_nova_lite_p5_gpu_ppo_rft.sh](../../launcher_scripts/nova/run_nova_lite_p5_gpu_ppo_rft.sh) |

### Nova Pro

| Technique | Sequence Length | Nodes | Instance Type | Recipe | Launch Script |
|-----------|-----------------|-------|---------------|--------|---------------|
| SFT (LoRA) | 65536 | 6 | ml.p5.48xlarge | [nova_pro_1_0_p5_p4d_gpu_lora_sft.yaml](fine-tuning/nova/nova_1_0/nova_pro/SFT/nova_pro_1_0_p5_p4d_gpu_lora_sft.yaml) | [run_nova_pro_p5_gpu_lora_sft.sh](../../launcher_scripts/nova/run_nova_pro_p5_gpu_lora_sft.sh) |
| SFT (Full) | 65536 | 6 | ml.p5.48xlarge | [nova_pro_1_0_p5_gpu_sft.yaml](fine-tuning/nova/nova_1_0/nova_pro/SFT/nova_pro_1_0_p5_gpu_sft.yaml) | [run_nova_pro_p5_gpu_sft.sh](../../launcher_scripts/nova/run_nova_pro_p5_gpu_sft.sh) |
| RFT (DPO) | 65536 | 6 | ml.p5.48xlarge | [nova_pro_1_0_p5_p4d_gpu_dpo_rft.yaml](fine-tuning/nova/nova_1_0/nova_pro/RFT/nova_pro_1_0_p5_p4d_gpu_dpo_rft.yaml) | [run_nova_pro_p5_gpu_dpo_rft.sh](../../launcher_scripts/nova/run_nova_pro_p5_gpu_dpo_rft.sh) |
| RFT (PPO) | 65536 | 6 | ml.p5.48xlarge | [nova_pro_1_0_p5_p4d_gpu_ppo_rft.yaml](fine-tuning/nova/nova_1_0/nova_pro/RFT/nova_pro_1_0_p5_p4d_gpu_ppo_rft.yaml) | [run_nova_pro_p5_gpu_ppo_rft.sh](../../launcher_scripts/nova/run_nova_pro_p5_gpu_ppo_rft.sh) |

## Evaluation Recipes

| Evaluation Type | Recipe | Launch Script |
|-----------------|--------|---------------|
| Deterministic Evaluation | [open_source_deterministic_eval.yaml](evaluation/open_source_deterministic_eval.yaml) | [run_open_source_deterministic_eval.sh](../../launcher_scripts/evaluation/run_open_source_deterministic_eval.sh) |
| LLM-as-Judge Evaluation | [open_source_llmaj_eval.yaml](evaluation/open_source_llmaj_eval.yaml) | [run_open_source_llmaj_eval.sh](../../launcher_scripts/evaluation/run_open_source_llmaj_eval.sh) |

## VERL (Versatile Reinforcement Learning) Recipes

VERL recipes support reinforcement learning from AI feedback (RLAIF) and reinforcement learning from verifiable rewards (RLVR) using the GRPO algorithm.

### Llama Models

| Model | Technique | Recipe |
|-------|-----------|--------|
| Llama 3.1 8B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-llama-3-dot-1-8b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-1-8b-instruct-lora.yaml) |
| Llama 3.1 8B Instruct | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-llama-3-dot-1-8b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-1-8b-instruct-fft.yaml) |
| Llama 3.1 8B Instruct | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-llama-3-dot-1-8b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-1-8b-instruct-lora.yaml) |
| Llama 3.1 8B Instruct | GRPO + RLVR (FFT) | [verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-1-8b-instruct-fft.yaml) |
| Llama 3.2 1B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-llama-3-dot-2-1b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-2-1b-instruct-lora.yaml) |
| Llama 3.2 1B Instruct | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-llama-3-dot-2-1b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-2-1b-instruct-lora.yaml) |
| Llama 3.2 1B Instruct | GRPO + RLVR (FFT) | [verl-grpo-rlvr-llama-3-dot-2-1b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-2-1b-instruct-fft.yaml) |
| Llama 3.2 3B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-llama-3-dot-2-3b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-2-3b-instruct-lora.yaml) |
| Llama 3.2 3B Instruct | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-llama-3-dot-2-3b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-2-3b-instruct-lora.yaml) |
| Llama 3.2 3B Instruct | GRPO + RLVR (FFT) | [verl-grpo-rlvr-llama-3-dot-2-3b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-2-3b-instruct-fft.yaml) |
| Llama 3.3 70B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-llama-3-dot-3-70b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-3-70b-instruct-lora.yaml) |
| Llama 3.3 70B Instruct | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-llama-3-dot-3-70b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlaif-llama-3-dot-3-70b-instruct-fft.yaml) |
| Llama 3.3 70B Instruct | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-llama-3-dot-3-70b-instruct-lora.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-3-70b-instruct-lora.yaml) |
| Llama 3.3 70B Instruct | GRPO + RLVR (FFT) | [verl-grpo-rlvr-llama-3-dot-3-70b-instruct-fft.yaml](fine-tuning/llama/verl-grpo-rlvr-llama-3-dot-3-70b-instruct-fft.yaml) |

### Qwen Models

| Model | Technique | Recipe |
|-------|-----------|--------|
| Qwen 2.5 7B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-lora.yaml) |
| Qwen 2.5 7B Instruct | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-fft.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-fft.yaml) |
| Qwen 2.5 14B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-2-dot-5-14b-instruct-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-14b-instruct-lora.yaml) |
| Qwen 2.5 14B Instruct | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-qwen-2-dot-5-14b-instruct-fft.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-14b-instruct-fft.yaml) |
| Qwen 2.5 32B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-2-dot-5-32b-instruct-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-32b-instruct-lora.yaml) |
| Qwen 2.5 32B Instruct | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-qwen-2-dot-5-32b-instruct-fft.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-32b-instruct-fft.yaml) |
| Qwen 2.5 72B Instruct | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-2-dot-5-72b-instruct-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-2-dot-5-72b-instruct-lora.yaml) |
| Qwen 3 0.6B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-3-0-dot-6b-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-0-dot-6b-lora.yaml) |
| Qwen 3 0.6B | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-qwen-3-0-dot-6b-fft.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-0-dot-6b-fft.yaml) |
| Qwen 3 1.7B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-3-1-dot-7b-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-1-dot-7b-lora.yaml) |
| Qwen 3 4B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-3-4b-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-4b-lora.yaml) |
| Qwen 3 14B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-3-14b-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-14b-lora.yaml) |
| Qwen 3 32B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-qwen-3-32b-lora.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-32b-lora.yaml) |
| Qwen 3 32B | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-qwen-3-32b-fft.yaml](fine-tuning/qwen/verl-grpo-rlaif-qwen-3-32b-fft.yaml) |

### DeepSeek R1 Distilled Models

| Model | Technique | Recipe |
|-------|-----------|--------|
| DeepSeek R1 Distilled Llama 8B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-llama-8b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-llama-8b-lora.yaml) |
| DeepSeek R1 Distilled Llama 8B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-llama-8b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-llama-8b-lora.yaml) |
| DeepSeek R1 Distilled Llama 8B | GRPO + RLVR (FFT) | [verl-grpo-rlvr-deepseek-r1-distilled-llama-8b-fft.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-llama-8b-fft.yaml) |
| DeepSeek R1 Distilled Llama 70B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-llama-70b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-llama-70b-lora.yaml) |
| DeepSeek R1 Distilled Llama 70B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-llama-70b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-llama-70b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 1.5B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-qwen-1-dot-5b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-qwen-1-dot-5b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 1.5B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-1-dot-5b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-1-dot-5b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 7B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-qwen-7b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-qwen-7b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 7B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-7b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-7b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 14B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-qwen-14b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-qwen-14b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 14B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-14b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-14b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 14B | GRPO + RLVR (FFT) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-14b-fft.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-14b-fft.yaml) |
| DeepSeek R1 Distilled Qwen 32B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-deepseek-r1-distilled-qwen-32b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-qwen-32b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 32B | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-deepseek-r1-distilled-qwen-32b-fft.yaml](fine-tuning/deepseek/verl-grpo-rlaif-deepseek-r1-distilled-qwen-32b-fft.yaml) |
| DeepSeek R1 Distilled Qwen 32B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-32b-lora.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-32b-lora.yaml) |
| DeepSeek R1 Distilled Qwen 32B | GRPO + RLVR (FFT) | [verl-grpo-rlvr-deepseek-r1-distilled-qwen-32b-fft.yaml](fine-tuning/deepseek/verl-grpo-rlvr-deepseek-r1-distilled-qwen-32b-fft.yaml) |

### GPT-OSS Models

| Model | Technique | Recipe |
|-------|-----------|--------|
| GPT-OSS 20B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-gpt-oss-20b-lora.yaml](fine-tuning/gpt_oss/verl-grpo-rlaif-gpt-oss-20b-lora.yaml) |
| GPT-OSS 20B | GRPO + RLAIF (FFT) | [verl-grpo-rlaif-gpt-oss-20b-fft.yaml](fine-tuning/gpt_oss/verl-grpo-rlaif-gpt-oss-20b-fft.yaml) |
| GPT-OSS 20B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-gpt-oss-20b-lora.yaml](fine-tuning/gpt_oss/verl-grpo-rlvr-gpt-oss-20b-lora.yaml) |
| GPT-OSS 20B | GRPO + RLVR (FFT) | [verl-grpo-rlvr-gpt-oss-20b-fft.yaml](fine-tuning/gpt_oss/verl-grpo-rlvr-gpt-oss-20b-fft.yaml) |
| GPT-OSS 120B | GRPO + RLAIF (LoRA) | [verl-grpo-rlaif-gpt-oss-120b-lora.yaml](fine-tuning/gpt_oss/verl-grpo-rlaif-gpt-oss-120b-lora.yaml) |
| GPT-OSS 120B | GRPO + RLVR (LoRA) | [verl-grpo-rlvr-gpt-oss-120b-lora.yaml](fine-tuning/gpt_oss/verl-grpo-rlvr-gpt-oss-120b-lora.yaml) |
