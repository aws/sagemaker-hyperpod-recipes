# HyperPod Recipe Hyperparameter Reference

This document contains the list of hyperparameters available when using the recipes repo through SMTJ Serverless Model Customization. All parameters are available in serverful usage but these are the ranges we recommend using for successful results.

## Table of Contents

- [LLMFT (LLM Fine-Tuning Framework)](#llmft-llm-fine-tuning-framework)
- [VERL (Versatile Reinforcement Learning)](#verl-versatile-reinforcement-learning)
- [Amazon Nova](#amazon-nova)
- [Checkpointless](#checkpointless)
- [Evaluation](#evaluation)

## LLMFT (LLM Fine-Tuning Framework)

### SFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. |
| `global_batch_size` | integer | Yes | 8, 16, 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step across all compute instances. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Lower for RL to avoid policy collapse; higher for SFT/DPO. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights. Helps prevent overfitting by penalizing large weights. |
| `gradient_clipping` | boolean | Yes | — | When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients. |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility of data shuffling and weight initialization. |
| `logging_steps` | integer | Yes | 1–100 | Frequency in optimizer steps at which training metrics are logged. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Dimensionality of the low-rank decomposition matrices. Lower rank means fewer trainable parameters. |
| `lora_dropout` | float | Yes | 0.0–1.0 | Dropout probability applied to LoRA adapter layers for regularization. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for adapters scales as alpha/rank. |
| `merge_weights` | boolean | Yes | — | When enabled, merges LoRA adapter weights into the base model after training. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output diversity. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |

### DPO (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step across all compute instances. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Lower for RL to avoid policy collapse; higher for SFT/DPO. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 to the target value. |
| `adam_beta` | float | Yes | 1e-03–0.1 | DPO inverse temperature parameter. Controls how strongly the model enforces preference rankings. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights. Helps prevent overfitting by penalizing large weights. |
| `gradient_clipping` | boolean | Yes | — | When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients. |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility of data shuffling and weight initialization. |
| `logging_steps` | integer | Yes | 1–100 | Frequency in optimizer steps at which training metrics are logged. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Dimensionality of the low-rank decomposition matrices. Lower rank means fewer trainable parameters. |
| `lora_dropout` | float | Yes | 0.0–1.0 | Dropout probability applied to LoRA adapter layers for regularization. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for adapters scales as alpha/rank. |
| `merge_weights` | boolean | Yes | — | When enabled, merges LoRA adapter weights into the base model after training. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output diversity. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |

### Fine-Tuning

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step across all compute instances. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Lower for RL to avoid policy collapse; higher for SFT/DPO. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights. Helps prevent overfitting by penalizing large weights. |
| `gradient_clipping` | boolean | Yes | — | When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients. |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility of data shuffling and weight initialization. |
| `logging_steps` | integer | Yes | 1–100 | Frequency in optimizer steps at which training metrics are logged. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output diversity. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |

### SFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. |
| `global_batch_size` | integer | Yes | 8, 16, 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step across all compute instances. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Lower for RL to avoid policy collapse; higher for SFT/DPO. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights. Helps prevent overfitting by penalizing large weights. |
| `gradient_clipping` | boolean | Yes | — | When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients. |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility of data shuffling and weight initialization. |
| `logging_steps` | integer | Yes | 1–100 | Frequency in optimizer steps at which training metrics are logged. |
| `max_response_length` | integer | Yes | 100–200000 | Maximum number of tokens allowed for the generated response portion. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output diversity. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |

### DPO (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step across all compute instances. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Lower for RL to avoid policy collapse; higher for SFT/DPO. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 to the target value. |
| `adam_beta` | float | Yes | 1e-03–0.1 | DPO inverse temperature parameter. Controls how strongly the model enforces preference rankings. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights. Helps prevent overfitting by penalizing large weights. |
| `gradient_clipping` | boolean | Yes | — | When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients. |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility of data shuffling and weight initialization. |
| `logging_steps` | integer | Yes | 1–100 | Frequency in optimizer steps at which training metrics are logged. |
| `max_response_length` | integer | Yes | 100–200000 | Maximum number of tokens allowed for the generated response portion. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output diversity. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |

## VERL (Versatile Reinforcement Learning)

### GRPO RLAIF (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_prompt_length` | integer | Yes | 512–16384 | Maximum number of tokens reserved for the prompt/input portion of each sequence. The remaining tokens are available for the response. |
| `judge_prompt_template` | string | No | /opt/ml/code/verl/cot.jinja, /opt/ml/code/verl/evaluate.jinja, /opt/ml/code/verl/faithfulness.jinja, /opt/ml/code/verl/summarize.jinja, bedrock/RLAIF/PandaLM/prompts/grader.jinja | — |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `clip_ratio` | float | Yes | 0.1–1.5 | PPO/GRPO clipping parameter that limits how much the policy can change in a single update. Too high leads to training instability; too low leads to slow learning. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Weight of the KL divergence penalty term in the RL loss function. Prevents the policy from diverging too far from the reference model. |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during GRPO rollouts. More samples improve advantage estimation quality but increase compute cost. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Sampling temperature used during rollout generation. Higher values produce more diverse candidate responses. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | LoRA rank — the dimensionality of the low-rank decomposition matrices. Lower rank means fewer trainable parameters; higher rank increases model capacity but uses more memory. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Upper clipping threshold for the PPO policy ratio. Limits the magnitude of positive policy updates. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Lower clipping threshold for the PPO policy ratio. Limits the magnitude of negative policy updates. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `use_kl_loss` | boolean | Yes | — | Boolean flag to add a KL divergence penalty to the training loss. Prevents the RL-trained policy from drifting too far from the reference model. |

### GRPO RLVR (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `preset_reward_function` | string | Yes | , gsm8k, prime_code, prime_math | — |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_prompt_length` | integer | Yes | 512–16384 | Maximum number of tokens reserved for the prompt/input portion of each sequence. The remaining tokens are available for the response. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `clip_ratio` | float | Yes | 0.1–1.5 | PPO/GRPO clipping parameter that limits how much the policy can change in a single update. Too high leads to training instability; too low leads to slow learning. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Weight of the KL divergence penalty term in the RL loss function. Prevents the policy from diverging too far from the reference model. |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during GRPO rollouts. More samples improve advantage estimation quality but increase compute cost. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Sampling temperature used during rollout generation. Higher values produce more diverse candidate responses. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | LoRA rank — the dimensionality of the low-rank decomposition matrices. Lower rank means fewer trainable parameters; higher rank increases model capacity but uses more memory. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Upper clipping threshold for the PPO policy ratio. Limits the magnitude of positive policy updates. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Lower clipping threshold for the PPO policy ratio. Limits the magnitude of negative policy updates. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `use_kl_loss` | boolean | Yes | — | Boolean flag to add a KL divergence penalty to the training loss. Prevents the RL-trained policy from drifting too far from the reference model. |

### GRPO RLAIF (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_prompt_length` | integer | Yes | 512–16384 | Maximum number of tokens reserved for the prompt/input portion of each sequence. The remaining tokens are available for the response. |
| `judge_prompt_template` | string | No | /opt/ml/code/verl/cot.jinja, /opt/ml/code/verl/evaluate.jinja, /opt/ml/code/verl/faithfulness.jinja, /opt/ml/code/verl/summarize.jinja, bedrock/RLAIF/PandaLM/prompts/grader.jinja | — |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `clip_ratio` | float | Yes | 0.1–1.5 | PPO/GRPO clipping parameter that limits how much the policy can change in a single update. Too high leads to training instability; too low leads to slow learning. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Weight of the KL divergence penalty term in the RL loss function. Prevents the policy from diverging too far from the reference model. |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during GRPO rollouts. More samples improve advantage estimation quality but increase compute cost. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Sampling temperature used during rollout generation. Higher values produce more diverse candidate responses. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Upper clipping threshold for the PPO policy ratio. Limits the magnitude of positive policy updates. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Lower clipping threshold for the PPO policy ratio. Limits the magnitude of negative policy updates. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `use_kl_loss` | boolean | Yes | — | Boolean flag to add a KL divergence penalty to the training loss. Prevents the RL-trained policy from drifting too far from the reference model. |

### SFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | LoRA rank — the dimensionality of the low-rank decomposition matrices. Lower rank means fewer trainable parameters; higher rank increases model capacity but uses more memory. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. 'cosine' anneals the LR smoothly from peak to near-zero following a cosine curve. |

### SFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. 'cosine' anneals the LR smoothly from peak to near-zero following a cosine curve. |

### GRPO RLVR (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `preset_reward_function` | string | Yes | , gsm8k, prime_code, prime_math | — |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `global_batch_size` | integer | Yes | 64, 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_prompt_length` | integer | Yes | 512–16384 | Maximum number of tokens reserved for the prompt/input portion of each sequence. The remaining tokens are available for the response. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `clip_ratio` | float | Yes | 0.1–1.5 | PPO/GRPO clipping parameter that limits how much the policy can change in a single update. Too high leads to training instability; too low leads to slow learning. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Weight of the KL divergence penalty term in the RL loss function. Prevents the policy from diverging too far from the reference model. |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during GRPO rollouts. More samples improve advantage estimation quality but increase compute cost. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Sampling temperature used during rollout generation. Higher values produce more diverse candidate responses. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Upper clipping threshold for the PPO policy ratio. Limits the magnitude of positive policy updates. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Lower clipping threshold for the PPO policy ratio. Limits the magnitude of negative policy updates. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `use_kl_loss` | boolean | Yes | — | Boolean flag to add a KL divergence penalty to the training loss. Prevents the RL-trained policy from drifting too far from the reference model. |

## Amazon Nova

### SFT (LoRA) 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | — |
| `global_batch_size` | integer | Yes | 16, 32, 64 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |

### SFT (LoRA) 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 4, 8 | — |
| `reasoning_enabled` | boolean | Yes | — | When enabled, activates chain-of-thought reasoning mode during training. |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_steps` | integer | Yes | 4–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `save_steps` | integer | Yes | 1–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `fine_tuned_model` | float | No | 0.0–1.0 | Model merging weight controlling the contribution of the fine-tuned model when merging with the base model. Ranges from 0.0 (base model only) to 1.0 (fine-tuned model only). |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `val_check_interval` | integer | No | 1–100000 | — |
| `limit_val_batches` | integer | No | 0–100000 | — |

### SFT (FFT) 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 4, 8 | — |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `reasoning_enabled` | boolean | Yes | — | When enabled, activates chain-of-thought reasoning mode during training. |
| `max_steps` | integer | Yes | 4–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `save_steps` | integer | Yes | 1–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `fine_tuned_model` | float | No | 0.0–1.0 | Model merging weight controlling the contribution of the fine-tuned model when merging with the base model. Ranges from 0.0 (base model only) to 1.0 (fine-tuned model only). |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `val_check_interval` | integer | No | 1–100000 | — |
| `limit_val_batches` | integer | No | 0–100000 | — |

### DPO (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | — |
| `global_batch_size` | integer | Yes | 16, 32, 64 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `adam_beta` | float | No | 1e-03–0.1 | DPO inverse temperature parameter. Controls how strongly the model enforces preference rankings. |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |

### SFT (FFT) 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | — |
| `global_batch_size` | integer | Yes | 16, 32, 64 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |

### DPO (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | — |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `adam_beta` | float | No | 1e-03–0.1 | DPO inverse temperature parameter. Controls how strongly the model enforces preference rankings. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |

### PPO

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `global_batch_size` | integer | Yes | 160 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `replicas` | integer | Yes | 2, 4, 6 | — |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |

### Distillation

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_response_length` | integer | No | 5000–5120 | Maximum number of tokens allowed for the generated response portion during training or evaluation. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |

### Text Benchmark Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |

### Text Benchmark Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–32768 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |

### Multi-Modal Benchmark Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |

### Multi-Modal Benchmark Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–32768 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |

### LLM-as-Judge Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |

### Custom Dataset Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `lambda_type` | string | Yes | rft, custom_metrics | — |

### Custom Dataset Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–32768 | — |
| `top_p` | float | No | 0.0–1.0 | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `lambda_type` | string | Yes | rft, custom_metrics | — |
| `preset_reward_function` | string | No | prime_math, prime_code | — |

### Pretraining 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 2, 4, 6, 8, 12, 16, 24, 32 | — |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |

### Pretraining 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 2, 4, 6, 8, 12, 16, 24, 32 | — |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_steps` | integer | Yes | 10–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |

### RFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 2, 4, 8, 16 | — |
| `learning_rate` | float | Yes | 1e-08–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `number_generation` | integer | Yes | 2–16 | — |
| `lora_alpha` | integer | Yes | 16, 32, 64 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `global_batch_size` | integer | Yes | 16–256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_length` | integer | Yes | 4096–32768 | — |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `save_steps` | integer | Yes | 0–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |

### RFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `replicas` | integer | Yes | 2, 4, 8, 16 | — |
| `learning_rate` | float | Yes | 1e-08–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `global_batch_size` | integer | Yes | 16–256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_length` | integer | Yes | 4096–32768 | — |
| `number_generation` | integer | Yes | 2–16 | — |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `save_steps` | integer | Yes | 0–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |

### SMTJ RFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `lora_alpha` | integer | Yes | 16, 32, 64 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `global_batch_size` | integer | Yes | 16–128 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_length` | integer | Yes | 4096–32768 | — |
| `learning_rate_ratio` | float | No | 0.0–100.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `save_steps` | integer | Yes | 0–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |

### SMTJ RFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `global_batch_size` | integer | Yes | 16–256 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `max_length` | integer | Yes | 4096–32768 | — |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `save_steps` | integer | Yes | 0–100000 | Frequency (in optimizer steps) at which model checkpoints are saved. A value of 0 saves only at the end of training. |

## Checkpointless

### Fine-Tuning

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `num_nodes` | integer | Yes | 1–128 | — |
| `max_steps` | integer | Yes | 1000–10000000 | — |
| `val_check_interval` | integer | Yes | 100–10000 | — |
| `log_every_n_steps` | integer | Yes | 1–1000 | — |
| `sequence_length` | integer | Yes | 1024, 2048, 4096, 8192, 16384, 32768 | — |
| `global_batch_size` | integer | Yes | 2, 4, 8, 16, 32, 64 | — |
| `learning_rate` | float | Yes | 1e-06–1e-03 | — |
| `instance_types` | string | Yes | ml.p5.48xlarge, ml.p5e.48xlarge | — |
| `instance_count` | integer | Yes | 1–128 | — |

### Pretraining

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `num_nodes` | integer | Yes | 1–128 | — |
| `max_steps` | integer | Yes | 1000–10000000 | — |
| `val_check_interval` | integer | Yes | 100–10000 | — |
| `log_every_n_steps` | integer | Yes | 1–1000 | — |
| `sequence_length` | integer | Yes | 1024, 2048, 4096, 8192, 16384, 32768 | — |
| `global_batch_size` | integer | Yes | 2, 4, 8, 16, 32, 64 | — |
| `learning_rate` | float | Yes | 1e-06–1e-03 | — |
| `instance_types` | string | Yes | ml.p5.48xlarge, ml.p5e.48xlarge | — |
| `instance_count` | integer | Yes | 1–128 | — |

## Evaluation

### Deterministic Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `task` | string | Yes | mmlu, mmlu_pro, bbh, gpqa, math, strong_reject, ifeval, gen_qa, inference_only | — |
| `strategy` | string | No | zs_cot, gen_qa, zs, fs_cot | — |
| `preset_reward_function` | string | No | prime_code, prime_math | — |
| `evaluation_metric` | string | No | all, deflection, accuracy, exact_match | — |
| `subtask` | string | No | abstract_algebra, anatomy, astronomy, business_ethics, … (91 values) | — |
| `aggregation` | string | No | average, median, max, min, sum | — |

### LLM-as-Judge Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `llmaj_metrics` | string | No | Correctness, Completeness, Faithfulness, Helpfulness, … (15 values) | — |
