# HyperPod Recipe Overridable Parameter Reference

This document contains the list of parameters that can be overridden when using the recipes repo through SMTJ Serverless Model Customization. All parameters are available in serverful usage but these are the ranges we recommend using for successful results.

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
| `data_path` | string | Yes | — | File path to training dataset |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `global_batch_size` | integer | Yes | 8, 16, 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `gradient_clipping` | boolean | Yes | — | Boolean flag to enable gradient norm clipping. When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients during training. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `logging_steps` | integer | Yes | 1–100 | Frequency (in optimizer steps) at which training metrics are logged. A value of 1 logs every step. Higher values reduce logging overhead. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_dropout` | float | Yes | 0.0–1.0 | Dropout probability applied to LoRA adapter layers for regularization. Typically set to 0.05 for SFT/DPO, and 0 for Mixture-of-Experts models. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Number of trainable dimensions in the LoRA adapter. Higher values increase model capacity but use more memory. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `merge_weights` | boolean | Yes | — | When enabled, merges LoRA adapter weights into the base model weights after training, producing a single consolidated model. Disable if you need to keep adapters separate (e.g., for model merging workflows). |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility. Controls data shuffling order, weight initialization, and other stochastic operations. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `training_data_name` | string | Yes | — | Name identifier for the training dataset |
| `validation_data_name` | string | Yes | — | Name identifier for the validation dataset |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | Yes | 1e-03–0.1 | Controls how strongly the model enforces preference rankings. Higher values make the model more sensitive to preference differences. |
| `data_path` | string | Yes | — | File path to training dataset |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `gradient_clipping` | boolean | Yes | — | Boolean flag to enable gradient norm clipping. When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients during training. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `logging_steps` | integer | Yes | 1–100 | Frequency (in optimizer steps) at which training metrics are logged. A value of 1 logs every step. Higher values reduce logging overhead. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_dropout` | float | Yes | 0.0–1.0 | Dropout probability applied to LoRA adapter layers for regularization. Typically set to 0.05 for SFT/DPO, and 0 for Mixture-of-Experts models. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Number of trainable dimensions in the LoRA adapter. Higher values increase model capacity but use more memory. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `merge_weights` | boolean | Yes | — | When enabled, merges LoRA adapter weights into the base model weights after training, producing a single consolidated model. Disable if you need to keep adapters separate (e.g., for model merging workflows). |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility. Controls data shuffling order, weight initialization, and other stochastic operations. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `training_data_name` | string | Yes | — | Name identifier for the training dataset |
| `validation_data_name` | string | Yes | — | Name identifier for the validation dataset |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### Fine-Tuning

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_path` | string | Yes | — | File path to training dataset |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `gradient_clipping` | boolean | Yes | — | Boolean flag to enable gradient norm clipping. When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients during training. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `logging_steps` | integer | Yes | 1–100 | Frequency (in optimizer steps) at which training metrics are logged. A value of 1 logs every step. Higher values reduce logging overhead. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility. Controls data shuffling order, weight initialization, and other stochastic operations. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `training_data_name` | string | Yes | — | Name identifier for the training dataset |
| `validation_data_name` | string | Yes | — | Name identifier for the validation dataset |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_path` | string | Yes | — | File path to training dataset |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `global_batch_size` | integer | Yes | 8, 16, 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `gradient_clipping` | boolean | Yes | — | Boolean flag to enable gradient norm clipping. When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients during training. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `logging_steps` | integer | Yes | 1–100 | Frequency (in optimizer steps) at which training metrics are logged. A value of 1 logs every step. Higher values reduce logging overhead. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `max_response_length` | integer | Yes | 100–200000 | Maximum number of tokens allowed for the generated response portion. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility. Controls data shuffling order, weight initialization, and other stochastic operations. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `training_data_name` | string | Yes | — | Name identifier for the training dataset |
| `validation_data_name` | string | Yes | — | Name identifier for the validation dataset |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | Yes | 1e-03–0.1 | Controls how strongly the model enforces preference rankings. Higher values make the model more sensitive to preference differences. |
| `data_path` | string | Yes | — | File path to training dataset |
| `dataset_max_len` | integer | Yes | 256–131072 | Maximum sequence length in tokens for tokenized training inputs. Sequences longer than this are truncated. |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `gradient_clipping` | boolean | Yes | — | Boolean flag to enable gradient norm clipping. When enabled, gradients are scaled down if their norm exceeds the threshold, preventing exploding gradients during training. |
| `gradient_clipping_threshold` | float | Yes | 0.0–5.0 | Maximum allowed gradient norm. Gradients exceeding this value are proportionally scaled down. |
| `learning_rate` | float | Yes | 5e-07–1e-04 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `logging_steps` | integer | Yes | 1–100 | Frequency (in optimizer steps) at which training metrics are logged. A value of 1 logs every step. Higher values reduce logging overhead. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `max_response_length` | integer | Yes | 100–200000 | Maximum number of tokens allowed for the generated response portion. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `seed` | integer | Yes | 0–2147483647 | Random seed for reproducibility. Controls data shuffling order, weight initialization, and other stochastic operations. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `training_data_name` | string | Yes | — | Name identifier for the training dataset |
| `validation_data_name` | string | Yes | — | Name identifier for the validation dataset |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

## VERL (Versatile Reinforcement Learning)

### GRPO RLAIF (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `clip_ratio` | float | Yes | 0.1–1.5 | Limits how much the model policy changes in a single update. Values that are too high cause instability, and values that are too low slow learning. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Sets the maximum allowed increase in token probability per update for positive-advantage tokens. Higher values allow more aggressive policy changes but may reduce output diversity. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Sets the maximum allowed decrease in token probability per update for negative-advantage tokens. Lower values promote exploration and help prevent the model from converging too quickly. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `judge_model_id` | string | Yes | — | Bedrock model ID used as the judge for RLAIF evaluation |
| `judge_prompt_template` | string | No | /opt/ml/code/verl/cot.jinja, /opt/ml/code/verl/evaluate.jinja, /opt/ml/code/verl/faithfulness.jinja, /opt/ml/code/verl/summarize.jinja | Judge prompt template for RLAIF |
| `kl_loss_coef` | float | Yes | 0–0.1 | Controls how closely the model stays to its original behavior during training. Higher values constrain changes more, and lower values allow the model to drift further. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Number of trainable dimensions in the LoRA adapter. Higher values increase model capacity but use more memory. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during training. More samples improve training quality but increase compute time. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Controls the diversity of generated responses during training rollouts. Higher values produce more varied outputs. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `use_kl_loss` | boolean | Yes | — | Adds a penalty that keeps the training policy close to the reference policy. Disable to rely onclipping alone for training stability. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### GRPO RLVR (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `clip_ratio` | float | Yes | 0.1–1.5 | Limits how much the model policy changes in a single update. Values that are too high cause instability, and values that are too low slow learning. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Sets the maximum allowed increase in token probability per update for positive-advantage tokens. Higher values allow more aggressive policy changes but may reduce output diversity. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Sets the maximum allowed decrease in token probability per update for negative-advantage tokens. Lower values promote exploration and help prevent the model from converging too quickly. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Controls how closely the model stays to its original behavior during training. Higher values constrain changes more, and lower values allow the model to drift further. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Number of trainable dimensions in the LoRA adapter. Higher values increase model capacity but use more memory. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `preset_reward_function` | string | Yes | , gsm8k, prime_code, prime_math | Preset reward function for RLVR |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `reward_lambda_arn` | string | No | — | ARN of the Lambda function used for reward computation |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during training. More samples improve training quality but increase compute time. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Controls the diversity of generated responses during training rollouts. Higher values produce more varied outputs. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `use_kl_loss` | boolean | Yes | — | Adds a penalty that keeps the training policy close to the reference policy. Disable to rely onclipping alone for training stability. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### GRPO RLAIF (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `clip_ratio` | float | Yes | 0.1–1.5 | Limits how much the model policy changes in a single update. Values that are too high cause instability, and values that are too low slow learning. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Sets the maximum allowed increase in token probability per update for positive-advantage tokens. Higher values allow more aggressive policy changes but may reduce output diversity. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Sets the maximum allowed decrease in token probability per update for negative-advantage tokens. Lower values promote exploration and help prevent the model from converging too quickly. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `judge_model_id` | string | Yes | — | Bedrock model ID used as the judge for RLAIF evaluation |
| `judge_prompt_template` | string | No | /opt/ml/code/verl/cot.jinja, /opt/ml/code/verl/evaluate.jinja, /opt/ml/code/verl/faithfulness.jinja, /opt/ml/code/verl/summarize.jinja | Judge prompt template for RLAIF |
| `kl_loss_coef` | float | Yes | 0–0.1 | Controls how closely the model stays to its original behavior during training. Higher values constrain changes more, and lower values allow the model to drift further. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during training. More samples improve training quality but increase compute time. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Controls the diversity of generated responses during training rollouts. Higher values produce more varied outputs. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `use_kl_loss` | boolean | Yes | — | Adds a penalty that keeps the training policy close to the reference policy. Disable to rely onclipping alone for training stability. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Number of trainable dimensions in the LoRA adapter. Higher values increase model capacity but use more memory. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. Cosine anneals smoothly from peak to near-zero. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### GRPO RLVR (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `clip_ratio` | float | Yes | 0.1–1.5 | Limits how much the model policy changes in a single update. Values that are too high cause instability, and values that are too low slow learning. |
| `clip_ratio_high` | float | Yes | 0.0–0.5 | Sets the maximum allowed increase in token probability per update for positive-advantage tokens. Higher values allow more aggressive policy changes but may reduce output diversity. |
| `clip_ratio_low` | float | Yes | 0.0–0.5 | Sets the maximum allowed decrease in token probability per update for negative-advantage tokens. Lower values promote exploration and help prevent the model from converging too quickly. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 64, 128, 256, 512, 1024 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `kl_loss_coef` | float | Yes | 0–0.1 | Controls how closely the model stays to its original behavior during training. Higher values constrain changes more, and lower values allow the model to drift further. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent gradually increasing the learning rate from zero. Helps stabilize early training. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `preset_reward_function` | string | Yes | , gsm8k, prime_code, prime_math | Preset reward function for RLVR |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `reward_lambda_arn` | string | No | — | ARN of the Lambda function used for reward computation |
| `rollout_n` | integer | Yes | 1, 2, 4, 8, 16, 32 | Number of candidate responses generated per prompt during training. More samples improve training quality but increase compute time. |
| `rollout_temperature` | float | Yes | 0.01–2.0 | Controls the diversity of generated responses during training rollouts. Higher values produce more varied outputs. |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `use_kl_loss` | boolean | Yes | — | Adds a penalty that keeps the training policy close to the reference policy. Disable to rely onclipping alone for training stability. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | Yes | 1e-03–1.0 | KL divergence penalty coefficient for DPO. Controls how much the policy is penalized for deviating from the reference model. Lower values allow more deviation. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. |
| `lora_alpha` | integer | Yes | 16, 32, 64, 128, 256 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `lora_rank` | integer | Yes | 8, 16, 32, 64, 128 | Rank of the low-rank decomposition in LoRA adapters. Higher rank increases expressiveness but uses more memory and compute. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. 'cosine' anneals the LR smoothly from peak to near-zero following a cosine curve. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | Yes | 1e-03–1.0 | KL divergence penalty coefficient for DPO. Controls how much the policy is penalized for deviating from the reference model. Lower values allow more deviation. |
| `data_path` | string | Yes | — | File path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256, 512, 1024 | Total number of samples processed per optimizer step across all GPUs and accumulation steps. Larger batch sizes provide more stable gradients but require more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. |
| `lr_scheduler` | string | Yes | cosine, constant | Learning rate decay schedule over training. 'cosine' anneals the LR smoothly from peak to near-zero following a cosine curve. |
| `lr_warmup_steps_ratio` | float | Yes | 0–1 | Fraction of total training steps spent linearly ramping the learning rate from 0 up to the target value. Stabilizes early training by avoiding large initial updates. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 0.0–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `resume_from_path` | string | No | — | Path to checkpoint for resuming training |
| `train_val_split_ratio` | float | No | 0.0–1.0 | Fraction of the dataset allocated to training versus validation. For example, 0.9 means 90% training and 10% validation. |
| `validation_data_path` | string | Yes | — | File path to validation dataset |
| `warmup_steps` | integer | Yes | -1–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

## Amazon Nova

### SFT (LoRA) 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 1e-07–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | Number of training pods (adjust based on cluster capacity) |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (LoRA) 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `fine_tuned_model` | float | No | 0.0–1.0 | Model merging weight controlling the contribution of the fine-tuned model when merging with the base model. Ranges from 0.0 (base model only) to 1.0 (fine-tuned model only). |
| `global_batch_size` | integer | Yes | 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `limit_val_batches` | integer | No | 0–100000 | Number of validation batches to run per evaluation step. |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_steps` | integer | Yes | 4–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `reasoning_enabled` | boolean | Yes | — | When enabled, activates chain-of-thought reasoning mode during training. |
| `replicas` | integer | Yes | 4, 8 | Number of training pods (adjust based on cluster capacity) |
| `save_steps` | integer | Yes | 1–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |
| `val_check_interval` | integer | No | 1–100000 | Number of training steps between validation runs. |
| `validation_s3_path` | string | No | — | S3 URI path to validation dataset |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (FFT) 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `fine_tuned_model` | float | No | 0.0–1.0 | Model merging weight controlling the contribution of the fine-tuned model when merging with the base model. Ranges from 0.0 (base model only) to 1.0 (fine-tuned model only). |
| `global_batch_size` | integer | Yes | 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `limit_val_batches` | integer | No | 0–100000 | Number of validation batches to run per evaluation step. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_steps` | integer | Yes | 4–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `min_lr` | float | Yes | 1e-07–1.0 | Minimum learning rate floor at the end of the LR schedule. Prevents the learning rate from decaying below this value. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `reasoning_enabled` | boolean | Yes | — | When enabled, activates chain-of-thought reasoning mode during training. |
| `replicas` | integer | Yes | 4, 8 | Number of training pods (adjust based on cluster capacity) |
| `save_steps` | integer | Yes | 1–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |
| `val_check_interval` | integer | No | 1–100000 | Number of training steps between validation runs. |
| `validation_s3_path` | string | No | — | S3 URI path to validation dataset |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | No | 1e-03–0.1 | Controls how strongly the model enforces preference rankings. Higher values make the model more sensitive to preference differences. |
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `lora_alpha` | integer | Yes | 32, 64, 96, 128, 160, 192 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 1e-07–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | Number of training pods (adjust based on cluster capacity) |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### SFT (FFT) 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 1e-07–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | Number of training pods (adjust based on cluster capacity) |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### DPO (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `adam_beta` | float | No | 1e-03–0.1 | Controls how strongly the model enforces preference rankings. Higher values make the model more sensitive to preference differences. |
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16, 32, 64, 128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `min_lr` | float | Yes | 1e-07–1.0 | Lowest learning rate the scheduler decays to during training. Prevents the learning rate from dropping to zero in later steps. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 1, 2, 4, 6, 8, 12, 16, 24 | Number of training pods (adjust based on cluster capacity) |
| `warmup_steps` | integer | Yes | 0–100 | Absolute number of training steps over which the learning rate linearly warms up from 0 to the target value. |
| `weight_decay` | float | Yes | 0.0–1.0 | L2 regularization coefficient applied to model weights during optimization. Helps prevent overfitting by penalizing large weights. |

### PPO

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 160 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 2, 4, 6 | Number of training pods (adjust based on cluster capacity) |

### Distillation

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `customer_bucket` | string | Yes | — | Customer S3 bucket for data storage |
| `kms_key` | string | Yes | — | S3 Encryption KMS key |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_response_length` | integer | No | 5000–5120 | Maximum number of tokens allowed for the generated response portion. |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |

### Text Benchmark Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `subtask` | string | No | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Text Benchmark Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–32768 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `subtask` | string | No | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Multi-Modal Benchmark Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `subtask` | string | No | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Multi-Modal Benchmark Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–32768 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `subtask` | string | No | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### LLM-as-Judge Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `max_new_tokens` | integer | No | 0–100000 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Custom Dataset Eval 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `lambda_arn` | string | Yes | — | — |
| `lambda_type` | string | Yes | rft, custom_metrics | — |
| `max_new_tokens` | integer | No | 0–100000 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `strategy` | string | Yes | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Custom Dataset Eval 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `lambda_arn` | string | Yes | — | — |
| `lambda_type` | string | Yes | rft, custom_metrics | — |
| `max_new_tokens` | integer | No | 0–32768 | — |
| `metric` | string | Yes | — | — |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `preset_reward_function` | string | Yes | prime_math, prime_code | Preset reward function for RLVR |
| `strategy` | string | Yes | — | — |
| `task` | string | Yes | — | — |
| `temperature` | float | Yes | 0.0–2.0 | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | float | No | 0.0–1.0 | Top-p nucleus sampling cutoff |

### Pretraining 1.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_epochs` | integer | Yes | 1–100 | Number of complete passes through the entire training dataset. More epochs allow the model to learn patterns more thoroughly but increase training time and risk of overfitting. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 2, 4, 6, 8, 12, 16, 24, 32 | Number of training pods (adjust based on cluster capacity) |

### Pretraining 2.0

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 32, 64, 128, 256 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 0–1 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_steps` | integer | Yes | 10–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 2, 4, 6, 8, 12, 16, 24, 32 | Number of training pods (adjust based on cluster capacity) |
| `validation_s3_path` | string | No | — | S3 URI path to validation dataset |

### RFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16–256 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-08–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `learning_rate_ratio` | float | No | 1.0–70.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `lora_alpha` | integer | Yes | 16, 32, 64 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `max_length` | integer | Yes | 4096–32768 | Maximum number of tokens in a single training example. Longer sequences capture more context but use more memory. |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `number_generation` | integer | Yes | 2–16 | Number of generations per prompt in rollout |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 2, 4, 8, 16 | Number of training pods (adjust based on cluster capacity) |
| `reward_lambda_arn` | string | Yes | — | ARN of the Lambda function used for reward computation |
| `save_steps` | integer | Yes | 0–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |

### RFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16–256 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-08–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_context_length` | integer | Yes | 1–131072 | Maximum sequence length for training inputs. Determines which instance types will be used based on memory requirements. |
| `max_length` | integer | Yes | 4096–32768 | Maximum number of tokens in a single training example. Longer sequences capture more context but use more memory. |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `namespace` | string | Yes | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `number_generation` | integer | Yes | 2–16 | Number of generations per prompt in rollout |
| `output_s3_path` | string | Yes | — | The S3 path where the trained model artifacts are stored after the job completes. |
| `replicas` | integer | Yes | 2, 4, 8, 16 | Number of training pods (adjust based on cluster capacity) |
| `reward_lambda_arn` | string | Yes | — | ARN of the Lambda function used for reward computation |
| `save_steps` | integer | Yes | 0–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |

### SMTJ RFT (LoRA)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16–128 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `learning_rate_ratio` | float | No | 0.0–100.0 | Scaling factor that controls the relative learning rate between LoRA adapter parameters and base model parameters. |
| `lora_alpha` | integer | Yes | 16, 32, 64 | LoRA scaling factor. The effective learning rate for LoRA adapters scales as alpha/rank. Typically set to 2x the LoRA rank. |
| `max_length` | integer | Yes | 4096–32768 | Maximum number of tokens in a single training example. Longer sequences capture more context but use more memory. |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `name` | string | Yes | — | Custom model name |
| `reward_lambda_arn` | string | Yes | — | ARN of the Lambda function used for reward computation |
| `save_steps` | integer | Yes | 0–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |

### SMTJ RFT (FFT)

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `data_s3_path` | string | Yes | — | S3 URI path to training dataset |
| `global_batch_size` | integer | Yes | 16–256 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `learning_rate` | float | Yes | 1e-07–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `max_length` | integer | Yes | 4096–32768 | Maximum number of tokens in a single training example. Longer sequences capture more context but use more memory. |
| `max_steps` | integer | Yes | 5–100000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `name` | string | Yes | — | Custom model name |
| `reward_lambda_arn` | string | Yes | — | ARN of the Lambda function used for reward computation |
| `save_steps` | integer | Yes | 0–100000 | How often to save model checkpoints, measured in optimizer steps. A value of 0 saves only at the end of training. |

## Checkpointless

### Fine-Tuning

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `dataset_path` | string | Yes | — | — |
| `global_batch_size` | integer | Yes | 2, 4, 8, 16, 32, 64 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `instance_count` | integer | Yes | 1–128 | Number of compute instances for training |
| `instance_types` | string | Yes | ml.p5.48xlarge, ml.p5e.48xlarge | — |
| `learning_rate` | float | Yes | 1e-06–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `log_directory` | string | Yes | — | — |
| `log_every_n_steps` | integer | Yes | 1–1000 | — |
| `max_steps` | integer | Yes | 1000–10000000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `model_name` | string | Yes | — | — |
| `namespace` | string | No | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `num_nodes` | integer | Yes | 1–128 | — |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `sequence_length` | integer | Yes | 1024, 2048, 4096, 8192, 16384, 32768 | — |
| `time_limit` | string | Yes | — | — |
| `val_check_interval` | integer | Yes | 100–10000 | Number of training steps between validation runs. |

### Pretraining

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `dataset_path` | string | Yes | — | — |
| `global_batch_size` | integer | Yes | 2, 4, 8, 16, 32, 64 | Total number of training samples processed per optimizer step. Larger values improve training stability but use more memory. |
| `instance_count` | integer | Yes | 1–128 | Number of compute instances for training |
| `instance_types` | string | Yes | ml.p5.48xlarge, ml.p5e.48xlarge | — |
| `learning_rate` | float | Yes | 1e-06–1e-03 | Step size for weight updates during optimization. Controls how much model weights change per gradient step. Set lower for RL (e.g., 1e-5) to avoid policy collapse; higher for SFT/DPO (e.g., 1e-4). |
| `log_directory` | string | Yes | — | — |
| `log_every_n_steps` | integer | Yes | 1–1000 | — |
| `max_steps` | integer | Yes | 1000–10000000 | Maximum number of optimizer steps. When set, training stops after this many steps regardless of epochs remaining. |
| `model_name` | string | Yes | — | — |
| `namespace` | string | No | — | Namespace for HyperPod cluster resources (default: kubeflow) |
| `num_nodes` | integer | Yes | 1–128 | — |
| `results_directory` | string | Yes | — | Directory for storing training results and logs |
| `sequence_length` | integer | Yes | 1024, 2048, 4096, 8192, 16384, 32768 | — |
| `time_limit` | string | Yes | — | — |
| `val_check_interval` | integer | Yes | 100–10000 | Number of training steps between validation runs. |

## Evaluation

### Deterministic Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `aggregation` | string | No | average, median, max, min, sum | — |
| `base_model_name` | string | Yes | — | — |
| `data_s3_path` | string | No | — | S3 URI path to training dataset |
| `eval_tensorboard_results_dir` | string | No | — | Directory for evaluation TensorBoard results |
| `evaluation_metric` | string | No | all, deflection, accuracy, exact_match | — |
| `instance_count` | integer | Yes | 1–128 | Number of compute instances for training |
| `kms_key_id` | string | No | — | — |
| `lambda_arn` | string | No | — | — |
| `max_model_len` | integer | No | — | — |
| `max_new_tokens` | integer | No | — | — |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `model_name_or_path` | string | Yes | — | Model identifier or path to pretrained model |
| `name` | string | Yes | — | Custom model name |
| `output_path` | string | Yes | — | Local output path for model artifacts |
| `postprocessing` | boolean | No | — | — |
| `preset_reward_function` | string | Yes | , gsm8k, prime_code, prime_math | Preset reward function for RLVR |
| `strategy` | string | No | zs_cot, gen_qa, zs, fs_cot | — |
| `subtask` | string | No | abstract_algebra, anatomy, astronomy, business_ethics, … (91 values) | — |
| `task` | string | Yes | mmlu, mmlu_pro, bbh, gpqa, math, strong_reject, ifeval, gen_qa, inference_only | — |
| `temperature` | integer | No | — | Sampling temperature for evaluation and generation. Higher values increase output randomness/diversity; lower values make outputs more deterministic. |
| `top_k` | integer | No | — | Top-k nucleus sampling cutoff |
| `top_p` | number | No | — | Top-p nucleus sampling cutoff |

### LLM-as-Judge Eval

| Parameter | Type | Required | Range / Values | Description |
|-----|-----|-----|-----|-----|
| `base_model_inference_data_s3_path` | string | No | — | — |
| `base_model_name` | string | Yes | — | — |
| `custom_metrics` | string | No | — | — |
| `eval_tensorboard_results_dir` | string | No | — | Directory for evaluation TensorBoard results |
| `inference_data_s3_path` | string | Yes | — | — |
| `judge_model_id` | string | Yes | — | Bedrock model ID used as the judge for RLAIF evaluation |
| `kms_key_id` | string | No | — | — |
| `llmaj_metrics` | string | No | Correctness, Completeness, Faithfulness, Helpfulness, … (15 values) | — |
| `mlflow_experiment_name` | string | No | — | MLflow Groups related training runs for comparison. Uses the custom model name by default. |
| `mlflow_run_id` | string | No | — | MLflow run ID for resuming or linking to an existing run |
| `mlflow_run_name` | string | No | — | Identifies this specific training run within the experiment. Autogenerated by default. |
| `mlflow_tracking_uri` | string | No | — | MLflow tracking server URI for experiment logging |
| `name` | string | Yes | — | Custom model name |
| `output_path` | string | No | — | Local output path for model artifacts |
