# AGENT_CONTEXT.md

> **Purpose:** Enable an AI agent to understand this system in under 2 minutes, use frameworks correctly, and avoid breaking architecture or constraints.

---

## Overview

- **This is a multi-package workspace** for LLM fine-tuning and reinforcement learning on Amazon SageMaker HyperPod
- **Multiple ML frameworks** drive training:
  - **AWSLLMFineTuning** — FSDP-based supervised/preference fine-tuning (SFT, DPO)
  - **AWSBedrockVerl** — Ray-based reinforcement learning for LLMs (PPO, GRPO, RLAIF, RLVR), forked from Volcano Engine's verl
  - **Nova** — Amazon Nova model training (SFT/PPO/RFT) with dedicated launchers and infrastructure (`launcher/nova/`, `novaCDL/`)
  - **Eval (SupayModelLens)** — Model evaluation framework (`eval/src/SupayModelLens/`, `launcher/evaluation/`)
  - **Checkpointless NeMo** — Checkpointless training via NeMo on K8s only (`launcher/nemo/`, `launcher/recipe_templatization/checkpointless/`)
- **SageMaker HyperPod Recipes** (both public and staging) provide the orchestration layer — pre-validated YAML recipe configs that launch training jobs on HyperPod clusters (EKS/Slurm/SageMaker Training Jobs).
- **Supported models**: Llama (3.1–4), Qwen (2.5, 3), DeepSeek R1 Distilled, GPT-OSS (20B/120B), Amazon Nova.
- **Training runs on HyperPod GPU clusters** (p4d/p4de/p5 instances) with shared storage on `/fsx/ubuntu/`. Models, data, and checkpoints reside on FSx.
- **Supporting packages** include HyperPod infrastructure (elastic agent, PyTorch operator, Nova CDK, CLI tools).

---

## Key Concepts

| Term | Definition |
|------|-----------|
| **SFT** | Supervised Fine-Tuning — training on input-output pairs |
| **DPO** | Direct Preference Optimization — preference learning with chosen/rejected pairs |
| **PPO/GRPO** | RL algorithms implemented in AWSBedrockVerl for policy gradient training |
| **RLAIF** | RL from AI Feedback — uses Bedrock LLM-as-judge or dedicated vLLM server for reward signals |
| **FSDP/FSDP2** | Fully Sharded Data Parallel — PyTorch distributed training strategy used by both frameworks |
| **HSDP** | Hybrid Sharded Data Parallel — used for multi-node training in AWSLLMFineTuning |
| **LoRA/QLoRA** | Parameter-efficient fine-tuning via low-rank adapters; QLoRA adds 4-bit quantization |
| **Hydra/OmegaConf** | Configuration framework — AWSLLMFineTuning uses Hydra for hierarchical YAML config composition |
| **Ray** | Distributed computing framework — AWSBedrockVerl uses Ray to orchestrate actor/critic/rollout workers |
| **Recipe** | A pre-validated YAML config in `recipes_collection/` that defines a complete training run |
| **Brazil workspace** | Amazon internal package management system — each subdirectory with `packageInfo` is a Brazil package |

---

## Architecture & Flow

### HyperPod Recipes Orchestration Flow

1. User runs `bash launcher_scripts/<model>/run_<recipe>.sh` with env vars (`TRAIN_DATA`, `EXP_DIR`, etc.)
2. Script invokes `python3 main.py` with Hydra overrides pointing to a recipe YAML
3. `main.py` loads config from `recipes_collection/config.yaml` + recipe YAML via Hydra
4. `@validate_config` decorator runs Pydantic validation (`VerlRecipeValidator`, `LLMFTRecipeValidator`, or `NovaRecipeValidator`)
5. `preprocess_config()` normalizes config, detects framework type via `model_type` field
6. `get_training_stage()` selects the appropriate stage class (GPU/Trainium/Elastic/HPCT)
7. Stage class generates cluster-specific job scripts (Helm chart for K8s, sbatch for Slurm)
8. Job is submitted; training runs inside a container with the appropriate framework

### Routing Logic (`model_type` → Framework)

| `model_type` value | Stage Class | Framework / Container |
|---------------------|-------------|----------------------|
| `"verl"` | `SMTrainingGPURecipe` | Ray cluster + VERL container (`hyperpod-recipes:verl-v1.0.0-eks` / `verl-v1.0.0-smtj`) |
| `"llm_finetuning_aws"` | `SMTrainingGPURecipe` | torchrun + LLMFT container (`hyperpod-recipes:llmft-v1.0.0`) |
| `"hf"` | `SMTrainingGPURecipe` | HuggingFace adapter |
| `"amazon.nova*"` | `NovaK8SLauncher` | Nova SFT/PPO/RFT sub-launchers |
| `"hyperpod_checkpointless_nemo"` | `SMTrainingHPCTRecipe` | Checkpointless NeMo (K8s only) |

### Recipe YAML Naming Convention

Pattern: `{framework}_{model}_{size}_{seq}_{device}_{technique}_{strategy}.yaml`
Example: `llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml`

> ⚠️ Launcher scripts in `launcher_scripts/` are auto-generated — do NOT edit them manually.

### AWSLLMFineTuning Execution Flow
1. Entry: `torchrun src/train.py` or `src/train_hp.py` (HyperPod schema)
2. Hydra resolves config from `src/experiments/configs/` (model + dataset + training_args + strategy)
3. `train_runner.py` initializes distributed env, merges OmegaConf configs
4. Routes to appropriate trainer: `train_sft.train()`, `train_dpo.train()`
5. BaseTrainer ABC manages: epoch loop → batch loop → `training_step()` → gradient accumulation → eval → checkpoint

### AWSBedrockVerl Execution Flow
1. Entry: Slurm script provisions Ray cluster → runs training script
2. Ray orchestrates distributed workers: Actor (policy model), Critic (value function), Rollout (inference via vLLM/SGLang), Reward Model/Manager
3. For SFT: `SFTTrainer` uses OmegaConf config → builds dataset/engine/dataloader → StatefulDataLoader training loop
4. For RL: PPO/GRPO trainer coordinates actor updates, rollout generation, reward computation, critic updates
5. Checkpoints saved to `trainer.default_local_dir` (must be FSx path)

---

## Framework Reference

### AWSLLMFineTuning

**Purpose:** Primary framework for SFT and offline preference optimization (DPO). Runs on HyperPod via torchrun with FSDP.

**Usage:**
- Standard: `torchrun --standalone --nnodes N --nproc-per-node 8 src/train.py` with Hydra overrides
- HyperPod schema: `torchrun src/train_hp.py training_config/model_config=... training_config/training_args=...`
- Multi-node via Slurm `sbatch` or `torchrun` with `--rdzv_backend=c10d`
- Override configs via CLI: `model_config=X training_args=Y training_args/strategy=Z datasets=W`

**Core Patterns:**
- **Hydra Config Composition** — Configs compose via `defaults:` blocks. Override with dotted CLI paths.
- **BaseTrainer ABC** — All trainers (SFT, DPO) inherit from `BaseTrainer`. Subclasses implement `training_step()` and `evaluation_step()`.
- **Strategy Pattern** — Parallelism via `training_args/strategy` (fsdp_fft, fsdp_peft, hsdp_peft, mp)
- **Reference Model via Adapter Disable** — Preference trainers compute reference logprobs by disabling LoRA adapters (`model.disable_adapter()`) rather than loading a separate model
- **Callback System** — `on_train_begin`, `on_log`, `on_checkpoint`, `on_step_begin/end`, `on_epoch_begin/end`
- **Data Collator Pattern** — Per-trainer collators: `DataCollatorForSft`, `DataCollatorForMultiTurnPreference`, etc.
- **PlatformException Wrapping** — `train_runner.py` catches `PermissionError`, `OSError`, CUDA errors → `PlatformException`

**When to use:** SFT, DPO, offline preference optimization
**When to avoid:** Online RL (PPO/GRPO), RLAIF — use AWSBedrockVerl

---

### AWSBedrockVerl v0.5.0 (mainline)

> **Branch:** `mainline` on `code.amazon.com/packages/AWSBedrockVerl`

**Purpose:** RL training of LLMs (PPO, GRPO, RLAIF, RLVR) and verl-native SFT. Uses Ray for distributed orchestration. Forked from volcengine/verl.

**Usage:**
- RL jobs: `sbatch --nodes=2 bedrock/scripts/slurm/ray_on_slurm.slurm`
- Custom job: `sbatch --nodes=N bedrock/scripts/slurm/ray_on_slurm_runner.slurm bedrock/scripts/slurm/example_job.sh`
- Env setup: `hatch run setup_venv` (creates `.venv` with uv)
- Docker: `hatch run local_push -- <tag>`

**Core Patterns:**
- **Ray-Based Worker Architecture** — Actor, Critic, Rollout, Reward workers via Ray. Base + DataParallel + Megatron implementations.
- **BaseConfig Dataclass-as-Dict** — Frozen dataclass implementing `collections.abc.Mapping`; fields immutable unless in `_mutable_fields`
- **Multiple Rollout Engines** — vLLM (production), SGLang (structured gen), HuggingFace, Naive
- **StatefulDataLoader** — SFT trainer auto-resumes from checkpoint without manual epoch/step tracking
- **Reward Manager Registry** — Pluggable strategies (naive, batch, DAPO, prime) via registry pattern
- **OmegaConf → Dataclass Config** — SFT trainer converts OmegaConf to typed dataclasses
- **FSDP + Ulysses Sequence Parallelism** — Supported via sharding manager; disabled by default (`ulysses_sequence_parallel_size: 1`)
- **Hatch Build System** — `hatch run setup_venv`, `hatch run release`, `hatch run local_push`

**When to use:** PPO, GRPO, RLAIF, RLVR, online RL, reward model integration; also SFT when needing StatefulDataLoader or dynamic batch sizing
**When to avoid:** Simple offline preference optimization (DPO) — use AWSLLMFineTuning

---

### AWSBedrockVerl v0.7.0 (oss-0.7.0-dev)

> **Branch:** `oss-0.7.0-dev` on `code.amazon.com/packages/AWSBedrockVerl`

**Purpose:** Next-generation verl with expanded algorithm support, typed config system, agent/tool infrastructure, and broader model coverage. Drop-in replacement for v0.5.0 with significant new capabilities.

**What's new vs v0.5.0:**
- **Typed Config System** — `verl/trainer/config/` has structured sub-configs: `actor/`, `algorithm/`, `critic/`, `data/`, `engine/`, `model/`, `optim/`, `rollout/`, `ref/`, `reward_model/`, `profiler/`
- **Expanded Algorithms** — DAPO, SPPO, SPIN, Prime, FlowRL, SpecRL, CollabLLM, GSPO, CISPO, GMPO, GPG, Reinforce++, RLOO, SAPO, one-step off-policy, fully async policy, rollout correction, router replay
- **Agent/Tool Infrastructure** — `verl/tools/` with tool registry, MCP clients, sandbox fusion tools, search tools, geo3k tools; `verl/experimental/agent_loop/` for agentic RL training
- **SGLang Rollout** — Full SGLang rollout support (`verl/workers/rollout/sglang_rollout/`) alongside vLLM, including multi-turn interaction with tool use
- **Megatron Model Support** — Llama Megatron, Qwen2 Megatron, Qwen2.5-VL Megatron, Qwen3-MoE, GPT-OSS MoE, Moonlight, DeepSeek V3 671B
- **VLA (Vision-Language-Action)** — `recipe/vla/` for robotics RL with Isaac/Libero environments
- **Expanded SageMaker Recipes** — RLAIF + RLVR configs for Qwen3 family (0.6B–32B), GPT-OSS (20B/120B), DeepSeek R1 Distilled, Nemotron
- **Multi-turn Training** — `examples/sglang_multiturn/` with config-driven interaction and tool configs
- **Experimental Features** — Dynamic datasets, reward loops with colocated reward models, agent loops

**Usage:** Same as v0.5.0 (`hatch run setup_venv`, Slurm/Ray launch), but with additional algorithm-specific entry points in `recipe/` and `examples/`.

**Key new directories:**

| Path | Purpose |
|------|---------|
| `verl/trainer/config/actor/` | Actor worker typed configs (base, DP, Megatron) |
| `verl/trainer/config/algorithm/` | Algorithm-specific configs (rollout correction, etc.) |
| `verl/trainer/config/engine/` | Engine configs (FSDP, Megatron, VeOmni) |
| `verl/trainer/config/rollout/` | Rollout worker configs |
| `verl/trainer/ppo/` | PPO trainer with core algos, early stopping, multi-reward |
| `verl/trainer/sft/` | SFT metrics utilities |
| `verl/tools/` | Tool infrastructure (MCP, sandbox, search, schemas) |
| `verl/experimental/agent_loop/` | Agentic RL training loops |
| `verl/experimental/reward_loop/` | Reward loop with colocated models |
| `verl/models/` | Model implementations (Llama, Qwen2, Qwen3-MoE, mcore) |
| `verl/interactions/` | Multi-turn interaction system |
| `recipe/dapo/` | DAPO algorithm recipe |
| `recipe/prime/` | Prime algorithm recipe |
| `recipe/fully_async_policy/` | Fully async policy training |
| `recipe/one_step_off_policy/` | One-step off-policy training |
| `recipe/vla/` | Vision-Language-Action for robotics |
| `sagemaker-recipes/sagemaker/grpo/` | SageMaker GRPO configs (RLAIF + RLVR) |

---

## Constraints & Pitfalls

> Single source of truth for all hard constraints, gotchas, and their fixes.

| # | Constraint / Pitfall | Framework | Fix / Notes |
|---|---------------------|-----------|-------------|
| 1 | **CUDA 12.4 required** — must set `CUDA_HOME=/usr/local/cuda-12.4` and symlink `/usr/local/cuda` | LLMFT | Set env var before building or training |
| 2 | **torch==2.6.0 pinned** — fixated to avoid licensing review blockers | LLMFT | Do not upgrade without review |
| 3 | **transformers>=4.55.0** — required for GPT-OSS model support | LLMFT | Pinned in pyproject.toml |
| 4 | **Python >=3.11** required | LLMFT | — |
| 5 | **Python >=3.10** required | Verl | — |
| 6 | **Checkpoint dir must be on /fsx** — `trainer.default_local_dir` must be absolute FSx path; default `/tmp/ray/` will exhaust local disk | Verl | Set to e.g. `/fsx/ubuntu/verl-checkpoints/XXX` |
| 7 | **Batch size divisibility** — `train_batch_size / (world_size * micro_train_batch_size)` must be integer > 0 | LLMFT | BaseTrainer validates `accumulated_gradient > 0` |
| 9 | **Truncation default = "error"** — sequences exceeding `max_length` (default 1024) raise errors, not silently truncate | Verl | Increase `max_length` or change `truncation` strategy |
| 10 | **Bedrock flex tier required** — HyperPod → Bedrock requests must use `service_tier = flex`; priority/default tier is blocked | Verl | Set `service_tier = flex` |
| 11 | **Llama 4 gradient checkpointing** — fails without reentrant=false | LLMFT | Add `training_args.gradient_checkpointing_use_reentrant=false` |
| 12 | **flash-attn build failure** — pip tries to build from source | LLMFT | `uv pip install flash-attn==2.7.4.post1 --no-build-isolation` |
| 13 | **BaseConfig fields are frozen** — modifying non-mutable fields raises `FrozenInstanceError` | Verl | Add field to `_mutable_fields` set, or set at init |
| 14 | **save_freq & test_freq default to -1 (disabled)** — no periodic checkpointing/eval by default | Verl | Explicitly configure both |
| 15 | **max_length default = 1024** — too short for production; must increase | Verl | Set in config YAML |
| 16 | **VERL not supported on Slurm** — only EKS and SMTJ work | Recipes | Do not attempt Slurm launch for VERL recipes |
| 17 | **Elastic training K8s only** — requires `cluster.use_hyperpod_pytorch_job=true` and `cluster.queue_name` | Recipes | Raises `ValueError` on non-K8s |
| 18 | **Checkpointless training K8s only** — `SMTrainingHPCTRecipe` | Recipes | Raises `ValueError` on non-K8s |
| 19 | **Container image mismatch** — LLMFT and VERL use different containers; wrong image causes silent failures | Recipes | LLMFT: `llmft-v1.0.0`, VERL EKS: `verl-v1.0.0-eks`, VERL SMTJ: `verl-v1.0.0-smtj` |
| 20 | **GPU OOM on checkpoint save** — near-limit GPU memory + S3 sharded save = core dump | Recipes | More nodes, smaller batch, or FSx instead of S3 |
| 21 | **Experiment output conflict** — Hydra reuses directory names | LLMFT | Set `force_rerun=true` to overwrite |
| 22 | **wandb logging enabled by default** — needs network access | Verl | Remove `wandb` from logger list for offline environments |
| 23 | **NCCL_DEBUG=WARN** — set at module import in verl SFT trainer | Verl | — |
| 24 | **GPT-OSS may need custom attention** — `model_config.attn_implementation=kernels-community/vllm-flash-attn3` | LLMFT | — |
| 25 | **mlflow URI in runtime_env.yaml** — changing it carelessly breaks metric tracking | Verl | Understand downstream impact before modifying |
| 26 | **Don't mix train.py vs train_hp.py** — different config hierarchies (standard Hydra vs HyperPod schema) | LLMFT | — |
| 27 | **model_type is the routing key** — determines the entire execution pipeline in Recipes | Recipes | See Routing Logic table above |
| 28 | **Dynamic batch sizing only in engine config** — `use_dynamic_bsz=True` not available in legacy trainer config | Verl | Use `sft_trainer_engine.yaml` |

---

## Quick Reference: Do / Don't

> Actionable workflow tips (constraints/pitfalls are in the table above — not repeated here).

### Do ✅

- Use `uv pip install` for dependency installation (both frameworks use uv)
- Use `-Lora` suffix naming convention for LoRA model configs in LLMFT
- Use `datasets.train_data.limit=N` for quick iteration and testing
- Run CPU tests before committing: `pytest tests/ -k "_on_cpu"` (verl) or `pytest tests/` (LLMFT)
- Use existing recipe YAMLs as templates — copy and modify
- Use `++recipes.training_config.<key>=<value>` for deep Hydra overrides in launcher scripts
- Add Pydantic validation rules when adding new recipe fields
- Use `model_config.lora_checkpoint_file_path=/path` for LoRA checkpoint loading
- Use `training_args/strategy=hsdp_peft` for multi-node HSDP

### Don't ❌

- Don't use `conda` for AWSBedrockVerl — uses uv venvs (conda can't import tensordict properly in LLMFT)
- Don't edit launcher scripts in `launcher_scripts/` manually — they are auto-generated

---

## Important Files & Directories

### AWSLLMFineTuning (`recipes/awsLLMFT/src/AWSLLMFineTuning/`)

| Path | Purpose |
|------|---------|
| `src/train.py` | Main training entry point (standard) |
| `src/train_hp.py` | HyperPod-schema training entry point |
| `src/train_hp_elastic.py` | Elastic training entry point |
| `src/evaluate.py` | Evaluation entry point |
| `src/amzn_awsllm_fine_tuning/cli/train_runner.py` | Training orchestrator — routes to trainer by type |
| `src/amzn_awsllm_fine_tuning/cli/train_sft.py` | SFT training CLI |
| `src/amzn_awsllm_fine_tuning/cli/train_dpo.py` | DPO training CLI |
| `src/amzn_awsllm_fine_tuning/trainer/base_trainer.py` | BaseTrainer ABC (~600 lines) |
| `src/amzn_awsllm_fine_tuning/trainer/sft_trainer_refactored.py` | SFT trainer implementation |
| `src/amzn_awsllm_fine_tuning/trainer/dpo_multiturn_trainer_refactored.py` | DPO trainer |
| `src/amzn_awsllm_fine_tuning/data/` | Data loading, collators, preprocessors, mappers |
| `src/experiments/configs/` | Hydra config root (model_config/, training_config/, evals/) |
| `src/experiments/configs/model_config/` | Per-model YAML configs (Llama, Qwen, GPT-OSS, etc.) |
| `src/experiments/configs/training_config/training_args/` | Training hyperparameters per algorithm |
| `src/experiments/configs/training_config/datasets/` | 20+ dataset configs |
| `src/tools/` | Utilities (FSDP→HF conversion, LoRA combiner, etc.) |
| `pyproject.toml` | Package metadata, dependencies, hatch config |
| `requirements.txt` | Frozen dependency versions |

### AWSBedrockVerl v0.5.0 (`recipes/verl/src/AWSBedrockVerl/` — mainline)

| Path | Purpose |
|------|---------|
| `verl/trainer/sft_trainer.py` | SFT trainer (distributed, StatefulDataLoader) |
| `verl/trainer/config/sft_trainer.yaml` | SFT config (legacy, 8-GPU default) |
| `verl/trainer/config/sft_trainer_engine.yaml` | SFT config (advanced, dynamic batching) |
| `verl/base_config.py` | BaseConfig frozen dataclass with dict-like interface |
| `verl/workers/actor/` | Actor (policy model) workers: base, DP, Megatron |
| `verl/workers/critic/` | Critic (value function) workers: base, DP, Megatron |
| `verl/workers/rollout/` | Rollout workers: vLLM, SGLang, HF, naive, custom |
| `verl/workers/rollout/vllm_rollout/` | vLLM async inference server integration |
| `verl/workers/reward_model/` | Reward model workers |
| `verl/workers/reward_manager/` | Pluggable reward strategies (naive, batch, DAPO, prime) |
| `verl/workers/engine/` | Distributed engines: FSDP, Megatron, Mindspeed, VeOmni |
| `verl/workers/config/` | Worker configuration schemas |
| `verl/utils/dataset/` | Dataset utilities (rl_dataset, multiturn_sft_dataset) |
| `bedrock/scripts/slurm/` | Slurm launch scripts for Ray cluster provisioning |
| `bedrock/scripts/ray/` | Ray runtime environment configs |
| `bedrock/docker/Dockerfile` | Docker image for verl |
| `examples/` | Training examples (grpo, ppo, sft, etc.) |
| `recipe/` | Community recipes (DAPO, R1, etc.) |
| `sagemaker-recipes/` | SageMaker-specific recipe configs |
| `pyproject.toml` | Package metadata, hatch scripts (setup_venv, release, local_push) |
| `requirements.txt` | Core dependencies |
| `setup.py` | Setuptools config (supplements hatch) |

### AWSBedrockVerl v0.7.0 (`recipes/verl/src/AWSBedrockVerl/` — oss-0.7.0-dev)

> See the "AWSBedrockVerl v0.7.0" section above for new directories. In addition to all v0.5.0 paths, key additions include:

### HyperPod Recipes

| Path | Purpose |
|------|---------|
| `recipes/sagemaker-hyperpod-recipes/` | Public recipes repo — launcher, recipe_collection, tests |
| `recipes/private-sagemaker-hyperpod-recipes-staging/` | Staging recipes — extended launcher, validations, CI workflows |
| `recipes/private-sagemaker-hyperpod-recipes-staging/scripts/validations/` | Validation infrastructure for recipe testing |

---

## Document History

| Date | Change |
|------|--------|
| 2025-03-25 | Created — initial version with full framework documentation |
| 2026-03-30 | Deduplicated — merged Critical Constraints, Common Pitfalls, Framework Dependencies, and Core Patterns into consolidated sections. Removed Self-Update Instructions. |
| 2026-04-02 | PR #779 review fixes — updated Overview to list Nova, Eval (SupayModelLens), and Checkpointless NeMo as separate frameworks (zachgk). Symlinked `.kiro/steering/agent.md` to `.clinerules/agent.md`. |
