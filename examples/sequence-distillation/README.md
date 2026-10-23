# Sequence-level Knowledge Distillation (SeqKD) for a Tool-Use Agent

Distill a teacher's tool-use behavior into **Qwen/Qwen3.5-4B** with **sequence-level knowledge distillation**:
supervised fine-tuning on the teacher's trajectories, trained with the serverless `SFTTrainer` (SageMaker Python SDK
v3) and evaluated as an agent on [ShoppingBench](https://github.com/yjwjy/ShoppingBench).

## Overview

1. **Data** (`1-distill-shopping-agent.ipynb`, step 01): a teacher model on Amazon Bedrock (default GLM-5.3) plays
   ShoppingBench's agent loop on the training problems; its solved trajectories become one training example per
   tool-calling step, with exactly the prompts the benchmark sends at evaluation time. The train/validation split is
   by query.
2. **Training** (step 02, same notebook): serverless `SFTTrainer`, LoRA (rank 64); the job registers a ready-to-deploy
   Model Package.
3. **Evaluation** — two complementary notebooks, run in order:
   - **Tool-call accuracy gate** (`2-tool-call-accuracy-gate.ipynb`): a fast, cheap gate. A SageMaker **Custom Scorer**
     managed evaluation job measures **per-step tool-call accuracy** (did the student pick the right tools, in order?).
     Run this first — if the student doesn't beat the base here, the heavy run isn't worth it.
   - **Full agentic eval** (`3-evaluate-shopping-agent.ipynb`): base and student on a shared SageMaker real-time
     endpoint, run through ShoppingBench's agent loop on **held-out** test problems (never used in training), scored
     with the benchmark's rules (ASR), plus optional IFEval and optional reference frontier models (served from frozen
     cached rollouts, not re-invoked live).

Reference results (LoRA r64, held-out problems, one sample run; expect a few points of run-to-run variation).

**Bedrock teacher (default; thinking on, reasoning kept in the agent history)**, ASR (see `asr_results_chart.png`):

| Model | product (103) | shop (105) | voucher (116) | mean |
|---|---|---|---|---|
| Base Qwen3.5-4B | 0.46 | 0.31 | 0.12 | 0.30 |
| **Student** (GLM-5.3 teacher, 2.6k solved-trajectory steps, 3 epochs) | **0.56** | **0.64** | **0.42** | **0.54** |
| Teacher GLM-5.3 (cached reference) | 0.55 | 0.62 | 0.36 | 0.51 |
| Qwen3-32B (cached reference, thinking on) | 0.44 | 0.16 | 0.09 | 0.23 |
| Frontier model on Amazon Bedrock (cached reference, adaptive thinking) | 0.58 | 0.67 | 0.39 | 0.55 |

IFEval prompt-strict: base 0.82, student 0.74. Tool-call accuracy gate: 48.9% → 63.5% tool selection.

> **Train/held-out split.** The ShoppingBench test problems are split once into 425 training problems (the
> teacher's prompts: 147 product / 145 shop / 133 voucher) and 324 held-out problems (103 / 105 / 116), which are the
> only ones evaluated. The split ships as line indices into ShoppingBench's test files (`split/<task>.json`, with a
> hash of each file) and is loaded by `scripts/split.py`. These numbers are on the held-out subset, not the full
> benchmark.

## Deployment

Base and student share **one** real-time endpoint using **SageMaker inference components (ICs)**: a base IC hosts
Qwen/Qwen3.5-4B with LoRA enabled, and the student is an **adapter IC** attached to it — so the base model loads once
and adapters share it. At invoke time, `InferenceComponentName` selects base vs. a specific adapter. Deployment uses
the **SageMaker Core** resource API (`Model` / `EndpointConfig` / `Endpoint` / `InferenceComponent`); capacity config
(an ordered instance-type preference list, or a Flexible Training Plan reservation) lives in the notebook, and the
`scripts/kd_serve.py` helpers do the deploy/attach/teardown.

The eval harness talks to the endpoint through a local OpenAI-compatible **relay** (`scripts/sm_openai_relay.py`),
which forwards each request via `sagemaker-runtime InvokeEndpoint` with the right `InferenceComponentName` (SigV4 per
call, so multi-hour runs never fail on token expiry). The endpoint's `/openai/v1` passthrough route is **not** used —
it can't route to a specific inference component.

## Files

### Notebooks

| File | Purpose |
|---|---|
| `1-distill-shopping-agent.ipynb` | Data prep, serverless SFT, Model Package registration |
| `2-tool-call-accuracy-gate.ipynb` | Fast gate: per-step tool-call accuracy via a SageMaker Custom Scorer eval job |
| `3-evaluate-shopping-agent.ipynb` | Full ShoppingBench ASR (and IFEval) for base, student, and optional cached reference frontier models |

### Scripts

| File | Purpose |
|---|---|
| `scripts/setup_shopbench.sh` | Install the ShoppingBench harness, catalog and search index (`--code-only` for data prep) |
| `scripts/split.py` | Load the fixed train/held-out split of the ShoppingBench test problems (indices in `split/`) |
| `scripts/encode_rollouts.py` | Turn solved trajectories into `{prompt, completion}` training examples, one per step |
| `scripts/shopbench_harness.py` | Search server, agent rollouts, scoring and health checks |
| `scripts/score_rollouts.py` | Score trajectories with ShoppingBench's rules |
| `scripts/eval_utils.py` | Scoring/table/plot plumbing for the full agentic eval |
| `scripts/kd_serve.py` | Deploy / attach / teardown of base + adapter inference components (SageMaker Core); Flexible Training Plan support |
| `scripts/sm_openai_relay.py` | Local OpenAI-compatible relay → `InvokeEndpoint` (IC routing) or a Bedrock model (Converse API) |
| `scripts/litmus_utils.py` | Tool-call accuracy gate: dataset assembly, quota-aware job scheduler, results reading |
| `scripts/quota_checker.py` | Endpoint / eval service-quota lookups (cached locally) |
| `scripts/run_ifeval.py` | IFEval with lm-evaluation-harness, with a reasoning-leak check |
| `scripts/sm_utils.py` | Small helper to stream a subprocess line-by-line in a notebook |

### Data

| Path | Purpose |
|---|---|
| `split/` | The fixed train/held-out split: line indices into ShoppingBench's test files, with a hash of each file |
| `teacher_trajectories/glm-5.3/` | The published GLM-5.3 teacher trajectories behind the reported student (`USE_PUBLISHED_TEACHER_TRAJECTORIES` in notebook 1) |
| `reference_evaluations/` | Frozen rollouts and configs of the cached comparison models (`CACHED_REF_TAGS` in notebook 3) |

### Reward function (tool-call accuracy gate)

| File | Purpose |
|---|---|
| `reward_functions/litmus_reward_function.py` | The Custom Scorer reward function shipped to the managed eval job (graded tool-call match: `toolset_match` gate, `format_match`, `kind_match`) |
| `reward_functions/test_litmus_reward_function.py` | Unit test, run before shipping the reward function to the evaluator |

## Notes on the model

Qwen3.5-4B is a **linear-attention / conv1d hybrid**, and that is the main reason to train it on the
managed serverless recipe rather than a training script of your own. The linear-attn path needs
`flash-linear-attention`, which requires `triton>=3.7.1` — and that collides with the `triton` pin a
stock PyTorch image carries, so a plain `pip install` either dead-ends on `ResolutionImpossible` or
bumps `torch` to a version that breaks `transformers`. The hybrid also wants `sdpa` attention, because
`flash_attention_2` hits CUDA illegal-memory-access on SageMaker's managed driver. The serverless recipe
handles both; you never see them.

Inference is simpler: the LMI image `djl-inference:0.36.0-lmi25.0.0-cu130` bundles **vLLM 0.20.1**,
which has native kernels for this hybrid, so **no custom inference image is needed**. Note that
`ModelServer.VLLM` (HF vLLM 0.29.0) is *not* a working alternative for this architecture — it
crash-loops on `Failed to promote local KV cache specs`.

## Operational notes

- **Edit the notebooks with Jupyter tooling**, not a text editor — direct `.ipynb` writes can corrupt them.
- **Search server is the throughput bottleneck.** It holds a ~3 GB Lucene index. Under high rollout concurrency it
  can saturate (connection limit, request timeouts, stalled rollouts) — restart the search cell and lower
  `ROLLOUT_THREADS` if you see repeated `find_product` read timeouts. Rollouts are resume-safe (finished problems are
  skipped on re-run).
- **Detached subprocesses survive kernel shutdown.** The search server and rollout workers are started in new
  sessions, so closing a kernel does not reclaim their memory — stop them explicitly (`harness.stop_all()` / the
  cleanup cell) or kill the orphans.
- **Validation loss** is not logged to MLflow by design for the single-dataset + `train_val_split_ratio` serverless
  configuration; assess overtraining from the training-loss curve.

## License

See the repository root `LICENSE`.
