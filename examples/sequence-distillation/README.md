# Sequence-level Knowledge Distillation (SeqKD) for a Tool-Use Agent

Distill a teacher's tool-use behavior into **Qwen/Qwen3.5-4B** with **sequence-level knowledge distillation**:
supervised fine-tuning on the teacher's trajectories, trained with the serverless `SFTTrainer` (SageMaker Python SDK
v3) and evaluated as an agent on [ShoppingBench](https://github.com/yjwjy/ShoppingBench).

> For the full design record, decisions, and changelog, see [`SPEC.md`](SPEC.md).

## Overview

1. **Data** (`1-distill-shopping-agent.ipynb`, step 01): teacher trajectories from
   `oro-ai/sn15-shoppingbench-sft-15k`, one training example per tool-calling step, built with ShoppingBench's own
   prompt functions so training text matches what the benchmark sends at evaluation time. The train/validation split
   is by query.
2. **Training** (step 02, same notebook): serverless `SFTTrainer`, LoRA (rank 64); the job registers a ready-to-deploy
   Model Package.
3. **Evaluation** — two complementary notebooks, run in order:
   - **Tool-call accuracy gate** (`2-tool-call-accuracy-gate.ipynb`): a fast, cheap gate. A SageMaker **Custom Scorer**
     managed evaluation job measures **per-step tool-call accuracy** (did the student pick the right tools, in order?).
     Run this first — if the student doesn't beat the base here, the heavy run isn't worth it.
   - **Full agentic eval** (`3-evaluate-shopping-agent.ipynb`): base and student on a shared SageMaker real-time
     endpoint, run through ShoppingBench's agent loop on **held-out** test problems (queries absent from the training
     corpus), scored with the benchmark's rules (ASR), plus optional IFEval and optional reference frontier models
     (served from frozen cached rollouts, not re-invoked live).

Reference results (LoRA r64, held-out problems, thinking on), base → student ASR:

| product (103) | shop (105) | voucher (116) | mean |
|---|---|---|---|
| 0.155 → 0.505 | 0.057 → 0.343 | 0.043 → 0.190 | 0.085 → 0.346 |

> **Why held-out problems.** The teacher corpus is built on ShoppingBench test queries (147 of the 250 product test
> queries appear in it verbatim), so a score on the full test set partly measures recall. The evaluation notebooks
> score only test problems whose query never appears in the corpus (`scripts/make_clean_split.py`).

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
| `scripts/make_clean_split.py` | Write the held-out (decontaminated) test problems |
| `scripts/shopbench_harness.py` | Search server, agent rollouts, scoring and health checks |
| `scripts/score_rollouts.py` | Score trajectories with ShoppingBench's rules |
| `scripts/eval_utils.py` | Scoring/table/plot plumbing for the full agentic eval |
| `scripts/kd_serve.py` | Deploy / attach / teardown of base + adapter inference components (SageMaker Core); Flexible Training Plan support |
| `scripts/sm_openai_relay.py` | Local OpenAI-compatible relay → `InvokeEndpoint` (IC routing) or an external provider API |
| `scripts/litmus_utils.py` | Tool-call accuracy gate: dataset assembly, quota-aware job scheduler, results reading |
| `scripts/quota_checker.py` | Endpoint / eval service-quota lookups (cached locally) |
| `scripts/run_ifeval.py` | IFEval with lm-evaluation-harness, with a reasoning-leak check |
| `scripts/inspect_teacher_dataset.py` | Inspect the teacher dataset |
| `scripts/sm_utils.py` | Small helper to stream a subprocess line-by-line in a notebook |

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
