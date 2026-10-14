# SPEC — Sequence-Level Knowledge Distillation for a Tool-Use Agent

Specification, design record, and changelog for the `distillation/` recipe. This document captures
**what** the project is, **why** it is built the way it is (the decisions and the alternatives that were
rejected), and **how** it evolved.

Audience: maintainers of this recipe and reviewers. For end-user instructions, see `README.md`.

---

## 1. Goal

Distill a teacher's **tool-use behaviour** into a small student model (`Qwen/Qwen3.5-4B`) using
**sequence-level knowledge distillation (SeqKD)** — i.e. supervised fine-tuning on the teacher's
decoded trajectories — and prove the student improves on a held-out agent benchmark.

- **Training technique:** LoRA SFT on teacher trajectories, run on **SageMaker serverless Model
  Customization** (`SFTTrainer`, SageMaker Python SDK v3). This is SeqKD: one training example per
  tool-calling step, text matched to what the benchmark sends at eval time.
- **Benchmark:** [ShoppingBench](https://github.com/yjwjy/ShoppingBench), scored by its own rules
  (Action Success Rate, **ASR**) across three task intents — `product`, `shop`, `voucher` — plus
  optional **IFEval**.
- **Headline result** (LoRA r64, held-out problems, thinking on), base → student ASR:

  | product (103) | shop (105) | voucher (116) | mean |
  |---|---|---|---|
  | 0.155 → 0.505 | 0.057 → 0.343 | 0.043 → 0.190 | 0.085 → 0.346 |

This is a **customer-facing example notebook recipe**, so clarity of the notebooks and markdown is a
first-class requirement, not an afterthought.

---

## 2. Scope and non-goals

**In scope**
- Teacher-trajectory data prep from `oro-ai/sn15-shoppingbench-sft-15k`.
- Serverless LoRA SFT and Model Package registration.
- Two evaluation paths: a **heavy agentic eval** (full ShoppingBench ASR) and a **lightweight litmus
  test** (per-step tool-call accuracy via a managed Custom Scorer job) used as a gate.
- Single-endpoint deployment of base + student via **inference components** (base model + LoRA adapter).

**Non-goals / rejected directions**
- **Logit-level / white-box KD** — out of scope; this recipe is sequence-level (black-box) distillation.
- **Full fine-tuning (FFT)** — only LoRA is used.
- **ModelBuilder-based deployment** — explicitly removed in favour of the SageMaker Core resource API
  (see §6, decision D-DEPLOY).
- **A homegrown `EndpointDeployer`/`sm_endpoints` abstraction** — removed; the notebook owns capacity
  config and calls the thin `kd_serve.py` helpers directly (D-DEPLOY).
- **A DJL-vs-vLLM container toggle** — removed; standardized on **DJL-LMI (vLLM backend)** (D-SERVE).

---

## 3. The core conceptual decision: is this distillation at all?

A central discussion shaped the whole project. The initial teacher dataset gave only the **final tool
calls** (labels). The question raised was: *if we only train on correct tool-call labels, is this
distillation or just supervised learning on a labeled set?*

Resolution (decision **D-REASON**):
- Training on bare tool-call labels alone is closer to plain SFT on labels than to distilling the
  teacher's *decision process*.
- The valuable thing to distill is the teacher's **reasoning about which tools to call, why, and with
  what parameters** — not just the final call.
- Therefore the project pursued **reasoning + correct tool calls** (teacher trajectories *with* thinking),
  and set out to **prove** that reasoning + correct tool calls beats correct-tool-call labels alone.
- To generate reasoning traces, a **teacher model** was stood up (Qwen3.5-27B on a SageMaker endpoint)
  rather than relying on a labels-only corpus.

This is why the recipe is framed as **SeqKD of tool-use behaviour**, and why the eval must measure both
tool **selection** and **format**, not just exact-string tool matches.

---

## 4. Data

- **Source:** `oro-ai/sn15-shoppingbench-sft-15k` teacher trajectories.
- **Granularity:** one training example **per tool-calling step** (not per conversation).
- **Prompt construction:** built with **ShoppingBench's own prompt functions**, so training text matches
  exactly what the benchmark sends at evaluation time. (Decision **D-PROMPT**: do not hand-roll prompts;
  reuse the harness's functions to avoid train/serve skew.)
- **Split:** train/validation split **by query** (not by row), to avoid leaking a query across splits.
- **Volume:** ~**15,269** trajectories used for training. This was explicitly questioned as possible
  overtraining; see decision **D-OVERTRAIN**.
- **Local files:** `data/seqkd_train.jsonl` (~435 MB), `data/seqkd_val.jsonl` (~16 MB) — written by the
  data-prep notebook into a `data/` folder, which is git-ignored in its entirety.

### Decontamination (held-out evaluation) — decision D-HOLDOUT

The teacher corpus is built on ShoppingBench **test** queries (≈147 of the 250 `product` test queries
appear verbatim in the corpus). Scoring on the full test set would partly measure **recall of training
data**, inflating the student's apparent gain.

**Decision:** evaluate only on test problems whose query **never appears** in the training corpus
(`scripts/make_clean_split.py` writes the held-out set, `synthesize_*_test_clean.jsonl`). All headline
numbers are on held-out problems. This is the single most important integrity decision in the recipe.

---

## 5. Training

- **Trainer:** serverless `SFTTrainer` (SageMaker Python SDK **v3**). The whole import surface was
  migrated to **v3 constructs** (decision **D-SDKV3**); the SDK is **pinned** in the notebook's install
  cell (decision **D-PIN**).
- **Adapter:** **LoRA, rank 64** (`max_lora_rank`/serving must be ≥ the adapter rank — the serving layer
  sets `OPTION_MAX_LORA_RANK=64`).
- **Experiment tracking:** MLflow (experiment `seqkd-shopping-exp`; a run such as
  `huggingface-vlm-qwen3-5-4b-sft-<timestamp>`).
- **Current student:** the deployed/evaluated student is **Model Package version 2**
  (`seqkd-qwen35-4b-shopping/2`), a retrain that superseded the original v1 student. The eval notebook's
  config cell pins `MODEL_PKG_ARN` to `/2` directly (rather than relying on `%store -r`, which may still
  hold the v1 ARN from the training notebook's earlier run).

### SFT vs "Distillation" job type — decision D-JOBTYPE
The managed catalog exposed both an **SFT** recipe and a **Distillation** option. This recipe uses the
**SFT** trainer: SeqKD *is* supervised fine-tuning on the teacher's decoded trajectories, so the SFT
recipe is the correct managed path. (The managed "Distillation" option targets a different workflow.)

### Validation-loss tracking — decision D-VALLOSS
Observed: **no validation loss** was logged to MLflow. Investigated against the documentation and
confirmed this is **by design** for serverless customization in this configuration — when a single
dataset is supplied with `train_val_split_ratio`, val loss is not emitted the way it would be with two
explicitly-supplied datasets. The notebook does **not** add machinery to force it; instead the loss
**curve cell** reads from MLflow when available, and overtraining is assessed from the **training** curve.

### Overtraining assessment — decision D-OVERTRAIN
The training loss reached roughly its final level by ~step 500 but training continued past ~step 1000,
indicating the model was being pushed past the point of marginal gain for a distillation run. This was
flagged in-notebook as a learning point (appropriate-amount-of-data / early-stopping discussion) rather
than silently "fixed".

### Why the serverless recipe (not a custom training script) — decision D-RECIPE
`Qwen3.5-4B` is a **linear-attention / conv1d hybrid**. The linear-attention path needs
`flash-linear-attention`, which requires `triton>=3.7.1`; that collides with the `triton` pin in a stock
PyTorch image (either `ResolutionImpossible`, or a `torch` bump that breaks `transformers`). The hybrid
also wants `sdpa` attention, because `flash_attention_2` triggers CUDA illegal-memory-access on
SageMaker's managed driver. The managed serverless recipe resolves both transparently — a decisive
reason to train on the recipe rather than a bespoke script.

---

## 6. Deployment architecture (serving)

### Single endpoint, inference components — decision D-IC
Base and student share **one** real-time endpoint using **SageMaker inference components (ICs)**:
- one **base IC** hosts `Qwen/Qwen3.5-4B` with LoRA enabled;
- each **student is an adapter IC** that references the base IC by name and points at the adapter
  artifacts in S3 (a loose `checkpoints/hf/` prefix works; a `.tar.gz` also works) — so the base model
  loads **once** and adapters share it;
- at invoke time, `InferenceComponentName` selects base vs a specific adapter.

This replaced an earlier two-endpoint design (base endpoint + separate student endpoint).

### Serving container — decision D-SERVE
Standardized on the **DJL-LMI container (vLLM backend)**, image
`djl-inference:0.36.0-lmi25.0.0-cu130` (bundles **vLLM 0.20.1**), which has native kernels for this
hybrid architecture — **no custom inference image needed**.
- `ModelServer.VLLM` (HF vLLM 0.29.0) is **not** a working alternative — it crash-loops on
  `Failed to promote local KV cache specs`.
- A DJL-vs-vLLM configuration toggle was **removed** (dead configuration surface).
- LMI/vLLM also maps Qwen3's per-projection q/k/v PEFT modules correctly.

### Why `InvokeEndpoint`, not the OpenAI passthrough — decision D-INVOKE
The endpoint's `/openai/v1` passthrough route does **not** support inference-component routing (it 400s
with "Inference Component Name header is required" and no way to supply one on that route). IC routing is
only honoured via `sagemaker-runtime InvokeEndpoint`'s `InferenceComponentName`. This is why the eval
harness talks to the endpoints through a **local OpenAI relay** (`sm_openai_relay.py`) that forwards each
request via `InvokeEndpoint` (SigV4 per call, so multi-hour runs never fail on token expiry).

### SDK layer — decision D-DEPLOY
Deployment uses the **SageMaker Core resource API** (`Model` / `EndpointConfig` / `Endpoint` /
`InferenceComponent`), **not** `ModelBuilder` and **not** boto3. The notebook **owns capacity config**;
`kd_serve.py` bakes in **no pool defaults**. A homegrown `EndpointDeployer`/`sm_endpoints` abstraction and
redundant helper methods (`lmi_image`, `container_env`, a `DJL/VLLM` switch) were deleted. Principle:
*build inline as needed, refactor into utils once stable* — hence the thin `kd_serve.py` module.

### Capacity: instance pools and Flexible Training Plans (FTP) — decision D-FTP
`deploy_base_endpoint` accepts an **ordered `instance_types` preference list** → SageMaker Core
`InstancePool`s (priority = list order), so capacity shortages fall back to the next type.
For **reserved capacity via a Flexible Training Plan (FTP)**:
- a reservation covers a **single instance type**, so `instance_types` must contain **exactly one** entry,
  which must match the plan's type;
- it is wired via `ProductionVariantCapacityReservationConfig(ml_reservation_arn=<plan arn>,
  capacity_reservation_preference="capacity-reservations-only")` on the production variant;
- the single-type rule is enforced **twice**: an **early `assert` in the notebook config cell** (fail
  before any deploy work) **and** a backstop `ValueError` in `kd_serve.deploy_base_endpoint`.

**Design note (D-FTP refinement):** an earlier version introduced a second variable
(`TRAINING_PLAN_INSTANCE_TYPE`) alongside `INSTANCE_TYPES`. This was rejected as redundant (two sources of
truth that can disagree). Final design: **one** source of truth, `INSTANCE_TYPES`; setting
`TRAINING_PLAN_ARN` simply requires `INSTANCE_TYPES` to be length 1.

### Adapter resolution & idempotency
- `resolve_adapter_uri` finds the LoRA adapter inside the training job's Model Package
  (`checkpoints/hf/` first, else the latest `global_step_*/lora_adapter/`), returning the **loose prefix**
  (DJL-LMI loads files directly; no `.tar.gz` repackaging needed).
- `attach_adapter` is **idempotent**: if an adapter IC exists and its artifact **matches**, it is reused
  (safe cell re-runs); if the artifact **differs** (new training run), it is deleted and recreated so the
  adapter is never silently stale. `replace=True` forces recreate.
- `teardown_endpoint` deletes in dependency order: **adapter ICs → base IC → endpoint → config**
  (ICs don't cascade-delete; adapters reference their base). Failed deploys **self-clean** the partial
  stack and re-raise.

---

## 7. Evaluation

Two complementary evaluations, by design (decision **D-TWOTIER**): a cheap **litmus gate** and the
expensive **full agentic eval**. *If the litmus test fails, skip the heavy test; if it passes, the heavy
run is worthwhile.*

### 7a. Full agentic eval — `3-evaluate-shopping-agent.ipynb`
Runs base, student, and optional **reference frontier models** through ShoppingBench's agent
loop on **held-out** problems, scored by the benchmark's rules (ASR), plus optional IFEval.

**Reference/frontier comparison rows — decision D-CACHEDREF.** The comparison rows (`CACHED_REF_TAGS`
in the config cell — currently `reference_qwen3-32b`, a same-family dense model, and `frontier_medium`,
the frontier ceiling) are served entirely from a **frozen local cache** (`out/reference_evaluations/`:
one `rollout_<tag>_<task>.jsonl` + `cfg_<tag>_<task>.json` per tag/task) rather than invoked live each
run. A restore step copies the cached files into the run's working `OUT` directory before rollouts
start; `run_rollouts()` then sees them as already complete and skips them. This means:
- no live call (and no model ID) is needed for these rows at run time — only the tag string;
- their identities and serving endpoints are decoupled from the live recipe, so results tables don't
  reveal which frontier models were used as comparison points;
- adding/dropping a comparison row is just editing `CACHED_REF_TAGS` and ensuring its cache files exist.

This replaced an earlier design where each comparison tag was a live call behind a configured model ID
(`FRONTIER_SMALL_MODEL`/`FRONTIER_MEDIUM_MODEL`/`REFERENCE_QWEN3_32B_MODEL` string variables), invoked
fresh on every run.

- **Harness:** `scripts/shopbench_harness.py` + the vendored `ShoppingBench/` (search server over a
  ~3 GB Lucene index of ~2.7 M products; per-rollout subprocesses; scoring subprocess that also loads a
  `Qwen3-Embedding-0.6B` model for title similarity).
- **Relay:** `scripts/sm_openai_relay.py` — one local OpenAI endpoint per model; forwards via
  `InvokeEndpoint` with the right `InferenceComponentName`; can also target an external provider API,
  and can force fields (e.g. `enable_thinking=False` for IFEval). (Reference frontier models in this
  recipe are served from cached rollouts rather than through the live relay.)
- **Scoring/plumbing:** `scripts/eval_utils.py` (`score_all` isolates per-cell failures; `plot_results`
  draws grouped bar charts).

**Decision D-SMEVAL (rejected for the heavy eval):** using SageMaker's managed model-evaluation capability
for the *full* agentic eval was rejected — ShoppingBench scoring is ~10% of the work and tool returns /
agent loop can't be expressed as a simple managed scorer. (It **is** used for the litmus test — §7b.)

**Decision D-TOOLRETURN:** since ground truth is the correct tool call, the litmus gate measures the
**tool call** and does not depend on what the tool *returns*; the heavy eval still runs the real tool loop.

**Decision D-ASR-AGG:** aggregate ASR is reported as a **simple mean over the three task intents** (plus
per-task breakdown). A weighted average was considered and discussed but **simple mean** was chosen for
both the heavy and litmus reports, so minority intents aren't swamped by the majority intent.

**Operational hardening in the eval notebook:**
- **Preflight quota check** before deployment (reads only the endpoint quotas it needs and persists them
  to a git-ignored local JSON cache, `.quota_codes_endpoint_usage_*.json`); replaced an earlier version
  that got **throttled** by hitting all quotas.
- Context-length guard: drop eval prompts `>= 12000` tokens (they exceed the serving window and fail the
  job); a subset size (`N_EXAMPLES`, e.g. 600) is used.
- MLflow link: resolves the app ARN from the **notebook's region-bound session** (a bare `sagemaker.Session()`
  could resolve the **wrong** app) and renders a compact presigned link.
- Search server **self-heals** / is reused if a healthy one is already on the port (customers may not run
  cleanup immediately).

### 7b. Tool-call accuracy gate — `2-tool-call-accuracy-gate.ipynb`
A fast, per-step **tool-call accuracy** gate using a **SageMaker Custom Scorer** managed evaluation job.

- **Reward function:** `reward_functions/litmus_reward_function.py` — shipped to the managed job via
  `Evaluator.create(source=...)` (it is **not** imported by the notebook). Three graded levels:
  - `kind_match` — right *kind* of turn (tool_call vs response)? (reported Metric)
  - `toolset_match` — right *tool names, in order*, params ignored, parsed leniently → **the gate**
    (`aggregate_reward_score`): measures tool **selection**, not formatting;
  - `format_match` — right tools **and** the exact harness `<tool_call>[{...}]</tool_call>` array form?
    (reported Metric; what the base model usually misses).
  - Contract: `lambda_handler` and `reward_function` names are required by the runtime.
  - A **unit test** (`test_litmus_reward_function.py`) runs before shipping to the evaluator.
- **Plumbing:** `scripts/litmus_utils.py` — `build_eval_datasets` (recovers each row's task by matching
  `<user>…</user>` against ShoppingBench's per-task test files; length-filters; writes one JSONL per task
  to `eval_inputs/` and uploads to S3), `run_eval_jobs` (**quota-aware scheduler**), `read_task_metrics`,
  `print_results_tables`, `mlflow_experiment_link`.
- **Quota-aware scheduling — decision D-EVALQUOTA:** the concurrent-eval-job quota is
  `L-619D690E` ("Maximum number of concurrent model evaluation serverless jobs per Region"). The scheduler
  reads the applied quota (exact code first, then a name scan, with backoff on throttling) and keeps
  `quota // jobs_per_task` tasks in flight (2 jobs per task: student + base). **If the quota lookup fails
  or is throttled, it falls back to 1 task in flight** (i.e. 3 sequential groups of 2).
- Datasets and results are written under git-ignored `eval_inputs/` (the datasets the jobs consume).

**Decision D-STUDENT-THRESH:** the student's pass threshold is set to **the base model's value** — the
student must beat the base it was distilled from.

---

## 8. Repository layout

| Path | Purpose |
|---|---|
| `1-distill-shopping-agent.ipynb` | Data prep, serverless SFT, Model Package registration |
| `3-evaluate-shopping-agent.ipynb` | Full ShoppingBench ASR (+ IFEval) for base, student, optional cached reference frontier models |
| `2-tool-call-accuracy-gate.ipynb` | Per-step tool-call accuracy gate via managed Custom Scorer |
| `scripts/kd_serve.py` | Deploy/attach/teardown of base + adapter inference components (SageMaker Core); FTP support |
| `scripts/sm_openai_relay.py` | Local OpenAI-compatible relay → `InvokeEndpoint` (IC routing) or an external provider API |
| `scripts/shopbench_harness.py` | Search server, agent rollouts, scoring, health checks |
| `scripts/eval_utils.py` | `score_all`, `plot_results` for the heavy eval |
| `scripts/litmus_utils.py` | Litmus datasets, quota-aware scheduler, results reading |
| `scripts/make_clean_split.py` | Writes the decontaminated held-out test problems |
| `scripts/score_rollouts.py` | Score trajectories with ShoppingBench's rules |
| `scripts/run_ifeval.py` | IFEval via lm-evaluation-harness, with a reasoning-leak check |
| `scripts/inspect_teacher_dataset.py` | Inspect the teacher dataset |
| `scripts/quota_checker.py` | Endpoint/eval quota lookups |
| `scripts/sm_utils.py` | `run()` — stream a subprocess line-by-line in a notebook |
| `scripts/setup_shopbench.sh` | Install the ShoppingBench harness, catalog, search index (`--code-only` for data prep) |
| `reward_functions/litmus_reward_function.py` | The Custom Scorer reward function (shipped to the managed job) |
| `reward_functions/test_litmus_reward_function.py` | Unit test for the reward function |
| `eval_inputs/` | Litmus eval datasets (git-ignored) |
| `data/` | Training/validation JSONL written by the data-prep notebook (git-ignored) |
| `shopbench-eval/ShoppingBench/` | Vendored benchmark harness + data/index |
| `README.md` | End-user instructions | 
| `SPEC.md` | This document |

### Conventions
- **Notebooks are edited only via the Jupyter MCP tools** (`insert_cell`/`edit_cell`/`delete_cell`/
  `read_notebook`) — direct `.ipynb` file writes are blocked to prevent corruption.
- Capacity config lives in the **notebook**, never baked into `scripts/`.
- Code the notebook does **not** execute (the reward function, its unit test) lives in top-level folders
  (`reward_functions/`), not in `scripts/`.
- `sys.path.insert(0, str(Path("scripts").resolve()))` in each notebook's config cell makes the script
  modules importable; `.resolve()` is kept for consistency across notebooks and robustness to `os.chdir`.
- Local imports from the vendored benchmark are namespaced (`ShoppingBench.src...`) rather than bare
  (`agent.`) for clarity and to reduce ambiguity.
- `.gitignore` excludes: `ShoppingBench/`, `data/`, `seqkd_*.jsonl`, `llmaj_*.jsonl`,
  `synthesize_*_test_clean.jsonl`, `scripts/__pycache__/`, `reward_functions/__pycache__/`,
  `eval_inputs/`, `.quota_codes_endpoint_usage_*.json`.

---

## 9. Known constraints & operational notes

- **Hybrid architecture pins** (training): see §5 D-RECIPE. The managed recipe hides the
  `flash-linear-attention`/`triton`/`sdpa` constraints.
- **Serving KV-cache crash:** `ModelServer.VLLM` (HF vLLM 0.29.0) crash-loops on
  `Failed to promote local KV cache specs`; use DJL-LMI (D-SERVE).
- **IC routing requires `InvokeEndpoint`:** the OpenAI passthrough route 400s on IC endpoints (D-INVOKE).
- **Search server is the memory & throughput bottleneck:** it holds a ~3 GB Lucene index and grows under
  query load. Under high concurrency (e.g. 9 jobs × `ROLLOUT_THREADS=4` ≈ 36 agents) the waitress server
  hits its connection limit (queue depth ~90) and `find_product` calls time out (60 s read timeout, up to
  ~10 retries) — rollouts then stall with flat completion counts. **Remedy:** restart the search server
  (re-run the search cell) and **lower `ROLLOUT_THREADS`** (e.g. 4 → 2). Rollouts are **resume-safe**
  (`run_rollout.py` skips already-finished problems).
- **Detached subprocesses survive kernel shutdown:** the search server/rollouts are started with
  `start_new_session=True`, so closing a kernel does **not** reclaim their memory — they must be stopped
  explicitly (`harness.stop_all()` / cleanup cell) or killed as orphans (reparented to PID 1).
- **Jupyter server heap growth:** a long-lived (multi-day) `jupyter-lab` **server** process can accumulate
  many GB of private heap independent of the notebook (the on-disk notebook and its outputs can be tiny).
  Reclaimed only by **restarting the Jupyter server** (not kernels).
- **Eval context limit:** drop prompts `>= 12000` tokens; they exceed the serving window and fail the job.
- **MLflow val loss:** not logged by design for this serverless config (§5 D-VALLOSS).

---

## 10. Decision index

| ID | Decision | Rationale |
|---|---|---|
| D-REASON | Distill reasoning + tool calls, not labels alone | Labels-only is plain SFT, not distilling the teacher's decision process |
| D-PROMPT | Build training prompts with ShoppingBench's own functions | Eliminate train/serve prompt skew |
| D-HOLDOUT | Evaluate only on queries absent from the training corpus | Teacher corpus overlaps the test set; full-set scores measure recall |
| D-SDKV3 | Migrate all imports to SageMaker SDK v3 | Current SDK; serverless `SFTTrainer` lives there |
| D-PIN | Pin all pip installs in the notebooks | Reproducibility; avoid "latest" drift |
| D-JOBTYPE | Use the SFT recipe (not the "Distillation" option) | SeqKD = SFT on teacher trajectories |
| D-VALLOSS | Don't force val-loss logging | Not emitted by design for single-dataset + split-ratio config |
| D-OVERTRAIN | Flag likely overtraining past ~step 500–1000 | Loss plateaued early; teaching point rather than silent fix |
| D-RECIPE | Train on managed serverless recipe, not custom script | Hybrid attn needs flash-linear-attention/triton/sdpa; recipe handles it |
| D-IC | One endpoint, base IC + adapter ICs | Base loads once; adapters share it; `InferenceComponentName` selects |
| D-SERVE | Standardize on DJL-LMI (vLLM 0.20.1); drop DJL/vLLM toggle | Native kernels for the hybrid; HF vLLM 0.29.0 crash-loops |
| D-INVOKE | Route via `InvokeEndpoint`, use a local OpenAI relay | OpenAI passthrough can't address inference components |
| D-DEPLOY | SageMaker Core resource API; delete ModelBuilder/EndpointDeployer | Notebook owns capacity; thin helpers in `kd_serve.py` |
| D-FTP | Support Flexible Training Plans; single `INSTANCE_TYPES` entry under FTP | Reservation = one instance type; one source of truth; assert early + backstop |
| D-TWOTIER | Cheap litmus gate + expensive full eval | Skip the heavy run if the gate fails |
| D-SMEVAL | Use managed eval for the litmus test only, not the full agentic eval | ShoppingBench scoring/tool loop isn't a simple managed scorer |
| D-TOOLRETURN | Litmus measures the tool call, independent of tool return | Ground truth is the correct call, not the return value |
| D-ASR-AGG | Report ASR as a simple mean over intents | Keep minority intents from being swamped |
| D-EVALQUOTA | Quota-aware eval scheduling (`L-619D690E`), fallback = 1 task | Avoid exceeding concurrent-eval-job quota / throttling |
| D-STUDENT-THRESH | Student pass threshold = base model's value | Student must beat the base it was distilled from |
| D-CACHEDREF | Serve reference/frontier comparison rows from a frozen local cache, not live calls | Decouples result tables from live frontier-model identity/endpoints; removes per-run model ID config |

---

## 11. Changelog

Chronological record of the project's evolution (most recent last within each phase).

### Phase 1 — Data & training foundation
- Reviewed the SeqKD dataset for format/suitability for serverless customization of Qwen3.5-4B.
- Chose hyperparameters (LoRA r64) weighing recommended vs. suggested; confirmed the job is **SFT**, not
  the managed "Distillation" option (D-JOBTYPE).
- Built per-step teacher trajectories using ShoppingBench's own prompt functions, split by query (D-PROMPT).
- Launched serverless SFT; registered a ready-to-deploy Model Package.

### Phase 2 — The reasoning pivot
- Identified that labels-only training isn't really distillation; pivoted to **reasoning + tool calls**
  and set the goal of proving it beats labels-only (D-REASON).
- Stood up a **teacher model** (Qwen3.5-27B) on a SageMaker endpoint to generate reasoning traces;
  debugged endpoint deployment (CloudWatch logs), FP8 considerations on H100, JumpStart vs HuggingFace
  DJL-LMI deployment paths; added training-plan **active/scheduled/expired** gating for the teacher job.

### Phase 3 — SDK & notebook hygiene
- Migrated imports to **SDK v3** (D-SDKV3); pinned installs; consolidated dependency installs into the
  setup cell (D-PIN).
- Fixed `BUCKET` to use the **SageMaker session default bucket**; namespaced vendored imports
  (`ShoppingBench.src...`); chunked a slow data-prep cell for performance.
- Rewrote customer-facing markdown across the notebooks to teach as the user runs.

### Phase 4 — Overtraining & loss curves
- Added a training/validation **loss-curve cell** reading from MLflow when available.
- Investigated missing **val loss**: confirmed it's by-design for the single-dataset + `train_val_split_ratio`
  config (D-VALLOSS); removed the ineffective `SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT` usage.
- Flagged likely **overtraining** (loss plateaued ~step 500, run continued past ~1000) as a teaching point
  (D-OVERTRAIN).

### Phase 5 — Evaluation design
- Rejected managed eval for the **full** agentic eval (scoring is ~10% of the work) (D-SMEVAL); kept the
  full ShoppingBench ASR path.
- Designed the **two-tier** approach: a lightweight **litmus** gate + the heavy eval (D-TWOTIER).
- Built the eval to run on **both base and student**, scored on **held-out** problems (D-HOLDOUT), with
  **per-task + aggregate** ASR via **simple mean** (D-ASR-AGG); student threshold = base (D-STUDENT-THRESH).
- Added a **quota-aware eval scheduler** on `L-619D690E` with a safe fallback of 2 concurrent jobs /
  1 task (D-EVALQUOTA); context-length purge at `>= 12000` tokens; `N_EXAMPLES` subset.

### Phase 6 — Deployment refactor
- Replaced **ModelBuilder** / `EndpointDeployer` / `sm_endpoints` with the **SageMaker Core** resource API
  in `kd_serve.py`; standardized on **DJL-LMI/vLLM** and removed the DJL-vs-vLLM toggle (D-DEPLOY, D-SERVE).
- Converted to **one endpoint = base IC + adapter IC(s)**; adapter resolved from the Model Package;
  idempotent `attach_adapter`; dependency-ordered `teardown_endpoint` (D-IC).
- Switched harness calls to `InvokeEndpoint` with `InferenceComponentName` via the local OpenAI relay
  after the OpenAI passthrough route rejected IC routing (D-INVOKE); made the relay reuse endpoints and
  re-mint SigV4 per call.
- Hardened the **preflight quota check** (endpoint quotas only, cached to a git-ignored JSON) after it was
  throttled.

### Phase 7 — Litmus notebook & reward function
- Built the litmus notebook on a **Custom Scorer** reward function with three graded levels
  (`toolset_match` gate, `format_match`, `kind_match`), a **unit test** before shipping, and clear
  pass/fail + reward-source rendering.
- Factored notebook logic into `litmus_utils.py`; added markdown parity with the eval notebook; moved
  non-executed source (`reward_functions/`) out of `scripts/`; wrote litmus datasets to git-ignored
  `eval_inputs/`.
- Fixed the gate notebook's (`2-tool-call-accuracy-gate.ipynb`) `ModuleNotFoundError: litmus_utils` by adding
  `sys.path.insert(0, str(Path("scripts").resolve()))` to its config cell (matching the sibling notebook).

### Phase 8 — Flexible Training Plan (FTP) support
- Added FTP capacity-reservation support to the deployment (`kd_serve.deploy_base_endpoint` +
  `_variant` build `ProductionVariantCapacityReservationConfig`, `capacity-reservations-only`).
- Wired FTP into the eval notebook: initially with a redundant second variable
  (`TRAINING_PLAN_INSTANCE_TYPE`), then **simplified** to a single source of truth — `TRAINING_PLAN_ARN`
  plus a length-1 `INSTANCE_TYPES` requirement (D-FTP).
- Added an **early `assert`** in the config cell so an FTP + multi-type mismatch fails before any deploy
  work, backed by the existing `ValueError` in `kd_serve` (D-FTP).

### Phase 9 — Operational debugging (eval run)
- Diagnosed stalled rollouts: the **search server** saturated (connection limit, queue depth ~90), causing
  `find_product` 60 s read timeouts and flat completion counts. Remedy: restart the search server, lower
  `ROLLOUT_THREADS`; rollouts are resume-safe.
- Diagnosed high host memory: a ~3 GB (growing) **search server**, long-lived **jupyter-lab server** heap
  (~11.7 GB over ~8 days uptime, reclaimable only by restarting the server), and detached subprocesses
  that survive kernel shutdown. Confirmed the notebook/eval themselves were behaving normally.

### Phase 10 — Packaging & cleanup
- Stale-code sweep across the notebooks (removed dead cells/sections, long duplicated comments folded into
  markdown).
- Produced a lean tarball of the `distillation/` folder (notebooks + scripts) for offline save.

### Phase 11 — Retrain, anonymized reference comparisons, data relocation
- Retrained the student (new Model Package version registered); the eval notebook's `MODEL_PKG_ARN` and
  the `student` inference component were updated to serve the new version, replacing the original student.
- Reworked the reference/frontier comparison rows: retired the single original reference row and
  replaced it with a stronger comparison model and a same-family dense model shown alongside base/student.
  (A second, faster comparison model was evaluated alongside these but dropped from the published
  comparison set — it didn't add a distinct point to the discussion once the same-family dense reference
  and the frontier ceiling were already in the table.)
- Switched those comparison rows from live per-run calls to a **frozen local cache**
  (`out/reference_evaluations/`) — `CACHED_REF_TAGS` lists the tags, no model ID config remains in the
  notebook for them (D-CACHEDREF). This also means the recipe's result tables no longer reveal which
  frontier models were used as comparison points.
- Removed all remaining references to specific frontier-model/vendor names from the notebooks, scripts,
  and this document, generalizing them to "reference/frontier comparison model" language, consistent with
  D-CACHEDREF.
- Relocated the training/validation JSONL files from the recipe root into a `data/` folder (git-ignored
  in its entirety); updated the data-prep notebook's write step and the S3 upload step (which previously
  relied on the files being bare filenames) and the litmus notebook's read path accordingly.

### Phase 12 — Pre-publish polish
- Reduced the published comparison set to two rows (`reference_qwen3-32b`, `frontier_medium`) across the
  eval notebook, plot, result tables, and `eval_utils.plot_results` ordering; refreshed the "Reading the
  results" section with a value table, a student-gap delta table, and an embedded ASR bar chart
  (`asr_results_chart.png`) carrying a "sample run, may differ" caption.
- Brought all headline/intro numbers current (data-prep notebook intro, README, SPEC) to one consistent
  held-out sample run; corrected a stale "reference model goes to Bedrock" note in the eval notebook to
  the cached-rollout reality.
- Trimmed overly verbose notebook code-cell comments (notably the eval notebook's config/deploy/relay
  cells) while keeping the non-obvious gotchas; scripts left unchanged.

---

*This SPEC is a living document — update the decision index (§10) and changelog (§11) when the recipe changes.*
