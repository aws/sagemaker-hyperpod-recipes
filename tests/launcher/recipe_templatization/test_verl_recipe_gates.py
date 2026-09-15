"""CPU pre-merge gates for verl recipes (no GPU, no verl import).

Renders every verl recipe with Hydra and asserts the static invariants each verl
recipe must satisfy before merge:

  * token-gate min (qwen-3-4b bug) -> "recipes right": min(all *_max_token_len_per_gpu,
                                      each credited by its section's ulysses_sp) must be
                                      >= the sequence length verl packs. Under Ulysses SP
                                      a sequence is sharded across sp ranks, so verl
                                      compares gate * sp against the packed sequence.
  * default lengths within budget  -> "recipes right": the recipe's baked prompt+response
                                      (RL) / max_length (SFT/DPO) must not exceed its
                                      budget (max_token_len_per_gpu * sp) -- the ceiling
                                      the UI enforces the length overrides against.
  * max_num_batched_tokens          -> "will it start" on verl 0.5: the vLLM cap must
                                      cover (max_model_len or prompt+response) under
                                      chunked prefill; 0.5-fatal, informational on 0.7.
  * sequence-length metadata (H3)  -> "recipes right": SequenceLength must be derivable
                                      and bucket to a valid published enum.
  * image version-consistency (4)  -> "image right": the recipe resolves to a container
                                      whose tag matches its verl version, present in
                                      every prod region.

verl's own validate_config would additionally validate batch/parallelism relationships, but it
must run with the verl wheel MATCHING each recipe's targeted image (0.5 vs 0.7 config schemas
differ) -- a per-version on-image lane, out of scope for this static CPU gate.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from launcher.recipe_templatization.base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
)
from launcher.recipe_templatization.verl.verl_recipe_template_processor import (
    VerlRecipeTemplateProcessor,
)
from scripts.generate_launch_jsons import LaunchJsonGenerator
from scripts.validations.validation_launchers.launcher_utils import (
    COMMON_CONFIG_PATH,
    _get_recipe_type_info,
)
from utils.recipe_utils import load_recipe_with_hydra

REPO = Path(__file__).resolve().parents[3]
RECIPES_ROOT = REPO / "recipes_collection" / "recipes"
REGIONAL_JSON = REPO / "launcher" / "recipe_templatization" / "verl" / "verl_regional_parameters.json"


@lru_cache(maxsize=1)
def _common_cfg():
    return OmegaConf.load(COMMON_CONFIG_PATH)


def resolve_container(recipe_rel_path: str):
    """(container_key, None) the real launch path would pick for a recipe, or (None, None).

    Inlined from the validation runner's config-driven routing (_get_recipe_type_info +
    container_info) so the recipe gate tests don't depend on the token_sweep_ui package, which
    isn't part of the installed wheel and so isn't importable in CI."""
    try:
        return _get_recipe_type_info(_common_cfg(), recipe_rel_path)["container_key"], None
    except Exception:
        return None, None


# The six *_max_token_len_per_gpu gates. verl asserts each (credited by its section's
# ulysses_sequence_parallel_size) >= the packed sequence: dp_actor.py scales the gate by
# sp before the packing assert, because under Ulysses SP a single sequence is sharded
# across sp ranks. _collect_gates applies the gate * sp credit; _sp_for_gate maps each
# gate to the sp that shards it (the policy workers share the actor's sp).
GATE_PATHS = (
    ("actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu"),
    ("actor_rollout_ref", "ref", "log_prob_max_token_len_per_gpu"),
    ("actor_rollout_ref", "rollout", "log_prob_max_token_len_per_gpu"),
    ("critic", "ppo_max_token_len_per_gpu"),
    ("critic", "forward_max_token_len_per_gpu"),
    ("reward_model", "forward_max_token_len_per_gpu"),
)

# container_key (filename-derived) -> verl_regional_parameters.json top-level key.
CONTAINER_KEY_TO_REGIONAL = {
    "verl": "verl",
    "verl_0_7_0": "verl-0.7.0",
    "verl_0_7_0_vllm012": "verl-0.7.0-vllm012",
    "verl_0_7_0_tf58": "verl-0.7.0-tf58",
    "verl_0_7_0_megatron": "verl-0.7.0-megatron",
}
# Expected image tag per verl version. 0.5 -> v1.0.0; the 0.7 variants are bumped
# independently as each image ships, so they are not expected to share a tag.
REGIONAL_KEY_TO_TAG = {
    "verl": "verl-v1.0.0",
    "verl-0.7.0": "verl-v1.1.3",
    "verl-0.7.0-vllm012": "verl-v1.1.0",
    "verl-0.7.0-tf58": "verl-v1.1.3",
    "verl-0.7.0-megatron": "verl-v1.1.2",
}
VALID_BUCKETS = {"1K", "2K", "4K", "8K", "16K", "32K", "64K", "128K"}


def _verl_recipe_rels() -> list[str]:
    """Every verl recipe as a path relative to recipes_collection/recipes, no .yaml."""
    found = LaunchJsonGenerator().discover_recipes(prefixes=["verl"])
    rels = [str(item[0].relative_to(RECIPES_ROOT).with_suffix("")) for item in found]
    return sorted(rels)


@lru_cache(maxsize=None)
def _render(rel: str) -> dict:
    """Fully-resolved recipe dict (cached; each recipe renders once per session)."""
    return load_recipe_with_hydra(rel, return_dict=True)


def _get(d: dict, *path):
    """Nested lookup returning None if any level is missing."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_rl(tc: dict) -> bool:
    """RL recipes carry an actor ppo token-gate; SFT/DPO do not."""
    return _get(tc, "actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu") is not None


def _sequence_length(tc: dict) -> int | None:
    """RL: max_prompt_length + max_response_length; else data.max_length.

    Mirrors VerlRecipeTemplateProcessor._extract_sequence_length so the gate and the
    published-metadata derivation agree.
    """
    data = tc.get("data", {})
    prompt = data.get("max_prompt_length")
    resp = data.get("max_response_length")
    if prompt is not None and resp is not None:
        return prompt + resp
    return data.get("max_length")


def _bucket(seq_len: int) -> str:
    """Round down to the nearest published SequenceLength bucket (1K..128K)."""
    ladder = [
        (1024, "1K"),
        (2048, "2K"),
        (4096, "4K"),
        (8192, "8K"),
        (16384, "16K"),
        (32768, "32K"),
        (65536, "64K"),
        (131072, "128K"),
    ]
    label = ladder[0][1]
    for size, name in ladder:
        if seq_len >= size:
            label = name
    return label


def _sp_for_gate(tc: dict, path: tuple) -> int:
    """ulysses_sequence_parallel_size that shards the sequence for this gate.

    The policy model's workers (actor/ref/rollout) share the actor's SP group, so
    every actor_rollout_ref.* gate credits the actor's sp; critic and reward_model
    use their own. Defaults to 1 when the section omits it."""
    if path[0] == "actor_rollout_ref":
        sp = _get(tc, "actor_rollout_ref", "actor", "ulysses_sequence_parallel_size")
    else:
        sp = _get(tc, path[0], "ulysses_sequence_parallel_size")
    return sp if isinstance(sp, int) and sp > 0 else 1


def _collect_gates(tc: dict) -> dict[str, int]:
    """{dotted-path: effective gate} for every *_max_token_len_per_gpu gate present.

    Effective gate = raw gate * that section's ulysses_sequence_parallel_size. Under
    Ulysses SP a single sequence is sharded across sp ranks, so verl compares
    gate * sp against the packed sequence (dp_actor scales the gate by sp before the
    assert). sp defaults to 1."""
    gates = {}
    for path in GATE_PATHS:
        val = _get(tc, *path)
        if isinstance(val, int):
            gates[".".join(path)] = val * _sp_for_gate(tc, path)
    return gates


RELS = _verl_recipe_rels()

# Known token-gate findings surfaced by this gate (min *_max_token_len_per_gpu < packed
# sequence -- the qwen-3-4b bug class). xfail'd strict=True so the gate stays green while
# recording them, but flips RED the moment a recipe is fixed (remove it here) or a NEW
# recipe regresses. Confirmed by verl source: seqlen_balancing.rearrange_micro_batches
# asserts max_token_len >= max_seq_len at the first training step.
KNOWN_UNDERSET_GATES = {
    # ref.log_prob_max_token_len_per_gpu 16384 with ulysses_sp=1 (no sp credit) sits
    # below the packed sequence -- a real under-set (verl 0.7). (qwen-3-14b is NOT
    # here: it ships ulysses_sp=2, so its gate 24576 * 2 = 49152 covers the sequence.)
    "fine-tuning/qwen-0_7_0/verl-grpo-rlaif-qwen-3-dot-5-4b-lora",
    "fine-tuning/qwen-0_7_0/verl-grpo-rlaif-qwen-3-dot-5-9b-lora",
}
_TOKEN_GATE_PARAMS = [
    pytest.param(rel, marks=pytest.mark.xfail(reason="known under-set token gate (qwen-3-4b bug class)", strict=True))
    if rel in KNOWN_UNDERSET_GATES
    else rel
    for rel in RELS
]


def _budget(tc: dict) -> int | None:
    """The recipe's sequence-length budget = max_token_len_per_gpu * ulysses_sp.

    Mirrors VerlRecipeTemplateProcessor._extract_context_length: RL uses the actor's
    ppo_max_token_len_per_gpu and ulysses_sequence_parallel_size; SFT/DPO use
    data.max_token_len_per_gpu and engine.ulysses_sequence_parallel_size. sp defaults
    to 1. Returns None when the token-budget field is absent."""
    if _is_rl(tc):
        gate = _get(tc, "actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu")
        sp = _get(tc, "actor_rollout_ref", "actor", "ulysses_sequence_parallel_size")
    else:
        gate = _get(tc, "data", "max_token_len_per_gpu")
        sp = _get(tc, "engine", "ulysses_sequence_parallel_size")
    if not isinstance(gate, int) or gate <= 0:
        return None
    sp = sp if isinstance(sp, int) and sp > 0 else 1
    return gate * sp


def _packed_sequence_length(tc: dict) -> int | None:
    """Max sequence verl packs at runtime = rollout.prompt_length + rollout.response_length
    for RL (what generation actually produces and the gate assert compares against);
    data.max_length for SFT/DPO.

    NOTE this differs from _sequence_length (the published-metadata derivation, which uses
    data.max_{prompt,response}_length). RLAIF recipes legitimately set data.max_response_length
    != rollout.response_length, so the runtime gate must key off the rollout caps.
    """
    rollout = _get(tc, "actor_rollout_ref", "rollout") or {}
    prompt = rollout.get("prompt_length")
    resp = rollout.get("response_length")
    if prompt is not None and resp is not None:
        return prompt + resp
    return tc.get("data", {}).get("max_length")


# NOTE on the "coupled-set" (G1): the real invariant is that an *override* of a length
# param writes both data.max_*_length AND rollout.*_length together -- an override-propagation
# property, NOT baked equality (RLAIF recipes deliberately ship
# data.max_response_length != rollout.response_length). This gate file only checks the
# as-shipped snapshot; the coupled-set / override invariants live in test_verl_override.py.


@pytest.mark.parametrize("rel", _TOKEN_GATE_PARAMS)
def test_token_gates_ge_sequence_length(rel):
    """qwen-3-4b bug class: min(all *_max_token_len_per_gpu * ulysses_sp) must be >= the
    packed sequence.

    verl asserts each gate * sp >= max_seq_len at the first training step
    (seqlen_balancing: "max_token_len must be greater than the sequence length"; dp_actor
    scales the gate by ulysses_sequence_parallel_size before the assert).
    """
    tc = _render(rel)["training_config"]
    gates = _collect_gates(tc)  # already credited by ulysses_sp
    if not gates:
        pytest.skip("no token-length gates in this recipe")
    seq_len = _packed_sequence_length(tc)
    assert seq_len is not None, f"{rel}: could not derive packed sequence length"
    offenders = {name: val for name, val in gates.items() if val < seq_len}
    assert not offenders, (
        f"{rel}: token gate(s) below packed sequence length {seq_len} "
        f"(rollout.prompt_length+response_length): {offenders} (values shown are gate * ulysses_sp). "
        f"verl asserts every *_max_token_len_per_gpu * sp >= seq; the min effective gate is the "
        f"ceiling (this is the qwen-3-4b gate-bug class)."
    )


@pytest.mark.parametrize("rel", RELS)
def test_max_num_batched_tokens_covers_model_len(rel):
    """verl 0.5: under chunked prefill, cap >= (max_model_len or prompt+response).

    0.5-fatal (vllm_rollout_spmd.py:151 guard); verl 0.7 deleted the guard, so it is
    only asserted for recipes that resolve to the 0.5 image.
    """
    key, _ = resolve_container(rel + ".yaml")
    if key != "verl":
        pytest.skip(f"not a verl-0.5 recipe (container_key={key}); guard deleted in 0.7")
    tc = _render(rel)["training_config"]
    rollout = _get(tc, "actor_rollout_ref", "rollout") or {}
    if rollout.get("enable_chunked_prefill") is False:
        pytest.skip("chunked prefill disabled: guard does not apply")
    cap = rollout.get("max_num_batched_tokens")
    if cap is None:
        pytest.skip("no max_num_batched_tokens set")
    mml = rollout.get("max_model_len")
    needed = mml if mml is not None else _sequence_length(tc)
    assert needed is not None, f"{rel}: cannot determine required max_model_len"
    assert cap >= needed, (
        f"{rel}: max_num_batched_tokens {cap} < required {needed} "
        f"(max_model_len={mml}, seq_len={_sequence_length(tc)}). verl 0.5 aborts vLLM init here. "
        f"Fix: raise max_num_batched_tokens to >= the sequence budget."
    )


@pytest.mark.parametrize("rel", RELS)
def test_default_lengths_within_budget(rel):
    """The recipe's baked default lengths must not exceed its sequence-length budget.

    budget = max_token_len_per_gpu * ulysses_sp -- the ceiling published as the length
    override params' constraint.max, which the UI enforces prompt+response (RL) /
    dataset_max_len (SFT/DPO) against. The form pre-fills these baked defaults, so a baked
    prompt+response (RL) / max_length (SFT/DPO) above the budget would load the form in an
    invalid state (and over-subscribe the packing budget)."""
    tc = _render(rel)["training_config"]
    budget = _budget(tc)
    if budget is None:
        pytest.skip("no token budget in this recipe")
    seq_len = _sequence_length(tc)
    assert seq_len is not None, f"{rel}: could not derive baked sequence length"
    assert seq_len <= budget, (
        f"{rel}: baked default sequence {seq_len} exceeds budget {budget} "
        f"(max_token_len_per_gpu * ulysses_sp). The UI pre-fills the baked default and enforces "
        f"prompt+response <= budget, so this recipe would load in an invalid state. "
        f"Fix: raise max_token_len_per_gpu (or ulysses_sp), or lower the baked lengths."
    )


@pytest.mark.parametrize("rel", RELS)
def test_sequence_length_metadata_derivable(rel):
    """H3: SequenceLength must be derivable and bucket to a valid published enum."""
    tc = _render(rel)["training_config"]
    seq_len = _sequence_length(tc)
    assert seq_len is not None and seq_len > 0, f"{rel}: no derivable sequence length"
    assert _bucket(seq_len) in VALID_BUCKETS, f"{rel}: seq_len {seq_len} buckets outside {VALID_BUCKETS}"


@pytest.mark.parametrize("rel", RELS)
def test_image_resolves_version_consistent(rel):
    """Step 4: recipe resolves to a container whose tag matches its verl version,
    and that version is present in every prod region of the regional matrix."""
    key, _ = resolve_container(rel + ".yaml")
    assert key is not None, f"{rel}: container_key did not resolve"
    regional_key = CONTAINER_KEY_TO_REGIONAL.get(key)
    assert regional_key is not None, f"{rel}: unmapped container_key {key!r}"

    matrix = json.loads(REGIONAL_JSON.read_text())
    assert regional_key in matrix, f"{rel}: {regional_key!r} missing from verl_regional_parameters.json"

    expected_tag = REGIONAL_KEY_TO_TAG[regional_key]
    checked_any = False
    for platform in ("k8s", "sm_jobs"):
        prod = _get(matrix[regional_key], platform, "container_image", "prod")
        if not prod:
            continue
        checked_any = True
        wrong = {region: uri for region, uri in prod.items() if expected_tag not in uri}
        assert not wrong, (
            f"{rel}: image tag mismatch for {regional_key!r} {platform}/prod "
            f"(expected {expected_tag!r}) in regions {list(wrong)}"
        )
    assert checked_any, f"{rel}: no k8s or sm_jobs prod block for {regional_key!r}"


# ---------------------------------------------------------------------------
# Published-metadata gates: the processor emits the expected ceiling + label for
# EVERY verl recipe (runs the real get_additional_data / get_recipe_metadata).
# ---------------------------------------------------------------------------
_BASE_OVERRIDE = json.loads(
    (REPO / "launcher" / "recipe_templatization" / "base_override_parameters.json").read_text()
)["fine_tuning"]
# One processor reused across recipes; process_recipe(rel) fully resets per-recipe
# state, so results are order-independent (staging_cfg=base -> defaults stay static,
# which does not affect the `max` since the budget dominates the static default).
_LENGTH_PROC = VerlRecipeTemplateProcessor(staging_cfg=_BASE_OVERRIDE, platform="k8s")
_LENGTH_PARAMS = ("max_prompt_length", "max_response_length", "dataset_max_len")


@lru_cache(maxsize=None)
def _published(rel: str):
    """(override_params, metadata) the processor publishes for a recipe (cached once)."""
    _LENGTH_PROC.process_recipe(recipe_file_path=rel)
    _, override_params, _ = _LENGTH_PROC.get_additional_data(rel)
    metadata = _LENGTH_PROC.get_recipe_metadata(rel)
    return override_params, metadata


@pytest.mark.parametrize("rel", RELS)
def test_override_max_equals_budget(rel):
    """Every verl recipe publishes each length override param's `max` == its budget
    (max_token_len_per_gpu * ulysses_sp) -- the ceiling the UI enforces prompt+response (RL) /
    dataset_max_len (SFT/DPO) against."""
    tc = _render(rel)["training_config"]
    budget = _budget(tc)
    if budget is None:
        pytest.skip("no token budget in this recipe")
    override_params, _ = _published(rel)
    present = [p for p in _LENGTH_PARAMS if p in override_params]
    assert present, f"{rel}: exposes a budget but publishes no length override param"
    for p in present:
        assert override_params[p]["max"] == budget, (
            f"{rel}: {p} published max {override_params[p]['max']} != budget {budget} "
            f"(max_token_len_per_gpu * ulysses_sp)"
        )


@pytest.mark.parametrize("rel", RELS)
def test_sequence_length_uses_budget_calculation(rel):
    """The SequenceLength metadata is derived from the budget: the processor's
    _extract_context_length equals max_token_len_per_gpu * ulysses_sp, and the published label
    is that budget bucketed by format_sequence_length."""
    full = _render(rel)
    tc = full["training_config"]
    budget = _budget(tc)
    if budget is None:
        pytest.skip("no token budget in this recipe")
    # _extract_context_length takes the FULL recipe cfg (it reads .training_config itself)
    assert (
        _LENGTH_PROC._extract_context_length(full) == budget
    ), f"{rel}: _extract_context_length != max_token_len_per_gpu * ulysses_sp ({budget})"
    _, metadata = _published(rel)
    expected_label = _LENGTH_PROC.format_sequence_length(budget)
    assert metadata["SequenceLength"] == expected_label, (
        f"{rel}: SequenceLength {metadata['SequenceLength']!r} != "
        f"format_sequence_length(budget) {expected_label!r} (budget {budget})"
    )


def test_context_length_clamp_is_verl_only():
    """The budget/ceiling machinery lives ONLY on VerlRecipeTemplateProcessor: non-verl
    frameworks must not clamp length params to a per-recipe budget. Base carries neither the
    budget helper nor the bound-params list, and verl overrides get_additional_data to add the
    clamp -- so nova/llmft/etc. inherit base's unaltered behavior (static base `max`)."""
    assert hasattr(VerlRecipeTemplateProcessor, "_extract_context_length")
    assert hasattr(VerlRecipeTemplateProcessor, "_CONTEXT_LENGTH_BOUND_PARAMS")
    assert not hasattr(BaseRecipeTemplateProcessor, "_extract_context_length")
    assert not hasattr(BaseRecipeTemplateProcessor, "_CONTEXT_LENGTH_BOUND_PARAMS")
    assert (
        VerlRecipeTemplateProcessor.get_additional_data is not BaseRecipeTemplateProcessor.get_additional_data
    ), "verl must override get_additional_data to apply the budget clamp"


def test_sp_for_gate_credits_the_right_section():
    """The load-bearing sp-credit mapping: the policy workers (actor/ref/rollout) all share the
    ACTOR's ulysses_sp, so every actor_rollout_ref.* gate credits actor sp -- even when ref/rollout
    carry no sp of their own (this is why qwen-3-14b, whose rollout gate has no sp, still gets the
    x2 credit). critic and reward_model use their own sp; absent -> 1."""
    tc = {
        "actor_rollout_ref": {
            "actor": {"ulysses_sequence_parallel_size": 2},
            "ref": {},  # no own sp
            "rollout": {},  # no own sp
        },
        "critic": {"ulysses_sequence_parallel_size": 4},
        "reward_model": {},  # no sp
    }
    # actor / ref / rollout gates all credit the ACTOR's sp (=2), including ref/rollout with no own sp
    assert _sp_for_gate(tc, ("actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu")) == 2
    assert _sp_for_gate(tc, ("actor_rollout_ref", "ref", "log_prob_max_token_len_per_gpu")) == 2
    assert _sp_for_gate(tc, ("actor_rollout_ref", "rollout", "log_prob_max_token_len_per_gpu")) == 2
    # critic uses its own sp
    assert _sp_for_gate(tc, ("critic", "ppo_max_token_len_per_gpu")) == 4
    # reward_model has no sp -> defaults to 1
    assert _sp_for_gate(tc, ("reward_model", "forward_max_token_len_per_gpu")) == 1
    # actor sp absent -> defaults to 1 (so actor_rollout_ref.* gates all fall back to 1)
    tc_no_sp = {"actor_rollout_ref": {"actor": {}}}
    assert _sp_for_gate(tc_no_sp, ("actor_rollout_ref", "rollout", "log_prob_max_token_len_per_gpu")) == 1
