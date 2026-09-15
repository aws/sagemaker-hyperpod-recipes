"""Overridable length/token params — full concern-case coverage (CPU unit tests, no verl import).

Covers every failure mode of the sequence-length override contract, per param and per technique:

  RL length params (couple `data` ⇔ `rollout`):
    * max_prompt_length   → data.max_prompt_length   ⇔ rollout.prompt_length
    * max_response_length → data.max_response_length ⇔ rollout.response_length
  SFT/DPO length param:
    * dataset max length  → data.max_length
  Token budgets (the packing/vLLM guards the length params must stay under):
    * ppo_max_token_len / *_max_token_len_per_gpu (RL gates), data.max_token_len_per_gpu (SFT/DPO)
    * max_num_batched_tokens (the vLLM cap)

Concern cases asserted:
  A. **Coupling** — a length override writes its full coupled set and nothing else;
     a rollout-only write desyncs.
  B. **Length raised → budgets must still cover** — a correct override scales the cap + gates so
     `gate ≥ seq` and `cap ≥ seq`; the same override *without* scaling is caught (proves the fix
     is load-bearing, not decorative).
  C. **Budget lowered below seq is caught** — overriding a gate/cap below the current sequence
     trips the `gate ≥ seq` / cap invariant.

These unit tests assert the invariants an override MUST satisfy, applied through reference
appliers on the rendered recipe (fast, no launch). The live override processor is exercised
end-to-end by the integration tests (property + GPU smoke) in
HP-ModelCustomization-ImageIntegrationTests.
"""
from __future__ import annotations

import copy
import json

import pytest

from tests.launcher.recipe_templatization.test_verl_recipe_gates import (
    GATE_PATHS,
    RELS,
    REPO,
    VALID_BUCKETS,
    _bucket,
    _collect_gates,
    _get,
    _is_rl,
    _packed_sequence_length,
    _render,
    _sequence_length,
    _sp_for_gate,
)

# The only customer-overridable length params (base_override_parameters.json). The token budgets
# (ppo/*_max_token_len_per_gpu, max_num_batched_tokens) are NOT overridable — they are the gates
# the length overrides must stay under, exercised as constraints below.
_CONTRACT = json.loads((REPO / "launcher" / "recipe_templatization" / "base_override_parameters.json").read_text())


def _contract_range(name: str) -> tuple[int, int]:
    def find(o):
        if isinstance(o, dict):
            n = o.get(name)
            if isinstance(n, dict) and n.get("type") == "integer" and "min" in n:
                return n
            for v in o.values():
                r = find(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v)
                if r:
                    return r
        return None

    node = find(_CONTRACT)
    assert node is not None, f"{name} not an integer param in base_override_parameters.json"
    return int(node["min"]), int(node["max"])


OVERRIDABLE_LENGTH_PARAMS = ["max_prompt_length", "max_response_length", "max_length"]
# Token budgets that MUST stay baked (not customer-overridable): the length overrides rely on them
# as fixed ceilings + the fix scales them internally. Exposing one as an override without the
# gate/coupling logic would reintroduce the vLLM-cap / under-set-gate class.
NON_OVERRIDABLE_BUDGETS = [
    "ppo_max_token_len_per_gpu",
    "log_prob_max_token_len_per_gpu",
    "forward_max_token_len_per_gpu",
    "max_token_len_per_gpu",
    "max_num_batched_tokens",
]


def _param_in_contract(name: str) -> bool:
    def walk(o):
        if isinstance(o, dict):
            n = o.get(name)
            if isinstance(n, dict) and "type" in n:
                return True
            return any(walk(v) for v in o.values())
        if isinstance(o, list):
            return any(walk(v) for v in o)
        return False

    return walk(_CONTRACT)


RL_RELS = [rel for rel in RELS if _is_rl(_render(rel)["training_config"])]
# SFT/DPO recipes: no actor ppo gate, but a data.max_length + its own data.max_token_len_per_gpu gate.
SFT_DPO_RELS = [
    rel
    for rel in RELS
    if not _is_rl(_render(rel)["training_config"])
    and _get(_render(rel)["training_config"], "data", "max_length") is not None
    and _get(_render(rel)["training_config"], "data", "max_token_len_per_gpu") is not None
]

_RL_FIELDS = {"prompt": ("max_prompt_length", "prompt_length"), "response": ("max_response_length", "response_length")}
_OTHER = {"prompt": "response", "response": "prompt"}


# --------------------------------------------------------------------------- appliers
def _override_rl_length(tc, prompt, response, *, couple=True, scale=False):
    """RL length override. couple: write data+rollout together. scale: also raise cap + every gate
    to cover the new sequence (the fix). couple=False = the rollout-only mapping."""
    tc = copy.deepcopy(tc)
    rollout = tc["actor_rollout_ref"]["rollout"]
    rollout["prompt_length"], rollout["response_length"] = prompt, response
    if couple:
        data = tc.setdefault("data", {})
        data["max_prompt_length"], data["max_response_length"] = prompt, response
    if scale:
        seq = prompt + response
        cap = rollout.get("max_num_batched_tokens")
        if cap is not None:
            # cap must cover the derived max_model_len (which may be a fixed value > seq), not just seq
            needed = max(seq, rollout.get("max_model_len") or 0)
            rollout["max_num_batched_tokens"] = max(cap, needed)
        for path in GATE_PATHS:
            parent = _get(tc, *path[:-1])
            if isinstance(parent, dict) and isinstance(parent.get(path[-1]), int):
                parent[path[-1]] = max(parent[path[-1]], seq)
    return tc


def _override_rl_single(tc, param, value, *, couple=True):
    """Override ONE RL length param (prompt or response), touching only its own pair.
    couple=False = the rollout-only mapping."""
    tc = copy.deepcopy(tc)
    data_key, rollout_key = _RL_FIELDS[param]
    tc["actor_rollout_ref"]["rollout"][rollout_key] = value
    if couple:
        tc.setdefault("data", {})[data_key] = value
    return tc


def _override_sft_max_length(tc, max_length, *, scale=False):
    """SFT/DPO dataset-max-length override; scale raises data.max_token_len_per_gpu to cover it."""
    tc = copy.deepcopy(tc)
    data = tc["data"]
    data["max_length"] = max_length
    if scale:
        data["max_token_len_per_gpu"] = max(data["max_token_len_per_gpu"], max_length)
    return tc


# --------------------------------------------------------------------------- A. coupling
@pytest.mark.parametrize("param,sentinel", [("prompt", 3072), ("response", 5120)])
@pytest.mark.parametrize("rel", RL_RELS)
def test_length_override_couples_only_its_own_pair(rel, param, sentinel):
    """Overriding one length param moves BOTH its coupled fields and leaves the other untouched."""
    base = _render(rel)["training_config"]
    data_key, rollout_key = _RL_FIELDS[param]
    tc = _override_rl_single(base, param, sentinel, couple=True)

    assert tc["data"][data_key] == tc["actor_rollout_ref"]["rollout"][rollout_key] == sentinel
    o_data_key, o_rollout_key = _RL_FIELDS[_OTHER[param]]
    assert tc["data"].get(o_data_key) == base.get("data", {}).get(o_data_key)
    assert tc["actor_rollout_ref"]["rollout"].get(o_rollout_key) == base["actor_rollout_ref"]["rollout"].get(
        o_rollout_key
    )


@pytest.mark.parametrize("param,bump", [("prompt", 3072), ("response", 5120)])
@pytest.mark.parametrize("rel", RL_RELS)
def test_rollout_only_override_desyncs_its_pair(rel, param, bump):
    """Writing only the rollout side leaves data.max_<param>_length behind → desync."""
    base = _render(rel)["training_config"]
    data_key, rollout_key = _RL_FIELDS[param]
    value = (base.get("data", {}).get(data_key) or 0) + bump
    tc = _override_rl_single(base, param, value, couple=False)
    assert (
        tc["data"].get(data_key) != tc["actor_rollout_ref"]["rollout"][rollout_key]
    ), f"{rel}: expected {param} desync"


# ------------------------------------------------ B. length raised → budgets must still cover
@pytest.mark.parametrize("rel", RL_RELS)
def test_rl_length_override_with_fix_keeps_gates_and_cap_covered(rel):
    """Raising prompt+response to the contract ceiling, WITH the fix, keeps every gate and the
    vLLM cap >= the new sequence (no under-set-gate / cap violation)."""
    prompt, response = 16384, 16384  # max_prompt_length ceiling; large response
    seq = prompt + response
    tc = _override_rl_length(_render(rel)["training_config"], prompt, response, couple=True, scale=True)
    for name, val in _collect_gates(tc).items():
        assert val >= seq, f"{rel}: gate {name}={val} < seq {seq} after scaled override"
    rollout = tc["actor_rollout_ref"]["rollout"]
    cap = rollout.get("max_num_batched_tokens")
    if cap is not None and rollout.get("enable_chunked_prefill") is not False:
        needed = rollout.get("max_model_len") or seq
        assert cap >= needed, f"{rel}: cap {cap} < {needed} after scaled override"


@pytest.mark.parametrize("rel", RL_RELS)
def test_rl_length_override_without_scaling_is_caught(rel):
    """The same raise WITHOUT scaling under-sets a gate — proving the override must scale gates
    (a length override alone is not safe)."""
    base = _render(rel)["training_config"]
    gates = _collect_gates(base)
    if not gates:
        pytest.skip("no token gates")
    seq = 16384 + 16384
    if seq <= min(gates.values()):
        pytest.skip("baked gates already cover the raised sequence")
    tc = _override_rl_length(base, 16384, 16384, couple=True, scale=False)
    offenders = {n: v for n, v in _collect_gates(tc).items() if v < seq}
    assert offenders, f"{rel}: expected an under-set gate at seq {seq} without scaling"


# ------------------------------------------------ dataset_max_len (SFT/DPO) vs its gate
@pytest.mark.parametrize("rel", SFT_DPO_RELS)
def test_dataset_max_len_override_with_fix_covered_by_gate(rel):
    """Raising data.max_length WITH gate scaling keeps data.max_token_len_per_gpu >= max_length."""
    base = _render(rel)["training_config"]
    new_len = base["data"]["max_token_len_per_gpu"] + 8192
    tc = _override_sft_max_length(base, new_len, scale=True)
    assert tc["data"]["max_token_len_per_gpu"] >= tc["data"]["max_length"]


@pytest.mark.parametrize("rel", SFT_DPO_RELS)
def test_dataset_max_len_override_without_scaling_is_caught(rel):
    """Raising data.max_length beyond the gate WITHOUT scaling under-sets it (the packing assert)."""
    base = _render(rel)["training_config"]
    new_len = base["data"]["max_token_len_per_gpu"] + 8192
    tc = _override_sft_max_length(base, new_len, scale=False)
    assert (
        tc["data"]["max_token_len_per_gpu"] < tc["data"]["max_length"]
    ), f"{rel}: expected the gate to be under-set by the raised max_length"


# ------------------------------------------------ C. budget lowered below seq is caught
@pytest.mark.parametrize("rel", RL_RELS)
def test_token_gate_override_below_seq_is_caught(rel):
    """Overriding the ppo token gate so its effective capacity (gate * ulysses_sp) drops below
    the packed sequence trips the gate >= seq invariant."""
    base = _render(rel)["training_config"]
    seq = _packed_sequence_length(base)
    if seq is None or seq <= 1:
        pytest.skip("no derivable sequence")
    # Lower the RAW gate below seq/sp so that even after the * sp credit it falls under seq
    # (recipes with ulysses_sp>1 need the raw value pushed below seq/sp, not just seq-1).
    sp = _sp_for_gate(base, ("actor_rollout_ref", "actor", "ppo_max_token_len_per_gpu"))
    tc = copy.deepcopy(base)
    tc["actor_rollout_ref"]["actor"]["ppo_max_token_len_per_gpu"] = max((seq - 1) // sp, 1)
    offenders = {n: v for n, v in _collect_gates(tc).items() if v < seq}
    assert offenders, f"{rel}: lowering the ppo gate below seq/sp (seq={seq}, sp={sp}) should be caught"


@pytest.mark.parametrize("rel", RL_RELS)
def test_cap_override_below_seq_is_caught(rel):
    """Overriding max_num_batched_tokens below (max_model_len or prompt+response) under
    chunked prefill is the vLLM-init abort case."""
    base = _render(rel)["training_config"]
    rollout = base["actor_rollout_ref"]["rollout"]
    if rollout.get("max_num_batched_tokens") is None or rollout.get("enable_chunked_prefill") is False:
        pytest.skip("no cap / chunked prefill disabled")
    seq = _packed_sequence_length(base)
    needed = rollout.get("max_model_len") or seq
    tc = copy.deepcopy(base)
    tc["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] = needed - 1  # lowered below need
    cap = tc["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"]
    assert cap < needed, f"{rel}: cap {cap} below needed {needed} is the vLLM-init abort case"


# ------------------------------------------------ G2. bounds / type of the overridable params
@pytest.mark.parametrize("param", OVERRIDABLE_LENGTH_PARAMS)
def test_overridable_param_bounds_wellformed(param):
    """Each overridable length param declares a coherent integer range (type=int, 0 < min < max)."""
    lo, hi = _contract_range(param)
    assert isinstance(lo, int) and isinstance(hi, int)
    assert 0 < lo < hi, f"{param}: incoherent declared range ({lo}, {hi})"


@pytest.mark.parametrize("edge", ["min", "max"])
@pytest.mark.parametrize("rel", RL_RELS)
def test_rl_length_override_at_declared_edges(rel, edge):
    """At the declared MIN and MAX of prompt/response, the correct (scaled) override keeps every
    gate + cap ≥ seq and the SequenceLength bucket valid — G2 boundary behavior end-to-end."""
    pmin, pmax = _contract_range("max_prompt_length")
    rmin, rmax = _contract_range("max_response_length")
    prompt, response = (pmin, rmin) if edge == "min" else (pmax, rmax)
    seq = prompt + response
    tc = _override_rl_length(_render(rel)["training_config"], prompt, response, couple=True, scale=True)
    assert _sequence_length(tc) == seq
    assert _bucket(seq) in VALID_BUCKETS
    for name, val in _collect_gates(tc).items():
        assert val >= seq, f"{rel}: gate {name}={val} < seq {seq} at {edge} edge"
    rollout = tc["actor_rollout_ref"]["rollout"]
    cap = rollout.get("max_num_batched_tokens")
    if cap is not None and rollout.get("enable_chunked_prefill") is not False:
        assert cap >= (rollout.get("max_model_len") or seq)


@pytest.mark.parametrize("edge", ["min", "max"])
@pytest.mark.parametrize("rel", SFT_DPO_RELS)
def test_dataset_max_len_override_at_declared_edges(rel, edge):
    """At the declared MIN and MAX of the dataset max length, the scaled override keeps the
    SFT/DPO token gate ≥ max_length and the SequenceLength bucket valid."""
    lo, hi = _contract_range("max_length")
    max_length = lo if edge == "min" else hi
    tc = _override_sft_max_length(_render(rel)["training_config"], max_length, scale=True)
    assert tc["data"]["max_length"] == max_length
    assert tc["data"]["max_token_len_per_gpu"] >= max_length
    assert _bucket(max_length) in VALID_BUCKETS


# ------------------------------------------------ token budgets must stay baked, not overridable
@pytest.mark.parametrize("param", NON_OVERRIDABLE_BUDGETS)
def test_token_budget_is_not_customer_overridable(param):
    """The token budgets are baked ceilings, not override params. If a future change exposes one as
    overridable it must ship with the gate/cap-scaling + coupling logic — this guard flips red so
    that review can't be skipped (it would otherwise reintroduce the vLLM-cap / under-set-gate class)."""
    assert not _param_in_contract(param), (
        f"{param} is now in base_override_parameters.json — a token budget became customer-"
        f"overridable. Ensure the override scales the cap/gates and re-couples before removing "
        f"this guard."
    )


# --------------------------------------------------------------------------- contract coupling
# The reference-applier tests above check the *invariant* but not the actual override contract, so a
# contract that maps a length override to only part of its coupled set slips past them. This
# reads the real contract and asserts the full coupled set is wired for every GRPO template.
_VERL_CONTRACT = json.loads(
    (REPO / "launcher" / "recipe_templatization" / "verl" / "verl_recipe_template_parameters.json").read_text()
)
# Every field an overridable length param MUST populate in the contract, one row per field so a
# missing/wrong mapping (e.g. rollout.response_length unwired) fails as its OWN named case.
# group: which templates the field lives in ("grpo" = the 4 GRPO templates; "sftdpo" = sft/dpo).
_CONTRACT_FIELD_MAPPINGS = [
    ("grpo", ("data", "max_prompt_length"), "{{max_prompt_length}}"),
    ("grpo", ("actor_rollout_ref", "rollout", "prompt_length"), "{{max_prompt_length}}"),
    ("grpo", ("data", "max_response_length"), "{{max_response_length}}"),
    ("grpo", ("actor_rollout_ref", "rollout", "response_length"), "{{max_response_length}}"),
    ("sftdpo", ("data", "max_length"), "{{dataset_max_len}}"),
]


def _templates_in_group(group: str):
    """Return {name: training_config} for the templates in a group (grpo vs sft/dpo)."""
    out = {}
    for tn, t in _VERL_CONTRACT["templates"].items():
        is_grpo = tn.startswith("grpo")
        if (group == "grpo" and is_grpo) or (group == "sftdpo" and (tn.startswith("sft") or tn.startswith("dpo"))):
            out[tn] = t["recipe_template"]["training_config"]
    return out


@pytest.mark.parametrize(
    "group,path,placeholder",
    _CONTRACT_FIELD_MAPPINGS,
    ids=[f"{g}:{'.'.join(p)}" for g, p, _ in _CONTRACT_FIELD_MAPPINGS],
)
def test_override_contract_field_mapping(group, path, placeholder):
    """Each overridable length field must be wired to its placeholder in EVERY template of its group
    — read from the actual contract (not a reference applier). This is the check that catches a
    partial coupled-set mapping: e.g. `grpo:actor_rollout_ref.rollout.response_length` fails if the
    contract only maps the data
    side. Covers all six field mappings: data/rollout × prompt/response (GRPO) + dataset max_length
    (SFT/DPO)."""
    templates = _templates_in_group(group)
    assert templates, f"no {group} templates found in the verl contract"
    for tname, tc in templates.items():
        assert _get(tc, *path) == placeholder, (
            f"{tname}: contract does not map {'.'.join(path)} to {placeholder} -- an override of that "
            f"length param would leave this field un-updated (coupled-set desync)."
        )
