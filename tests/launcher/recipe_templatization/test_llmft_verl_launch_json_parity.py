# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""llmft vs verl override-parameter parity, compared through generated launch.json files.

For one representative model from each of five families (Llama, DeepSeek, Qwen2.5, Qwen3,
GPT-OSS) and each of SFT/DPO x LoRA/FFT, this test generates the launch.json of the llmft
recipe and of its verl counterpart on both platforms (sm_jobs and k8s), using the same
``scripts/generate_launch_jsons.py`` generator as ``test_launch_json_generation_and_validation.py``,
and compares the ``recipe_override_parameters`` the two files advertise.

Parity rule: every override name the llmft launch.json advertises is also advertised by the
verl one, with the same declared ``type``, so a job spec written against the llmft recipe still
validates against the verl recipe. Extra verl names are fine (``min_lr``, ``warmup_steps``).
Only names and types are compared; value ranges and defaults are recipe-specific (verl caps
``dataset_max_len`` at the length each recipe was sized for, and the Llama recipes round that
down to a power of 2).

Known gaps are pinned exactly in ``EXPECTED_GAPS`` instead of being allowlisted, so the test
fails both when a new gap appears and when a listed gap is closed (delete the entry then). A
verl recipe that still has a gap must stay excluded from JumpStart, since that is where job
specs written for the llmft recipe get carried over to it. The exception is a name in
``DELIBERATE_OMISSIONS``, which verl leaves out on purpose and which does not keep a recipe
off JumpStart.

``test_llmft_verl_override_parity.py`` pins the same rule for all 16 released Llama pairs at the
template and processor level; this module covers the other families and the final launch.json,
including the k8s path.
"""

import json
from pathlib import Path

import pytest

from scripts.generate_launch_jsons import LaunchJsonGenerator

PLATFORMS = ("sm_jobs", "k8s")

# family -> (technique, peft) -> (llmft recipe, verl recipe), relative to
# recipes_collection/recipes/fine-tuning, without the .yaml suffix. Every instance-type variant
# of a verl (model, technique, PEFT) advertises the same override names, so the variant picked
# here does not change the result.
SAMPLE = {
    "llama": {  # Llama 3.1 8B Instruct
        ("sft", "lora"): (
            "llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora",
            "llama-0_7_0/verl-sft-llama-3-dot-1-8b-instruct-g6-48xl-11k-tt-lora",
        ),
        ("sft", "fft"): (
            "llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft",
            "llama-0_7_0/verl-sft-llama-3-dot-1-8b-instruct-g6-48xl-5k-tt-fft",
        ),
        ("dpo", "lora"): (
            "llama/llmft_llama3_1_8b_instruct_seq4k_gpu_dpo",
            "llama-0_7_0/verl-dpo-llama-3-dot-1-8b-instruct-g5-48xl-6k-tt-lora",
        ),
        ("dpo", "fft"): (
            "llama/llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_fft",
            "llama-0_7_0/verl-dpo-llama-3-dot-1-8b-instruct-p5-48xl-12k-tt-fft",
        ),
    },
    "deepseek": {  # DeepSeek R1 Distill Qwen 7B
        ("sft", "lora"): (
            "deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_sft_lora",
            "deepseek-0_7_0/verl-sft-deepseek-r1-distilled-qwen-7b-p4d-24xl-20k-tt-lora",
        ),
        ("sft", "fft"): (
            "deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_sft_fft",
            "deepseek-0_7_0/verl-sft-deepseek-r1-distilled-qwen-7b-p5-48xl-21k-tt-fft",
        ),
        ("dpo", "lora"): (
            "deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_dpo",
            "deepseek-0_7_0/verl-dpo-deepseek-r1-distilled-qwen-7b-p5-48xl-22k-tt-lora",
        ),
        ("dpo", "fft"): (
            "deepseek/llmft_deepseek_r1_distilled_qwen_7b_seq4k_gpu_dpo_fft",
            "deepseek-0_7_0/verl-dpo-deepseek-r1-distilled-qwen-7b-p5-48xl-10k-tt-fft",
        ),
    },
    "qwen2.5": {  # Qwen2.5 7B (Instruct)
        ("sft", "lora"): (
            "qwen/llmft_qwen2_5_7b_seq4k_gpu_sft_lora",
            "qwen-0_7_0/verl-sft-qwen-2-5-7b-instruct-p4d-24xl-10k-tt-lora",
        ),
        ("sft", "fft"): (
            "qwen/llmft_qwen2_5_7b_instruct_seq4k_gpu_sft_fft",
            "qwen-0_7_0/verl-sft-qwen-2-5-7b-instruct-p4d-24xl-18k-tt-fft",
        ),
        ("dpo", "lora"): (
            "qwen/llmft_qwen2_5_7b_seq4k_gpu_dpo",
            "qwen-0_7_0/verl-dpo-qwen-2-5-7b-instruct-p5-48xl-22k-tt-lora",
        ),
        ("dpo", "fft"): (
            "qwen/llmft_qwen2_5_7b_instruct_seq4k_gpu_dpo_fft",
            "qwen-0_7_0/verl-dpo-qwen-2-5-7b-instruct-p5-48xl-16k-tt-fft",
        ),
    },
    "qwen3": {  # Qwen3 8B
        ("sft", "lora"): (
            "qwen/llmft_qwen3_8b_seq4k_gpu_sft_lora",
            "qwen-0_7_0/verl-sft-qwen-3-8b-p4d-24xl-10k-tt-lora",
        ),
        ("sft", "fft"): (
            "qwen/llmft_qwen3_8b_seq4k_gpu_sft_fft",
            "qwen-0_7_0/verl-sft-qwen-3-8b-p4d-24xl-8k-tt-fft",
        ),
        ("dpo", "lora"): (
            "qwen/llmft_qwen3_8b_seq4k_gpu_dpo",
            "qwen-0_7_0/verl-dpo-qwen-3-8b-p4d-24xl-10k-tt-lora",
        ),
        ("dpo", "fft"): (
            "qwen/llmft_qwen3_8b_seq4k_gpu_dpo_fft",
            "qwen-0_7_0/verl-dpo-qwen-3-8b-p5-48xl-9k-tt-fft",
        ),
    },
    "gpt_oss": {  # GPT-OSS 20B
        ("sft", "lora"): (
            "gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_sft_lora",
            "gpt_oss-0_7_0/verl-sft-gpt-oss-20b-p5-48xl-34k-tt-lora",
        ),
        ("sft", "fft"): (
            "gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_sft_fft",
            "gpt_oss-0_7_0/verl-sft-gpt-oss-20b-p5-48xl-16k-tt-fft",
        ),
        ("dpo", "lora"): (
            "gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_dpo",
            "gpt_oss-0_7_0/verl-dpo-gpt-oss-20b-p5-48xl-8k-tt-lora",
        ),
        ("dpo", "fft"): (
            "gpt_oss/llmft_gpt_oss_20b_seq4k_gpu_dpo_fft",
            "gpt_oss-0_7_0/verl-dpo-gpt-oss-20b-p5-48xl-12k-tt-fft",
        ),
    },
}

# llmft names a family's verl recipes do not advertise yet. Every verl SFT/DPO recipe outside
# Nemotron uses the default sft/sft_fft/dpo/dpo_fft templates, which name every llmft key except
# merge_weights, but a recipe advertises a name only if it carries the recipe key the template
# writes it to (VerlRecipeTemplateProcessor drops the rest). The Llama 3.x, DeepSeek, Qwen 2.5 and
# Qwen 3 recipes carry every key; the other families carry optim.clip_grad (gradient_clipping_threshold) and,
# for DPO, data.seed (seed) and model.lora_dropout (lora_dropout, LoRA only). To close a family's
# gap, add the missing keys to its verl recipes, then delete the names here.
_NOT_PORTED = {
    ("sft", "lora"): {
        "gradient_clipping",
        "logging_steps",
        # lora_dropout is not listed: llmft gpt-oss-20b pins it to 0.0 (min == max), so the
        # llmft launch.json does not advertise it either.
        "merge_weights",
        "seed",
        "temperature",
        "training_data_name",
        "validation_data_name",
    },
    ("sft", "fft"): {
        "gradient_clipping",
        "logging_steps",
        "max_response_length",
        "seed",
        "temperature",
        "training_data_name",
        "validation_data_name",
    },
    ("dpo", "lora"): {
        "gradient_clipping",
        "logging_steps",
        "merge_weights",
        "temperature",
        "training_data_name",
        "validation_data_name",
    },
    ("dpo", "fft"): {
        "gradient_clipping",
        "logging_steps",
        "max_response_length",
        "temperature",
        "training_data_name",
        "validation_data_name",
    },
}
EXPECTED_GAPS = {
    "llama": {
        ("sft", "lora"): {"merge_weights"},
        ("dpo", "lora"): {"merge_weights"},
    },
    "deepseek": {
        ("sft", "lora"): {"merge_weights"},
        ("dpo", "lora"): {"merge_weights"},
    },
    "qwen2.5": {
        ("sft", "lora"): {"merge_weights"},
        ("dpo", "lora"): {"merge_weights"},
    },
    "qwen3": {
        ("sft", "lora"): {"merge_weights"},
        ("dpo", "lora"): {"merge_weights"},
    },
    "gpt_oss": _NOT_PORTED,
}

# llmft names verl leaves out on purpose. They still count as gaps above, but they do not keep a
# recipe off JumpStart. merge_weights: verl LoRA recipes always merge the adapter
# (merge_lora_on_final_save: true, pinned by tests/test_verl_recipes_constraints.py) and verl
# does not support adapter-only output yet, so the switch is not offered.
DELIBERATE_OMISSIONS = {"merge_weights"}

# Names verl never advertises on a platform, whatever the recipe holds: the k8s launch.json adds
# an instance_type override for non-verl recipes only (launcher/nemo/stages.py, "Exclude
# instance_type overridable for verl templates").
PLATFORM_EXCLUSIONS = {"sm_jobs": set(), "k8s": {"instance_type"}}

CASES = [
    pytest.param(family, technique, peft, platform, id=f"{family}-{technique}-{peft}-{platform}")
    for family, coords in SAMPLE.items()
    for technique, peft in coords
    for platform in PLATFORMS
]


@pytest.fixture(scope="module")
def generator():
    return LaunchJsonGenerator()


def _override_params(generator, recipe_rel, platform, out_dir):
    """Generate the recipe's launch.json for ``platform`` and return its recipe_override_parameters."""
    recipe = generator.recipes_dir / "fine-tuning" / f"{recipe_rel}.yaml"
    assert recipe.is_file(), f"sample recipe not found (renamed?): {recipe_rel}"
    result = generator.generate_launch_json(recipe, platform, str(out_dir))
    assert result and result[1] == "success", f"{platform} launch.json generation failed for {recipe_rel}: {result}"
    return json.loads(Path(result[0]).read_text())["recipe_override_parameters"]


def _jumpstart_excluded(generator, recipe_rel):
    recipe_id = f"fine-tuning/{recipe_rel}.yaml"
    return any(pattern.search(recipe_id) for pattern in generator.exclusion_patterns)


@pytest.mark.parametrize("family,technique,peft,platform", CASES)
def test_verl_launch_json_advertises_llmft_overrides(family, technique, peft, platform, generator, tmp_path):
    llmft_rel, verl_rel = SAMPLE[family][(technique, peft)]
    llmft = _override_params(generator, llmft_rel, platform, tmp_path)
    verl = _override_params(generator, verl_rel, platform, tmp_path)

    missing = set(llmft) - set(verl)
    expected = EXPECTED_GAPS[family].get((technique, peft), set()) | PLATFORM_EXCLUSIONS[platform]
    assert missing == expected, (
        f"{verl_rel} vs {llmft_rel} ({platform}):\n"
        f"  new gap, llmft overrides the verl launch.json lacks: {sorted(missing - expected)}\n"
        f"  closed gap, listed but now advertised (update EXPECTED_GAPS): {sorted(expected - missing)}"
    )

    type_drift = {
        name: (llmft[name].get("type"), verl[name].get("type"))
        for name in sorted(set(llmft) & set(verl))
        if llmft[name].get("type") != verl[name].get("type")
    }
    assert not type_drift, f"{verl_rel} ({platform}): shared override types differ, (llmft, verl): {type_drift}"

    recipe_gap = missing - PLATFORM_EXCLUSIONS[platform] - DELIBERATE_OMISSIONS
    if recipe_gap:
        assert _jumpstart_excluded(generator, verl_rel), (
            f"{verl_rel} is released to JumpStart but lacks llmft override(s) {sorted(recipe_gap)}; "
            f"close the gap before releasing it"
        )
