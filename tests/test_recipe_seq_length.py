"""
Test that SequenceLengthFilter is present in all llmft SFT/DPO recipes,
and that sequence length metadata is consistent.
"""

import re

from hyperpod_recipes import list_recipes


def _get_llmft_recipes():
    """Get all llmft SFT and DPO recipes via Hydra-resolved config."""
    recipes = []
    for recipe in list_recipes():
        if "llmft" in recipe.name and "vision" not in recipe.name:
            recipes.append(recipe)
    return recipes


class TestRecipeSeqLength:
    """Test sequence length consistency and SequenceLengthFilter presence."""

    def test_llmft_sft_dpo_recipes_have_sequence_length_filter(self):
        """Verify all llmft SFT/DPO recipes have SequenceLengthFilter in their preprocessor chain."""
        missing_filter = []
        for recipe in _get_llmft_recipes():
            cfg = recipe.config
            preprocessor_cfgs = cfg.get("training_config", {}).get("datasets", {}).get("preprocessor_cfgs", [])
            if not preprocessor_cfgs:
                continue

            has_filter = any(
                (cfg_item.get("type") if hasattr(cfg_item, "get") else None) == "SequenceLengthFilter"
                for cfg_item in preprocessor_cfgs
            )
            if not has_filter:
                missing_filter.append(recipe.name)

        assert not missing_filter, (
            f"The following recipes are missing SequenceLengthFilter "
            f"(required to prevent OOM from long sequences):\n" + "\n".join(missing_filter)
        )

    def test_sequence_length_filter_not_less_than_max_len(self):
        """Verify that if SequenceLengthFilter has an explicit max_seq_length, it is >= max_len.

        An explicit max_seq_length can be larger than max_len (e.g., to allow some headroom),
        but should never be smaller — that would filter samples the recipe claims to support.
        """
        violations = []
        for recipe in _get_llmft_recipes():
            cfg = recipe.config
            max_len = cfg.get("training_config", {}).get("training_args", {}).get("max_len")
            preprocessor_cfgs = cfg.get("training_config", {}).get("datasets", {}).get("preprocessor_cfgs", [])
            if not preprocessor_cfgs or not max_len:
                continue

            for cfg_item in preprocessor_cfgs:
                if not hasattr(cfg_item, "get"):
                    continue
                if cfg_item.get("type") == "SequenceLengthFilter":
                    filter_max = cfg_item.get("max_seq_length")
                    if filter_max is not None and filter_max < max_len:
                        violations.append(
                            f"{recipe.name}: max_len={max_len}, "
                            f"SequenceLengthFilter.max_seq_length={filter_max} (too small!)"
                        )

        assert (
            not violations
        ), f"SequenceLengthFilter max_seq_length is smaller than training_args.max_len:\n" + "\n".join(violations)

    def test_recipe_filename_seq_length_matches_max_len(self):
        """Verify the sequence length in the recipe filename matches training_args.max_len.

        The filename (e.g., 'seq4k') is used by the template processor to set
        SequenceLength metadata on the hub. training_args.max_len is what actually
        controls filtering at runtime. These must be consistent to prevent hub
        metadata from misrepresenting the recipe's actual behavior.
        """
        mismatches = []
        for recipe in _get_llmft_recipes():
            match = re.search(r"seq(\d+)k", recipe.name)
            if not match:
                continue
            filename_seq_len = int(match.group(1)) * 1024

            cfg = recipe.config
            max_len = cfg.get("training_config", {}).get("training_args", {}).get("max_len")
            if max_len and max_len != filename_seq_len:
                mismatches.append(f"{recipe.name}: filename implies {filename_seq_len}, max_len={max_len}")

        assert not mismatches, (
            "Recipe filename seq length doesn't match training_args.max_len "
            "(hub metadata will be inconsistent with runtime behavior):\n" + "\n".join(mismatches)
        )
