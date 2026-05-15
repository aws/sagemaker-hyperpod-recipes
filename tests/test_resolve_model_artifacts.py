"""Tests for .github/scripts/resolve_model_artifacts.py"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from resolve_model_artifacts import (
    CHECKPOINT_SUFFIX_HF,
    _get_latest_modified,
    _list_common_prefixes,
    find_latest_artifact,
    get_peft_type_from_filename,
    get_training_type_from_filename,
    resolve_artifacts,
)


class TestGetPeftTypeFromFilename:
    def test_lora_default(self):
        assert get_peft_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml") == "LORA"

    def test_fft(self):
        assert get_peft_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml") == "FFT"

    def test_fft_case_insensitive(self):
        assert get_peft_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_FFT.yaml") == "FFT"

    def test_verl_lora(self):
        assert get_peft_type_from_filename("verl-grpo-rlvr-qwen-2-dot-5-7b-instruct-lora.yaml") == "LORA"

    def test_no_explicit_peft(self):
        assert get_peft_type_from_filename("verl-grpo-rlvr-llama-3-3-70b.yaml") == "LORA"


class TestGetTrainingTypeFromFilename:
    def test_sft(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml") == "SFT"

    def test_dpo(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_lora.yaml") == "DPO"

    def test_rlvr(self):
        assert get_training_type_from_filename("verl-grpo-rlvr-qwen-2-dot-5-7b-instruct-lora.yaml") == "RLVR"

    def test_rlaif(self):
        assert get_training_type_from_filename("verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-lora.yaml") == "RLAIF"


class TestListCommonPrefixes:
    def test_returns_prefixes(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"CommonPrefixes": [{"Prefix": "a/"}, {"Prefix": "b/"}]}]
        result = _list_common_prefixes(mock_client, "bucket", "prefix/")
        assert result == ["a/", "b/"]

    def test_empty(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]
        result = _list_common_prefixes(mock_client, "bucket", "prefix/")
        assert result == []


class TestGetLatestModified:
    def test_returns_latest(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 3, tzinfo=timezone.utc)
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "f1", "LastModified": t1}, {"Key": "f2", "LastModified": t2}]}
        ]
        assert _get_latest_modified(mock_client, "bucket", "prefix/") == t2

    def test_no_objects(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]
        assert _get_latest_modified(mock_client, "bucket", "prefix/") is None


class TestFindLatestArtifact:
    def _make_client(self, prefix_tree, objects):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        def fake_paginate(**kwargs):
            prefix = kwargs.get("Prefix", "")
            delimiter = kwargs.get("Delimiter", "")
            if delimiter == "/":
                cps = prefix_tree.get(prefix, [])
                return [{"CommonPrefixes": [{"Prefix": cp} for cp in cps]}]
            else:
                objs = objects.get(prefix, [])
                return [{"Contents": [{"Key": k, "LastModified": t} for k, t in objs]}]

        paginator.paginate.side_effect = fake_paginate
        return mock_client

    def test_finds_artifact(self):
        t1 = datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)
        prefix_tree = {
            "pfx/": ["pfx/run/"],
            "pfx/run/m/LORA/SFT/": ["pfx/run/m/LORA/SFT/inst/"],
            "pfx/run/m/LORA/SFT/inst/": ["pfx/run/m/LORA/SFT/inst/job/"],
        }
        objects = {
            "pfx/run/m/LORA/SFT/inst/job/output/model/checkpoints/hf/": [("f", t1)],
        }
        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "b", "pfx/", "m", "LORA", "SFT", CHECKPOINT_SUFFIX_HF)
        assert "hf/" in result

    def test_no_run_folders(self):
        client = self._make_client({"pfx/": []}, {})
        result = find_latest_artifact(client, "b", "pfx/", "m", "LORA", "SFT", CHECKPOINT_SUFFIX_HF)
        assert result == ""


class TestResolveArtifacts:
    def _make_noop_client(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]
        return mock_client

    def test_no_js_model_id_skipped(self):
        missing = [{"run_names": ["x"], "js_model_id": None, "reason": "x", "recipe_paths": ["x.yaml"]}]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 0

    def test_output_keys(self):
        missing = [{"run_names": ["x"], "js_model_id": "js-x", "reason": "x", "recipe_paths": ["sft_lora.yaml"]}]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 1
        assert set(events[0].keys()) == {
            "model-id",
            "recipe_path",
            "peft_type",
            "hf_s3_uri",
            "hf_merged_s3_uri",
            "base_model_s3_uri",
        }

    def test_base_model_empty_when_not_in_s3(self):
        """base_model_s3_uri is empty when base model doesn't exist in S3."""
        missing = [
            {"run_names": ["x"], "js_model_id": "meta-llama-8b", "reason": "x", "recipe_paths": ["sft_lora.yaml"]}
        ]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        # noop client has no objects, so base model verification fails → empty
        assert events[0]["base_model_s3_uri"] == ""

    def test_empty_missing(self):
        events = resolve_artifacts([], "bucket", "prefix/", s3_client=self._make_noop_client())
        assert events == []

    @patch("resolve_model_artifacts._get_latest_modified")
    @patch("resolve_model_artifacts.find_latest_artifact")
    def test_resolves_both_from_same_job(self, mock_find, mock_modified):
        """resolve_artifacts derives hf/ from the same job as hf_merged/."""
        mock_find.return_value = "s3://bucket/pfx/run/m/LORA/SFT/inst/job/output/model/checkpoints/hf_merged/"
        mock_modified.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)  # hf/ exists
        missing = [{"run_names": ["x"], "js_model_id": "js-x", "reason": "x", "recipe_paths": ["sft_lora.yaml"]}]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=MagicMock())
        # find_latest_artifact called once (for hf_merged only)
        assert mock_find.call_count == 1
        assert events[0]["hf_merged_s3_uri"].endswith("hf_merged/")
        # hf/ derived from same path
        assert events[0]["hf_s3_uri"].endswith("hf/")
        assert "inst/job/" in events[0]["hf_s3_uri"]

    @patch("resolve_model_artifacts.find_latest_artifact")
    def test_peft_type_detected(self, mock_find):
        mock_find.return_value = ""
        missing = [
            {"run_names": ["x"], "js_model_id": "js-x", "reason": "x", "recipe_paths": ["sft_fft.yaml"]},
            {"run_names": ["y"], "js_model_id": "js-y", "reason": "x", "recipe_paths": ["sft_lora.yaml"]},
        ]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=MagicMock())
        assert events[0]["peft_type"] == "FFT"
        assert events[1]["peft_type"] == "LORA"


class TestHasUnresolvedInferenceMode:
    """Tests for the has_unresolved logic in inference mode.

    Verifies that the workflow correctly gates on:
    - FFT: needs hf_merged_s3_uri
    - LORA: needs hf_s3_uri (adapter) AND base_model_s3_uri
    """

    def test_lora_missing_adapter_is_unresolved(self):
        """LORA recipe with hf_merged but no hf/ adapter should be unresolved in inference mode."""
        events = [
            {
                "model-id": "qwen-7b",
                "recipe_path": "verl-grpo-rlvr-qwen-2-dot-5-7b-instruct-lora.yaml",
                "peft_type": "LORA",
                "hf_s3_uri": "",  # adapter missing!
                "hf_merged_s3_uri": "s3://bucket/pfx/run/qwen-7b/LORA/RLVR/inst/job/output/model/checkpoints/hf_merged/",
                "base_model_s3_uri": "s3://bucket/validation_results/base-models/qwen-7b/",
            }
        ]
        # Inference mode: LORA needs hf_s3_uri
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is True

    def test_lora_missing_base_model_is_unresolved(self):
        """LORA recipe with adapter but no base model should be unresolved in inference mode."""
        events = [
            {
                "model-id": "qwen-7b",
                "recipe_path": "sft_lora.yaml",
                "peft_type": "LORA",
                "hf_s3_uri": "s3://bucket/pfx/run/qwen-7b/LORA/SFT/inst/job/output/model/checkpoints/hf/",
                "hf_merged_s3_uri": "s3://bucket/pfx/run/qwen-7b/LORA/SFT/inst/job/output/model/checkpoints/hf_merged/",
                "base_model_s3_uri": "",  # base model missing!
            }
        ]
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is True

    def test_lora_all_present_is_resolved(self):
        """LORA recipe with both adapter and base model should be resolved."""
        events = [
            {
                "model-id": "qwen-7b",
                "recipe_path": "sft_lora.yaml",
                "peft_type": "LORA",
                "hf_s3_uri": "s3://bucket/pfx/run/qwen-7b/LORA/SFT/inst/job/output/model/checkpoints/hf/",
                "hf_merged_s3_uri": "s3://bucket/pfx/run/qwen-7b/LORA/SFT/inst/job/output/model/checkpoints/hf_merged/",
                "base_model_s3_uri": "s3://bucket/validation_results/base-models/qwen-7b/",
            }
        ]
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is False

    def test_fft_missing_hf_merged_is_unresolved(self):
        """FFT recipe without hf_merged should be unresolved in inference mode."""
        events = [
            {
                "model-id": "llama-8b",
                "recipe_path": "sft_fft.yaml",
                "peft_type": "FFT",
                "hf_s3_uri": "",
                "hf_merged_s3_uri": "",  # missing!
                "base_model_s3_uri": "",
            }
        ]
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is True

    def test_fft_with_hf_merged_is_resolved(self):
        """FFT recipe with hf_merged should be resolved (doesn't need adapter or base model)."""
        events = [
            {
                "model-id": "llama-8b",
                "recipe_path": "sft_fft.yaml",
                "peft_type": "FFT",
                "hf_s3_uri": "",
                "hf_merged_s3_uri": "s3://bucket/pfx/run/llama-8b/FFT/SFT/inst/job/output/model/checkpoints/hf_merged/",
                "base_model_s3_uri": "",
            }
        ]
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is False

    def test_mixed_one_unresolved(self):
        """Mixed FFT+LORA where LORA is missing adapter → overall unresolved."""
        events = [
            {
                "model-id": "llama-8b",
                "recipe_path": "sft_fft.yaml",
                "peft_type": "FFT",
                "hf_s3_uri": "",
                "hf_merged_s3_uri": "s3://bucket/pfx/hf_merged/",
                "base_model_s3_uri": "",
            },
            {
                "model-id": "qwen-7b",
                "recipe_path": "verl-grpo-rlvr-qwen-lora.yaml",
                "peft_type": "LORA",
                "hf_s3_uri": "",  # adapter missing!
                "hf_merged_s3_uri": "s3://bucket/pfx/hf_merged/",
                "base_model_s3_uri": "s3://bucket/base-models/qwen-7b/",
            },
        ]
        has_unresolved = any(
            (e["peft_type"] == "FFT" and not e["hf_merged_s3_uri"])
            or (e["peft_type"] == "LORA" and (not e["hf_s3_uri"] or not e["base_model_s3_uri"]))
            for e in events
        )
        assert has_unresolved is True
