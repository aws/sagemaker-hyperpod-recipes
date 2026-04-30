"""Tests for .github/scripts/resolve_model_artifacts.py"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add .github/scripts to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from resolve_model_artifacts import (
    DEFAULT_CHECKPOINT_SUFFIX,
    _get_latest_modified,
    _list_common_prefixes,
    find_latest_artifact,
    get_peft_type_from_filename,
    get_training_type_from_filename,
    resolve_artifacts,
)

# ── get_peft_type_from_filename ──────────────────────────────────────────────


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
        """When neither lora nor fft appears, default to LORA."""
        assert get_peft_type_from_filename("verl-grpo-rlvr-llama-3-3-70b.yaml") == "LORA"

    def test_path_with_directory(self):
        assert (
            get_peft_type_from_filename("fine-tuning/llama/llmft_llama3_1_8b_instruct_seq4k_gpu_sft_fft.yaml") == "FFT"
        )


# ── get_training_type_from_filename ──────────────────────────────────────────


class TestGetTrainingTypeFromFilename:
    def test_sft_from_sft(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_sft_lora.yaml") == "SFT"

    def test_sft_from_lora(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_lora.yaml") == "SFT"

    def test_sft_from_fft(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_fft.yaml") == "SFT"

    def test_dpo(self):
        assert get_training_type_from_filename("llmft_llama3_1_8b_instruct_seq4k_gpu_dpo_lora.yaml") == "DPO"

    def test_rlvr(self):
        assert get_training_type_from_filename("verl-grpo-rlvr-qwen-2-dot-5-7b-instruct-lora.yaml") == "RLVR"

    def test_rlaif(self):
        assert get_training_type_from_filename("verl-grpo-rlaif-qwen-2-dot-5-7b-instruct-lora.yaml") == "RLAIF"

    def test_rlvr_takes_priority_over_dpo(self):
        """RLVR is checked before DPO."""
        assert get_training_type_from_filename("some_recipe_rlvr_dpo.yaml") == "RLVR"

    def test_rlaif_takes_priority_over_dpo(self):
        assert get_training_type_from_filename("some_recipe_rlaif_dpo.yaml") == "RLAIF"


# ── _list_common_prefixes (mock) ─────────────────────────────────────────────


class TestListCommonPrefixes:
    def test_returns_prefixes(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "validation_results/2026-01-01_run_0/"},
                    {"Prefix": "validation_results/2026-02-03_run_0/"},
                ]
            }
        ]

        result = _list_common_prefixes(mock_client, "my-bucket", "validation_results/")
        assert result == [
            "validation_results/2026-01-01_run_0/",
            "validation_results/2026-02-03_run_0/",
        ]

    def test_empty_bucket(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]

        result = _list_common_prefixes(mock_client, "my-bucket", "validation_results/")
        assert result == []

    def test_max_prefixes_keeps_newest(self):
        """When more prefixes exist than max_prefixes, the newest (trailing) entries are kept."""
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        # Return 10 prefixes in a single page (S3 returns lex-ascending)
        paginator.paginate.return_value = [
            {"CommonPrefixes": [{"Prefix": f"validation_results/2026-01-{i:02d}_run_0/"} for i in range(10)]}
        ]

        result = _list_common_prefixes(mock_client, "my-bucket", "validation_results/", max_prefixes=3)
        assert len(result) == 3
        # Should keep the NEWEST (last) 3 entries, not the oldest (first) 3
        assert result == [
            "validation_results/2026-01-07_run_0/",
            "validation_results/2026-01-08_run_0/",
            "validation_results/2026-01-09_run_0/",
        ]

    def test_max_prefixes_logs_warning(self, caplog):
        import logging

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"CommonPrefixes": [{"Prefix": f"validation_results/run_{i}/"} for i in range(5)]}
        ]

        with caplog.at_level(logging.WARNING, logger="resolve_model_artifacts"):
            result = _list_common_prefixes(mock_client, "my-bucket", "validation_results/", max_prefixes=2)
        assert len(result) == 2
        assert "Found 5 prefixes" in caplog.text
        assert "keeping newest 2" in caplog.text
        assert "s3://my-bucket/validation_results/" in caplog.text

    def test_no_warning_when_under_limit(self, caplog):
        import logging

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "CommonPrefixes": [
                    {"Prefix": "validation_results/run_0/"},
                    {"Prefix": "validation_results/run_1/"},
                ]
            }
        ]

        with caplog.at_level(logging.WARNING, logger="resolve_model_artifacts"):
            result = _list_common_prefixes(mock_client, "my-bucket", "validation_results/", max_prefixes=1000)
        assert len(result) == 2
        assert "max_prefixes" not in caplog.text


# ── _get_latest_modified (mock) ──────────────────────────────────────────────


class TestGetLatestModified:
    def test_returns_latest(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 3, tzinfo=timezone.utc)

        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "some/path/file1.bin", "LastModified": t1},
                    {"Key": "some/path/file2.bin", "LastModified": t2},
                ]
            }
        ]

        result = _get_latest_modified(mock_client, "my-bucket", "some/path/")
        assert result == t2

    def test_no_objects(self):
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]

        result = _get_latest_modified(mock_client, "my-bucket", "empty/prefix/")
        assert result is None


# ── find_latest_artifact (mock) ──────────────────────────────────────────────


class TestFindLatestArtifact:
    def _make_client(self, prefix_tree: dict, objects: dict):
        """Build a mock S3 client.

        Args:
            prefix_tree: mapping from prefix → list of common-prefix strings
            objects: mapping from prefix → list of (key, LastModified) tuples
        """
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        def fake_paginate(**kwargs):
            prefix = kwargs.get("Prefix", "")
            delimiter = kwargs.get("Delimiter", "")

            if delimiter == "/":
                # list_common_prefixes path
                cps = prefix_tree.get(prefix, [])
                return [{"CommonPrefixes": [{"Prefix": cp} for cp in cps]}]
            else:
                # list_objects path
                objs = objects.get(prefix, [])
                return [{"Contents": [{"Key": k, "LastModified": t} for k, t in objs]}]

        paginator.paginate.side_effect = fake_paginate
        return mock_client

    def test_finds_artifact(self):
        t1 = datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": ["validation_results/2026-02-03_run_0/"],
            "validation_results/2026-02-03_run_0/my-model/LORA/SFT/": [
                "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/": [
                "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/job-123/"
            ],
        }
        objects = {
            "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/job-123/output/model/checkpoints/hf_merged/": [
                (
                    "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/job-123/output/model/checkpoints/hf_merged/config.json",
                    t1,
                ),
            ]
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "my-model", "LORA", "SFT")
        assert (
            result
            == "s3://my-bucket/validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/job-123/output/model/checkpoints/hf_merged/"
        )

    def test_no_run_folders(self):
        client = self._make_client({"validation_results/": []}, {})
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "my-model", "LORA", "SFT")
        assert result == ""

    def test_no_matching_model(self):
        prefix_tree = {
            "validation_results/": ["validation_results/2026-02-03_run_0/"],
            "validation_results/2026-02-03_run_0/my-model/LORA/SFT/": [],
        }
        client = self._make_client(prefix_tree, {})
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "my-model", "LORA", "SFT")
        assert result == ""

    def test_picks_latest_across_runs(self):
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 3, 1, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": [
                "validation_results/2026-01-01_run_0/",
                "validation_results/2026-03-01_run_0/",
            ],
            # Run 1
            "validation_results/2026-01-01_run_0/m/LORA/SFT/": [
                "validation_results/2026-01-01_run_0/m/LORA/SFT/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-01_run_0/m/LORA/SFT/ml.p4d.24xlarge/": [
                "validation_results/2026-01-01_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-old/"
            ],
            # Run 2
            "validation_results/2026-03-01_run_0/m/LORA/SFT/": [
                "validation_results/2026-03-01_run_0/m/LORA/SFT/ml.p5.48xlarge/"
            ],
            "validation_results/2026-03-01_run_0/m/LORA/SFT/ml.p5.48xlarge/": [
                "validation_results/2026-03-01_run_0/m/LORA/SFT/ml.p5.48xlarge/job-new/"
            ],
        }
        objects = {
            "validation_results/2026-01-01_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-old/output/model/checkpoints/hf_merged/": [
                ("...config.json", t1),
            ],
            "validation_results/2026-03-01_run_0/m/LORA/SFT/ml.p5.48xlarge/job-new/output/model/checkpoints/hf_merged/": [
                ("...config.json", t2),
            ],
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "m", "LORA", "SFT")
        assert "2026-03-01_run_0" in result
        assert "job-new" in result

    def test_early_stop_skips_older_runs(self):
        """Once the newest matching run is found, older runs are not queried."""
        t1 = datetime(2026, 1, 20, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": [
                "validation_results/2026-01-10_run_0/",
                "validation_results/2026-01-20_run_0/",
            ],
            # Newest run has the model
            "validation_results/2026-01-20_run_0/m/FFT/DPO/": [
                "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/": [
                "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-1/"
            ],
            # Older run also has the model — should NOT be queried
            "validation_results/2026-01-10_run_0/m/FFT/DPO/": [
                "validation_results/2026-01-10_run_0/m/FFT/DPO/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-10_run_0/m/FFT/DPO/ml.p4d.24xlarge/": [
                "validation_results/2026-01-10_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-old/"
            ],
        }
        objects = {
            "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-1/output/model/checkpoints/hf_merged/": [
                ("config.json", t1),
            ],
            # Old run objects exist but should never be checked
            "validation_results/2026-01-10_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-old/output/model/checkpoints/hf_merged/": [
                ("config.json", datetime(2026, 1, 10, tzinfo=timezone.utc)),
            ],
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "m", "FFT", "DPO")
        assert "2026-01-20_run_0" in result
        assert "job-1" in result
        # Older run should not appear
        assert "2026-01-10" not in result

    def test_multiple_runs_same_day(self):
        """Multiple runs on the same day (_run_0 vs _run_1) — picks the newest."""
        t1 = datetime(2026, 1, 20, 12, 0, 0, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": [
                "validation_results/2026-01-20_run_0/",
                "validation_results/2026-01-20_run_1/",
            ],
            # _run_0 does NOT have the model
            "validation_results/2026-01-20_run_0/huggingface-reasoning-qwen3-06b/FFT/DPO/": [],
            # _run_1 DOES have the model
            "validation_results/2026-01-20_run_1/huggingface-reasoning-qwen3-06b/FFT/DPO/": [
                "validation_results/2026-01-20_run_1/huggingface-reasoning-qwen3-06b/FFT/DPO/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-20_run_1/huggingface-reasoning-qwen3-06b/FFT/DPO/ml.p4d.24xlarge/": [
                "validation_results/2026-01-20_run_1/huggingface-reasoning-qwen3-06b/FFT/DPO/ml.p4d.24xlarge/qwen-3-0d6b-c30lg-2026-01-30-08-00-31-918/"
            ],
        }
        objects = {
            "validation_results/2026-01-20_run_1/huggingface-reasoning-qwen3-06b/FFT/DPO/ml.p4d.24xlarge/qwen-3-0d6b-c30lg-2026-01-30-08-00-31-918/output/model/checkpoints/hf_merged/": [
                ("config.json", t1),
            ],
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(
            client,
            "hyperpod-recipes-validation-artifacts",
            "validation_results/",
            "huggingface-reasoning-qwen3-06b",
            "FFT",
            "DPO",
        )
        assert (
            result == "s3://hyperpod-recipes-validation-artifacts/validation_results/2026-01-20_run_1/"
            "huggingface-reasoning-qwen3-06b/FFT/DPO/ml.p4d.24xlarge/"
            "qwen-3-0d6b-c30lg-2026-01-30-08-00-31-918/output/model/checkpoints/hf_merged/"
        )

    def test_newest_run_empty_hf_merged_falls_through(self):
        """If the newest matching run has the prefix structure but no hf_merged objects,
        fall through to the next-newest run."""
        t_old = datetime(2026, 1, 10, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": [
                "validation_results/2026-01-10_run_0/",
                "validation_results/2026-01-20_run_0/",
            ],
            # Newest run has path structure but no objects
            "validation_results/2026-01-20_run_0/m/LORA/SFT/": [
                "validation_results/2026-01-20_run_0/m/LORA/SFT/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-20_run_0/m/LORA/SFT/ml.p4d.24xlarge/": [
                "validation_results/2026-01-20_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-empty/"
            ],
            # Older run has actual objects
            "validation_results/2026-01-10_run_0/m/LORA/SFT/": [
                "validation_results/2026-01-10_run_0/m/LORA/SFT/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-10_run_0/m/LORA/SFT/ml.p4d.24xlarge/": [
                "validation_results/2026-01-10_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-good/"
            ],
        }
        objects = {
            # Newest run: hf_merged/ is empty (no objects)
            "validation_results/2026-01-20_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-empty/output/model/checkpoints/hf_merged/": [],
            # Older run: has actual objects
            "validation_results/2026-01-10_run_0/m/LORA/SFT/ml.p4d.24xlarge/job-good/output/model/checkpoints/hf_merged/": [
                ("config.json", t_old),
            ],
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "m", "LORA", "SFT")
        assert "2026-01-10_run_0" in result
        assert "job-good" in result

    def test_multiple_jobs_picks_latest(self):
        """Multiple jobs under the same instance type — picks the one with latest LastModified."""
        t_old = datetime(2026, 1, 15, tzinfo=timezone.utc)
        t_new = datetime(2026, 1, 30, tzinfo=timezone.utc)

        prefix_tree = {
            "validation_results/": ["validation_results/2026-01-20_run_0/"],
            "validation_results/2026-01-20_run_0/m/FFT/DPO/": [
                "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/"
            ],
            "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/": [
                "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-old/",
                "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-new/",
            ],
        }
        objects = {
            "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-old/output/model/checkpoints/hf_merged/": [
                ("config.json", t_old),
            ],
            "validation_results/2026-01-20_run_0/m/FFT/DPO/ml.p4d.24xlarge/job-new/output/model/checkpoints/hf_merged/": [
                ("config.json", t_new),
            ],
        }

        client = self._make_client(prefix_tree, objects)
        result = find_latest_artifact(client, "my-bucket", "validation_results/", "m", "FFT", "DPO")
        assert "job-new" in result
        assert "job-old" not in result


# ── resolve_artifacts (integration with mocks) ──────────────────────────────


class TestResolveArtifacts:
    def _make_noop_client(self):
        """S3 client that returns nothing."""
        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]
        return mock_client

    def test_no_js_model_id_skipped(self):
        """Entries without js_model_id are skipped (handled by workflow gate)."""
        missing = [
            {
                "run_names": ["new-model"],
                "js_model_id": None,
                "reason": "run.name not found in jumpstart_model-id_map.json",
                "recipe_paths": ["recipes_collection/recipes/fine-tuning/llama/new_recipe.yaml"],
            }
        ]

        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 0

    def test_with_js_model_id_no_artifact(self):
        """Entries with js_model_id but no S3 artifact get empty s3Uri."""
        missing = [
            {
                "run_names": ["qwen-model"],
                "js_model_id": "huggingface-llm-qwen2-5-7b",
                "reason": "JumpStart model ID not in eval js_model_name_instance_mapping",
                "recipe_paths": ["recipes_collection/recipes/fine-tuning/qwen/sft_lora.yaml"],
            }
        ]

        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 1
        assert events[0]["model-id"] == "huggingface-llm-qwen2-5-7b"
        assert events[0]["s3Uri"] == ""
        assert events[0]["recipe_path"] == "recipes_collection/recipes/fine-tuning/qwen/sft_lora.yaml"

    def test_s3_cache_for_same_model_peft_training(self):
        """Two recipe paths for same model/peft/training produce two entries but only one S3 lookup."""
        missing = [
            {
                "run_names": ["llama-8b", "llama-8b-v2"],
                "js_model_id": "meta-llama-8b",
                "reason": "missing",
                "recipe_paths": [
                    "fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml",
                    "fine-tuning/llama/verl_llama3_1_8b_sft_lora.yaml",
                ],
            }
        ]

        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        # One entry per recipe path
        assert len(events) == 2
        assert events[0]["model-id"] == "meta-llama-8b"
        assert events[1]["model-id"] == "meta-llama-8b"
        # Each has its own recipe_path
        assert events[0]["recipe_path"] == "fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml"
        assert events[1]["recipe_path"] == "fine-tuning/llama/verl_llama3_1_8b_sft_lora.yaml"
        # Both share the same s3Uri (cached)
        assert events[0]["s3Uri"] == events[1]["s3Uri"]

    def test_different_training_types_produce_separate_events(self):
        """SFT and DPO recipes for the same model produce separate events."""
        missing = [
            {
                "run_names": ["llama-8b"],
                "js_model_id": "meta-llama-8b",
                "reason": "missing",
                "recipe_paths": [
                    "fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml",
                    "fine-tuning/llama/llmft_llama3_1_8b_dpo_lora.yaml",
                ],
            }
        ]

        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 2
        model_ids = {e["model-id"] for e in events}
        assert model_ids == {"meta-llama-8b"}

    def test_output_contains_expected_keys(self):
        """Output dicts should contain model-id, s3Uri, and recipe_path keys."""
        missing = [
            {
                "run_names": ["model-x"],
                "js_model_id": "js-model-x",
                "reason": "missing",
                "recipe_paths": ["fine-tuning/x/sft_lora.yaml"],
            }
        ]
        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=self._make_noop_client())
        assert len(events) == 1
        assert set(events[0].keys()) == {"model-id", "s3Uri", "recipe_path"}
        assert events[0]["recipe_path"] == "fine-tuning/x/sft_lora.yaml"

    def test_empty_missing_recipes(self):
        events = resolve_artifacts([], "bucket", "prefix/", s3_client=self._make_noop_client())
        assert events == []

    @patch("resolve_model_artifacts.find_latest_artifact")
    def test_with_resolved_artifact(self, mock_find):
        """When S3 has an artifact, s3Uri is populated."""
        mock_find.return_value = (
            "s3://bucket/validation_results/run/model/LORA/SFT/ml.p4d/job/output/model/checkpoints/hf_merged/"
        )

        missing = [
            {
                "run_names": ["llama-8b"],
                "js_model_id": "meta-llama-8b",
                "reason": "missing",
                "recipe_paths": ["fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml"],
            }
        ]

        events = resolve_artifacts(missing, "bucket", "prefix/", s3_client=MagicMock())
        assert len(events) == 1
        assert events[0]["s3Uri"].endswith("hf_merged/")
        assert events[0]["model-id"] == "meta-llama-8b"
        assert events[0]["recipe_path"] == "fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml"


# ── checkpoint_suffix parameter ──────────────────────────────────────────────


class TestCheckpointSuffix:
    """Tests for the configurable checkpoint_suffix parameter."""

    def test_default_suffix_is_hf_merged(self):
        """DEFAULT_CHECKPOINT_SUFFIX should be hf_merged/ for backward compatibility."""
        assert DEFAULT_CHECKPOINT_SUFFIX == "output/model/checkpoints/hf_merged/"

    def test_find_latest_artifact_with_custom_suffix(self):
        """find_latest_artifact uses the provided checkpoint_suffix."""
        t1 = datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)
        custom_suffix = "output/model/checkpoints/hf/"

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        def fake_paginate(**kwargs):
            prefix = kwargs.get("Prefix", "")
            delimiter = kwargs.get("Delimiter", "")

            if delimiter == "/":
                tree = {
                    "validation_results/": ["validation_results/2026-02-03_run_0/"],
                    "validation_results/2026-02-03_run_0/my-model/LORA/SFT/": [
                        "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/"
                    ],
                    "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/": [
                        "validation_results/2026-02-03_run_0/my-model/LORA/SFT/ml.p4d.24xlarge/job-123/"
                    ],
                }
                cps = tree.get(prefix, [])
                return [{"CommonPrefixes": [{"Prefix": cp} for cp in cps]}]
            else:
                # Only match the custom suffix path, NOT hf_merged
                hf_path = (
                    "validation_results/2026-02-03_run_0/my-model/LORA/SFT/"
                    "ml.p4d.24xlarge/job-123/output/model/checkpoints/hf/"
                )
                if prefix == hf_path:
                    return [{"Contents": [{"Key": f"{hf_path}config.json", "LastModified": t1}]}]
                return [{}]

        paginator.paginate.side_effect = fake_paginate

        result = find_latest_artifact(
            mock_client,
            "my-bucket",
            "validation_results/",
            "my-model",
            "LORA",
            "SFT",
            checkpoint_suffix=custom_suffix,
        )
        assert "output/model/checkpoints/hf/" in result
        assert "hf_merged" not in result

    def test_find_latest_artifact_default_suffix(self):
        """find_latest_artifact defaults to hf_merged/ when no suffix provided."""
        t1 = datetime(2026, 2, 3, 12, 0, 0, tzinfo=timezone.utc)

        mock_client = MagicMock()
        paginator = MagicMock()
        mock_client.get_paginator.return_value = paginator

        def fake_paginate(**kwargs):
            prefix = kwargs.get("Prefix", "")
            delimiter = kwargs.get("Delimiter", "")

            if delimiter == "/":
                tree = {
                    "pfx/": ["pfx/2026-02-03_run_0/"],
                    "pfx/2026-02-03_run_0/m/LORA/SFT/": ["pfx/2026-02-03_run_0/m/LORA/SFT/ml.p4d/"],
                    "pfx/2026-02-03_run_0/m/LORA/SFT/ml.p4d/": ["pfx/2026-02-03_run_0/m/LORA/SFT/ml.p4d/job/"],
                }
                cps = tree.get(prefix, [])
                return [{"CommonPrefixes": [{"Prefix": cp} for cp in cps]}]
            else:
                merged_path = "pfx/2026-02-03_run_0/m/LORA/SFT/ml.p4d/job/output/model/checkpoints/hf_merged/"
                if prefix == merged_path:
                    return [{"Contents": [{"Key": f"{merged_path}config.json", "LastModified": t1}]}]
                return [{}]

        paginator.paginate.side_effect = fake_paginate

        # No checkpoint_suffix arg — should default to hf_merged/
        result = find_latest_artifact(
            mock_client,
            "my-bucket",
            "pfx/",
            "m",
            "LORA",
            "SFT",
        )
        assert "hf_merged/" in result

    @patch("resolve_model_artifacts.find_latest_artifact")
    def test_resolve_artifacts_passes_suffix(self, mock_find):
        """resolve_artifacts forwards checkpoint_suffix to find_latest_artifact."""
        mock_find.return_value = "s3://bucket/path/hf/"

        missing = [
            {
                "run_names": ["llama-8b"],
                "js_model_id": "meta-llama-8b",
                "reason": "missing",
                "recipe_paths": ["fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml"],
            }
        ]

        custom_suffix = "output/model/checkpoints/hf/"
        events = resolve_artifacts(
            missing,
            "bucket",
            "prefix/",
            s3_client=MagicMock(),
            checkpoint_suffix=custom_suffix,
        )

        # Verify find_latest_artifact was called with the custom suffix
        mock_find.assert_called_once()
        call_kwargs = mock_find.call_args
        assert call_kwargs.kwargs["checkpoint_suffix"] == custom_suffix

        assert len(events) == 1
        assert events[0]["s3Uri"] == "s3://bucket/path/hf/"

    @patch("resolve_model_artifacts.find_latest_artifact")
    def test_resolve_artifacts_default_suffix(self, mock_find):
        """resolve_artifacts defaults to DEFAULT_CHECKPOINT_SUFFIX."""
        mock_find.return_value = ""

        missing = [
            {
                "run_names": ["llama-8b"],
                "js_model_id": "meta-llama-8b",
                "reason": "missing",
                "recipe_paths": ["fine-tuning/llama/llmft_llama3_1_8b_sft_lora.yaml"],
            }
        ]

        resolve_artifacts(missing, "bucket", "prefix/", s3_client=MagicMock())

        # Verify find_latest_artifact was called with the default suffix
        mock_find.assert_called_once()
        call_kwargs = mock_find.call_args
        assert call_kwargs.kwargs["checkpoint_suffix"] == DEFAULT_CHECKPOINT_SUFFIX
