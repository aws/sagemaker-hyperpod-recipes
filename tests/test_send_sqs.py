"""Tests for .github/scripts/send_sqs.py

send_sqs.py is a thin sender: it receives the fully-built per-model SQS message
array (produced by process_recipe_metadata.py, with hyperpodBranch injected by
the workflow) and enqueues one message per model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from send_sqs import main, send_messages

MODEL_A = "nova-textgeneration-lite-v2"
MODEL_B = "meta-textgeneration-llama-3-1-8b-instruct"


def _message(model_id, recipes, branch="release"):
    return {
        "modelId": model_id,
        "recipes": recipes,
        "options": {"subscription_model": False, "hyperpodBranch": branch},
    }


# ── send_messages ────────────────────────────────────────────────────────────


class TestSendMessages:
    def test_one_send_per_model(self):
        sqs = mock.Mock()
        sqs.send_message.return_value = {"MessageId": "abc"}
        messages = [
            _message(MODEL_A, [{"recipe": "a", "version": "1"}]),
            _message(MODEL_B, [{"recipe": "b", "version": "2"}]),
        ]
        failures = send_messages(sqs, "queue", messages)
        assert failures == 0
        assert sqs.send_message.call_count == 2
        first_body = json.loads(sqs.send_message.call_args_list[0].kwargs["MessageBody"])
        assert first_body["modelId"] == MODEL_A

    def test_exception_counts_as_failure(self):
        sqs = mock.Mock()
        sqs.send_message.side_effect = RuntimeError("boom")
        assert send_messages(sqs, "queue", [_message(MODEL_A, [{"recipe": "a", "version": "1"}])]) == 1

    def test_dry_run_does_not_send(self):
        sqs = mock.Mock()
        assert send_messages(sqs, "queue", [_message(MODEL_A, [{"recipe": "a", "version": "1"}])], dry_run=True) == 0
        sqs.send_message.assert_not_called()


# ── main() ───────────────────────────────────────────────────────────────────


class TestMain:
    def _args(self, metadata):
        return [
            "send_sqs.py",
            "--metadata",
            json.dumps(metadata),
            "--queue-url",
            "https://sqs.us-west-2.amazonaws.com/1/q",
        ]

    def test_sends_each_message_verbatim(self, monkeypatch):
        messages = [
            _message(MODEL_A, [{"recipe": "fine-tuning/nova/rft", "version": "1.0.19"}]),
            _message(MODEL_B, [{"recipe": "fine-tuning/llama/sft", "version": "2.0.0"}]),
        ]
        sqs = mock.Mock()
        sqs.send_message.return_value = {"MessageId": "abc"}
        monkeypatch.setattr("send_sqs.boto3.client", lambda *a, **k: sqs)
        monkeypatch.setattr(sys, "argv", self._args(messages))

        main()

        assert sqs.send_message.call_count == 2
        bodies = [json.loads(c.kwargs["MessageBody"]) for c in sqs.send_message.call_args_list]
        assert bodies == messages  # sent exactly as received

    def test_empty_array_sends_nothing(self, monkeypatch):
        sqs = mock.Mock()
        monkeypatch.setattr("send_sqs.boto3.client", lambda *a, **k: sqs)
        monkeypatch.setattr(sys, "argv", self._args([]))

        main()

        sqs.send_message.assert_not_called()

    def test_send_failure_exits_nonzero(self, monkeypatch):
        sqs = mock.Mock()
        sqs.send_message.side_effect = RuntimeError("boom")
        monkeypatch.setattr("send_sqs.boto3.client", lambda *a, **k: sqs)
        monkeypatch.setattr(sys, "argv", self._args([_message(MODEL_A, [{"recipe": "a", "version": "1"}])]))

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_non_array_metadata_exits_nonzero(self, monkeypatch):
        sqs = mock.Mock()
        monkeypatch.setattr("send_sqs.boto3.client", lambda *a, **k: sqs)
        monkeypatch.setattr(sys, "argv", ["send_sqs.py", "--metadata", '{"modelId": "x"}', "--queue-url", "q"])

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
