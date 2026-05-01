"""Tests for .github/scripts/post_pr_comment.py"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add .github/scripts to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

import pytest
import requests
from post_pr_comment import (
    DEFAULT_MARKER,
    _request_with_retry,
    create_comment,
    find_existing_comment,
    post_or_update_comment,
    update_comment,
)

# ── find_existing_comment ────────────────────────────────────────────────────


class TestFindExistingComment:
    def test_finds_bot_comment_with_marker(self):
        """Should find a comment from github-actions[bot] containing the marker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 111,
                "user": {"login": "some-user"},
                "body": "unrelated comment",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": 222,
                "user": {"login": "github-actions[bot]"},
                "body": f"{DEFAULT_MARKER}\n## Eval Validation",
                "created_at": "2026-01-02T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", return_value=mock_response) as mock_get:
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result == 222
        mock_get.assert_called_once()

    def test_returns_none_when_no_match(self):
        """Should return None when no comment has the marker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 111,
                "user": {"login": "github-actions[bot]"},
                "body": "different comment without marker",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", return_value=mock_response):
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result is None

    def test_returns_none_for_empty_comments(self):
        """Should return None when PR has no comments."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", return_value=mock_response):
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result is None

    def test_ignores_non_bot_comments_with_marker(self):
        """Should ignore comments from non-bot users even if they contain the marker."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 111,
                "user": {"login": "human-user"},
                "body": f"{DEFAULT_MARKER}\nmanual comment",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", return_value=mock_response):
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result is None

    def test_selects_most_recent_when_multiple_match(self):
        """Should return the most recently created comment when multiple match."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": 111,
                "user": {"login": "github-actions[bot]"},
                "body": f"{DEFAULT_MARKER}\nold comment",
                "created_at": "2026-01-01T00:00:00Z",
            },
            {
                "id": 222,
                "user": {"login": "github-actions[bot]"},
                "body": f"{DEFAULT_MARKER}\nnew comment",
                "created_at": "2026-01-02T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", return_value=mock_response):
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result == 222

    def test_paginates_through_multiple_pages(self):
        """Should paginate when first page is full (100 comments)."""
        # Page 1: 100 comments, none matching
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = [
            {"id": i, "user": {"login": "someone"}, "body": "no marker", "created_at": "2026-01-01T00:00:00Z"}
            for i in range(100)
        ]
        page1.raise_for_status = MagicMock()

        # Page 2: matching comment
        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = [
            {
                "id": 999,
                "user": {"login": "github-actions[bot]"},
                "body": f"{DEFAULT_MARKER}\nfound it",
                "created_at": "2026-01-02T00:00:00Z",
            },
        ]
        page2.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", side_effect=[page1, page2]):
            result = find_existing_comment("fake-token", "owner/repo", 42, DEFAULT_MARKER)

        assert result == 999


# ── update_comment ───────────────────────────────────────────────────────────


class TestUpdateComment:
    def test_calls_patch_with_correct_params(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.patch", return_value=mock_response) as mock_patch:
            update_comment("fake-token", "owner/repo", 123, "new body")

        mock_patch.assert_called_once()
        call_args = mock_patch.call_args
        assert "/repos/owner/repo/issues/comments/123" in call_args[0][0]
        assert call_args[1]["json"] == {"body": "new body"}


# ── create_comment ───────────────────────────────────────────────────────────


class TestCreateComment:
    def test_calls_post_with_correct_params(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 456}
        mock_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.post", return_value=mock_response) as mock_post:
            result = create_comment("fake-token", "owner/repo", 42, "comment body")

        assert result == 456
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/repos/owner/repo/issues/42/comments" in call_args[0][0]
        assert call_args[1]["json"] == {"body": "comment body"}


# ── post_or_update_comment ───────────────────────────────────────────────────


class TestPostOrUpdateComment:
    @patch("post_pr_comment.create_comment", return_value=789)
    @patch("post_pr_comment.find_existing_comment", return_value=None)
    def test_creates_when_no_existing(self, mock_find, mock_create):
        """Should create a new comment when none exists."""
        result = post_or_update_comment("token", "owner/repo", 42, "body")
        assert result == 789
        mock_find.assert_called_once()
        mock_create.assert_called_once_with("token", "owner/repo", 42, "body")

    @patch("post_pr_comment.update_comment")
    @patch("post_pr_comment.find_existing_comment", return_value=123)
    def test_updates_when_existing(self, mock_find, mock_update):
        """Should update the existing comment when one is found."""
        result = post_or_update_comment("token", "owner/repo", 42, "new body")
        assert result == 123
        mock_find.assert_called_once()
        mock_update.assert_called_once_with("token", "owner/repo", 123, "new body")

    @patch("post_pr_comment.update_comment")
    @patch("post_pr_comment.find_existing_comment", return_value=123)
    def test_uses_custom_marker(self, mock_find, mock_update):
        """Should pass custom marker to find_existing_comment."""
        custom_marker = "<!-- custom-marker -->"
        post_or_update_comment("token", "owner/repo", 42, "body", marker=custom_marker)
        mock_find.assert_called_once_with("token", "owner/repo", 42, custom_marker)


# ── _request_with_retry ─────────────────────────────────────────────────────


class TestRequestWithRetry:
    @patch("post_pr_comment.time.sleep")
    def test_retries_on_502_then_succeeds(self, mock_sleep):
        """Should retry on 502 and return the successful response."""
        fail_response = MagicMock()
        fail_response.status_code = 502
        fail_response.headers = {}

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", side_effect=[fail_response, ok_response]):
            result = _request_with_retry("GET", "https://api.example.com/test", max_retries=3)

        assert result == ok_response
        mock_sleep.assert_called_once()

    @patch("post_pr_comment.time.sleep")
    def test_retries_on_429_with_retry_after_header(self, mock_sleep):
        """Should use Retry-After header value for backoff on 429."""
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "5"}

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.get", side_effect=[rate_limited, ok_response]):
            result = _request_with_retry("GET", "https://api.example.com/test", max_retries=3)

        assert result == ok_response
        # Should sleep for the Retry-After value (5 seconds)
        mock_sleep.assert_called_once_with(5.0)

    @patch("post_pr_comment.time.sleep")
    def test_raises_after_max_retries_exhausted(self, mock_sleep):
        """Should raise HTTPError after all retries are exhausted."""
        fail_response = MagicMock()
        fail_response.status_code = 503
        fail_response.headers = {}
        fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError("503 Server Error")

        # max_retries=2 means: initial attempt + 2 retries = 3 total calls
        with patch("post_pr_comment.requests.get", return_value=fail_response):
            with pytest.raises(requests.exceptions.HTTPError):
                _request_with_retry("GET", "https://api.example.com/test", max_retries=2)

        # Should have slept twice (between attempt 0→1 and 1→2)
        assert mock_sleep.call_count == 2

    @patch("post_pr_comment.time.sleep")
    def test_no_retry_on_non_retryable_error(self, mock_sleep):
        """Should not retry on non-retryable errors like 404."""
        not_found = MagicMock()
        not_found.status_code = 404
        not_found.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with patch("post_pr_comment.requests.get", return_value=not_found):
            with pytest.raises(requests.exceptions.HTTPError):
                _request_with_retry("GET", "https://api.example.com/test", max_retries=3)

        # No retries for non-retryable status
        mock_sleep.assert_not_called()

    @patch("post_pr_comment.time.sleep")
    def test_success_on_first_attempt_no_retry(self, mock_sleep):
        """Should return immediately on success without any retries."""
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()

        with patch("post_pr_comment.requests.post", return_value=ok_response):
            result = _request_with_retry("POST", "https://api.example.com/test", json={"body": "test"})

        assert result == ok_response
        mock_sleep.assert_not_called()

    @patch("post_pr_comment.time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep):
        """Should use exponential backoff: 1s, 2s, 4s."""
        fail_response = MagicMock()
        fail_response.status_code = 504
        fail_response.headers = {}
        fail_response.raise_for_status.side_effect = requests.exceptions.HTTPError("504 Gateway Timeout")

        with patch("post_pr_comment.requests.get", return_value=fail_response):
            with pytest.raises(requests.exceptions.HTTPError):
                _request_with_retry("GET", "https://api.example.com/test", max_retries=3)

        # Backoff: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0, 1.0 * 2^2 = 4.0
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)
