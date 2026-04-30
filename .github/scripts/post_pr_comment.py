#!/usr/bin/env python3
"""
Post or update a PR comment with deduplication via a hidden HTML marker.

This script reliably finds an existing comment by searching for a hidden
HTML marker string, then either updates it or creates a new one. It uses
the GitHub REST API with proper pagination to avoid the duplicate-comment
bugs that occur with ``gh pr view | jq`` pipelines.

All HTTP requests include retry logic with exponential backoff to handle
transient failures (rate limiting, 502/503/504 server errors). Retries
default to 3 attempts with 1s/2s/4s backoff, and respect the
``Retry-After`` header on 429 responses.

Usage (from GitHub Actions):
    python .github/scripts/post_pr_comment.py \
        --repo "owner/repo" \
        --pr-number 123 \
        --body-file /tmp/comment.md

    # Or with inline body:
    python .github/scripts/post_pr_comment.py \
        --repo "owner/repo" \
        --pr-number 123 \
        --body "## My Comment"

Environment:
    GITHUB_TOKEN: Required. The GitHub token for API authentication.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

# Hidden HTML comment used to identify our workflow's comments
DEFAULT_MARKER = "<!-- eval-validation-workflow-comment -->"

# GitHub API base URL
GITHUB_API = "https://api.github.com"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}


def _get_headers(token: str) -> dict[str, str]:
    """Build GitHub API request headers."""
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: dict | None = None,
    params: dict | None = None,
    timeout: int = 30,
    max_retries: int = MAX_RETRIES,
) -> requests.Response:
    """Make an HTTP request with retry logic and exponential backoff.

    Retries on transient failures: HTTP 429 (rate limited), 502, 503, 504.
    Respects the ``Retry-After`` header on 429 responses.

    Args:
        method: HTTP method (``GET``, ``POST``, ``PATCH``).
        url: Full URL to request.
        headers: HTTP headers.
        json: JSON body (for POST/PATCH).
        params: Query parameters (for GET).
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.

    Returns:
        The successful :class:`requests.Response`.

    Raises:
        requests.exceptions.HTTPError: After all retries are exhausted.
    """
    request_fn = getattr(requests, method.lower())
    kwargs: dict = {"headers": headers, "timeout": timeout}
    if json is not None:
        kwargs["json"] = json
    if params is not None:
        kwargs["params"] = params

    last_response = None
    for attempt in range(max_retries + 1):
        response = request_fn(url, **kwargs)
        last_response = response

        if response.status_code not in RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response

        # Retryable error — decide whether to retry
        if attempt >= max_retries:
            break

        # Calculate backoff: use Retry-After header for 429, else exponential
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = float(retry_after)
                except (ValueError, TypeError):
                    wait = INITIAL_BACKOFF_SECONDS * (2**attempt)
            else:
                wait = INITIAL_BACKOFF_SECONDS * (2**attempt)
        else:
            wait = INITIAL_BACKOFF_SECONDS * (2**attempt)

        print(
            f"Request to {url} returned {response.status_code}, retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})"
        )
        time.sleep(wait)

    # All retries exhausted — raise the error from the last response
    assert last_response is not None
    last_response.raise_for_status()
    return last_response  # unreachable, but keeps type-checker happy


def find_existing_comment(
    token: str,
    repo: str,
    pr_number: int,
    marker: str,
) -> int | None:
    """Find an existing PR comment containing the marker string.

    Paginates through all issue comments to find the most recent one
    from ``github-actions[bot]`` that contains the marker.

    Args:
        token: GitHub API token.
        repo: Repository in ``owner/repo`` format.
        pr_number: PR number.
        marker: Hidden HTML marker to search for.

    Returns:
        The comment ID if found, or ``None``.
    """
    headers = _get_headers(token)
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"

    best_comment_id = None
    best_created_at = ""

    page = 1
    per_page = 100

    while True:
        params = {"page": page, "per_page": per_page}
        response = _request_with_retry("GET", url, headers=headers, params=params)

        comments = response.json()
        if not comments:
            break

        for comment in comments:
            # Check if it's from the bot and contains our marker
            user = comment.get("user", {})
            login = user.get("login", "")
            body = comment.get("body", "")

            if login == "github-actions[bot]" and marker in body:
                created_at = comment.get("created_at", "")
                comment_id = comment.get("id")
                # Keep the most recent matching comment
                if created_at > best_created_at:
                    best_created_at = created_at
                    best_comment_id = comment_id

        # Check if there are more pages
        if len(comments) < per_page:
            break
        page += 1

    return best_comment_id


def update_comment(token: str, repo: str, comment_id: int, body: str) -> None:
    """Update an existing comment by ID.

    Args:
        token: GitHub API token.
        repo: Repository in ``owner/repo`` format.
        comment_id: The comment ID to update.
        body: New comment body (Markdown).
    """
    headers = _get_headers(token)
    url = f"{GITHUB_API}/repos/{repo}/issues/comments/{comment_id}"

    response = _request_with_retry("PATCH", url, headers=headers, json={"body": body})
    print(f"Updated existing comment {comment_id}")


def create_comment(token: str, repo: str, pr_number: int, body: str) -> int:
    """Create a new comment on a PR.

    Args:
        token: GitHub API token.
        repo: Repository in ``owner/repo`` format.
        pr_number: PR number.
        body: Comment body (Markdown).

    Returns:
        The new comment ID.
    """
    headers = _get_headers(token)
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"

    response = _request_with_retry("POST", url, headers=headers, json={"body": body})

    comment_id = response.json().get("id")
    print(f"Created new comment {comment_id}")
    return comment_id


def post_or_update_comment(
    token: str,
    repo: str,
    pr_number: int,
    body: str,
    marker: str = DEFAULT_MARKER,
) -> int:
    """Post a new PR comment or update the existing one.

    Searches for an existing comment with the marker. If found, updates it.
    If not found, creates a new one.

    The body should already contain the marker string (typically as the
    first line).

    Args:
        token: GitHub API token.
        repo: Repository in ``owner/repo`` format.
        pr_number: PR number.
        body: Full comment body including marker.
        marker: Hidden HTML marker for deduplication.

    Returns:
        The comment ID (created or updated).
    """
    existing_id = find_existing_comment(token, repo, pr_number, marker)

    if existing_id is not None:
        update_comment(token, repo, existing_id, body)
        return existing_id
    else:
        return create_comment(token, repo, pr_number, body)


def main():
    parser = argparse.ArgumentParser(
        description="Post or update a PR comment with deduplication via hidden marker",
    )
    parser.add_argument("--repo", required=True, help="Repository in owner/repo format")
    parser.add_argument("--pr-number", required=True, type=int, help="PR number")
    parser.add_argument("--body", default=None, help="Comment body (Markdown)")
    parser.add_argument("--body-file", default=None, help="Path to file containing comment body")
    parser.add_argument("--marker", default=DEFAULT_MARKER, help="Hidden HTML marker for deduplication")

    args = parser.parse_args()

    # Get token from environment
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is required", file=sys.stderr)
        sys.exit(1)

    # Get body from --body or --body-file
    if args.body_file:
        with open(args.body_file, "r") as f:
            body = f.read()
    elif args.body:
        body = args.body
    else:
        print("Error: Either --body or --body-file is required", file=sys.stderr)
        sys.exit(1)

    post_or_update_comment(
        token=token,
        repo=args.repo,
        pr_number=args.pr_number,
        body=body,
        marker=args.marker,
    )


if __name__ == "__main__":
    main()
