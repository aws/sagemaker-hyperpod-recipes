#!/usr/bin/env python3
"""
Tickety API client library for GitHub Actions.

This module provides a Python client for creating and managing Tickety tickets
from GitHub Actions workflows. It implements idempotent ticket management with:
- Creating new tickets with deduplication
- Searching for existing tickets (both open and closed)
- Updating existing open tickets (only when content changes)
- Adding comments to tickets on update

The library uses SigV4A signing for authentication with Tickety's global API endpoint.

Workflow behavior:
- New tickets are created when no existing ticket is found for the deduplicator
- Open tickets are updated only if title or description has changed
- Closed/resolved tickets are ignored (a new ticket is created instead)
- GitHub Actions outputs (ticket_id, action) are set via GITHUB_OUTPUT

Example usage:
    from tickety import Tickety

    client = Tickety(
        tickety_endpoint="https://global.api.tickety.amazon.dev",
        ticketing_system_name="Default",
        aws_region="us-west-2"
    )

    result = client.create_or_update_ticket(
        title="Validation needed",
        description="Please review...",
        severity="SEV_3",
        cti="AWS/SageMaker/HP Recipes Customer Issues",
        deduplicator_string="validation-pr-123",
        update_comment="PR updated with new changes.",
    )
    # result["action"] will be: CREATED, UPDATED, or NO_CHANGE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from botocore.awsrequest import AWSRequest
from botocore.crt.auth import CrtSigV4AsymAuth


class TicketAction(str, Enum):
    """
    Enum representing possible actions taken on a ticket during create_or_update_ticket.

    Values inherit from str to make them JSON-serializable and easy to use in string contexts.
    """

    CREATED = "CREATED"  # New ticket was created
    UPDATED = "UPDATED"  # Existing open ticket was updated (title or description changed)
    NO_CHANGE = "NO_CHANGE"  # Open ticket exists with same title and description


class TicketStatus(str, Enum):
    """
    Enum representing ticket statuses in the Tickety system.
    """

    # Open statuses
    ASSIGNED = "Assigned"
    WORK_IN_PROGRESS = "Work In Progress"
    PENDING = "Pending"
    RESEARCHING = "Researching"

    # Closed statuses
    RESOLVED = "Resolved"
    CLOSED = "Closed"

    @classmethod
    def is_open_status(cls, status: str) -> bool:
        """Check if a status value represents an open ticket."""
        return status in {cls.ASSIGNED.value, cls.WORK_IN_PROGRESS.value, cls.PENDING.value, cls.RESEARCHING.value}

    @classmethod
    def is_closed_status(cls, status: str) -> bool:
        """Check if a status value represents a closed ticket."""
        return status in {cls.RESOLVED.value, cls.CLOSED.value}

    @classmethod
    def get_all_open_statuses(cls) -> list[str]:
        """Get list of all open status values."""
        return [cls.ASSIGNED.value, cls.WORK_IN_PROGRESS.value, cls.PENDING.value, cls.RESEARCHING.value]

    @classmethod
    def get_all_closed_statuses(cls) -> list[str]:
        """Get list of all closed status values."""
        return [cls.RESOLVED.value, cls.CLOSED.value]

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Get list of all status values (open + closed)."""
        return cls.get_all_open_statuses() + cls.get_all_closed_statuses()


class Tickety:
    """
    Tickety API client for managing tickets in GitHub Actions workflows.

    This client provides methods for creating, searching, updating, and commenting on
    Tickety tickets using the Tickety API with SigV4A authentication.
    """

    def __init__(
        self,
        tickety_endpoint: str,
        ticketing_system_name: str = "Default",
        ticketing_account_id: str = "Default",
        aws_region: str = "us-west-2",
    ):
        """
        Initialize the Tickety client.

        Args:
            tickety_endpoint: The Tickety API endpoint (e.g., https://global.api.tickety.amazon.dev)
            ticketing_system_name: The ticketing system name (default: Default for SIM-T)
            ticketing_account_id: The ticketing account ID for URL path (default: Default for SIM-T)
            aws_region: AWS region for signing (default: us-west-2)

        Raises:
            ValueError: If required parameters are invalid
            boto3.exceptions.NoCredentialsError: If AWS credentials are not available
        """
        # Validate parameters
        if not tickety_endpoint or not tickety_endpoint.strip():
            raise ValueError("Tickety endpoint cannot be empty")
        if not tickety_endpoint.startswith("https://"):
            raise ValueError("Tickety endpoint must use HTTPS")
        if not ticketing_system_name or not ticketing_system_name.strip():
            raise ValueError("Ticketing system name cannot be empty")
        if not ticketing_account_id or not ticketing_account_id.strip():
            raise ValueError("Ticketing account ID cannot be empty")
        if not aws_region or not aws_region.strip():
            raise ValueError("AWS region cannot be empty")

        self.tickety_endpoint = tickety_endpoint
        self.ticketing_system_name = ticketing_system_name
        self.ticketing_account_id = ticketing_account_id
        self.aws_region = aws_region

        # Verify AWS credentials are available (needed for SigV4A signing)
        try:
            sts_client = boto3.client("sts", region_name=aws_region)
            caller_identity = sts_client.get_caller_identity()
            aws_account = caller_identity["Account"]
        except Exception as e:
            raise RuntimeError(f"Failed to get AWS credentials: {e}") from e

        print(f"=== Tickety Client Initialized ===")
        print(f"Endpoint: {tickety_endpoint}")
        print(f"System: {ticketing_system_name}")
        print(f"Account ID (URL): {ticketing_account_id}")
        print(f"AWS Account (Auth): {aws_account}")
        print(f"Region: {aws_region}")
        print()

    def _parse_cti(self, cti: str) -> list[dict[str, str]]:
        """
        Parse a CTI (Category/Type/Item) string into the format required by Tickety API.

        Args:
            cti: CTI string in format "Category/Type/Item"

        Returns:
            List of dictionaries with key-value pairs for category, type, and item

        Example:
            >>> _parse_cti("AWS/SageMaker/HP Recipes Customer Issues")
            [
                {"key": "category", "value": "AWS"},
                {"key": "type", "value": "SageMaker"},
                {"key": "item", "value": "HP Recipes Customer Issues"}
            ]
        """
        parts = cti.split("/", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid CTI format: {cti}. Expected format: 'Category/Type/Item'")

        return [
            {"key": "category", "value": parts[0]},
            {"key": "type", "value": parts[1]},
            {"key": "item", "value": parts[2]},
        ]

    def _sign_request_sigv4a(self, method: str, url: str, headers: dict[str, str], body: str) -> dict[str, str]:
        """
        Sign an HTTP request using SigV4A (AWS Signature Version 4A) for global endpoints.

        SigV4A is required for Tickety's global API endpoint to support multi-region signing.

        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            url: Full URL to sign
            headers: HTTP headers to include in signing
            body: Request body as string

        Returns:
            Dictionary of signed headers to use in the request

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/BestPractices/
        """
        session = boto3.Session()
        credentials = session.get_credentials()

        # Use CrtSigV4AsymAuth for SigV4A signing (global endpoint support)
        # This is imported from botocore.crt.auth, not botocore.auth
        # Service name must be "tickety" for the Tickety API
        signer = CrtSigV4AsymAuth(
            credentials=credentials.get_frozen_credentials(),
            service_name="tickety",
            region_name="*",  # Global signing - use "*" for multi-region
        )

        # Create an AWSRequest object (required by CrtSigV4AsymAuth.add_auth)
        request = AWSRequest(
            method=method,
            url=url,
            headers=headers,
            data=body.encode("utf-8") if body else b"",
        )

        # Sign the request - this modifies the request headers in place
        signer.add_auth(request)

        # Return the signed headers as a dict
        return dict(request.headers)

    def _make_request(self, method: str, url: str, body: str = "", description: str = "") -> dict[str, Any]:
        """
        Make a signed HTTP request to the Tickety API.

        Args:
            method: HTTP method (GET, POST, PUT)
            url: Full URL to request
            body: Request body (JSON string)
            description: Human-readable description of the request for logging

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On request failure
        """
        if description:
            print(f"=== {description} ===")

        # Prepare headers
        headers = {"Content-Type": "application/json", "Host": urlparse(url).netloc}

        # Sign the request
        signed_headers = self._sign_request_sigv4a(method=method, url=url, headers=headers, body=body)

        # Make the request
        try:
            if method == "GET":
                response = requests.get(url, headers=signed_headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, data=body, headers=signed_headers, timeout=30)
            elif method == "PUT":
                response = requests.put(url, data=body, headers=signed_headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            print(f"Response Status: {response.status_code}")

            if response.status_code in (200, 201, 204):
                if response.content:
                    return response.json()
                return {"success": True}
            else:
                error_msg = f"Request failed with status {response.status_code}"

                # Provide more specific error messages based on status code
                if response.status_code == 400:
                    error_msg += " (Bad Request - check request parameters)"
                elif response.status_code == 401:
                    error_msg += " (Unauthorized - check AWS credentials)"
                elif response.status_code == 403:
                    error_msg += " (Forbidden - check IAM permissions)"
                elif response.status_code == 404:
                    error_msg += " (Not Found - check endpoint and ticket ID)"
                elif response.status_code == 429:
                    error_msg += " (Rate Limited - too many requests)"
                elif response.status_code >= 500:
                    error_msg += " (Server Error - Tickety service issue)"

                print(f"Error: {error_msg}")
                print(f"Response: {response.text}")
                response.raise_for_status()
                return {}

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """
        Get full details of a specific ticket.

        Args:
            ticket_id: ID of the ticket to retrieve

        Returns:
            Full ticket details including title, description, status, etc.

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/Documentation/#HGetTicket
        """
        url = f"{self.tickety_endpoint}/{self.ticketing_system_name}/{self.ticketing_account_id}/tickets/{ticket_id}"
        print(f"Ticket ID: {ticket_id}\n")

        try:
            return self._make_request("GET", url, description="Getting Ticket Details")
        except requests.exceptions.RequestException:
            return {}

    def search_tickets(self, deduplicator_string: str, cti: str) -> list[dict[str, Any]]:
        """
        Search for existing tickets by deduplicator string.

        Searches across all ticket statuses (both open and closed). Caller can filter
        by status after retrieval using TicketStatus.is_open_status() / is_closed_status().

        Args:
            deduplicator_string: Unique string to search for
            cti: CTI (Category/Type/Item) for filtering

        Returns:
            List of matching ticket summaries

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/Documentation/#HListTickets
        """
        categorization = self._parse_cti(cti)

        # Search across all statuses (open and closed)
        statuses = TicketStatus.get_all_statuses()

        # Build filter for ListTickets API
        # categorizations format: list of lists, where each inner list is a categorization
        # Reference: https://code.amazon.com/packages/TicketyServicePythonExamples
        filters = {
            "deduplicatorString": deduplicator_string,
            "categorizations": [categorization],  # categorization is already [{"key": "category", ...}, ...]
            "statuses": statuses,
        }

        payload = {"filters": filters}

        # Build the API URL for ListTickets
        url = f"{self.tickety_endpoint}/{self.ticketing_system_name}/{self.ticketing_account_id}/list/tickets"

        print(f"Deduplicator: {deduplicator_string}")
        print(f"Searching all statuses (open and closed)")
        print()

        # Prepare the request body
        body = json.dumps(payload)

        try:
            result = self._make_request("POST", url, body, "Searching for Existing Tickets")
            tickets = result.get("ticketSummaries", [])
            print(f"Found {len(tickets)} ticket(s)")
            if tickets:
                for ticket in tickets:
                    status = ticket.get("status", "")
                    ticket_id = ticket.get("ticketId", "")
                    print(f"  - {ticket_id}: {status}")
            return tickets

        except requests.exceptions.RequestException as e:
            print(f"Error searching tickets: {e}")
            # Don't raise - return empty list and proceed
            return []

    def create_ticket(
        self,
        title: str,
        description: str,
        severity: str,
        cti: str,
        deduplicator_string: str,
    ) -> dict[str, Any]:
        """
        Create a new Tickety ticket.

        Args:
            title: Ticket title
            description: Ticket description (Markdown supported)
            severity: Severity level (e.g., SEV_3, SEV_2)
            cti: CTI (Category/Type/Item) in format "Category/Type/Item"
            deduplicator_string: Unique string to prevent duplicate tickets

        Returns:
            Response from Tickety API containing ticketId and dedupeStatus

        Raises:
            ValueError: If required parameters are empty or invalid

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/Documentation/#HCreateTicket
        """
        # Validate required parameters
        if not title or not title.strip():
            raise ValueError("Ticket title cannot be empty")
        if not description or not description.strip():
            raise ValueError("Ticket description cannot be empty")
        if not severity or not severity.strip():
            raise ValueError("Ticket severity cannot be empty")
        if not deduplicator_string or not deduplicator_string.strip():
            raise ValueError("Deduplicator string cannot be empty")

        categorization = self._parse_cti(cti)

        # Build the ticket payload
        payload = {
            "title": title,
            "description": description,
            "descriptionContentType": "text/amz-markdown-sim",
            "severity": severity,
            "categorization": categorization,
            "deduplicatorString": deduplicator_string,
        }

        # Build the API URL
        url = f"{self.tickety_endpoint}/{self.ticketing_system_name}/{self.ticketing_account_id}/tickets"

        print(f"Title: {title}")
        print(f"Severity: {severity}")
        print(f"CTI: {cti}")
        print(f"Deduplicator: {deduplicator_string}")
        print()

        # Prepare the request body
        body = json.dumps(payload)

        try:
            result = self._make_request("POST", url, body, "Creating Ticket")
            return result

        except requests.exceptions.RequestException as e:
            print(f"Error creating ticket: {e}")
            raise

    def update_ticket(
        self,
        ticket_id: str,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing ticket's title, description, and/or status.

        Args:
            ticket_id: ID of the ticket to update
            title: New ticket title (optional)
            description: New ticket description (optional)
            status: New ticket status - e.g., "Assigned" to reopen (optional)

        Returns:
            Response from Tickety API

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/Documentation/#HUpdateTicket
        """
        # Build update payload with only provided fields
        update_payload = {}

        if title is not None:
            update_payload["title"] = title

        if description is not None:
            update_payload["description"] = description
            update_payload["descriptionContentType"] = "text/amz-markdown-sim"

        if status is not None:
            update_payload["status"] = status

        if not update_payload:
            print("Warning: No updates provided")
            return {"success": True}

        # Build the API URL for UpdateTicket
        url = f"{self.tickety_endpoint}/{self.ticketing_system_name}/{self.ticketing_account_id}/tickets/{ticket_id}"

        print(f"Ticket ID: {ticket_id}")
        print(f"Updates:")
        if title:
            print(f"  - Title: {title}")
        if description:
            print(f"  - Description: {len(description)} characters")
        if status:
            print(f"  - Status: {status}")
        print()

        # Prepare the request body - fields go directly at root level (no wrapper)
        # Reference: TicketyServiceCurlExamples - UpdateTicket example
        body = json.dumps(update_payload)

        try:
            self._make_request("POST", url, body, "Updating Ticket")
            print("Ticket updated successfully")
            return {"success": True}

        except requests.exceptions.RequestException as e:
            print(f"Error updating ticket: {e}")
            raise

    def add_comment(self, ticket_id: str, comment: str) -> dict[str, Any]:
        """
        Add a comment to an existing ticket.

        Args:
            ticket_id: ID of the ticket to comment on
            comment: Comment text (Markdown supported)

        Returns:
            Response from Tickety API

        Reference:
            https://w.amazon.com/bin/view/IssueManagement/SIMTicketing/TicketyAPI/Documentation/#HCreateTicketComment
        """
        # Build comment payload
        payload = {
            "message": comment,
            "contentType": "text/amz-markdown-sim",
            "threadName": "CORRESPONDENCE",  # Use CORRESPONDENCE for customer-visible updates
        }

        # Build the API URL for CreateTicketComment
        url = f"{self.tickety_endpoint}/{self.ticketing_system_name}/{self.ticketing_account_id}/tickets/{ticket_id}/comments"

        print(f"Ticket ID: {ticket_id}")
        print(f"Comment length: {len(comment)} characters")
        print()

        # Prepare the request body
        body = json.dumps(payload)

        try:
            result = self._make_request("POST", url, body, "Adding Comment to Ticket")
            print("Comment added successfully")
            return result

        except requests.exceptions.RequestException as e:
            print(f"Error adding comment: {e}")
            raise

    def create_or_update_ticket(
        self,
        title: str,
        description: str,
        severity: str,
        cti: str,
        deduplicator_string: str,
        update_comment: str | None = None,
    ) -> dict[str, Any]:
        """
        Create or update a ticket in an idempotent manner.

        This method implements the full workflow:
        1. Search for existing tickets (both open and closed) by deduplicator
        2. If an open ticket exists:
           - Compare title and description with existing values
           - If changed: update ticket and add optional comment
           - If unchanged: return NO_CHANGE (skip update)
        3. If only closed/resolved tickets exist: create a new ticket
        4. If no ticket exists: create a new one

        Args:
            title: Ticket title
            description: Ticket description (Markdown supported)
            severity: Severity level (e.g., SEV_3, SEV_2)
            cti: CTI (Category/Type/Item) in format "Category/Type/Item"
            deduplicator_string: Unique string to identify this ticket
            update_comment: Comment to add when updating ticket (timestamp added automatically)

        Returns:
            Dictionary containing:
                - ticket_id: The ticket ID
                - action: TicketAction enum value (CREATED, UPDATED, or NO_CHANGE)
                - was_deduped: Whether ticket creation was deduped by Tickety API

        Raises:
            ValueError: If required parameters are empty or invalid
        """
        # Validate required parameters
        if not title or not title.strip():
            raise ValueError("Ticket title cannot be empty")
        if not description or not description.strip():
            raise ValueError("Ticket description cannot be empty")
        if not severity or not severity.strip():
            raise ValueError("Ticket severity cannot be empty")
        if not deduplicator_string or not deduplicator_string.strip():
            raise ValueError("Deduplicator string cannot be empty")

        print("=== Step 1: Checking for Existing Tickets ===")
        all_tickets = self.search_tickets(deduplicator_string=deduplicator_string, cti=cti)

        # Separate tickets by status
        open_tickets = [t for t in all_tickets if TicketStatus.is_open_status(t.get("status", ""))]
        closed_tickets = [t for t in all_tickets if TicketStatus.is_closed_status(t.get("status", ""))]

        # Sort tickets by modification date (most recent first)
        # Use lastModifiedDate if available, otherwise use empty string for stable sorting
        open_tickets.sort(key=lambda t: t.get("lastModifiedDate", ""), reverse=True)
        closed_tickets.sort(key=lambda t: t.get("lastModifiedDate", ""), reverse=True)

        # Only consider open tickets — closed tickets cannot be reopened via the API,
        # so we ignore them and create a new ticket instead.
        if open_tickets:
            ticket_id = open_tickets[0].get("ticketId", "")
            ticket_status = open_tickets[0].get("status", "")
            print(f"\nFound {len(open_tickets)} open ticket(s), using most recently modified")
            print(f"Selected ticket: {ticket_id} (Status: {ticket_status})")
            print(f"Ticket URL: https://t.corp.amazon.com/{ticket_id}")

            # Ticket exists - determine action
            print("\n=== Step 2: Determining Action ===")

            # Open ticket found - check if title or description has actually changed
            print("\nOpen ticket found. Checking if content has changed...")

            # Get full ticket details to compare title and description
            ticket_details = self.get_ticket(ticket_id)
            existing_title = ticket_details.get("title", "")
            existing_description = ticket_details.get("description", "")

            # Normalize strings for comparison (strip whitespace)
            title_changed = title.strip() != existing_title.strip()
            description_changed = description.strip() != existing_description.strip()

            print(f"Title changed: {title_changed}")
            print(f"Description changed: {description_changed}")

            if not title_changed and not description_changed:
                # Content hasn't changed - skip update
                print("\nNo changes detected. Skipping update.")
                print(f"Ticket ID: {ticket_id}")
                print(f"Ticket URL: https://t.corp.amazon.com/{ticket_id}")

                return {
                    "ticket_id": ticket_id,
                    "action": TicketAction.NO_CHANGE,
                    "was_deduped": False,
                }

            # Content has changed - proceed with update
            print("\nContent has changed. Updating ticket...")
            print("\n=== Step 3: Updating Existing Ticket ===")

            # Update ticket title and/or description (only what changed)
            update_args = {"ticket_id": ticket_id}
            if title_changed:
                update_args["title"] = title
            if description_changed:
                update_args["description"] = description

            self.update_ticket(**update_args)

            # Add a comment with the update timestamp
            if update_comment is not None:
                print("\n=== Step 4: Adding Update Comment ===")
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                comment = f"## Ticket Updated: {timestamp}\n\n{update_comment}"
                self.add_comment(ticket_id=ticket_id, comment=comment)

            print("\n=== Ticket Update Complete ===")
            print(f"Ticket ID: {ticket_id}")
            print(f"Action: {TicketAction.UPDATED}")

            return {
                "ticket_id": ticket_id,
                "action": TicketAction.UPDATED,
                "was_deduped": False,
            }

        else:
            # No open ticket found (may have closed tickets) — create a new one
            if closed_tickets:
                print(f"\nFound {len(closed_tickets)} closed ticket(s), but closed tickets cannot be reopened.")
                print("Creating a new ticket instead.")
            else:
                print("\nNo existing tickets found.")
            print("\nCreating new ticket...")

            result = self.create_ticket(
                title=title,
                description=description,
                severity=severity,
                cti=cti,
                deduplicator_string=deduplicator_string,
            )

            # Extract ticket ID from response - CreateTicket returns "id" at root level
            # Reference: https://code.amazon.com/packages/TicketyServicePythonExamples
            ticket_id = result.get("id", "")
            dedupe_status = result.get("dedupeStatus", "CREATED")

            if dedupe_status == "DEDUPED":
                print(f"Note: Existing ticket found via API deduplication")

            return {
                "ticket_id": ticket_id,
                "action": TicketAction.CREATED,
                "was_deduped": dedupe_status == "DEDUPED",
            }


def _set_github_output(name: str, value: str) -> None:
    """Set a GitHub Actions output variable (silent - no console output)."""
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{name}={value}\n")


def main():
    """CLI entry point for GitHub Actions workflows."""
    parser = argparse.ArgumentParser(
        description="Create or update Tickety tickets from GitHub Actions workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--title", required=True, help="Ticket title")
    parser.add_argument("--description", required=True, help="Ticket description (Markdown supported)")
    parser.add_argument("--severity", required=True, help="Severity level (e.g., SEV_3)")
    parser.add_argument("--cti", required=True, help="CTI in format 'Category/Type/Item'")
    parser.add_argument("--deduplicator", required=True, help="Unique deduplicator string")

    # Optional arguments
    parser.add_argument(
        "--update-comment",
        default=None,
        help="Comment to add when updating existing ticket (optional)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TICKETY_ENDPOINT", "https://global.api.tickety.amazon.dev"),
        help="Tickety API endpoint (default: from TICKETY_ENDPOINT env or global endpoint)",
    )
    parser.add_argument(
        "--system-name",
        default="Default",
        help="Ticketing system name (default: Default for SIM-T)",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )

    args = parser.parse_args()

    try:
        # Initialize Tickety client
        client = Tickety(
            tickety_endpoint=args.endpoint,
            ticketing_system_name=args.system_name,
            aws_region=args.region,
        )

        # Create or update ticket
        result = client.create_or_update_ticket(
            title=args.title,
            description=args.description,
            severity=args.severity,
            cti=args.cti,
            deduplicator_string=args.deduplicator,
            update_comment=args.update_comment,
        )

        # Set GitHub Actions outputs
        ticket_id = result["ticket_id"]
        action = result["action"]
        _set_github_output("ticket_id", ticket_id)
        _set_github_output("action", action)

        # Final summary
        print(f"\n✓ {action}: https://t.corp.amazon.com/{ticket_id}")

        sys.exit(0)

    except Exception as e:
        print(f"\nError: Failed to create/update ticket: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        _set_github_output("ticket_id", "")
        _set_github_output("action", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
