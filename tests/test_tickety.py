#!/usr/bin/env python3
"""
Test suite for the Tickety API client library.

This test suite includes:
- Unit tests for helper methods (CTI parsing, validation)
- Mock-based tests for API interactions
- Integration tests for the full workflow

Usage:
    python test_tickety.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Mock the CRT dependencies before importing tickety
sys.modules["awscrt"] = MagicMock()
sys.modules["botocore.crt"] = MagicMock()
sys.modules["botocore.crt.auth"] = MagicMock()

# Add .github/scripts to path to import tickety module
scripts_dir = Path(__file__).parent.parent / ".github" / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the Tickety class and enums
from tickety import TicketAction, Tickety


class TestCTIParsing(unittest.TestCase):
    """Test cases for CTI (Category/Type/Item) parsing."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock AWS credentials to avoid actual STS calls
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            self.client = Tickety(
                tickety_endpoint="https://global.api.tickety.amazon.dev",
                ticketing_system_name="Default",
                aws_region="us-west-2",
            )

    def test_parse_cti_valid(self):
        """Test parsing a valid CTI string."""
        cti = "AWS/SageMaker/HP Recipes Customer Issues"
        expected = [
            {"key": "category", "value": "AWS"},
            {"key": "type", "value": "SageMaker"},
            {"key": "item", "value": "HP Recipes Customer Issues"},
        ]
        result = self.client._parse_cti(cti)
        self.assertEqual(result, expected)

    def test_parse_cti_with_slashes_in_item(self):
        """Test parsing CTI with slashes in the item field."""
        cti = "AWS/EC2/Some/Path/With/Slashes"
        expected = [
            {"key": "category", "value": "AWS"},
            {"key": "type", "value": "EC2"},
            {"key": "item", "value": "Some/Path/With/Slashes"},
        ]
        result = self.client._parse_cti(cti)
        self.assertEqual(result, expected)

    def test_parse_cti_invalid_format(self):
        """Test parsing an invalid CTI string raises ValueError."""
        invalid_ctis = [
            "AWS",  # Only one part
            "AWS/SageMaker",  # Only two parts
            "",  # Empty string
        ]

        for cti in invalid_ctis:
            with self.subTest(cti=cti):
                with self.assertRaises(ValueError) as context:
                    self.client._parse_cti(cti)
                self.assertIn("Invalid CTI format", str(context.exception))


class TestParameterValidation(unittest.TestCase):
    """Test cases for parameter validation."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            self.client = Tickety(
                tickety_endpoint="https://global.api.tickety.amazon.dev",
                ticketing_system_name="Default",
                aws_region="us-west-2",
            )

    def test_init_with_invalid_endpoint(self):
        """Test initialization with invalid endpoint."""
        with patch("boto3.client"):
            # Empty endpoint
            with self.assertRaises(ValueError) as context:
                Tickety(tickety_endpoint="", ticketing_system_name="Default")
            self.assertIn("endpoint cannot be empty", str(context.exception))

            # Non-HTTPS endpoint
            with self.assertRaises(ValueError) as context:
                Tickety(tickety_endpoint="http://api.tickety.amazon.dev", ticketing_system_name="Default")
            self.assertIn("must use HTTPS", str(context.exception))

    def test_create_ticket_with_empty_parameters(self):
        """Test create_ticket with empty parameters raises ValueError."""
        test_cases = [
            ("", "Description", "SEV_3", "AWS/Service/Item", "dedup-1", "title cannot be empty"),
            ("Title", "", "SEV_3", "AWS/Service/Item", "dedup-1", "description cannot be empty"),
            ("Title", "Description", "", "AWS/Service/Item", "dedup-1", "severity cannot be empty"),
            ("Title", "Description", "SEV_3", "AWS/Service/Item", "", "deduplicator string cannot be empty"),
        ]

        for title, desc, sev, cti, dedup, error_msg in test_cases:
            with self.subTest(error=error_msg):
                with self.assertRaises(ValueError) as context:
                    self.client.create_ticket(
                        title=title, description=desc, severity=sev, cti=cti, deduplicator_string=dedup
                    )
                self.assertIn(error_msg, str(context.exception).lower())

    def test_create_or_update_ticket_with_empty_parameters(self):
        """Test create_or_update_ticket with empty parameters raises ValueError."""
        with self.assertRaises(ValueError):
            self.client.create_or_update_ticket(
                title="",
                description="Description",
                severity="SEV_3",
                cti="AWS/Service/Item",
                deduplicator_string="dedup-1",
            )


class TestMakeRequest(unittest.TestCase):
    """Test cases for the _make_request helper method."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            self.client = Tickety(
                tickety_endpoint="https://global.api.tickety.amazon.dev",
                ticketing_system_name="Default",
                aws_region="us-west-2",
            )

    @patch("tickety.requests.post")
    @patch.object(Tickety, "_sign_request_sigv4a")
    def test_make_request_success(self, mock_sign, mock_post):
        """Test successful API request."""
        # Mock signing
        mock_sign.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"ticketId": "V123456"}'
        mock_response.json.return_value = {"ticketId": "V123456"}
        mock_post.return_value = mock_response

        result = self.client._make_request(
            method="POST",
            url="https://global.api.tickety.amazon.dev/Default/123456789012/tickets",
            body='{"title": "Test"}',
            description="Test Request",
        )

        self.assertEqual(result, {"ticketId": "V123456"})
        mock_post.assert_called_once()

    @patch("tickety.requests.get")
    @patch.object(Tickety, "_sign_request_sigv4a")
    def test_make_request_404(self, mock_sign, mock_get):
        """Test API request with 404 response."""
        mock_sign.return_value = {"Authorization": "AWS4-HMAC-SHA256 ..."}

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(Exception):
            self.client._make_request(
                method="GET",
                url="https://global.api.tickety.amazon.dev/Default/123456789012/tickets/V999",
                description="Get Ticket",
            )


class TestSearchTickets(unittest.TestCase):
    """Test cases for searching tickets."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            self.client = Tickety(
                tickety_endpoint="https://global.api.tickety.amazon.dev",
                ticketing_system_name="Default",
                aws_region="us-west-2",
            )

    @patch.object(Tickety, "_make_request")
    def test_search_tickets_with_results(self, mock_request):
        """Test searching tickets that returns results."""
        mock_request.return_value = {
            "ticketSummaries": [{"ticketId": "V123456", "status": "Assigned", "title": "Test Ticket"}]
        }

        tickets = self.client.search_tickets(
            deduplicator_string="test-dedup", cti="AWS/SageMaker/HP Recipes Customer Issues"
        )

        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["ticketId"], "V123456")

    @patch.object(Tickety, "_make_request")
    def test_search_tickets_no_results(self, mock_request):
        """Test searching tickets with no results."""
        mock_request.return_value = {"ticketSummaries": []}

        tickets = self.client.search_tickets(
            deduplicator_string="nonexistent-dedup", cti="AWS/SageMaker/HP Recipes Customer Issues"
        )

        self.assertEqual(len(tickets), 0)


class TestCreateOrUpdateTicket(unittest.TestCase):
    """Test cases for the idempotent create_or_update_ticket workflow."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("boto3.client") as mock_boto3:
            mock_sts = MagicMock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_boto3.return_value = mock_sts

            self.client = Tickety(
                tickety_endpoint="https://global.api.tickety.amazon.dev",
                ticketing_system_name="Default",
                aws_region="us-west-2",
            )

    @patch.object(Tickety, "search_tickets")
    @patch.object(Tickety, "create_ticket")
    def test_create_new_ticket(self, mock_create, mock_search):
        """Test creating a new ticket when none exists."""
        # No existing tickets
        mock_search.return_value = []

        # Mock ticket creation - API returns "id" at root level
        mock_create.return_value = {"id": "V123456", "dedupeStatus": "CREATED"}

        result = self.client.create_or_update_ticket(
            title="Test Ticket",
            description="Test Description",
            severity="SEV_3",
            cti="AWS/SageMaker/HP Recipes Customer Issues",
            deduplicator_string="test-dedup-123",
        )

        self.assertEqual(result["action"], TicketAction.CREATED)
        self.assertEqual(result["ticket_id"], "V123456")
        mock_create.assert_called_once()

    @patch.object(Tickety, "search_tickets")
    @patch.object(Tickety, "get_ticket")
    @patch.object(Tickety, "update_ticket")
    def test_no_update_when_unchanged(self, mock_update, mock_get, mock_search):
        """Test that open ticket is NOT updated when title and description match."""
        # Existing open ticket
        mock_search.return_value = [{"ticketId": "V123456", "status": "Assigned"}]

        # Mock get_ticket to return matching content
        mock_get.return_value = {
            "ticketId": "V123456",
            "title": "Test Ticket",
            "description": "Test Description",
            "status": "Assigned",
        }

        mock_update.return_value = {"success": True}

        result = self.client.create_or_update_ticket(
            title="Test Ticket",
            description="Test Description",
            severity="SEV_3",
            cti="AWS/SageMaker/HP Recipes Customer Issues",
            deduplicator_string="test-dedup-123",
        )

        # When title and description match, no update should occur
        self.assertEqual(result["action"], TicketAction.NO_CHANGE)
        self.assertEqual(result["ticket_id"], "V123456")
        mock_update.assert_not_called()

    @patch.object(Tickety, "search_tickets")
    @patch.object(Tickety, "get_ticket")
    @patch.object(Tickety, "update_ticket")
    def test_update_when_description_changed(self, mock_update, mock_get, mock_search):
        """Test that update occurs when description changes."""
        # Existing open ticket
        mock_search.return_value = [{"ticketId": "V123456", "status": "Assigned"}]

        # Mock get_ticket to return different description
        mock_get.return_value = {
            "ticketId": "V123456",
            "title": "Test Ticket",
            "description": "Old Description",
            "status": "Assigned",
        }

        mock_update.return_value = {"success": True}

        result = self.client.create_or_update_ticket(
            title="Test Ticket",
            description="New Description",
            severity="SEV_3",
            cti="AWS/SageMaker/HP Recipes Customer Issues",
            deduplicator_string="test-dedup-123",
        )

        self.assertEqual(result["action"], TicketAction.UPDATED)
        self.assertEqual(result["ticket_id"], "V123456")
        mock_update.assert_called_once()

    @patch.object(Tickety, "search_tickets")
    @patch.object(Tickety, "get_ticket")
    @patch.object(Tickety, "update_ticket")
    def test_update_when_title_changed(self, mock_update, mock_get, mock_search):
        """Test that update occurs when title changes."""
        # Existing open ticket
        mock_search.return_value = [{"ticketId": "V123456", "status": "Assigned"}]

        # Mock get_ticket to return different title
        mock_get.return_value = {
            "ticketId": "V123456",
            "title": "Old Title",
            "description": "Test Description",
            "status": "Assigned",
        }

        mock_update.return_value = {"success": True}

        result = self.client.create_or_update_ticket(
            title="New Title",
            description="Test Description",
            severity="SEV_3",
            cti="AWS/SageMaker/HP Recipes Customer Issues",
            deduplicator_string="test-dedup-123",
        )

        self.assertEqual(result["action"], TicketAction.UPDATED)
        self.assertEqual(result["ticket_id"], "V123456")
        mock_update.assert_called_once()

    @patch.object(Tickety, "add_comment")
    @patch.object(Tickety, "update_ticket")
    @patch.object(Tickety, "search_tickets")
    def test_closed_ticket_creates_new(self, mock_search, mock_update, mock_add_comment):
        """Test that closed tickets are ignored and a new ticket is created instead."""
        # Existing closed ticket
        mock_search.return_value = [{"ticketId": "V123456", "status": "Closed"}]

        # Mock create_ticket for the new ticket path
        with patch.object(self.client, "create_ticket") as mock_create:
            mock_create.return_value = {"id": "V999999", "dedupeStatus": "CREATED"}

            result = self.client.create_or_update_ticket(
                title="Test Ticket",
                description="Test Description",
                severity="SEV_3",
                cti="AWS/SageMaker/HP Recipes Customer Issues",
                deduplicator_string="test-dedup-123",
            )

        # Closed tickets cannot be reopened — a new ticket should be created
        self.assertEqual(result["action"], TicketAction.CREATED)
        self.assertEqual(result["ticket_id"], "V999999")
        mock_create.assert_called_once()
        mock_update.assert_not_called()
        mock_add_comment.assert_not_called()

    @patch.object(Tickety, "add_comment")
    @patch.object(Tickety, "update_ticket")
    @patch.object(Tickety, "search_tickets")
    def test_resolved_ticket_creates_new(self, mock_search, mock_update, mock_add_comment):
        """Test that resolved tickets are ignored and a new ticket is created instead."""
        # Existing resolved ticket
        mock_search.return_value = [{"ticketId": "V123456", "status": "Resolved"}]

        with patch.object(self.client, "create_ticket") as mock_create:
            mock_create.return_value = {"id": "V999999", "dedupeStatus": "CREATED"}

            result = self.client.create_or_update_ticket(
                title="Test Ticket",
                description="Test Description",
                severity="SEV_3",
                cti="AWS/SageMaker/HP Recipes Customer Issues",
                deduplicator_string="test-dedup-123",
            )

        self.assertEqual(result["action"], TicketAction.CREATED)
        self.assertEqual(result["ticket_id"], "V999999")
        mock_create.assert_called_once()
        mock_update.assert_not_called()

    @patch.object(Tickety, "search_tickets")
    @patch.object(Tickety, "get_ticket")
    @patch.object(Tickety, "update_ticket")
    def test_multiple_open_tickets_selects_most_recent(self, mock_update, mock_get, mock_search):
        """Test that when multiple open tickets exist, the most recently modified is selected."""
        # Multiple open tickets with different modification dates
        mock_search.return_value = [
            {"ticketId": "V111111", "status": "Assigned", "lastModifiedDate": "2024-01-01T10:00:00Z"},
            {"ticketId": "V333333", "status": "Assigned", "lastModifiedDate": "2024-01-03T10:00:00Z"},  # Most recent
            {"ticketId": "V222222", "status": "Assigned", "lastModifiedDate": "2024-01-02T10:00:00Z"},
        ]

        # Mock GetTicket to return human-readable ID and use different description to trigger update
        mock_get.return_value = {
            "id": "V333333",
            "ticketId": "uuid-333",
            "title": "Test Ticket",
            "description": "Old Description",
            "status": "Assigned",
        }

        mock_update.return_value = {"success": True}

        result = self.client.create_or_update_ticket(
            title="Test Ticket",
            description="New Description",
            severity="SEV_3",
            cti="AWS/SageMaker/HP Recipes Customer Issues",
            deduplicator_string="test-dedup-123",
        )

        # Should select the most recently modified ticket (V333333) and update it
        self.assertEqual(result["ticket_id"], "V333333")
        self.assertEqual(result["action"], TicketAction.UPDATED)

    @patch.object(Tickety, "add_comment")
    @patch.object(Tickety, "update_ticket")
    @patch.object(Tickety, "search_tickets")
    def test_multiple_closed_tickets_creates_new(self, mock_search, mock_update, mock_add_comment):
        """Test that when multiple closed tickets exist, a new ticket is created instead of reopening."""
        # Multiple closed tickets with different modification dates
        mock_search.return_value = [
            {"ticketId": "V111111", "status": "Closed", "lastModifiedDate": "2024-01-01T10:00:00Z"},
            {"ticketId": "V222222", "status": "Closed", "lastModifiedDate": "2024-01-02T10:00:00Z"},
            {"ticketId": "V333333", "status": "Closed", "lastModifiedDate": "2024-01-03T10:00:00Z"},
        ]

        with patch.object(self.client, "create_ticket") as mock_create:
            mock_create.return_value = {"id": "V999999", "dedupeStatus": "CREATED"}

            result = self.client.create_or_update_ticket(
                title="Test Ticket",
                description="Test Description",
                severity="SEV_3",
                cti="AWS/SageMaker/HP Recipes Customer Issues",
                deduplicator_string="test-dedup-123",
            )

        # Closed tickets cannot be reopened — a new ticket should be created
        self.assertEqual(result["action"], TicketAction.CREATED)
        self.assertEqual(result["ticket_id"], "V999999")
        mock_create.assert_called_once()
        mock_update.assert_not_called()


def main():
    """Run all tests."""
    # Run tests with verbose output
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
