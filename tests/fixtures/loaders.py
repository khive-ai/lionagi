"""
Test Data Loading Utilities

Provides centralized loading and management of test data files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TestDataLoader:
    """Centralized loader for test data files."""

    def __init__(self, fixtures_dir: Path | None = None):
        """
        Initialize the test data loader.

        Args:
            fixtures_dir: Path to fixtures directory. If None, uses default.
        """
        if fixtures_dir is None:
            # Auto-detect fixtures directory
            current_file = Path(__file__)
            self.fixtures_dir = current_file.parent
        else:
            self.fixtures_dir = Path(fixtures_dir)

        self.data_dir = self.fixtures_dir / "data"

    def load_json(self, filename: str) -> dict[str, Any]:
        """
        Load JSON data from fixtures/data directory.

        Args:
            filename: JSON filename (with or without .json extension)

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not filename.endswith(".json"):
            filename += ".json"

        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            return json.load(f)

    def get_conversation_data(self, scenario: str) -> dict[str, Any]:
        """
        Get conversation data for a specific scenario.

        Args:
            scenario: Scenario name (e.g., 'basic_chat', 'technical_discussion')

        Returns:
            Conversation data including messages and metadata
        """
        conversations = self.load_json("sample_conversations")
        if scenario not in conversations:
            available = list(conversations.keys())
            raise ValueError(
                f"Scenario '{scenario}' not found. Available: {available}"
            )

        return conversations[scenario]

    def get_api_response(self, response_type: str) -> dict[str, Any]:
        """
        Get API response data for testing.

        Args:
            response_type: Type of response (e.g., 'successful_chat_response')

        Returns:
            API response data
        """
        responses = self.load_json("api_responses")
        if response_type not in responses:
            available = list(responses.keys())
            raise ValueError(
                f"Response type '{response_type}' not found. Available: {available}"
            )

        return responses[response_type]

    def get_error_scenario(self, error_type: str) -> dict[str, Any]:
        """
        Get error scenario data for testing error handling.

        Args:
            error_type: Type of error (e.g., 'api_rate_limit_error')

        Returns:
            Error scenario data
        """
        errors = self.load_json("error_scenarios")
        if error_type not in errors:
            available = list(errors.keys())
            raise ValueError(
                f"Error type '{error_type}' not found. Available: {available}"
            )

        return errors[error_type]

    def list_conversations(self) -> list[str]:
        """List available conversation scenarios."""
        conversations = self.load_json("sample_conversations")
        return list(conversations.keys())

    def list_api_responses(self) -> list[str]:
        """List available API response types."""
        responses = self.load_json("api_responses")
        return list(responses.keys())

    def list_error_scenarios(self) -> list[str]:
        """List available error scenarios."""
        errors = self.load_json("error_scenarios")
        return list(errors.keys())


# Create default loader instance
_default_loader = TestDataLoader()


def load_test_data(filename: str) -> dict[str, Any]:
    """
    Convenience function to load test data using default loader.

    Args:
        filename: JSON filename (with or without .json extension)

    Returns:
        Parsed JSON data
    """
    return _default_loader.load_json(filename)


def get_conversation(scenario: str) -> dict[str, Any]:
    """Get conversation data for scenario."""
    return _default_loader.get_conversation_data(scenario)


def get_api_response(response_type: str) -> dict[str, Any]:
    """Get API response data."""
    return _default_loader.get_api_response(response_type)


def get_error_scenario(error_type: str) -> dict[str, Any]:
    """Get error scenario data."""
    return _default_loader.get_error_scenario(error_type)
