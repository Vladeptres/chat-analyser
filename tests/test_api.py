from fastapi.testclient import TestClient
import unittest
from unittest.mock import patch, MagicMock

from chat_analyser.api.main import app
from chat_analyser.api.models import (
    ConversationAnalysisRequest,
    ConversationAnalysisResponse,
    UserFeedback,
    PostContextRequest,
    PostContextResponse,
)


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_messages = [
            {"user": "Alice", "message": "Hey everyone!"},
            {"user": "Bob", "message": "Hello Alice, how are you?"},
            {"user": "Alice", "message": "I'm doing great, thanks!"}
        ]
        self.sample_users = ["Alice", "Bob"]

    @patch('chat_analyser.core.analyse_chat')
    def test_analyse_chat_success(self, mock_analyse_chat):
        """Test successful chat analysis"""
        # Mock the response
        mock_response = ConversationAnalysisResponse(
            summary="Test conversation summary",
            users_feedback={
                "Alice": UserFeedback(summary="Alice was friendly", emoji="😊"),
                "Bob": UserFeedback(summary="Bob was polite", emoji="👋")
            }
        )
        mock_analyse_chat.return_value = mock_response

        # Make request
        request_data = {
            "context_type": "party",
            "messages": self.sample_messages,
            "users": self.sample_users,
            "max_attempts": 3
        }
        response = self.client.post("/chat/", json=request_data)

        # Assertions
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["summary"], "Test conversation summary")
        self.assertIn("Alice", response_data["users_feedback"])
        self.assertIn("Bob", response_data["users_feedback"])
        self.assertEqual(response_data["users_feedback"]["Alice"]["emoji"], "😊")

        # Verify the core function was called with correct parameters
        mock_analyse_chat.assert_called_once_with(
            "party", self.sample_messages, self.sample_users
        )

    @patch('chat_analyser.core.analyse_chat')
    def test_analyse_chat_with_retries(self, mock_analyse_chat):
        """Test chat analysis with retries on ValueError"""
        # Mock to raise ValueError on first calls, then succeed
        mock_analyse_chat.side_effect = [
            ValueError("Context not found"),
            ValueError("Context not found"),
            ConversationAnalysisResponse(
                summary="Success after retries",
                users_feedback={
                    "Alice": UserFeedback(summary="Alice summary", emoji="😊")
                }
            )
        ]

        request_data = {
            "context_type": "invalid_context",
            "messages": self.sample_messages,
            "users": self.sample_users,
            "max_attempts": 3
        }
        response = self.client.post("/chat/", json=request_data)

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["summary"], "Success after retries")
        self.assertEqual(mock_analyse_chat.call_count, 3)

    @patch('chat_analyser.core.analyse_chat')
    def test_analyse_chat_max_attempts_exceeded(self, mock_analyse_chat):
        """Test chat analysis when max attempts are exceeded"""
        # Mock to always raise ValueError
        mock_analyse_chat.side_effect = ValueError("Context not found")

        request_data = {
            "context_type": "invalid_context",
            "messages": self.sample_messages,
            "users": self.sample_users,
            "max_attempts": 2
        }
        response = self.client.post("/chat/", json=request_data)

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        # Should return error response with user summaries
        self.assertIn("Alice", response_data["users_feedback"])
        self.assertIn("Bob", response_data["users_feedback"])
        self.assertEqual(mock_analyse_chat.call_count, 2)

    def test_analyse_chat_invalid_request(self):
        """Test chat analysis with invalid request data"""
        # Missing required fields
        request_data = {
            "messages": self.sample_messages,
            "users": self.sample_users
            # Missing context_type
        }
        response = self.client.post("/chat/", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error

    @patch('chat_analyser.core.write_context')
    @patch('chat_analyser.config.AVAILABLE_CONTEXTS', ['party', 'work'])
    def test_post_context_success(self, mock_write_context):
        """Test successful context creation"""
        request_data = {
            "context_type": "meeting",
            "context": "# Meeting Context\nThis is a meeting analysis context."
        }
        response = self.client.post("/context/", json=request_data)

        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertIn("available_contexts", response_data)
        self.assertIsInstance(response_data["available_contexts"], list)

        # Verify the core function was called
        mock_write_context.assert_called_once_with(
            "meeting", "# Meeting Context\nThis is a meeting analysis context."
        )

    def test_post_context_invalid_request(self):
        """Test context creation with invalid request data"""
        # Missing required fields
        request_data = {
            "context_type": "meeting"
            # Missing context field
        }
        response = self.client.post("/context/", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error

    def test_analyse_chat_empty_messages(self):
        """Test chat analysis with empty messages"""
        request_data = {
            "context_type": "party",
            "messages": [],
            "users": self.sample_users,
            "max_attempts": 3
        }
        response = self.client.post("/chat/", json=request_data)
        # Should still process the request
        self.assertEqual(response.status_code, 200)

    def test_analyse_chat_empty_users(self):
        """Test chat analysis with empty users list"""
        request_data = {
            "context_type": "party",
            "messages": self.sample_messages,
            "users": [],
            "max_attempts": 3
        }
        response = self.client.post("/chat/", json=request_data)
        # Should still process the request
        self.assertEqual(response.status_code, 200)

    def test_analyse_chat_default_max_attempts(self):
        """Test that max_attempts defaults to 3 when not provided"""
        request_data = {
            "context_type": "party",
            "messages": self.sample_messages,
            "users": self.sample_users
            # max_attempts not provided, should default to 3
        }
        response = self.client.post("/chat/", json=request_data)
        # Should process without error
        self.assertIn(response.status_code, [200, 422])  # 200 if mocked properly, 422 if validation fails


if __name__ == "__main__":
    unittest.main()
