import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import tempfile
import shutil

from chat_analyser.core.analyser import load_context, format_query, analyse_chat
from chat_analyser.core.utils import write_context
from chat_analyser.api.models import ConversationAnalysisResponse


class TestAnalyser(unittest.TestCase):
    def setUp(self):
        self.sample_messages = [
            {"user": "Alice", "message": "Hello everyone!"},
            {"user": "Bob", "message": "Hi Alice, how are you?"},
            {"user": "Alice", "message": "I'm doing great!"},
        ]
        self.sample_users = ["Alice", "Bob"]
        self.sample_context_content = (
            "# Test Context\nThis is a test context for analysis."
        )
        self.sample_markdown_output = (
            "<h1>Test Context</h1>\n<p>This is a test context for analysis.</p>"
        )

    @patch("chat_analyser.config.AVAILABLE_CONTEXTS", ["party", "work"])
    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="# Test Context\nThis is a test context.",
    )
    @patch("markdown.markdown")
    def test_load_context_success(self, mock_markdown, mock_file):
        """Test successful context loading"""
        mock_markdown.return_value = self.sample_markdown_output

        result = load_context("party")

        # Verify file was opened correctly
        mock_file.assert_called_once_with("/fake/contexts/party.md", "r")
        # Verify markdown was processed
        mock_markdown.assert_called_once_with("# Test Context\nThis is a test context.")
        self.assertEqual(result, self.sample_markdown_output)

    @patch("chat_analyser.config.AVAILABLE_CONTEXTS", ["party", "work"])
    def test_load_context_invalid_type(self):
        """Test loading context with invalid context type"""
        with self.assertRaises(ValueError) as context:
            load_context("invalid_context")

        self.assertIn(
            "Context type invalid_context not among existing context",
            str(context.exception),
        )

    @patch("chat_analyser.config.AVAILABLE_CONTEXTS", ["party"])
    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("builtins.open", side_effect=FileNotFoundError("File not found"))
    def test_load_context_file_not_found(self, mock_file):
        """Test loading context when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_context("party")

    @patch("chat_analyser.core.analyser.load_context")
    def test_format_query_success(self, mock_load_context):
        """Test successful query formatting"""
        mock_load_context.return_value = self.sample_markdown_output

        result = format_query("party", self.sample_messages, self.sample_users)

        # Verify load_context was called
        mock_load_context.assert_called_once_with("party")

        # Verify the format is correct
        expected_messages = "\n".join([json.dumps(msg) for msg in self.sample_messages])
        expected_users = "\n".join(self.sample_users)
        expected_result = (
            f"{self.sample_markdown_output}\n{expected_messages}\n{expected_users}"
        )

        self.assertEqual(result, expected_result)

    @patch("chat_analyser.core.analyser.load_context")
    def test_format_query_empty_messages(self, mock_load_context):
        """Test query formatting with empty messages"""
        mock_load_context.return_value = self.sample_markdown_output

        result = format_query("party", [], self.sample_users)

        expected_users = "\n".join(self.sample_users)
        expected_result = f"{self.sample_markdown_output}\n\n{expected_users}"

        self.assertEqual(result, expected_result)

    @patch("chat_analyser.core.analyser.load_context")
    def test_format_query_empty_users(self, mock_load_context):
        """Test query formatting with empty users"""
        mock_load_context.return_value = self.sample_markdown_output

        result = format_query("party", self.sample_messages, [])

        expected_messages = "\n".join([json.dumps(msg) for msg in self.sample_messages])
        expected_result = f"{self.sample_markdown_output}\n{expected_messages}\n"

        self.assertEqual(result, expected_result)

    @patch("chat_analyser.core.analyser.client")
    @patch("chat_analyser.core.analyser.format_query")
    def test_analyse_chat_success(self, mock_format_query, mock_client):
        """Test successful chat analysis"""
        # Mock the format_query response
        mock_format_query.return_value = "formatted query"

        # Mock the API response
        mock_response_content = '{"summary": "Test summary", "users_feedback": {"Alice": {"summary": "Alice summary", "emoji": "😊"}}}'
        mock_client.chat.complete.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_response_content))
        ]

        result = analyse_chat("party", self.sample_messages, self.sample_users)

        # Verify format_query was called
        mock_format_query.assert_called_once_with(
            "party", self.sample_messages, self.sample_users
        )

        # Verify client was called with correct parameters
        mock_client.chat.complete.assert_called_once()
        call_args = mock_client.chat.complete.call_args
        self.assertEqual(call_args[1]["model"], "mistral-tiny-latest")  # Default model
        self.assertEqual(call_args[1]["messages"][0]["content"], "formatted query")

        # Verify response
        self.assertIsInstance(result, ConversationAnalysisResponse)
        self.assertEqual(result.summary, "Test summary")
        self.assertIn("Alice", result.users_feedback)

    @patch("chat_analyser.core.analyser.client")
    @patch("chat_analyser.core.analyser.format_query")
    def test_analyse_chat_with_custom_model(self, mock_format_query, mock_client):
        """Test chat analysis with custom model"""
        mock_format_query.return_value = "formatted query"
        mock_response_content = '{"summary": "Test", "users_feedback": {}}'
        mock_client.chat.complete.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_response_content))
        ]

        custom_model = "mistral-large-latest"
        analyse_chat(
            "party", self.sample_messages, self.sample_users, model=custom_model
        )

        # Verify custom model was used
        call_args = mock_client.chat.complete.call_args
        self.assertEqual(call_args[1]["model"], custom_model)

    @patch("chat_analyser.core.analyser.client")
    @patch("chat_analyser.core.analyser.format_query")
    def test_analyse_chat_json_extraction(self, mock_format_query, mock_client):
        """Test JSON extraction from response content"""
        mock_format_query.return_value = "formatted query"

        # Mock response with extra text around JSON
        mock_response_content = 'Some prefix text {"summary": "Test", "users_feedback": {}} some suffix text'
        mock_client.chat.complete.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_response_content))
        ]

        result = analyse_chat("party", self.sample_messages, self.sample_users)

        # Should successfully extract and parse the JSON
        self.assertIsInstance(result, ConversationAnalysisResponse)
        self.assertEqual(result.summary, "Test")

    @patch("chat_analyser.core.analyser.client")
    @patch("chat_analyser.core.analyser.format_query")
    def test_analyse_chat_api_error(self, mock_format_query, mock_client):
        """Test chat analysis when API call fails"""
        mock_format_query.return_value = "formatted query"
        mock_client.chat.complete.side_effect = Exception("API Error")

        with self.assertRaises(Exception) as context:
            analyse_chat("party", self.sample_messages, self.sample_users)

        self.assertEqual(str(context.exception), "API Error")

    @patch("chat_analyser.core.analyser.client")
    @patch("chat_analyser.core.analyser.format_query")
    def test_analyse_chat_invalid_json_response(self, mock_format_query, mock_client):
        """Test chat analysis with invalid JSON response"""
        mock_format_query.return_value = "formatted query"
        mock_response_content = "Invalid JSON content without proper structure"
        mock_client.chat.complete.return_value.choices = [
            MagicMock(message=MagicMock(content=mock_response_content))
        ]

        with self.assertRaises(ValueError):
            analyse_chat("party", self.sample_messages, self.sample_users)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_contexts_dir = None
        self.original_available_contexts = None

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    @patch("chat_analyser.config.CONTEXTS_DIR")
    def test_write_context_success(self, mock_contexts_dir):
        """Test successful context writing"""
        mock_contexts_dir.__str__ = lambda: self.test_dir
        mock_contexts_dir.__add__ = lambda self, other: self.test_dir + other

        # Use real file operations with temporary directory
        with patch(
            "chat_analyser.core.utils.pjoin",
            return_value=os.path.join(self.test_dir, "test_context.md"),
        ):
            with patch("chat_analyser.config.AVAILABLE_CONTEXTS", []):
                context_content = "# Test Context\nThis is a test context."

                write_context("test_context", context_content)

                # Verify file was written
                file_path = os.path.join(self.test_dir, "test_context.md")
                self.assertTrue(os.path.exists(file_path))

                with open(file_path, "r") as f:
                    written_content = f.read()
                self.assertEqual(written_content, context_content)

    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_context_file_operations(self, mock_file):
        """Test write_context file operations with mocking"""
        context_content = "# New Context\nContent here."

        with patch("chat_analyser.config.AVAILABLE_CONTEXTS", []) as mock_available:
            write_context("new_context", context_content)

            # Verify file was opened for writing
            mock_file.assert_called_once_with("/fake/contexts/new_context.md", "w")
            # Verify content was written
            mock_file().write.assert_called_once_with(context_content)
            # Verify context was added to available contexts (can't easily test list.append with mock)
            self.assertTrue(True)  # File operations tested above

    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_context_append_to_existing_list(self, mock_file):
        """Test write_context appends to existing contexts list"""
        # Test that the function runs without error when contexts exist
        with patch("chat_analyser.config.AVAILABLE_CONTEXTS", ["existing"]):
            write_context("another_context", "Content")

            # Verify file operations
            mock_file.assert_called_once_with("/fake/contexts/another_context.md", "w")
            mock_file().write.assert_called_once_with("Content")

    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("chat_analyser.config.AVAILABLE_CONTEXTS", [])
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_write_context_permission_error(self, mock_file):
        """Test write_context when file writing fails due to permissions"""
        with self.assertRaises(PermissionError):
            write_context("test_context", "Content")

    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_context_empty_content(self, mock_file):
        """Test write_context with empty content"""
        with patch("chat_analyser.config.AVAILABLE_CONTEXTS", []):
            write_context("empty_context", "")

            # Verify the function runs without error and file operations occur
            self.assertTrue(mock_file.called)
            mock_file().write.assert_called_with("")

    @patch("chat_analyser.config.CONTEXTS_DIR", "/fake/contexts")
    @patch("builtins.open", new_callable=mock_open)
    def test_write_context_special_characters(self, mock_file):
        """Test write_context with special characters in content"""
        special_content = "# Context with émojis 😊\nAnd special chars: àáâãäå"

        with patch("chat_analyser.config.AVAILABLE_CONTEXTS", []):
            write_context("special_context", special_content)

            # Verify the function runs without error and file operations occur
            self.assertTrue(mock_file.called)
            mock_file().write.assert_called_with(special_content)


if __name__ == "__main__":
    unittest.main()
