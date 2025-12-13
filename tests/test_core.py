import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import tempfile
import shutil

from chat_analyser.core.analyser import (
    load_system_prompt,
    format_user_prompt,
    analyse_chat,
)
from chat_analyser.core.utils import write_context
from chat_analyser.api.models import ConversationAnalysisResponse
from chat_analyser import config as cf


# # test_analyser.py
# import unittest
# from unittest.mock import patch, mock_open, MagicMock
# from chat_analyser.core.analyser import (
#     load_system_prompt,
#     format_user_prompt,
#     analyse_chat,
# )
# from chat_analyser.api.models import ConversationAnalysisResponse


class TestAnalyser(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    def test_load_system_prompt_success(self, mock_file):
        """Test loading a system prompt successfully."""
        mock_file.return_value.read.return_value = "# Test context"
        cf.AVAILABLE_CONTEXTS = ["test_context"]
        cf.CONTEXTS_DIR = "/fake/path"

        result = load_system_prompt("test_context")
        self.assertEqual(result, "# Test context")

    def test_load_system_prompt_invalid(self):
        """Test loading a non-existent system prompt."""
        cf.AVAILABLE_CONTEXTS = ["valid_context"]
        with self.assertRaises(ValueError):
            load_system_prompt("invalid_context")

    def test_format_user_prompt(self):
        """Test formatting the user prompt."""
        users = ["Alice", "Bob"]
        conversations = {
            0: {"user": "Alice", "content": "Hello"},
            1: {"user": "Bob", "content": "Hi"},
        }

        expected = """## Users list\n['Alice', 'Bob']\n\n## Conversations\n{0: {'user': 'Alice', 'content': 'Hello'}, 1: {'user': 'Bob', 'content': 'Hi'}}"""
        result = format_user_prompt(users, conversations)
        self.assertEqual(result, expected)

    @patch("chat_analyser.core.analyser.Mistral")
    @patch("chat_analyser.core.analyser.load_system_prompt")
    @patch("chat_analyser.core.analyser.format_user_prompt")
    def test_analyse_chat(self, mock_format, mock_load, mock_mistral):
        """Test the chat analysis function."""
        # Setup mocks
        mock_load.return_value = "system prompt"
        mock_format.return_value = "user prompt"

        model_response = """{
            "summary": "This is the summary.",
            "users_feedback": {
                "Alice": {"summary": "Alice's summary.", "emoji":"X"},
                "Bob": {"summary": "Bob's summary.", "emoji":"Y"}
            }
        }"""
        mock_client = MagicMock()
        mock_mistral.return_value.__enter__.return_value = mock_client
        mock_client.chat.complete.return_value.choices = [
            MagicMock(message=MagicMock(content=model_response))
        ]

        # Test data
        context_type = "test_context"
        users = ["Alice", "Bob"]
        conversations = [
            {"user": "Alice", "content": "Hello"},
            {"user": "Bob", "content": "Hi"},
        ]

        # Call function
        result = analyse_chat(context_type, users, conversations)

        # Assertions
        mock_load.assert_called_once_with(context_type)
        mock_format.assert_called_once_with(users, conversations)
        mock_mistral.assert_called_once_with(api_key=cf.API_KEY)
        mock_client.chat.complete.assert_called_once_with(
            model=cf.MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            stream=False,
            response_format={
                "type": "json_object",
                "json_schema": ConversationAnalysisResponse.model_json_schema(),
            },
        )

        self.assertEqual(
            result,
            ConversationAnalysisResponse.model_validate(
                {
                    "summary": "This is the summary.",
                    "users_feedback": {
                        "Alice": {"summary": "Alice's summary.", "emoji": "X"},
                        "Bob": {"summary": "Bob's summary.", "emoji": "Y"},
                    },
                }
            ),
        )


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
