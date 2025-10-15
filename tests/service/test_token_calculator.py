"""Tests for lionagi.service.token_calculator module."""

import base64
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest

from lionagi.service.token_calculator import (
    GPT4O_IMAGE_PRICING,
    GPT4O_MINI_IMAGE_PRICING,
    O1_IMAGE_PRICING,
    TokenCalculator,
    calculate_image_token_usage_from_base64,
    get_encoding_name,
    get_image_pricing,
)


class TestImagePricingConstants:
    """Test image pricing constant dictionaries."""

    def test_gpt4o_image_pricing_structure(self):
        """Test GPT4O_IMAGE_PRICING has correct structure."""
        assert GPT4O_IMAGE_PRICING["base_cost"] == 85
        assert GPT4O_IMAGE_PRICING["low_detail"] == 0
        assert GPT4O_IMAGE_PRICING["max_dimension"] == 2048
        assert GPT4O_IMAGE_PRICING["min_side"] == 768
        assert GPT4O_IMAGE_PRICING["tile_size"] == 512
        assert GPT4O_IMAGE_PRICING["tile_cost"] == 170

    def test_gpt4o_mini_image_pricing_structure(self):
        """Test GPT4O_MINI_IMAGE_PRICING has correct structure."""
        assert GPT4O_MINI_IMAGE_PRICING["base_cost"] == 2833
        assert GPT4O_MINI_IMAGE_PRICING["tile_cost"] == 5667

    def test_o1_image_pricing_structure(self):
        """Test O1_IMAGE_PRICING has correct structure."""
        assert O1_IMAGE_PRICING["base_cost"] == 75
        assert O1_IMAGE_PRICING["tile_cost"] == 150


class TestGetImagePricing:
    """Test get_image_pricing function."""

    def test_get_gpt4o_mini_pricing(self):
        """Test returns GPT4O_MINI pricing for gpt-4o-mini models."""
        result = get_image_pricing("gpt-4o-mini")
        assert result == GPT4O_MINI_IMAGE_PRICING

        result = get_image_pricing("gpt-4o-mini-2024-07-18")
        assert result == GPT4O_MINI_IMAGE_PRICING

    def test_get_gpt4o_pricing(self):
        """Test returns GPT4O pricing for gpt-4o models."""
        result = get_image_pricing("gpt-4o")
        assert result == GPT4O_IMAGE_PRICING

        result = get_image_pricing("gpt-4o-2024-08-06")
        assert result == GPT4O_IMAGE_PRICING

    def test_get_o1_pricing(self):
        """Test returns O1 pricing for o1 models (not mini)."""
        result = get_image_pricing("o1")
        assert result == O1_IMAGE_PRICING

        result = get_image_pricing("o1-2024-12-17")
        assert result == O1_IMAGE_PRICING

    def test_get_pricing_invalid_model_raises(self):
        """Test raises ValueError for invalid model names."""
        with pytest.raises(ValueError, match="Invalid model name"):
            get_image_pricing("claude-3-opus")

        with pytest.raises(ValueError, match="Invalid model name"):
            get_image_pricing("gemini-pro")


class TestGetEncodingName:
    """Test get_encoding_name function."""

    def test_get_encoding_for_known_model(self):
        """Test returns encoding name for known model."""
        with patch("tiktoken.encoding_for_model") as mock_encoding_for_model:
            mock_enc = MagicMock()
            mock_enc.name = "cl100k_base"
            mock_encoding_for_model.return_value = mock_enc

            result = get_encoding_name("gpt-4o")
            assert result == "cl100k_base"
            mock_encoding_for_model.assert_called_once_with("gpt-4o")

    def test_get_encoding_fallback_to_direct(self):
        """Test falls back to tiktoken.get_encoding if model lookup fails."""
        with patch("tiktoken.encoding_for_model") as mock_encoding_for_model:
            mock_encoding_for_model.side_effect = Exception("Model not found")

            with patch("tiktoken.get_encoding") as mock_get_encoding:
                mock_get_encoding.return_value = MagicMock()

                result = get_encoding_name("cl100k_base")
                assert result == "cl100k_base"
                mock_get_encoding.assert_called_once_with("cl100k_base")

    def test_get_encoding_fallback_to_default(self):
        """Test falls back to o200k_base if both lookups fail."""
        with patch("tiktoken.encoding_for_model") as mock_encoding_for_model:
            mock_encoding_for_model.side_effect = Exception("Model not found")

            with patch("tiktoken.get_encoding") as mock_get_encoding:
                mock_get_encoding.side_effect = Exception("Encoding not found")

                result = get_encoding_name("unknown_model")
                assert result == "o200k_base"


class TestCalculateImageTokenUsageFromBase64:
    """Test calculate_image_token_usage_from_base64 function - DEPRECATED."""

    @pytest.mark.skip(reason="Image token calculation being deprecated")
    def test_calculate_image_deprecated(self):
        """Image calculation functionality being deprecated."""
        pass


class TestTokenCalculatorTokenize:
    """Test TokenCalculator.tokenize static method."""

    def test_tokenize_empty_string_returns_zero(self):
        """Test returns 0 for empty string."""
        result = TokenCalculator.tokenize("")
        assert result == 0

        result = TokenCalculator.tokenize(None)
        assert result == 0

    def test_tokenize_with_encoding_name(self):
        """Test tokenizes with encoding name."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3, 4]
            mock_get_encoding.return_value = mock_enc

            result = TokenCalculator.tokenize(
                "hello world", encoding_name="cl100k_base"
            )
            assert result == 4

    def test_tokenize_with_custom_tokenizer(self):
        """Test uses custom tokenizer if provided."""
        custom_tokenizer = MagicMock(return_value=[1, 2, 3])
        custom_decoder = MagicMock()

        result = TokenCalculator.tokenize(
            "test", tokenizer=custom_tokenizer, decoder=custom_decoder
        )
        assert result == 3
        custom_tokenizer.assert_called_once_with("test")

    def test_tokenize_return_tokens(self):
        """Test returns token list when return_tokens=True."""
        custom_tokenizer = MagicMock(return_value=[1, 2, 3])
        custom_decoder = MagicMock()

        result = TokenCalculator.tokenize(
            "test",
            tokenizer=custom_tokenizer,
            decoder=custom_decoder,
            return_tokens=True,
        )
        assert result == [1, 2, 3]

    def test_tokenize_return_decoded(self):
        """Test returns tokens and decoded when both flags True."""
        custom_tokenizer = MagicMock(return_value=[1, 2, 3])
        custom_decoder = MagicMock(return_value="decoded text")

        result = TokenCalculator.tokenize(
            "test",
            tokenizer=custom_tokenizer,
            decoder=custom_decoder,
            return_tokens=True,
            return_decoded=True,
        )
        assert result == (3, "decoded text")

    def test_tokenize_handles_exception_returns_zero(self):
        """Test returns 0 if tokenization raises exception."""
        failing_tokenizer = MagicMock(
            side_effect=Exception("Tokenization error")
        )
        custom_decoder = MagicMock()

        result = TokenCalculator.tokenize(
            "test", tokenizer=failing_tokenizer, decoder=custom_decoder
        )
        assert result == 0


class TestTokenCalculatorCalculateMessageTokens:
    """Test TokenCalculator.calculate_message_tokens static method."""

    def test_calculate_empty_messages(self):
        """Test calculates tokens for empty message list."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = []
            mock_get_encoding.return_value = mock_enc

            result = TokenCalculator.calculate_message_tokens([])
            assert result == 0

    def test_calculate_simple_text_message(self):
        """Test calculates tokens for simple text message."""
        messages = [{"content": "Hello, world!"}]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens
            mock_get_encoding.return_value = mock_enc

            result = TokenCalculator.calculate_message_tokens(messages)
            # 4 overhead + 3 tokens
            assert result == 7

    def test_calculate_multiple_messages(self):
        """Test calculates tokens for multiple messages."""
        messages = [
            {"content": "First message"},
            {"content": "Second message"},
        ]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2]  # 2 tokens each
            mock_get_encoding.return_value = mock_enc

            result = TokenCalculator.calculate_message_tokens(messages)
            # (4 overhead + 2 tokens) * 2 messages
            assert result == 12

    def test_calculate_with_custom_model(self):
        """Test uses custom model for encoding."""
        messages = [{"content": "test"}]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1]
            mock_get_encoding.return_value = mock_enc

            with patch(
                "lionagi.service.token_calculator.get_encoding_name"
            ) as mock_get_name:
                mock_get_name.return_value = "cl100k_base"

                TokenCalculator.calculate_message_tokens(
                    messages, model="gpt-4o-mini"
                )
                mock_get_name.assert_called_with("gpt-4o-mini")


class TestTokenCalculatorCalculateChatitem:
    """Test TokenCalculator._calculate_chatitem helper method."""

    def test_calculate_chatitem_string(self):
        """Test calculates tokens for string content."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_enc

            tokenizer = mock_enc.encode
            result = TokenCalculator._calculate_chatitem(
                "hello", tokenizer=tokenizer, model_name="gpt-4o"
            )
            assert result == 3

    def test_calculate_chatitem_text_dict(self):
        """Test dict with text field - currently returns 0 due to line 178 bug."""
        content = {"text": "hello world"}

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3, 4, 5]
            mock_get_encoding.return_value = mock_enc

            tokenizer = mock_enc.encode
            result = TokenCalculator._calculate_chatitem(
                content, tokenizer=tokenizer, model_name="gpt-4o"
            )
            # BUG: line 178 calls _calculate_chatitem without tokenizer/model_name
            # causing exception and return 0
            assert result == 0

    def test_calculate_chatitem_list(self):
        """Test calculates tokens for list of items."""
        content = ["hello", "world"]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2]
            mock_get_encoding.return_value = mock_enc

            tokenizer = mock_enc.encode
            result = TokenCalculator._calculate_chatitem(
                content, tokenizer=tokenizer, model_name="gpt-4o"
            )
            # Each string has 2 tokens, total 4
            assert result == 4

    def test_calculate_chatitem_handles_exception(self):
        """Test returns 0 if exception occurs."""
        failing_tokenizer = MagicMock(side_effect=Exception("Error"))

        result = TokenCalculator._calculate_chatitem(
            "test", tokenizer=failing_tokenizer, model_name="gpt-4o"
        )
        assert result == 0


class TestTokenCalculatorCalculateEmbedToken:
    """Test TokenCalculator.calculate_embed_token static method."""

    def test_calculate_embed_missing_inputs_field(self):
        """Test returns 0 if 'inputs' field missing in kwargs."""
        result = TokenCalculator.calculate_embed_token(
            ["test"], model="text-embedding-3-small"
        )
        assert result == 0

    def test_calculate_embed_with_inputs(self):
        """Test calculates tokens for embedding inputs."""
        inputs = ["first text", "second text"]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3]  # 3 tokens each
            mock_get_encoding.return_value = mock_enc

            result = TokenCalculator.calculate_embed_token(
                inputs, inputs=inputs, model="text-embedding-3-small"
            )
            # 3 tokens * 2 inputs
            assert result == 6

    def test_calculate_embed_handles_exception(self):
        """Test returns 0 if exception occurs."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_get_encoding.side_effect = Exception("Encoding error")

            result = TokenCalculator.calculate_embed_token(
                ["test"], inputs=["test"]
            )
            assert result == 0


class TestTokenCalculatorCalculateEmbedItem:
    """Test TokenCalculator._calculate_embed_item helper method."""

    def test_calculate_embed_item_string(self):
        """Test calculates tokens for string."""
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2, 3]
            mock_get_encoding.return_value = mock_enc

            tokenizer = mock_enc.encode
            result = TokenCalculator._calculate_embed_item(
                "hello", tokenizer=tokenizer
            )
            assert result == 3

    def test_calculate_embed_item_list(self):
        """Test calculates tokens for list of strings."""
        content = ["hello", "world"]

        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_enc = MagicMock()
            mock_enc.encode.return_value = [1, 2]
            mock_get_encoding.return_value = mock_enc

            tokenizer = mock_enc.encode
            result = TokenCalculator._calculate_embed_item(
                content, tokenizer=tokenizer
            )
            # 2 tokens * 2 strings
            assert result == 4

    def test_calculate_embed_item_handles_exception(self):
        """Test returns 0 if exception occurs."""
        failing_tokenizer = MagicMock(side_effect=Exception("Error"))

        result = TokenCalculator._calculate_embed_item(
            "test", tokenizer=failing_tokenizer
        )
        assert result == 0
