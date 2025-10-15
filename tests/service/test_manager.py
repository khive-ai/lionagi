"""Tests for lionagi.service.manager module."""

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from lionagi.service.imodel import iModel
from lionagi.service.manager import iModelManager


class TestiModelManagerInit:
    """Test iModelManager initialization."""

    def test_init_empty(self):
        """Test initialization with no arguments."""
        manager = iModelManager()

        assert isinstance(manager.registry, dict)
        assert len(manager.registry) == 0

    def test_init_with_args(self):
        """Test initialization with positional iModel arguments."""
        # Create mock iModel instances with proper spec
        mock_model1 = create_autospec(iModel, instance=True)
        mock_model1.endpoint = MagicMock()
        mock_model1.endpoint.endpoint = "model1"
        mock_model2 = create_autospec(iModel, instance=True)
        mock_model2.endpoint = MagicMock()
        mock_model2.endpoint.endpoint = "model2"

        with patch("lionagi.service.manager.is_same_dtype", return_value=True):
            manager = iModelManager(mock_model1, mock_model2)

            assert "model1" in manager.registry
            assert "model2" in manager.registry
            assert manager.registry["model1"] is mock_model1
            assert manager.registry["model2"] is mock_model2

    def test_init_with_args_invalid_type_raises(self):
        """Test initialization with non-iModel args raises TypeError."""
        not_a_model = "not an iModel"

        with patch(
            "lionagi.service.manager.is_same_dtype", return_value=False
        ):
            with pytest.raises(
                TypeError, match="Input models are not instances of iModel"
            ):
                iModelManager(not_a_model)

    def test_init_with_kwargs(self):
        """Test initialization with keyword arguments."""
        mock_chat_model = create_autospec(iModel, instance=True)
        mock_parse_model = create_autospec(iModel, instance=True)

        manager = iModelManager(chat=mock_chat_model, parse=mock_parse_model)

        assert "chat" in manager.registry
        assert "parse" in manager.registry
        assert manager.registry["chat"] is mock_chat_model
        assert manager.registry["parse"] is mock_parse_model

    def test_init_with_args_and_kwargs(self):
        """Test initialization with both args and kwargs."""
        mock_model1 = create_autospec(iModel, instance=True)
        mock_model1.endpoint = MagicMock()
        mock_model1.endpoint.endpoint = "model1"
        mock_custom_model = create_autospec(iModel, instance=True)

        with patch("lionagi.service.manager.is_same_dtype", return_value=True):
            manager = iModelManager(mock_model1, custom=mock_custom_model)

            assert "model1" in manager.registry
            assert "custom" in manager.registry
            assert manager.registry["model1"] is mock_model1
            assert manager.registry["custom"] is mock_custom_model


class TestiModelManagerProperties:
    """Test iModelManager property accessors."""

    def test_chat_property_exists(self):
        """Test chat property returns registered chat model."""
        manager = iModelManager()
        mock_chat_model = MagicMock()
        manager.registry["chat"] = mock_chat_model

        result = manager.chat
        assert result is mock_chat_model

    def test_chat_property_none_if_not_registered(self):
        """Test chat property returns None if not registered."""
        manager = iModelManager()

        result = manager.chat
        assert result is None

    def test_parse_property_exists(self):
        """Test parse property returns registered parse model."""
        manager = iModelManager()
        mock_parse_model = MagicMock()
        manager.registry["parse"] = mock_parse_model

        result = manager.parse
        assert result is mock_parse_model

    def test_parse_property_none_if_not_registered(self):
        """Test parse property returns None if not registered."""
        manager = iModelManager()

        result = manager.parse
        assert result is None


class TestiModelManagerRegisterIModel:
    """Test iModelManager.register_imodel method."""

    def test_register_imodel_valid(self):
        """Test registering a valid iModel."""
        manager = iModelManager()
        mock_model = create_autospec(iModel, instance=True)

        manager.register_imodel("test_model", mock_model)

        assert "test_model" in manager.registry
        assert manager.registry["test_model"] is mock_model

    def test_register_imodel_invalid_type_raises(self):
        """Test registering non-iModel raises TypeError."""
        manager = iModelManager()
        not_a_model = "not an iModel"

        with pytest.raises(
            TypeError, match="Input model is not an instance of iModel"
        ):
            manager.register_imodel("invalid", not_a_model)

    def test_register_imodel_overwrites_existing(self):
        """Test registering overwrites existing model with same name."""
        manager = iModelManager()
        mock_model1 = create_autospec(iModel, instance=True)
        mock_model2 = create_autospec(iModel, instance=True)

        manager.register_imodel("model", mock_model1)
        assert manager.registry["model"] is mock_model1

        manager.register_imodel("model", mock_model2)
        assert manager.registry["model"] is mock_model2

    def test_register_multiple_models(self):
        """Test registering multiple different models."""
        manager = iModelManager()
        mock_model1 = create_autospec(iModel, instance=True)
        mock_model2 = create_autospec(iModel, instance=True)
        mock_model3 = create_autospec(iModel, instance=True)

        manager.register_imodel("chat", mock_model1)
        manager.register_imodel("parse", mock_model2)
        manager.register_imodel("embed", mock_model3)

        assert len(manager.registry) == 3
        assert manager.registry["chat"] is mock_model1
        assert manager.registry["parse"] is mock_model2
        assert manager.registry["embed"] is mock_model3
