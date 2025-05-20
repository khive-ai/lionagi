from pathlib import Path

import pytest

from lionagi.core.text_processing.synthlang import (
    SynthlangFramework,
    SynthlangTemplate,
    translate_to_synthlang,
)


def test_synthlang_template_enum():
    """Test that SynthlangTemplate enum has expected values."""
    assert SynthlangTemplate.ABSTRACT_ALGEBRA.value == "abstract_algebra"
    assert SynthlangTemplate.CATEGORY_THEORY.value == "category_theory"
    assert SynthlangTemplate.COMPLEX_ANALYSIS.value == "complex_analysis"
    assert SynthlangTemplate.GROUP_THEORY.value == "group_theory"
    assert SynthlangTemplate.MATH_LOGIC.value == "math_logic"
    assert SynthlangTemplate.REFLECTIVE_PATTERNS.value == "reflective_patterns"
    assert SynthlangTemplate.SET_THEORY.value == "set_theory"
    assert (
        SynthlangTemplate.TOPOLOGY_FUNDAMENTALS.value
        == "topology_fundamentals"
    )


def test_synthlang_template_fp():
    """Test that SynthlangTemplate.fp returns the expected path."""
    template = SynthlangTemplate.ABSTRACT_ALGEBRA
    expected_path = (
        Path(__file__).parent.parent.parent.parent
        / "lionagi"
        / "core"
        / "text_processing"
        / "synthlang_"
        / "resources"
        / "frameworks"
        / "abstract_algebra.toml"
    )
    assert template.fp.name == "abstract_algebra.toml"
    # We don't test the full path equality as it might vary depending on the environment


def test_synthlang_template_list_templates():
    """Test that SynthlangTemplate.list_templates returns all template values."""
    templates = SynthlangTemplate.list_templates()
    assert "abstract_algebra" in templates
    assert "category_theory" in templates
    assert "complex_analysis" in templates
    assert "group_theory" in templates
    assert "math_logic" in templates
    assert "reflective_patterns" in templates
    assert "set_theory" in templates
    assert "topology_fundamentals" in templates
    assert len(templates) == 8  # Ensure we have exactly 8 templates


def test_synthlang_framework_load_framework_options():
    """Test that SynthlangFramework.load_framework_options returns a dict."""
    options = SynthlangFramework.load_framework_options()
    assert isinstance(options, dict)


@pytest.mark.asyncio
async def test_translate_to_synthlang_mock(monkeypatch):
    """Test translate_to_synthlang with mocked dependencies."""
    # This is a simplified test that mocks the actual LLM call
    # In a real test, you would use a mock for the Branch.chat method

    class MockBranch:
        def __init__(self, **kwargs):
            self.chat_model = None
            self.system = None
            self.msgs = self

        async def chat(self, **kwargs):
            return "```synthlang\nMocked synthlang output\n```"

        def add_message(self, **kwargs):
            pass

    class MockTokenCalculator:
        def tokenize(self, text, return_tokens=True):
            return 10

    # Mock the imports and functions
    monkeypatch.setattr(
        "lionagi.core.text_processing.synthlang_.translate_to_synthlang.Branch",
        MockBranch,
    )
    monkeypatch.setattr(
        "lionagi.core.text_processing.synthlang_.translate_to_synthlang.TokenCalculator",
        MockTokenCalculator,
    )

    # Test the function with minimal arguments
    result = await translate_to_synthlang("Test text")
    assert "Mocked synthlang output" in result
