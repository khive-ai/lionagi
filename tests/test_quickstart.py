"""
Test all code examples from the quickstart documentation.
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Mock the API calls to avoid actual API usage during tests
@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses"""
    with patch('lionagi.providers.openai.OpenAIProvider.invoke') as mock:
        mock.return_value = AsyncMock(return_value="Mocked response")
        yield mock


@pytest.mark.asyncio
async def test_quick_example_multi_perspective():
    """Test the multi-perspective analysis example from index.md"""
    from lionagi import Branch, iModel
    
    # Mock the chat model
    with patch.object(Branch, 'chat') as mock_chat:
        mock_chat.return_value = "Mocked analysis"
        
        async def multi_perspective_analysis():
            # Create specialized agents
            technical = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                system="You analyze technical feasibility"
            )
            business = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                system="You analyze business impact"
            )
            user = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                system="You analyze user experience"
            )
            
            question = "Should we rewrite our backend in Rust?"
            
            # Get all perspectives in parallel
            tech_view, biz_view, user_view = await asyncio.gather(
                technical.chat(question),
                business.chat(question),
                user.chat(question)
            )
            
            return {"technical": tech_view, "business": biz_view, "user": user_view}
        
        result = await multi_perspective_analysis()
        
        # Assertions
        assert isinstance(result, dict)
        assert "technical" in result
        assert "business" in result
        assert "user" in result
        assert mock_chat.call_count >= 3  # At least 3 calls


@pytest.mark.asyncio
async def test_first_agent():
    """Test the first agent example from quickstart"""
    from lionagi import Branch, iModel
    
    with patch.object(Branch, 'chat') as mock_chat:
        mock_chat.return_value = "REST APIs are..."
        
        async def main():
            agent = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
            response = await agent.chat("Explain REST APIs")
            return response
        
        result = await main()
        
        # Assertions
        assert result is not None
        assert isinstance(result, str)
        mock_chat.assert_called_once_with("Explain REST APIs")


def test_installation_import():
    """Test that basic imports work"""
    try:
        from lionagi import Branch, iModel, Session
        assert Branch is not None
        assert iModel is not None
        assert Session is not None
    except ImportError as e:
        pytest.fail(f"Failed to import lionagi components: {e}")


@pytest.mark.asyncio
async def test_simple_parallel():
    """Test simple parallel execution pattern"""
    from lionagi import Branch, iModel
    
    with patch.object(Branch, 'communicate') as mock_comm:
        mock_comm.return_value = "Mocked result"
        
        branch = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini")
        )
        
        tasks = ["Task 1", "Task 2", "Task 3"]
        results = await asyncio.gather(*[
            branch.communicate(task) for task in tasks
        ])
        
        # Assertions
        assert len(results) == 3
        assert all(r == "Mocked result" for r in results)
        assert mock_comm.call_count == 3


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])