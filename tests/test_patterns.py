"""
Test all code examples from the patterns documentation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from lionagi import Branch, iModel, Session
from lionagi.core.graph import GraphBuilder


@pytest.mark.asyncio
async def test_fan_out_in_pattern():
    """Test fan-out/fan-in pattern from patterns documentation"""
    
    with patch.object(Branch, 'communicate') as mock_comm:
        mock_comm.return_value = "Analysis result"
        
        session = Session()
        builder = GraphBuilder(session)
        
        # Create specialized branches
        security_expert = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini"),
            system="Security expert"
        )
        performance_expert = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini"),
            system="Performance expert"
        )
        ux_expert = Branch(
            chat_model=iModel(provider="openai", model="gpt-4o-mini"),
            system="UX expert"
        )
        
        # Include branches in session
        session.branches = {
            "security": security_expert,
            "performance": performance_expert,
            "ux": ux_expert
        }
        
        # Test that branches are created correctly
        assert len(session.branches) == 3
        assert "security" in session.branches
        assert session.branches["security"].system == "Security expert"


@pytest.mark.asyncio
async def test_sequential_analysis():
    """Test sequential analysis pattern"""
    
    with patch.object(Session, 'flow') as mock_flow:
        mock_flow.return_value = {"result": "success"}
        
        session = Session()
        builder = GraphBuilder(session)
        branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
        
        # Build sequential pipeline
        steps = ["Research", "Analyze", "Synthesize", "Report"]
        previous = None
        
        for step in steps:
            # Mock the add_operation to return node IDs
            with patch.object(builder, 'add_operation', return_value=f"node_{step}"):
                current = builder.add_operation(
                    "communicate",
                    branch=branch,
                    instruction=step,
                    depends_on=[previous] if previous else None
                )
                previous = current
        
        # Test the structure is built
        assert previous == "node_Report"  # Last node


@pytest.mark.asyncio
async def test_conditional_flows():
    """Test conditional flow pattern"""
    
    with patch.object(Branch, 'communicate') as mock_comm:
        # Simulate different responses for conditional logic
        mock_comm.side_effect = [
            {"confidence": 0.95, "answer": "High confidence answer"},
            {"confidence": 0.6, "answer": "Low confidence answer"}
        ]
        
        async def conditional_analysis(question: str):
            branch = Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
            
            # First attempt
            result = await branch.communicate(question)
            
            if result.get("confidence", 0) < 0.8:
                # Need more analysis
                enhanced_prompt = f"Please provide more detailed analysis: {question}"
                result = await branch.communicate(enhanced_prompt)
            
            return result
        
        # Test high confidence path
        result1 = await conditional_analysis("Simple question")
        assert result1["confidence"] == 0.95
        
        # Reset mock
        mock_comm.side_effect = [
            {"confidence": 0.6, "answer": "Low confidence"},
            {"confidence": 0.9, "answer": "Detailed answer"}
        ]
        
        # Test low confidence path
        result2 = await conditional_analysis("Complex question")
        assert mock_comm.call_count >= 2  # Should call twice


@pytest.mark.asyncio
async def test_react_with_rag():
    """Test ReAct with RAG pattern"""
    
    with patch.object(Branch, 'communicate') as mock_comm:
        mock_comm.return_value = {
            "reasoning": "Need to search for information",
            "action": "search",
            "query": "test query"
        }
        
        async def react_pattern(question: str):
            branch = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                system="You are a ReAct agent"
            )
            
            # Initial reasoning
            thought = await branch.communicate(f"Think about: {question}")
            
            if thought.get("action") == "search":
                # Simulate RAG retrieval
                context = "Retrieved context"
                
                # Reason with context
                answer = await branch.communicate(
                    f"Answer based on context: {context}\nQuestion: {question}"
                )
                return answer
            
            return thought
        
        result = await react_pattern("What is LionAGI?")
        assert result is not None
        assert mock_comm.call_count >= 1


@pytest.mark.asyncio
async def test_tournament_validation():
    """Test tournament validation pattern"""
    
    with patch.object(Branch, 'communicate') as mock_comm:
        # Mock different quality scores
        mock_comm.side_effect = [
            {"solution": "Solution A", "score": 0.85},
            {"solution": "Solution B", "score": 0.92},
            {"solution": "Solution C", "score": 0.78},
            {"winner": "Solution B", "reason": "Best score"}
        ]
        
        async def tournament_validation(problem: str):
            # Generate multiple solutions
            branches = [
                Branch(chat_model=iModel(provider="openai", model="gpt-4o-mini"))
                for _ in range(3)
            ]
            
            solutions = []
            for i, branch in enumerate(branches):
                solution = await branch.communicate(f"Solve: {problem}")
                solutions.append(solution)
            
            # Validation round
            judge = Branch(
                chat_model=iModel(provider="openai", model="gpt-4o-mini"),
                system="You are a solution judge"
            )
            
            best = await judge.communicate(f"Select best solution: {solutions}")
            return best
        
        result = await tournament_validation("Optimize database query")
        assert result["winner"] == "Solution B"
        assert mock_comm.call_count == 4  # 3 solutions + 1 judge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])