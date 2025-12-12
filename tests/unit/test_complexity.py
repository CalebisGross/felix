"""
Unit tests for task complexity classification.

Tests the SynthesisEngine's classify_task_complexity method to ensure
simple, medium, and complex queries are categorized correctly.
"""

import pytest
from src.communication.synthesis_engine import SynthesisEngine


@pytest.mark.unit
class TestComplexityClassification:
    """Tests for task complexity classification."""

    @pytest.fixture
    def synthesis_engine(self, mock_llm_provider):
        """Create SynthesisEngine instance for testing."""
        from unittest.mock import Mock
        mock_central_post = Mock()
        return SynthesisEngine(mock_central_post, mock_llm_provider)

    def test_simple_factual_time_query(self, synthesis_engine):
        """Test that 'What time is it?' is classified as SIMPLE_FACTUAL."""
        result = synthesis_engine.classify_task_complexity("What time is it?")
        assert result == "SIMPLE_FACTUAL"

    def test_simple_factual_date_query(self, synthesis_engine):
        """Test that date queries are classified as SIMPLE_FACTUAL."""
        result = synthesis_engine.classify_task_complexity("What is today's date?")
        assert result == "SIMPLE_FACTUAL"

    def test_simple_factual_current_events(self, synthesis_engine):
        """Test that current event queries are SIMPLE_FACTUAL."""
        result = synthesis_engine.classify_task_complexity("Who is the current president?")
        assert result == "SIMPLE_FACTUAL"

    def test_medium_explain_query(self, synthesis_engine):
        """Test that 'explain' queries are MEDIUM complexity."""
        result = synthesis_engine.classify_task_complexity("Explain quantum computing")
        assert result == "MEDIUM"

    def test_medium_compare_query(self, synthesis_engine):
        """Test that 'compare' queries are MEDIUM complexity."""
        result = synthesis_engine.classify_task_complexity("Compare Python and JavaScript")
        assert result == "MEDIUM"

    def test_medium_how_to_query(self, synthesis_engine):
        """Test that 'how to' queries are MEDIUM complexity."""
        result = synthesis_engine.classify_task_complexity("How to build a REST API?")
        assert result == "MEDIUM"

    def test_complex_design_query(self, synthesis_engine):
        """Test that design queries are COMPLEX."""
        result = synthesis_engine.classify_task_complexity(
            "Design a microservices architecture for an e-commerce platform"
        )
        assert result == "COMPLEX"

    def test_complex_analytical_query(self, synthesis_engine):
        """Test that analytical queries are COMPLEX."""
        result = synthesis_engine.classify_task_complexity(
            "Analyze the impact of AI on the job market over the next decade"
        )
        assert result == "COMPLEX"

    def test_complex_open_ended_query(self, synthesis_engine):
        """Test that open-ended queries default to COMPLEX."""
        result = synthesis_engine.classify_task_complexity(
            "Tell me your thoughts on consciousness"
        )
        assert result == "COMPLEX"

    def test_empty_query_defaults_to_complex(self, synthesis_engine):
        """Test that empty queries default to COMPLEX."""
        result = synthesis_engine.classify_task_complexity("")
        assert result == "COMPLEX"

    def test_case_insensitive_matching(self, synthesis_engine):
        """Test that pattern matching is case-insensitive."""
        result1 = synthesis_engine.classify_task_complexity("WHAT TIME IS IT?")
        result2 = synthesis_engine.classify_task_complexity("what time is it?")
        result3 = synthesis_engine.classify_task_complexity("What Time Is It?")

        assert result1 == result2 == result3 == "SIMPLE_FACTUAL"
