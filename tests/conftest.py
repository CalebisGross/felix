"""
Pytest configuration and shared fixtures for Felix Framework tests.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Add src to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a mock LLM response for testing purposes.",
        "tokens_used": 50,
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "response_time": 0.5,
        "model": "test-model",
        "provider": "mock",
        "finish_reason": "stop"
    }


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    from src.llm.base_provider import BaseLLMProvider, LLMResponse

    class MockProvider(BaseLLMProvider):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            self.should_fail = False

        def complete(self, request):
            self.call_count += 1
            if self.should_fail:
                raise Exception("Mock provider failure")
            return LLMResponse(
                content="Mock response",
                tokens_used=50,
                prompt_tokens=20,
                completion_tokens=30,
                response_time=0.1,
                model="mock-model",
                provider="mock",
                finish_reason="stop"
            )

        def complete_streaming(self, request, callback):
            self.call_count += 1
            if self.should_fail:
                raise Exception("Mock provider failure")
            callback("Mock ")
            callback("streaming ")
            callback("response")
            return LLMResponse(
                content="Mock streaming response",
                tokens_used=50,
                prompt_tokens=20,
                completion_tokens=30,
                response_time=0.1,
                model="mock-model",
                provider="mock",
                finish_reason="stop"
            )

        def test_connection(self):
            return not self.should_fail

        def get_available_models(self):
            return ["mock-model-1", "mock-model-2"]

    return MockProvider()


@pytest.fixture
def mock_confidence_metrics():
    """Mock confidence metrics for testing spawning logic."""
    from src.agents.dynamic_spawning import ConfidenceMetrics, ConfidenceTrend

    return ConfidenceMetrics(
        current_average=0.6,
        trend=ConfidenceTrend.STABLE,
        volatility=0.1,
        time_window_minutes=5.0,
        agent_type_breakdown={"research": 0.7, "analysis": 0.6, "critic": 0.5},
        position_breakdown={"0.0-0.3": 0.65, "0.3-0.7": 0.6, "0.7-1.0": 0.55},
        recent_samples=[(0.0, 0.6), (1.0, 0.6), (2.0, 0.6)]
    )


@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing."""
    config_content = """
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "test-model"
  timeout: 120

fallbacks: []

router:
  retry_on_rate_limit: false
  verbose_logging: false

cost_tracking:
  enabled: true
  daily_limit_usd: 10.0
  monthly_limit_usd: 100.0
  alert_threshold: 0.8

aliases:
  fast: "test-model-fast"
  good: "test-model-good"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_llm_request():
    """Sample LLM request for testing."""
    from src.llm.base_provider import LLMRequest

    return LLMRequest(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is 2+2?",
        temperature=0.7,
        max_tokens=100,
        stream=False,
        agent_id="test-agent-1",
        model="test-model"
    )


@pytest.fixture
def mock_helix_geometry():
    """Mock helix geometry for testing."""
    from src.core.helix_geometry import HelixGeometry

    return HelixGeometry(
        top_radius=3.0,
        bottom_radius=0.5,
        height=8.0,
        turns=2
    )


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging
    # Clear all handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Reset to WARNING level
    logging.root.setLevel(logging.WARNING)
    yield


@pytest.fixture
def mock_central_post():
    """Mock CentralPost for testing agents."""
    mock = Mock()
    mock.register_agent = Mock()
    mock.unregister_agent = Mock()
    mock.post_message = Mock()
    mock.get_messages = Mock(return_value=[])
    mock.get_recent_confidences = Mock(return_value=[0.7, 0.8, 0.75])
    return mock


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (can be skipped with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "llm: Tests requiring LLM provider"
    )
    config.addinivalue_line(
        "markers", "providers: Tests for LLM providers"
    )
    config.addinivalue_line(
        "markers", "spawning: Tests for agent spawning logic"
    )
    config.addinivalue_line(
        "markers", "router: Tests for LLM router"
    )
