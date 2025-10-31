"""
Unit tests for LLM provider system.

Tests the provider abstraction, individual providers, and router.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm.base_provider import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthenticationError
)


@pytest.mark.unit
@pytest.mark.providers
class TestBaseLLMProvider:
    """Tests for base provider interface."""

    def test_llm_request_creation(self):
        """Test creating LLMRequest with required fields."""
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello",
            temperature=0.7,
            max_tokens=100
        )

        assert request.system_prompt == "You are helpful"
        assert request.user_prompt == "Hello"
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.stream == False  # Default
        assert request.agent_id is None  # Optional

    def test_llm_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Test response",
            tokens_used=50,
            prompt_tokens=20,
            completion_tokens=30,
            response_time=0.5,
            model="test-model",
            provider="test"
        )

        assert response.content == "Test response"
        assert response.tokens_used == 50
        assert response.prompt_tokens == 20
        assert response.completion_tokens == 30
        assert response.response_time == 0.5
        assert response.model == "test-model"
        assert response.provider == "test"
        assert response.finish_reason == "stop"  # Default


@pytest.mark.unit
@pytest.mark.providers
class TestLMStudioProvider:
    """Tests for LM Studio provider wrapper."""

    @patch('src.llm.providers.lm_studio_provider.LMStudioClient')
    def test_lm_studio_provider_complete(self, mock_client_class):
        """Test LM Studio provider completion."""
        from src.llm.providers.lm_studio_provider import LMStudioProvider

        # Setup mock
        mock_client = Mock()
        mock_lm_response = Mock()
        mock_lm_response.content = "Test response"
        mock_lm_response.tokens_used = 50
        mock_lm_response.model = "local-model"
        mock_client.complete.return_value = mock_lm_response
        mock_client_class.return_value = mock_client

        # Create provider
        provider = LMStudioProvider(base_url="http://localhost:1234/v1")

        # Make request
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello",
            temperature=0.7
        )

        response = provider.complete(request)

        # Verify
        assert response.content == "Test response"
        assert response.tokens_used == 50
        assert response.provider == "lm_studio"
        assert mock_client.complete.called

    @patch('src.llm.providers.lm_studio_provider.LMStudioClient')
    def test_lm_studio_provider_cost_is_zero(self, mock_client_class):
        """Test that LM Studio provider reports zero cost."""
        from src.llm.providers.lm_studio_provider import LMStudioProvider

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        provider = LMStudioProvider()
        cost = provider.estimate_cost(tokens_used=1000, model="any-model")

        assert cost == 0.0, "LM Studio should have zero cost (local)"


@pytest.mark.unit
@pytest.mark.providers
class TestProviderConfig:
    """Tests for provider configuration loader."""

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        from src.llm.provider_config import ProviderConfigLoader

        loader = ProviderConfigLoader(config_path="nonexistent.yaml")
        config = loader._get_default_config()

        assert config["primary"]["type"] == "lm_studio"
        assert "fallbacks" in config
        assert "router" in config

    def test_expand_env_vars(self, monkeypatch):
        """Test environment variable expansion in config."""
        from src.llm.provider_config import ProviderConfigLoader

        monkeypatch.setenv("TEST_API_KEY", "secret123")

        loader = ProviderConfigLoader()
        expanded = loader._expand_env_vars("${TEST_API_KEY}")

        assert expanded == "secret123"

    def test_expand_env_vars_not_found(self):
        """Test env var expansion with missing variable."""
        from src.llm.provider_config import ProviderConfigLoader

        loader = ProviderConfigLoader()
        expanded = loader._expand_env_vars("${NONEXISTENT_VAR}")

        assert expanded == ""

    @patch('src.llm.providers.lm_studio_provider.LMStudioProvider')
    def test_create_lm_studio_provider(self, mock_provider_class):
        """Test creating LM Studio provider from config."""
        from src.llm.provider_config import ProviderConfigLoader

        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider

        loader = ProviderConfigLoader()
        config = {
            "type": "lm_studio",
            "base_url": "http://localhost:1234/v1",
            "model": "test-model",
            "timeout": 120
        }

        provider = loader._create_provider(config)

        assert provider is not None
        mock_provider_class.assert_called_once()


@pytest.mark.unit
@pytest.mark.router
class TestLLMRouter:
    """Tests for LLM router with fallback."""

    def test_router_primary_success(self, mock_llm_provider):
        """Test router uses primary provider when it succeeds."""
        from src.llm.llm_router import LLMRouter

        router = LLMRouter(
            primary_provider=mock_llm_provider,
            fallback_providers=[]
        )

        request = LLMRequest(
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7
        )

        response = router.complete(request)

        assert response.content == "Mock response"
        assert mock_llm_provider.call_count == 1

    def test_router_fallback_on_primary_failure(self, mock_llm_provider):
        """Test router falls back when primary fails."""
        from src.llm.llm_router import LLMRouter
        from src.llm.base_provider import ProviderConnectionError

        # Create two providers
        primary = mock_llm_provider
        primary.should_fail = True  # Make primary fail

        fallback = mock_llm_provider.__class__()  # Create new instance

        router = LLMRouter(
            primary_provider=primary,
            fallback_providers=[fallback]
        )

        request = LLMRequest(
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7
        )

        response = router.complete(request)

        # Should succeed with fallback
        assert response.content == "Mock response"
        assert primary.call_count == 1  # Primary tried
        assert fallback.call_count == 1  # Fallback used

    def test_router_all_providers_fail(self, mock_llm_provider):
        """Test router raises error when all providers fail."""
        from src.llm.llm_router import LLMRouter

        primary = mock_llm_provider
        primary.should_fail = True

        fallback = mock_llm_provider.__class__()
        fallback.should_fail = True

        router = LLMRouter(
            primary_provider=primary,
            fallback_providers=[fallback]
        )

        request = LLMRequest(
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7
        )

        with pytest.raises(ProviderError, match="All LLM providers failed"):
            router.complete(request)

    def test_router_statistics_tracking(self, mock_llm_provider):
        """Test router tracks usage statistics."""
        from src.llm.llm_router import LLMRouter

        router = LLMRouter(primary_provider=mock_llm_provider)

        request = LLMRequest(
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7
        )

        # Make 3 successful requests
        for _ in range(3):
            router.complete(request)

        stats = router.get_statistics()

        assert stats["total_requests"] == 3
        assert stats["primary_successes"] == 3
        assert stats["fallback_successes"] == 0
        assert stats["total_failures"] == 0
        assert stats["primary_success_rate"] == 1.0

    def test_router_test_all_connections(self, mock_llm_provider):
        """Test router can test all provider connections."""
        from src.llm.llm_router import LLMRouter

        primary = mock_llm_provider
        fallback = mock_llm_provider.__class__()

        router = LLMRouter(
            primary_provider=primary,
            fallback_providers=[fallback]
        )

        results = router.test_all_connections()

        assert len(results) == 2
        assert all(results.values()), "All connections should succeed"


@pytest.mark.unit
@pytest.mark.router
class TestRouterAdapter:
    """Tests for router adapter (backward compatibility)."""

    def test_router_adapter_complete(self, mock_llm_provider):
        """Test adapter provides LMStudioClient-compatible interface."""
        from src.llm.llm_router import LLMRouter
        from src.llm.router_adapter import RouterAdapter

        router = LLMRouter(primary_provider=mock_llm_provider)
        adapter = RouterAdapter(router)

        # Call with LMStudioClient interface
        response = adapter.complete(
            agent_id="test-agent",
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7,
            max_tokens=100,
            model="test-model"
        )

        # Should return LMStudioResponse format
        assert response.content == "Mock response"
        assert response.agent_id == "test-agent"
        assert response.tokens_used == 50

    def test_router_adapter_streaming(self, mock_llm_provider):
        """Test adapter streaming interface."""
        from src.llm.llm_router import LLMRouter
        from src.llm.router_adapter import RouterAdapter

        router = LLMRouter(primary_provider=mock_llm_provider)
        adapter = RouterAdapter(router)

        chunks = []

        def callback(chunk):
            chunks.append(chunk)

        response = adapter.complete_streaming(
            agent_id="test-agent",
            system_prompt="Test",
            user_prompt="Hello",
            temperature=0.7,
            chunk_callback=callback
        )

        # Should call callback with chunks
        assert len(chunks) > 0
        assert response.content == "Mock streaming response"

    def test_router_adapter_test_connection(self, mock_llm_provider):
        """Test adapter connection testing."""
        from src.llm.llm_router import LLMRouter
        from src.llm.router_adapter import RouterAdapter

        router = LLMRouter(primary_provider=mock_llm_provider)
        adapter = RouterAdapter(router)

        # Should delegate to router's primary provider
        result = adapter.test_connection()
        assert result == True


@pytest.mark.unit
@pytest.mark.providers
class TestProviderErrorHandling:
    """Tests for provider error handling."""

    def test_provider_connection_error(self):
        """Test ProviderConnectionError is raised correctly."""
        with pytest.raises(ProviderConnectionError):
            raise ProviderConnectionError("Connection failed")

    def test_provider_authentication_error(self):
        """Test ProviderAuthenticationError is raised correctly."""
        with pytest.raises(ProviderAuthenticationError):
            raise ProviderAuthenticationError("Auth failed")

    def test_provider_error_base_class(self):
        """Test ProviderError is base for all provider errors."""
        # ProviderConnectionError should be catchable as ProviderError
        try:
            raise ProviderConnectionError("Test")
        except ProviderError:
            pass  # Should catch it

        # ProviderAuthenticationError should be catchable as ProviderError
        try:
            raise ProviderAuthenticationError("Test")
        except ProviderError:
            pass  # Should catch it
