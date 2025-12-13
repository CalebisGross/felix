"""
Base Provider Abstraction for Felix LLM Integration

This module defines the abstract base class and data structures for all LLM providers.
It enables Felix to work with multiple LLM backends (LM Studio, Anthropic, Gemini, etc.)
through a unified interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from enum import Enum


class ProviderType(Enum):
    """Supported LLM provider types."""
    LM_STUDIO = "lm_studio"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"  # For future use
    AZURE = "azure"    # For future use


@dataclass
class LLMRequest:
    """
    Unified request structure for all LLM providers.

    Attributes:
        system_prompt: System-level instructions for the model
        user_prompt: User's actual prompt/query
        temperature: Sampling temperature (0.0-2.0, typically 0.0-1.0)
        max_tokens: Maximum tokens to generate (None for provider default)
        stream: Whether to stream the response
        agent_id: Optional agent identifier for logging
        model: Optional model override (uses provider default if None)
    """
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    agent_id: Optional[str] = None
    model: Optional[str] = None

    # Additional provider-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    Unified response structure from all LLM providers.

    Attributes:
        content: The generated text content
        tokens_used: Total tokens consumed (prompt + completion)
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        response_time: Time taken to generate response (seconds)
        model: Actual model used
        provider: Provider that generated this response
        finish_reason: Why generation stopped (e.g., "stop", "length", "error")
    """
    content: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    response_time: float
    model: str
    provider: str
    finish_reason: str = "stop"

    # Provider-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class ProviderConnectionError(ProviderError):
    """Raised when provider connection fails."""
    pass


class ProviderCircuitOpenError(ProviderConnectionError):
    """Raised when circuit breaker is open for a provider."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""
    pass


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""
    pass


class ProviderModelError(ProviderError):
    """Raised when requested model is not available."""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    All provider implementations (LM Studio, Anthropic, Gemini, etc.) must
    implement this interface to be compatible with Felix's agent system.
    """

    def __init__(self, **kwargs):
        """
        Initialize the provider.

        Common kwargs:
            base_url: Base URL for API (for local/custom providers)
            api_key: API key for authentication
            model: Default model to use
            timeout: Request timeout in seconds
            verbose_logging: Enable detailed logging
        """
        self.provider_type = None
        self.default_model = None
        self.verbose_logging = kwargs.get('verbose_logging', False)

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion (non-streaming).

        Args:
            request: LLMRequest with prompt and parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ProviderError: If completion fails
        """
        pass

    @abstractmethod
    def complete_streaming(
        self,
        request: LLMRequest,
        callback: Callable[[str], None]
    ) -> LLMResponse:
        """
        Generate a completion with streaming.

        Args:
            request: LLMRequest with prompt and parameters
            callback: Function called with each token/chunk as it arrives

        Returns:
            LLMResponse with full generated content and metadata

        Raises:
            ProviderError: If streaming fails
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the provider is reachable and operational.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from this provider.

        Returns:
            List of model identifiers
        """
        pass

    def get_provider_name(self) -> str:
        """Get human-readable provider name."""
        return self.provider_type.value if self.provider_type else "unknown"

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """
        Estimate cost in USD for the given token usage.

        Args:
            tokens_used: Total tokens (prompt + completion)
            model: Model identifier

        Returns:
            Estimated cost in USD (0.0 for local providers)
        """
        # Default implementation - override for cloud providers
        return 0.0

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.get_provider_name()} model={self.default_model}>"
