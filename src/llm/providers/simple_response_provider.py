"""
Simple Response Provider for Graceful Fallback

This provider serves as the last-resort fallback when all other LLM providers
are unavailable. It returns a graceful "service unavailable" message instead
of raising an exception, ensuring agents don't crash.

Part of Issue #16: Add LLM Provider Fallback for Inference
"""

import time
import logging
import threading
from typing import List, Callable, Optional

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderType,
)

logger = logging.getLogger('felix_workflows')


# Default unavailable message - clear, helpful, and non-alarming
DEFAULT_UNAVAILABLE_MESSAGE = (
    "I apologize, but I'm unable to process your request at this time. "
    "The LLM service is temporarily unavailable. Please check that:\n\n"
    "1. LM Studio is running with a model loaded, or\n"
    "2. A cloud provider (Anthropic/Gemini) is configured in config/llm.yaml\n\n"
    "Once connectivity is restored, I'll be able to assist you fully."
)


class SimpleResponseProvider(BaseLLMProvider):
    """
    Fallback provider that returns graceful 'unavailable' responses.

    This provider is designed to be the last fallback in a provider chain.
    It never fails - instead, it returns a helpful message indicating
    that the LLM service is temporarily unavailable.

    Usage in config/llm.yaml:
        fallbacks:
          - type: "simple_response"
            message: "Custom unavailable message..."  # Optional
    """

    def __init__(
        self,
        message: str = DEFAULT_UNAVAILABLE_MESSAGE,
        verbose_logging: bool = False,
        **kwargs
    ):
        """
        Initialize the simple response provider.

        Args:
            message: Custom unavailable message (optional)
            verbose_logging: Enable detailed logging
            **kwargs: Additional parameters (ignored, for interface compatibility)
        """
        super().__init__(verbose_logging=verbose_logging)
        self.provider_type = ProviderType.LM_STUDIO  # Reuse existing enum
        self.default_model = "simple_response"
        self.unavailable_message = message

        logger.info("SimpleResponseProvider initialized as last-resort fallback")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Return a graceful 'service unavailable' response.

        This method always succeeds - it never raises an exception.

        Args:
            request: LLMRequest (used for agent_id logging only)

        Returns:
            LLMResponse with unavailable message
        """
        start_time = time.time()

        agent_info = f" for agent {request.agent_id}" if request.agent_id else ""
        logger.warning(f"SimpleResponseProvider returning fallback response{agent_info}")

        return LLMResponse(
            content=self.unavailable_message,
            tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            response_time=time.time() - start_time,
            model="simple_response",
            provider="simple_response",
            finish_reason="fallback",
            metadata={
                "fallback": True,
                "reason": "all_providers_unavailable"
            }
        )

    def complete_streaming(
        self,
        request: LLMRequest,
        callback: Callable[[str], None],
        cancel_event: Optional[threading.Event] = None
    ) -> LLMResponse:
        """
        Return a graceful 'service unavailable' response with streaming simulation.

        Streams the unavailable message word-by-word to provide visual feedback
        that something is happening, even though it's a fallback response.

        Args:
            request: LLMRequest (used for agent_id logging only)
            callback: Function called with each word chunk
            cancel_event: Optional threading.Event to signal cancellation

        Returns:
            LLMResponse with unavailable message
        """
        start_time = time.time()

        agent_info = f" for agent {request.agent_id}" if request.agent_id else ""
        logger.warning(f"SimpleResponseProvider streaming fallback response{agent_info}")

        # Stream the message word-by-word for visual feedback
        words = self.unavailable_message.split(' ')
        for i, word in enumerate(words):
            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                break
            # Add space before words (except first)
            chunk = word if i == 0 else ' ' + word
            callback(chunk)
            # Small delay to simulate streaming (10ms per word)
            time.sleep(0.01)

        return LLMResponse(
            content=self.unavailable_message,
            tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            response_time=time.time() - start_time,
            model="simple_response",
            provider="simple_response",
            finish_reason="fallback",
            metadata={
                "fallback": True,
                "reason": "all_providers_unavailable",
                "streamed": True
            }
        )

    def test_connection(self) -> bool:
        """
        Test connection - always returns True.

        This provider is always 'available' since it doesn't depend on
        any external service.

        Returns:
            True (always)
        """
        return True

    def get_available_models(self) -> List[str]:
        """
        Get available models - returns single pseudo-model.

        Returns:
            List containing 'simple_response'
        """
        return ["simple_response"]

    def get_provider_name(self) -> str:
        """Get provider name for logging/display."""
        return "simple_response"

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """
        Estimate cost - always free.

        Returns:
            0.0 (no cost)
        """
        return 0.0

    def __repr__(self) -> str:
        return "<SimpleResponseProvider fallback=True>"
