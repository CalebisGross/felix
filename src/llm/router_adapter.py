"""
Router Adapter for Backward Compatibility

Provides LMStudioClient-like interface for LLMRouter to maintain
backward compatibility with existing code.
"""

import logging
from typing import Optional, Callable

from src.llm.llm_router import LLMRouter
from src.llm.base_provider import LLMRequest, ProviderError

# Import the original LLMResponse from lm_studio_client for compatibility
from src.llm.lm_studio_client import LLMResponse as LMStudioResponse

logger = logging.getLogger('felix_workflows')


class RouterAdapter:
    """
    Adapts LLMRouter to provide LMStudioClient-compatible interface.

    This allows existing code that expects LMStudioClient to use the new
    multi-provider router without modifications.
    """

    def __init__(self, router: LLMRouter):
        """
        Initialize adapter.

        Args:
            router: Configured LLMRouter instance
        """
        self.router = router
        self.verbose_logging = False

    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.7, max_tokens: Optional[int] = None,
                model: str = "local-model") -> LMStudioResponse:
        """
        Complete a prompt (compatible with LMStudioClient interface).

        Args:
            agent_id: Agent identifier
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model identifier

        Returns:
            LMStudioResponse (compatible format)
        """
        # Create request in new format
        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            agent_id=agent_id,
            model=model
        )

        try:
            # Route through router
            response = self.router.complete(request)

            # Convert to LMStudioResponse format for compatibility
            return LMStudioResponse(
                content=response.content,
                tokens_used=response.tokens_used,
                response_time=response.response_time,
                model=response.model,
                temperature=temperature,
                agent_id=agent_id,
                timestamp=0.0  # Not tracked in new format
            )

        except ProviderError as e:
            logger.error(f"Router completion failed for {agent_id}: {e}")
            raise RuntimeError(f"LLM completion failed: {e}")

    def complete_streaming(self, agent_id: str, system_prompt: str, user_prompt: str,
                          temperature: float = 0.7, max_tokens: Optional[int] = None,
                          model: str = "local-model",
                          chunk_callback: Optional[Callable[[str], None]] = None) -> LMStudioResponse:
        """
        Complete a prompt with streaming (compatible with LMStudioClient interface).

        Args:
            agent_id: Agent identifier
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model identifier
            chunk_callback: Callback for streaming chunks

        Returns:
            LMStudioResponse (compatible format)
        """
        # Create request in new format
        request = LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            agent_id=agent_id,
            model=model
        )

        try:
            # Route through router with streaming
            response = self.router.complete_streaming(request, chunk_callback or (lambda x: None))

            # Convert to LMStudioResponse format for compatibility
            return LMStudioResponse(
                content=response.content,
                tokens_used=response.tokens_used,
                response_time=response.response_time,
                model=response.model,
                temperature=temperature,
                agent_id=agent_id,
                timestamp=0.0
            )

        except ProviderError as e:
            logger.error(f"Router streaming failed for {agent_id}: {e}")
            raise RuntimeError(f"LLM streaming failed: {e}")

    def test_connection(self) -> bool:
        """Test connection to LLM provider(s)."""
        try:
            # Test primary provider
            return self.router.get_primary_provider().test_connection()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get router statistics."""
        return self.router.get_statistics()


def create_router_adapter(config_path: str = "config/llm.yaml") -> RouterAdapter:
    """
    Create RouterAdapter from configuration.

    Args:
        config_path: Path to LLM configuration file

    Returns:
        RouterAdapter instance

    Raises:
        RuntimeError: If configuration fails
    """
    from src.llm.provider_config import get_llm_router

    router = get_llm_router(config_path)
    return RouterAdapter(router)
