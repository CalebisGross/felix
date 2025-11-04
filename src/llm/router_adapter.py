"""
Router Adapter for Backward Compatibility

Provides LMStudioClient-like interface for LLMRouter to maintain
backward compatibility with existing code.
"""

import logging
import time
from typing import Optional, Callable, Any, Union

from src.llm.llm_router import LLMRouter
from src.llm.base_provider import LLMRequest, ProviderError

# Import the original LLMResponse (aliased as LMStudioResponse) and StreamingChunk from lm_studio_client for compatibility
from src.llm.lm_studio_client import LLMResponse as LMStudioResponse, StreamingChunk

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
                          chunk_callback: Optional[Callable] = None,
                          callback: Optional[Callable] = None,
                          batch_interval: float = 0.1,
                          token_controller: Optional[Any] = None) -> LMStudioResponse:
        """
        Complete a prompt with streaming (compatible with LMStudioClient interface).

        Args:
            agent_id: Agent identifier
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model identifier
            chunk_callback: Legacy callback parameter (accepts StreamingChunk)
            callback: Primary callback parameter (accepts StreamingChunk)
            batch_interval: Time batching interval (accepted for compatibility, handled by providers)
            token_controller: Token budget controller (accepted for compatibility, not enforced)

        Returns:
            LMStudioResponse (compatible format)

        Note:
            batch_interval and token_controller are accepted for backward compatibility
            but are not enforced at the router level. Providers handle batching internally.
        """
        # Use whichever callback was provided (prefer 'callback' over 'chunk_callback')
        user_callback = callback or chunk_callback

        # State for accumulation and token tracking
        accumulated_text = ""
        chunk_count = 0

        def streaming_wrapper(chunk: Union[str, StreamingChunk]) -> None:
            """
            Wrap provider callback to handle both strings and StreamingChunk objects.

            This maintains compatibility with LLMAgent which expects StreamingChunk objects
            with .content, .accumulated, and .tokens_so_far attributes.

            Handles both:
            - String chunks from providers that return simple strings
            - StreamingChunk objects from providers (like LMStudioClient) that return rich objects
            """
            nonlocal accumulated_text, chunk_count

            # Handle both string chunks and StreamingChunk objects
            if isinstance(chunk, StreamingChunk):
                # Provider sent a StreamingChunk - extract data directly
                chunk_str = chunk.content
                accumulated_text = chunk.accumulated  # Use provider's accumulated value
                chunk_count = chunk.tokens_so_far     # Use provider's token count
            else:
                # Provider sent a plain string - accumulate manually
                chunk_str = chunk
                accumulated_text += chunk_str
                chunk_count += 1

            # Call user callback with StreamingChunk object if provided
            if user_callback:
                streaming_chunk = StreamingChunk(
                    content=chunk_str,
                    accumulated=accumulated_text,
                    tokens_so_far=chunk_count,
                    agent_id=agent_id,
                    timestamp=time.time()
                )
                user_callback(streaming_chunk)

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
            # Route through router with wrapped streaming callback
            response = self.router.complete_streaming(request, streaming_wrapper)

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
