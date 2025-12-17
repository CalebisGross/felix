"""
Router Adapter for Backward Compatibility

Provides LMStudioClient-like interface for LLMRouter to maintain
backward compatibility with existing code.
"""

import logging
import time
import threading
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
        self._connection_verified = False

        # Background processing control (for priority between background tasks and user interactions)
        self._background_processing = threading.Event()
        self._background_processing.set()  # Initially not paused
        self._sync_semaphore = threading.Semaphore(2)  # Max 2 concurrent sync requests

    def signal_user_activity(self, active: bool = True) -> None:
        """
        Signal that user is actively using the system.

        When user activity is signaled, background processing (like Knowledge Brain
        batch processing) will pause to give priority to user-initiated requests.

        Args:
            active: True to pause background processing, False to resume
        """
        if active:
            self._background_processing.clear()  # Pause background tasks
            logger.debug("User activity signaled - background processing paused")
        else:
            self._background_processing.set()  # Resume background tasks
            logger.debug("User activity ended - background processing resumed")

        # Also signal to underlying LMStudioClient if available
        try:
            provider = self.router.get_primary_provider()
            if hasattr(provider, 'client') and hasattr(provider.client, 'signal_user_activity'):
                provider.client.signal_user_activity(active)
        except Exception:
            pass  # Silently ignore if not available

    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                temperature: float = 0.7, max_tokens: Optional[int] = None,
                model: str = "local-model", is_background: bool = False) -> LMStudioResponse:
        """
        Complete a prompt (compatible with LMStudioClient interface).

        Args:
            agent_id: Agent identifier
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Model identifier
            is_background: If True, this is a background task that should yield
                          to user-initiated requests (e.g., Knowledge Brain batch processing)

        Returns:
            LMStudioResponse (compatible format)
        """
        # Background tasks wait for user activity to finish before proceeding
        if is_background:
            # Wait up to 5 seconds for user activity to stop, then proceed anyway
            self._background_processing.wait(timeout=5.0)

        # Acquire semaphore to limit concurrent sync requests
        with self._sync_semaphore:
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
                          token_controller: Optional[Any] = None,
                          cancel_event: Optional[threading.Event] = None) -> LMStudioResponse:
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
            cancel_event: Optional threading.Event to signal cancellation

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
            response = self.router.complete_streaming(request, streaming_wrapper, cancel_event)

            # FALLBACK: Use locally accumulated text if provider returned empty
            final_content = response.content
            if not final_content and accumulated_text:
                logger.warning(
                    f"Provider returned empty content for {agent_id}, "
                    f"using locally accumulated text ({len(accumulated_text)} chars)"
                )
                final_content = accumulated_text

            # Convert to LMStudioResponse format for compatibility
            return LMStudioResponse(
                content=final_content,
                tokens_used=response.tokens_used if response.tokens_used else chunk_count,
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
            result = self.router.get_primary_provider().test_connection()
            self._connection_verified = result
            return result
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self._connection_verified = False
            return False

    def ensure_connection(self) -> None:
        """Ensure connection to LLM provider or raise exception."""
        if not self._connection_verified and not self.test_connection():
            raise RuntimeError("Cannot connect to LLM provider. Check configuration.")

    def generate_embedding(self, text: str, model: str = "local-model"):
        """
        Generate embedding - delegates to LM Studio provider if available.

        Note: Embeddings are only supported with LM Studio provider.
        Returns None if not using LM Studio.
        """
        try:
            provider = self.router.get_primary_provider()
            # Check if this is an LM Studio provider with embedding support
            if hasattr(provider, 'client') and hasattr(provider.client, 'generate_embedding'):
                return provider.client.generate_embedding(text, model)
        except Exception as e:
            logger.debug(f"Embedding generation not available: {e}")
        return None

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
