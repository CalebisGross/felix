"""
LM Studio Provider for Felix

Wraps the existing LMStudioClient to implement the BaseLLMProvider interface.
Maintains backward compatibility while enabling multi-provider support.
"""

import time
import logging
from typing import Callable, List

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderType,
    ProviderError,
    ProviderConnectionError
)

# Import existing LM Studio client
from src.llm.lm_studio_client import LMStudioClient

logger = logging.getLogger('felix_workflows')


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio provider implementation.

    Wraps the existing LMStudioClient to provide BaseLLMProvider interface
    while maintaining all existing functionality.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "local-model", timeout: int = 120,
                 verbose_logging: bool = False):
        """
        Initialize LM Studio provider.

        Args:
            base_url: LM Studio API endpoint
            model: Model identifier (informational, LM Studio uses loaded model)
            timeout: Request timeout in seconds
            verbose_logging: Enable detailed logging
        """
        super().__init__(verbose_logging=verbose_logging)

        self.provider_type = ProviderType.LM_STUDIO
        self.base_url = base_url
        self.default_model = model
        self.timeout = timeout

        # Initialize wrapped LM Studio client
        try:
            self.client = LMStudioClient(
                base_url=base_url,
                timeout=timeout,
                verbose_logging=verbose_logging
            )
        except Exception as e:
            raise ProviderConnectionError(f"Failed to initialize LM Studio client: {e}")

        if self.verbose_logging:
            logger.info(f"LM Studio provider initialized: {base_url}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion using LM Studio."""
        start_time = time.time()

        if self.verbose_logging:
            logger.info(f"LM Studio API call for {request.agent_id or 'unknown'}")
            logger.info(f"  Temperature: {request.temperature}")
            logger.info(f"  Max tokens: {request.max_tokens or 'default'}")

        try:
            # Call existing LM Studio client
            lm_response = self.client.complete(
                agent_id=request.agent_id or "unknown",
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model=request.model or self.default_model
            )

            # Convert to BaseLLMProvider response format
            response_time = time.time() - start_time

            # LM Studio doesn't provide token breakdown, estimate 50/50
            prompt_tokens = lm_response.tokens_used // 2
            completion_tokens = lm_response.tokens_used - prompt_tokens

            if self.verbose_logging:
                logger.info(f"  Response time: {response_time:.2f}s")
                logger.info(f"  Tokens: {lm_response.tokens_used}")

            return LLMResponse(
                content=lm_response.content,
                tokens_used=lm_response.tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                model=lm_response.model,
                provider="lm_studio",
                finish_reason="stop"
            )

        except Exception as e:
            raise ProviderError(f"LM Studio completion failed: {e}")

    def complete_streaming(self, request: LLMRequest,
                          callback: Callable[[str], None]) -> LLMResponse:
        """Generate a streaming completion using LM Studio."""
        start_time = time.time()

        if self.verbose_logging:
            logger.info(f"LM Studio streaming API call for {request.agent_id or 'unknown'}")

        try:
            # Use existing LM Studio streaming
            lm_response = self.client.complete_streaming(
                agent_id=request.agent_id or "unknown",
                system_prompt=request.system_prompt,
                user_prompt=request.user_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model=request.model or self.default_model,
                chunk_callback=callback  # Pass callback directly
            )

            response_time = time.time() - start_time

            # Estimate token breakdown
            prompt_tokens = lm_response.tokens_used // 2
            completion_tokens = lm_response.tokens_used - prompt_tokens

            if self.verbose_logging:
                logger.info(f"  Streaming complete: {response_time:.2f}s")
                logger.info(f"  Tokens: {lm_response.tokens_used}")

            return LLMResponse(
                content=lm_response.content,
                tokens_used=lm_response.tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                model=lm_response.model,
                provider="lm_studio",
                finish_reason="stop"
            )

        except Exception as e:
            raise ProviderError(f"LM Studio streaming failed: {e}")

    def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            return self.client.test_connection()
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"LM Studio connection test failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """
        Get available models from LM Studio.

        Note: LM Studio uses whatever model is currently loaded,
        so we return the configured model name.
        """
        return [self.default_model]

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """
        Estimate cost for LM Studio usage.

        Returns 0.0 since LM Studio is local and free.
        """
        return 0.0

    def get_underlying_client(self) -> LMStudioClient:
        """Get the underlying LMStudioClient for advanced usage."""
        return self.client
