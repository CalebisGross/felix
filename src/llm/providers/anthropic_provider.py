"""
Anthropic Claude Provider for Felix

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku.
"""

import time
import logging
from typing import Callable, List, Dict, Any

try:
    from anthropic import Anthropic, APIError, APIConnectionError, RateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderType,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderModelError
)

logger = logging.getLogger('felix_workflows')


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider implementation.

    Supports all Claude 3 models through the Anthropic API.
    """

    # Pricing per 1M tokens (as of 2025-01)
    PRICING = {
        "claude-3-5-sonnet-20241022": {"prompt": 3.00, "completion": 15.00},
        "claude-3-5-sonnet-20240620": {"prompt": 3.00, "completion": 15.00},
        "claude-3-opus-20240229": {"prompt": 15.00, "completion": 75.00},
        "claude-3-sonnet-20240229": {"prompt": 3.00, "completion": 15.00},
        "claude-3-haiku-20240307": {"prompt": 0.25, "completion": 1.25},
    }

    DEFAULT_MODELS = {
        "sonnet": "claude-3-5-sonnet-20241022",
        "opus": "claude-3-opus-20240229",
        "haiku": "claude-3-haiku-20240307",
    }

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022",
                 timeout: int = 120, verbose_logging: bool = False):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model to use (default: Claude 3.5 Sonnet)
            timeout: Request timeout in seconds
            verbose_logging: Enable detailed logging
        """
        super().__init__(verbose_logging=verbose_logging)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        if not api_key:
            raise ProviderAuthenticationError("Anthropic API key is required")

        self.provider_type = ProviderType.ANTHROPIC
        self.api_key = api_key
        self.default_model = model
        self.timeout = timeout

        # Initialize Anthropic client
        try:
            self.client = Anthropic(api_key=api_key, timeout=timeout)
        except Exception as e:
            raise ProviderConnectionError(f"Failed to initialize Anthropic client: {e}")

        if self.verbose_logging:
            logger.info(f"Anthropic provider initialized with model: {self.default_model}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        start_time = time.time()
        model = request.model or self.default_model

        if self.verbose_logging:
            logger.info(f"Anthropic API call for {request.agent_id or 'unknown'}")
            logger.info(f"  Model: {model}")
            logger.info(f"  Temperature: {request.temperature}")
            logger.info(f"  Max tokens: {request.max_tokens or 'default'}")

        try:
            # Build messages (Anthropic uses array of message objects)
            messages = [{"role": "user", "content": request.user_prompt}]

            # Make API call
            response = self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                system=request.system_prompt,  # Anthropic separates system prompt
                messages=messages
            )

            # Extract response
            content = response.content[0].text
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            tokens_used = prompt_tokens + completion_tokens

            response_time = time.time() - start_time

            if self.verbose_logging:
                logger.info(f"  Response time: {response_time:.2f}s")
                logger.info(f"  Tokens: {tokens_used} (prompt: {prompt_tokens}, completion: {completion_tokens})")

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                model=model,
                provider="anthropic",
                finish_reason=response.stop_reason or "stop",
                metadata={
                    "message_id": response.id,
                    "model_version": response.model,
                }
            )

        except RateLimitError as e:
            raise ProviderRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except APIConnectionError as e:
            raise ProviderConnectionError(f"Anthropic connection error: {e}")
        except APIError as e:
            if "model" in str(e).lower():
                raise ProviderModelError(f"Anthropic model error: {e}")
            raise ProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Anthropic: {e}")

    def complete_streaming(self, request: LLMRequest,
                          callback: Callable[[str], None]) -> LLMResponse:
        """Generate a streaming completion using Anthropic API."""
        start_time = time.time()
        model = request.model or self.default_model

        if self.verbose_logging:
            logger.info(f"Anthropic streaming API call for {request.agent_id or 'unknown'}")
            logger.info(f"  Model: {model}")

        try:
            # Build messages
            messages = [{"role": "user", "content": request.user_prompt}]

            # Accumulate response
            full_content = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Stream response
            with self.client.messages.stream(
                model=model,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    full_content += text
                    callback(text)  # Send chunk to callback

                # Get final message for token counts
                final_message = stream.get_final_message()
                prompt_tokens = final_message.usage.input_tokens
                completion_tokens = final_message.usage.output_tokens

            tokens_used = prompt_tokens + completion_tokens
            response_time = time.time() - start_time

            if self.verbose_logging:
                logger.info(f"  Streaming complete: {response_time:.2f}s")
                logger.info(f"  Tokens: {tokens_used}")

            return LLMResponse(
                content=full_content,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                model=model,
                provider="anthropic",
                finish_reason="stop"
            )

        except RateLimitError as e:
            raise ProviderRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except APIConnectionError as e:
            raise ProviderConnectionError(f"Anthropic connection error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error streaming from Anthropic: {e}")

    def test_connection(self) -> bool:
        """Test Anthropic API connection."""
        try:
            # Simple test with minimal tokens
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Anthropic connection test failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        # Anthropic doesn't have a models list endpoint
        # Return known models
        return list(self.PRICING.keys())

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """
        Estimate cost for Anthropic API usage.

        Note: This is a simplified estimate. Actual costs depend on
        prompt vs completion token split.
        """
        if model not in self.PRICING:
            return 0.0

        # Rough estimate: assume 50/50 split
        pricing = self.PRICING[model]
        avg_price_per_1m = (pricing["prompt"] + pricing["completion"]) / 2
        cost = (tokens_used / 1_000_000) * avg_price_per_1m

        return cost

    def get_cost_breakdown(self, prompt_tokens: int, completion_tokens: int,
                          model: str) -> Dict[str, float]:
        """Get detailed cost breakdown."""
        if model not in self.PRICING:
            return {"prompt_cost": 0.0, "completion_cost": 0.0, "total_cost": 0.0}

        pricing = self.PRICING[model]
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": prompt_cost + completion_cost
        }
