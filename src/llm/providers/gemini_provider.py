"""
Google Gemini Provider for Felix

Supports Gemini 1.5 Pro and Gemini 1.5 Flash.
"""

import time
import logging
from typing import Callable, List, Dict, Any

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider implementation.

    Supports Gemini 1.5 Pro and Gemini 1.5 Flash through Google AI Studio.
    """

    # Pricing per 1M tokens (as of 2025-01, free tier exists)
    PRICING = {
        "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
        "gemini-1.5-pro-latest": {"prompt": 1.25, "completion": 5.00},
        "gemini-1.5-flash": {"prompt": 0.075, "completion": 0.30},
        "gemini-1.5-flash-latest": {"prompt": 0.075, "completion": 0.30},
        "gemini-1.0-pro": {"prompt": 0.50, "completion": 1.50},
    }

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest",
                 timeout: int = 120, verbose_logging: bool = False):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI Studio API key
            model: Model to use (default: Gemini 1.5 Flash)
            timeout: Request timeout in seconds
            verbose_logging: Enable detailed logging
        """
        super().__init__(verbose_logging=verbose_logging)

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )

        if not api_key:
            raise ProviderAuthenticationError("Google AI Studio API key is required")

        self.provider_type = ProviderType.GEMINI
        self.api_key = api_key
        self.default_model = model
        self.timeout = timeout

        # Configure Gemini
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
        except Exception as e:
            raise ProviderConnectionError(f"Failed to initialize Gemini: {e}")

        if self.verbose_logging:
            logger.info(f"Gemini provider initialized with model: {self.default_model}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion using Gemini API."""
        start_time = time.time()
        model_name = request.model or self.default_model

        if self.verbose_logging:
            logger.info(f"Gemini API call for {request.agent_id or 'unknown'}")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Temperature: {request.temperature}")
            logger.info(f"  Max tokens: {request.max_tokens or 'default'}")

        try:
            # If model changed, reinitialize
            if model_name != self.default_model:
                model = genai.GenerativeModel(model_name)
            else:
                model = self.model

            # Build prompt (Gemini combines system + user)
            full_prompt = f"{request.system_prompt}\n\n{request.user_prompt}"

            # Generation config
            gen_config = GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens or 8192,
            )

            # Generate
            response = model.generate_content(
                full_prompt,
                generation_config=gen_config
            )

            # Extract content
            content = response.text

            # Token counts (if available)
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

            tokens_used = prompt_tokens + completion_tokens
            response_time = time.time() - start_time

            if self.verbose_logging:
                logger.info(f"  Response time: {response_time:.2f}s")
                logger.info(f"  Tokens: {tokens_used} (prompt: {prompt_tokens}, completion: {completion_tokens})")

            # Determine finish reason
            finish_reason = "stop"
            if hasattr(response, 'candidates') and response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                model=model_name,
                provider="gemini",
                finish_reason=finish_reason
            )

        except google_exceptions.ResourceExhausted as e:
            raise ProviderRateLimitError(f"Gemini rate limit exceeded: {e}")
        except google_exceptions.PermissionDenied as e:
            raise ProviderAuthenticationError(f"Gemini authentication failed: {e}")
        except google_exceptions.InvalidArgument as e:
            raise ProviderModelError(f"Gemini model error: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error calling Gemini: {e}")

    def complete_streaming(self, request: LLMRequest,
                          callback: Callable[[str], None]) -> LLMResponse:
        """Generate a streaming completion using Gemini API."""
        start_time = time.time()
        model_name = request.model or self.default_model

        if self.verbose_logging:
            logger.info(f"Gemini streaming API call for {request.agent_id or 'unknown'}")
            logger.info(f"  Model: {model_name}")

        try:
            # If model changed, reinitialize
            if model_name != self.default_model:
                model = genai.GenerativeModel(model_name)
            else:
                model = self.model

            # Build prompt
            full_prompt = f"{request.system_prompt}\n\n{request.user_prompt}"

            # Generation config
            gen_config = GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens or 8192,
            )

            # Stream response
            full_content = ""
            prompt_tokens = 0
            completion_tokens = 0

            response = model.generate_content(
                full_prompt,
                generation_config=gen_config,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    full_content += chunk_text
                    callback(chunk_text)

            # Get token counts from last chunk (if available)
            if hasattr(response, 'usage_metadata'):
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

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
                model=model_name,
                provider="gemini",
                finish_reason="stop"
            )

        except google_exceptions.ResourceExhausted as e:
            raise ProviderRateLimitError(f"Gemini rate limit exceeded: {e}")
        except google_exceptions.PermissionDenied as e:
            raise ProviderAuthenticationError(f"Gemini authentication failed: {e}")
        except Exception as e:
            raise ProviderError(f"Unexpected error streaming from Gemini: {e}")

    def test_connection(self) -> bool:
        """Test Gemini API connection."""
        try:
            # Simple test with minimal tokens
            model = genai.GenerativeModel(self.default_model)
            gen_config = GenerationConfig(max_output_tokens=10)
            response = model.generate_content("Hi", generation_config=gen_config)
            return True
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Gemini connection test failed: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        try:
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace('models/', ''))
            return models
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Failed to list Gemini models: {e}")
            return list(self.PRICING.keys())

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """
        Estimate cost for Gemini API usage.

        Note: Gemini has a generous free tier. This returns paid tier pricing.
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
