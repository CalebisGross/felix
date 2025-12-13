"""
LLM Provider Router for Felix

Handles routing between multiple LLM providers with automatic fallback.
Supports primary provider with fallback chain for reliability.
"""

import logging
from typing import List, Callable, Optional

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthenticationError,
    ProviderRateLimitError
)

logger = logging.getLogger('felix_workflows')


class LLMRouter:
    """
    Routes LLM requests to providers with automatic fallback.

    Tries primary provider first, falls back to secondary providers on failure.
    Provides unified interface for all LLM operations in Felix.
    """

    def __init__(self, primary_provider: BaseLLMProvider,
                 fallback_providers: Optional[List[BaseLLMProvider]] = None,
                 retry_on_rate_limit: bool = False,
                 verbose_logging: bool = False):
        """
        Initialize LLM router.

        Args:
            primary_provider: Primary LLM provider to use
            fallback_providers: List of fallback providers (tried in order)
            retry_on_rate_limit: Whether to retry with fallbacks on rate limit
            verbose_logging: Enable detailed logging
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.retry_on_rate_limit = retry_on_rate_limit
        self.verbose_logging = verbose_logging

        # Statistics tracking
        self.request_count = 0
        self.primary_success_count = 0
        self.fallback_success_count = 0
        self.total_failure_count = 0

        if self.verbose_logging:
            logger.info(f"LLM Router initialized:")
            logger.info(f"  Primary: {primary_provider.get_provider_name()}")
            if self.fallback_providers:
                fallback_names = [p.get_provider_name() for p in self.fallback_providers]
                logger.info(f"  Fallbacks: {', '.join(fallback_names)}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a completion, trying primary then fallback providers.

        Args:
            request: LLM request

        Returns:
            LLM response from successful provider

        Raises:
            ProviderError: If all providers fail
        """
        self.request_count += 1
        providers_to_try = [self.primary_provider] + self.fallback_providers
        errors = []

        for i, provider in enumerate(providers_to_try):
            is_primary = (i == 0)
            provider_name = provider.get_provider_name()

            try:
                if self.verbose_logging:
                    logger.info(f"Attempting completion with {provider_name} "
                              f"({'primary' if is_primary else f'fallback {i}'})")

                response = provider.complete(request)

                # Success!
                if is_primary:
                    self.primary_success_count += 1
                else:
                    self.fallback_success_count += 1
                    logger.warning(f"Primary provider failed, succeeded with fallback: {provider_name}")

                return response

            except ProviderRateLimitError as e:
                error_msg = f"{provider_name} rate limited: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

                if not self.retry_on_rate_limit:
                    # Don't try fallbacks on rate limit unless explicitly enabled
                    break

            except ProviderAuthenticationError as e:
                error_msg = f"{provider_name} authentication failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Don't try fallbacks on auth errors (won't help)
                break

            except ProviderConnectionError as e:
                error_msg = f"{provider_name} connection failed: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # Continue to fallbacks

            except ProviderError as e:
                error_msg = f"{provider_name} error: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # Continue to fallbacks

            except Exception as e:
                error_msg = f"{provider_name} unexpected error: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue to fallbacks

        # All providers failed
        self.total_failure_count += 1
        error_summary = "\n".join(errors)
        raise ProviderError(f"All LLM providers failed:\n{error_summary}")

    def complete_streaming(self, request: LLMRequest,
                          callback: Callable[[str], None]) -> LLMResponse:
        """
        Generate a streaming completion with fallback support.

        Args:
            request: LLM request
            callback: Function called with each token/chunk

        Returns:
            LLM response from successful provider

        Raises:
            ProviderError: If all providers fail
        """
        self.request_count += 1
        providers_to_try = [self.primary_provider] + self.fallback_providers
        errors = []

        for i, provider in enumerate(providers_to_try):
            is_primary = (i == 0)
            provider_name = provider.get_provider_name()

            try:
                if self.verbose_logging:
                    logger.info(f"Attempting streaming with {provider_name} "
                              f"({'primary' if is_primary else f'fallback {i}'})")

                response = provider.complete_streaming(request, callback)

                # Success!
                if is_primary:
                    self.primary_success_count += 1
                else:
                    self.fallback_success_count += 1
                    logger.warning(f"Primary provider failed, succeeded with fallback: {provider_name}")

                return response

            except ProviderRateLimitError as e:
                error_msg = f"{provider_name} rate limited: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

                if not self.retry_on_rate_limit:
                    break

            except ProviderAuthenticationError as e:
                error_msg = f"{provider_name} authentication failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                break

            except ProviderConnectionError as e:
                error_msg = f"{provider_name} connection failed: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

            except ProviderError as e:
                error_msg = f"{provider_name} error: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)

            except Exception as e:
                error_msg = f"{provider_name} unexpected error: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # All providers failed
        self.total_failure_count += 1
        error_summary = "\n".join(errors)
        raise ProviderError(f"All LLM providers failed for streaming:\n{error_summary}")

    def test_all_connections(self) -> dict:
        """
        Test connectivity for all configured providers.

        Returns:
            Dict mapping provider names to connection status (bool)
        """
        results = {}

        # Test primary
        primary_name = self.primary_provider.get_provider_name()
        results[f"{primary_name} (primary)"] = self.primary_provider.test_connection()

        # Test fallbacks
        for i, provider in enumerate(self.fallback_providers):
            provider_name = provider.get_provider_name()
            results[f"{provider_name} (fallback {i+1})"] = provider.test_connection()

        return results

    def get_statistics(self) -> dict:
        """Get router statistics including circuit breaker status."""
        stats = {
            "total_requests": self.request_count,
            "primary_successes": self.primary_success_count,
            "fallback_successes": self.fallback_success_count,
            "total_failures": self.total_failure_count,
            "primary_success_rate": (
                self.primary_success_count / self.request_count
                if self.request_count > 0 else 0.0
            ),
            "overall_success_rate": (
                (self.primary_success_count + self.fallback_success_count) / self.request_count
                if self.request_count > 0 else 0.0
            )
        }

        # Add circuit breaker metrics if available
        circuit_metrics = {}
        for provider in [self.primary_provider] + self.fallback_providers:
            if hasattr(provider, 'get_circuit_breaker_metrics'):
                name = provider.get_provider_name()
                circuit_metrics[name] = provider.get_circuit_breaker_metrics()

        if circuit_metrics:
            stats['circuit_breakers'] = circuit_metrics

        return stats

    def get_primary_provider(self) -> BaseLLMProvider:
        """Get the primary provider."""
        return self.primary_provider

    def get_fallback_providers(self) -> List[BaseLLMProvider]:
        """Get the list of fallback providers."""
        return self.fallback_providers

    def __repr__(self) -> str:
        primary_name = self.primary_provider.get_provider_name()
        fallback_count = len(self.fallback_providers)
        return f"<LLMRouter primary={primary_name} fallbacks={fallback_count}>"
