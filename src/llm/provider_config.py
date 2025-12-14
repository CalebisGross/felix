"""
LLM Provider Configuration Loader

Reads config/llm.yaml and instantiates providers and router.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional

from src.llm.base_provider import BaseLLMProvider
from src.llm.llm_router import LLMRouter
from src.llm.providers import LMStudioProvider, AnthropicProvider, GeminiProvider, SimpleResponseProvider

logger = logging.getLogger('felix_workflows')


class ProviderConfigLoader:
    """Loads provider configuration from YAML and creates router."""

    def __init__(self, config_path: str = "config/llm.yaml"):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not os.path.exists(self.config_path):
            logger.warning(f"LLM config not found: {self.config_path}, using defaults")
            return self._get_default_config()

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config or self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load LLM config: {e}, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration (LM Studio only)."""
        return {
            "primary": {
                "type": "lm_studio",
                "base_url": "http://localhost:1234/v1",
                "model": "local-model",
                "timeout": 120
            },
            "fallbacks": [],
            "router": {
                "retry_on_rate_limit": False,
                "verbose_logging": False
            }
        }

    def _expand_env_vars(self, value: str) -> str:
        """Expand environment variables in config values."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
        return value

    def _create_provider(self, config: Dict[str, Any]) -> Optional[BaseLLMProvider]:
        """
        Create a provider from configuration.

        Args:
            config: Provider configuration dict

        Returns:
            Instantiated provider or None if creation fails
        """
        provider_type = config.get("type", "").lower()

        # Expand environment variables in API keys
        if "api_key" in config:
            config["api_key"] = self._expand_env_vars(config["api_key"])

        provider: Optional[BaseLLMProvider] = None

        try:
            if provider_type == "lm_studio":
                provider = LMStudioProvider(
                    base_url=config.get("base_url", "http://localhost:1234/v1"),
                    model=config.get("model", "local-model"),
                    timeout=config.get("timeout", 120),
                    verbose_logging=config.get("verbose_logging", False)
                )

            elif provider_type == "anthropic":
                api_key = config.get("api_key", "")
                if not api_key:
                    logger.error("Anthropic API key not provided")
                    return None

                provider = AnthropicProvider(
                    api_key=api_key,
                    model=config.get("model", "claude-3-5-sonnet-20241022"),
                    timeout=config.get("timeout", 120),
                    verbose_logging=config.get("verbose_logging", False)
                )

            elif provider_type == "gemini":
                api_key = config.get("api_key", "")
                if not api_key:
                    logger.error("Gemini API key not provided")
                    return None

                provider = GeminiProvider(
                    api_key=api_key,
                    model=config.get("model", "gemini-1.5-flash-latest"),
                    timeout=config.get("timeout", 120),
                    verbose_logging=config.get("verbose_logging", False)
                )

            elif provider_type == "simple_response":
                # Last-resort fallback provider - always available
                # Only pass message if explicitly provided in config (otherwise use provider's default)
                simple_kwargs = {"verbose_logging": config.get("verbose_logging", False)}
                if "message" in config:
                    simple_kwargs["message"] = config["message"]
                provider = SimpleResponseProvider(**simple_kwargs)

            else:
                logger.error(f"Unknown provider type: {provider_type}")
                return None

            # Wrap with circuit breaker if configured (enabled by default)
            if provider is not None:
                circuit_config = config.get('circuit_breaker', {})
                if circuit_config.get('enabled', True):
                    from src.llm.circuit_breaker import CircuitBreakerProvider, CircuitBreakerConfig

                    breaker_config = CircuitBreakerConfig(
                        failure_threshold=circuit_config.get('failure_threshold', 3),
                        recovery_timeout=circuit_config.get('recovery_timeout', 60.0),
                        half_open_max_calls=circuit_config.get('half_open_max_calls', 1)
                    )
                    provider = CircuitBreakerProvider(provider, config=breaker_config)
                    logger.debug(f"Circuit breaker enabled for {provider_type} provider")

            return provider

        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            return None

    def create_router(self) -> Optional[LLMRouter]:
        """
        Create LLM router from configuration.

        Returns:
            Configured LLMRouter or None if creation fails
        """
        # Create primary provider
        primary_config = self.config.get("primary", {})
        primary_provider = self._create_provider(primary_config)

        if not primary_provider:
            logger.error("Failed to create primary provider")
            return None

        # Create fallback providers
        fallback_providers = []
        fallback_configs = self.config.get("fallbacks", [])

        for fallback_config in fallback_configs:
            fallback_provider = self._create_provider(fallback_config)
            if fallback_provider:
                fallback_providers.append(fallback_provider)
            else:
                logger.warning(f"Skipping fallback provider: {fallback_config.get('type')}")

        # Get router settings
        router_config = self.config.get("router", {})
        retry_on_rate_limit = router_config.get("retry_on_rate_limit", False)
        verbose_logging = router_config.get("verbose_logging", False)

        # Create router
        try:
            router = LLMRouter(
                primary_provider=primary_provider,
                fallback_providers=fallback_providers,
                retry_on_rate_limit=retry_on_rate_limit,
                verbose_logging=verbose_logging
            )

            logger.info(f"LLM Router created with {len(fallback_providers)} fallback(s)")
            return router

        except Exception as e:
            logger.error(f"Failed to create LLM router: {e}")
            return None

    def get_cost_limits(self) -> Dict[str, float]:
        """Get cost tracking limits from configuration."""
        cost_config = self.config.get("cost_tracking", {})
        return {
            "enabled": cost_config.get("enabled", True),
            "daily_limit_usd": cost_config.get("daily_limit_usd", 50.0),
            "monthly_limit_usd": cost_config.get("monthly_limit_usd", 500.0),
            "alert_threshold": cost_config.get("alert_threshold", 0.80)
        }

    def get_model_alias(self, alias: str) -> Optional[str]:
        """
        Resolve model alias to full model name.

        Args:
            alias: Model alias (e.g., "sonnet", "gemini-flash")

        Returns:
            Full model name or None if alias not found
        """
        aliases = self.config.get("aliases", {})
        return aliases.get(alias)


# Global instance (lazy-loaded)
_global_router: Optional[LLMRouter] = None


def get_llm_router(config_path: str = "config/llm.yaml", force_reload: bool = False) -> LLMRouter:
    """
    Get global LLM router instance.

    Args:
        config_path: Path to configuration file
        force_reload: Force reload configuration

    Returns:
        Configured LLMRouter

    Raises:
        RuntimeError: If router creation fails
    """
    global _global_router

    if _global_router is None or force_reload:
        loader = ProviderConfigLoader(config_path)
        _global_router = loader.create_router()

        if _global_router is None:
            raise RuntimeError("Failed to create LLM router from configuration")

    return _global_router
