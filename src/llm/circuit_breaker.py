"""
Circuit Breaker Pattern for LLM Providers

Prevents cascading failures by failing fast when a provider is experiencing issues.
Implements three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Provider is failing, requests fail immediately
- HALF_OPEN: Testing if provider has recovered

Issue #17: https://github.com/CalebisGross/felix/issues/17
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass
from threading import Lock, Event
from typing import Optional, Callable, List

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderConnectionError,
)

logger = logging.getLogger('felix_workflows')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing fast, not attempting calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 3        # Consecutive failures before opening
    recovery_timeout: float = 60.0    # Seconds before testing recovery
    half_open_max_calls: int = 1      # Test calls allowed in half-open state


class CircuitBreakerError(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(self, provider_name: str, time_until_retry: float):
        self.provider_name = provider_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker OPEN for {provider_name}. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation.

    Usage:
        breaker = CircuitBreaker("lm_studio")

        with breaker:
            # Call that might fail
            result = provider.complete(request)
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = Lock()

        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._rejected_calls = 0
        self._state_transitions: List[dict] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit state (checks for automatic transitions)."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state

        self._state_transitions.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': time.time()
        })

        logger.info(f"Circuit breaker [{self.name}]: {old_state.value} -> {new_state.value}")

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

    def _allow_request(self) -> bool:
        """Check if request should be allowed through (called within lock)."""
        self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            return False
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited test calls
            return self._half_open_calls < self.config.half_open_max_calls
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._successful_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                # Recovery successful - close circuit
                self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success (consecutive failures matter)
                self._failure_count = 0

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker [{self.name}]: "
                f"Failure {self._failure_count}/{self.config.failure_threshold}"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed - reopen circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def __enter__(self):
        """Context manager entry - check if call is allowed."""
        with self._lock:
            self._total_calls += 1
            self._check_state_transition()

            if not self._allow_request():
                self._rejected_calls += 1
                time_until_retry = 0.0
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    time_until_retry = max(0, self.config.recovery_timeout - elapsed)
                raise CircuitBreakerError(self.name, time_until_retry)

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record result."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'total_calls': self._total_calls,
                'successful_calls': self._successful_calls,
                'rejected_calls': self._rejected_calls,
                'last_failure_time': self._last_failure_time,
                'recent_transitions': self._state_transitions[-5:]
            }

    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._last_failure_time = None


class CircuitBreakerProvider(BaseLLMProvider):
    """
    Wraps any BaseLLMProvider with circuit breaker protection.

    Usage:
        base_provider = LMStudioProvider(...)
        protected_provider = CircuitBreakerProvider(
            base_provider,
            config=CircuitBreakerConfig(failure_threshold=3)
        )
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: Optional[CircuitBreakerConfig] = None
    ):
        super().__init__()
        self._provider = provider
        self.provider_type = provider.provider_type
        self.default_model = provider.default_model
        self.verbose_logging = provider.verbose_logging

        # Create circuit breaker for this provider
        self._breaker = CircuitBreaker(
            name=provider.get_provider_name(),
            config=config
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete with circuit breaker protection."""
        try:
            with self._breaker:
                return self._provider.complete(request)
        except CircuitBreakerError as e:
            # Convert to ProviderConnectionError for router compatibility
            raise ProviderConnectionError(
                f"Circuit breaker open for {self._provider.get_provider_name()}: {e}"
            )

    def complete_streaming(
        self,
        request: LLMRequest,
        callback: Callable[[str], None],
        cancel_event: Optional[Event] = None
    ) -> LLMResponse:
        """Streaming complete with circuit breaker protection."""
        try:
            with self._breaker:
                return self._provider.complete_streaming(request, callback, cancel_event)
        except CircuitBreakerError as e:
            raise ProviderConnectionError(
                f"Circuit breaker open for {self._provider.get_provider_name()}: {e}"
            )

    def test_connection(self) -> bool:
        """Test connection - bypass circuit breaker for explicit tests."""
        return self._provider.test_connection()

    def get_available_models(self) -> List[str]:
        """Get models from wrapped provider."""
        return self._provider.get_available_models()

    def estimate_cost(self, tokens_used: int, model: str) -> float:
        """Estimate cost from wrapped provider."""
        return self._provider.estimate_cost(tokens_used, model)

    def get_provider_name(self) -> str:
        """Get provider name from wrapped provider."""
        return self._provider.get_provider_name()

    def get_circuit_breaker_metrics(self) -> dict:
        """Get circuit breaker status and metrics."""
        return self._breaker.get_metrics()

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self._breaker.reset()

    @property
    def circuit_state(self) -> CircuitState:
        """Current circuit breaker state."""
        return self._breaker.state

    def __repr__(self) -> str:
        return (
            f"<CircuitBreakerProvider wrapping={self._provider.get_provider_name()} "
            f"state={self._breaker.state.value}>"
        )
