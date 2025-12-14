"""
LM Studio client integration for Felix Framework.

This module provides a client interface to LM Studio's OpenAI-compatible API,
enabling LLM-powered agents in the Felix multi-agent system.

LM Studio runs a local server (typically http://localhost:1234/v1) that 
provides OpenAI-compatible API endpoints for local language model inference.
"""

import asyncio
import time
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import httpx
from collections import deque

logger = logging.getLogger(__name__)

# Suppress verbose httpx logging (INFO level logs every HTTP request)
# This prevents log spam with hundreds of "HTTP/1.1 200 OK" messages
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class RequestPriority(Enum):
    """Priority levels for async requests."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AsyncRequest:
    """Async request with priority and metadata."""
    agent_id: str
    system_prompt: str
    user_prompt: str
    temperature: float
    max_tokens: Optional[int]
    model: str
    priority: RequestPriority
    future: asyncio.Future
    timestamp: float


@dataclass
class LLMResponse:
    """
    Internal response from LM Studio client.

    NOTE: This is the low-level client response format used internally by LMStudioClient.
    For provider-agnostic code, use src.llm.base_provider.LLMResponse instead.
    The LMStudioProvider converts this internal format to the standard LLMResponse.
    """
    content: str
    tokens_used: int
    response_time: float
    model: str
    temperature: float
    agent_id: str
    timestamp: float


@dataclass
class StreamingChunk:
    """Incremental streaming chunk from LLM with time-batching."""
    content: str           # Partial text since last batch
    accumulated: str       # Full content accumulated so far
    tokens_so_far: int     # Running token count (approximate)
    agent_id: str
    timestamp: float


class TokenAwareStreamController:
    """
    Controls streaming LLM generation with real-time token budget awareness.

    Monitors token usage during streaming and provides graceful conclusion
    signals when approaching budget limits, preventing abrupt truncation
    while maintaining output quality.

    Features:
    - Real-time token counting during generation
    - Soft limit (85%) triggers conclusion signal
    - Hard limit (100%) stops stream
    - Token efficiency metrics for agent learning
    """

    def __init__(self, token_budget: int, soft_limit_ratio: float = 0.85,
                 conclusion_signal: Optional[str] = None):
        """
        Initialize token-aware stream controller.

        Args:
            token_budget: Maximum tokens allowed for this generation
            soft_limit_ratio: Ratio at which to inject conclusion signal (default 0.85)
            conclusion_signal: Optional signal to inject at soft limit
        """
        self.token_budget = token_budget
        self.soft_limit = int(token_budget * soft_limit_ratio)
        self.hard_limit = token_budget
        self.tokens_generated = 0
        self.should_stop = False
        self.conclusion_injected = False

        # Default conclusion signal (subtle)
        self.conclusion_signal = conclusion_signal or ""

        logger.debug(f"TokenAwareStreamController initialized: "
                    f"budget={token_budget}, soft={self.soft_limit}, hard={self.hard_limit}")

    def process_chunk(self, chunk: str) -> tuple[str, bool]:
        """
        Process streaming chunk and determine if generation should continue.

        Uses rough token estimation: ~0.75 words per token (1.33 tokens per word).
        More accurate than character count, accounts for multi-byte tokens.

        Args:
            chunk: Text chunk from LLM stream

        Returns:
            Tuple of (processed_chunk, should_continue)
            - processed_chunk: Chunk potentially modified with signals
            - should_continue: False to stop streaming, True to continue
        """
        # Estimate tokens in this chunk (rough but effective)
        # Average: 1 token ≈ 0.75 words, so 1 word ≈ 1.33 tokens
        word_count = len(chunk.split())
        estimated_tokens = int(word_count * 1.33)
        self.tokens_generated += estimated_tokens

        # HARD LIMIT - stop immediately, don't emit this chunk
        if self.tokens_generated >= self.hard_limit:
            logger.debug(f"TokenAwareStreamController: HARD LIMIT reached "
                        f"({self.tokens_generated}/{self.hard_limit} tokens) - stopping stream")
            return "", False

        # SOFT LIMIT - inject conclusion signal once, continue briefly
        elif self.tokens_generated >= self.soft_limit and not self.conclusion_injected:
            logger.debug(f"TokenAwareStreamController: SOFT LIMIT reached "
                        f"({self.tokens_generated}/{self.soft_limit} tokens) - injecting conclusion signal")
            self.conclusion_injected = True
            self.should_stop = True

            # Inject conclusion signal if provided
            if self.conclusion_signal:
                return chunk + self.conclusion_signal, True

            return chunk, True

        # Normal operation - continue streaming
        return chunk, True

    def get_efficiency_ratio(self) -> float:
        """
        Calculate token efficiency ratio (used / allocated).

        Returns:
            Efficiency ratio (e.g., 0.85 means used 85% of budget)
        """
        if self.token_budget == 0:
            return 1.0
        return self.tokens_generated / self.token_budget

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get token usage metrics for agent learning.

        Returns:
            Dictionary with token usage statistics
        """
        return {
            "tokens_allocated": self.token_budget,
            "tokens_used": self.tokens_generated,
            "efficiency_ratio": self.get_efficiency_ratio(),
            "soft_limit_reached": self.conclusion_injected,
            "hard_limit_reached": self.tokens_generated >= self.hard_limit,
            "tokens_remaining": max(0, self.token_budget - self.tokens_generated)
        }


class LMStudioConnectionError(Exception):
    """Raised when cannot connect to LM Studio."""
    pass


class LMStudioClient:
    """
    Client for communicating with LM Studio's local API server.
    
    Provides both synchronous and asynchronous methods for LLM completion,
    with built-in error handling, connection testing, and usage tracking.
    """
    
    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 timeout: float = 120.0, max_concurrent_requests: int = 4,
                 debug_mode: bool = False, verbose_logging: bool = True):
        """
        Initialize LM Studio client.

        Args:
            base_url: LM Studio API endpoint
            timeout: Request timeout in seconds
            max_concurrent_requests: Maximum concurrent async requests
            debug_mode: Enable verbose debug output to console
            verbose_logging: Enable detailed logging to Felix logging system (for GUI)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.debug_mode = debug_mode
        self.verbose_logging = verbose_logging

        # Ensure logger is configured
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        logger.propagate = True

        # Log initialization with verbose setting
        logger.info(f"LMStudioClient initialized: base_url={base_url}, verbose_logging={verbose_logging}")
        if debug_mode:
            logger.debug(f"Debug mode enabled for LMStudioClient")

        # Sync client
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # LM Studio doesn't require API keys
            timeout=timeout
        )
        
        # Async client and connection pool
        self._async_client: Optional[httpx.AsyncClient] = None
        self._connection_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Request queue for async processing
        self._request_queue: deque[AsyncRequest] = deque()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._is_processing_queue = False
        
        # Usage tracking
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
        self.concurrent_requests = 0
        
        # Connection state
        self._connection_verified = False

        # Sync request concurrency control (for background vs foreground priority)
        self._sync_semaphore = threading.Semaphore(2)  # Max 2 concurrent sync requests
        self._background_processing = threading.Event()  # Pause flag for background tasks
        self._background_processing.set()  # Initially not paused (background can proceed)
    
    def test_connection(self) -> bool:
        """
        Test connection to LM Studio server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = httpx.get(f"{self.base_url}/models", timeout=5.0)
            self._connection_verified = response.status_code == 200
            return self._connection_verified
        except RecursionError:
            # Catch recursion errors explicitly to prevent them from being logged
            # in a way that might cause further recursion
            logger.warning("LM Studio connection test failed: recursion error")
            return False
        except Exception as e:
            # Use type(e).__name__ to safely get exception type without potential
            # recursion issues from complex exception __str__ methods
            error_type = type(e).__name__
            error_msg = str(e) if len(str(e)) < 200 else f"{str(e)[:200]}..."
            logger.warning(f"LM Studio connection test failed ({error_type}): {error_msg}")
            return False
    
    def ensure_connection(self) -> None:
        """Ensure connection to LM Studio or raise exception."""
        if not self._connection_verified and not self.test_connection():
            raise LMStudioConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Ensure LM Studio is running with a model loaded."
            )

    def test_embedding_availability(self, timeout: float = 5.0) -> bool:
        """
        Quick test if embedding capability is available.

        Uses a short timeout suitable for initialization checks.
        Does not use the main OpenAI client to avoid long timeouts.

        Args:
            timeout: Maximum seconds to wait for response

        Returns:
            True if embeddings are available, False otherwise
        """
        if not self.test_connection():
            return False

        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": "local-model", "input": "test"},
                timeout=timeout
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Embedding availability check failed: {e}")
            return False

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

    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: Optional[int] = 500,
                 model: str = "local-model", is_background: bool = False) -> LLMResponse:
        """
        Synchronous completion request to LM Studio.

        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier (LM Studio will use loaded model)
            is_background: If True, this is a background task that should yield
                          to user-initiated requests (e.g., Knowledge Brain batch processing)

        Returns:
            LLMResponse with content and metadata

        Raises:
            LMStudioConnectionError: If cannot connect to LM Studio
        """
        self.ensure_connection()

        # Background tasks wait for user activity to finish before proceeding
        if is_background:
            # Wait up to 5 seconds for user activity to stop, then proceed anyway
            self._background_processing.wait(timeout=5.0)

        # Acquire semaphore to limit concurrent sync requests
        with self._sync_semaphore:
            start_time = time.perf_counter()

            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Log request details at INFO level for GUI visibility (if enabled)
                if self.verbose_logging:
                    system_preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
                    user_preview = user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt

                    logger.info("=" * 60)
                    logger.info(f"LLM REQUEST to LM Studio")
                    logger.info(f"  Agent: {agent_id}")
                    logger.info(f"  Model: {model}")
                    logger.info(f"  Temperature: {temperature}, Max Tokens: {max_tokens}")
                    logger.info(f"  System Prompt ({len(system_prompt)} chars): {system_preview}")
                    logger.info(f"  User Prompt ({len(user_prompt)} chars): {user_preview}")
                    logger.info("=" * 60)

                if self.debug_mode:
                    print(f"\n🔍 DEBUG LLM CALL for {agent_id}")
                    print(f"📝 System Prompt:\n{system_prompt}")
                    print(f"🎯 User Prompt:\n{user_prompt}")
                    print(f"🌡️ Temperature: {temperature}, Max Tokens: {max_tokens}")
                    print("━" * 60)

                completion_args = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }

                if max_tokens:
                    completion_args["max_tokens"] = max_tokens

                response = self.client.chat.completions.create(**completion_args)

                end_time = time.perf_counter()
                response_time = end_time - start_time

                # Extract response data
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0

                # Update usage tracking
                self.total_tokens += tokens_used
                self.total_requests += 1
                self.total_response_time += response_time

                # Log response details at INFO level for GUI visibility (if enabled)
                if self.verbose_logging:
                    content_preview = content[:200] + "..." if len(content) > 200 else content

                    logger.info("=" * 60)
                    logger.info(f"LLM RESPONSE from LM Studio")
                    logger.info(f"  Agent: {agent_id}")
                    logger.info(f"  Model: {model}")
                    logger.info(f"  Response Time: {response_time:.2f}s")
                    logger.info(f"  Tokens: {tokens_used} total (prompt: {prompt_tokens}, completion: {completion_tokens})")
                    logger.info(f"  Content ({len(content)} chars): {content_preview}")
                    logger.info("=" * 60)

                if self.debug_mode:
                    print(f"✅ LLM RESPONSE for {agent_id}")
                    print(f"📄 Content ({len(content)} chars):\n{content}")
                    print(f"📊 Tokens Used: {tokens_used}, Time: {response_time:.2f}s")
                    print("━" * 60)

                return LLMResponse(
                    content=content,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    model=model,
                    temperature=temperature,
                    agent_id=agent_id,
                    timestamp=time.time()
                )

            except Exception as e:
                logger.error(f"LLM completion failed for {agent_id}: {e}")
                raise

    def generate_embedding(self, text: str, model: str = "local-model") -> Optional[List[float]]:
        """
        Generate embedding vector for text using LM Studio's embedding endpoint.

        LM Studio supports OpenAI-compatible embeddings API when an embedding-capable
        model is loaded (e.g., text-embedding-ada-002, nomic-embed-text).

        Args:
            text: Input text to embed
            model: Model identifier (LM Studio will use loaded embedding model)

        Returns:
            List of floats (embedding vector) or None if failed

        Raises:
            LMStudioConnectionError: If cannot connect to LM Studio
        """
        self.ensure_connection()

        try:
            # Use OpenAI-compatible embeddings API
            response = self.client.embeddings.create(
                model=model,
                input=text
            )

            # Extract embedding from response
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding

                if self.verbose_logging:
                    logger.info(f"Generated embedding for text ({len(text)} chars), "
                               f"dimension: {len(embedding)}")

                return embedding

            logger.warning("LM Studio embedding response contained no data")
            return None

        except Exception as e:
            # This is expected if LM Studio doesn't have an embedding model loaded
            logger.debug(f"Embedding generation failed: {e}")
            return None

    def complete_streaming(
        self,
        agent_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: str = "local-model",
        batch_interval: float = 0.1,
        callback: Optional[Callable[[StreamingChunk], None]] = None,
        token_controller: Optional[TokenAwareStreamController] = None,
        cancel_event: Optional[threading.Event] = None
    ) -> LLMResponse:
        """
        Stream LLM completion with time-batched token delivery and optional budget control.

        Tokens are accumulated and sent to callback every `batch_interval` seconds
        to create real-time feel without excessive overhead.

        Args:
            agent_id: Identifier for requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier
            batch_interval: Time between partial updates (default 0.1s = 100ms)
            callback: Called with batched chunks for real-time updates
            token_controller: Optional TokenAwareStreamController for budget enforcement
            cancel_event: Optional threading.Event to signal cancellation

        Returns:
            Complete LLMResponse after stream finishes

        Raises:
            LMStudioConnectionError: If cannot connect to LM Studio
        """
        self.ensure_connection()

        start_time = time.perf_counter()
        accumulated_content = ""
        batch_buffer = ""  # Buffer for time-based batching
        last_batch_time = time.time()
        tokens_so_far = 0

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if self.verbose_logging:
                system_preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
                user_preview = user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt

                logger.info("=" * 60)
                logger.info(f"STREAMING LLM REQUEST (time-batched, interval={batch_interval}s)")
                logger.info(f"  Agent: {agent_id}")
                logger.info(f"  Model: {model}")
                logger.info(f"  Temperature: {temperature}, Max Tokens: {max_tokens}")
                logger.info(f"  System Prompt ({len(system_prompt)} chars): {system_preview}")
                logger.info(f"  User Prompt ({len(user_prompt)} chars): {user_preview}")
                logger.info("=" * 60)

            if self.debug_mode:
                print(f"\n🔍 DEBUG STREAMING LLM CALL for {agent_id}")
                print(f"📝 System Prompt:\n{system_prompt}")
                print(f"🎯 User Prompt:\n{user_prompt}")
                print(f"🌡️ Temperature: {temperature}, Max Tokens: {max_tokens}")
                print(f"⏱️ Batch Interval: {batch_interval}s")
                print("━" * 60)

            # Create streaming completion using OpenAI client
            # LM Studio supports this on /v1/chat/completions endpoint
            completion_args = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True  # Enable streaming!
            }

            if max_tokens:
                completion_args["max_tokens"] = max_tokens

            stream = self.client.chat.completions.create(**completion_args)

            # Process stream with time-based batching and optional token control
            for chunk in stream:
                # Check for cancellation signal
                if cancel_event and cancel_event.is_set():
                    logger.info(f"Streaming cancelled by user for {agent_id}")
                    break

                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content

                    # Token-aware processing (if controller provided)
                    if token_controller:
                        processed_chunk, should_continue = token_controller.process_chunk(delta_content)
                        if not should_continue:
                            logger.info(f"Token controller stopped stream for {agent_id} "
                                       f"at {token_controller.tokens_generated} tokens")
                            # Don't add the rejected chunk, break immediately
                            break
                        # Use processed chunk (may have conclusion signal injected)
                        delta_content = processed_chunk

                    accumulated_content += delta_content
                    batch_buffer += delta_content
                    tokens_so_far += 1  # Approximate (1 token ≈ 1 chunk)

                    # Check if batch interval elapsed
                    current_time = time.time()
                    if current_time - last_batch_time >= batch_interval:
                        # Send batched chunk via callback
                        if callback and batch_buffer:
                            streaming_chunk = StreamingChunk(
                                content=batch_buffer,
                                accumulated=accumulated_content,
                                tokens_so_far=tokens_so_far,
                                agent_id=agent_id,
                                timestamp=current_time
                            )
                            callback(streaming_chunk)

                        # Reset batch
                        batch_buffer = ""
                        last_batch_time = current_time

            # Send final batch if remaining
            if batch_buffer and callback:
                streaming_chunk = StreamingChunk(
                    content=batch_buffer,
                    accumulated=accumulated_content,
                    tokens_so_far=tokens_so_far,
                    agent_id=agent_id,
                    timestamp=time.time()
                )
                callback(streaming_chunk)

            end_time = time.perf_counter()
            response_time = end_time - start_time

            # Update tracking
            self.total_tokens += tokens_so_far
            self.total_requests += 1
            self.total_response_time += response_time

            if self.verbose_logging:
                content_preview = accumulated_content[:200] + "..." if len(accumulated_content) > 200 else accumulated_content

                logger.info("=" * 60)
                logger.info(f"STREAMING LLM COMPLETE")
                logger.info(f"  Agent: {agent_id}")
                logger.info(f"  Duration: {response_time:.2f}s")
                logger.info(f"  Tokens: ~{tokens_so_far}")
                logger.info(f"  Content ({len(accumulated_content)} chars): {content_preview}")
                logger.info("=" * 60)

            if self.debug_mode:
                print(f"✅ STREAMING LLM COMPLETE for {agent_id}")
                print(f"📄 Content ({len(accumulated_content)} chars):\n{accumulated_content}")
                print(f"📊 Tokens Used: ~{tokens_so_far}, Time: {response_time:.2f}s")
                print("━" * 60)

            return LLMResponse(
                content=accumulated_content,
                tokens_used=tokens_so_far,
                response_time=response_time,
                model=model,
                temperature=temperature,
                agent_id=agent_id,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"Streaming LLM completion failed for {agent_id}: {e}")
            raise

    async def _ensure_async_client(self) -> httpx.AsyncClient:
        """Ensure async client is initialized."""
        if self._async_client is None:
            limits = httpx.Limits(
                max_connections=self.max_concurrent_requests,
                max_keepalive_connections=self.max_concurrent_requests
            )
            timeout = httpx.Timeout(self.timeout)
            
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                limits=limits,
                headers={"Content-Type": "application/json"}
            )
        return self._async_client
    
    async def _make_async_request(self, agent_id: str, system_prompt: str, 
                                user_prompt: str, temperature: float = 0.7,
                                max_tokens: Optional[int] = None,
                                model: str = "local-model") -> LLMResponse:
        """Make actual async HTTP request to LM Studio."""
        async with self._connection_semaphore:
            start_time = time.perf_counter()
            self.concurrent_requests += 1
            
            try:
                client = await self._ensure_async_client()
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Log request details at INFO level for GUI visibility (if enabled)
                if self.verbose_logging:
                    system_preview = system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
                    user_preview = user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt

                    logger.info("=" * 60)
                    logger.info(f"ASYNC LLM REQUEST to LM Studio")
                    logger.info(f"  Agent: {agent_id}")
                    logger.info(f"  Model: {model}")
                    logger.info(f"  Temperature: {temperature}, Max Tokens: {max_tokens}")
                    logger.info(f"  System Prompt ({len(system_prompt)} chars): {system_preview}")
                    logger.info(f"  User Prompt ({len(user_prompt)} chars): {user_preview}")
                    logger.info("=" * 60)

                if self.debug_mode:
                    print(f"\n🔍 DEBUG ASYNC LLM CALL for {agent_id}")
                    print(f"📝 System Prompt:\n{system_prompt}")
                    print(f"🎯 User Prompt:\n{user_prompt}")
                    print(f"🌡️ Temperature: {temperature}, Max Tokens: {max_tokens}")
                    print("━" * 60)

                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False
                }

                if max_tokens:
                    payload["max_tokens"] = max_tokens

                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()

                data = response.json()

                end_time = time.perf_counter()
                response_time = end_time - start_time

                # Extract response data
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = data.get("usage", {}).get("completion_tokens", 0)

                # Update usage tracking
                self.total_tokens += tokens_used
                self.total_requests += 1
                self.total_response_time += response_time

                # Log response details at INFO level for GUI visibility (if enabled)
                if self.verbose_logging:
                    content_preview = content[:200] + "..." if len(content) > 200 else content

                    logger.info("=" * 60)
                    logger.info(f"ASYNC LLM RESPONSE from LM Studio")
                    logger.info(f"  Agent: {agent_id}")
                    logger.info(f"  Model: {model}")
                    logger.info(f"  Response Time: {response_time:.2f}s")
                    logger.info(f"  Tokens: {tokens_used} total (prompt: {prompt_tokens}, completion: {completion_tokens})")
                    logger.info(f"  Content ({len(content)} chars): {content_preview}")
                    logger.info("=" * 60)

                if self.debug_mode:
                    print(f"✅ ASYNC LLM RESPONSE for {agent_id}")
                    print(f"📄 Content ({len(content)} chars):\n{content}")
                    print(f"📊 Tokens Used: {tokens_used}, Time: {response_time:.2f}s")
                    print("━" * 60)
                
                return LLMResponse(
                    content=content,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    model=model,
                    temperature=temperature,
                    agent_id=agent_id,
                    timestamp=time.time()
                )
                
            except Exception as e:
                logger.error(f"Async LLM completion failed for {agent_id}: {e}")
                raise
            finally:
                self.concurrent_requests -= 1
    
    async def complete_async(self, agent_id: str, system_prompt: str, 
                           user_prompt: str, temperature: float = 0.7,
                           max_tokens: Optional[int] = None,
                           model: str = "local-model",
                           priority: RequestPriority = RequestPriority.NORMAL) -> LLMResponse:
        """
        Asynchronous completion request to LM Studio with priority support.
        
        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier
            priority: Request priority level
            
        Returns:
            LLMResponse with content and metadata
        """
        # For high priority requests, execute immediately
        if priority == RequestPriority.URGENT:
            return await self._make_async_request(
                agent_id, system_prompt, user_prompt, temperature, max_tokens, model
            )
        
        # For normal/low priority, use queue system
        future = asyncio.Future()
        request = AsyncRequest(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            priority=priority,
            future=future,
            timestamp=time.time()
        )
        
        self._request_queue.append(request)
        await self._ensure_queue_processor()
        
        return await future
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get client usage statistics.
        
        Returns:
            Dictionary with usage metrics
        """
        avg_response_time = (self.total_response_time / self.total_requests 
                           if self.total_requests > 0 else 0.0)
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time,
            "average_tokens_per_request": (self.total_tokens / self.total_requests
                                         if self.total_requests > 0 else 0.0),
            "connection_verified": self._connection_verified,
            "max_concurrent_requests": self.max_concurrent_requests,
            "current_concurrent_requests": self.concurrent_requests,
            "queue_size": len(self._request_queue)
        }
    
    async def _ensure_queue_processor(self) -> None:
        """Ensure queue processor task is running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_request_queue())
    
    async def _process_request_queue(self) -> None:
        """Process queued async requests with priority ordering."""
        if self._is_processing_queue:
            return
        
        self._is_processing_queue = True
        
        try:
            while self._request_queue:
                # Sort queue by priority (higher priority first)
                sorted_requests = sorted(self._request_queue, key=lambda r: r.priority.value, reverse=True)
                
                # Process up to max_concurrent_requests at once
                batch_size = min(len(sorted_requests), self.max_concurrent_requests)
                batch = [sorted_requests[i] for i in range(batch_size)]
                
                # Remove processed requests from queue
                for req in batch:
                    self._request_queue.remove(req)
                
                # Process batch concurrently
                tasks = []
                for req in batch:
                    task = asyncio.create_task(self._execute_queued_request(req))
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay to prevent busy waiting
                if not self._request_queue:
                    break
                await asyncio.sleep(0.01)
                    
        finally:
            self._is_processing_queue = False
    
    async def _execute_queued_request(self, request: AsyncRequest) -> None:
        """Execute a single queued request."""
        try:
            result = await self._make_async_request(
                request.agent_id,
                request.system_prompt, 
                request.user_prompt,
                request.temperature,
                request.max_tokens,
                request.model
            )
            request.future.set_result(result)
        except Exception as e:
            request.future.set_exception(e)
    
    async def close_async(self) -> None:
        """Close async client and cleanup resources."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_requests = 0
        self.total_response_time = 0.0
    


def create_default_client(max_concurrent_requests: int = 4) -> LMStudioClient:
    """Create LM Studio client with default settings."""
    return LMStudioClient(max_concurrent_requests=max_concurrent_requests)