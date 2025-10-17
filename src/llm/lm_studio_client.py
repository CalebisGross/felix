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
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI
import httpx
from collections import deque

logger = logging.getLogger(__name__)


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
    """Response from LLM completion."""
    content: str
    tokens_used: int
    response_time: float
    model: str
    temperature: float
    agent_id: str
    timestamp: float


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
                 timeout: float = 120.0, max_concurrent_requests: int = 4, debug_mode: bool = False):
        """
        Initialize LM Studio client.
        
        Args:
            base_url: LM Studio API endpoint
            timeout: Request timeout in seconds
            max_concurrent_requests: Maximum concurrent async requests
            debug_mode: Enable verbose debug output
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_concurrent_requests = max_concurrent_requests
        self.debug_mode = debug_mode
        
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
    
    def test_connection(self) -> bool:
        """
        Test connection to LM Studio server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = httpx.get(f"{self.base_url.rstrip('/v1')}/health", 
                               timeout=5.0)
            self._connection_verified = response.status_code == 200
            return self._connection_verified
        except Exception as e:
            logger.warning(f"LM Studio connection test failed: {e}")
            return False
    
    def ensure_connection(self) -> None:
        """Ensure connection to LM Studio or raise exception."""
        if not self._connection_verified and not self.test_connection():
            raise LMStudioConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Ensure LM Studio is running with a model loaded."
            )
    
    def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                 temperature: float = 0.7, max_tokens: Optional[int] = 500,
                 model: str = "local-model") -> LLMResponse:
        """
        Synchronous completion request to LM Studio.
        
        Args:
            agent_id: Identifier for the requesting agent
            system_prompt: System/context prompt
            user_prompt: User query/task
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            model: Model identifier (LM Studio will use loaded model)
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            LMStudioConnectionError: If cannot connect to LM Studio
        """
        self.ensure_connection()
        
        start_time = time.perf_counter()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
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
            
            # Update usage tracking
            self.total_tokens += tokens_used
            self.total_requests += 1
            self.total_response_time += response_time
            
            if self.debug_mode:
                print(f"✅ LLM RESPONSE for {agent_id}")
                print(f"📄 Content ({len(content)} chars):\n{content}")
                print(f"📊 Tokens Used: {tokens_used}, Time: {response_time:.2f}s")
                print("━" * 60)
            
            logger.debug(f"LLM completion for {agent_id}: {tokens_used} tokens, "
                        f"{response_time:.2f}s")
            
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
                
                # Update usage tracking
                self.total_tokens += tokens_used
                self.total_requests += 1
                self.total_response_time += response_time
                
                if self.debug_mode:
                    print(f"✅ ASYNC LLM RESPONSE for {agent_id}")
                    print(f"📄 Content ({len(content)} chars):\n{content}")
                    print(f"📊 Tokens Used: {tokens_used}, Time: {response_time:.2f}s")
                    print("━" * 60)
                
                logger.debug(f"Async LLM completion for {agent_id}: {tokens_used} tokens, "
                           f"{response_time:.2f}s")
                
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
    
    def create_agent_system_prompt(self, agent_type: str, position_info: Dict[str, float],
                                 task_context: str = "") -> str:
        """
        Create system prompt for Felix agent based on position and type.
        
        Args:
            agent_type: Type of agent (research, analysis, synthesis, critic)
            position_info: Agent's position on helix (x, y, z, radius, depth_ratio)
            task_context: Additional context about the current task
            
        Returns:
            Formatted system prompt
        """
        depth_ratio = position_info.get("depth_ratio", 0.0)
        radius = position_info.get("radius", 0.0)
        
        base_prompt = f"""🚨 IMPORTANT: You are a {agent_type} agent in the Felix multi-agent system with STRICT OUTPUT LIMITS.

⚠️ CRITICAL INSTRUCTION: Your response will be HARD-LIMITED and CUT OFF if too long. WRITE CONCISELY.

Current Position:
- Depth: {depth_ratio:.2f} (0.0 = top/start, 1.0 = bottom/end)
- Radius: {radius:.2f} (decreasing as you progress)
- Processing Stage: {"Early/Broad" if depth_ratio < 0.3 else "Middle/Focused" if depth_ratio < 0.7 else "Final/Precise"}

Your Role Based on Position:
"""
        
        if agent_type == "research":
            if depth_ratio < 0.3:
                base_prompt += "- MAXIMUM 5 bullet points with key facts ONLY\n"
                base_prompt += "- NO explanations, NO introductions, NO conclusions\n"
                base_prompt += "- Raw findings only - be direct\n"
            else:
                base_prompt += "- MAXIMUM 3 specific facts with numbers/dates/quotes\n"
                base_prompt += "- NO background context or elaboration\n"
                base_prompt += "- Prepare key points for analysis (concise)\n"
        
        elif agent_type == "analysis":
            base_prompt += "- MAXIMUM 2 numbered insights/patterns ONLY\n"
            base_prompt += "- NO background explanation or context\n"
            base_prompt += "- Direct analytical findings only\n"
            
        elif agent_type == "synthesis":
            base_prompt += "- FINAL output ONLY - NO process description\n"
            base_prompt += "- MAXIMUM 3 short paragraphs\n"
            base_prompt += "- Direct, actionable content without fluff\n"
            
        elif agent_type == "critic":
            base_prompt += "- MAXIMUM 3 specific issues/fixes ONLY\n"
            base_prompt += "- NO praise, NO general comments\n"
            base_prompt += "- Direct problems and solutions only\n"
        
        if task_context:
            base_prompt += f"\nTask Context: {task_context}\n"
        
        base_prompt += "\n🚨 FINAL REMINDER: Your response must be EXTREMELY CONCISE. Your output will be CUT OFF if it exceeds your token limit. "
        base_prompt += "Early positions focus on breadth, later positions focus on depth and precision. BE BRIEF!"
        
        return base_prompt


def create_default_client(max_concurrent_requests: int = 4) -> LMStudioClient:
    """Create LM Studio client with default settings."""
    return LMStudioClient(max_concurrent_requests=max_concurrent_requests)