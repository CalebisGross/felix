"""
Multi-Server LM Studio Client Pool for Felix Framework.

This module enables true parallel processing by allowing agents to use
different LM Studio servers and models simultaneously, removing the
bottleneck of a single server.

Key Features:
- Multiple LM Studio server support
- Agent-type to server/model mapping
- Load balancing and health checks
- Failover and fault tolerance
- Performance monitoring per server

Usage:
    pool = LMStudioClientPool("config/server_config.json")
    response = await pool.complete_for_agent_type("research", system_prompt, user_prompt)
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .lm_studio_client import LMStudioClient, LLMResponse, RequestPriority, LMStudioConnectionError

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """Load balancing strategies for server selection."""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    FASTEST_RESPONSE = "fastest_response"
    AGENT_TYPE_MAPPING = "agent_type_mapping"


@dataclass
class ServerConfig:
    """Configuration for a single LM Studio server."""
    name: str
    url: str
    model: str
    timeout: float = 120.0
    max_concurrent: int = 4
    weight: float = 1.0  # For weighted load balancing
    enabled: bool = True


@dataclass
class ServerStats:
    """Runtime statistics for a server."""
    total_requests: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    current_load: int = 0  # Current active requests
    last_health_check: float = 0.0
    health_status: bool = True
    average_response_time: float = 0.0
    
    def update_stats(self, tokens: int, response_time: float):
        """Update server statistics."""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.total_requests


class LMStudioClientPool:
    """
    Pool of LM Studio clients for multi-server parallel processing.
    
    Manages multiple LM Studio servers, assigns requests to appropriate
    servers based on agent types, and provides load balancing and
    failover capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, debug_mode: bool = False):
        """
        Initialize the client pool.
        
        Args:
            config_path: Path to server configuration JSON file
            debug_mode: Enable verbose debug output
        """
        self.debug_mode = debug_mode
        self.config_path = config_path
        
        # Server management
        self.servers: Dict[str, ServerConfig] = {}
        self.clients: Dict[str, LMStudioClient] = {}
        self.stats: Dict[str, ServerStats] = {}
        
        # Agent type mapping
        self.agent_mappings: Dict[str, str] = {}
        
        # Load balancing
        self.load_balance_strategy = LoadBalanceStrategy.AGENT_TYPE_MAPPING
        self._round_robin_index = 0
        
        # Health monitoring
        self._health_check_interval = 30.0  # seconds
        self._last_global_health_check = 0.0
        
        # Load configuration
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default single-server configuration."""
        default_server = ServerConfig(
            name="default",
            url="http://localhost:1234/v1",
            model="local-model",
            timeout=120.0,
            max_concurrent=4
        )
        
        self.servers["default"] = default_server
        self.clients["default"] = LMStudioClient(
            base_url=default_server.url,
            timeout=default_server.timeout,
            max_concurrent_requests=default_server.max_concurrent,
            debug_mode=self.debug_mode
        )
        self.stats["default"] = ServerStats()
        
        # Default mapping: all agent types use default server
        self.agent_mappings = {
            "research": "default",
            "analysis": "default", 
            "synthesis": "default",
            "critic": "default"
        }
        
        if self.debug_mode:
            print("🔧 Using default single-server configuration")
    
    def load_config(self, config_path: str):
        """
        Load server configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load servers
            for server_data in config_data.get("servers", []):
                server_config = ServerConfig(
                    name=server_data["name"],
                    url=server_data["url"],
                    model=server_data["model"],
                    timeout=server_data.get("timeout", 120.0),
                    max_concurrent=server_data.get("max_concurrent", 4),
                    weight=server_data.get("weight", 1.0),
                    enabled=server_data.get("enabled", True)
                )
                
                self.servers[server_config.name] = server_config
                
                if server_config.enabled:
                    self.clients[server_config.name] = LMStudioClient(
                        base_url=server_config.url,
                        timeout=server_config.timeout,
                        max_concurrent_requests=server_config.max_concurrent,
                        debug_mode=self.debug_mode
                    )
                    self.stats[server_config.name] = ServerStats()
            
            # Load agent mappings
            self.agent_mappings = config_data.get("agent_mapping", {})
            
            # Load load balancing strategy
            strategy_name = config_data.get("load_balance_strategy", "agent_type_mapping")
            try:
                self.load_balance_strategy = LoadBalanceStrategy(strategy_name)
            except ValueError:
                self.load_balance_strategy = LoadBalanceStrategy.AGENT_TYPE_MAPPING
            
            if self.debug_mode:
                print(f"🔧 Loaded multi-server config: {len(self.servers)} servers")
                for name, server in self.servers.items():
                    status = "enabled" if server.enabled else "disabled"
                    print(f"   - {name}: {server.url} ({server.model}) [{status}]")
                
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            self._create_default_config()
    
    def get_server_for_agent_type(self, agent_type: str) -> Optional[str]:
        """
        Get the appropriate server for an agent type.
        
        Args:
            agent_type: Type of agent (research, analysis, synthesis, critic)
            
        Returns:
            Server name or None if no suitable server
        """
        if self.load_balance_strategy == LoadBalanceStrategy.AGENT_TYPE_MAPPING:
            return self.agent_mappings.get(agent_type, self._get_fallback_server())
        
        elif self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._get_round_robin_server()
        
        elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_BUSY:
            return self._get_least_busy_server()
        
        elif self.load_balance_strategy == LoadBalanceStrategy.FASTEST_RESPONSE:
            return self._get_fastest_server()
        
        return self._get_fallback_server()
    
    def _get_fallback_server(self) -> Optional[str]:
        """Get first available server as fallback."""
        for name, server in self.servers.items():
            if server.enabled and name in self.clients:
                return name
        return None
    
    def _get_round_robin_server(self) -> Optional[str]:
        """Get next server using round-robin selection."""
        available_servers = [name for name, server in self.servers.items() 
                           if server.enabled and name in self.clients]
        
        if not available_servers:
            return None
        
        server_name = available_servers[self._round_robin_index % len(available_servers)]
        self._round_robin_index = (self._round_robin_index + 1) % len(available_servers)
        return server_name
    
    def _get_least_busy_server(self) -> Optional[str]:
        """Get server with lowest current load."""
        available_servers = [(name, self.stats[name].current_load) 
                           for name, server in self.servers.items() 
                           if server.enabled and name in self.clients]
        
        if not available_servers:
            return None
        
        # Sort by current load (ascending)
        available_servers.sort(key=lambda x: x[1])
        return available_servers[0][0]
    
    def _get_fastest_server(self) -> Optional[str]:
        """Get server with fastest average response time."""
        available_servers = [(name, self.stats[name].average_response_time) 
                           for name, server in self.servers.items() 
                           if server.enabled and name in self.clients and self.stats[name].total_requests > 0]
        
        if not available_servers:
            return self._get_fallback_server()
        
        # Sort by average response time (ascending)
        available_servers.sort(key=lambda x: x[1])
        return available_servers[0][0]
    
    async def complete_for_agent_type(self, agent_type: str, agent_id: str, 
                                    system_prompt: str, user_prompt: str,
                                    temperature: float = 0.7, max_tokens: Optional[int] = None,
                                    priority: RequestPriority = RequestPriority.NORMAL) -> LLMResponse:
        """
        Complete request using appropriate server for agent type.
        
        Args:
            agent_type: Type of agent making request
            agent_id: ID of the requesting agent
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            priority: Request priority
            
        Returns:
            LLM response
            
        Raises:
            LMStudioConnectionError: If no servers available
        """
        server_name = self.get_server_for_agent_type(agent_type)
        
        if not server_name or server_name not in self.clients:
            raise LMStudioConnectionError(f"No available server for agent type: {agent_type}")
        
        # Check health if needed
        await self._check_server_health(server_name)
        
        client = self.clients[server_name]
        server_config = self.servers[server_name]
        stats = self.stats[server_name]
        
        if self.debug_mode:
            print(f"🌐 {agent_id} ({agent_type}) → {server_name} ({server_config.url}, {server_config.model})")
        
        # Track load
        stats.current_load += 1
        
        try:
            start_time = time.perf_counter()
            
            # Make request with specified model
            response = await client.complete_async(
                agent_id=agent_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=server_config.model,
                priority=priority
            )
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            # Update statistics
            stats.update_stats(response.tokens_used, response_time)
            
            return response
            
        finally:
            # Decrease load counter
            stats.current_load = max(0, stats.current_load - 1)
    
    async def complete(self, agent_id: str, system_prompt: str, user_prompt: str,
                     temperature: float = 0.7, max_tokens: Optional[int] = None,
                     priority: RequestPriority = RequestPriority.NORMAL) -> LLMResponse:
        """
        Complete request using default load balancing (fallback method).
        
        Args:
            agent_id: ID of the requesting agent  
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            priority: Request priority
            
        Returns:
            LLM response
        """
        # Extract agent type from agent_id if possible
        agent_type = "general"
        if "_" in agent_id:
            agent_type = agent_id.split("_")[0]
        
        return await self.complete_for_agent_type(
            agent_type, agent_id, system_prompt, user_prompt,
            temperature, max_tokens, priority
        )
    
    async def _check_server_health(self, server_name: str):
        """Check health of specific server."""
        current_time = time.time()
        stats = self.stats[server_name]
        
        # Only check if enough time has passed
        if current_time - stats.last_health_check < self._health_check_interval:
            return
        
        client = self.clients[server_name]
        
        try:
            health_ok = client.test_connection()
            stats.health_status = health_ok
            stats.last_health_check = current_time
            
            if not health_ok and self.debug_mode:
                print(f"⚠️  Server {server_name} health check failed")
                
        except Exception as e:
            stats.health_status = False
            stats.last_health_check = current_time
            logger.warning(f"Health check failed for {server_name}: {e}")
    
    async def health_check_all_servers(self) -> Dict[str, bool]:
        """
        Check health of all servers.
        
        Returns:
            Dictionary mapping server names to health status
        """
        health_results = {}
        
        for server_name in self.clients:
            await self._check_server_health(server_name)
            health_results[server_name] = self.stats[server_name].health_status
        
        return health_results
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the entire pool.
        
        Returns:
            Dictionary with pool statistics
        """
        pool_stats = {
            "total_servers": len(self.servers),
            "active_servers": len(self.clients),
            "load_balance_strategy": self.load_balance_strategy.value,
            "servers": {}
        }
        
        for name, stats in self.stats.items():
            server_config = self.servers[name]
            pool_stats["servers"][name] = {
                "config": {
                    "url": server_config.url,
                    "model": server_config.model,
                    "enabled": server_config.enabled
                },
                "stats": {
                    "total_requests": stats.total_requests,
                    "total_tokens": stats.total_tokens,
                    "average_response_time": stats.average_response_time,
                    "current_load": stats.current_load,
                    "health_status": stats.health_status
                }
            }
        
        return pool_stats
    
    def get_agent_mapping_info(self) -> Dict[str, str]:
        """Get current agent type to server mappings."""
        return self.agent_mappings.copy()
    
    async def close_all(self):
        """Close all client connections."""
        for client in self.clients.values():
            if hasattr(client, 'close_async'):
                await client.close_async()
    
    def display_pool_status(self):
        """Display current pool status for debugging."""
        if not self.debug_mode:
            return
            
        print(f"\n╭─ LM STUDIO POOL STATUS ─╮")
        print(f"│ Strategy: {self.load_balance_strategy.value}")
        print(f"│ Servers: {len(self.clients)} active / {len(self.servers)} total")
        
        for name, stats in self.stats.items():
            server = self.servers[name]
            status = "🟢" if stats.health_status else "🔴"
            load = f"{stats.current_load}/{server.max_concurrent}"
            avg_time = f"{stats.average_response_time:.2f}s" if stats.total_requests > 0 else "N/A"
            
            print(f"│ {status} {name}: {load} load, {stats.total_requests} reqs, {avg_time} avg")
        
        print(f"│ Agent Mapping:")
        for agent_type, server_name in self.agent_mappings.items():
            print(f"│   {agent_type} → {server_name}")
        print(f"╰{'─'*35}╯")