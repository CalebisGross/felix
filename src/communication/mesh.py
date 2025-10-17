"""
Mesh communication architecture implementation for Felix Framework comparison.

This module implements an all-to-all mesh communication topology to serve
as a baseline comparison against the spoke-based architecture. The mesh
topology provides O(N²) communication complexity for statistical validation
of Hypothesis H2.

Mathematical Foundation:
- Mesh topology: each agent connects to all other agents
- Message complexity: M_mesh = Σᵢ Σⱼ≠ᵢ m_ij = O(N²)
- Connection count: C_mesh = N(N-1)/2 (undirected connections)
- Latency model: L_ij = α + β·d_ij + ε_ij (distance-dependent)
- Memory overhead: O(N²) connection tracking vs O(N) spoke system

Key Features:
- All-to-all connectivity between agents
- Distance-based message latency simulation
- O(N²) scaling characteristics for comparison
- Performance metrics collection for Hypothesis H2 validation
- Message queuing and processing with delivery guarantees

This implementation supports Hypothesis H2 validation by providing measurable
communication overhead characteristics that demonstrate the complexity
advantage of spoke-based architecture.

Mathematical references:
- docs/hypothesis_mathematics.md, Section H2: Communication complexity analysis
- docs/mathematical_model.md: Theoretical foundations for comparison framework
"""

import time
import math
import statistics
import uuid
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass

from src.agents.agent import Agent


@dataclass
class MeshMessage:
    """
    Message structure for mesh communication system.
    
    Represents messages sent between agents in the mesh topology,
    including metadata for latency analysis and routing.
    """
    
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    message_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize message with validation and ID generation."""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        
        self._validate_message()
    
    def _validate_message(self):
        """Validate message parameters."""
        valid_types = {
            "TASK_REQUEST", "TASK_RESPONSE", "STATUS_UPDATE", 
            "COORDINATION", "ERROR"
        }
        
        if not self.sender_id or self.sender_id.strip() == "":
            raise ValueError("sender_id cannot be empty")
        
        if not self.recipient_id or self.recipient_id.strip() == "":
            raise ValueError("recipient_id cannot be empty")
        
        if self.message_type not in valid_types:
            raise ValueError(f"Invalid message type: {self.message_type}")
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize message for transmission."""
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'MeshMessage':
        """Deserialize message from transmission data."""
        return cls(
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=data["message_type"],
            content=data["content"],
            timestamp=data["timestamp"],
            message_id=data.get("message_id")
        )


class MeshConnection:
    """
    Connection between two agents in mesh topology.
    
    Manages message queuing, transmission, and latency calculation
    for a pair of agents in the mesh network.
    """
    
    def __init__(self, agent_a: str, agent_b: str, distance: float):
        """
        Initialize mesh connection between two agents.
        
        Args:
            agent_a: ID of first agent
            agent_b: ID of second agent  
            distance: Euclidean distance between agents
            
        Raises:
            ValueError: If parameters are invalid
        """
        if distance < 0:
            raise ValueError("Distance must be non-negative")
        
        if agent_a == agent_b:
            raise ValueError("Cannot create connection to self")
        
        # Store agents in canonical order for consistency
        if agent_a < agent_b:
            self.agent_a = agent_a
            self.agent_b = agent_b
        else:
            self.agent_a = agent_b
            self.agent_b = agent_a
        
        self.distance = distance
        self.message_queue: List[MeshMessage] = []
        self.message_count = 0
        self.total_latency = 0.0
        self.creation_time = time.perf_counter()
    
    def queue_message(self, message: MeshMessage) -> None:
        """
        Queue message for transmission.
        
        Args:
            message: Message to queue for delivery
        """
        self.message_queue.append(message)
        self.message_count += 1
    
    def process_messages(self, base_latency: float, distance_coefficient: float) -> List[MeshMessage]:
        """
        Process queued messages with latency calculation.
        
        Implements the latency model: L = α + β·d + ε
        where α is base latency, β is distance coefficient, d is distance.
        
        Args:
            base_latency: Base processing latency (α)
            distance_coefficient: Distance multiplier (β)
            
        Returns:
            List of messages ready for delivery
        """
        delivered_messages = []
        
        while self.message_queue:
            message = self.message_queue.pop(0)
            
            # Calculate latency: L = α + β·d
            message_latency = base_latency + distance_coefficient * self.distance
            self.total_latency += message_latency
            
            delivered_messages.append(message)
        
        return delivered_messages
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connection performance metrics.
        
        Returns:
            Dictionary containing connection metrics
        """
        average_latency = (
            self.total_latency / self.message_count 
            if self.message_count > 0 else 0.0
        )
        
        connection_age = time.perf_counter() - self.creation_time
        throughput = self.message_count / connection_age if connection_age > 0 else 0.0
        
        return {
            "message_count": self.message_count,
            "distance": self.distance,
            "total_latency": self.total_latency,
            "average_latency": average_latency,
            "throughput": throughput,
            "queue_size": len(self.message_queue)
        }


class MeshCommunication:
    """
    Mesh communication system for architecture comparison.
    
    Implements full mesh topology where each agent can communicate
    directly with every other agent. Provides O(N²) communication
    complexity for statistical comparison against O(N) spoke architecture.
    
    Mathematical Model:
    - Connection count: N(N-1)/2 undirected connections
    - Message complexity: O(N²) scaling
    - Latency distribution: distance-dependent with statistical variation
    - Memory overhead: O(N²) vs O(N) for spoke architecture
    
    This implementation supports Hypothesis H2 validation by providing
    measurable communication characteristics that demonstrate the 
    scalability advantage of spoke-based systems.
    """
    
    def __init__(self, max_agents: int, enable_metrics: bool = True):
        """
        Initialize mesh communication system.
        
        Args:
            max_agents: Maximum number of agents supported
            enable_metrics: Whether to collect performance metrics
            
        Raises:
            ValueError: If parameters are invalid
        """
        if max_agents <= 0:
            raise ValueError("max_agents must be positive")
        
        self.max_agents = max_agents
        self.enable_metrics = enable_metrics
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, MeshConnection] = {}
        self.message_count = 0
        self.total_latency = 0.0
        self.creation_time = time.perf_counter()
        
        # Latency model parameters
        self.base_latency = 0.001  # 1ms base latency
        self.distance_coefficient = 0.0001  # Distance factor
    
    def register_agent(self, agent: Agent) -> Optional[str]:
        """
        Register agent in mesh network.
        
        Creates connections to all previously registered agents,
        implementing the O(N²) connectivity characteristic of mesh topology.
        
        Args:
            agent: Agent to register in the mesh
            
        Returns:
            Connection ID for the registered agent, or None if failed
            
        Raises:
            ValueError: If agent capacity exceeded
        """
        if len(self.registered_agents) >= self.max_agents:
            raise ValueError("Maximum agent connections exceeded")
        
        if agent.agent_id in self.registered_agents:
            return self.registered_agents[agent.agent_id]["connection_id"]
        
        # Generate unique connection ID
        connection_id = f"mesh_{agent.agent_id}_{len(self.registered_agents)}"
        
        # Register agent
        self.registered_agents[agent.agent_id] = {
            "agent": agent,
            "connection_id": connection_id,
            "registration_time": time.perf_counter()
        }
        
        # Create connections to all other registered agents
        for other_agent_id, other_data in self.registered_agents.items():
            if other_agent_id != agent.agent_id:
                other_agent = other_data["agent"]
                
                # Calculate distance between agents
                distance = self._calculate_agent_distance(agent, other_agent)
                
                # Create bidirectional connection
                connection_key = self._get_connection_key(agent.agent_id, other_agent_id)
                self.connections[connection_key] = MeshConnection(
                    agent.agent_id, other_agent_id, distance
                )
        
        return connection_id
    
    def _calculate_agent_distance(self, agent1: Agent, agent2: Agent) -> float:
        """
        Calculate Euclidean distance between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            Euclidean distance between agent positions
        """
        if agent1.current_position is None or agent2.current_position is None:
            # If positions not available, use spawn time difference as proxy
            return abs(agent1.spawn_time - agent2.spawn_time) * 100
        
        pos1 = agent1.current_position
        pos2 = agent2.current_position
        
        # Calculate 3D Euclidean distance
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _get_connection_key(self, agent_a: str, agent_b: str) -> str:
        """
        Get canonical connection key for agent pair.
        
        Args:
            agent_a: ID of first agent
            agent_b: ID of second agent
            
        Returns:
            Canonical connection key (lexicographically ordered)
        """
        if agent_a < agent_b:
            return f"{agent_a}↔{agent_b}"
        else:
            return f"{agent_b}↔{agent_a}"
    
    def send_message(self, message: MeshMessage) -> bool:
        """
        Send message between agents in mesh topology.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued successfully, False otherwise
        """
        # Verify both agents are registered
        if (message.sender_id not in self.registered_agents or 
            message.recipient_id not in self.registered_agents):
            return False
        
        # Find connection for this agent pair
        connection_key = self._get_connection_key(message.sender_id, message.recipient_id)
        if connection_key not in self.connections:
            return False
        
        # Queue message in connection
        connection = self.connections[connection_key]
        connection.queue_message(message)
        
        return True
    
    def process_all_messages(self) -> int:
        """
        Process all queued messages across all connections.
        
        Returns:
            Number of messages processed
        """
        total_processed = 0
        
        for connection in self.connections.values():
            delivered_messages = connection.process_messages(
                self.base_latency, self.distance_coefficient
            )
            
            total_processed += len(delivered_messages)
            
            if self.enable_metrics:
                for message in delivered_messages:
                    self.message_count += 1
        
        return total_processed
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive performance metrics.
        
        Returns:
            Dictionary containing performance metrics for analysis
        """
        connection_count = len(self.connections)
        agent_count = len(self.registered_agents)
        
        # Collect latency statistics
        latencies = []
        for connection in self.connections.values():
            if connection.message_count > 0:
                avg_latency = connection.total_latency / connection.message_count
                latencies.append(avg_latency)
        
        average_latency = statistics.mean(latencies) if latencies else 0.0
        
        # Calculate throughput
        system_age = time.perf_counter() - self.creation_time
        throughput = self.message_count / system_age if system_age > 0 else 0.0
        
        # Message density (messages per connection)
        message_density = (
            self.message_count / connection_count 
            if connection_count > 0 else 0.0
        )
        
        return {
            "total_messages": self.message_count,
            "connection_count": connection_count,
            "agent_count": agent_count,
            "average_latency": average_latency,
            "throughput": throughput,
            "message_density": message_density,
            "system_age": system_age
        }
    
    def get_comparison_metrics(self) -> Dict[str, Any]:
        """
        Get metrics specifically for architecture comparison.
        
        Returns:
            Dictionary with metrics for mesh vs spoke comparison
        """
        agent_count = len(self.registered_agents)
        connection_count = len(self.connections)
        
        # Calculate memory overhead (connections + message queues)
        total_queue_size = sum(len(conn.message_queue) for conn in self.connections.values())
        
        # Collect distance statistics
        distances = [conn.distance for conn in self.connections.values()]
        max_distance = max(distances) if distances else 0.0
        
        # Calculate expected O(N²) theoretical connections
        theoretical_connections = agent_count * (agent_count - 1) // 2 if agent_count > 1 else 0
        
        return {
            "memory_overhead": connection_count + total_queue_size,
            "connection_memory": connection_count,
            "message_queue_size": total_queue_size,
            "max_distance": max_distance,
            "theoretical_connections": theoretical_connections,
            "throughput": self.get_performance_metrics()["throughput"]
        }
    
    def get_hypothesis_h2_metrics(self) -> Dict[str, Any]:
        """
        Get metrics specifically for Hypothesis H2 validation.
        
        Collects statistical measures needed for comparing mesh vs spoke
        communication overhead and complexity.
        
        Returns:
            Dictionary with H2 validation metrics
        """
        agent_count = len(self.registered_agents)
        connection_count = len(self.connections)
        
        # Collect latency data for statistical tests
        latencies = []
        for connection in self.connections.values():
            if connection.message_count > 0:
                avg_latency = connection.total_latency / connection.message_count
                latencies.append(avg_latency)
        
        average_latency = statistics.mean(latencies) if latencies else 0.0
        latency_variance = statistics.variance(latencies) if len(latencies) > 1 else 0.0
        
        # Message complexity (actual connections vs theoretical O(N²))
        theoretical_max_connections = agent_count * (agent_count - 1) // 2
        message_complexity = connection_count
        
        # Communication distance bounds
        distances = [conn.distance for conn in self.connections.values()]
        max_distance = max(distances) if distances else 0.0
        
        # Connection overhead (O(N²) scaling)
        connection_overhead = connection_count
        
        # Throughput calculation
        system_age = time.perf_counter() - self.creation_time
        throughput_msgs_per_sec = self.message_count / system_age if system_age > 0 else 0.0
        
        return {
            "message_complexity": message_complexity,
            "average_latency": average_latency,
            "latency_variance": latency_variance,
            "connection_overhead": connection_overhead,
            "max_distance": max_distance,
            "throughput_msgs_per_sec": throughput_msgs_per_sec,
            "theoretical_max_connections": theoretical_max_connections,
            "scaling_factor": (
                connection_overhead / agent_count if agent_count > 0 else 0.0
            )
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"MeshCommunication(agents={len(self.registered_agents)}, "
                f"connections={len(self.connections)}, messages={self.message_count})")
