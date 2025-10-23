"""
Mesh Communication Baseline Implementation
O(N²) communication complexity for comparison with Felix's hub-spoke O(N) model
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
import time
import random
from collections import defaultdict

@dataclass
class MeshAgent:
    """Agent in a mesh topology where everyone talks to everyone"""
    id: str
    agent_type: str
    position: float
    connections: Set[str] = field(default_factory=set)
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    sent_messages: int = 0
    received_messages: int = 0
    processing_time: float = 0.0

class MeshCommunicationSystem:
    """Mesh topology where every agent can communicate with every other agent"""

    def __init__(self, num_agents: int = 10):
        self.agents: Dict[str, MeshAgent] = {}
        self.num_agents = num_agents
        self.total_messages = 0
        self.total_routing_time = 0.0
        self.message_history: List[Dict[str, Any]] = []
        self._initialize_agents()

    def _initialize_agents(self):
        """Create agents and establish full mesh connections"""
        # Create agents
        for i in range(self.num_agents):
            agent_type = self._get_agent_type(i / self.num_agents)
            agent = MeshAgent(
                id=f"mesh_{agent_type}_{i}",
                agent_type=agent_type,
                position=i / self.num_agents
            )
            self.agents[agent.id] = agent

        # Establish full mesh connections (O(N²) connections)
        agent_ids = list(self.agents.keys())
        for i, agent_id in enumerate(agent_ids):
            for j, other_id in enumerate(agent_ids):
                if i != j:
                    self.agents[agent_id].connections.add(other_id)

    def _get_agent_type(self, position: float) -> str:
        """Determine agent type based on position"""
        if position < 0.3:
            return "research"
        elif position < 0.7:
            return "analysis"
        else:
            return "synthesis"

    async def send_message(self, from_agent: str, to_agent: str, content: Dict[str, Any]):
        """Send message directly from one agent to another (mesh style)"""
        start_time = time.time()

        # Check if connection exists (it should in full mesh)
        if to_agent not in self.agents[from_agent].connections:
            # In mesh, agents need to establish connection first
            self.agents[from_agent].connections.add(to_agent)
            self.agents[to_agent].connections.add(from_agent)

        # Simulate routing delay (increases with network size)
        routing_delay = 0.001 * len(self.agents)  # O(N) routing lookup
        await asyncio.sleep(routing_delay)

        # Deliver message
        message = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "timestamp": time.time()
        }
        self.agents[to_agent].message_queue.append(message)

        # Update metrics
        self.agents[from_agent].sent_messages += 1
        self.agents[to_agent].received_messages += 1
        self.total_messages += 1
        self.total_routing_time += (time.time() - start_time)
        self.message_history.append(message)

    async def broadcast_message(self, from_agent: str, content: Dict[str, Any]):
        """Broadcast message to all connected agents (O(N) messages)"""
        tasks = []
        for to_agent in self.agents[from_agent].connections:
            tasks.append(self.send_message(from_agent, to_agent, content))
        await asyncio.gather(*tasks)

    async def process_communication_round(self):
        """Simulate one round of communication where agents exchange messages"""
        tasks = []

        # Each agent processes its queue and potentially sends messages
        for agent_id, agent in self.agents.items():
            # Process incoming messages
            while agent.message_queue:
                message = agent.message_queue.pop(0)
                processing_time = random.uniform(0.01, 0.05)
                agent.processing_time += processing_time
                await asyncio.sleep(processing_time)

                # Potentially respond or forward (50% chance)
                if random.random() < 0.5:
                    # Pick random agent to communicate with
                    target = random.choice(list(agent.connections))
                    response = {
                        "type": "response",
                        "original_from": message["from"],
                        "data": f"Response from {agent_id}"
                    }
                    tasks.append(self.send_message(agent_id, target, response))

        if tasks:
            await asyncio.gather(*tasks)

    async def run_workflow(self, rounds: int = 10) -> Dict[str, Any]:
        """Run mesh communication workflow for specified rounds"""
        start_time = time.time()
        initial_memory = self._get_memory_usage()

        for round_num in range(rounds):
            # Each agent broadcasts status (simulating collaboration)
            for agent_id in self.agents:
                content = {
                    "round": round_num,
                    "status": f"Agent {agent_id} status at round {round_num}",
                    "confidence": random.uniform(0.3, 0.9)
                }
                # Broadcast to 30% of network (not full broadcast to avoid explosion)
                targets = random.sample(
                    list(self.agents[agent_id].connections),
                    max(1, int(len(self.agents[agent_id].connections) * 0.3))
                )
                for target in targets:
                    await self.send_message(agent_id, target, content)

            # Process messages
            await self.process_communication_round()

        # Calculate metrics
        elapsed_time = time.time() - start_time
        final_memory = self._get_memory_usage()

        # Calculate message complexity
        avg_messages_per_agent = self.total_messages / len(self.agents)
        connection_count = sum(len(a.connections) for a in self.agents.values())

        return {
            "elapsed_time": elapsed_time,
            "total_messages": self.total_messages,
            "avg_messages_per_agent": avg_messages_per_agent,
            "total_routing_time": self.total_routing_time,
            "connection_count": connection_count,  # Should be O(N²)
            "memory_usage": final_memory - initial_memory,
            "avg_processing_time": sum(a.processing_time for a in self.agents.values()) / len(self.agents),
            "message_complexity": self._calculate_complexity(),
            "num_agents": len(self.agents)
        }

    def _calculate_complexity(self) -> str:
        """Calculate actual communication complexity"""
        n = len(self.agents)
        connections = sum(len(a.connections) for a in self.agents.values())

        # In full mesh, connections should be n * (n-1)
        expected_mesh = n * (n - 1)

        if connections >= expected_mesh * 0.9:  # Allow 10% variance
            return f"O(N²) - {connections} connections for {n} agents"
        else:
            return f"Partial mesh - {connections} connections for {n} agents"

    def _get_memory_usage(self) -> int:
        """Estimate memory usage of message queues and connections"""
        # Rough estimate: each message ~100 bytes, each connection ~50 bytes
        message_memory = len(self.message_history) * 100
        connection_memory = sum(len(a.connections) for a in self.agents.values()) * 50
        return message_memory + connection_memory

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of communication metrics"""
        return {
            "agents": len(self.agents),
            "total_connections": sum(len(a.connections) for a in self.agents.values()),
            "total_messages": self.total_messages,
            "avg_routing_time": self.total_routing_time / max(1, self.total_messages),
            "messages_per_agent": {
                agent_id: {
                    "sent": agent.sent_messages,
                    "received": agent.received_messages,
                    "connections": len(agent.connections)
                }
                for agent_id, agent in self.agents.items()
            }
        }