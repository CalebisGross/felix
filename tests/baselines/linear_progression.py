"""
Linear Progression Baseline Implementation
Non-helical agent progression for comparison with Felix's helical model
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import random
from collections import defaultdict

@dataclass
class LinearAgent:
    """Simple agent with linear progression (no helix)"""
    id: str
    agent_type: str
    position: float  # Linear position from 0.0 to 1.0
    confidence: float = 0.0
    token_budget: int = 2048
    temperature: float = 0.7
    messages: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def progress(self, delta: float = 0.1):
        """Move agent forward linearly"""
        self.position = min(1.0, self.position + delta)
        # Simple linear temperature adjustment (no helix math)
        self.temperature = 1.0 - (0.8 * self.position)
        # Linear token budget increase
        self.token_budget = int(2048 + (1000 * self.position))

    def calculate_confidence(self) -> float:
        """Simple confidence calculation without helical influence"""
        base_confidence = 0.3 + (0.5 * self.position)  # Linear increase

        # Type-based caps (same as Felix for fair comparison)
        if self.agent_type == "research":
            max_conf = 0.6
        elif self.agent_type == "analysis":
            max_conf = 0.8
        else:  # synthesis
            max_conf = 0.95

        return min(base_confidence, max_conf)


class LinearWorkflow:
    """Linear multi-agent workflow without helical coordination"""

    def __init__(self, num_agents: int = 5):
        self.agents: List[LinearAgent] = []
        self.messages: List[Dict[str, Any]] = []
        self.num_agents = num_agents
        self.spawn_agents()

    def spawn_agents(self):
        """Create agents with linear distribution"""
        for i in range(self.num_agents):
            agent_type = self._get_agent_type_by_position(i / self.num_agents)
            agent = LinearAgent(
                id=f"linear_{agent_type}_{i}",
                agent_type=agent_type,
                position=i / self.num_agents  # Evenly distributed
            )
            self.agents.append(agent)

    def _get_agent_type_by_position(self, position: float) -> str:
        """Assign agent type based on linear position"""
        if position < 0.3:
            return "research"
        elif position < 0.7:
            return "analysis"
        else:
            return "synthesis"

    async def process_task(self, task: str, steps: int = 20) -> Dict[str, Any]:
        """Process task with linear progression"""
        start_time = time.time()
        token_usage = defaultdict(int)
        workload_distribution = defaultdict(float)
        message_count = 0

        for step in range(steps):
            # Progress all agents linearly
            for agent in self.agents:
                agent.progress(1.0 / steps)

                # Simulate processing (would call LLM in real implementation)
                if random.random() < 0.7:  # 70% chance to process
                    processing_time = random.uniform(0.1, 0.5)
                    await asyncio.sleep(processing_time * 0.01)  # Simulate work

                    # Track metrics
                    tokens_used = random.randint(100, agent.token_budget)
                    token_usage[agent.id] += tokens_used
                    workload_distribution[agent.id] += processing_time

                    # Update confidence
                    agent.confidence = agent.calculate_confidence()

                    # Create message
                    message = {
                        "agent_id": agent.id,
                        "step": step,
                        "confidence": agent.confidence,
                        "tokens": tokens_used,
                        "position": agent.position
                    }
                    self.messages.append(message)
                    message_count += 1

                    # Check exit condition
                    if agent.agent_type == "synthesis" and agent.confidence >= 0.8:
                        break

        # Calculate metrics
        elapsed_time = time.time() - start_time

        # Workload variance (measure of distribution evenness)
        workloads = list(workload_distribution.values())
        avg_workload = sum(workloads) / len(workloads) if workloads else 0
        variance = sum((w - avg_workload) ** 2 for w in workloads) / len(workloads) if workloads else 0

        return {
            "elapsed_time": elapsed_time,
            "total_tokens": sum(token_usage.values()),
            "message_count": message_count,
            "workload_variance": variance,
            "workload_distribution": dict(workload_distribution),
            "token_usage": dict(token_usage),
            "final_confidence": max(a.confidence for a in self.agents),
            "num_agents": len(self.agents),
            "convergence_step": step
        }

    def get_agent_positions(self) -> Dict[str, float]:
        """Get current positions of all agents"""
        return {agent.id: agent.position for agent in self.agents}

    def get_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed metrics for each agent"""
        metrics = {}
        for agent in self.agents:
            metrics[agent.id] = {
                "type": agent.agent_type,
                "position": agent.position,
                "confidence": agent.confidence,
                "temperature": agent.temperature,
                "token_budget": agent.token_budget,
                "message_count": len([m for m in self.messages if m["agent_id"] == agent.id])
            }
        return metrics