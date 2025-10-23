"""
H2 Hypothesis Test: Communication Efficiency
Tests whether hub-spoke communication optimizes resource allocation with 15% efficiency gain
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import statistics
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.validation.validation_utils import (
    TestResult, MetricsCalculator, TaskGenerator, TestRunner
)
from tests.baselines.mesh_communication import MeshCommunicationSystem
from src.communication.central_post import CentralPost, AgentFactory
from src.communication.spoke import SpokeManager
from src.agents.agent import Agent

class H2CommunicationEfficiencyTest(TestRunner):
    """Test communication efficiency improvement with hub-spoke topology"""

    def __init__(self, num_iterations: int = 10):
        super().__init__(num_iterations)

    async def test_felix_hub_spoke(self, num_agents: int = 20) -> Dict[str, Any]:
        """Test Felix's hub-spoke communication pattern"""

        # Initialize CentralPost (hub)
        central_post = CentralPost()
        factory = AgentFactory(central_post)

        # Create agents and register with hub
        agents = []
        message_count = 0
        routing_time = 0.0
        start_time = time.time()

        # Spawn agents through factory (ensures registration)
        for i in range(num_agents):
            position = i / num_agents
            if position < 0.3:
                agent_type = "research"
            elif position < 0.7:
                agent_type = "analysis"
            else:
                agent_type = "synthesis"

            agent = factory.create_agent(
                agent_type=agent_type,
                t_value=position,
                spawn_time=position
            )
            agents.append(agent)

        # Simulate communication rounds
        for round_num in range(10):
            # Each agent sends a message to hub
            for agent in agents:
                msg_start = time.time()

                # Message to hub (O(1) operation)
                message = {
                    "agent_id": agent.agent_id,
                    "round": round_num,
                    "content": f"Status update from {agent.agent_id}",
                    "confidence": 0.5 + (round_num * 0.05)
                }

                # Hub processes and potentially broadcasts
                central_post.broadcast(message, sender_id=agent.agent_id)
                message_count += 1

                # Hub routes to relevant agents (not all)
                # Based on phase and relevance
                relevant_agents = central_post.get_agents_by_phase(agent.agent_type)
                message_count += len(relevant_agents)  # Messages from hub to agents

                routing_time += (time.time() - msg_start)

            # Small delay between rounds
            await asyncio.sleep(0.01)

        elapsed_time = time.time() - start_time

        # Calculate metrics
        connections = num_agents * 2  # Each agent connects to hub and hub to agent
        avg_routing_time = routing_time / max(1, message_count)

        # Memory usage estimation
        memory_estimate = message_count * 100  # bytes per message
        memory_estimate += connections * 50  # bytes per connection

        return {
            "elapsed_time": elapsed_time,
            "total_messages": message_count,
            "avg_messages_per_agent": message_count / num_agents,
            "total_routing_time": routing_time,
            "avg_routing_time": avg_routing_time,
            "connection_count": connections,
            "memory_usage": memory_estimate,
            "num_agents": num_agents,
            "complexity": f"O(N) - {connections} connections for {num_agents} agents",
            "messages_per_round": message_count / 10
        }

    async def test_mesh_communication(self, num_agents: int = 20) -> Dict[str, Any]:
        """Test baseline mesh communication pattern"""

        mesh_system = MeshCommunicationSystem(num_agents=num_agents)
        metrics = await mesh_system.run_workflow(rounds=10)

        return metrics

    async def run_test(self, test_name: str, hypothesis: str,
                       target_percentage: float) -> TestResult:
        """Run single test iteration comparing hub-spoke to mesh"""

        # Test with different agent counts to verify scalability
        agent_counts = [10, 20, 30]
        felix_results = []
        mesh_results = []

        for count in agent_counts:
            print(f"  Testing with {count} agents...")

            # Test Felix hub-spoke
            felix_metrics = await self.test_felix_hub_spoke(count)
            felix_results.append(felix_metrics)

            # Test mesh baseline
            mesh_metrics = await self.test_mesh_communication(count)
            mesh_results.append(mesh_metrics)

        # Aggregate results
        felix_aggregated = self._aggregate_metrics(felix_results)
        mesh_aggregated = self._aggregate_metrics(mesh_results)

        # Calculate improvements
        message_improvement = MetricsCalculator.calculate_improvement(
            mesh_aggregated["avg_messages"],
            felix_aggregated["avg_messages"],
            lower_is_better=True
        )

        routing_improvement = MetricsCalculator.calculate_improvement(
            mesh_aggregated["avg_routing_time"],
            felix_aggregated["avg_routing_time"],
            lower_is_better=True
        )

        memory_improvement = MetricsCalculator.calculate_improvement(
            mesh_aggregated["avg_memory"],
            felix_aggregated["avg_memory"],
            lower_is_better=True
        )

        # Overall efficiency improvement (average of all metrics)
        overall_improvement = (message_improvement + routing_improvement + memory_improvement) / 3

        # Check if target met
        passed = overall_improvement >= target_percentage

        # Scalability analysis
        scalability_analysis = self._analyze_scalability(felix_results, mesh_results)

        details = {
            "message_improvement": message_improvement,
            "routing_improvement": routing_improvement,
            "memory_improvement": memory_improvement,
            "overall_improvement": overall_improvement,
            "scalability": scalability_analysis,
            "agent_counts_tested": agent_counts
        }

        return TestResult(
            hypothesis=hypothesis,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            felix_metrics=felix_aggregated,
            baseline_metrics=mesh_aggregated,
            improvement_percentage=overall_improvement,
            target_percentage=target_percentage,
            passed=passed,
            details=details
        )

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple test runs"""
        return {
            "avg_messages": statistics.mean(r["total_messages"] for r in results),
            "avg_routing_time": statistics.mean(r["avg_routing_time"] for r in results),
            "avg_memory": statistics.mean(r["memory_usage"] for r in results),
            "connection_counts": [r["connection_count"] for r in results],
            "complexities": [r.get("complexity", "Unknown") for r in results]
        }

    def _analyze_scalability(self, felix_results: List[Dict], mesh_results: List[Dict]) -> Dict[str, Any]:
        """Analyze how systems scale with agent count"""

        felix_scaling = []
        mesh_scaling = []

        for i in range(1, len(felix_results)):
            # Calculate growth rate of messages
            felix_growth = felix_results[i]["total_messages"] / felix_results[i-1]["total_messages"]
            mesh_growth = mesh_results[i]["total_messages"] / mesh_results[i-1]["total_messages"]

            felix_scaling.append(felix_growth)
            mesh_scaling.append(mesh_growth)

        return {
            "felix_avg_growth": statistics.mean(felix_scaling) if felix_scaling else 1.0,
            "mesh_avg_growth": statistics.mean(mesh_scaling) if mesh_scaling else 1.0,
            "felix_is_linear": statistics.mean(felix_scaling) < 1.5 if felix_scaling else False,
            "mesh_is_quadratic": statistics.mean(mesh_scaling) > 2.0 if mesh_scaling else False
        }


async def main():
    """Main test execution"""
    print("=" * 60)
    print("H2 HYPOTHESIS TEST: COMMUNICATION EFFICIENCY")
    print("Target: 15% efficiency gain with hub-spoke vs mesh")
    print("=" * 60)

    # Initialize test
    test = H2CommunicationEfficiencyTest(num_iterations=5)

    # Run multiple iterations
    print("\nRunning test iterations...")
    results = await test.run_multiple(
        test_name="H2_Communication_Efficiency",
        hypothesis="H2",
        target_percentage=15.0
    )

    # Calculate statistics
    improvements = [r.improvement_percentage for r in results]
    avg_improvement = statistics.mean(improvements)
    success_rate = sum(1 for r in results if r.passed) / len(results) * 100

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"Target: 15%")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Test Passed: {'YES' if avg_improvement >= 15 else 'NO'}")

    print("\nIndividual Iterations:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Improvement: {result.improvement_percentage:.2f}% - {'PASS' if result.passed else 'FAIL'}")

    # Show detailed breakdown from latest iteration
    if results:
        latest = results[-1]
        print("\nEfficiency Breakdown (Latest Iteration):")
        details = latest.details
        print(f"  Message Reduction: {details['message_improvement']:.1f}%")
        print(f"  Routing Time Reduction: {details['routing_improvement']:.1f}%")
        print(f"  Memory Usage Reduction: {details['memory_improvement']:.1f}%")

        print("\nScalability Analysis:")
        scalability = details["scalability"]
        print(f"  Felix scales linearly: {scalability['felix_is_linear']}")
        print(f"  Mesh scales quadratically: {scalability['mesh_is_quadratic']}")
        print(f"  Felix growth rate: {scalability['felix_avg_growth']:.2f}x per size increase")
        print(f"  Mesh growth rate: {scalability['mesh_avg_growth']:.2f}x per size increase")

    # Save results
    import json
    results_file = "tests/results/h2_communication_efficiency_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "hypothesis": "H2",
            "test": "Communication Efficiency",
            "target_improvement": 15.0,
            "average_improvement": avg_improvement,
            "success_rate": success_rate,
            "passed": avg_improvement >= 15,
            "iterations": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())