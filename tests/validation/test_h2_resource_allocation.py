"""
H2 Hypothesis Test: Resource Allocation
Tests whether hub-spoke communication optimizes resource allocation efficiency
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import statistics
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.validation.validation_utils import (
    TestResult, MetricsCalculator, TaskGenerator, TestRunner, LLMSimulator
)
from src.llm.token_budget import TokenBudgetManager
from src.communication.central_post import CentralPost, AgentFactory

class H2ResourceAllocationTest(TestRunner):
    """Test resource allocation efficiency with centralized hub-spoke management"""

    def __init__(self, num_iterations: int = 10):
        super().__init__(num_iterations)
        self.llm_simulator = LLMSimulator()

    async def test_felix_resource_allocation(self, task: str, num_agents: int = 15) -> Dict[str, Any]:
        """Test Felix's centralized resource allocation through CentralPost"""

        # Initialize central components
        central_post = CentralPost()
        factory = AgentFactory(central_post)
        token_budget_mgr = TokenBudgetManager(
            total_budget=20000,  # Total tokens available
            base_budget=1000
        )

        # Track resource usage
        token_allocations = {}
        token_usage = {}
        waste = {}
        quality_scores = {}

        # Create agents with different roles
        agents = []
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

            # Centralized allocation through token budget manager
            allocated = token_budget_mgr.allocate_budget(
                agent_type,
                position,
                token_budget_mgr.remaining_budget
            )
            token_allocations[agent.agent_id] = allocated

            # Simulate agent processing with allocated budget
            response = self.llm_simulator.generate_response(
                f"Agent {agent.agent_id} processing task",
                temperature=0.7 - (0.5 * position),  # Position-based temp
                max_tokens=allocated
            )

            # Track usage
            used = response["tokens_used"]
            token_usage[agent.agent_id] = used
            waste[agent.agent_id] = max(0, allocated - used)
            quality_scores[agent.agent_id] = response["confidence"]

            # Update remaining budget
            token_budget_mgr.track_usage(agent.agent_id, used)

        # Calculate efficiency metrics
        total_allocated = sum(token_allocations.values())
        total_used = sum(token_usage.values())
        total_waste = sum(waste.values())

        efficiency = total_used / max(1, total_allocated)
        avg_quality = statistics.mean(quality_scores.values())

        # Allocation fairness (lower variance is better)
        allocation_variance = statistics.variance(token_allocations.values())

        return {
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_waste": total_waste,
            "efficiency": efficiency,
            "avg_quality": avg_quality,
            "allocation_variance": allocation_variance,
            "num_agents": num_agents,
            "by_agent_type": self._analyze_by_type(agents, token_usage, quality_scores),
            "budget_remaining": token_budget_mgr.remaining_budget
        }

    async def test_distributed_resource_allocation(self, task: str, num_agents: int = 15) -> Dict[str, Any]:
        """Test baseline distributed resource allocation (each agent manages own budget)"""

        total_budget = 20000
        per_agent_budget = total_budget // num_agents  # Equal distribution

        # Track resource usage
        token_allocations = {}
        token_usage = {}
        waste = {}
        quality_scores = {}

        for i in range(num_agents):
            position = i / num_agents
            agent_id = f"distributed_{i}"

            # Each agent gets equal budget (no central optimization)
            allocated = per_agent_budget
            token_allocations[agent_id] = allocated

            # Agents don't know their optimal role from position
            # They use a fixed temperature
            temperature = 0.7  # No position-based adaptation

            # Simulate processing
            response = self.llm_simulator.generate_response(
                f"Agent {agent_id} processing task",
                temperature=temperature,
                max_tokens=allocated
            )

            # Track usage
            used = response["tokens_used"]
            token_usage[agent_id] = used
            waste[agent_id] = max(0, allocated - used)
            quality_scores[agent_id] = response["confidence"]

        # Calculate metrics
        total_allocated = sum(token_allocations.values())
        total_used = sum(token_usage.values())
        total_waste = sum(waste.values())

        efficiency = total_used / max(1, total_allocated)
        avg_quality = statistics.mean(quality_scores.values())
        allocation_variance = statistics.variance(token_allocations.values())

        return {
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_waste": total_waste,
            "efficiency": efficiency,
            "avg_quality": avg_quality,
            "allocation_variance": allocation_variance,
            "num_agents": num_agents,
            "by_agent_type": {"all": {"avg_tokens": per_agent_budget, "avg_quality": avg_quality}},
            "budget_remaining": total_budget - total_used
        }

    def _analyze_by_type(self, agents: List, token_usage: Dict, quality_scores: Dict) -> Dict[str, Any]:
        """Analyze resource usage by agent type"""
        by_type = {"research": [], "analysis": [], "synthesis": []}

        for agent in agents:
            agent_type = agent.agent_type
            if agent.agent_id in token_usage:
                by_type[agent_type].append({
                    "tokens": token_usage[agent.agent_id],
                    "quality": quality_scores.get(agent.agent_id, 0)
                })

        result = {}
        for agent_type, data in by_type.items():
            if data:
                result[agent_type] = {
                    "avg_tokens": statistics.mean(d["tokens"] for d in data),
                    "avg_quality": statistics.mean(d["quality"] for d in data)
                }
            else:
                result[agent_type] = {"avg_tokens": 0, "avg_quality": 0}

        return result

    async def run_test(self, test_name: str, hypothesis: str,
                       target_percentage: float) -> TestResult:
        """Run single test iteration comparing resource allocation strategies"""

        task = TaskGenerator.generate_analysis_task()
        num_agents = 15

        print(f"  Testing Felix centralized allocation...")
        felix_metrics = await self.test_felix_resource_allocation(task, num_agents)

        print(f"  Testing distributed allocation baseline...")
        baseline_metrics = await self.test_distributed_resource_allocation(task, num_agents)

        # Calculate improvements
        efficiency_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["efficiency"],
            felix_metrics["efficiency"],
            lower_is_better=False
        )

        waste_reduction = MetricsCalculator.calculate_improvement(
            baseline_metrics["total_waste"],
            felix_metrics["total_waste"],
            lower_is_better=True
        )

        quality_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["avg_quality"],
            felix_metrics["avg_quality"],
            lower_is_better=False
        )

        # Fairness improvement (lower variance is better)
        fairness_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["allocation_variance"],
            felix_metrics["allocation_variance"],
            lower_is_better=True
        )

        # Overall resource allocation efficiency
        overall_improvement = (efficiency_improvement + waste_reduction + quality_improvement) / 3

        passed = overall_improvement >= target_percentage

        details = {
            "efficiency_improvement": efficiency_improvement,
            "waste_reduction": waste_reduction,
            "quality_improvement": quality_improvement,
            "fairness_improvement": fairness_improvement,
            "overall_improvement": overall_improvement,
            "felix_by_type": felix_metrics["by_agent_type"],
            "baseline_by_type": baseline_metrics["by_agent_type"]
        }

        return TestResult(
            hypothesis=hypothesis,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            felix_metrics=felix_metrics,
            baseline_metrics=baseline_metrics,
            improvement_percentage=overall_improvement,
            target_percentage=target_percentage,
            passed=passed,
            details=details
        )


async def main():
    """Main test execution"""
    print("=" * 60)
    print("H2 HYPOTHESIS TEST: RESOURCE ALLOCATION")
    print("Testing centralized vs distributed resource management")
    print("=" * 60)

    # Initialize test
    test = H2ResourceAllocationTest(num_iterations=5)

    # Run multiple iterations
    print("\nRunning test iterations...")
    results = await test.run_multiple(
        test_name="H2_Resource_Allocation",
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

    # Show detailed breakdown
    if results:
        latest = results[-1]
        print("\nResource Allocation Breakdown (Latest Iteration):")
        details = latest.details
        print(f"  Efficiency Improvement: {details['efficiency_improvement']:.1f}%")
        print(f"  Waste Reduction: {details['waste_reduction']:.1f}%")
        print(f"  Quality Improvement: {details['quality_improvement']:.1f}%")
        print(f"  Fairness Improvement: {details['fairness_improvement']:.1f}%")

        print("\nFelix Allocation by Type:")
        for agent_type, metrics in details["felix_by_type"].items():
            print(f"  {agent_type}: {metrics['avg_tokens']:.0f} tokens, {metrics['avg_quality']:.2f} quality")

    # Save results
    import json
    results_file = "tests/results/h2_resource_allocation_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "hypothesis": "H2",
            "test": "Resource Allocation",
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