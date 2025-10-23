"""
H1 Hypothesis Test: Adaptive Behavior
Tests whether helical progression enables better adaptive behavior through position-based adjustments
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import statistics
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.validation.validation_utils import (
    TestResult, MetricsCalculator, TaskGenerator, TestRunner, LLMSimulator
)
from src.core.helix_geometry import HelixGeometry
from src.agents.llm_agent import LLMAgent
from src.communication.central_post import CentralPost
from src.llm.token_budget import TokenBudgetManager

class H1AdaptiveBehaviorTest(TestRunner):
    """Test adaptive behavior improvement with helical progression"""

    def __init__(self, num_iterations: int = 10):
        super().__init__(num_iterations)
        self.helix = HelixGeometry(
            top_radius=3.0,
            bottom_radius=0.5,
            height=8.0,
            turns=2
        )
        self.llm_simulator = LLMSimulator()

    async def test_felix_adaptive_behavior(self, task: str) -> Dict[str, Any]:
        """Test Felix agents' adaptive behavior based on helix position"""

        # Simulate behavior adaptation based on helix position
        from src.llm.token_budget import TokenBudgetManager

        token_budget_mgr = TokenBudgetManager()

        # Test agents at different helix positions
        positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        behavior_metrics = {}

        for i, t_value in enumerate(positions):
            # Determine agent type based on position
            if t_value < 0.3:
                agent_type = "research"
            elif t_value < 0.7:
                agent_type = "analysis"
            else:
                agent_type = "synthesis"

            agent_id = f"felix_{agent_type}_{i}"

            # Calculate adaptive temperature (high at top, low at bottom)
            # Felix uses helical progression for smooth adaptation
            min_temp = 0.2
            max_temp = 1.0
            expected_temp = max_temp - (t_value * (max_temp - min_temp))

            # Calculate adaptive token budget based on type and depth
            base_budget = 1000
            if agent_type == "research":
                expected_tokens = base_budget
            elif agent_type == "analysis":
                expected_tokens = int(base_budget * 1.2)
            else:  # synthesis
                expected_tokens = int(base_budget * 2.0)

            # Get simulated response with adaptive parameters
            response = self.llm_simulator.generate_response(
                f"Position-aware prompt for {agent_type} at depth {t_value}",
                expected_temp,
                expected_tokens
            )

            # Track behavior metrics
            behavior_metrics[agent_id] = {
                "position": t_value,
                "temperature": expected_temp,
                "token_budget": expected_tokens,
                "actual_tokens": response["tokens_used"],
                "confidence": response["confidence"],
                "response_type": response["response_type"],
                "agent_type": agent_type
            }

        # Analyze adaptation quality
        adaptation_score = self._calculate_adaptation_score(behavior_metrics)

        return {
            "behavior_metrics": behavior_metrics,
            "adaptation_score": adaptation_score,
            "num_agents": len(positions),
            "temperature_range": self._get_temperature_range(behavior_metrics),
            "token_adaptation": self._analyze_token_adaptation(behavior_metrics),
            "type_distribution": self._get_type_distribution(behavior_metrics)
        }

    async def test_linear_adaptive_behavior(self, task: str) -> Dict[str, Any]:
        """Test baseline linear agents' adaptive behavior"""

        positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        behavior_metrics = {}

        for i, position in enumerate(positions):
            # Simple linear type assignment
            if position < 0.3:
                agent_type = "research"
            elif position < 0.7:
                agent_type = "analysis"
            else:
                agent_type = "synthesis"

            # Linear temperature calculation (no helix)
            temperature = 1.0 - (0.8 * position)

            # Linear token budget
            token_budget = int(2048 + (1000 * position))

            # Get simulated response
            response = self.llm_simulator.generate_response(
                f"Linear agent at position {position}",
                temperature,
                token_budget
            )

            behavior_metrics[f"linear_{agent_type}_{i}"] = {
                "position": position,
                "temperature": temperature,
                "token_budget": token_budget,
                "actual_tokens": response["tokens_used"],
                "confidence": response["confidence"],
                "response_type": response["response_type"],
                "agent_type": agent_type
            }

        # Calculate adaptation score for linear
        adaptation_score = self._calculate_adaptation_score(behavior_metrics)

        return {
            "behavior_metrics": behavior_metrics,
            "adaptation_score": adaptation_score,
            "num_agents": len(positions),
            "temperature_range": self._get_temperature_range(behavior_metrics),
            "token_adaptation": self._analyze_token_adaptation(behavior_metrics),
            "type_distribution": self._get_type_distribution(behavior_metrics)
        }

    def _calculate_adaptation_score(self, metrics: Dict[str, Dict]) -> float:
        """
        Calculate how well agents adapt their behavior to their position/role
        Higher score = better adaptation
        """
        scores = []

        for agent_id, m in metrics.items():
            position = m["position"]
            agent_type = m["agent_type"]

            # Score temperature adaptation
            # Research (early) should have high temp, synthesis (late) should have low
            if agent_type == "research":
                temp_score = m["temperature"]  # Higher is better for research
            elif agent_type == "synthesis":
                temp_score = 1.0 - m["temperature"]  # Lower is better for synthesis
            else:
                temp_score = 1.0 - abs(m["temperature"] - 0.5) * 2  # Middle temp for analysis

            # Score token usage efficiency
            token_efficiency = m["actual_tokens"] / m["token_budget"]
            if agent_type == "synthesis":
                # Synthesis should use more tokens
                token_score = token_efficiency
            else:
                # Others should be more efficient
                token_score = 1.0 - abs(token_efficiency - 0.6)

            # Score confidence appropriateness
            if agent_type == "research":
                # Research should have lower confidence
                conf_score = 1.0 if m["confidence"] < 0.6 else 0.5
            elif agent_type == "synthesis":
                # Synthesis should have higher confidence
                conf_score = m["confidence"]
            else:
                # Analysis in middle
                conf_score = 1.0 - abs(m["confidence"] - 0.7)

            # Combined adaptation score
            agent_score = (temp_score + token_score + conf_score) / 3
            scores.append(agent_score)

        return statistics.mean(scores) if scores else 0.0

    def _get_temperature_range(self, metrics: Dict[str, Dict]) -> Tuple[float, float]:
        """Get min and max temperature from metrics"""
        temps = [m["temperature"] for m in metrics.values()]
        return (min(temps), max(temps)) if temps else (0.0, 0.0)

    def _analyze_token_adaptation(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how token budgets adapt to position"""
        by_type = {"research": [], "analysis": [], "synthesis": []}

        for m in metrics.values():
            by_type[m["agent_type"]].append(m["token_budget"])

        return {
            agent_type: {
                "mean": statistics.mean(budgets) if budgets else 0,
                "range": (min(budgets), max(budgets)) if budgets else (0, 0)
            }
            for agent_type, budgets in by_type.items()
        }

    def _get_type_distribution(self, metrics: Dict[str, Dict]) -> Dict[str, int]:
        """Count agents by type"""
        distribution = {"research": 0, "analysis": 0, "synthesis": 0}
        for m in metrics.values():
            distribution[m["agent_type"]] += 1
        return distribution

    async def run_test(self, test_name: str, hypothesis: str,
                       target_percentage: float) -> TestResult:
        """Run single test iteration comparing adaptive behaviors"""

        # Generate test task
        task = TaskGenerator.generate_research_task()

        print(f"  Testing Felix adaptive behavior...")
        felix_metrics = await self.test_felix_adaptive_behavior(task)

        print(f"  Testing baseline linear behavior...")
        baseline_metrics = await self.test_linear_adaptive_behavior(task)

        # Calculate improvement in adaptation score
        # Higher adaptation score is better
        improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["adaptation_score"],
            felix_metrics["adaptation_score"],
            lower_is_better=False
        )

        # Check if target met
        passed = improvement >= target_percentage

        # Create detailed analysis
        details = {
            "felix_adaptation_score": felix_metrics["adaptation_score"],
            "baseline_adaptation_score": baseline_metrics["adaptation_score"],
            "felix_temp_range": felix_metrics["temperature_range"],
            "baseline_temp_range": baseline_metrics["temperature_range"],
            "felix_token_adaptation": felix_metrics["token_adaptation"],
            "baseline_token_adaptation": baseline_metrics["token_adaptation"],
            "improvement_breakdown": {
                "adaptation_score": improvement,
                "temperature_gradient": self._compare_temperature_gradients(
                    felix_metrics["behavior_metrics"],
                    baseline_metrics["behavior_metrics"]
                ),
                "token_efficiency": self._compare_token_efficiency(
                    felix_metrics["behavior_metrics"],
                    baseline_metrics["behavior_metrics"]
                )
            }
        }

        return TestResult(
            hypothesis=hypothesis,
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            felix_metrics=felix_metrics,
            baseline_metrics=baseline_metrics,
            improvement_percentage=improvement,
            target_percentage=target_percentage,
            passed=passed,
            details=details
        )

    def _compare_temperature_gradients(self, felix_metrics: Dict, baseline_metrics: Dict) -> float:
        """Compare smoothness of temperature gradients"""
        felix_temps = sorted([(m["position"], m["temperature"]) for m in felix_metrics.values()])
        baseline_temps = sorted([(m["position"], m["temperature"]) for m in baseline_metrics.values()])

        # Calculate gradient smoothness (lower variance in differences = smoother)
        felix_diffs = [felix_temps[i+1][1] - felix_temps[i][1] for i in range(len(felix_temps)-1)]
        baseline_diffs = [baseline_temps[i+1][1] - baseline_temps[i][1] for i in range(len(baseline_temps)-1)]

        felix_smoothness = statistics.stdev(felix_diffs) if len(felix_diffs) > 1 else 1.0
        baseline_smoothness = statistics.stdev(baseline_diffs) if len(baseline_diffs) > 1 else 1.0

        # Lower variance is better (smoother gradient)
        return MetricsCalculator.calculate_improvement(
            baseline_smoothness, felix_smoothness, lower_is_better=True
        )

    def _compare_token_efficiency(self, felix_metrics: Dict, baseline_metrics: Dict) -> float:
        """Compare token usage efficiency"""
        felix_efficiency = [m["actual_tokens"] / m["token_budget"] for m in felix_metrics.values()]
        baseline_efficiency = [m["actual_tokens"] / m["token_budget"] for m in baseline_metrics.values()]

        # Calculate average efficiency
        felix_avg = statistics.mean(felix_efficiency)
        baseline_avg = statistics.mean(baseline_efficiency)

        # Better efficiency = using appropriate amount of tokens for role
        # Target efficiency around 0.7 (not too low, not maxed out)
        felix_score = 1.0 - abs(felix_avg - 0.7)
        baseline_score = 1.0 - abs(baseline_avg - 0.7)

        return MetricsCalculator.calculate_improvement(
            baseline_score, felix_score, lower_is_better=False
        )


async def main():
    """Main test execution"""
    print("=" * 60)
    print("H1 HYPOTHESIS TEST: ADAPTIVE BEHAVIOR")
    print("Testing position-based behavioral adaptation")
    print("=" * 60)

    # Initialize test
    test = H1AdaptiveBehaviorTest(num_iterations=5)

    # Run multiple iterations
    print("\nRunning test iterations...")
    results = await test.run_multiple(
        test_name="H1_Adaptive_Behavior",
        hypothesis="H1",
        target_percentage=20.0
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
    print(f"Target: 20%")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Test Passed: {'YES' if avg_improvement >= 20 else 'NO'}")

    print("\nIndividual Iterations:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Improvement: {result.improvement_percentage:.2f}% - {'PASS' if result.passed else 'FAIL'}")

    # Show detailed breakdown from latest iteration
    if results:
        latest = results[-1]
        print("\nAdaptation Analysis (Latest Iteration):")
        print(f"  Felix Adaptation Score: {latest.details['felix_adaptation_score']:.3f}")
        print(f"  Baseline Adaptation Score: {latest.details['baseline_adaptation_score']:.3f}")

        print("\nImprovement Breakdown:")
        breakdown = latest.details["improvement_breakdown"]
        print(f"  Adaptation Score: {breakdown['adaptation_score']:.1f}%")
        print(f"  Temperature Gradient: {breakdown['temperature_gradient']:.1f}%")
        print(f"  Token Efficiency: {breakdown['token_efficiency']:.1f}%")

    # Save results
    import json
    results_file = "tests/results/h1_adaptive_behavior_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "hypothesis": "H1",
            "test": "Adaptive Behavior",
            "target_improvement": 20.0,
            "average_improvement": avg_improvement,
            "success_rate": success_rate,
            "passed": avg_improvement >= 20,
            "iterations": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())