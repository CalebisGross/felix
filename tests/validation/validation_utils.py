"""
Validation Utilities for Hypothesis Testing
Shared metrics, helpers, and statistical functions for H1-H2-H3 validation
"""

import time
import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

@dataclass
class TestResult:
    """Container for individual test results"""
    hypothesis: str  # H1, H2, or H3
    test_name: str
    timestamp: str
    felix_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    improvement_percentage: float
    target_percentage: float
    passed: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

@dataclass
class ValidationReport:
    """Complete validation report for all hypotheses"""
    h1_results: List[TestResult]
    h2_results: List[TestResult]
    h3_results: List[TestResult]
    summary: Dict[str, Any]
    timestamp: str

    def save(self, filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "h1_results": [r.to_dict() for r in self.h1_results],
            "h2_results": [r.to_dict() for r in self.h2_results],
            "h3_results": [r.to_dict() for r in self.h3_results],
            "summary": self.summary,
            "timestamp": self.timestamp
        }


class MetricsCalculator:
    """Calculate various metrics for hypothesis validation"""

    @staticmethod
    def calculate_workload_variance(workload_distribution: Dict[str, float]) -> float:
        """Calculate variance in workload distribution across agents"""
        if not workload_distribution:
            return 0.0

        workloads = list(workload_distribution.values())
        if len(workloads) <= 1:
            return 0.0

        return statistics.variance(workloads)

    @staticmethod
    def calculate_improvement(baseline_value: float, felix_value: float,
                            lower_is_better: bool = True) -> float:
        """
        Calculate percentage improvement from baseline to Felix
        Args:
            baseline_value: Baseline metric value
            felix_value: Felix metric value
            lower_is_better: True if lower values are better (e.g., latency, variance)
        Returns:
            Improvement percentage (positive means Felix is better)
        """
        if baseline_value == 0:
            return 0.0

        if lower_is_better:
            improvement = (baseline_value - felix_value) / baseline_value * 100
        else:
            improvement = (felix_value - baseline_value) / baseline_value * 100

        return improvement

    @staticmethod
    def calculate_token_efficiency(tokens_used: int, output_quality: float,
                                  token_budget: int) -> float:
        """Calculate token usage efficiency"""
        if token_budget == 0:
            return 0.0

        usage_ratio = tokens_used / token_budget
        # Efficiency is high when we use fewer tokens but maintain quality
        efficiency = output_quality / max(0.1, usage_ratio)
        return min(1.0, efficiency)

    @staticmethod
    def calculate_communication_complexity(num_agents: int,
                                          num_messages: int,
                                          num_connections: int) -> Tuple[str, float]:
        """
        Determine communication complexity (O(N) vs O(N²))
        Returns: (complexity_type, score)
        """
        if num_agents == 0:
            return "O(1)", 1.0

        # For hub-spoke: messages should be ~2N (to hub and from hub)
        # For mesh: messages should be ~N²
        expected_hub_spoke = 2 * num_agents
        expected_mesh = num_agents * (num_agents - 1)

        # Calculate which pattern matches better
        hub_spoke_deviation = abs(num_messages - expected_hub_spoke) / expected_hub_spoke
        mesh_deviation = abs(num_messages - expected_mesh) / expected_mesh

        if hub_spoke_deviation < mesh_deviation:
            return "O(N)", 1.0 - hub_spoke_deviation
        else:
            return "O(N²)", 1.0 - mesh_deviation

    @staticmethod
    def calculate_attention_focus(relevant_tokens: int, total_tokens: int) -> float:
        """Calculate attention focus metric (ratio of relevant to total tokens)"""
        if total_tokens == 0:
            return 0.0
        return relevant_tokens / total_tokens

    @staticmethod
    def calculate_convergence_rate(confidence_history: List[float]) -> float:
        """Calculate how quickly confidence converges to threshold"""
        if len(confidence_history) < 2:
            return 0.0

        # Calculate average rate of confidence increase
        deltas = [confidence_history[i] - confidence_history[i-1]
                  for i in range(1, len(confidence_history))]

        return statistics.mean(deltas) if deltas else 0.0


class TaskGenerator:
    """Generate standardized tasks for testing"""

    @staticmethod
    def generate_research_task() -> str:
        """Generate a research synthesis task"""
        return """
        Analyze the impact of quantum computing on cryptography.
        Consider current encryption methods, quantum threats, and post-quantum solutions.
        Synthesize findings into actionable recommendations.
        """

    @staticmethod
    def generate_analysis_task() -> str:
        """Generate a document analysis task"""
        return """
        Review the provided technical documentation and identify:
        1. Key architectural decisions
        2. Potential scalability issues
        3. Security vulnerabilities
        4. Performance optimization opportunities
        Provide a comprehensive analysis with prioritized recommendations.
        """

    @staticmethod
    def generate_creative_task() -> str:
        """Generate a creative generation task"""
        return """
        Design a sustainable city of the future that addresses:
        - Climate change adaptation
        - Resource efficiency
        - Social equity
        - Economic viability
        Create a detailed concept with innovative solutions.
        """

    @staticmethod
    def generate_complex_task() -> str:
        """Generate a complex multi-faceted task"""
        return """
        Develop a comprehensive strategy for a global technology company to:
        1. Transition to renewable energy by 2030
        2. Achieve carbon neutrality by 2035
        3. Maintain competitive advantage
        4. Ensure stakeholder buy-in
        Include financial projections, risk analysis, and implementation roadmap.
        """


class TestRunner:
    """Base class for running hypothesis validation tests"""

    def __init__(self, num_iterations: int = 10):
        self.num_iterations = num_iterations
        self.results: List[TestResult] = []

    async def run_test(self, test_name: str, hypothesis: str,
                      target_percentage: float) -> TestResult:
        """Override in subclasses to implement specific test logic"""
        raise NotImplementedError

    async def run_multiple(self, test_name: str, hypothesis: str,
                          target_percentage: float) -> List[TestResult]:
        """Run test multiple times for statistical significance"""
        results = []
        for i in range(self.num_iterations):
            print(f"Running iteration {i+1}/{self.num_iterations} for {test_name}")
            result = await self.run_test(test_name, hypothesis, target_percentage)
            results.append(result)
            # Add small delay between iterations
            await asyncio.sleep(0.5)

        return results

    def calculate_average_improvement(self, results: List[TestResult]) -> float:
        """Calculate average improvement across multiple test runs"""
        improvements = [r.improvement_percentage for r in results]
        return statistics.mean(improvements) if improvements else 0.0

    def calculate_success_rate(self, results: List[TestResult]) -> float:
        """Calculate percentage of tests that passed"""
        if not results:
            return 0.0
        passed = sum(1 for r in results if r.passed)
        return (passed / len(results)) * 100


class LLMSimulator:
    """Simulate LLM responses for testing without real LLM calls"""

    @staticmethod
    def generate_response(prompt: str, temperature: float,
                         max_tokens: int) -> Dict[str, Any]:
        """Generate simulated LLM response"""
        # Simulate response based on temperature
        creativity_score = temperature  # Higher temp = more creative

        # Generate mock content
        if temperature > 0.7:
            response_type = "creative"
            content = f"Creative exploration with temperature {temperature:.2f}"
        elif temperature > 0.4:
            response_type = "balanced"
            content = f"Balanced analysis with temperature {temperature:.2f}"
        else:
            response_type = "focused"
            content = f"Focused synthesis with temperature {temperature:.2f}"

        # Simulate token usage (higher temp uses more tokens generally)
        tokens_used = int(max_tokens * (0.5 + temperature * 0.3))

        return {
            "content": content,
            "tokens_used": tokens_used,
            "response_type": response_type,
            "temperature": temperature,
            "confidence": 0.9 - (temperature * 0.3),  # Lower temp = higher confidence
            "processing_time": 0.1 + (tokens_used / 10000)  # Simulate processing time
        }


def compare_metrics(felix_metrics: Dict[str, Any],
                   baseline_metrics: Dict[str, Any],
                   comparison_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """Compare metrics between Felix and baseline"""
    comparison = {}

    for key in comparison_keys:
        if key in felix_metrics and key in baseline_metrics:
            felix_val = felix_metrics[key]
            baseline_val = baseline_metrics[key]

            # Determine if lower is better based on metric name
            lower_is_better = any(x in key.lower() for x in
                                 ['variance', 'time', 'latency', 'cost'])

            improvement = MetricsCalculator.calculate_improvement(
                baseline_val, felix_val, lower_is_better
            )

            comparison[key] = {
                "felix": felix_val,
                "baseline": baseline_val,
                "improvement": improvement,
                "lower_is_better": lower_is_better
            }

    return comparison


def generate_test_report(results: List[TestResult],
                        hypothesis: str,
                        target: float) -> Dict[str, Any]:
    """Generate summary report for hypothesis test results"""
    if not results:
        return {"error": "No results to report"}

    avg_improvement = statistics.mean(r.improvement_percentage for r in results)
    success_rate = sum(1 for r in results if r.passed) / len(results) * 100

    # Calculate confidence interval (95%)
    improvements = [r.improvement_percentage for r in results]
    std_dev = statistics.stdev(improvements) if len(improvements) > 1 else 0
    confidence_interval = 1.96 * (std_dev / (len(improvements) ** 0.5))

    return {
        "hypothesis": hypothesis,
        "target_improvement": target,
        "average_improvement": avg_improvement,
        "confidence_interval": confidence_interval,
        "success_rate": success_rate,
        "num_tests": len(results),
        "passed": avg_improvement >= target,
        "details": {
            "min_improvement": min(improvements),
            "max_improvement": max(improvements),
            "std_deviation": std_dev,
            "individual_results": [r.improvement_percentage for r in results]
        }
    }


# Import asyncio for async tests
import asyncio