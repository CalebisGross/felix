"""
H3 Hypothesis Test: Attention Focus
Tests whether memory compression improves attention focus by 25%
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import statistics
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.validation.validation_utils import (
    TestResult, MetricsCalculator, TaskGenerator, TestRunner
)
from src.memory.context_compression import ContextCompressor
from src.workflows.context_builder import CollaborativeContextBuilder

class H3AttentionFocusTest(TestRunner):
    """Test attention focus improvement with compressed context"""

    def __init__(self, num_iterations: int = 10):
        super().__init__(num_iterations)

    async def test_felix_attention_focus(self, task: str, agent_outputs: List[Dict]) -> Dict[str, Any]:
        """Test Felix with collaborative context compression for attention focus"""

        # Initialize components
        compressor = ContextCompressor(compression_ratio=0.3)
        context_builder = CollaborativeContextBuilder()

        # Simulate building collaborative context
        for output in agent_outputs:
            context_builder.add_agent_output(
                agent_id=output["agent_id"],
                output=output["content"],
                confidence=output["confidence"],
                timestamp=output["timestamp"]
            )

        # Build compressed collaborative context
        compressed_context = context_builder.build_context(
            current_agent_id="synthesis_agent",
            max_context_length=2000,
            include_confidence=True
        )

        # Analyze attention focus
        focus_metrics = self._analyze_attention_focus(
            compressed_context,
            task,
            agent_outputs
        )

        # Calculate relevance scores
        relevant_tokens = focus_metrics["relevant_tokens"]
        total_tokens = focus_metrics["total_tokens"]
        focus_score = relevant_tokens / max(1, total_tokens)

        return {
            "context_size": len(compressed_context),
            "focus_score": focus_score,
            "relevant_tokens": relevant_tokens,
            "total_tokens": total_tokens,
            "noise_ratio": focus_metrics["noise_ratio"],
            "key_concepts_retained": focus_metrics["key_concepts_retained"],
            "redundancy_removed": focus_metrics["redundancy_removed"],
            "num_sources": len(agent_outputs)
        }

    async def test_baseline_attention_focus(self, task: str, agent_outputs: List[Dict]) -> Dict[str, Any]:
        """Test baseline without compression (full context)"""

        # Concatenate all outputs without compression
        full_context = "\n\n".join([
            f"Agent {output['agent_id']} (confidence {output['confidence']}):\n{output['content']}"
            for output in agent_outputs
        ])

        # May need truncation if too large
        max_context = 4096
        if len(full_context) > max_context:
            truncated_context = full_context[:max_context]
            information_loss = 1 - (max_context / len(full_context))
        else:
            truncated_context = full_context
            information_loss = 0

        # Analyze attention focus
        focus_metrics = self._analyze_attention_focus(
            truncated_context,
            task,
            agent_outputs
        )

        # Calculate relevance with penalty for truncation
        relevant_tokens = focus_metrics["relevant_tokens"] * (1 - information_loss)
        total_tokens = focus_metrics["total_tokens"]
        focus_score = relevant_tokens / max(1, total_tokens)

        return {
            "context_size": len(truncated_context),
            "focus_score": focus_score,
            "relevant_tokens": relevant_tokens,
            "total_tokens": total_tokens,
            "noise_ratio": focus_metrics["noise_ratio"],
            "key_concepts_retained": focus_metrics["key_concepts_retained"] * (1 - information_loss),
            "redundancy_removed": 0,  # No redundancy removal without compression
            "num_sources": len(agent_outputs),
            "information_loss": information_loss
        }

    def _generate_agent_outputs(self, task: str, num_agents: int = 10) -> List[Dict]:
        """Generate simulated agent outputs with varying relevance"""
        outputs = []
        task_keywords = task.lower().split()[:5]  # Key task words

        for i in range(num_agents):
            agent_type = ["research", "analysis", "critic"][i % 3]

            # Generate content with varying relevance
            relevance_level = random.choice(["high", "medium", "low"])

            if relevance_level == "high":
                # Highly relevant content
                content = f"Directly addressing {' '.join(task_keywords[:3])}. "
                content += f"Key finding: {random.choice(task_keywords)} is critical. "
                content += "This aligns with the core objective. " * 3
            elif relevance_level == "medium":
                # Somewhat relevant
                content = f"Related to {random.choice(task_keywords)}. "
                content += "Some tangential observations. " * 5
                content += f"Might connect to {random.choice(task_keywords)}. "
            else:
                # Low relevance (noise)
                content = "General observations unrelated to main task. " * 10

            outputs.append({
                "agent_id": f"{agent_type}_{i}",
                "content": content,
                "confidence": 0.5 + (i * 0.05),
                "timestamp": i,
                "relevance": relevance_level
            })

        return outputs

    def _analyze_attention_focus(self, context: str, task: str,
                                 agent_outputs: List[Dict]) -> Dict[str, Any]:
        """Analyze how well the context focuses on relevant information"""

        task_keywords = set(task.lower().split()[:10])
        context_lower = context.lower()

        # Count relevant tokens (task keywords appearing in context)
        relevant_tokens = sum(
            context_lower.count(keyword) for keyword in task_keywords
        )

        # Total tokens (word count)
        total_tokens = len(context.split())

        # Calculate noise (content from low-relevance agents)
        noise_count = sum(
            1 for output in agent_outputs
            if output.get("relevance") == "low" and output["agent_id"] in context
        )
        noise_ratio = noise_count / max(1, len(agent_outputs))

        # Key concepts retained (high relevance content present)
        key_concepts = sum(
            1 for output in agent_outputs
            if output.get("relevance") == "high" and output["agent_id"] in context
        )
        key_concepts_ratio = key_concepts / max(1, sum(
            1 for o in agent_outputs if o.get("relevance") == "high"
        ))

        # Redundancy (repeated phrases)
        words = context.split()
        unique_phrases = set()
        redundancy = 0
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if phrase in unique_phrases:
                redundancy += 1
            unique_phrases.add(phrase)

        return {
            "relevant_tokens": relevant_tokens,
            "total_tokens": total_tokens,
            "noise_ratio": noise_ratio,
            "key_concepts_retained": key_concepts_ratio,
            "redundancy_removed": redundancy
        }

    async def run_test(self, test_name: str, hypothesis: str,
                       target_percentage: float) -> TestResult:
        """Run single test iteration comparing attention focus"""

        # Generate test scenario
        task = TaskGenerator.generate_complex_task()
        agent_outputs = self._generate_agent_outputs(task, num_agents=15)

        print(f"  Testing Felix attention focus with compression...")
        felix_metrics = await self.test_felix_attention_focus(task, agent_outputs)

        print(f"  Testing baseline attention focus without compression...")
        baseline_metrics = await self.test_baseline_attention_focus(task, agent_outputs)

        # Calculate attention focus improvement
        focus_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["focus_score"],
            felix_metrics["focus_score"],
            lower_is_better=False
        )

        # Noise reduction improvement
        noise_reduction = MetricsCalculator.calculate_improvement(
            baseline_metrics["noise_ratio"],
            felix_metrics["noise_ratio"],
            lower_is_better=True
        )

        # Key concepts retention
        concept_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["key_concepts_retained"],
            felix_metrics["key_concepts_retained"],
            lower_is_better=False
        )

        # Overall attention improvement
        overall_improvement = (focus_improvement + noise_reduction + concept_improvement) / 3

        passed = overall_improvement >= target_percentage

        details = {
            "focus_improvement": focus_improvement,
            "noise_reduction": noise_reduction,
            "concept_improvement": concept_improvement,
            "overall_improvement": overall_improvement,
            "felix_context_size": felix_metrics["context_size"],
            "baseline_context_size": baseline_metrics["context_size"],
            "compression_achieved": 1 - (felix_metrics["context_size"] /
                                        max(1, baseline_metrics["context_size"])),
            "information_loss_avoided": baseline_metrics.get("information_loss", 0)
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
    print("H3 HYPOTHESIS TEST: ATTENTION FOCUS")
    print("Target: 25% improvement in attention focus")
    print("=" * 60)

    # Initialize test
    test = H3AttentionFocusTest(num_iterations=5)

    # Run multiple iterations
    print("\nRunning test iterations...")
    results = await test.run_multiple(
        test_name="H3_Attention_Focus",
        hypothesis="H3",
        target_percentage=25.0
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
    print(f"Target: 25%")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Test Passed: {'YES' if avg_improvement >= 25 else 'NO'}")

    print("\nIndividual Iterations:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Improvement: {result.improvement_percentage:.2f}% - {'PASS' if result.passed else 'FAIL'}")

    # Show detailed breakdown
    if results:
        latest = results[-1]
        print("\nAttention Focus Analysis (Latest Iteration):")
        details = latest.details
        print(f"  Focus Score Improvement: {details['focus_improvement']:.1f}%")
        print(f"  Noise Reduction: {details['noise_reduction']:.1f}%")
        print(f"  Key Concepts Retention: {details['concept_improvement']:.1f}%")
        print(f"  Overall Improvement: {details['overall_improvement']:.1f}%")
        print(f"\nContext Compression:")
        print(f"  Compression Achieved: {details['compression_achieved']:.1%}")
        print(f"  Information Loss Avoided: {details['information_loss_avoided']:.1%}")

    # Save results
    import json
    results_file = "tests/results/h3_attention_focus_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "hypothesis": "H3",
            "test": "Attention Focus",
            "target_improvement": 25.0,
            "average_improvement": avg_improvement,
            "success_rate": success_rate,
            "passed": avg_improvement >= 25,
            "iterations": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())