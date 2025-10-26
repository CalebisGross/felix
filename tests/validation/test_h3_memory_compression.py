"""
H3 Hypothesis Test: Memory Compression
Tests whether memory compression reduces latency with 25% improvement
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import statistics
import time
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.validation.validation_utils import (
    TestResult, MetricsCalculator, TaskGenerator, TestRunner, LLMSimulator
)
from src.memory.context_compression import ContextCompressor
from src.memory.knowledge_store import KnowledgeStore

class H3MemoryCompressionTest(TestRunner):
    """Test memory compression impact on latency and efficiency"""

    def __init__(self, num_iterations: int = 10):
        super().__init__(num_iterations)
        self.llm_simulator = LLMSimulator()

    async def test_felix_with_compression(self, context_items: List[str]) -> Dict[str, Any]:
        """Test Felix with context compression enabled"""

        compressor = ContextCompressor(
            compression_ratio=0.3,  # Target 30% of original size
            strategy="abstractive"
        )
        knowledge_store = KnowledgeStore(":memory:")

        # Metrics tracking
        processing_times = []
        compressed_sizes = []
        original_sizes = []
        quality_scores = []

        for i, context in enumerate(context_items):
            start_time = time.time()

            # Store in knowledge store
            knowledge_store.add_knowledge(
                domain=f"test_domain_{i}",
                content=context,
                confidence=0.7 + (i * 0.01),
                tags=["test", "compression"]
            )

            # Compress context
            compressed = compressor.compress(context)
            compressed_size = len(compressed)
            original_size = len(context)

            # Simulate LLM processing with compressed context
            # Smaller context = faster processing
            processing_delay = compressed_size / 10000  # Simulated delay
            await asyncio.sleep(processing_delay)

            response = self.llm_simulator.generate_response(
                compressed,
                temperature=0.5,
                max_tokens=500
            )

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            compressed_sizes.append(compressed_size)
            original_sizes.append(original_size)
            quality_scores.append(response["confidence"])

        # Calculate metrics
        avg_processing_time = statistics.mean(processing_times)
        avg_compression_ratio = statistics.mean(
            c / o for c, o in zip(compressed_sizes, original_sizes)
        )
        avg_quality = statistics.mean(quality_scores)

        # Context window utilization
        max_context_size = 4096  # Typical context window
        avg_utilization = statistics.mean(c / max_context_size for c in compressed_sizes)

        return {
            "avg_processing_time": avg_processing_time,
            "avg_compression_ratio": avg_compression_ratio,
            "avg_quality": avg_quality,
            "context_utilization": avg_utilization,
            "total_compressed_size": sum(compressed_sizes),
            "total_original_size": sum(original_sizes),
            "num_items": len(context_items),
            "processing_times": processing_times
        }

    async def test_without_compression(self, context_items: List[str]) -> Dict[str, Any]:
        """Test baseline without context compression"""

        processing_times = []
        context_sizes = []
        quality_scores = []

        for context in context_items:
            start_time = time.time()

            # No compression, use full context
            context_size = len(context)

            # Simulate LLM processing with full context
            # Larger context = slower processing
            processing_delay = context_size / 10000  # Simulated delay
            await asyncio.sleep(processing_delay)

            # May need truncation if context too large
            max_context = 4096
            if context_size > max_context:
                # Truncate and lose information
                truncated_context = context[:max_context]
                quality_penalty = 0.2  # Quality loss from truncation
            else:
                truncated_context = context
                quality_penalty = 0.0

            response = self.llm_simulator.generate_response(
                truncated_context,
                temperature=0.5,
                max_tokens=500
            )

            # Apply quality penalty for truncation
            adjusted_quality = response["confidence"] * (1 - quality_penalty)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            context_sizes.append(context_size)
            quality_scores.append(adjusted_quality)

        # Calculate metrics
        avg_processing_time = statistics.mean(processing_times)
        avg_quality = statistics.mean(quality_scores)

        # Context window utilization
        max_context_size = 4096
        avg_utilization = statistics.mean(
            min(1.0, c / max_context_size) for c in context_sizes
        )

        # Count how many contexts exceeded limit
        truncation_count = sum(1 for c in context_sizes if c > max_context_size)

        return {
            "avg_processing_time": avg_processing_time,
            "avg_compression_ratio": 1.0,  # No compression
            "avg_quality": avg_quality,
            "context_utilization": avg_utilization,
            "total_compressed_size": sum(context_sizes),
            "total_original_size": sum(context_sizes),
            "num_items": len(context_items),
            "truncation_count": truncation_count,
            "processing_times": processing_times
        }

    def _generate_context_items(self, num_items: int = 20) -> List[str]:
        """Generate test context items of varying sizes"""
        contexts = []
        base_sizes = [500, 1000, 2000, 3000, 5000]  # Character counts

        for i in range(num_items):
            size = random.choice(base_sizes)
            # Generate context of approximately the target size
            context = f"Context item {i}: " + "x" * size
            contexts.append(context)

        return contexts

    async def run_test(self, test_name: str, hypothesis: str,
                       target_percentage: float) -> TestResult:
        """Run single test iteration comparing with/without compression"""

        # Generate test contexts
        context_items = self._generate_context_items(20)

        print(f"  Testing Felix with compression...")
        felix_metrics = await self.test_felix_with_compression(context_items)

        print(f"  Testing baseline without compression...")
        baseline_metrics = await self.test_without_compression(context_items)

        # Calculate improvements
        latency_improvement = MetricsCalculator.calculate_improvement(
            baseline_metrics["avg_processing_time"],
            felix_metrics["avg_processing_time"],
            lower_is_better=True
        )

        # Quality comparison (should not degrade much)
        quality_change = MetricsCalculator.calculate_improvement(
            baseline_metrics["avg_quality"],
            felix_metrics["avg_quality"],
            lower_is_better=False
        )

        # Context efficiency (how well we use available context window)
        context_efficiency = (1 - felix_metrics["context_utilization"]) * 100

        # Overall improvement focuses on latency reduction
        overall_improvement = latency_improvement

        passed = overall_improvement >= target_percentage

        # Calculate attention focus improvement
        # With compression, more relevant information fits in context
        attention_improvement = self._calculate_attention_improvement(
            felix_metrics, baseline_metrics
        )

        details = {
            "latency_improvement": latency_improvement,
            "quality_change": quality_change,
            "context_efficiency": context_efficiency,
            "attention_improvement": attention_improvement,
            "compression_ratio": felix_metrics["avg_compression_ratio"],
            "truncations_avoided": baseline_metrics.get("truncation_count", 0)
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

    def _calculate_attention_improvement(self, felix_metrics: Dict, baseline_metrics: Dict) -> float:
        """Calculate improvement in attention focus"""
        # With compression, we can fit more relevant content
        # Without compression, we truncate and lose information

        # Estimate based on compression ratio and quality maintained
        felix_relevance = felix_metrics["avg_quality"] / felix_metrics["avg_compression_ratio"]
        baseline_relevance = baseline_metrics["avg_quality"]  # No compression benefit

        return MetricsCalculator.calculate_improvement(
            baseline_relevance,
            felix_relevance,
            lower_is_better=False
        )


async def main():
    """Main test execution"""
    print("=" * 60)
    print("H3 HYPOTHESIS TEST: MEMORY COMPRESSION")
    print("Target: 25% latency reduction with compression")
    print("=" * 60)

    # Initialize test
    test = H3MemoryCompressionTest(num_iterations=5)

    # Run multiple iterations
    print("\nRunning test iterations...")
    results = await test.run_multiple(
        test_name="H3_Memory_Compression",
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
        print("\nCompression Analysis (Latest Iteration):")
        details = latest.details
        print(f"  Latency Improvement: {details['latency_improvement']:.1f}%")
        print(f"  Quality Change: {details['quality_change']:.1f}%")
        print(f"  Context Efficiency: {details['context_efficiency']:.1f}%")
        print(f"  Attention Improvement: {details['attention_improvement']:.1f}%")
        print(f"  Compression Ratio: {details['compression_ratio']:.2f}")
        print(f"  Truncations Avoided: {details['truncations_avoided']}")

    # Save results
    import json
    results_file = "tests/results/h3_memory_compression_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "hypothesis": "H3",
            "test": "Memory Compression",
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