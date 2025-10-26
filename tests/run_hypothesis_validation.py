#!/usr/bin/env python3
"""
Main Hypothesis Validation Runner for Felix Framework
Runs all H1, H2, and H3 hypothesis tests and generates comprehensive report
"""

import asyncio
import sys
import os
from datetime import datetime
import json
import argparse
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tests.validation.test_h1_workload_distribution import H1WorkloadDistributionTest
from tests.validation.test_h1_adaptive_behavior import H1AdaptiveBehaviorTest
from tests.validation.test_h2_communication_efficiency import H2CommunicationEfficiencyTest
from tests.validation.test_h2_resource_allocation import H2ResourceAllocationTest
from tests.validation.test_h3_memory_compression import H3MemoryCompressionTest
from tests.validation.test_h3_attention_focus import H3AttentionFocusTest
from tests.validation.validation_utils import ValidationReport, generate_test_report

class HypothesisValidator:
    """Main validation orchestrator for all hypotheses"""

    def __init__(self, num_iterations: int = 5, use_real_llm: bool = False):
        self.num_iterations = num_iterations
        self.use_real_llm = use_real_llm
        self.results = {
            "H1": [],
            "H2": [],
            "H3": []
        }

    async def validate_h1(self) -> Dict[str, Any]:
        """Validate H1: Helical progression enhances adaptation (20% improvement)"""
        print("\n" + "=" * 70)
        print("VALIDATING H1: HELICAL PROGRESSION ENHANCES ADAPTATION")
        print("=" * 70)

        all_results = []

        # Test 1: Workload Distribution
        print("\n[H1.1] Testing Workload Distribution...")
        test = H1WorkloadDistributionTest(
            num_iterations=self.num_iterations,
            use_real_llm=self.use_real_llm
        )
        results = await test.run_multiple(
            "H1_Workload_Distribution", "H1", 20.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H1", 20.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        # Test 2: Adaptive Behavior
        print("\n[H1.2] Testing Adaptive Behavior...")
        test = H1AdaptiveBehaviorTest(num_iterations=self.num_iterations)
        results = await test.run_multiple(
            "H1_Adaptive_Behavior", "H1", 20.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H1", 20.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        self.results["H1"] = all_results
        return self._summarize_hypothesis(all_results, "H1", 20.0)

    async def validate_h2(self) -> Dict[str, Any]:
        """Validate H2: Hub-spoke optimizes resources (15% improvement)"""
        print("\n" + "=" * 70)
        print("VALIDATING H2: HUB-SPOKE COMMUNICATION OPTIMIZES RESOURCES")
        print("=" * 70)

        all_results = []

        # Test 1: Communication Efficiency
        print("\n[H2.1] Testing Communication Efficiency...")
        test = H2CommunicationEfficiencyTest(num_iterations=self.num_iterations)
        results = await test.run_multiple(
            "H2_Communication_Efficiency", "H2", 15.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H2", 15.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        # Test 2: Resource Allocation
        print("\n[H2.2] Testing Resource Allocation...")
        test = H2ResourceAllocationTest(num_iterations=self.num_iterations)
        results = await test.run_multiple(
            "H2_Resource_Allocation", "H2", 15.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H2", 15.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        self.results["H2"] = all_results
        return self._summarize_hypothesis(all_results, "H2", 15.0)

    async def validate_h3(self) -> Dict[str, Any]:
        """Validate H3: Memory compression reduces latency (25% improvement)"""
        print("\n" + "=" * 70)
        print("VALIDATING H3: MEMORY COMPRESSION IMPROVES ATTENTION FOCUS")
        print("=" * 70)

        all_results = []

        # Test 1: Memory Compression
        print("\n[H3.1] Testing Memory Compression...")
        test = H3MemoryCompressionTest(num_iterations=self.num_iterations)
        results = await test.run_multiple(
            "H3_Memory_Compression", "H3", 25.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H3", 25.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        # Test 2: Attention Focus
        print("\n[H3.2] Testing Attention Focus...")
        test = H3AttentionFocusTest(num_iterations=self.num_iterations)
        results = await test.run_multiple(
            "H3_Attention_Focus", "H3", 25.0
        )
        all_results.extend(results)
        report = generate_test_report(results, "H3", 25.0)
        print(f"  Result: {report['average_improvement']:.1f}% improvement")

        self.results["H3"] = all_results
        return self._summarize_hypothesis(all_results, "H3", 25.0)

    def _summarize_hypothesis(self, results: List, hypothesis: str,
                             target: float) -> Dict[str, Any]:
        """Summarize results for a hypothesis"""
        if not results:
            return {"error": "No results"}

        improvements = [r.improvement_percentage for r in results]
        avg_improvement = sum(improvements) / len(improvements)
        success_rate = sum(1 for r in results if r.passed) / len(results) * 100

        return {
            "hypothesis": hypothesis,
            "target": target,
            "average_improvement": avg_improvement,
            "success_rate": success_rate,
            "passed": avg_improvement >= target,
            "num_tests": len(results)
        }

    async def run_full_validation(self) -> ValidationReport:
        """Run complete validation suite"""
        print("\n" + "=" * 80)
        print(" " * 20 + "FELIX HYPOTHESIS VALIDATION SUITE")
        print(" " * 25 + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Iterations per test: {self.num_iterations}")
        print(f"  Using real LLM: {self.use_real_llm}")

        # Validate each hypothesis
        h1_summary = await self.validate_h1()
        h2_summary = await self.validate_h2()
        h3_summary = await self.validate_h3()

        # Generate final report
        overall_success = all([
            h1_summary["passed"],
            h2_summary["passed"],
            h3_summary["passed"]
        ])

        summary = {
            "overall_success": overall_success,
            "h1": h1_summary,
            "h2": h2_summary,
            "h3": h3_summary,
            "configuration": {
                "iterations": self.num_iterations,
                "real_llm": self.use_real_llm
            }
        }

        report = ValidationReport(
            h1_results=self.results["H1"],
            h2_results=self.results["H2"],
            h3_results=self.results["H3"],
            summary=summary,
            timestamp=datetime.now().isoformat()
        )

        return report

    def print_final_summary(self, report: ValidationReport):
        """Print comprehensive summary of validation results"""
        print("\n" + "=" * 80)
        print(" " * 30 + "FINAL RESULTS SUMMARY")
        print("=" * 80)

        summary = report.summary

        # H1 Results
        h1 = summary["h1"]
        print(f"\n📊 H1: Helical Progression Enhances Adaptation")
        print(f"   Target: 20% improvement")
        print(f"   Achieved: {h1['average_improvement']:.1f}%")
        print(f"   Success Rate: {h1['success_rate']:.1f}%")
        print(f"   Status: {'✅ PASSED' if h1['passed'] else '❌ FAILED'}")

        # H2 Results
        h2 = summary["h2"]
        print(f"\n📊 H2: Hub-Spoke Communication Optimizes Resources")
        print(f"   Target: 15% improvement")
        print(f"   Achieved: {h2['average_improvement']:.1f}%")
        print(f"   Success Rate: {h2['success_rate']:.1f}%")
        print(f"   Status: {'✅ PASSED' if h2['passed'] else '❌ FAILED'}")

        # H3 Results
        h3 = summary["h3"]
        print(f"\n📊 H3: Memory Compression Improves Attention Focus")
        print(f"   Target: 25% improvement")
        print(f"   Achieved: {h3['average_improvement']:.1f}%")
        print(f"   Success Rate: {h3['success_rate']:.1f}%")
        print(f"   Status: {'✅ PASSED' if h3['passed'] else '❌ FAILED'}")

        # Overall Status
        print("\n" + "=" * 80)
        if summary["overall_success"]:
            print(" " * 20 + "🎉 ALL HYPOTHESES VALIDATED SUCCESSFULLY! 🎉")
        else:
            print(" " * 25 + "⚠️ VALIDATION INCOMPLETE ⚠️")
        print("=" * 80)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Felix Framework Hypothesis Validation Suite"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=5,
        help="Number of iterations per test (default: 5)"
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use real LLM via LM Studio (must be running on port 1234)"
    )
    parser.add_argument(
        "--hypothesis",
        choices=["H1", "H2", "H3", "all"],
        default="all",
        help="Which hypothesis to validate (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tests/results/validation_report.json",
        help="Output file for validation report"
    )

    args = parser.parse_args()

    # Create validator
    validator = HypothesisValidator(
        num_iterations=args.iterations,
        use_real_llm=args.real_llm
    )

    # Run validation based on selection
    if args.hypothesis == "all":
        report = await validator.run_full_validation()
    elif args.hypothesis == "H1":
        h1_summary = await validator.validate_h1()
        report = ValidationReport(
            h1_results=validator.results["H1"],
            h2_results=[],
            h3_results=[],
            summary={"h1": h1_summary},
            timestamp=datetime.now().isoformat()
        )
    elif args.hypothesis == "H2":
        h2_summary = await validator.validate_h2()
        report = ValidationReport(
            h1_results=[],
            h2_results=validator.results["H2"],
            h3_results=[],
            summary={"h2": h2_summary},
            timestamp=datetime.now().isoformat()
        )
    else:  # H3
        h3_summary = await validator.validate_h3()
        report = ValidationReport(
            h1_results=[],
            h2_results=[],
            h3_results=validator.results["H3"],
            summary={"h3": h3_summary},
            timestamp=datetime.now().isoformat()
        )

    # Print summary
    validator.print_final_summary(report)

    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report.save(args.output)
    print(f"\n📁 Full report saved to: {args.output}")

    # Return exit code based on validation success
    if args.hypothesis == "all":
        sys.exit(0 if report.summary["overall_success"] else 1)
    else:
        hypothesis_key = args.hypothesis.lower()
        sys.exit(0 if report.summary[hypothesis_key]["passed"] else 1)


if __name__ == "__main__":
    asyncio.run(main())