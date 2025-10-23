#!/usr/bin/env python3
"""
Simple test to verify the hypothesis validation framework is working
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from tests.validation.validation_utils import TaskGenerator, MetricsCalculator

async def test_basic_functionality():
    """Test basic functionality of validation framework"""

    print("Testing validation framework components...")

    # Test task generator
    print("\n1. Testing TaskGenerator:")
    task = TaskGenerator.generate_research_task()
    print(f"   Generated task: {task[:50]}...")

    # Test metrics calculator
    print("\n2. Testing MetricsCalculator:")
    workload = {"agent1": 10.5, "agent2": 12.3, "agent3": 9.8}
    variance = MetricsCalculator.calculate_workload_variance(workload)
    print(f"   Workload variance: {variance:.4f}")

    improvement = MetricsCalculator.calculate_improvement(100, 80, lower_is_better=True)
    print(f"   Improvement (100 -> 80): {improvement:.1f}%")

    # Test baseline linear workflow
    print("\n3. Testing Linear Baseline:")
    from tests.baselines.linear_progression import LinearWorkflow
    workflow = LinearWorkflow(num_agents=3)
    print(f"   Created workflow with {len(workflow.agents)} agents")

    # Test mesh communication baseline
    print("\n4. Testing Mesh Communication Baseline:")
    from tests.baselines.mesh_communication import MeshCommunicationSystem
    mesh = MeshCommunicationSystem(num_agents=3)
    print(f"   Created mesh with {len(mesh.agents)} agents")
    connections = sum(len(a.connections) for a in mesh.agents.values())
    print(f"   Total connections: {connections} (expected ~6 for full mesh)")

    print("\n✅ All basic components working!")
    return True

async def test_h1_import():
    """Test H1 hypothesis test imports"""

    print("\nTesting H1 hypothesis tests...")

    try:
        from tests.validation.test_h1_workload_distribution import H1WorkloadDistributionTest
        print("   ✅ H1 Workload Distribution test imported")

        from tests.validation.test_h1_adaptive_behavior import H1AdaptiveBehaviorTest
        print("   ✅ H1 Adaptive Behavior test imported")

        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

async def test_h2_import():
    """Test H2 hypothesis test imports"""

    print("\nTesting H2 hypothesis tests...")

    try:
        from tests.validation.test_h2_communication_efficiency import H2CommunicationEfficiencyTest
        print("   ✅ H2 Communication Efficiency test imported")

        from tests.validation.test_h2_resource_allocation import H2ResourceAllocationTest
        print("   ✅ H2 Resource Allocation test imported")

        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

async def test_h3_import():
    """Test H3 hypothesis test imports"""

    print("\nTesting H3 hypothesis tests...")

    try:
        from tests.validation.test_h3_memory_compression import H3MemoryCompressionTest
        print("   ✅ H3 Memory Compression test imported")

        from tests.validation.test_h3_attention_focus import H3AttentionFocusTest
        print("   ✅ H3 Attention Focus test imported")

        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

async def main():
    """Main test execution"""
    print("=" * 60)
    print("FELIX HYPOTHESIS VALIDATION FRAMEWORK TEST")
    print("=" * 60)

    results = []

    # Test basic functionality
    results.append(await test_basic_functionality())

    # Test imports
    results.append(await test_h1_import())
    results.append(await test_h2_import())
    results.append(await test_h3_import())

    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ ALL TESTS PASSED - Framework is ready!")
        print("\nYou can now run the full validation suite:")
        print("  python tests/run_hypothesis_validation.py")
        print("\nOr run individual tests:")
        print("  python tests/run_hypothesis_validation.py --hypothesis H1")
        print("  python tests/run_hypothesis_validation.py --hypothesis H2")
        print("  python tests/run_hypothesis_validation.py --hypothesis H3")
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
    print("=" * 60)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))