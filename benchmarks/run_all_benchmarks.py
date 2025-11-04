"""
Run All Benchmarks

Executes all Felix benchmarks and generates a summary report.
"""

import subprocess
import sys
from datetime import datetime


def run_benchmark(script_name: str, description: str) -> bool:
    """
    Run a single benchmark script.

    Args:
        script_name: Name of benchmark script to run
        description: Description of the benchmark

    Returns:
        True if successful, False otherwise
    """
    print()
    print("=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print()

    try:
        result = subprocess.run(
            [sys.executable, f"benchmarks/{script_name}"],
            check=True,
            capture_output=False
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with error code {e.returncode}")
        return False

    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {str(e)}")
        return False


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("Felix Framework - Complete Benchmark Suite")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    benchmarks = [
        ("benchmark_communication.py", "Communication Overhead (O(N) vs O(N²))"),
        ("benchmark_airgapped.py", "Air-Gapped Startup Test"),
        ("benchmark_retrieval.py", "Meta-Learning Retrieval Performance"),
    ]

    results = {}

    for script, description in benchmarks:
        success = run_benchmark(script, description)
        results[description] = success

    # Print summary
    print()
    print("=" * 70)
    print("Benchmark Suite Complete!")
    print("=" * 70)
    print()
    print("Results:")
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}  {description}")

    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Results saved in benchmarks/results/")
    print()

    # Exit with error code if any benchmark failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
