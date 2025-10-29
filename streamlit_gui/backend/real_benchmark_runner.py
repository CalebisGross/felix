"""Real benchmark runner integrating with tests/run_hypothesis_validation.py."""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RealBenchmarkRunner:
    """Runs Felix hypothesis validation tests via subprocess integration."""

    def __init__(self):
        """Initialize runner with paths to test suite."""
        self.results = {}
        self.test_dir = Path(__file__).parent.parent.parent / "tests"
        self.results_dir = self.test_dir / "results"
        self.runner_script = self.test_dir / "run_hypothesis_validation.py"

    def validate_test_suite_available(self) -> bool:
        """Check if test suite script exists."""
        return self.runner_script.exists()

    def run_hypothesis_validation(
        self,
        hypothesis: str = "all",
        iterations: int = 5,
        use_real_llm: bool = False,
        callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Run hypothesis validation tests via subprocess.

        Args:
            hypothesis: Which hypothesis to test ("all", "H1", "H2", "H3")
            iterations: Number of test iterations (1-20)
            use_real_llm: Use real LLM via LM Studio on port 1234
            callback: Optional progress callback function

        Returns:
            Test results dictionary or error dict
        """
        if not self.validate_test_suite_available():
            logger.error("Test suite not found at tests/run_hypothesis_validation.py")
            return {
                'error': 'Test suite not available',
                'message': f'Script not found: {self.runner_script}'
            }

        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            sys.executable,
            str(self.runner_script),
            "--iterations", str(iterations),
            "--hypothesis", hypothesis,
            "--output", str(self.results_dir / "validation_report.json")
        ]

        if use_real_llm:
            cmd.append("--real-llm")

        try:
            if callback:
                callback(f"Starting {hypothesis} validation with {iterations} iterations...")

            # Run tests
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.test_dir.parent),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if callback:
                callback("Tests complete, loading results...")

            # Load results
            report_path = self.results_dir / "validation_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    results = json.load(f)

                logger.info(f"Successfully loaded results from {report_path}")
                return results
            else:
                logger.warning("No results file generated")
                return {
                    'error': 'No results file',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }

        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out after 5 minutes")
            return {
                'error': 'Timeout',
                'message': 'Tests took longer than 5 minutes'
            }
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {
                'error': str(type(e).__name__),
                'message': str(e)
            }

    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Load most recent validation report."""
        report_path = self.results_dir / "validation_report.json"
        if not report_path.exists():
            return None

        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return None

    def get_individual_test_results(self, hypothesis: str) -> List[Dict[str, Any]]:
        """
        Load individual test result files for a hypothesis.

        Args:
            hypothesis: "H1", "H2", or "H3"

        Returns:
            List of test result dictionaries
        """
        if not self.results_dir.exists():
            return []

        results = []
        pattern = f"{hypothesis.lower()}_*.json"

        for result_file in self.results_dir.glob(pattern):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                logger.error(f"Error loading {result_file}: {e}")

        return results
