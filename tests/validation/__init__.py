"""
Felix Hypothesis Validation Test Suite
Testing H1, H2, and H3 hypotheses with empirical measurements
"""

from .validation_utils import (
    TestResult,
    ValidationReport,
    MetricsCalculator,
    TaskGenerator,
    TestRunner,
    LLMSimulator,
    compare_metrics,
    generate_test_report
)

__all__ = [
    'TestResult',
    'ValidationReport',
    'MetricsCalculator',
    'TaskGenerator',
    'TestRunner',
    'LLMSimulator',
    'compare_metrics',
    'generate_test_report'
]