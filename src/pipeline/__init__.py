"""
Pipeline support modules for the Felix Framework.

This module provides pipeline utilities like chunking for task processing.

Note: linear_pipeline.py has been moved to exp/benchmarks/ as it is a
comparison baseline for research validation, not part of the core Felix
framework architecture. The proper Felix workflow implementation is in
src/workflows/felix_workflow.py which uses CentralPost, AgentFactory,
and the helix-based agent system.
"""

from . import chunking

__all__ = ['chunking']