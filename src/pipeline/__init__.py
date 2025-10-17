"""
Linear pipeline architecture for the Felix Framework comparison.

This module provides a traditional linear processing pipeline that serves
as a baseline comparison against the helix-based architecture.

Components:
- LinearPipeline: Sequential stage-based processing system
- PipelineAgent: Agent implementation for linear progression
- PipelineStage: Individual pipeline stage with capacity management

Mathematical Foundation:
- Sequential processing: agents progress through fixed stages 0→1→...→N
- Uniform workload distribution across pipeline stages
- Linear message passing between adjacent stages
- Performance baseline for Hypothesis H1 validation

This implementation supports research validation by providing a controlled
comparison architecture with measurably different characteristics from
the helix-based system.
"""

from .linear_pipeline import LinearPipeline, PipelineAgent, PipelineStage

__all__ = ['LinearPipeline', 'PipelineAgent', 'PipelineStage']