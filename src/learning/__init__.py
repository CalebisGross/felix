"""
Felix Learning Infrastructure

This module provides adaptive learning capabilities for Felix:
- PatternLearner: Activates TaskMemory recommendations
- ConfidenceCalibrator: Learns agent confidence biases
- ThresholdLearner: Optimizes confidence thresholds per task type
- RecommendationEngine: Hybrid B+A logic for auto-apply vs recommend
"""

from .pattern_learner import PatternLearner
from .confidence_calibrator import ConfidenceCalibrator
from .threshold_learner import ThresholdLearner
from .recommendation_engine import RecommendationEngine

__all__ = [
    'PatternLearner',
    'ConfidenceCalibrator',
    'ThresholdLearner',
    'RecommendationEngine'
]
