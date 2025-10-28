"""
Feedback system for Felix learning and quality control.

This module provides three-tier feedback collection:
- Quick workflow ratings (thumbs up/down)
- Detailed workflow feedback (multi-dimensional ratings)
- Knowledge-level feedback (entry-specific corrections)
- Integration with knowledge store and learning systems
"""

from .feedback_manager import (
    FeedbackManager,
    WorkflowRating,
    DetailedFeedback,
    KnowledgeFeedback,
    FeedbackType,
    ReasonCategory
)
from .feedback_integrator import FeedbackIntegrator

__all__ = [
    'FeedbackManager',
    'FeedbackIntegrator',
    'WorkflowRating',
    'DetailedFeedback',
    'KnowledgeFeedback',
    'FeedbackType',
    'ReasonCategory'
]
