"""Custom widgets for Felix GUI."""

from .message_bubble import MessageBubble, UserBubble, AssistantBubble, StreamingBubble, SystemBubble
from .typing_indicator import TypingIndicator
from .action_bubble import ActionBubble, ActionStatus
from .progress_bubble import ProgressBubble, CompactProgressIndicator, WorkflowStep, WorkflowStepStatus
from .synthesis_review_bubble import SynthesisReviewBubble, SynthesisReviewStatus

__all__ = [
    "MessageBubble",
    "UserBubble",
    "AssistantBubble",
    "StreamingBubble",
    "SystemBubble",
    "TypingIndicator",
    "ActionBubble",
    "ActionStatus",
    "ProgressBubble",
    "CompactProgressIndicator",
    "WorkflowStep",
    "WorkflowStepStatus",
    "SynthesisReviewBubble",
    "SynthesisReviewStatus",
]
