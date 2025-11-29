"""
Tab components for Felix GUI.

Each tab is a separate module for better maintainability.
"""

from .dashboard import DashboardTab
from .agents import AgentsTab
from .memory import MemoryTab
from .workflows import WorkflowsTab
from .approvals import ApprovalsTab
from .terminal import TerminalTab
from .prompts import PromptsTab
from .settings import SettingsTab
from .learning import LearningTab
from .knowledge_brain import KnowledgeBrainTab

__all__ = [
    'DashboardTab',
    'AgentsTab',
    'MemoryTab',
    'WorkflowsTab',
    'ApprovalsTab',
    'TerminalTab',
    'PromptsTab',
    'SettingsTab',
    'LearningTab',
    'KnowledgeBrainTab'
]
