"""
Tab components for Felix GUI.

Each tab is a separate module for better maintainability.
"""

from .base_tab import ResponsiveTab, SimpleResponsiveTab
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

# Import Chat tab (with fallback for missing dependencies)
try:
    from .chat import ChatTab
except ImportError:
    ChatTab = None

__all__ = [
    'ResponsiveTab',
    'SimpleResponsiveTab',
    'DashboardTab',
    'AgentsTab',
    'MemoryTab',
    'WorkflowsTab',
    'ChatTab',
    'ApprovalsTab',
    'TerminalTab',
    'PromptsTab',
    'SettingsTab',
    'LearningTab',
    'KnowledgeBrainTab'
]
