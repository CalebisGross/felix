"""
Knowledge Brain tab components for Felix CustomTkinter GUI.

This package contains the decomposed Knowledge Brain (originally 3,529 lines)
broken into focused, maintainable modules:

- control_panel: Daemon control, processing queue, statistics, and activity feed
- knowledge_base_panel: Documents and Concepts split view
- relationships_panel: Knowledge graph relationships explorer
- maintenance_panel: Quality, Audit, and Cleanup sub-tabs
- knowledge_brain_tab: Main tab that assembles all panels
"""

from .control_panel import ControlPanel
from .knowledge_base_panel import KnowledgeBasePanel
from .relationships_panel import RelationshipsPanel
from .maintenance_panel import MaintenancePanel
from .knowledge_brain_tab import KnowledgeBrainTab

__all__ = [
    'ControlPanel',
    'KnowledgeBasePanel',
    'RelationshipsPanel',
    'MaintenancePanel',
    'KnowledgeBrainTab'
]
