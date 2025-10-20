"""
Felix GUI Module

This module provides a Tkinter-based GUI interface for the Felix multi-agent framework.

Components:
- MainApp: Main application window with tabbed interface
- FelixSystem: Unified system manager for Felix components
- DashboardFrame: System control and monitoring
- WorkflowsFrame: Task execution through linear pipeline
- MemoryFrame: Memory and knowledge browsing/management
- AgentsFrame: Agent spawning and interaction

Usage:
    from src.gui import MainApp
    app = MainApp()
    app.mainloop()
"""

from .main import MainApp
from .felix_system import FelixSystem, FelixConfig, AgentManager
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame
from .utils import ThreadManager, DBHelper, logger
from .logging_handler import (
    setup_gui_logging,
    TkinterTextHandler,
    QueueHandler,
    add_text_widget_to_logger
)

__all__ = [
    # Main application
    'MainApp',

    # System management
    'FelixSystem',
    'FelixConfig',
    'AgentManager',

    # GUI frames
    'DashboardFrame',
    'WorkflowsFrame',
    'MemoryFrame',
    'AgentsFrame',

    # Utilities
    'ThreadManager',
    'DBHelper',
    'logger',

    # Logging
    'setup_gui_logging',
    'TkinterTextHandler',
    'QueueHandler',
    'add_text_widget_to_logger'
]

__version__ = '1.0.0'
__author__ = 'Felix Framework'
