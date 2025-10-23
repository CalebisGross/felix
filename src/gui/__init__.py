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
    'MainApp',
    'FelixSystem',
    'FelixConfig',
    'AgentManager',
    'DashboardFrame',
    'WorkflowsFrame',
    'MemoryFrame',
    'AgentsFrame',
    'ThreadManager',
    'DBHelper',
    'logger',
    'setup_gui_logging',
    'TkinterTextHandler',
    'QueueHandler',
    'add_text_widget_to_logger'
]

__version__ = '1.0.0'
__author__ = 'Felix Framework'
