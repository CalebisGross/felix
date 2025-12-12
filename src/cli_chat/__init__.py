"""
Felix Conversational CLI Package

Provides interactive chat interface for Felix with session management and tools.

The CLI now properly integrates with Felix's multi-agent architecture through
the CLIWorkflowOrchestrator, ensuring proper use of:
- Helical agent progression
- CentralPost hub-spoke communication
- CollaborativeContextBuilder
- Knowledge store with meta-learning
- Self-improvement architecture
- Workflow-session continuity
"""

from .session_manager import SessionManager, Message, Session
from .formatters import OutputFormatter, RichOutputFormatter, get_formatter
from .command_handler import CommandHandler
from .tools import ToolRegistry
from .chat import FelixChat, run_chat, run_single_query
from .custom_commands import CustomCommand, CustomCommandLoader
from .completers import FelixCompleter, create_felix_completer
from .cli_workflow_orchestrator import CLIWorkflowOrchestrator

__all__ = [
    'SessionManager',
    'Message',
    'Session',
    'OutputFormatter',
    'RichOutputFormatter',
    'get_formatter',
    'CommandHandler',
    'ToolRegistry',
    'FelixChat',
    'run_chat',
    'run_single_query',
    'CustomCommand',
    'CustomCommandLoader',
    'FelixCompleter',
    'create_felix_completer',
    'CLIWorkflowOrchestrator',
]
