"""
System Execution Module for Felix Framework.

This module provides system autonomy capabilities including:
- Command execution with safety controls
- Trust-based approval system
- Command history tracking
- Virtual environment detection
- Pattern learning from execution history

Components:
- system_executor: Command execution with resource limits
- trust_manager: Command classification and approval workflow
- command_history: Database operations for command tracking
"""

from .system_executor import SystemExecutor, CommandResult, ExecutionError
from .trust_manager import TrustManager, TrustLevel, ApprovalRequest
from .command_history import CommandHistory

__all__ = [
    'SystemExecutor',
    'CommandResult',
    'ExecutionError',
    'TrustManager',
    'TrustLevel',
    'ApprovalRequest',
    'CommandHistory'
]
