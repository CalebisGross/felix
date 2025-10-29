"""
Message types and structures for the Felix Framework communication system.

Defines the core message types and Message dataclass used throughout
the Felix multi-agent communication infrastructure.
"""

import uuid
from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass, field


class MessageType(Enum):
    """Types of messages in the communication system."""
    TASK_REQUEST = "task_request"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    TASK_COMPLETE = "task_complete"
    ERROR_REPORT = "error_report"
    # Phase-aware message types for agent awareness
    PHASE_ANNOUNCE = "phase_announce"  # Agent announces entering new phase
    CONVERGENCE_SIGNAL = "convergence_signal"  # Agent signals convergence readiness
    COLLABORATION_REQUEST = "collaboration_request"  # Agent seeks peers in same phase
    SYNTHESIS_READY = "synthesis_ready"  # Signal that synthesis criteria met
    AGENT_QUERY = "agent_query"  # Agent queries for awareness information
    AGENT_DISCOVERY = "agent_discovery"  # Response with agent information
    # System action message types for system autonomy
    SYSTEM_ACTION_REQUEST = "system_action_request"  # Agent requests command execution
    SYSTEM_ACTION_RESULT = "system_action_result"  # CentralPost broadcasts execution result
    SYSTEM_ACTION_APPROVAL_NEEDED = "system_action_approval_needed"  # Command needs approval
    SYSTEM_ACTION_DENIED = "system_action_denied"  # Command blocked or denied
    SYSTEM_ACTION_START = "system_action_start"  # Command execution started (for Terminal tab)
    SYSTEM_ACTION_OUTPUT = "system_action_output"  # Real-time command output line (for Terminal tab)
    SYSTEM_ACTION_COMPLETE = "system_action_complete"  # Command execution completed (for Terminal tab)


@dataclass
class Message:
    """Message structure for communication between agents and central post."""
    sender_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
