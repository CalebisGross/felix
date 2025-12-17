"""Custom Qt signals for Felix GUI events."""

from PySide6.QtCore import QObject, Signal


class FelixSignals(QObject):
    """Central signal hub for Felix GUI events.

    This class provides a single source of truth for all Felix-related
    signals. Components connect to these signals to receive updates.
    """

    # System lifecycle
    system_starting = Signal()
    system_started = Signal()
    system_stopping = Signal()
    system_stopped = Signal()
    system_error = Signal(str)  # error message

    # Status updates
    status_updated = Signal(dict)  # full status dict from FelixSystem

    # Chat/Workflow signals
    request_started = Signal(str)  # request_id
    chunk_received = Signal(str)  # content chunk for streaming
    thinking_step = Signal(str, str)  # agent_name, content
    response_complete = Signal(dict)  # full result dict
    request_failed = Signal(str, str)  # request_id, error message
    request_cancelled = Signal(str)  # request_id

    # Approval signals (for system commands)
    approval_requested = Signal(str, dict)  # approval_id, command details
    approval_resolved = Signal(str, str)  # approval_id, decision (approved/denied)

    # Command execution
    command_started = Signal(str, str)  # exec_id, command
    command_output = Signal(str, str)  # exec_id, output line
    command_completed = Signal(str, bool, int)  # exec_id, success, exit_code

    # Knowledge brain
    knowledge_updated = Signal(int)  # entry count

    # Log messages
    log_message = Signal(str, str)  # level, message


# Global singleton instance
_signals_instance = None


def get_signals() -> FelixSignals:
    """Get the global FelixSignals instance."""
    global _signals_instance
    if _signals_instance is None:
        _signals_instance = FelixSignals()
    return _signals_instance
