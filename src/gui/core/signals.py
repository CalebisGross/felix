"""Custom Qt signals for Felix GUI events."""

import logging
from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


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

    # Synthesis review signals (for low-confidence synthesis approval)
    synthesis_review_requested = Signal(str, dict)  # review_id, review_data
    synthesis_review_response = Signal(str, dict)  # review_id, decision dict

    # Command execution
    command_started = Signal(str, str)  # exec_id, command
    command_output = Signal(str, str)  # exec_id, output line
    command_completed = Signal(str, bool, int)  # exec_id, success, exit_code

    # Knowledge brain
    knowledge_updated = Signal(int)  # entry count

    # Log messages
    log_message = Signal(str, str)  # level, message

    # Layout changes
    layout_mode_changed = Signal(str)  # "compact" | "standard" | "wide"

    def __init__(self, parent=None):
        super().__init__(parent)
        logger.debug("FelixSignals instance created - signal hub ready for connections")

    def log_available_signals(self):
        """Log all available signals for debugging purposes (Issue #56.6)."""
        signal_names = [
            "system_starting", "system_started", "system_stopping", "system_stopped",
            "system_error", "status_updated", "request_started", "chunk_received",
            "thinking_step", "response_complete", "request_failed", "request_cancelled",
            "approval_requested", "approval_resolved", "synthesis_review_requested",
            "synthesis_review_response", "command_started", "command_output",
            "command_completed", "knowledge_updated", "log_message", "layout_mode_changed"
        ]
        logger.info(f"FelixSignals: {len(signal_names)} signals available for connection")
        logger.debug(f"Available signals: {', '.join(signal_names)}")


# Global singleton instance
_signals_instance = None


def get_signals() -> FelixSignals:
    """Get the global FelixSignals instance."""
    global _signals_instance
    if _signals_instance is None:
        _signals_instance = FelixSignals()
    return _signals_instance
