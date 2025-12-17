"""Adapter wrapping FelixSystem for Qt signal-based communication."""

import logging
import threading
from typing import Optional, Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from ..core.signals import get_signals
from ..core.worker import StreamingWorker

# Import ApprovalDecision enum for type conversion
from src.execution.approval_manager import ApprovalDecision

logger = logging.getLogger(__name__)


class FelixAdapter(QObject):
    """Adapter that wraps FelixSystem and emits Qt signals.

    This adapter provides a Qt-friendly interface to the Felix backend.
    All operations run in background threads to keep the UI responsive.

    Usage:
        adapter = FelixAdapter()

        # Connect to signals
        adapter.system_started.connect(on_started)
        adapter.chunk_received.connect(on_chunk)

        # Start system
        adapter.start_system(config)

        # Send message
        adapter.send_message("Hello Felix", mode="direct")
    """

    # System signals
    system_started = Signal()
    system_stopped = Signal()
    system_error = Signal(str)
    status_updated = Signal(dict)

    # Chat signals
    chunk_received = Signal(str)
    response_complete = Signal(dict)
    request_failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        self._felix_system = None
        self._config = None
        self._current_worker: Optional[StreamingWorker] = None
        self._system_worker = None  # Worker for start/stop operations
        self._conversation_history: List[Dict[str, str]] = []

        # Status polling timer
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)

        # Approval polling timer (faster for responsiveness)
        self._approval_timer = QTimer(self)
        self._approval_timer.timeout.connect(self._check_pending_approvals)
        self._emitted_approvals: set = set()  # Track approvals we've already signaled

    @property
    def is_running(self) -> bool:
        """Check if Felix system is running."""
        return self._felix_system is not None and self._felix_system.running

    @property
    def felix_system(self):
        """Get the underlying FelixSystem instance."""
        return self._felix_system

    def set_conversation_history(self, history: List[Dict[str, str]]):
        """Set the conversation history (used when restoring sessions).

        Args:
            history: List of {"role": "user"/"assistant", "content": "..."}
        """
        self._conversation_history = history.copy()

    def clear_conversation_history(self):
        """Clear the conversation history (used when starting a new session)."""
        self._conversation_history.clear()

    @Slot(dict)
    def start_system(self, config: Optional[Dict[str, Any]] = None):
        """Start the Felix system in a background thread.

        Args:
            config: Optional configuration dict to override defaults
        """
        if self.is_running:
            logger.warning("Felix system already running")
            return

        def do_start(progress_callback, cancel_check):
            try:
                from src.core.felix_system import FelixSystem, FelixConfig

                # Build config from dict or use defaults
                if config:
                    felix_config = FelixConfig(**config)
                else:
                    felix_config = FelixConfig()

                self._config = felix_config
                self._felix_system = FelixSystem(felix_config)

                progress_callback(50, "Connecting to LLM...")

                if self._felix_system.start():
                    progress_callback(100, "System started")
                    return True
                else:
                    return False

            except Exception as e:
                logger.exception(f"Failed to start Felix: {e}")
                raise

        from ..core.worker import Worker
        self._system_worker = Worker(do_start)
        self._system_worker.signals.result.connect(self._on_start_complete)
        self._system_worker.signals.error.connect(self._on_start_error)
        self._system_worker.start()

    def _on_start_complete(self, success: bool):
        """Handle system start completion."""
        self._system_worker = None  # Clean up worker reference
        if success:
            logger.info("Felix system started successfully")
            self.system_started.emit()
            # Start status polling
            self._status_timer.start(2000)  # Poll every 2 seconds
            # Start approval polling (faster for UI responsiveness)
            self._approval_timer.start(500)  # Poll every 500ms
            # Emit initial status
            self._poll_status()
        else:
            self.system_error.emit("Failed to start Felix system")

    def _on_start_error(self, error: str):
        """Handle system start error."""
        self._system_worker = None  # Clean up worker reference
        logger.error(f"Felix start error: {error}")
        self.system_error.emit(error)

    @Slot()
    def stop_system(self):
        """Stop the Felix system."""
        if not self.is_running:
            logger.warning("Felix system not running")
            return

        # Stop polling timers
        self._status_timer.stop()
        self._approval_timer.stop()
        self._emitted_approvals.clear()

        # Cancel any ongoing request
        self.cancel_request()

        def do_stop(progress_callback, cancel_check):
            try:
                self._felix_system.stop()
                self._felix_system = None
                return True
            except Exception as e:
                logger.exception(f"Error stopping Felix: {e}")
                raise

        from ..core.worker import Worker
        self._system_worker = Worker(do_stop)
        self._system_worker.signals.result.connect(self._on_stop_complete)
        self._system_worker.signals.error.connect(self._on_stop_error)
        self._system_worker.start()

    def _on_stop_complete(self, success: bool):
        """Handle system stop completion."""
        self._system_worker = None  # Clean up worker reference
        self.system_stopped.emit()

    def _on_stop_error(self, error: str):
        """Handle system stop error."""
        self._system_worker = None  # Clean up worker reference
        self.system_error.emit(error)

    def _poll_status(self):
        """Poll system status and emit update."""
        if self._felix_system:
            try:
                status = self._felix_system.get_system_status()
                self.status_updated.emit(status)
            except Exception as e:
                logger.warning(f"Error polling status: {e}")

    def _check_pending_approvals(self):
        """Poll for pending approval requests and emit signals."""
        if not self._felix_system or not self._felix_system.central_post:
            return

        try:
            scm = self._felix_system.central_post.system_command_manager
            if not scm or not scm.approval_manager:
                return

            pending = scm.approval_manager.get_pending_approvals()

            for approval in pending:
                if approval.approval_id not in self._emitted_approvals:
                    self._emitted_approvals.add(approval.approval_id)
                    signals = get_signals()
                    signals.approval_requested.emit(approval.approval_id, {
                        'command': approval.command,
                        'context': approval.context or '',
                        'risk_level': approval.risk_assessment if isinstance(approval.risk_assessment, str) else (approval.risk_assessment.get('level', 'MEDIUM') if approval.risk_assessment else 'MEDIUM'),
                        'agent_id': approval.agent_id or ''
                    })
                    logger.info(f"Emitted approval request signal: {approval.approval_id}")
        except Exception as e:
            logger.warning(f"Error checking pending approvals: {e}")

    def respond_to_approval(self, approval_id: str, decision: str):
        """Send approval decision back to SystemCommandManager.

        Args:
            approval_id: The approval request ID
            decision: One of 'approve_once', 'approve_always_exact',
                     'approve_always_command', 'deny' (lowercase to match enum values)
        """
        if not self._felix_system or not self._felix_system.central_post:
            logger.warning("Cannot respond to approval - system not running")
            return

        try:
            scm = self._felix_system.central_post.system_command_manager
            # Convert string to ApprovalDecision enum
            # The enum values are lowercase: 'approve_once', 'deny', etc.
            decision_enum = ApprovalDecision(decision.lower())
            scm.approve_system_action(approval_id, decision_enum)
            logger.info(f"Sent approval response: {approval_id} -> {decision_enum.value}")
        except ValueError as e:
            logger.error(f"Invalid approval decision '{decision}': {e}")
        except Exception as e:
            logger.error(f"Error responding to approval: {e}")

    @Slot(str, str, bool)
    def send_message(
        self,
        message: str,
        mode: str = "auto",
        knowledge_enabled: bool = True
    ):
        """Send a message to Felix.

        Args:
            message: User's message text
            mode: Processing mode - "auto", "direct", or "full"
            knowledge_enabled: Whether to include knowledge brain context
        """
        if not self.is_running:
            self.request_failed.emit("Felix system not running")
            return

        if self._current_worker and self._current_worker.isRunning():
            logger.warning("Request already in progress")
            return

        # Add to conversation history
        self._conversation_history.append({
            "role": "user",
            "content": message
        })

        # Import here to avoid circular imports
        from src.workflows.felix_inference import run_felix

        # Create streaming worker
        self._current_worker = StreamingWorker(
            run_felix,
            self._felix_system,
            message,
            mode=mode,
            knowledge_enabled=knowledge_enabled,
            conversation_history=self._conversation_history.copy()
        )

        # Connect signals
        self._current_worker.chunk_received.connect(self.chunk_received.emit)
        self._current_worker.finished_with_result.connect(self._on_response_complete)
        self._current_worker.error_occurred.connect(self.request_failed.emit)

        # Start worker
        self._current_worker.start()

    def _on_response_complete(self, result: Dict[str, Any]):
        """Handle completion of a request."""
        # Add assistant response to history
        content = result.get("content", "")
        if content:
            self._conversation_history.append({
                "role": "assistant",
                "content": content
            })

        # Save to workflow history database (for Memory dev view)
        self._save_to_workflow_history(result)

        self.response_complete.emit(result)
        self._current_worker = None

    def _save_to_workflow_history(self, result: Dict[str, Any]):
        """Save workflow result to history database for persistence and dev view access."""
        try:
            from src.memory.workflow_history import WorkflowHistory

            # Only save if we have actual content
            if not result.get("content"):
                return

            workflow_history = WorkflowHistory()
            workflow_id = workflow_history.save_workflow_output(result)

            if workflow_id:
                logger.info(f"Saved workflow to history (ID: {workflow_id})")
            else:
                logger.warning("Failed to save workflow to history database")

        except Exception as e:
            logger.warning(f"Could not save to workflow history: {e}")

    @Slot()
    def cancel_request(self):
        """Cancel the current request if any."""
        if self._current_worker and self._current_worker.isRunning():
            logger.info("Cancelling current request")
            self._current_worker.cancel()
            self._current_worker.wait(1000)  # Wait up to 1 second
            self._current_worker = None

    @Slot()
    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self._conversation_history.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get current system status synchronously."""
        if self._felix_system:
            return self._felix_system.get_system_status()
        return {
            "running": False,
            "agents": 0,
            "messages_processed": 0
        }
