"""Adapter wrapping FelixSystem for Qt signal-based communication."""

import logging
import threading
from typing import Optional, Dict, Any, List

from PySide6.QtCore import QObject, Signal, Slot, QTimer

from ..core.signals import get_signals
from ..core.worker import StreamingWorker

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

    @property
    def is_running(self) -> bool:
        """Check if Felix system is running."""
        return self._felix_system is not None and self._felix_system.running

    @property
    def felix_system(self):
        """Get the underlying FelixSystem instance."""
        return self._felix_system

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

        # Stop status polling
        self._status_timer.stop()

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

        self.response_complete.emit(result)
        self._current_worker = None

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
