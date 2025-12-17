"""QThread-based worker classes for background operations."""

import logging
import threading
from typing import Callable, Optional, Any

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals emitted by Worker threads."""

    started = Signal()
    progress = Signal(int, str)  # percentage, status message
    result = Signal(object)  # result data
    error = Signal(str)  # error message
    finished = Signal()


class Worker(QThread):
    """Generic worker thread for background operations.

    Usage:
        def my_task(progress_callback, cancel_check):
            for i in range(100):
                if cancel_check():
                    return None
                progress_callback(i, f"Processing {i}%")
                # do work...
            return result

        worker = Worker(my_task)
        worker.signals.progress.connect(on_progress)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error)
        worker.start()
    """

    def __init__(
        self,
        fn: Callable,
        *args,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancelled = threading.Event()
        self._external_progress = progress_callback

    def run(self):
        """Execute the worker function."""
        self.signals.started.emit()
        try:
            # Inject progress callback and cancel check into kwargs
            result = self.fn(
                *self.args,
                progress_callback=self._emit_progress,
                cancel_check=self._is_cancelled,
                **self.kwargs
            )
            if not self._cancelled.is_set():
                self.signals.result.emit(result)
        except Exception as e:
            logger.exception(f"Worker error: {e}")
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

    def cancel(self):
        """Request cancellation of the worker."""
        self._cancelled.set()

    def _is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled.is_set()

    def _emit_progress(self, percentage: int, status: str):
        """Emit progress signal and call external callback if set."""
        self.signals.progress.emit(percentage, status)
        if self._external_progress:
            self._external_progress(percentage, status)


class StreamingWorker(QThread):
    """Worker specifically for streaming operations with chunk output.

    This worker is optimized for Felix's streaming responses where
    content arrives in chunks.

    Usage:
        worker = StreamingWorker(run_felix, felix_system, user_input, mode="direct")
        worker.chunk_received.connect(on_chunk)
        worker.finished_with_result.connect(on_complete)
        worker.start()
    """

    chunk_received = Signal(str)  # content chunk
    thinking_received = Signal(str, str)  # agent_name, content
    finished_with_result = Signal(dict)  # full result
    error_occurred = Signal(str)

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self._cancel_event = threading.Event()

    def run(self):
        """Execute the streaming function."""
        try:
            # Set up streaming callback that emits signals
            # Handle both callback signatures:
            # - direct mode: callback(chunk_text: str)
            # - full mode: callback(agent_name: str, chunk_text: str)
            def streaming_callback(*args):
                if not self._cancel_event.is_set():
                    if len(args) == 1:
                        chunk = args[0]
                    else:
                        agent_name, chunk = args[0], args[1]
                    self.chunk_received.emit(chunk)

            # Add our callbacks to kwargs
            self.kwargs["streaming_callback"] = streaming_callback
            self.kwargs["cancel_event"] = self._cancel_event

            result = self.fn(*self.args, **self.kwargs)

            if not self._cancel_event.is_set():
                self.finished_with_result.emit(result if result else {})

        except Exception as e:
            logger.exception(f"StreamingWorker error: {e}")
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Request cancellation."""
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancel_event.is_set()
