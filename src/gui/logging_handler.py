"""
Custom logging handler for GUI integration.

This module provides a logging handler that routes log messages to Tkinter
text widgets, ensuring that all Felix framework logging is visible in the GUI.
"""

import logging
import tkinter as tk
from queue import Queue
from typing import Optional


class TkinterTextHandler(logging.Handler):
    """
    Thread-safe logging handler that writes to a Tkinter Text widget.

    This handler uses a queue-based approach where log messages from ANY thread
    are put into a queue, and a polling method running on the main thread drains
    the queue and updates the widget. This avoids calling widget.after() from
    background threads, which causes "RuntimeError: main thread is not in main loop".

    Usage:
        handler = TkinterTextHandler(text_widget)
        logger.addHandler(handler)
        # Start polling on main thread (call this once):
        handler.start_polling()
    """

    def __init__(self, text_widget: tk.Text):
        """
        Initialize the handler with a text widget.

        Args:
            text_widget: Tkinter Text widget to write log messages to
        """
        super().__init__()
        self.text_widget = text_widget
        self.log_queue = Queue()
        self._polling = False
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        """
        Emit a log record to the queue (thread-safe).

        This method is called from ANY thread and simply puts the
        formatted message into a thread-safe queue.

        Args:
            record: LogRecord to emit
        """
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)

    def start_polling(self, poll_interval: int = 100):
        """
        Start polling the queue and updating the widget (call from main thread).

        Args:
            poll_interval: Milliseconds between polls (default: 100)
        """
        if not self._polling:
            self._polling = True
            self._poll_queue(poll_interval)

    def _poll_queue(self, poll_interval: int):
        """
        Poll the queue and update the widget (runs on main thread).

        Args:
            poll_interval: Milliseconds between polls
        """
        try:
            # Drain all pending messages from queue
            while not self.log_queue.empty():
                msg = self.log_queue.get_nowait()
                self._append_text(msg)
        except Exception:
            pass  # Ignore errors during polling

        # Schedule next poll if still active and widget exists
        if self._polling and self.text_widget.winfo_exists():
            self.text_widget.after(poll_interval, lambda: self._poll_queue(poll_interval))

    def _append_text(self, msg: str):
        """
        Append text to widget (must run on main thread).

        Args:
            msg: Formatted log message to append
        """
        try:
            if self.text_widget.winfo_exists():
                # Enable editing temporarily
                state = self.text_widget.cget('state')
                if state == 'disabled':
                    self.text_widget.config(state='normal')

                # Append message
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)

                # Restore original state
                if state == 'disabled':
                    self.text_widget.config(state='disabled')
        except Exception:
            pass  # Ignore errors during text append

    def stop_polling(self):
        """Stop polling the queue."""
        self._polling = False


class QueueHandler(logging.Handler):
    """
    Logging handler that writes to a queue for async processing.

    This is useful for GUI applications where log messages need to be
    processed on a separate thread or deferred.
    """

    def __init__(self, log_queue: Queue):
        """
        Initialize the handler with a queue.

        Args:
            log_queue: Queue to write log messages to
        """
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        """
        Emit a log record to the queue.

        Args:
            record: LogRecord to emit
        """
        try:
            msg = self.format(record)
            self.log_queue.put(msg)
        except Exception:
            self.handleError(record)


def setup_gui_logging(text_widget: Optional[tk.Text] = None,
                     log_queue: Optional[Queue] = None,
                     level: int = logging.INFO,
                     module_name: str = 'felix_gui') -> logging.Logger:
    """
    Setup logging to route to GUI components.

    This function configures a module-specific logger instead of the root logger
    to avoid conflicts between different GUI components.

    Args:
        text_widget: Optional text widget to write logs to
        log_queue: Optional queue to write logs to
        level: Logging level (default: INFO)
        module_name: Name for the logger (default: 'felix_gui')

    Returns:
        Configured logger instance
    """
    # Create or get module-specific logger (NOT root logger)
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates for THIS logger only
    logger.handlers.clear()

    # Disable propagation to avoid conflicts with root logger
    logger.propagate = False

    # Add handlers based on what was provided
    if text_widget:
        handler = TkinterTextHandler(text_widget)
        handler.setLevel(level)
        handler.start_polling()  # Start polling on main thread
        logger.addHandler(handler)

    if log_queue:
        handler = QueueHandler(log_queue)
        handler.setLevel(level)
        logger.addHandler(handler)

    # Setup Felix module loggers to propagate to this logger
    # CRITICAL FIX: Attach handlers to Felix module loggers so their logs appear in GUI
    felix_modules = [
        'src.agents',
        'src.communication',  # <-- This is where central_post.py logs come from!
        'src.llm',
        'src.memory',
        'src.pipeline',
        'src.core',
        'src.gui'  # Include GUI module logs (felix_system.py, etc.)
    ]

    for mod_name in felix_modules:
        module_logger = logging.getLogger(mod_name)
        module_logger.setLevel(level)
        module_logger.handlers.clear()  # Remove any existing handlers

        # Add the SAME handlers as the main logger so logs appear in GUI
        if text_widget:
            handler = TkinterTextHandler(text_widget)
            handler.setLevel(level)
            handler.start_polling()  # Start polling on main thread
            module_logger.addHandler(handler)

        if log_queue:
            handler = QueueHandler(log_queue)
            handler.setLevel(level)
            module_logger.addHandler(handler)

        module_logger.propagate = False  # Don't propagate since we have direct handlers

    return logger


def add_text_widget_to_logger(logger: logging.Logger, text_widget: tk.Text,
                              level: int = logging.INFO) -> logging.Handler:
    """
    Add a text widget handler to a specific logger.

    Args:
        logger: Logger to add handler to
        text_widget: Text widget to write to
        level: Logging level for this handler

    Returns:
        The created handler (so it can be removed later if needed)
    """
    handler = TkinterTextHandler(text_widget)
    handler.setLevel(level)
    handler.start_polling()  # Start polling on main thread
    logger.addHandler(handler)
    return handler


def remove_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    """
    Remove a handler from a logger.

    Args:
        logger: Logger to remove handler from
        handler: Handler to remove
    """
    logger.removeHandler(handler)
