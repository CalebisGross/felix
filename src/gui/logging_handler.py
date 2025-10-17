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
    Logging handler that writes to a Tkinter Text widget.

    This handler is thread-safe and properly schedules GUI updates
    on the main thread using the 'after' method.
    """

    def __init__(self, text_widget: tk.Text):
        """
        Initialize the handler with a text widget.

        Args:
            text_widget: Tkinter Text widget to write log messages to
        """
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))

    def emit(self, record):
        """
        Emit a log record to the text widget.

        Args:
            record: LogRecord to emit
        """
        try:
            msg = self.format(record)

            def append_text():
                """Append text to widget (must run on main thread)."""
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

            # Schedule on main thread
            self.text_widget.after(0, append_text)

        except Exception:
            self.handleError(record)


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
                     level: int = logging.INFO) -> None:
    """
    Setup logging to route to GUI components.

    This function configures the root logger and all Felix module loggers
    to output to GUI components instead of console.

    Args:
        text_widget: Optional text widget to write logs to
        log_queue: Optional queue to write logs to
        level: Logging level (default: INFO)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers based on what was provided
    if text_widget:
        handler = TkinterTextHandler(text_widget)
        handler.setLevel(level)
        root_logger.addHandler(handler)

    if log_queue:
        handler = QueueHandler(log_queue)
        handler.setLevel(level)
        root_logger.addHandler(handler)

    # Also setup for specific Felix modules
    felix_modules = [
        'src.agents',
        'src.communication',
        'src.llm',
        'src.memory',
        'src.pipeline',
        'src.core',
        'src.gui'
    ]

    for module_name in felix_modules:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(level)
        module_logger.propagate = True  # Propagate to root logger


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
