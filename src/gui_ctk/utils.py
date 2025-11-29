"""
Utility classes for Felix GUI.

ThreadManager: Background thread management with thread-safe GUI communication.
DBHelper: Database access with thread-safe locking.
QueueHandler: Logging handler for GUI log display.
"""

import threading
import queue
import sqlite3
import logging
import os


class ThreadManager:
    """
    Manages background threads and provides thread-safe communication with the main GUI thread.

    Background threads should NEVER call GUI methods directly. Instead, they should:
    1. Put results in their component's queue
    2. Let the component's poll method (running on main thread) update the GUI
    """

    def __init__(self, root):
        """
        Initialize ThreadManager.

        Args:
            root: The main CTk window for scheduling callbacks
        """
        self.root = root
        self.threads = []
        self._active = True

    def start_thread(self, target, args=()):
        """
        Start a daemon thread for background work.

        Args:
            target: The function to run in the thread
            args: Arguments to pass to the function
        """
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        self.threads.append(thread)

    def join_threads(self, timeout=1.0):
        """
        Wait for all threads to complete (with timeout).

        Args:
            timeout: Maximum time to wait for each thread
        """
        for thread in self.threads:
            thread.join(timeout=timeout)

    def shutdown(self):
        """Signal shutdown and join threads."""
        self._active = False
        self.join_threads()

    @property
    def is_active(self):
        """Check if thread manager is active."""
        return self._active


class DBHelper:
    """
    Thread-safe database helper for SQLite operations.
    """

    def __init__(self, memory_db='felix_memory.db', knowledge_db='felix_knowledge.db'):
        """
        Initialize DBHelper.

        Args:
            memory_db: Path to memory database
            knowledge_db: Path to knowledge database
        """
        self.memory_db = memory_db
        self.knowledge_db = knowledge_db
        self.lock = threading.Lock()

        # Try to import Felix memory stores
        self.ks = None
        self.tm = None
        try:
            from src.memory import knowledge_store, task_memory
            self.ks = knowledge_store.KnowledgeStore(self.knowledge_db)
            self.tm = task_memory.TaskMemory(self.memory_db)
        except ImportError:
            pass

    def connect(self, db_name):
        """
        Create a connection to a database.

        Args:
            db_name: Database file name

        Returns:
            sqlite3.Connection
        """
        return sqlite3.connect(db_name)

    def query(self, db_name, sql, params=()):
        """
        Execute a query and return results (thread-safe).

        Args:
            db_name: Database file name
            sql: SQL query string
            params: Query parameters

        Returns:
            List of result rows
        """
        with self.lock:
            try:
                conn = self.connect(db_name)
                cursor = conn.cursor()
                cursor.execute(sql, params)
                results = cursor.fetchall()
                conn.close()
                return results
            except sqlite3.Error as e:
                logger.warning(f"Query failed on {db_name}: {e}")
                return []

    def execute(self, db_name, sql, params=()):
        """
        Execute a statement (thread-safe).

        Args:
            db_name: Database file name
            sql: SQL statement
            params: Statement parameters
        """
        with self.lock:
            try:
                conn = self.connect(db_name)
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                logger.warning(f"Execute failed on {db_name}: {e}")

    def get_table_names(self, db_name):
        """
        Get list of table names in the database.

        Args:
            db_name: Database file name

        Returns:
            List of table names
        """
        try:
            results = self.query(db_name, "SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in results]
        except sqlite3.Error as e:
            logger.warning(f"Failed to get table names from {db_name}: {e}")
            return []


# Logging setup
log_queue = queue.Queue()


class QueueHandler(logging.Handler):
    """Logging handler that puts messages in a queue for GUI display."""

    def emit(self, record):
        """Emit a log record to the queue."""
        log_queue.put(self.format(record))


# Configure logger
logger = logging.getLogger('felix_gui_ctk')
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root to avoid duplicates

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler('felix_gui.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Queue handler for GUI
queue_handler = QueueHandler()
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

# Configure root logger to catch logs from Felix modules (src.*)
# This ensures all Felix system logs appear in the GUI dashboard
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, QueueHandler) for h in root_logger.handlers):
    root_logger.addHandler(queue_handler)


def enable_mouse_scroll(widget):
    """
    Enable mouse wheel scrolling for Treeview widgets on Linux.

    CTkTextbox and CTkScrollableFrame handle their own scrolling.
    This function only adds scroll support for ttk.Treeview which
    doesn't receive scroll events by default on Linux.

    Args:
        widget: The root widget to enable scrolling on
    """
    import platform

    # Only needed on Linux
    if platform.system() != 'Linux':
        return

    def _on_mousewheel(event):
        """Handle mouse wheel scroll for Treeview only."""
        try:
            widget_under_cursor = event.widget.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return None

        if not widget_under_cursor:
            return None

        # Try to find a Treeview in the widget hierarchy
        current = widget_under_cursor
        for _ in range(10):  # Limit search depth
            if current is None:
                break

            try:
                widget_class = current.winfo_class()
            except Exception:
                break

            # Only handle Treeview - CTk widgets handle their own scroll
            if widget_class == 'Treeview':
                try:
                    if event.num == 4:
                        current.yview_scroll(-3, "units")
                    elif event.num == 5:
                        current.yview_scroll(3, "units")
                    return "break"
                except Exception:
                    pass
                break

            # Move to parent
            try:
                current = current.master
            except Exception:
                break

        return None

    # Linux uses Button-4 (scroll up) and Button-5 (scroll down)
    widget.bind_all('<Button-4>', _on_mousewheel)
    widget.bind_all('<Button-5>', _on_mousewheel)
