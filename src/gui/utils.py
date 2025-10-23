import threading
import queue
import sqlite3
import logging
import os
from tkinter import messagebox

class ThreadManager:
    def __init__(self, root):
        self.root = root
        self.threads = []
        self.queue = queue.Queue()

    def start_thread(self, target, args=()):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        self.threads.append(thread)
        self.poll_queue()

    def poll_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == 'update_gui':
                    # Placeholder for GUI updates
                    pass
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)

    def join_threads(self):
        for thread in self.threads:
            thread.join(timeout=1.0)

class DBHelper:
    def __init__(self):
        self.memory_db = 'felix_memory.db'
        self.knowledge_db = 'felix_knowledge.db'
        self.lock = threading.Lock()
        try:
            from src.memory import knowledge_store, task_memory
            self.ks = knowledge_store.KnowledgeStore(self.knowledge_db)
            self.tm = task_memory.TaskMemory(self.memory_db)
        except ImportError:
            self.ks = None
            self.tm = None

    def connect(self, db_name):
        return sqlite3.connect(db_name)

    def query(self, db_name, sql, params=()):
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
        """Get list of table names in the database."""
        try:
            results = self.query(db_name, "SELECT name FROM sqlite_master WHERE type='table';")
            return [row[0] for row in results]
        except sqlite3.Error as e:
            logger.warning(f"Failed to get table names from {db_name}: {e}")
            return []

# Logging setup
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

logger = logging.getLogger('felix_gui')
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