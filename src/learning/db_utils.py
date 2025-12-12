"""
Database utilities for learning systems.

Provides:
- Connection management with WAL mode
- Retry logic for transient database locking
- Shared error handling
"""

import sqlite3
import time
import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Any

logger = logging.getLogger(__name__)


def get_connection_with_wal(db_path: Path, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Get database connection with WAL mode and proper timeout.

    WAL (Write-Ahead Logging) allows concurrent readers and one writer,
    significantly reducing database locking issues.

    Args:
        db_path: Path to SQLite database
        timeout: Connection timeout in seconds (default: 30.0)

    Returns:
        SQLite connection with WAL mode enabled
    """
    conn = sqlite3.connect(db_path, timeout=timeout)

    # Enable WAL mode for better concurrency
    # WAL allows multiple readers + 1 writer simultaneously
    conn.execute("PRAGMA journal_mode=WAL")

    # Set busy timeout (milliseconds) - how long to wait when database is locked
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")

    # Enable foreign key constraints (good practice)
    conn.execute("PRAGMA foreign_keys=ON")

    return conn


def retry_on_locked(max_attempts: int = 3, base_delay: float = 0.1) -> Callable:
    """
    Decorator to retry database operations on transient lock errors.

    Uses exponential backoff: delay * (attempt + 1)
    - Attempt 1: 0.1s delay
    - Attempt 2: 0.2s delay
    - Attempt 3: 0.3s delay

    Args:
        max_attempts: Maximum retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 0.1)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_on_locked(max_attempts=3)
        def record_data(self, ...):
            conn = self._get_connection()
            conn.execute(...)
            conn.commit()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except sqlite3.OperationalError as e:
                    last_exception = e
                    error_msg = str(e).lower()

                    # Only retry on database locked errors
                    if "locked" in error_msg or "busy" in error_msg:
                        if attempt < max_attempts - 1:
                            delay = base_delay * (attempt + 1)
                            logger.warning(
                                f"Database locked in {func.__name__}, "
                                f"retry {attempt + 1}/{max_attempts} after {delay:.1f}s"
                            )
                            time.sleep(delay)
                            continue

                    # Not a lock error or out of retries
                    raise

            # All retries exhausted
            logger.error(
                f"Database operation {func.__name__} failed after {max_attempts} attempts: "
                f"{last_exception}"
            )
            raise last_exception

        return wrapper
    return decorator


def safe_execute(conn: sqlite3.Connection, query: str, params: tuple = (),
                 max_attempts: int = 3) -> sqlite3.Cursor:
    """
    Execute SQL with automatic retry on lock errors.

    Args:
        conn: SQLite connection
        query: SQL query to execute
        params: Query parameters (default: empty tuple)
        max_attempts: Maximum retry attempts (default: 3)

    Returns:
        Cursor with query results

    Raises:
        sqlite3.Error: If query fails after all retries
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return conn.execute(query, params)

        except sqlite3.OperationalError as e:
            last_exception = e
            error_msg = str(e).lower()

            if ("locked" in error_msg or "busy" in error_msg) and attempt < max_attempts - 1:
                delay = 0.1 * (attempt + 1)
                logger.debug(f"Database locked, retry {attempt + 1}/{max_attempts} after {delay:.1f}s")
                time.sleep(delay)
                continue

            raise

    raise last_exception


def safe_commit(conn: sqlite3.Connection, max_attempts: int = 3) -> bool:
    """
    Commit transaction with automatic retry on lock errors.

    Args:
        conn: SQLite connection
        max_attempts: Maximum retry attempts (default: 3)

    Returns:
        True if commit succeeded, False otherwise
    """
    for attempt in range(max_attempts):
        try:
            conn.commit()
            return True

        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()

            if ("locked" in error_msg or "busy" in error_msg) and attempt < max_attempts - 1:
                delay = 0.1 * (attempt + 1)
                logger.debug(f"Commit locked, retry {attempt + 1}/{max_attempts} after {delay:.1f}s")
                time.sleep(delay)
                continue

            logger.error(f"Failed to commit transaction: {e}")
            return False

    return False


class TransactionContext:
    """
    Context manager for database transactions with automatic retry.

    Example:
        with TransactionContext(db_path) as conn:
            conn.execute("INSERT INTO ...")
            # Automatically commits on success, rolls back on error
    """

    def __init__(self, db_path: Path, timeout: float = 30.0, max_retries: int = 3):
        self.db_path = db_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.conn = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = get_connection_with_wal(self.db_path, self.timeout)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # No exception - try to commit
            safe_commit(self.conn, self.max_retries)
        else:
            # Exception occurred - rollback
            try:
                self.conn.rollback()
            except Exception as e:
                logger.warning(f"Failed to rollback transaction: {e}")

        # Close connection
        try:
            self.conn.close()
        except Exception as e:
            logger.warning(f"Failed to close connection: {e}")

        # Don't suppress exceptions
        return False


def ensure_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """
    Check if table exists in database.

    Args:
        conn: SQLite connection
        table_name: Name of table to check

    Returns:
        True if table exists, False otherwise
    """
    try:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        return cursor.fetchone() is not None
    except sqlite3.Error as e:
        logger.error(f"Failed to check table existence: {e}")
        return False


def get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """
    Get number of rows in table.

    Args:
        conn: SQLite connection
        table_name: Name of table

    Returns:
        Row count, or -1 if error
    """
    try:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    except sqlite3.Error as e:
        logger.error(f"Failed to get row count for {table_name}: {e}")
        return -1
