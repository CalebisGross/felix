"""
Backend components for Streamlit GUI.

Provides read-only monitoring and database access for the Felix system.
"""

from .system_monitor import SystemMonitor
from .db_reader import DatabaseReader

__all__ = ["SystemMonitor", "DatabaseReader"]