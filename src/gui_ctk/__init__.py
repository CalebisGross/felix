"""
Felix GUI - CustomTkinter Edition

A modern, clean GUI for the Felix multi-agent AI framework.
"""

from .app import FelixApp
from .theme_manager import ThemeManager
from .utils import ThreadManager, DBHelper, logger

__all__ = ['FelixApp', 'ThemeManager', 'ThreadManager', 'DBHelper', 'logger']
