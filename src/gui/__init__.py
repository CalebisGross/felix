"""Felix PySide6 GUI.

A modern, responsive GUI for the Felix multi-agent AI framework.

Usage:
    python -m src.gui

Or via entry point:
    felix-gui
"""

import sys
import logging

from PySide6.QtWidgets import QApplication

from .core.theme import apply_dark_theme
from .main_window import MainWindow

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Felix GUI."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    logger.info("Starting Felix GUI...")

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Felix")
    app.setApplicationVersion(__version__)

    # Apply dark theme
    apply_dark_theme(app)

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("Felix GUI started")

    # Run event loop
    sys.exit(app.exec())


__all__ = ["main", "MainWindow", "__version__"]
