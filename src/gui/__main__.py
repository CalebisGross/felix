"""
Entry point for running Felix GUI as a module.

Usage:
    python -m src.gui

This allows running the GUI without the RuntimeWarning about module imports.
"""

from .main import MainApp

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
