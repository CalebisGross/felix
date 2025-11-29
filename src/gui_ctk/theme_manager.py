"""
Theme Manager for Felix GUI (CustomTkinter Edition)

CustomTkinter handles most theming natively. This module provides:
- Easy dark/light mode switching
- Color constants for custom widgets
- TTK style configuration for TreeView compatibility
"""

import customtkinter as ctk
from tkinter import ttk
from typing import Dict, Callable, List
import logging

logger = logging.getLogger(__name__)


# Color constants for custom widgets (matches CTk dark/light modes)
COLORS = {
    "dark": {
        "bg_primary": "#1a1a1a",
        "bg_secondary": "#242424",
        "bg_tertiary": "#2b2b2b",
        "bg_hover": "#333333",
        "fg_primary": "#ffffff",
        "fg_secondary": "#b3b3b3",
        "fg_muted": "#808080",
        "accent": "#1f6aa5",
        "accent_hover": "#144870",
        "success": "#2fa572",
        "warning": "#d97706",
        "error": "#dc2626",
        "border": "#3f3f3f",
        "selection": "#1f6aa5",
    },
    "light": {
        "bg_primary": "#f9f9f9",
        "bg_secondary": "#ebebeb",
        "bg_tertiary": "#dbdbdb",
        "bg_hover": "#d1d1d1",
        "fg_primary": "#1a1a1a",
        "fg_secondary": "#4a4a4a",
        "fg_muted": "#7a7a7a",
        "accent": "#1f6aa5",
        "accent_hover": "#144870",
        "success": "#2fa572",
        "warning": "#d97706",
        "error": "#dc2626",
        "border": "#c7c7c7",
        "selection": "#1f6aa5",
    }
}


class ThemeManager:
    """
    Manages application themes using CustomTkinter's native theming.

    Unlike the original Tkinter version, most theming is handled automatically
    by CTk. This class primarily provides:
    - Easy toggle between dark/light modes
    - TTK widget styling (TreeView, etc.)
    - Color constants access
    - Theme change callbacks
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one ThemeManager per app."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, root=None):
        """
        Initialize ThemeManager.

        Args:
            root: The main CTk window (optional, used for TTK styling)
        """
        if self._initialized:
            return

        self.root = root
        self._callbacks: List[Callable] = []
        self._current_mode = "dark"  # Default to dark mode

        # Initialize CustomTkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Configure TTK styles for TreeView compatibility
        if root:
            self._configure_ttk_styles()

        self._initialized = True
        logger.info("ThemeManager initialized (dark mode)")

    def set_mode(self, mode: str):
        """
        Set the appearance mode.

        Args:
            mode: "dark", "light", or "system"
        """
        self._current_mode = mode if mode in ("dark", "light") else "dark"
        ctk.set_appearance_mode(mode)

        # Update TTK styles
        self._configure_ttk_styles()

        # Notify callbacks
        self._notify_callbacks()

        logger.info(f"Theme changed to: {mode}")

    def toggle_mode(self):
        """Toggle between dark and light modes."""
        new_mode = "light" if self._current_mode == "dark" else "dark"
        self.set_mode(new_mode)

    def is_dark_mode(self) -> bool:
        """Check if dark mode is currently active."""
        return ctk.get_appearance_mode().lower() == "dark"

    @property
    def mode(self) -> str:
        """Get current appearance mode."""
        return ctk.get_appearance_mode().lower()

    @property
    def colors(self) -> Dict[str, str]:
        """Get color dictionary for current mode."""
        return COLORS.get(self.mode, COLORS["dark"])

    def get_color(self, name: str) -> str:
        """
        Get a specific color for current mode.

        Args:
            name: Color name (e.g., "bg_primary", "accent")

        Returns:
            Hex color string
        """
        return self.colors.get(name, "#000000")

    def register_callback(self, callback: Callable):
        """
        Register a callback to be called when theme changes.

        Args:
            callback: Function to call (receives mode string)
        """
        self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks of theme change."""
        for callback in self._callbacks:
            try:
                callback(self._current_mode)
            except Exception as e:
                logger.error(f"Error in theme callback: {e}")

    def _configure_ttk_styles(self):
        """Configure TTK styles for TreeView and other ttk widgets."""
        style = ttk.Style()

        # Use 'clam' theme as base (works well with custom colors)
        try:
            style.theme_use('clam')
        except Exception:
            pass  # Theme might not be available

        colors = self.colors

        # Treeview styling
        style.configure("Treeview",
                        background=colors["bg_primary"],
                        foreground=colors["fg_primary"],
                        fieldbackground=colors["bg_primary"],
                        borderwidth=0,
                        rowheight=28)

        style.configure("Treeview.Heading",
                        background=colors["bg_secondary"],
                        foreground=colors["fg_primary"],
                        borderwidth=1,
                        relief="flat")

        style.map("Treeview",
                  background=[("selected", colors["selection"])],
                  foreground=[("selected", "#ffffff")])

        style.map("Treeview.Heading",
                  background=[("active", colors["bg_tertiary"])])

        # TFrame styling (for any remaining ttk.Frame usage)
        style.configure("TFrame", background=colors["bg_primary"])

        # TLabel styling
        style.configure("TLabel",
                        background=colors["bg_primary"],
                        foreground=colors["fg_primary"])

        # TButton styling
        style.configure("TButton",
                        background=colors["bg_secondary"],
                        foreground=colors["fg_primary"])

        # TSeparator styling
        style.configure("TSeparator", background=colors["border"])

        logger.debug(f"TTK styles configured for {self.mode} mode")


def get_theme_manager() -> ThemeManager:
    """Get the singleton ThemeManager instance."""
    return ThemeManager()
