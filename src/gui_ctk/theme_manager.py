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


# Color constants for custom widgets - Felix Modern Design System Palette
COLORS = {
    "dark": {
        # Background: Obsidian Black - Main app canvas, dark mode base
        "bg_primary": "#121212",
        # Surface: Slate Gray - Cards, panels, secondary backgrounds
        "bg_secondary": "#2A2A2A",
        # Primary: Deep Indigo - Headers, navigation, agent indicators
        "bg_tertiary": "#1F2833",
        "bg_hover": "#3A3A3A",
        # Text Primary: Soft Silver - Main body text
        "fg_primary": "#C5C6C7",
        # Text Secondary: Muted Ash - Subtle labels
        "fg_secondary": "#969696",
        "fg_muted": "#808080",
        # Accent 1: Helix Blue - Interactive elements, buttons
        "accent": "#0C7BDC",
        "accent_hover": "#0A66B8",
        # Accent 2: Elegant Teal - Highlights, success states
        "success": "#45A29E",
        "warning": "#d97706",
        # Warning/Error: Subdued Crimson - Alerts, BLOCKED indicators
        "error": "#A94442",
        "border": "#3f3f3f",
        "selection": "#0C7BDC",
    },
    "light": {
        # Light mode uses inverted palette with appropriate contrast
        "bg_primary": "#FFFFFF",
        "bg_secondary": "#F5F5F5",
        "bg_tertiary": "#E8E9ED",
        "bg_hover": "#E0E0E0",
        "fg_primary": "#1F2833",
        "fg_secondary": "#4A4A4A",
        "fg_muted": "#7A7A7A",
        # Keep accent colors consistent across modes
        "accent": "#0C7BDC",
        "accent_hover": "#0A66B8",
        "success": "#45A29E",
        "warning": "#d97706",
        "error": "#A94442",
        "border": "#C7C7C7",
        "selection": "#0C7BDC",
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
