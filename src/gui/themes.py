import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


class ColorScheme:
    """Defines a color scheme for the application."""

    def __init__(self, name: str, colors: Dict[str, str]):
        self.name = name
        self.colors = colors

    def __getitem__(self, key: str) -> str:
        return self.colors.get(key, "#000000")


# Light theme color scheme
LIGHT_THEME = ColorScheme("light", {
    # General backgrounds
    "bg_primary": "#ffffff",
    "bg_secondary": "#f5f5f5",
    "bg_tertiary": "#e8e8e8",

    # Text colors
    "fg_primary": "#000000",
    "fg_secondary": "#333333",
    "fg_tertiary": "#666666",
    "fg_disabled": "#999999",

    # Text widget colors
    "text_bg": "#ffffff",
    "text_fg": "#000000",
    "text_select_bg": "#0078d7",
    "text_select_fg": "#ffffff",
    "text_insert": "#000000",

    # Status colors
    "status_info": "#0066cc",
    "status_success": "#008000",
    "status_warning": "#ff8800",
    "status_error": "#cc0000",

    # Border and frame colors
    "border": "#cccccc",
    "highlight": "#0078d7",

    # Button colors (for non-ttk buttons)
    "button_bg": "#f0f0f0",
    "button_fg": "#000000",
    "button_active_bg": "#e0e0e0",
})

# Dark theme color scheme
DARK_THEME = ColorScheme("dark", {
    # General backgrounds
    "bg_primary": "#1e1e1e",
    "bg_secondary": "#252526",
    "bg_tertiary": "#2d2d30",

    # Text colors
    "fg_primary": "#cccccc",
    "fg_secondary": "#bbbbbb",
    "fg_tertiary": "#999999",
    "fg_disabled": "#666666",

    # Text widget colors
    "text_bg": "#1e1e1e",
    "text_fg": "#d4d4d4",
    "text_select_bg": "#264f78",
    "text_select_fg": "#ffffff",
    "text_insert": "#ffffff",

    # Status colors
    "status_info": "#4fc3f7",
    "status_success": "#4caf50",
    "status_warning": "#ff9800",
    "status_error": "#f44336",

    # Border and frame colors
    "border": "#3e3e42",
    "highlight": "#007acc",

    # Button colors (for non-ttk buttons)
    "button_bg": "#2d2d30",
    "button_fg": "#cccccc",
    "button_active_bg": "#3e3e42",
})


class ThemeManager:
    """Manages application themes and provides methods to apply them."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_theme = LIGHT_THEME
        self.theme_change_callbacks: List[Callable] = []

        # Store references to widgets for dynamic updates
        self.registered_widgets: List[tk.Widget] = []

    def get_current_theme(self) -> ColorScheme:
        """Get the currently active theme."""
        return self.current_theme

    def is_dark_mode(self) -> bool:
        """Check if dark mode is currently active."""
        return self.current_theme.name == "dark"

    def set_theme(self, theme_name: str):
        """Set the active theme by name ('light' or 'dark')."""
        if theme_name == "dark":
            self.current_theme = DARK_THEME
        else:
            self.current_theme = LIGHT_THEME

        self._apply_theme()
        self._notify_callbacks()
        logger.info(f"Theme changed to: {theme_name}")

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        if self.is_dark_mode():
            self.set_theme("light")
        else:
            self.set_theme("dark")

    def register_callback(self, callback: Callable):
        """Register a callback to be called when theme changes."""
        self.theme_change_callbacks.append(callback)

    def register_widget(self, widget: tk.Widget):
        """Register a widget for theme updates."""
        if widget not in self.registered_widgets:
            self.registered_widgets.append(widget)

    def _notify_callbacks(self):
        """Notify all registered callbacks of theme change."""
        for callback in self.theme_change_callbacks:
            try:
                callback(self.current_theme)
            except Exception as e:
                logger.error(f"Error in theme callback: {e}")

    def _apply_theme(self):
        """Apply the current theme to the root window and ttk styles."""
        theme = self.current_theme

        # Configure ttk styles
        style = ttk.Style()

        # TFrame
        style.configure("TFrame", background=theme["bg_primary"])

        # TLabel
        style.configure("TLabel",
                       background=theme["bg_primary"],
                       foreground=theme["fg_primary"])

        # TButton
        style.configure("TButton",
                       background=theme["button_bg"],
                       foreground=theme["fg_primary"])

        # TEntry
        style.configure("TEntry",
                       fieldbackground=theme["text_bg"],
                       foreground=theme["text_fg"],
                       insertcolor=theme["text_insert"])

        # TNotebook
        style.configure("TNotebook", background=theme["bg_primary"])
        style.configure("TNotebook.Tab",
                       background=theme["bg_secondary"],
                       foreground=theme["fg_primary"])
        style.map("TNotebook.Tab",
                 background=[("selected", theme["bg_primary"])],
                 foreground=[("selected", theme["fg_primary"])])

        # TCheckbutton
        style.configure("TCheckbutton",
                       background=theme["bg_primary"],
                       foreground=theme["fg_primary"])

        # TCombobox
        style.configure("TCombobox",
                       fieldbackground=theme["text_bg"],
                       foreground=theme["text_fg"],
                       selectbackground=theme["text_select_bg"],
                       selectforeground=theme["text_select_fg"])

        # Treeview (for tables)
        style.configure("Treeview",
                       background=theme["text_bg"],
                       foreground=theme["text_fg"],
                       fieldbackground=theme["text_bg"])
        style.map("Treeview",
                 background=[("selected", theme["text_select_bg"])],
                 foreground=[("selected", theme["text_select_fg"])])

        # TSeparator
        style.configure("TSeparator", background=theme["border"])

        # Configure root window
        self.root.configure(bg=theme["bg_primary"])

    def apply_to_text_widget(self, widget: tk.Text):
        """Apply theme to a Text widget."""
        theme = self.current_theme
        widget.configure(
            bg=theme["text_bg"],
            fg=theme["text_fg"],
            insertbackground=theme["text_insert"],
            selectbackground=theme["text_select_bg"],
            selectforeground=theme["text_select_fg"]
        )

    def apply_to_label(self, widget: tk.Label, color_type: str = "primary"):
        """Apply theme to a Label widget."""
        theme = self.current_theme

        # Get foreground color based on type
        fg_color = theme[f"fg_{color_type}"]

        widget.configure(
            bg=theme["bg_primary"],
            fg=fg_color
        )

    def apply_to_frame(self, widget: tk.Frame, bg_type: str = "primary"):
        """Apply theme to a Frame widget."""
        theme = self.current_theme
        widget.configure(bg=theme[f"bg_{bg_type}"])

    def apply_to_button(self, widget: tk.Button):
        """Apply theme to a Button widget (non-ttk)."""
        theme = self.current_theme
        widget.configure(
            bg=theme["button_bg"],
            fg=theme["button_fg"],
            activebackground=theme["button_active_bg"],
            activeforeground=theme["fg_primary"]
        )

    def get_status_color(self, status_type: str) -> str:
        """Get color for status messages (info, success, warning, error)."""
        return self.current_theme[f"status_{status_type}"]

    def apply_to_all_children(self, widget: tk.Widget):
        """Recursively apply theme to a widget and all its children."""
        theme = self.current_theme

        # Apply to current widget
        widget_type = widget.winfo_class()

        if widget_type == "Text":
            self.apply_to_text_widget(widget)
        elif widget_type == "Label":
            widget.configure(bg=theme["bg_primary"], fg=theme["fg_primary"])
        elif widget_type == "Frame":
            widget.configure(bg=theme["bg_primary"])
        elif widget_type == "Button":
            self.apply_to_button(widget)

        # Recursively apply to children
        for child in widget.winfo_children():
            try:
                self.apply_to_all_children(child)
            except Exception as e:
                # Some widgets may not support configuration
                pass
