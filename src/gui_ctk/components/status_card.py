"""
Status Card component for Felix GUI.

A modern card-style widget for displaying status information with
icon, title, value, and optional subtitle.
"""

import customtkinter as ctk
from typing import Optional
import logging

from ..theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class StatusCard(ctk.CTkFrame):
    """
    A modern status card component.

    Displays:
    - Title (small, muted)
    - Value (large, prominent)
    - Optional subtitle (small, muted)
    - Optional status indicator color
    """

    def __init__(
        self,
        master,
        title: str,
        value: str = "--",
        subtitle: str = "",
        status_color: Optional[str] = None,
        width: int = 150,
        **kwargs
    ):
        """
        Initialize StatusCard.

        Args:
            master: Parent widget
            title: Card title text
            value: Main value to display
            subtitle: Optional subtitle text
            status_color: Optional color for status indicator
            width: Card width
            **kwargs: Additional arguments passed to CTkFrame
        """
        # Set default appearance
        kwargs.setdefault("corner_radius", 10)

        super().__init__(master, width=width, **kwargs)

        self.theme_manager = get_theme_manager()

        # Prevent frame from shrinking
        self.grid_propagate(False)
        self.pack_propagate(False)

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Status indicator (optional colored bar at top)
        if status_color:
            self.status_bar = ctk.CTkFrame(
                self,
                height=4,
                corner_radius=2,
                fg_color=status_color
            )
            self.status_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        else:
            self.status_bar = None

        # Title label
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.title_label.grid(row=1, column=0, sticky="w", padx=15, pady=(10 if not status_color else 5, 0))

        # Value label
        self.value_label = ctk.CTkLabel(
            self,
            text=value,
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.value_label.grid(row=2, column=0, sticky="w", padx=15, pady=2)

        # Subtitle label
        self.subtitle_label = ctk.CTkLabel(
            self,
            text=subtitle,
            font=ctk.CTkFont(size=10),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.subtitle_label.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 10))

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event):
        """Clean up when widget is destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        self.title_label.configure(text_color=self.theme_manager.get_color("fg_muted"))
        self.subtitle_label.configure(text_color=self.theme_manager.get_color("fg_muted"))

    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.configure(text=value)

    def set_subtitle(self, subtitle: str):
        """Update the subtitle text."""
        self.subtitle_label.configure(text=subtitle)

    def set_title(self, title: str):
        """Update the title text."""
        self.title_label.configure(text=title)

    def set_status_color(self, color: str):
        """Update the status indicator color."""
        if self.status_bar:
            self.status_bar.configure(fg_color=color)
