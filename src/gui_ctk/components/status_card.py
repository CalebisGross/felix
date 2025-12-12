"""
Status Card component for Felix GUI.

A modern card-style widget for displaying status information with
icon, title, value, and optional subtitle.

Features:
- Width-responsive layout modes (compact vs expanded)
- Hover elevation effect with border highlight
- Design token integration
"""

import customtkinter as ctk
from typing import Optional
import logging

from ..theme_manager import get_theme_manager
from ..styles import (
    FONT_CAPTION, FONT_DISPLAY, FONT_SMALL, CARD_MD,
    SPACE_XS, SPACE_SM, SPACE_MD, RADIUS_MD, COLOR_ACCENT_SECONDARY
)

logger = logging.getLogger(__name__)


class StatusCard(ctk.CTkFrame):
    """
    A modern status card component with responsive layout and hover effects.

    Displays:
    - Title (small, muted)
    - Value (large, prominent)
    - Optional subtitle (small, muted)
    - Optional status indicator color

    Responsive Behavior:
    - Compact mode (width < 200px): Reduced padding, smaller fonts
    - Expanded mode (width >= 200px): Full padding, standard fonts
    """

    def __init__(
        self,
        master,
        title: str,
        value: str = "--",
        subtitle: str = "",
        status_color: Optional[str] = None,
        width: int = CARD_MD,
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
        # Set default appearance with design tokens
        kwargs.setdefault("corner_radius", RADIUS_MD)
        kwargs.setdefault("border_width", 1)

        super().__init__(master, width=width, **kwargs)

        self.theme_manager = get_theme_manager()
        self._status_color = status_color
        self._is_hovered = False
        self._card_width = width

        # Store original colors for hover effects
        self._default_border_color = self.theme_manager.get_color("border")
        self._hover_border_color = COLOR_ACCENT_SECONDARY
        self._default_bg_color = self.theme_manager.get_color("bg_secondary")
        self._hover_bg_color = self.theme_manager.get_color("bg_hover")

        # Apply default colors
        self.configure(
            border_color=self._default_border_color,
            fg_color=self._default_bg_color
        )

        # Prevent frame from shrinking
        self.grid_propagate(False)
        self.pack_propagate(False)

        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Determine layout mode based on width
        self._layout_mode = self._get_layout_mode(width)

        # Status indicator (optional colored bar at top)
        if status_color:
            self.status_bar = ctk.CTkFrame(
                self,
                height=4,
                corner_radius=2,
                fg_color=status_color
            )
            self.status_bar.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        else:
            self.status_bar = None

        # Title label
        title_font_size = FONT_CAPTION if self._layout_mode == "expanded" else FONT_CAPTION - 1
        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=title_font_size),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.title_label.grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_SM if not status_color else SPACE_XS, 0))

        # Value label
        value_font_size = FONT_DISPLAY if self._layout_mode == "expanded" else FONT_DISPLAY - 4
        self.value_label = ctk.CTkLabel(
            self,
            text=value,
            font=ctk.CTkFont(size=value_font_size, weight="bold")
        )
        self.value_label.grid(row=2, column=0, sticky="w", padx=SPACE_MD, pady=2)

        # Subtitle label
        subtitle_font_size = FONT_SMALL if self._layout_mode == "expanded" else FONT_SMALL - 1
        self.subtitle_label = ctk.CTkLabel(
            self,
            text=subtitle,
            font=ctk.CTkFont(size=subtitle_font_size),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.subtitle_label.grid(row=3, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_SM))

        # Bind hover events for elevation effect
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)
        self.bind("<Destroy>", self._on_destroy)

    def _get_layout_mode(self, width: int) -> str:
        """
        Determine layout mode based on width.

        Args:
            width: Card width in pixels

        Returns:
            "compact" or "expanded"
        """
        return "compact" if width < 200 else "expanded"

    def _on_enter(self, event):
        """Handle mouse enter event for hover effect."""
        self._is_hovered = True
        self.configure(
            border_color=self._hover_border_color,
            fg_color=self._hover_bg_color
        )

    def _on_leave(self, event):
        """Handle mouse leave event to remove hover effect."""
        self._is_hovered = False
        self.configure(
            border_color=self._default_border_color,
            fg_color=self._default_bg_color
        )

    def _on_destroy(self, event):
        """Clean up when widget is destroyed."""
        # Only handle our own destruction, not child widgets
        if event.widget != self:
            return
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except (ValueError, AttributeError):
            pass  # Already unregistered or theme_manager unavailable

    def _on_theme_change(self, mode: str):
        """Handle theme change and update hover colors."""
        # Update text colors
        self.title_label.configure(text_color=self.theme_manager.get_color("fg_muted"))
        self.subtitle_label.configure(text_color=self.theme_manager.get_color("fg_muted"))

        # Update stored colors for hover effects
        self._default_border_color = self.theme_manager.get_color("border")
        self._default_bg_color = self.theme_manager.get_color("bg_secondary")
        self._hover_bg_color = self.theme_manager.get_color("bg_hover")

        # Apply current state
        if self._is_hovered:
            self.configure(
                border_color=self._hover_border_color,
                fg_color=self._hover_bg_color
            )
        else:
            self.configure(
                border_color=self._default_border_color,
                fg_color=self._default_bg_color
            )

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

    def set_width(self, width: int):
        """
        Update card width and adjust layout mode if needed.

        Args:
            width: New card width in pixels
        """
        self._card_width = width
        self.configure(width=width)

        # Check if layout mode needs to change
        new_mode = self._get_layout_mode(width)
        if new_mode != self._layout_mode:
            self._layout_mode = new_mode
            self._update_layout_mode()

    def _update_layout_mode(self):
        """Update font sizes based on current layout mode."""
        if self._layout_mode == "compact":
            title_size = FONT_CAPTION - 1
            value_size = FONT_DISPLAY - 4
            subtitle_size = FONT_SMALL - 1
        else:
            title_size = FONT_CAPTION
            value_size = FONT_DISPLAY
            subtitle_size = FONT_SMALL

        self.title_label.configure(font=ctk.CTkFont(size=title_size))
        self.value_label.configure(font=ctk.CTkFont(size=value_size, weight="bold"))
        self.subtitle_label.configure(font=ctk.CTkFont(size=subtitle_size))
