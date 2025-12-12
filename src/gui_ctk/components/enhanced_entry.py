"""
Enhanced Entry Component with Focus and Error States

Features:
- Default state: Standard border
- Focus state: Accent border with subtle glow (increased border width)
- Error state: Red border with error message below
- Disabled state: Muted appearance
- Optional label above entry
- Placeholder text support
"""

import customtkinter as ctk
from typing import Optional
from ..theme_manager import ThemeManager, get_theme_manager
from ..styles import INPUT_MD, RADIUS_SM, SPACE_XS, FONT_BODY, FONT_CAPTION


class EnhancedEntry(ctk.CTkFrame):
    """
    Enhanced text entry with focus states and error handling.

    Args:
        parent: Parent widget
        label: Optional label text above entry
        placeholder: Placeholder text
        error_message: Optional error message (shows red border)
        width: Entry width in pixels
        disabled: Whether entry is disabled
        **kwargs: Additional arguments passed to CTkEntry
    """

    def __init__(
        self,
        parent,
        label: Optional[str] = None,
        placeholder: str = "",
        error_message: Optional[str] = None,
        width: int = INPUT_MD,
        disabled: bool = False,
        **kwargs
    ):
        self.theme_manager = get_theme_manager()
        self._label_text = label
        self._error_message = error_message
        self._disabled = disabled
        self._is_focused = False

        # Initialize frame (transparent container)
        super().__init__(parent, fg_color="transparent")

        # Create label if provided
        if label:
            self.label = ctk.CTkLabel(
                self,
                text=label,
                font=ctk.CTkFont(size=FONT_BODY),
                text_color=self.theme_manager.colors["fg_primary"],
                anchor="w"
            )
            self.label.pack(fill="x", pady=(0, SPACE_XS))
        else:
            self.label = None

        # Create entry with initial styling
        border_color = self._get_border_color()
        border_width = self._get_border_width()

        self.entry = ctk.CTkEntry(
            self,
            width=width,
            placeholder_text=placeholder,
            corner_radius=RADIUS_SM,
            border_width=border_width,
            border_color=border_color,
            fg_color=self.theme_manager.colors["bg_secondary"],
            text_color=self.theme_manager.colors["fg_primary"],
            state="disabled" if disabled else "normal",
            **kwargs
        )
        self.entry.pack(fill="x")

        # Create error message label (hidden by default)
        self.error_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.colors["error"],
            anchor="w"
        )
        if error_message:
            self.error_label.configure(text=error_message)
            self.error_label.pack(fill="x", pady=(SPACE_XS, 0))

        # Bind focus events
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _get_border_color(self) -> str:
        """Get border color based on current state."""
        colors = self.theme_manager.colors

        if self._error_message:
            # Error state: red border
            return colors["error"]
        elif self._is_focused and not self._disabled:
            # Focus state: accent border
            return colors["accent"]
        else:
            # Default state: standard border
            return colors["border"]

    def _get_border_width(self) -> int:
        """Get border width based on current state."""
        if self._error_message:
            # Error state: thicker border
            return 2
        elif self._is_focused and not self._disabled:
            # Focus state: glow effect via thicker border
            return 2
        else:
            # Default state
            return 1

    def _update_appearance(self):
        """Update entry appearance based on current state."""
        border_color = self._get_border_color()
        border_width = self._get_border_width()
        colors = self.theme_manager.colors

        # Update entry styling
        self.entry.configure(
            border_color=border_color,
            border_width=border_width,
            fg_color=colors["bg_secondary"] if not self._disabled else colors["bg_tertiary"],
            text_color=colors["fg_primary"] if not self._disabled else colors["fg_muted"],
            state="disabled" if self._disabled else "normal"
        )

        # Update label color
        if self.label:
            self.label.configure(
                text_color=colors["fg_primary"] if not self._disabled else colors["fg_muted"]
            )

        # Update error message
        if self._error_message:
            self.error_label.configure(text=self._error_message)
            if not self.error_label.winfo_manager():
                self.error_label.pack(fill="x", pady=(SPACE_XS, 0))
        else:
            self.error_label.pack_forget()

    def _on_focus_in(self, event):
        """Handle focus gained."""
        if not self._disabled:
            self._is_focused = True
            self._update_appearance()

    def _on_focus_out(self, event):
        """Handle focus lost."""
        self._is_focused = False
        self._update_appearance()

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        self._update_appearance()

    def set_error(self, error_message: Optional[str] = None):
        """Set error message (None to clear error state)."""
        self._error_message = error_message
        self._update_appearance()

    def set_disabled(self, disabled: bool):
        """Set disabled state."""
        self._disabled = disabled
        self._update_appearance()

    def get(self) -> str:
        """Get entry value."""
        return self.entry.get()

    def set(self, value: str):
        """Set entry value."""
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

    def clear(self):
        """Clear entry value."""
        self.entry.delete(0, "end")

    def destroy(self):
        """Clean up on destruction."""
        self.theme_manager.unregister_callback(self._on_theme_change)
        super().destroy()
