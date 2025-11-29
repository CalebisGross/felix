"""
Search Entry component for Felix GUI.

A search input with placeholder text and clear button.
"""

import customtkinter as ctk
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SearchEntry(ctk.CTkFrame):
    """
    A search entry component with placeholder and clear button.
    """

    def __init__(
        self,
        master,
        placeholder: str = "Search...",
        on_search: Optional[Callable[[str], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        width: int = 200,
        **kwargs
    ):
        """
        Initialize SearchEntry.

        Args:
            master: Parent widget
            placeholder: Placeholder text
            on_search: Callback when search is triggered (Enter key or typing)
            on_clear: Callback when clear button is clicked
            width: Entry width
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, fg_color="transparent", **kwargs)

        self.on_search = on_search
        self.on_clear = on_clear
        self._placeholder = placeholder

        # Search entry
        self.entry = ctk.CTkEntry(
            self,
            placeholder_text=placeholder,
            width=width
        )
        self.entry.pack(side="left", fill="x", expand=True)

        # Clear button
        self.clear_button = ctk.CTkButton(
            self,
            text="X",
            width=28,
            height=28,
            command=self._clear,
            fg_color="transparent",
            hover_color=("gray70", "gray30")
        )
        self.clear_button.pack(side="left", padx=(5, 0))

        # Bind events
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<KeyRelease>", self._on_key_release)

    def _on_enter(self, event):
        """Handle Enter key press."""
        if self.on_search:
            self.on_search(self.get())

    def _on_key_release(self, event):
        """Handle key release (for live search)."""
        # Could implement live search here if needed
        pass

    def _clear(self):
        """Clear the entry and trigger callback."""
        self.entry.delete(0, "end")
        if self.on_clear:
            self.on_clear()
        if self.on_search:
            self.on_search("")

    def get(self) -> str:
        """Get the current entry value."""
        return self.entry.get()

    def set(self, value: str):
        """Set the entry value."""
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

    def focus(self):
        """Focus the entry."""
        self.entry.focus()
