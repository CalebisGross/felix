"""
Themed TreeView component for Felix GUI.

CustomTkinter doesn't have a TreeView equivalent, so we use ttk.Treeview
with custom styling to match the CTk dark/light modes.
"""

import customtkinter as ctk
from tkinter import ttk
import tkinter as tk
from typing import List, Tuple, Optional, Callable
import logging

from ..theme_manager import get_theme_manager
from ..styles import TREEVIEW_ROW_HEIGHT

logger = logging.getLogger(__name__)


class ThemedTreeview(ctk.CTkFrame):
    """
    A themed TreeView component that integrates with CustomTkinter.

    Wraps ttk.Treeview in a CTkFrame with automatic dark/light mode styling
    and built-in scrollbars.
    """

    def __init__(
        self,
        master,
        columns: List[str],
        headings: Optional[List[str]] = None,
        widths: Optional[List[int]] = None,
        show_tree: bool = False,
        height: int = 10,
        selectmode: str = "browse",
        **kwargs
    ):
        """
        Initialize ThemedTreeview.

        Args:
            master: Parent widget
            columns: List of column identifiers
            headings: List of column heading texts (defaults to column ids)
            widths: List of column widths (defaults to 100)
            show_tree: Whether to show tree column (#0)
            height: Number of visible rows
            selectmode: Selection mode ("browse", "extended", "none")
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.columns = columns
        self.headings = headings or columns
        self.widths = widths or [100] * len(columns)

        # Get theme manager
        self.theme_manager = get_theme_manager()

        # Create container frame for treeview and scrollbars
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)

        # Create treeview
        show_value = "tree headings" if show_tree else "headings"
        self.tree = ttk.Treeview(
            self.container,
            columns=columns,
            show=show_value,
            height=height,
            selectmode=selectmode
        )

        # Configure columns
        for i, (col, heading, width) in enumerate(zip(columns, self.headings, self.widths)):
            self.tree.column(col, width=width, minwidth=50)
            self.tree.heading(col, text=heading, anchor="w")

        # Hide tree column if not showing tree
        if not show_tree:
            self.tree.column("#0", width=0, stretch=False)

        # Create scrollbars
        self.vsb = ctk.CTkScrollbar(self.container, orientation="vertical", command=self.tree.yview)
        self.hsb = ctk.CTkScrollbar(self.container, orientation="horizontal", command=self.tree.xview)

        self.tree.configure(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb.grid(row=1, column=0, sticky="ew")

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Apply theme
        self._apply_theme()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

        # Bind cleanup
        self.bind("<Destroy>", self._on_destroy)

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
        """Handle theme change."""
        self._apply_theme()

    def _apply_theme(self):
        """Apply current theme to the treeview."""
        colors = self.theme_manager.colors
        style = ttk.Style()

        # Configure treeview style
        style.configure("Treeview",
                        background=colors["bg_primary"],
                        foreground=colors["fg_primary"],
                        fieldbackground=colors["bg_primary"],
                        borderwidth=0,
                        rowheight=TREEVIEW_ROW_HEIGHT)

        style.configure("Treeview.Heading",
                        background=colors["bg_secondary"],
                        foreground=colors["fg_primary"],
                        borderwidth=1,
                        relief="flat")

        style.map("Treeview",
                  background=[("selected", colors["selection"])],
                  foreground=[("selected", "#ffffff")])

    # TreeView method proxies
    def insert(self, parent: str = "", index: str = "end", iid: str = None,
               values: Tuple = None, **kwargs) -> str:
        """Insert an item into the treeview."""
        return self.tree.insert(parent, index, iid=iid, values=values, **kwargs)

    def delete(self, *items):
        """Delete items from the treeview."""
        self.tree.delete(*items)

    def clear(self):
        """Clear all items from the treeview."""
        for item in self.tree.get_children():
            self.tree.delete(item)

    def get_children(self, item: str = "") -> Tuple[str, ...]:
        """Get child items."""
        return self.tree.get_children(item)

    def selection(self) -> Tuple[str, ...]:
        """Get selected items."""
        return self.tree.selection()

    def selection_set(self, *items):
        """Set selection."""
        self.tree.selection_set(*items)

    def item(self, item: str, option: str = None, **kwargs):
        """Query or modify item options."""
        return self.tree.item(item, option, **kwargs)

    def set(self, item: str, column: str = None, value=None):
        """Set item column value."""
        return self.tree.set(item, column, value)

    def focus(self, item: str = None):
        """Set or get focused item."""
        return self.tree.focus(item)

    def see(self, item: str):
        """Scroll to make item visible."""
        self.tree.see(item)

    def bind_tree(self, event: str, callback: Callable):
        """Bind an event to the treeview."""
        self.tree.bind(event, callback)

    def heading(self, column: str, **kwargs):
        """Configure column heading."""
        return self.tree.heading(column, **kwargs)

    def column(self, column: str, **kwargs):
        """Configure column."""
        return self.tree.column(column, **kwargs)

    def tag_configure(self, tag: str, **kwargs):
        """Configure a tag."""
        self.tree.tag_configure(tag, **kwargs)

    def identify_row(self, y: int) -> str:
        """Identify row at y coordinate."""
        return self.tree.identify_row(y)

    def identify_column(self, x: int) -> str:
        """Identify column at x coordinate."""
        return self.tree.identify_column(x)

    def bbox(self, item: str, column: str = None):
        """Get bounding box of item."""
        return self.tree.bbox(item, column)
