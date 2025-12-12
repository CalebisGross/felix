"""
Resizable Separator component for Felix GUI.

A draggable pane divider for master-detail views with visual feedback.
Supports both horizontal and vertical orientations.
"""

import customtkinter as ctk
from typing import Optional, Callable, Literal
import logging

from ..theme_manager import get_theme_manager
from ..styles import SEPARATOR_HEIGHT, SEPARATOR_WIDTH, COLOR_ACCENT_SECONDARY

logger = logging.getLogger(__name__)


class ResizableSeparator(ctk.CTkFrame):
    """
    A draggable separator/divider for resizable panes.

    Features:
    - Horizontal or vertical orientation
    - Visual feedback on hover (cursor change, color highlight)
    - Callback on drag completion with new position/ratio
    - Elegant Teal accent color for drag handle
    """

    def __init__(
        self,
        master,
        orientation: Literal["horizontal", "vertical"] = "vertical",
        on_drag_complete: Optional[Callable[[float], None]] = None,
        size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize ResizableSeparator.

        Args:
            master: Parent widget
            orientation: "horizontal" or "vertical"
            on_drag_complete: Callback when drag completes, receives ratio (0.0-1.0)
            size: Separator thickness in pixels (default: 8 for vertical, 8 for horizontal)
            **kwargs: Additional arguments passed to CTkFrame
        """
        self.theme_manager = get_theme_manager()
        self.orientation = orientation
        self.on_drag_complete = on_drag_complete

        # Default size based on orientation
        if size is None:
            size = 8

        # Set frame dimensions based on orientation
        if orientation == "vertical":
            kwargs.setdefault("width", size)
            kwargs.setdefault("height", 0)  # Will expand to fill
        else:
            kwargs.setdefault("width", 0)  # Will expand to fill
            kwargs.setdefault("height", size)

        kwargs.setdefault("corner_radius", 0)
        kwargs.setdefault("fg_color", self.theme_manager.get_color("border"))

        super().__init__(master, **kwargs)

        self._size = size
        self._is_dragging = False
        self._is_hovered = False
        self._drag_start_x = 0
        self._drag_start_y = 0

        # Store colors
        self._default_color = self.theme_manager.get_color("border")
        self._hover_color = self.theme_manager.get_color("bg_hover")
        self._drag_color = COLOR_ACCENT_SECONDARY

        # Create drag handle (visible indicator in the middle)
        self._create_drag_handle()

        # Bind mouse events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

        # Also bind to drag handle
        self.drag_handle.bind("<Enter>", self._on_enter)
        self.drag_handle.bind("<Leave>", self._on_leave)
        self.drag_handle.bind("<Button-1>", self._on_press)
        self.drag_handle.bind("<B1-Motion>", self._on_drag)
        self.drag_handle.bind("<ButtonRelease-1>", self._on_release)

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)
        self.bind("<Destroy>", self._on_destroy)

    def _create_drag_handle(self):
        """Create the visual drag handle in the center."""
        if self.orientation == "vertical":
            # Vertical separator: show 3 horizontal dots/lines in the middle
            self.drag_handle = ctk.CTkFrame(
                self,
                width=self._size,
                height=30,
                corner_radius=2,
                fg_color=self._default_color
            )
            self.drag_handle.place(relx=0.5, rely=0.5, anchor="center")
        else:
            # Horizontal separator: show 3 vertical dots/lines in the middle
            self.drag_handle = ctk.CTkFrame(
                self,
                width=30,
                height=self._size,
                corner_radius=2,
                fg_color=self._default_color
            )
            self.drag_handle.place(relx=0.5, rely=0.5, anchor="center")

    def _update_cursor(self, cursor: str):
        """Update cursor for both separator and drag handle."""
        try:
            self.configure(cursor=cursor)
            self.drag_handle.configure(cursor=cursor)
        except Exception as e:
            logger.debug(f"Could not set cursor: {e}")

    def _on_enter(self, event):
        """Handle mouse enter event."""
        self._is_hovered = True

        # Change cursor based on orientation
        if self.orientation == "vertical":
            self._update_cursor("sb_h_double_arrow")
        else:
            self._update_cursor("sb_v_double_arrow")

        # Update colors
        if not self._is_dragging:
            self.configure(fg_color=self._hover_color)
            self.drag_handle.configure(fg_color=self._hover_color)

    def _on_leave(self, event):
        """Handle mouse leave event."""
        self._is_hovered = False

        # Reset cursor
        self._update_cursor("")

        # Reset colors if not dragging
        if not self._is_dragging:
            self.configure(fg_color=self._default_color)
            self.drag_handle.configure(fg_color=self._default_color)

    def _on_press(self, event):
        """Handle mouse button press to start dragging."""
        self._is_dragging = True
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

        # Apply drag color
        self.configure(fg_color=self._drag_color)
        self.drag_handle.configure(fg_color=self._drag_color)

        logger.debug(f"Separator drag started at ({event.x_root}, {event.y_root})")

    def _on_drag(self, event):
        """Handle dragging motion."""
        if not self._is_dragging:
            return

        # Calculate delta
        if self.orientation == "vertical":
            delta = event.x_root - self._drag_start_x
        else:
            delta = event.y_root - self._drag_start_y

        # Get parent dimensions to calculate ratio
        try:
            parent = self.master
            if self.orientation == "vertical":
                parent_size = parent.winfo_width()
                current_pos = self.winfo_x()
            else:
                parent_size = parent.winfo_height()
                current_pos = self.winfo_y()

            # Calculate new position
            new_pos = current_pos + delta

            # Clamp to reasonable bounds (10% to 90% of parent)
            min_pos = int(parent_size * 0.1)
            max_pos = int(parent_size * 0.9)
            new_pos = max(min_pos, min(max_pos, new_pos))

            # Calculate ratio
            ratio = new_pos / parent_size if parent_size > 0 else 0.5

            # Update position (handled by parent's layout manager)
            # This is just visual feedback during drag
            logger.debug(f"Dragging: delta={delta}, ratio={ratio:.2f}")

        except Exception as e:
            logger.debug(f"Error during drag: {e}")

    def _on_release(self, event):
        """Handle mouse button release to complete dragging."""
        if not self._is_dragging:
            return

        self._is_dragging = False

        # Calculate final position ratio
        try:
            parent = self.master
            if self.orientation == "vertical":
                parent_size = parent.winfo_width()
                final_pos = self.winfo_x() + (event.x_root - self._drag_start_x)
            else:
                parent_size = parent.winfo_height()
                final_pos = self.winfo_y() + (event.y_root - self._drag_start_y)

            # Clamp to bounds
            min_pos = int(parent_size * 0.1)
            max_pos = int(parent_size * 0.9)
            final_pos = max(min_pos, min(max_pos, final_pos))

            # Calculate final ratio
            ratio = final_pos / parent_size if parent_size > 0 else 0.5

            logger.debug(f"Separator drag complete: ratio={ratio:.2f}")

            # Call callback with new ratio
            if self.on_drag_complete:
                self.on_drag_complete(ratio)

        except Exception as e:
            logger.error(f"Error completing drag: {e}")

        # Reset colors
        if self._is_hovered:
            self.configure(fg_color=self._hover_color)
            self.drag_handle.configure(fg_color=self._hover_color)
        else:
            self.configure(fg_color=self._default_color)
            self.drag_handle.configure(fg_color=self._default_color)

    def _on_destroy(self, event):
        """Clean up when widget is destroyed."""
        if event.widget != self:
            return
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except (ValueError, AttributeError):
            pass

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        # Update stored colors
        self._default_color = self.theme_manager.get_color("border")
        self._hover_color = self.theme_manager.get_color("bg_hover")

        # Apply current state
        if self._is_dragging:
            self.configure(fg_color=self._drag_color)
            self.drag_handle.configure(fg_color=self._drag_color)
        elif self._is_hovered:
            self.configure(fg_color=self._hover_color)
            self.drag_handle.configure(fg_color=self._hover_color)
        else:
            self.configure(fg_color=self._default_color)
            self.drag_handle.configure(fg_color=self._default_color)

    def set_callback(self, callback: Callable[[float], None]):
        """
        Set or update the drag completion callback.

        Args:
            callback: Function to call when drag completes, receives ratio (0.0-1.0)
        """
        self.on_drag_complete = callback
