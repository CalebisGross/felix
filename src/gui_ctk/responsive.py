"""
Responsive Layout System for Felix GUI.

Provides breakpoint detection, debounced resize handling, and layout configuration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, List, Optional
import customtkinter as ctk


class Breakpoint(Enum):
    """Window width breakpoints for responsive layouts."""
    COMPACT = "compact"      # < 1280px: Single column
    STANDARD = "standard"    # 1280-1920px: 2 columns
    WIDE = "wide"            # 1920-2560px: Full 2-column
    ULTRAWIDE = "ultrawide"  # 2560px+: 3 columns


@dataclass
class BreakpointConfig:
    """Configuration for a responsive breakpoint."""
    min_width: int
    max_width: Optional[int]
    card_columns: int
    card_min_width: int
    card_max_width: int
    spacing_multiplier: float


# Define breakpoint configurations
BREAKPOINTS: Dict[Breakpoint, BreakpointConfig] = {
    Breakpoint.COMPACT: BreakpointConfig(
        min_width=0, max_width=1279,
        card_columns=1, card_min_width=280, card_max_width=600,
        spacing_multiplier=0.8
    ),
    Breakpoint.STANDARD: BreakpointConfig(
        min_width=1280, max_width=1919,
        card_columns=2, card_min_width=300, card_max_width=500,
        spacing_multiplier=1.0
    ),
    Breakpoint.WIDE: BreakpointConfig(
        min_width=1920, max_width=2559,
        card_columns=3, card_min_width=320, card_max_width=480,
        spacing_multiplier=1.2
    ),
    Breakpoint.ULTRAWIDE: BreakpointConfig(
        min_width=2560, max_width=None,
        card_columns=4, card_min_width=350, card_max_width=500,
        spacing_multiplier=1.4
    ),
}


def get_breakpoint(width: int) -> Breakpoint:
    """Determine current breakpoint from window width."""
    if width < 1280:
        return Breakpoint.COMPACT
    elif width < 1920:
        return Breakpoint.STANDARD
    elif width < 2560:
        return Breakpoint.WIDE
    else:
        return Breakpoint.ULTRAWIDE


def get_breakpoint_config(width: int) -> BreakpointConfig:
    """Get configuration for current window width."""
    return BREAKPOINTS[get_breakpoint(width)]


class ResponsiveLayoutManager:
    """
    Manages responsive layout updates based on window resize events.

    Features:
    - Debounced resize handling (prevents excessive updates)
    - Breakpoint change callbacks
    - Current breakpoint tracking
    """

    def __init__(self, root: ctk.CTk, debounce_ms: int = 100):
        """
        Initialize the responsive layout manager.

        Args:
            root: The root CTk window
            debounce_ms: Milliseconds to wait before processing resize events
        """
        self.root = root
        self.debounce_ms = debounce_ms
        self.current_breakpoint: Optional[Breakpoint] = None
        self.callbacks: List[Callable[[Breakpoint, BreakpointConfig], None]] = []
        self._debounce_timer: Optional[str] = None

        # Initialize current breakpoint
        initial_width = root.winfo_width()
        if initial_width > 1:  # Valid width
            self.current_breakpoint = get_breakpoint(initial_width)

        # Bind to window configure events
        self.root.bind("<Configure>", self._on_configure)

    def register_callback(self, callback: Callable[[Breakpoint, BreakpointConfig], None]):
        """
        Register a callback for breakpoint changes.

        Args:
            callback: Function to call when breakpoint changes.
                     Receives (Breakpoint, BreakpointConfig) as arguments.
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable):
        """
        Remove a registered callback.

        Args:
            callback: The callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def get_current_breakpoint(self) -> Breakpoint:
        """
        Get current breakpoint.

        Returns:
            Current Breakpoint enum value
        """
        if self.current_breakpoint is None:
            width = self.root.winfo_width()
            self.current_breakpoint = get_breakpoint(width if width > 1 else 1280)
        return self.current_breakpoint

    def get_current_config(self) -> BreakpointConfig:
        """
        Get current breakpoint configuration.

        Returns:
            BreakpointConfig for current breakpoint
        """
        return BREAKPOINTS[self.get_current_breakpoint()]

    def force_update(self):
        """Force a layout update with current window dimensions."""
        width = self.root.winfo_width()
        if width > 1:  # Valid width
            self._check_breakpoint_change(width)

    def _on_configure(self, event):
        """
        Handle window configure event with debouncing.

        Args:
            event: Tkinter configure event
        """
        # Only process events from the root window
        if event.widget != self.root:
            return

        # Cancel previous timer if it exists
        if self._debounce_timer is not None:
            self.root.after_cancel(self._debounce_timer)

        # Schedule new check after debounce period
        self._debounce_timer = self.root.after(
            self.debounce_ms,
            lambda: self._check_breakpoint_change(event.width)
        )

    def _check_breakpoint_change(self, width: int):
        """
        Check if breakpoint changed and notify callbacks.

        Args:
            width: Current window width
        """
        new_breakpoint = get_breakpoint(width)

        # Check if breakpoint actually changed
        if new_breakpoint != self.current_breakpoint:
            self.current_breakpoint = new_breakpoint
            config = BREAKPOINTS[new_breakpoint]

            # Notify all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(new_breakpoint, config)
                except Exception as e:
                    print(f"Error in responsive layout callback: {e}")
