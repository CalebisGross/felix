"""
Responsive Card Grid Component for Felix GUI.

A grid container that automatically reflows cards based on available width.
"""

import customtkinter as ctk
from typing import List
from ..responsive import get_breakpoint_config
from ..styles import SPACE_MD


class ResponsiveCardGrid(ctk.CTkFrame):
    """
    A responsive grid container for cards that adapts to available width.

    Features:
    - Automatic column count adjustment based on width
    - Card min/max width constraints
    - Uniform card sizing within rows
    - Reflows on resize
    """

    def __init__(
        self,
        master,
        min_card_width: int = 280,
        max_card_width: int = 500,
        gap: int = SPACE_MD,
        **kwargs
    ):
        """
        Initialize the responsive card grid.

        Args:
            master: Parent widget
            min_card_width: Minimum width for each card
            max_card_width: Maximum width for each card
            gap: Spacing between cards
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(master, **kwargs)

        self.min_card_width = min_card_width
        self.max_card_width = max_card_width
        self.gap = gap
        self.cards: List[ctk.CTkFrame] = []
        self._last_width = 0
        self._current_columns = 0

        # Bind to resize events
        self.bind("<Configure>", self._on_resize)

    def add_card(self, card: ctk.CTkFrame):
        """
        Add a card to the grid.

        Args:
            card: The card widget to add
        """
        if card not in self.cards:
            self.cards.append(card)
            self._reflow_cards()

    def remove_card(self, card: ctk.CTkFrame):
        """
        Remove a card from the grid.

        Args:
            card: The card widget to remove
        """
        if card in self.cards:
            self.cards.remove(card)
            card.grid_forget()
            self._reflow_cards()

    def clear_cards(self):
        """Remove all cards from the grid."""
        for card in self.cards:
            card.grid_forget()
        self.cards.clear()
        self._current_columns = 0

    def _on_resize(self, event):
        """
        Handle resize event.

        Args:
            event: Tkinter configure event
        """
        # Only reflow if width changed significantly (avoid micro-adjustments)
        if abs(event.width - self._last_width) > 10:
            self._last_width = event.width
            self._reflow_cards()

    def _calculate_columns(self, available_width: int) -> int:
        """
        Calculate optimal number of columns for available width.

        Args:
            available_width: Available width in pixels

        Returns:
            Number of columns that fit
        """
        if available_width <= self.min_card_width:
            return 1

        # Calculate how many columns can fit with minimum width + gaps
        max_possible = (available_width + self.gap) // (self.min_card_width + self.gap)

        # Limit by maximum card width
        # If cards would exceed max_width, reduce column count
        card_width = (available_width - (max_possible - 1) * self.gap) / max_possible
        while card_width > self.max_card_width and max_possible > 1:
            max_possible -= 1
            card_width = (available_width - (max_possible - 1) * self.gap) / max_possible

        return max(1, max_possible)

    def _reflow_cards(self):
        """Reflow cards into grid based on current width."""
        if not self.cards:
            return

        # Get current available width
        width = self.winfo_width()
        if width <= 1:  # Not yet rendered
            return

        # Calculate optimal number of columns
        num_columns = self._calculate_columns(width)

        # Only update if column count changed
        if num_columns != self._current_columns:
            self._current_columns = num_columns

            # Configure grid columns with equal weights and uniform sizing
            for col in range(num_columns):
                self.grid_columnconfigure(col, weight=1, uniform="card")

            # Clear any extra column configurations
            for col in range(num_columns, max(10, self._current_columns + 5)):
                self.grid_columnconfigure(col, weight=0, uniform="")

        # Place cards in grid
        for idx, card in enumerate(self.cards):
            row = idx // num_columns
            col = idx % num_columns

            # Configure row to expand
            self.grid_rowconfigure(row, weight=0)

            # Place card with padding
            card.grid(
                row=row,
                column=col,
                sticky="nsew",
                padx=(0 if col == 0 else self.gap // 2, 0 if col == num_columns - 1 else self.gap // 2),
                pady=(0, self.gap)
            )

    def on_breakpoint_change(self, breakpoint, config):
        """
        Handle breakpoint change notification from ResponsiveLayoutManager.

        Args:
            breakpoint: New Breakpoint enum value
            config: New BreakpointConfig object
        """
        # Update card width constraints based on breakpoint
        self.min_card_width = config.card_min_width
        self.max_card_width = config.card_max_width

        # Update gap spacing
        self.gap = int(SPACE_MD * config.spacing_multiplier)

        # Force reflow with new constraints
        self._current_columns = 0  # Force recalculation
        self._reflow_cards()
