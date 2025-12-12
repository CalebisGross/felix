"""
Base Tab Infrastructure for Responsive Felix GUI

Provides a responsive base class that all tabs can inherit from to support
adaptive layouts across different screen sizes.
"""

import customtkinter as ctk
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable

from ..responsive import (
    ResponsiveLayoutManager,
    Breakpoint,
    BreakpointConfig,
    get_breakpoint_config
)
from ..styles import SPACE_SM, SPACE_MD


class ResponsiveTab(ctk.CTkFrame, ABC):
    """
    Base class for responsive tabs.

    All tabs should inherit from this to get automatic responsive layout support.

    Features:
    - Automatic breakpoint change handling
    - Helper methods for common responsive patterns
    - Access to current breakpoint configuration

    Usage:
        class MyTab(ResponsiveTab):
            def __init__(self, master, thread_manager, main_app=None, **kwargs):
                super().__init__(master, thread_manager, main_app, **kwargs)
                self._setup_ui()

            def _setup_ui(self):
                # Use helper methods like create_master_detail_layout()
                pass

            def on_breakpoint_change(self, breakpoint, config):
                # Respond to breakpoint changes
                self._update_layout(breakpoint, config)
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize responsive tab.

        Args:
            master: Parent widget (typically CTkTabview tab)
            thread_manager: ThreadManager instance
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self._layout_manager: Optional[ResponsiveLayoutManager] = None
        self._registered_callback = False

        # Store references to layout components for responsive updates
        self._responsive_components: Dict[str, Any] = {}

    def set_layout_manager(self, layout_manager: ResponsiveLayoutManager):
        """
        Set the layout manager and register for breakpoint updates.

        Args:
            layout_manager: ResponsiveLayoutManager instance from main app
        """
        self._layout_manager = layout_manager

        # Register callback if not already registered
        if not self._registered_callback:
            layout_manager.register_callback(self._handle_breakpoint_change)
            self._registered_callback = True

            # Trigger initial layout update
            current_breakpoint = layout_manager.get_current_breakpoint()
            current_config = layout_manager.get_current_config()
            self.on_breakpoint_change(current_breakpoint, current_config)

    def get_current_breakpoint(self) -> Optional[Breakpoint]:
        """
        Get current breakpoint.

        Returns:
            Current Breakpoint enum value, or None if no layout manager
        """
        if self._layout_manager:
            return self._layout_manager.get_current_breakpoint()
        return None

    def get_current_config(self) -> Optional[BreakpointConfig]:
        """
        Get current breakpoint configuration.

        Returns:
            BreakpointConfig for current breakpoint, or None if no layout manager
        """
        if self._layout_manager:
            return self._layout_manager.get_current_config()
        return None

    def _handle_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """
        Internal handler for breakpoint changes.

        Args:
            breakpoint: New breakpoint
            config: New configuration
        """
        try:
            self.on_breakpoint_change(breakpoint, config)
        except Exception as e:
            # Log error but don't crash the app
            print(f"Error handling breakpoint change in {self.__class__.__name__}: {e}")

    @abstractmethod
    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """
        Handle breakpoint changes. Tabs must implement this method.

        Args:
            breakpoint: New breakpoint
            config: New breakpoint configuration
        """
        pass

    # =========================================================================
    # RESPONSIVE LAYOUT HELPERS
    # =========================================================================

    def create_master_detail_layout(
        self,
        master_widget: ctk.CTkFrame,
        detail_widget: ctk.CTkFrame,
        compact_mode: bool = False
    ):
        """
        Create a master-detail layout (list + detail pane pattern).

        In compact mode: Vertical stack (master on top, detail below)
        In standard/wide mode: Horizontal split (master left, detail right)

        Args:
            master_widget: The "list" or navigation widget
            detail_widget: The "detail" or content widget
            compact_mode: True for vertical stack, False for horizontal split

        Returns:
            Tuple of (container_frame, master_widget, detail_widget)
        """
        container = ctk.CTkFrame(self, fg_color="transparent")

        if compact_mode:
            # Vertical layout for compact screens
            container.grid_columnconfigure(0, weight=1)
            container.grid_rowconfigure(0, weight=0)  # Master: fixed size
            container.grid_rowconfigure(1, weight=1)  # Detail: expands

            master_widget.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, 0))
            detail_widget.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        else:
            # Horizontal layout for standard/wide screens
            container.grid_columnconfigure(0, weight=0)  # Master: fixed width
            container.grid_columnconfigure(1, weight=1)  # Detail: expands
            container.grid_rowconfigure(0, weight=1)

            master_widget.grid(row=0, column=0, sticky="nsew", padx=(SPACE_SM, 0), pady=SPACE_SM)
            detail_widget.grid(row=0, column=1, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)

        return container, master_widget, detail_widget

    def create_column_layout(
        self,
        columns: int,
        equal_width: bool = True,
        spacing: int = SPACE_MD
    ) -> ctk.CTkFrame:
        """
        Create a multi-column layout container.

        Args:
            columns: Number of columns
            equal_width: If True, all columns have equal weight
            spacing: Space between columns

        Returns:
            CTkFrame configured for column layout
        """
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid_rowconfigure(0, weight=1)

        for col in range(columns):
            weight = 1 if equal_width else 0
            container.grid_columnconfigure(col, weight=weight)

        return container

    def get_responsive_spacing(self, base_spacing: int = SPACE_MD) -> int:
        """
        Get spacing adjusted for current breakpoint.

        Args:
            base_spacing: Base spacing value

        Returns:
            Adjusted spacing based on breakpoint multiplier
        """
        config = self.get_current_config()
        if config:
            return int(base_spacing * config.spacing_multiplier)
        return base_spacing

    def is_compact_mode(self) -> bool:
        """
        Check if currently in compact mode.

        Returns:
            True if current breakpoint is COMPACT
        """
        breakpoint = self.get_current_breakpoint()
        return breakpoint == Breakpoint.COMPACT if breakpoint else False

    def is_wide_mode(self) -> bool:
        """
        Check if currently in wide or ultrawide mode.

        Returns:
            True if current breakpoint is WIDE or ULTRAWIDE
        """
        breakpoint = self.get_current_breakpoint()
        return breakpoint in (Breakpoint.WIDE, Breakpoint.ULTRAWIDE) if breakpoint else False

    def store_component(self, key: str, component: Any):
        """
        Store a reference to a component for responsive updates.

        Args:
            key: Identifier for the component
            component: The widget or object to store
        """
        self._responsive_components[key] = component

    def get_component(self, key: str) -> Optional[Any]:
        """
        Retrieve a stored component.

        Args:
            key: Identifier for the component

        Returns:
            The stored component, or None if not found
        """
        return self._responsive_components.get(key)

    def destroy(self):
        """Clean up resources when tab is destroyed."""
        # Unregister callback if registered
        if self._registered_callback and self._layout_manager:
            try:
                self._layout_manager.unregister_callback(self._handle_breakpoint_change)
            except Exception:
                pass  # Ignore errors during cleanup

        super().destroy()


class SimpleResponsiveTab(ResponsiveTab):
    """
    Simple responsive tab that doesn't require custom breakpoint handling.

    Use this for tabs that don't need special responsive behavior.
    The default implementation does nothing on breakpoint changes.
    """

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Default implementation: do nothing on breakpoint change."""
        pass
