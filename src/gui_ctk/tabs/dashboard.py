"""
Dashboard Tab for Felix GUI (CustomTkinter Edition)

The dashboard provides:
- System start/stop controls
- Real-time log display
- System status overview
"""

import customtkinter as ctk
from typing import Optional
import logging

from ..utils import log_queue, logger
from ..theme_manager import get_theme_manager
from ..components.status_card import StatusCard
from ..components.responsive_grid import ResponsiveCardGrid
from ..responsive import Breakpoint, BreakpointConfig
from .base_tab import ResponsiveTab
from ..styles import (
    BUTTON_XS, BUTTON_MD, BUTTON_LG,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, CARD_MD
)

# Import Felix modules (with fallback)
try:
    from src.communication import central_post
    from src.llm import lm_studio_client
except ImportError as e:
    logger.error(f"Failed to import Felix modules: {e}")
    central_post = None
    lm_studio_client = None

# Log buffer rotation settings
MAX_LOG_LINES = 5000  # Maximum lines to keep in log display


class DashboardTab(ResponsiveTab):
    """
    Dashboard tab with system controls and log display.

    Responsive layout:
    - COMPACT: Single column (status cards, log, control panel stacked)
    - STANDARD: 2 columns (status cards | log)
    - WIDE: 2 columns (status + metrics | log)
    - ULTRAWIDE: 3 columns (status cards | log | metrics)
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize Dashboard tab.

        Args:
            master: Parent widget (typically CTkTabview)
            thread_manager: ThreadManager instance
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, thread_manager, main_app, **kwargs)

        self.theme_manager = get_theme_manager()
        self.system_running = False

        # Store references to layout containers
        self._main_container = None
        self._status_grid = None
        self._log_container = None
        self._control_panel = None
        self._current_layout = None

        self._setup_ui()
        self._start_log_polling()

    def _setup_ui(self):
        """Set up the dashboard UI."""
        # Configure grid for main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Content expands

        # Header section (always at top)
        self._create_header()

        # Main container for responsive layout
        self._main_container = ctk.CTkFrame(self, fg_color="transparent")
        self._main_container.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)

        # Create components (initial layout will be set by on_breakpoint_change)
        self._create_status_cards()
        self._create_log_section()

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Handle responsive layout changes based on breakpoint."""
        if not self._main_container:
            return

        # Skip redundant updates
        if self._current_layout == breakpoint:
            return

        self._current_layout = breakpoint

        # Clear existing layout
        for widget in self._main_container.winfo_children():
            widget.grid_forget()

        # Apply breakpoint-specific layout
        if breakpoint == Breakpoint.COMPACT:
            self._layout_compact()
        elif breakpoint == Breakpoint.STANDARD:
            self._layout_standard()
        elif breakpoint == Breakpoint.WIDE:
            self._layout_wide()
        else:  # ULTRAWIDE
            self._layout_ultrawide()

    def _layout_compact(self):
        """Single column: status cards, log (stacked vertically)."""
        self._main_container.grid_columnconfigure(0, weight=1)
        self._main_container.grid_rowconfigure(0, weight=0)  # Status cards
        self._main_container.grid_rowconfigure(1, weight=1)  # Log expands

        # Status grid on top
        if self._status_grid:
            self._status_grid.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_SM, SPACE_MD))

        # Log below
        if self._log_container:
            self._log_container.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=(0, SPACE_LG))

    def _layout_standard(self):
        """2 columns: status cards | log."""
        self._main_container.grid_columnconfigure(0, weight=1)  # Status side
        self._main_container.grid_columnconfigure(1, weight=2)  # Log side (wider)
        self._main_container.grid_rowconfigure(0, weight=1)

        # Status cards on left
        if self._status_grid:
            self._status_grid.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_SM)

        # Log on right
        if self._log_container:
            self._log_container.grid(row=0, column=1, sticky="nsew", padx=(SPACE_SM, SPACE_LG), pady=SPACE_SM)

    def _layout_wide(self):
        """2 columns: status cards + metrics | log (full height)."""
        # For now, use same layout as STANDARD (can be enhanced later with metrics)
        self._layout_standard()

    def _layout_ultrawide(self):
        """3 columns: status cards | log | metrics placeholder."""
        self._main_container.grid_columnconfigure(0, weight=1)  # Status
        self._main_container.grid_columnconfigure(1, weight=2)  # Log (widest)
        self._main_container.grid_columnconfigure(2, weight=1)  # Metrics placeholder
        self._main_container.grid_rowconfigure(0, weight=1)

        # Status cards on left
        if self._status_grid:
            self._status_grid.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_SM)

        # Log in center
        if self._log_container:
            self._log_container.grid(row=0, column=1, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)

        # Metrics placeholder on right
        metrics_placeholder = ctk.CTkFrame(self._main_container)
        metrics_placeholder.grid(row=0, column=2, sticky="nsew", padx=(SPACE_SM, SPACE_LG), pady=SPACE_SM)

        metrics_label = ctk.CTkLabel(
            metrics_placeholder,
            text="Live Metrics\n(Coming Soon)",
            font=ctk.CTkFont(size=FONT_SECTION),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        metrics_label.pack(expand=True)

    def _create_header(self):
        """Create the header section with title and controls."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))
        header_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        self.title_label = ctk.CTkLabel(
            title_frame,
            text="Felix Framework",
            font=ctk.CTkFont(size=FONT_TITLE, weight="bold")
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Multi-Agent AI Framework with Helical Geometry",
            font=ctk.CTkFont(size=FONT_BODY),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.subtitle_label.pack(anchor="w")

        # Control buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.grid(row=0, column=2, sticky="e")

        # Use accent colors from theme
        accent = self.theme_manager.get_color("accent")
        accent_hover = self.theme_manager.get_color("accent_hover")
        success = self.theme_manager.get_color("success")
        error = self.theme_manager.get_color("error")

        self.start_button = ctk.CTkButton(
            button_frame,
            text="▶ Start Felix",
            command=self._start_system,
            width=BUTTON_LG[0] + 20,
            height=BUTTON_LG[1] + 4,
            fg_color=success,
            hover_color="#2D8F69",
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.start_button.pack(side="left", padx=SPACE_XS)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="■ Stop Felix",
            command=self._stop_system,
            width=BUTTON_LG[0] + 20,
            height=BUTTON_LG[1] + 4,
            fg_color=error,
            hover_color="#8B3639",
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=SPACE_XS)

        # Theme toggle with accent styling
        self.theme_button = ctk.CTkButton(
            button_frame,
            text="🎨 Theme",
            command=self._toggle_theme,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            fg_color="transparent",
            border_width=2,
            border_color=accent,
            text_color=accent,
            hover_color=self.theme_manager.get_color("bg_hover"),
            corner_radius=8
        )
        self.theme_button.pack(side="left", padx=(SPACE_LG, SPACE_XS))

    def _create_status_cards(self):
        """Create the status cards using responsive grid."""
        # Create responsive card grid
        self._status_grid = ResponsiveCardGrid(
            self._main_container,
            min_card_width=CARD_MD,
            max_card_width=CARD_MD + 50,
            gap=SPACE_MD,
            fg_color="transparent"
        )

        # Create status cards
        self.status_card = StatusCard(
            self._status_grid,
            title="System Status",
            value="Stopped",
            subtitle="Not running",
            status_color=self.theme_manager.get_color("error"),
            width=CARD_MD
        )
        self._status_grid.add_card(self.status_card)

        self.agents_card = StatusCard(
            self._status_grid,
            title="Active Agents",
            value="0",
            subtitle="No agents spawned",
            width=CARD_MD
        )
        self._status_grid.add_card(self.agents_card)

        self.messages_card = StatusCard(
            self._status_grid,
            title="Messages",
            value="0",
            subtitle="Processed",
            width=CARD_MD
        )
        self._status_grid.add_card(self.messages_card)

        self.knowledge_card = StatusCard(
            self._status_grid,
            title="Knowledge",
            value="0",
            subtitle="Entries",
            width=CARD_MD
        )
        self._status_grid.add_card(self.knowledge_card)

    def _create_log_section(self):
        """Create the log display section."""
        self._log_container = ctk.CTkFrame(self._main_container)
        self._log_container.grid_columnconfigure(0, weight=1)
        self._log_container.grid_rowconfigure(1, weight=1)

        # Log header
        log_header = ctk.CTkFrame(self._log_container, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(
            log_header,
            text="System Log",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).pack(side="left")

        self.clear_log_button = ctk.CTkButton(
            log_header,
            text="Clear",
            command=self._clear_log,
            width=BUTTON_XS[0],
            height=BUTTON_XS[1],
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.clear_log_button.pack(side="right")

        # Log textbox
        self.log_textbox = ctk.CTkTextbox(
            self._log_container,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        # Make log read-only by disabling after setup
        self.log_textbox.configure(state="disabled")

    def _start_log_polling(self):
        """Start polling the log queue."""
        self._poll_log_queue()

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                return self.main_app.tabview.get() == "Dashboard"
        except Exception:
            pass
        return True  # Default to visible for dashboard (it's the first tab)

    def _poll_log_queue(self):
        """Poll log queue and update display."""
        # Check if shutdown signaled
        if not self.thread_manager.is_active:
            return  # Stop polling

        try:
            messages = []
            while True:
                try:
                    msg = log_queue.get_nowait()
                    messages.append(msg)
                except:
                    break

            if messages:
                self.log_textbox.configure(state="normal")
                for msg in messages:
                    self.log_textbox.insert("end", msg + '\n')

                # Rotate log buffer if it exceeds MAX_LOG_LINES
                line_count = int(self.log_textbox.index('end-1c').split('.')[0])
                if line_count > MAX_LOG_LINES:
                    lines_to_delete = line_count - MAX_LOG_LINES
                    self.log_textbox.delete("1.0", f"{lines_to_delete}.0")

                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")

        except Exception as e:
            logger.error(f"Error polling log queue: {e}")

        # Schedule next poll - slower when not visible
        if self.thread_manager.is_active:
            poll_interval = 100 if self._is_visible() else 500
            self.after(poll_interval, self._poll_log_queue)

    def _start_system(self):
        """Start the Felix system."""
        if self.system_running:
            return

        self.start_button.configure(state="disabled")
        logger.info("Starting Felix system...")

        if self.main_app:
            self.main_app.start_system()
            # Poll for system readiness
            self._poll_system_ready(max_attempts=20, poll_interval=250)
        else:
            logger.error("Cannot start system: main_app not available")
            self.start_button.configure(state="normal")

    def _poll_system_ready(self, max_attempts=20, poll_interval=250, attempt=0):
        """Poll for system ready state."""
        if attempt >= max_attempts:
            logger.warning("System startup polling timed out")
            self.start_button.configure(state="normal")
            return

        if self.main_app and self.main_app.system_running and self.main_app.felix_system:
            # System is ready
            self._update_state(running=True)
        else:
            # Keep polling
            self.after(poll_interval, lambda: self._poll_system_ready(
                max_attempts, poll_interval, attempt + 1
            ))

    def _stop_system(self):
        """Stop the Felix system."""
        if not self.system_running:
            return

        self.stop_button.configure(state="disabled")
        logger.info("Stopping Felix system...")

        if self.main_app:
            self.main_app.stop_system()
            # Poll for system stopped
            self._poll_system_stopped(max_attempts=20, poll_interval=250)
        else:
            logger.error("Cannot stop system: main_app not available")
            self.stop_button.configure(state="normal")

    def _poll_system_stopped(self, max_attempts=20, poll_interval=250, attempt=0):
        """Poll for system stopped state."""
        if attempt >= max_attempts:
            logger.warning("System shutdown polling timed out")
            self.stop_button.configure(state="normal")
            return

        if self.main_app and not self.main_app.system_running:
            # System is stopped
            self._update_state(running=False)
        else:
            # Keep polling
            self.after(poll_interval, lambda: self._poll_system_stopped(
                max_attempts, poll_interval, attempt + 1
            ))

    def _update_state(self, running: bool):
        """Update UI state based on system status."""
        self.system_running = running

        if running:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_card.set_value("Running")
            self.status_card.set_subtitle("System active")
            self.status_card.set_status_color(self.theme_manager.get_color("success"))

            # Update status from system
            if self.main_app and self.main_app.felix_system:
                status = self.main_app.felix_system.get_system_status()
                self.agents_card.set_value(str(status.get('agents', 0)))
                self.messages_card.set_value(str(status.get('messages_processed', 0)))
                self.knowledge_card.set_value(str(status.get('knowledge_entries', 0)))

            # Enable features in other tabs
            if self.main_app:
                self.main_app._enable_all_features()
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_card.set_value("Stopped")
            self.status_card.set_subtitle("Not running")
            self.status_card.set_status_color(self.theme_manager.get_color("error"))
            self.agents_card.set_value("0")
            self.agents_card.set_subtitle("No agents spawned")

            # Disable features in other tabs
            if self.main_app:
                self.main_app._disable_all_features()

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self.theme_manager.toggle_mode()

        # Update subtitle color
        self.subtitle_label.configure(
            text_color=self.theme_manager.get_color("fg_muted")
        )

    def _clear_log(self):
        """Clear the log display."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def _enable_features(self):
        """Enable features when system is running."""
        # Don't call _update_state here - it would cause recursion since
        # _update_state calls _enable_all_features which calls this method.
        # Dashboard state is already managed by _update_state itself.
        pass

    def _disable_features(self):
        """Disable features when system is not running."""
        # Don't call _update_state here - it would cause recursion since
        # _update_state calls _disable_all_features which calls this method.
        # Dashboard state is already managed by _update_state itself.
        pass

    def refresh_status(self):
        """Refresh status cards with current system data."""
        if self.main_app and self.main_app.felix_system and self.system_running:
            status = self.main_app.felix_system.get_system_status()
            self.agents_card.set_value(str(status.get('agents', 0)))
            self.messages_card.set_value(str(status.get('messages_processed', 0)))
            self.knowledge_card.set_value(str(status.get('knowledge_entries', 0)))
