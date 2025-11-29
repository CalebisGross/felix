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

# Import Felix modules (with fallback)
try:
    from src.communication import central_post
    from src.llm import lm_studio_client
except ImportError as e:
    logger.error(f"Failed to import Felix modules: {e}")
    central_post = None
    lm_studio_client = None


class DashboardTab(ctk.CTkFrame):
    """
    Dashboard tab with system controls and log display.
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
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()
        self.system_running = False

        self._setup_ui()
        self._start_log_polling()

    def _setup_ui(self):
        """Set up the dashboard UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Log area expands

        # Header section
        self._create_header()

        # Status cards section
        self._create_status_section()

        # Log display section
        self._create_log_section()

    def _create_header(self):
        """Create the header section with title and controls."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        self.title_label = ctk.CTkLabel(
            title_frame,
            text="Felix Framework",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Multi-Agent AI Framework with Helical Geometry",
            font=ctk.CTkFont(size=12),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.subtitle_label.pack(anchor="w")

        # Control buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.grid(row=0, column=2, sticky="e")

        self.start_button = ctk.CTkButton(
            button_frame,
            text="Start Felix",
            command=self._start_system,
            width=120,
            height=36,
            fg_color=self.theme_manager.get_color("success"),
            hover_color="#1e8449"
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(
            button_frame,
            text="Stop Felix",
            command=self._stop_system,
            width=120,
            height=36,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226",
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)

        # Theme toggle
        self.theme_button = ctk.CTkButton(
            button_frame,
            text="Toggle Theme",
            command=self._toggle_theme,
            width=100,
            height=36,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.theme_button.pack(side="left", padx=(20, 5))

    def _create_status_section(self):
        """Create the status cards section."""
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        # Status cards
        self.status_card = StatusCard(
            status_frame,
            title="System Status",
            value="Stopped",
            subtitle="Not running",
            status_color=self.theme_manager.get_color("error"),
            width=180
        )
        self.status_card.pack(side="left", padx=(0, 10))

        self.agents_card = StatusCard(
            status_frame,
            title="Active Agents",
            value="0",
            subtitle="No agents spawned",
            width=180
        )
        self.agents_card.pack(side="left", padx=10)

        self.messages_card = StatusCard(
            status_frame,
            title="Messages",
            value="0",
            subtitle="Processed",
            width=180
        )
        self.messages_card.pack(side="left", padx=10)

        self.knowledge_card = StatusCard(
            status_frame,
            title="Knowledge",
            value="0",
            subtitle="Entries",
            width=180
        )
        self.knowledge_card.pack(side="left", padx=10)

    def _create_log_section(self):
        """Create the log display section."""
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(10, 20))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        # Log header
        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            log_header,
            text="System Log",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")

        self.clear_log_button = ctk.CTkButton(
            log_header,
            text="Clear",
            command=self._clear_log,
            width=60,
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.clear_log_button.pack(side="right")

        # Log textbox
        self.log_textbox = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Courier", size=11),
            wrap="word"
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # Make log read-only by disabling after setup
        self.log_textbox.configure(state="disabled")

    def _start_log_polling(self):
        """Start polling the log queue."""
        self._poll_log_queue()

    def _poll_log_queue(self):
        """Poll log queue and update display."""
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
                self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")

        except Exception as e:
            logger.error(f"Error polling log queue: {e}")

        # Schedule next poll
        self.after(100, self._poll_log_queue)

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
        self._update_state(running=True)

    def _disable_features(self):
        """Disable features when system is not running."""
        self._update_state(running=False)

    def refresh_status(self):
        """Refresh status cards with current system data."""
        if self.main_app and self.main_app.felix_system and self.system_running:
            status = self.main_app.felix_system.get_system_status()
            self.agents_card.set_value(str(status.get('agents', 0)))
            self.messages_card.set_value(str(status.get('messages_processed', 0)))
            self.knowledge_card.set_value(str(status.get('knowledge_entries', 0)))
