"""
Terminal Tab for Felix GUI (CustomTkinter Edition)

Provides:
- Active Commands panel: Live output streaming for running commands
- Command History panel: Filterable history of all executed commands
- Kill command functionality for active processes
"""

import customtkinter as ctk
from tkinter import messagebox
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..utils import logger
from ..theme_manager import get_theme_manager
from ..components.themed_treeview import ThemedTreeview
from ..styles import (
    BUTTON_SM, BUTTON_MD,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    INPUT_MD, TEXTBOX_SM, TEXTBOX_MD, TEXTBOX_LG
)


class CommandDetailsDialog(ctk.CTkToplevel):
    """Modal dialog for viewing complete command execution details."""

    def __init__(self, parent, execution_details: Dict[str, Any]):
        """
        Initialize command details dialog.

        Args:
            parent: Parent window
            execution_details: Complete execution details from CommandHistory
        """
        super().__init__(parent)

        self.execution_details = execution_details
        self.theme_manager = get_theme_manager()

        # Window setup
        self.title(f"Command Execution #{execution_details.get('execution_id', 'N/A')}")
        self.geometry("800x900")
        self.resizable(True, True)

        # Make modal
        self.transient(parent)
        self.after(100, self._delayed_grab)

        # Build UI
        self._build_ui()

        # Center window
        self.update_idletasks()
        self._center_window()

    def _center_window(self):
        """Center the dialog on screen."""
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')

    def _build_ui(self):
        """Build the details dialog UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))

        status = self.execution_details.get('status', 'unknown')
        success = self.execution_details.get('success')
        status_icon = "✓" if success else "✗" if success is False else "⏸"

        title_label = ctk.CTkLabel(
            header_frame,
            text=f"{status_icon} Command Execution Details",
            font=ctk.CTkFont(size=FONT_TITLE, weight="bold")
        )
        title_label.pack(anchor="w")

        # Scrollable content frame
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=SPACE_SM)
        scroll_frame.grid_columnconfigure(0, weight=1)

        # Content sections
        self._add_section(scroll_frame, "Execution Info", {
            "Execution ID": self.execution_details.get('execution_id', 'N/A'),
            "Status": self.execution_details.get('status', 'unknown'),
            "Agent ID": self.execution_details.get('agent_id', 'N/A'),
            "Agent Type": self.execution_details.get('agent_type', 'N/A'),
            "Workflow ID": self.execution_details.get('workflow_id', 'N/A'),
            "Trust Level": self.execution_details.get('trust_level', 'N/A'),
            "Approved By": self.execution_details.get('approved_by', 'N/A')
        })

        self._add_section(scroll_frame, "Command", {
            "Command": self.execution_details.get('command', 'N/A'),
            "Context": self.execution_details.get('context', 'No context provided')
        }, multiline=['Command', 'Context'])

        # Extract values that may be None from database
        duration = self.execution_details.get('duration')
        output_size = self.execution_details.get('output_size')
        exit_code = self.execution_details.get('exit_code')

        self._add_section(scroll_frame, "Execution Results", {
            "Exit Code": exit_code if exit_code is not None else 'N/A',
            "Duration": f"{duration:.3f}s" if duration is not None else "Running...",
            "Success": "Yes" if success else "No" if success is False else "Pending",
            "Error Category": self.execution_details.get('error_category', 'None'),
            "Output Size": f"{output_size:,} bytes" if output_size is not None else "N/A"
        })

        # Output sections
        self._add_output_section(scroll_frame, "Standard Output",
                                self.execution_details.get('stdout_preview', '(no output)'))
        self._add_output_section(scroll_frame, "Standard Error",
                                self.execution_details.get('stderr_preview', '(no errors)'))

        self._add_section(scroll_frame, "Environment", {
            "Working Directory": self.execution_details.get('cwd', 'N/A'),
            "Virtual Env Active": "Yes" if self.execution_details.get('venv_active') else "No",
            "Execution Time": datetime.fromtimestamp(
                self.execution_details.get('execution_timestamp', 0)
            ).strftime('%Y-%m-%d %H:%M:%S')
        })

        # Close button
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_LG))

        ctk.CTkButton(
            button_frame,
            text="Close",
            command=self.destroy,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        ).pack(side="right")

    def _add_section(self, parent, title: str, data: Dict[str, str], multiline: List[str] = None):
        """Add a details section."""
        multiline = multiline or []

        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Section data
        for i, (label, value) in enumerate(data.items(), start=1):
            label_widget = ctk.CTkLabel(
                frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                anchor="ne"
            )
            label_widget.grid(row=i, column=0, sticky="ne", padx=(SPACE_SM, SPACE_SM), pady=SPACE_XS)

            if label in multiline:
                value_widget = ctk.CTkTextbox(frame, height=TEXTBOX_SM, wrap="word")
                value_widget.insert("1.0", str(value))
                value_widget.configure(state="disabled")
                value_widget.grid(row=i, column=1, sticky="ew", padx=(0, SPACE_SM), pady=SPACE_XS)
            else:
                value_widget = ctk.CTkLabel(
                    frame,
                    text=str(value),
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    anchor="w"
                )
                value_widget.grid(row=i, column=1, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

    def _add_output_section(self, parent, title: str, content: str):
        """Add an output section with scrolled text."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, pady=(0, SPACE_SM))

        # Section title
        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title_label.pack(anchor="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Output textbox
        text_widget = ctk.CTkTextbox(
            frame,
            height=TEXTBOX_MD,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=FONT_SMALL)
        )
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=SPACE_SM, pady=(0, SPACE_SM))

    def _delayed_grab(self):
        """Grab focus after window is rendered."""
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass  # Silently ignore if grab fails


from .base_tab import ResponsiveTab
from ..responsive import Breakpoint, BreakpointConfig
from ..components.resizable_separator import ResizableSeparator


class TerminalTab(ResponsiveTab):
    """
    Terminal tab for monitoring command execution.

    Features:
    - Active commands panel with live output streaming
    - Command history browser with filtering
    - Real-time updates via message polling
    - Kill command functionality
    - Responsive layout with resizable separator
    """

    def __init__(self, parent, thread_manager, main_app, **kwargs):
        """
        Initialize terminal tab.

        Args:
            parent: Parent widget
            thread_manager: Thread manager for background tasks
            main_app: Main application instance
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(parent, thread_manager, main_app, **kwargs)

        self.theme_manager = get_theme_manager()

        # Active commands tracking
        self.active_commands: Dict[int, Dict[str, Any]] = {}  # execution_id -> command info
        self.active_outputs: Dict[int, List[str]] = {}  # execution_id -> output lines

        # Polling state
        self.polling_active = False
        self.polling_job = None
        self._orphan_cleanup_done = False  # One-time cleanup flag

        # Layout state
        self.history_split_ratio = 0.7  # Output gets 70%, history gets 30%
        self.history_visible = True  # Track if history is visible

        # Build UI
        self._setup_ui()

        # Start polling when tab becomes visible
        self.bind("<Visibility>", self._on_visibility_change)

    def _setup_ui(self):
        """Build the terminal UI with responsive layout."""
        # Configure grid for responsive layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew")

        # Create sections (will be laid out in on_breakpoint_change)
        self.active_frame = None
        self.history_frame = None
        self.separator = None
        self.toggle_button = None

        # Build sections
        self._create_active_section()
        self._create_history_section()

        # Initial layout will be set by on_breakpoint_change

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Handle breakpoint changes for responsive layout."""
        # Clear existing layout
        for widget in self.main_container.winfo_children():
            widget.grid_forget()

        if breakpoint == Breakpoint.COMPACT:
            # COMPACT: History hidden, toggle button to show
            self._layout_compact()
        elif breakpoint == Breakpoint.STANDARD:
            # STANDARD: History as collapsible sidebar
            self._layout_standard()
        else:
            # WIDE/ULTRAWIDE: Side-by-side with separator
            self._layout_wide()

    def _layout_compact(self):
        """Layout for compact screens: output only, history accessible via button."""
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=0)  # Toggle button
        self.main_container.grid_rowconfigure(1, weight=1)  # Active/History

        # Create toggle button if needed
        if not self.toggle_button:
            self.toggle_button = ctk.CTkButton(
                self.main_container,
                text="Show History",
                command=self._toggle_history,
                width=BUTTON_MD[0],
                height=BUTTON_SM[1]
            )

        self.toggle_button.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_XS))

        # Show either active or history based on toggle state
        if self.history_visible:
            self.history_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=(SPACE_XS, SPACE_LG))
            self.toggle_button.configure(text="Show Output")
            if self.active_frame:
                self.active_frame.grid_forget()
        else:
            self.active_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=(SPACE_XS, SPACE_LG))
            self.toggle_button.configure(text="Show History")
            if self.history_frame:
                self.history_frame.grid_forget()

    def _layout_standard(self):
        """Layout for standard screens: history as collapsible sidebar."""
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=0 if not self.history_visible else 0)
        self.main_container.grid_rowconfigure(0, weight=0)
        self.main_container.grid_rowconfigure(1, weight=1)

        # Create toggle button if needed
        if not self.toggle_button:
            self.toggle_button = ctk.CTkButton(
                self.main_container,
                text="Hide History",
                command=self._toggle_history,
                width=BUTTON_SM[0],
                height=BUTTON_SM[1]
            )

        self.toggle_button.grid(row=0, column=0, columnspan=2, sticky="e", padx=SPACE_LG, pady=(SPACE_LG, SPACE_XS))

        # Active section always visible
        self.active_frame.grid(row=1, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM if self.history_visible else SPACE_LG), pady=(SPACE_XS, SPACE_LG))

        # History sidebar (collapsible)
        if self.history_visible:
            # Create separator if needed
            if not self.separator:
                self.separator = ResizableSeparator(
                    self.main_container,
                    orientation="vertical",
                    on_drag_complete=self._on_separator_drag
                )
            self.separator.grid(row=1, column=1, sticky="ns", pady=(SPACE_XS, SPACE_LG))

            self.history_frame.grid(row=1, column=2, sticky="nsew", padx=(0, SPACE_LG), pady=(SPACE_XS, SPACE_LG))
            self.main_container.grid_columnconfigure(2, weight=0, minsize=250)
            self.toggle_button.configure(text="Hide History")
        else:
            if self.separator:
                self.separator.grid_forget()
            self.history_frame.grid_forget()
            self.toggle_button.configure(text="Show History")

    def _layout_wide(self):
        """Layout for wide/ultrawide screens: side-by-side with resizable separator."""
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=0)
        self.main_container.grid_columnconfigure(2, weight=0, minsize=300)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Hide toggle button in wide mode
        if self.toggle_button:
            self.toggle_button.grid_forget()

        # Output pane
        self.active_frame.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_LG)

        # Create separator if needed
        if not self.separator:
            self.separator = ResizableSeparator(
                self.main_container,
                orientation="vertical",
                on_drag_complete=self._on_separator_drag
            )
        self.separator.grid(row=0, column=1, sticky="ns", pady=SPACE_LG)

        # History pane
        self.history_frame.grid(row=0, column=2, sticky="nsew", padx=(0, SPACE_LG), pady=SPACE_LG)
        self.history_visible = True

    def _toggle_history(self):
        """Toggle history visibility."""
        self.history_visible = not self.history_visible
        # Trigger layout update
        breakpoint = self.get_current_breakpoint()
        config = self.get_current_config()
        if breakpoint and config:
            self.on_breakpoint_change(breakpoint, config)

    def _on_separator_drag(self, ratio: float):
        """Handle separator drag completion."""
        self.history_split_ratio = ratio
        # Update column weights based on ratio
        # This is a simplified version; full implementation would resize columns

    def _create_active_section(self):
        """Create the active commands section."""
        self.active_frame = ctk.CTkFrame(self.main_container)
        self.active_frame.grid_columnconfigure(0, weight=1)
        self.active_frame.grid_rowconfigure(2, weight=1)

        active_frame = self.active_frame

        # Header
        header_frame = ctk.CTkFrame(active_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Active Commands",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        # Kill button
        self.kill_button = ctk.CTkButton(
            header_frame,
            text="Kill Command",
            command=self._kill_selected_command,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226",
            state="disabled"
        )
        self.kill_button.grid(row=0, column=1, sticky="e", padx=SPACE_XS)

        # Info label
        info_label = ctk.CTkLabel(
            active_frame,
            text="Real-time output from currently executing commands",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.grid(row=1, column=0, sticky="w", padx=SPACE_SM, pady=(0, SPACE_XS))

        # Active commands tree
        self.active_tree = ThemedTreeview(
            active_frame,
            columns=["command", "agent", "duration"],
            headings=["Command", "Agent", "Duration"],
            widths=[400, 150, 80],
            height=5
        )
        self.active_tree.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        # Bind selection
        self.active_tree.bind_tree("<<TreeviewSelect>>", self._on_active_select)

        # Output display header
        output_header_frame = ctk.CTkFrame(active_frame, fg_color="transparent")
        output_header_frame.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(
            output_header_frame,
            text="Live Output:",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold")
        ).pack(anchor="w")

        # Output textbox
        self.output_text = ctk.CTkTextbox(
            active_frame,
            height=TEXTBOX_MD,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=FONT_SMALL)
        )
        self.output_text.grid(row=4, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self.output_text.configure(state="disabled")

        # Configure row weights for resizing
        active_frame.grid_rowconfigure(2, weight=1)
        active_frame.grid_rowconfigure(4, weight=1)

    def _create_history_section(self):
        """Create the command history section."""
        self.history_frame = ctk.CTkFrame(self.main_container)
        self.history_frame.grid_columnconfigure(0, weight=1)
        self.history_frame.grid_rowconfigure(2, weight=1)

        history_frame = self.history_frame

        # Header
        header_label = ctk.CTkLabel(
            history_frame,
            text="Command History",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        header_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Filter controls
        filter_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
        filter_frame.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        ctk.CTkLabel(filter_frame, text="Filter:", font=ctk.CTkFont(size=FONT_BODY)).pack(side="left", padx=(0, SPACE_XS))

        self.search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(filter_frame, textvariable=self.search_var, width=INPUT_MD)
        search_entry.pack(side="left", padx=(0, SPACE_SM))
        search_entry.bind("<Return>", lambda e: self._refresh_history())

        ctk.CTkLabel(filter_frame, text="Status:", font=ctk.CTkFont(size=FONT_BODY)).pack(side="left", padx=(SPACE_SM, SPACE_XS))

        self.status_filter = ctk.StringVar(value="all")
        status_combo = ctk.CTkComboBox(
            filter_frame,
            variable=self.status_filter,
            values=["all", "completed", "failed", "running"],
            width=BUTTON_MD[0],
            command=lambda choice: self._refresh_history(),
            state="readonly"
        )
        status_combo.pack(side="left", padx=(0, SPACE_SM))

        ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self._refresh_history,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        ).pack(side="left", padx=(SPACE_SM, SPACE_XS))

        ctk.CTkButton(
            filter_frame,
            text="Clear Filters",
            command=self._clear_filters,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color="transparent",
            border_width=1
        ).pack(side="left", padx=SPACE_XS)

        # History tree
        self.history_tree = ThemedTreeview(
            history_frame,
            columns=["id", "command", "agent", "status", "exit_code", "duration", "timestamp"],
            headings=["ID", "Command", "Agent", "Status", "Exit Code", "Duration", "Timestamp"],
            widths=[50, 300, 120, 80, 70, 80, 150],
            height=10
        )
        self.history_tree.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))

        # Double-click to view details
        self.history_tree.bind_tree("<Double-Button-1>", self._on_history_double_click)

    def _on_visibility_change(self, _event):
        """Handle visibility changes to start/stop polling."""
        self._start_polling()

    def _start_polling(self):
        """Start polling for active commands."""
        if not self.polling_active:
            self.polling_active = True

            # One-time cleanup of orphaned commands from previous sessions
            if not self._orphan_cleanup_done:
                self._orphan_cleanup_done = True
                try:
                    if self.main_app and self.main_app.felix_system:
                        central_post = self.main_app.felix_system.central_post
                        if central_post and central_post.command_history:
                            count = central_post.command_history.cleanup_orphaned_commands()
                            if count > 0:
                                logger.info(f"Cleaned up {count} orphaned commands from previous session")
                except Exception as e:
                    logger.error(f"Error cleaning up orphaned commands: {e}")

            self._poll_active_commands()
            logger.info("Terminal polling started")

    def _stop_polling(self):
        """Stop polling."""
        if self.polling_active:
            self.polling_active = False
            if self.polling_job:
                self.after_cancel(self.polling_job)
                self.polling_job = None
            logger.info("Terminal polling stopped")

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                return self.main_app.tabview.get() == "Terminal"
        except Exception:
            pass
        return False

    def _poll_active_commands(self):
        """Poll for active commands and recent completions."""
        if not self.polling_active:
            return

        # Check visibility before doing expensive work
        if self._is_visible():
            try:
                # Query both active commands AND recently completed commands (last 30 seconds)
                if self.main_app and self.main_app.felix_system:
                    central_post = self.main_app.felix_system.central_post
                    if central_post:
                        # Get truly active (status='running')
                        active = central_post.command_history.get_active_commands()

                        # Get recent completions (last 30 seconds, any status)
                        recent_cutoff = time.time() - 30.0
                        recent_all = central_post.command_history.get_filtered_history(
                            date_from=recent_cutoff,
                            limit=20
                        )

                        # Combine: active + recent (deduplicate by execution_id)
                        combined = []
                        seen_ids = set()

                        for cmd in active:
                            if cmd['execution_id'] not in seen_ids:
                                combined.append(cmd)
                                seen_ids.add(cmd['execution_id'])

                        for cmd in recent_all:
                            if cmd['execution_id'] not in seen_ids:
                                combined.append(cmd)
                                seen_ids.add(cmd['execution_id'])

                        self._update_active_commands(combined)
            except Exception as e:
                logger.error(f"Error polling active commands: {e}")

        # Always reschedule next poll (500ms for responsive updates)
        self.polling_job = self.after(500, self._poll_active_commands)

    def _update_active_commands(self, active_commands: List[Dict[str, Any]]):
        """Update active commands display and retrieve live output."""
        # Track current selection
        selection = self.active_tree.selection()
        selected_id = int(selection[0]) if selection else None

        # Clear tree
        self.active_tree.clear()

        # Update active commands dict and repopulate
        new_active = {}
        for cmd in active_commands:
            exec_id = cmd['execution_id']
            new_active[exec_id] = cmd

            # Calculate duration
            start_time = cmd.get('execution_timestamp', time.time())
            duration = time.time() - start_time

            # Truncate command and add status indicator
            command = cmd['command']
            status = cmd.get('status', 'unknown')

            # Add status emoji prefix
            if status == 'running':
                status_icon = "⏳"
            elif status == 'completed':
                status_icon = "✓"
            elif status == 'failed':
                status_icon = "✗"
            else:
                status_icon = "?"

            # Truncate command (leave room for icon)
            if len(command) > 55:
                command = command[:52] + "..."

            display_command = f"{status_icon} {command}"

            # Insert into tree
            self.active_tree.insert(
                "",
                "end",
                iid=str(exec_id),
                values=(display_command, cmd.get('agent_id', 'N/A'), f"{duration:.1f}s")
            )

            # Retrieve live output from CentralPost
            try:
                if self.main_app and self.main_app.felix_system:
                    central_post = self.main_app.felix_system.central_post
                    if central_post:
                        live_output = central_post.get_live_command_output(exec_id)
                        if live_output:
                            # Initialize or update output buffer
                            if exec_id not in self.active_outputs:
                                self.active_outputs[exec_id] = []

                            # Get current buffer size to track new lines
                            current_size = len(self.active_outputs[exec_id])

                            # Append new lines (if any beyond current buffer)
                            for i, (line, stream_type) in enumerate(live_output):
                                if i >= current_size:
                                    prefix = "[ERR] " if stream_type == "stderr" else ""
                                    self.active_outputs[exec_id].append(f"{prefix}{line}")
            except Exception as e:
                logger.error(f"Error retrieving live output for {exec_id}: {e}")

        self.active_commands = new_active

        # Restore selection if still active
        if selected_id and selected_id in self.active_commands:
            self.active_tree.selection_set(str(selected_id))
            self._update_output_display(selected_id)
            # Enable kill button for running commands
            if self.active_commands[selected_id].get('status') == 'running':
                self.kill_button.configure(state="normal")

    def _on_active_select(self, event):
        """Handle active command selection."""
        selection = self.active_tree.selection()
        if selection:
            exec_id = int(selection[0])
            self._update_output_display(exec_id)

            # Enable/disable kill button based on command status
            if exec_id in self.active_commands:
                if self.active_commands[exec_id].get('status') == 'running':
                    self.kill_button.configure(state="normal")
                else:
                    self.kill_button.configure(state="disabled")
        else:
            self.kill_button.configure(state="disabled")

    def _update_output_display(self, execution_id: int):
        """Update output display for selected command."""
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")

        if execution_id in self.active_outputs:
            output = "\n".join(self.active_outputs[execution_id])
            self.output_text.insert("1.0", output)
            self.output_text.see("end")  # Scroll to bottom

        self.output_text.configure(state="disabled")

    def _kill_selected_command(self):
        """Kill the currently selected command."""
        selection = self.active_tree.selection()
        if not selection:
            return

        exec_id = int(selection[0])
        if exec_id not in self.active_commands:
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Kill",
            f"Are you sure you want to kill command #{exec_id}?\n\n"
            f"Command: {self.active_commands[exec_id].get('command', 'N/A')}",
            parent=self
        )

        if not result:
            return

        # Cancel command via CommandHistory (marks as failed in database)
        try:
            if self.main_app and self.main_app.felix_system:
                central_post = self.main_app.felix_system.central_post
                if central_post and central_post.command_history:
                    success = central_post.command_history.cancel_command(exec_id)
                    if success:
                        logger.info(f"Successfully cancelled command #{exec_id}")
                        messagebox.showinfo("Success", f"Command #{exec_id} cancelled", parent=self)
                        # Remove from active commands display
                        if exec_id in self.active_commands:
                            del self.active_commands[exec_id]
                        self._refresh_history()
                    else:
                        logger.warning(f"Failed to cancel command #{exec_id}")
                        messagebox.showerror("Error", f"Command #{exec_id} not found or already completed", parent=self)
                else:
                    messagebox.showerror("Error", "Command history not available", parent=self)
        except Exception as e:
            logger.error(f"Error cancelling command: {e}")
            messagebox.showerror("Error", f"Failed to cancel command: {e}", parent=self)

    def _refresh_history(self):
        """Refresh command history with current filters."""
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            if not central_post:
                return

            # Build filter parameters
            search_query = self.search_var.get().strip() or None
            status = self.status_filter.get()
            status_filter = None if status == "all" else status

            # Query history
            history = central_post.command_history.get_filtered_history(
                search_query=search_query,
                status=status_filter,
                limit=100
            )

            # Clear and populate tree
            self.history_tree.clear()

            for cmd in history:
                # Truncate command
                command = cmd['command']
                if len(command) > 50:
                    command = command[:47] + "..."

                # Format timestamp
                ts = datetime.fromtimestamp(cmd['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

                # Format duration
                duration = cmd.get('duration')
                duration_str = f"{duration:.2f}s" if duration is not None else "N/A"

                self.history_tree.insert(
                    "",
                    "end",
                    values=(
                        cmd['execution_id'],
                        command,
                        cmd.get('agent_id', 'N/A'),
                        cmd.get('status', 'unknown'),
                        cmd.get('exit_code', 'N/A'),
                        duration_str,
                        ts
                    )
                )

            logger.info(f"Refreshed history: {len(history)} commands")

        except Exception as e:
            logger.error(f"Error refreshing history: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to refresh history: {e}", parent=self)

    def _clear_filters(self):
        """Clear all filters and refresh."""
        self.search_var.set("")
        self.status_filter.set("all")
        self._refresh_history()

    def _on_history_double_click(self, event):
        """Handle double-click on history item to show details."""
        selection = self.history_tree.selection()
        if not selection:
            return

        item = self.history_tree.item(selection[0])
        exec_id = item['values'][0]

        # Get full details from CommandHistory
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            if not central_post:
                return

            details = central_post.command_history.get_command_details(exec_id)
            if details:
                CommandDetailsDialog(self, details)
            else:
                messagebox.showwarning(
                    "Not Found",
                    f"Command details not found for ID {exec_id}",
                    parent=self
                )

        except Exception as e:
            logger.error(f"Error showing command details: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load details: {e}", parent=self)

    def handle_command_start(self, execution_id: int, command: str, agent_id: str):
        """Handle command start broadcast."""
        # Initialize output tracking
        self.active_outputs[execution_id] = []
        logger.debug(f"Terminal: Command started - {execution_id}")

    def handle_command_output(self, execution_id: int, output_line: str, stream_type: str):
        """Handle command output broadcast."""
        # Append to output buffer
        if execution_id not in self.active_outputs:
            self.active_outputs[execution_id] = []

        prefix = "[ERR] " if stream_type == "stderr" else ""
        self.active_outputs[execution_id].append(f"{prefix}{output_line}")

        # Update display if this command is selected
        selection = self.active_tree.selection()
        if selection and int(selection[0]) == execution_id:
            self._update_output_display(execution_id)

    def handle_command_complete(self, execution_id: int, success: bool):
        """Handle command completion broadcast."""
        # Remove from active outputs after delay
        self.after(2000, lambda: self.active_outputs.pop(execution_id, None))
        logger.debug(f"Terminal: Command completed - {execution_id} (success={success})")

        # Refresh history to show completed command
        self._refresh_history()

    def _enable_features(self):
        """Enable terminal features when system starts."""
        self._start_polling()
        self._refresh_history()
        logger.info("Terminal tab features enabled")

    def _disable_features(self):
        """Disable terminal features when system stops."""
        self._stop_polling()
        self.kill_button.configure(state="disabled")
        logger.info("Terminal tab features disabled")
