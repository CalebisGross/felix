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
        self.grab_set()

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
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        status = self.execution_details.get('status', 'unknown')
        success = self.execution_details.get('success')
        status_icon = "✓" if success else "✗" if success is False else "⏸"

        title_label = ctk.CTkLabel(
            header_frame,
            text=f"{status_icon} Command Execution Details",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(anchor="w")

        # Scrollable content frame
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
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
        button_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))

        ctk.CTkButton(
            button_frame,
            text="Close",
            command=self.destroy,
            width=100
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
            font=ctk.CTkFont(size=13, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Section data
        for i, (label, value) in enumerate(data.items(), start=1):
            label_widget = ctk.CTkLabel(
                frame,
                text=f"{label}:",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="ne"
            )
            label_widget.grid(row=i, column=0, sticky="ne", padx=(10, 10), pady=5)

            if label in multiline:
                value_widget = ctk.CTkTextbox(frame, height=60, wrap="word")
                value_widget.insert("1.0", str(value))
                value_widget.configure(state="disabled")
                value_widget.grid(row=i, column=1, sticky="ew", padx=(0, 10), pady=5)
            else:
                value_widget = ctk.CTkLabel(
                    frame,
                    text=str(value),
                    font=ctk.CTkFont(size=11),
                    anchor="w"
                )
                value_widget.grid(row=i, column=1, sticky="w", padx=(0, 10), pady=5)

    def _add_output_section(self, parent, title: str, content: str):
        """Add an output section with scrolled text."""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, pady=(0, 10))

        # Section title
        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        title_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Output textbox
        text_widget = ctk.CTkTextbox(
            frame,
            height=150,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=10)
        )
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")
        text_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))


class TerminalTab(ctk.CTkFrame):
    """
    Terminal tab for monitoring command execution.

    Features:
    - Active commands panel with live output streaming
    - Command history browser with filtering
    - Real-time updates via message polling
    - Kill command functionality
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
        super().__init__(parent, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()

        # Active commands tracking
        self.active_commands: Dict[int, Dict[str, Any]] = {}  # execution_id -> command info
        self.active_outputs: Dict[int, List[str]] = {}  # execution_id -> output lines

        # Polling state
        self.polling_active = False
        self.polling_job = None

        # Build UI
        self._build_ui()

        # Start polling when tab becomes visible
        self.bind("<Visibility>", self._on_visibility_change)

    def _build_ui(self):
        """Build the terminal UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top: Active Commands
        self._create_active_section()

        # Bottom: Command History
        self._create_history_section()

    def _create_active_section(self):
        """Create the active commands section."""
        active_frame = ctk.CTkFrame(self)
        active_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        active_frame.grid_columnconfigure(0, weight=1)
        active_frame.grid_rowconfigure(2, weight=1)

        # Header
        header_frame = ctk.CTkFrame(active_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text="Active Commands",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        # Kill button
        self.kill_button = ctk.CTkButton(
            header_frame,
            text="Kill Command",
            command=self._kill_selected_command,
            width=100,
            height=28,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226",
            state="disabled"
        )
        self.kill_button.grid(row=0, column=1, sticky="e", padx=5)

        # Info label
        info_label = ctk.CTkLabel(
            active_frame,
            text="Real-time output from currently executing commands",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))

        # Active commands tree
        self.active_tree = ThemedTreeview(
            active_frame,
            columns=["command", "agent", "duration"],
            headings=["Command", "Agent", "Duration"],
            widths=[400, 150, 80],
            height=5
        )
        self.active_tree.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # Bind selection
        self.active_tree.bind_tree("<<TreeviewSelect>>", self._on_active_select)

        # Output display header
        output_header_frame = ctk.CTkFrame(active_frame, fg_color="transparent")
        output_header_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            output_header_frame,
            text="Live Output:",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")

        # Output textbox
        self.output_text = ctk.CTkTextbox(
            active_frame,
            height=150,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=10)
        )
        self.output_text.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.output_text.configure(state="disabled")

        # Configure row weights for resizing
        active_frame.grid_rowconfigure(2, weight=1)
        active_frame.grid_rowconfigure(4, weight=1)

    def _create_history_section(self):
        """Create the command history section."""
        history_frame = ctk.CTkFrame(self)
        history_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        history_frame.grid_columnconfigure(0, weight=1)
        history_frame.grid_rowconfigure(2, weight=1)

        # Header
        header_label = ctk.CTkLabel(
            history_frame,
            text="Command History",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        header_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))

        # Filter controls
        filter_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
        filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 10))

        ctk.CTkLabel(filter_frame, text="Filter:").pack(side="left", padx=(0, 5))

        self.search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(filter_frame, textvariable=self.search_var, width=200)
        search_entry.pack(side="left", padx=(0, 10))
        search_entry.bind("<Return>", lambda e: self._refresh_history())

        ctk.CTkLabel(filter_frame, text="Status:").pack(side="left", padx=(10, 5))

        self.status_filter = ctk.StringVar(value="all")
        status_combo = ctk.CTkComboBox(
            filter_frame,
            variable=self.status_filter,
            values=["all", "completed", "failed", "running"],
            width=120,
            command=lambda choice: self._refresh_history(),
            state="readonly"
        )
        status_combo.pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self._refresh_history,
            width=80,
            height=28
        ).pack(side="left", padx=(10, 5))

        ctk.CTkButton(
            filter_frame,
            text="Clear Filters",
            command=self._clear_filters,
            width=100,
            height=28,
            fg_color="transparent",
            border_width=1
        ).pack(side="left", padx=5)

        # History tree
        self.history_tree = ThemedTreeview(
            history_frame,
            columns=["id", "command", "agent", "status", "exit_code", "duration", "timestamp"],
            headings=["ID", "Command", "Agent", "Status", "Exit Code", "Duration", "Timestamp"],
            widths=[50, 300, 120, 80, 70, 80, 150],
            height=10
        )
        self.history_tree.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Double-click to view details
        self.history_tree.bind_tree("<Double-Button-1>", self._on_history_double_click)

    def _on_visibility_change(self, _event):
        """Handle visibility changes to start/stop polling."""
        self._start_polling()

    def _start_polling(self):
        """Start polling for active commands."""
        if not self.polling_active:
            self.polling_active = True
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

    def _poll_active_commands(self):
        """Poll for active commands and recent completions."""
        if not self.polling_active:
            return

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

        # Schedule next poll (500ms for responsive updates)
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

        # Kill command via CentralPost
        try:
            if self.main_app and self.main_app.felix_system:
                central_post = self.main_app.felix_system.central_post
                if central_post and hasattr(central_post, 'kill_command'):
                    success = central_post.kill_command(exec_id)
                    if success:
                        logger.info(f"Successfully killed command #{exec_id}")
                        messagebox.showinfo("Success", f"Command #{exec_id} killed", parent=self)
                        self._refresh_history()
                    else:
                        logger.warning(f"Failed to kill command #{exec_id}")
                        messagebox.showerror("Error", f"Failed to kill command #{exec_id}", parent=self)
                else:
                    messagebox.showerror("Error", "Kill functionality not available", parent=self)
        except Exception as e:
            logger.error(f"Error killing command: {e}")
            messagebox.showerror("Error", f"Failed to kill command: {e}", parent=self)

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
