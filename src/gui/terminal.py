"""
Terminal GUI component for real-time command execution monitoring.

Provides:
- Active Commands panel: Live output streaming for running commands
- Command History panel: Filterable history of all executed commands
- CommandDetailsDialog: Modal for viewing complete command details
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from .utils import ThreadManager, logger

logger = logging.getLogger(__name__)


class CommandDetailsDialog(tk.Toplevel):
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

        # Window setup
        self.title(f"Command Execution #{execution_details.get('execution_id', 'N/A')}")
        self.geometry("800x900")
        self.resizable(True, True)

        # Make modal
        self.transient(parent)

        # Build UI
        self._build_ui()

        # Center and grab focus
        self.update_idletasks()
        self._center_window()
        self.after(100, self._delayed_grab)

    def _center_window(self):
        """Center the dialog on screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')

    def _delayed_grab(self):
        """Grab focus after window is fully rendered."""
        try:
            self.grab_set()
            self.focus_set()
        except Exception as e:
            logger.warning(f"Could not grab dialog focus: {e}")

    def _build_ui(self):
        """Build the details dialog UI."""
        # Header
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(fill=tk.X)

        status = self.execution_details.get('status', 'unknown')
        success = self.execution_details.get('success')
        status_icon = "✓" if success else "✗" if success is False else "⏸"

        title_label = ttk.Label(
            header_frame,
            text=f"{status_icon} Command Execution Details",
            font=("TkDefaultFont", 14, "bold")
        )
        title_label.pack(anchor=tk.W)

        # Separator
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Scrollable content
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Content sections
        self._add_section(scrollable_frame, "Execution Info", {
            "Execution ID": self.execution_details.get('execution_id', 'N/A'),
            "Status": self.execution_details.get('status', 'unknown'),
            "Agent ID": self.execution_details.get('agent_id', 'N/A'),
            "Agent Type": self.execution_details.get('agent_type', 'N/A'),
            "Workflow ID": self.execution_details.get('workflow_id', 'N/A'),
            "Trust Level": self.execution_details.get('trust_level', 'N/A'),
            "Approved By": self.execution_details.get('approved_by', 'N/A')
        })

        self._add_section(scrollable_frame, "Command", {
            "Command": self.execution_details.get('command', 'N/A'),
            "Context": self.execution_details.get('context', 'No context provided')
        }, multiline=['Command', 'Context'])

        # Extract values that may be None from database
        duration = self.execution_details.get('duration')
        output_size = self.execution_details.get('output_size')
        exit_code = self.execution_details.get('exit_code')

        self._add_section(scrollable_frame, "Execution Results", {
            "Exit Code": exit_code if exit_code is not None else 'N/A',
            "Duration": f"{duration:.3f}s" if duration is not None else "Running...",
            "Success": "Yes" if success else "No" if success is False else "Pending",
            "Error Category": self.execution_details.get('error_category', 'None'),
            "Output Size": f"{output_size:,} bytes" if output_size is not None else "N/A"
        })

        # Output sections with scrolled text
        self._add_output_section(scrollable_frame, "Standard Output",
                                self.execution_details.get('stdout_preview', '(no output)'))
        self._add_output_section(scrollable_frame, "Standard Error",
                                self.execution_details.get('stderr_preview', '(no errors)'))

        self._add_section(scrollable_frame, "Environment", {
            "Working Directory": self.execution_details.get('cwd', 'N/A'),
            "Virtual Env Active": "Yes" if self.execution_details.get('venv_active') else "No",
            "Execution Time": datetime.fromtimestamp(
                self.execution_details.get('execution_timestamp', 0)
            ).strftime('%Y-%m-%d %H:%M:%S')
        })

        # Close button
        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT)

    def _add_section(self, parent, title: str, data: Dict[str, str], multiline: List[str] = None):
        """Add a details section."""
        multiline = multiline or []

        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.X, pady=(0, 10))

        for i, (label, value) in enumerate(data.items()):
            ttk.Label(frame, text=f"{label}:", font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=0, sticky=tk.NW, padx=(0, 10), pady=5
            )

            if label in multiline:
                text_widget = tk.Text(frame, height=3, wrap=tk.WORD, font=("TkDefaultFont", 9))
                text_widget.insert("1.0", str(value))
                text_widget.config(state=tk.DISABLED, bg="#f0f0f0")
                text_widget.grid(row=i, column=1, sticky=tk.EW, pady=5)
            else:
                value_label = ttk.Label(frame, text=str(value))
                value_label.grid(row=i, column=1, sticky=tk.W, pady=5)

        frame.columnconfigure(1, weight=1)

    def _add_output_section(self, parent, title: str, content: str):
        """Add an output section with scrolled text."""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        text_widget = scrolledtext.ScrolledText(
            frame,
            height=8,
            wrap=tk.WORD,
            font=("Courier", 9)
        )
        text_widget.insert("1.0", content)
        text_widget.config(state=tk.DISABLED, bg="#f0f0f0")
        text_widget.pack(fill=tk.BOTH, expand=True)


class TerminalFrame(ttk.Frame):
    """
    Terminal frame for monitoring command execution.

    Features:
    - Active commands panel with live output streaming
    - Command history browser with filtering
    - Real-time updates via message polling
    """

    def __init__(self, parent, thread_manager: ThreadManager, main_app, theme_manager=None):
        """
        Initialize terminal frame.

        Args:
            parent: Parent widget
            thread_manager: Thread manager for background tasks
            main_app: Main application instance
            theme_manager: Theme manager for styling
        """
        super().__init__(parent)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager

        # Active commands tracking
        self.active_commands: Dict[int, Dict[str, Any]] = {}  # execution_id -> command info
        self.active_outputs: Dict[int, List[str]] = {}  # execution_id -> output lines

        # Polling state
        self.polling_active = False
        self.polling_job = None

        # Build UI
        self._build_ui()

        # Apply theme if available
        if self.theme_manager:
            self.apply_theme()

        # Start polling when tab becomes visible
        self.bind("<Visibility>", self._on_visibility_change)

    def _build_ui(self):
        """Build the terminal UI."""
        # Split into top (active) and bottom (history)
        paned = ttk.PanedWindow(self, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Top: Active Commands
        active_frame = ttk.LabelFrame(paned, text="Active Commands", padding=10)
        paned.add(active_frame, weight=1)

        active_info = ttk.Label(
            active_frame,
            text="Real-time output from currently executing commands",
            font=("TkDefaultFont", 9)
        )
        active_info.pack(anchor=tk.W, pady=(0, 5))

        # Active commands list (TreeView)
        active_tree_frame = ttk.Frame(active_frame)
        active_tree_frame.pack(fill=tk.BOTH, expand=True)

        active_scrollbar = ttk.Scrollbar(active_tree_frame, orient=tk.VERTICAL)
        active_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.active_tree = ttk.Treeview(
            active_tree_frame,
            columns=("command", "agent", "duration"),
            show="headings",
            yscrollcommand=active_scrollbar.set,
            height=5
        )
        self.active_tree.heading("command", text="Command")
        self.active_tree.heading("agent", text="Agent")
        self.active_tree.heading("duration", text="Duration")
        self.active_tree.column("command", width=400)
        self.active_tree.column("agent", width=150)
        self.active_tree.column("duration", width=80)

        self.active_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        active_scrollbar.config(command=self.active_tree.yview)

        # Bind selection to show output
        self.active_tree.bind("<<TreeviewSelect>>", self._on_active_select)

        # Output display
        output_label_frame = ttk.Frame(active_frame)
        output_label_frame.pack(fill=tk.X, pady=(10, 5))
        ttk.Label(output_label_frame, text="Live Output:", font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)

        self.output_text = scrolledtext.ScrolledText(
            active_frame,
            height=10,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg="#1e1e1e",
            fg="#00ff00",
            insertbackground="#00ff00"
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)

        # Bottom: Command History
        history_frame = ttk.LabelFrame(paned, text="Command History", padding=10)
        paned.add(history_frame, weight=1)

        # Filter controls
        filter_frame = ttk.Frame(history_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 10))
        search_entry.bind("<Return>", lambda e: self._refresh_history())

        ttk.Label(filter_frame, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        self.status_filter = tk.StringVar(value="all")
        status_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.status_filter,
            values=["all", "completed", "failed", "running"],
            width=12,
            state="readonly"
        )
        status_combo.pack(side=tk.LEFT, padx=(0, 10))
        status_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_history())

        ttk.Button(filter_frame, text="Refresh", command=self._refresh_history).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(filter_frame, text="Clear Filters", command=self._clear_filters).pack(side=tk.LEFT)

        # History tree
        history_tree_frame = ttk.Frame(history_frame)
        history_tree_frame.pack(fill=tk.BOTH, expand=True)

        history_scrollbar = ttk.Scrollbar(history_tree_frame, orient=tk.VERTICAL)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_tree = ttk.Treeview(
            history_tree_frame,
            columns=("id", "command", "agent", "status", "exit_code", "duration", "timestamp"),
            show="headings",
            yscrollcommand=history_scrollbar.set
        )
        self.history_tree.heading("id", text="ID")
        self.history_tree.heading("command", text="Command")
        self.history_tree.heading("agent", text="Agent")
        self.history_tree.heading("status", text="Status")
        self.history_tree.heading("exit_code", text="Exit Code")
        self.history_tree.heading("duration", text="Duration")
        self.history_tree.heading("timestamp", text="Timestamp")

        self.history_tree.column("id", width=50)
        self.history_tree.column("command", width=300)
        self.history_tree.column("agent", width=120)
        self.history_tree.column("status", width=80)
        self.history_tree.column("exit_code", width=70)
        self.history_tree.column("duration", width=80)
        self.history_tree.column("timestamp", width=150)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.config(command=self.history_tree.yview)

        # Double-click to view details
        self.history_tree.bind("<Double-Button-1>", self._on_history_double_click)

    def _on_visibility_change(self, _event):
        """Handle visibility changes to start/stop polling."""
        self._start_polling()
        # Don't stop polling on obscure - keep updating in background

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
            # This catches commands that fail instantly (< 100ms) before next poll
            if self.main_app and self.main_app.felix_system:
                central_post = self.main_app.felix_system.central_post
                if central_post:
                    # Get truly active (status='running')
                    active = central_post.command_history.get_active_commands()

                    # Get recent completions (last 30 seconds, any status)
                    import time
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

        # Schedule next poll (100ms for faster response to catch quick failures)
        self.polling_job = self.after(100, self._poll_active_commands)

    def _update_active_commands(self, active_commands: List[Dict[str, Any]]):
        """Update active commands display and retrieve live output."""
        # Track current selection
        selection = self.active_tree.selection()
        selected_id = int(selection[0]) if selection else None

        # Clear tree
        for item in self.active_tree.get_children():
            self.active_tree.delete(item)

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

            # Add status emoji prefix for completed commands
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
                tk.END,
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

    def _on_active_select(self, event):
        """Handle active command selection."""
        selection = self.active_tree.selection()
        if selection:
            exec_id = int(selection[0])
            self._update_output_display(exec_id)

    def _update_output_display(self, execution_id: int):
        """Update output display for selected command."""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)

        if execution_id in self.active_outputs:
            output = "\n".join(self.active_outputs[execution_id])
            self.output_text.insert("1.0", output)
            self.output_text.see(tk.END)  # Scroll to bottom

        self.output_text.config(state=tk.DISABLED)

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
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

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
                    tk.END,
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
            messagebox.showerror("Error", f"Failed to refresh history: {e}")

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
                messagebox.showwarning("Not Found", f"Command details not found for ID {exec_id}")

        except Exception as e:
            logger.error(f"Error showing command details: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load details: {e}")

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

    def apply_theme(self):
        """Apply current theme to terminal frame."""
        if not self.theme_manager:
            return

        theme = self.theme_manager.get_current_theme()

        # Apply to output text widget (use terminal colors for output display)
        try:
            if self.theme_manager.is_dark_mode():
                self.output_text.config(bg="#1e1e1e", fg="#00ff00", insertbackground="#00ff00")
            else:
                self.output_text.config(bg="#ffffff", fg="#006600", insertbackground="#006600")
        except Exception as e:
            logger.warning(f"Could not theme output_text: {e}")

        # Apply to Treeview widgets
        try:
            style = ttk.Style()
            style.configure("Treeview",
                          background=theme["text_bg"],
                          foreground=theme["text_fg"],
                          fieldbackground=theme["text_bg"])
            style.map('Treeview',
                     background=[('selected', theme["text_select_bg"])],
                     foreground=[('selected', theme["text_select_fg"])])
        except Exception as e:
            logger.warning(f"Could not theme Treeview: {e}")

        # Recursively apply theme to all children (frames, labels, etc.)
        try:
            self.theme_manager.apply_to_all_children(self)
        except Exception as e:
            logger.warning(f"Could not recursively apply theme: {e}")

    def _enable_features(self):
        """Enable terminal features when system starts."""
        self._start_polling()
        self._refresh_history()
        logger.info("Terminal frame features enabled")

    def _disable_features(self):
        """Disable terminal features when system stops."""
        self._stop_polling()
        logger.info("Terminal frame features disabled")
