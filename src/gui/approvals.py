"""
Approvals GUI components for Felix System Autonomy.

Provides:
- ApprovalDialog: Popup dialog for user to approve/deny system commands
- ApprovalsFrame: Tab showing approval request history and current status
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from .utils import ThreadManager, logger

logger = logging.getLogger(__name__)


class ApprovalDialog(tk.Toplevel):
    """
    Popup dialog for approving system commands.

    Displays command details, risk assessment, and approval options:
    - Approve Once
    - Approve Always (Exact)
    - Approve Always (Command Type)
    - Approve Always (Path Pattern)
    - Deny
    """

    def __init__(self, parent, approval_request: Dict[str, Any], on_decision_callback):
        """
        Initialize approval dialog.

        Args:
            parent: Parent window
            approval_request: Approval request dictionary with keys:
                - approval_id
                - command
                - agent_id
                - context
                - trust_level
                - risk_assessment
                - workflow_id
            on_decision_callback: Callback function(approval_id, decision_type)
        """
        super().__init__(parent)

        self.approval_request = approval_request
        self.on_decision_callback = on_decision_callback
        self.decision_made = False

        # Window setup
        self.title("System Command Approval Required")
        self.geometry("700x700")  # Increased height to accommodate all content
        self.resizable(True, True)  # Allow resizing for debugging

        # Make modal (block interaction with parent)
        self.transient(parent)

        # Build UI
        self._build_ui()

        # Update geometry and center AFTER building UI
        self.update_idletasks()  # Force layout calculation
        self.minsize(700, 700)  # Set minimum size
        self._center_window()

        # Delay grab_set() until window is fully rendered (prevents "window not viewable" error)
        self.after(100, self._delayed_grab)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _center_window(self):
        """Center the dialog on screen."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')

    def _delayed_grab(self):
        """Grab focus after window is fully rendered.

        Prevents "grab failed: window not viewable" error by ensuring
        the window is fully visible before attempting to grab focus.
        """
        try:
            self.grab_set()
            self.focus_set()
            logger.debug("✓ Dialog focus grabbed successfully")
        except Exception as e:
            logger.warning(f"Could not grab dialog focus: {e}")
            # Continue anyway - dialog still works without modal grab

    def _build_ui(self):
        """Build the approval dialog UI."""

        # Header
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(fill=tk.X)

        title_label = ttk.Label(
            header_frame,
            text="⚠️ System Command Requires Approval",
            font=("TkDefaultFont", 14, "bold")
        )
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="An agent has requested permission to execute a system command.",
            font=("TkDefaultFont", 9)
        )
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))

        # Separator
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Content frame (scrollable)
        content_frame = ttk.Frame(self, padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Command details
        details_frame = ttk.LabelFrame(content_frame, text="Command Details", padding=10)
        details_frame.pack(fill=tk.X, pady=(0, 10))

        # Command
        ttk.Label(details_frame, text="Command:", font=("TkDefaultFont", 9, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        command_text = tk.Text(details_frame, height=2, wrap=tk.WORD, font=("TkDefaultFont", 10))  # Reduced height
        command_text.insert("1.0", self.approval_request['command'])
        command_text.config(state=tk.DISABLED, bg="#f0f0f0")
        command_text.grid(row=0, column=1, sticky=tk.EW, pady=5)
        details_frame.columnconfigure(1, weight=1)

        # Agent
        ttk.Label(details_frame, text="Requested by:", font=("TkDefaultFont", 9, "bold")).grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        ttk.Label(details_frame, text=self.approval_request['agent_id']).grid(
            row=1, column=1, sticky=tk.W, pady=5
        )

        # Trust level
        ttk.Label(details_frame, text="Trust Level:", font=("TkDefaultFont", 9, "bold")).grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=5
        )
        trust_label = ttk.Label(
            details_frame,
            text=self.approval_request['trust_level'].upper(),
            foreground="orange" if self.approval_request['trust_level'] == 'review' else "black"
        )
        trust_label.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Context
        if self.approval_request.get('context'):
            ttk.Label(details_frame, text="Context:", font=("TkDefaultFont", 9, "bold")).grid(
                row=3, column=0, sticky=tk.W, padx=(0, 10), pady=5
            )
            context_text = tk.Text(details_frame, height=2, wrap=tk.WORD, font=("TkDefaultFont", 9))
            context_text.insert("1.0", self.approval_request['context'])
            context_text.config(state=tk.DISABLED, bg="#f9f9f9")
            context_text.grid(row=3, column=1, sticky=tk.EW, pady=5)

        # Risk assessment
        risk_frame = ttk.LabelFrame(content_frame, text="Risk Assessment", padding=10)
        risk_frame.pack(fill=tk.X, pady=(0, 10))

        risk_text = tk.Text(risk_frame, height=2, wrap=tk.WORD, font=("TkDefaultFont", 9))  # Reduced height
        risk_text.insert("1.0", self.approval_request.get('risk_assessment', 'Modifies system state - requires user confirmation.'))
        risk_text.config(state=tk.DISABLED, bg="#fff9e6")
        risk_text.pack(fill=tk.X)

        # Decision options
        options_frame = ttk.LabelFrame(content_frame, text="Approval Options", padding=10)
        options_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            options_frame,
            text="Choose how to handle this command:",
            font=("TkDefaultFont", 9, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))

        # Decision variable
        self.decision_var = tk.StringVar(value="approve_once")

        # Radio buttons
        decisions = [
            ("approve_once", "Approve Once", "Execute this command one time only"),
            ("approve_always_exact", "Always Approve (Exact)", "Always approve this exact command in this workflow"),
            ("approve_always_command", "Always Approve (Command Type)", "Always approve this command type (e.g., all 'mkdir') in this workflow"),
            ("approve_always_path_pattern", "Always Approve (Path Pattern)", "Always approve commands in this path pattern in this workflow"),
            ("deny", "Deny", "Reject this command and continue workflow")
        ]

        for value, label, description in decisions:
            radio = ttk.Radiobutton(
                options_frame,
                text=label,
                value=value,
                variable=self.decision_var
            )
            radio.pack(anchor=tk.W, pady=2)

            desc_label = ttk.Label(
                options_frame,
                text=f"  └─ {description}",
                font=("TkDefaultFont", 8),
                foreground="gray"
            )
            desc_label.pack(anchor=tk.W, padx=(20, 0))

        # Separator
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Button frame
        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X)

        # Cancel button (left)
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel (Deny)",
            command=self._on_cancel,
            width=15
        )
        cancel_btn.pack(side=tk.LEFT)

        # Submit button (right)
        submit_btn = ttk.Button(
            button_frame,
            text="Submit Decision",
            command=self._on_submit,
            width=15
        )
        submit_btn.pack(side=tk.RIGHT)

        # Auto-focus submit button
        submit_btn.focus_set()

    def _on_submit(self):
        """Handle submit button click."""
        from src.execution.approval_manager import ApprovalDecision

        decision_type = self.decision_var.get()
        decision_enum = ApprovalDecision(decision_type)

        # Confirm "always approve" decisions
        if decision_type.startswith("approve_always"):
            confirm = messagebox.askyesno(
                "Confirm Always Approve",
                f"This will automatically approve similar commands for the rest of this workflow.\n\n"
                f"Decision: {decision_enum.value}\n"
                f"Command: {self.approval_request['command']}\n\n"
                f"Continue?",
                parent=self
            )
            if not confirm:
                return

        # Mark decision made
        self.decision_made = True

        # Call callback
        try:
            self.on_decision_callback(
                self.approval_request['approval_id'],
                decision_enum
            )
        except Exception as e:
            logger.error(f"Error processing approval decision: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to process approval decision:\n{str(e)}",
                parent=self
            )
            return

        # Close dialog
        self.destroy()

    def _on_cancel(self):
        """Handle cancel button click."""
        from src.execution.approval_manager import ApprovalDecision

        confirm = messagebox.askyesno(
            "Confirm Denial",
            "This will deny the command and continue the workflow.\n\n"
            "The agent will be notified that the command was denied.\n\n"
            "Continue?",
            parent=self
        )

        if confirm:
            self.decision_made = True
            self.on_decision_callback(
                self.approval_request['approval_id'],
                ApprovalDecision.DENY
            )
            self.destroy()

    def _on_close(self):
        """Handle window close button."""
        if not self.decision_made:
            messagebox.showwarning(
                "Approval Required",
                "You must make a decision on this approval request.\n\n"
                "The workflow cannot continue until you approve or deny this command.",
                parent=self
            )


class ApprovalsFrame(ttk.Frame):
    """
    Tab showing approval request history and current pending approvals.

    Features:
    - Real-time display of pending approval requests
    - History of all approval decisions
    - Quick approve/deny actions from list
    - Filtering by workflow, status
    """

    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager
        self.polling_active = False
        self.poll_interval = 2000  # Poll every 2 seconds

        # Build UI
        self._build_ui()

        # Apply initial theme
        if self.theme_manager:
            self.apply_theme()

    def _build_ui(self):
        """Build the approvals tab UI."""

        # Header
        header_frame = ttk.Frame(self, padding=10)
        header_frame.pack(fill=tk.X)

        title_label = ttk.Label(
            header_frame,
            text="System Command Approvals",
            font=("TkDefaultFont", 12, "bold")
        )
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="Pending approvals and approval history",
            font=("TkDefaultFont", 9)
        )
        subtitle_label.pack(anchor=tk.W)

        # Control frame
        control_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        control_frame.pack(fill=tk.X)

        # Refresh button
        self.refresh_button = ttk.Button(
            control_frame,
            text="Refresh",
            command=self._refresh_approvals,
            state=tk.DISABLED
        )
        self.refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        # Auto-refresh toggle
        self.auto_refresh_var = tk.BooleanVar(value=False)
        auto_refresh_check = ttk.Checkbutton(
            control_frame,
            text="Auto-refresh (2s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh,
            state=tk.DISABLED
        )
        auto_refresh_check.pack(side=tk.LEFT, padx=(0, 10))

        # Status label
        self.status_label = ttk.Label(
            control_frame,
            text="System not running",
            foreground="gray"
        )
        self.status_label.pack(side=tk.LEFT)

        # Pending approvals section
        pending_frame = ttk.LabelFrame(self, text="Pending Approvals", padding=10)
        pending_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Treeview for pending approvals
        tree_frame = ttk.Frame(pending_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview
        self.pending_tree = ttk.Treeview(
            tree_frame,
            columns=("approval_id", "command", "agent", "workflow", "requested_at"),
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )

        tree_scroll_y.config(command=self.pending_tree.yview)
        tree_scroll_x.config(command=self.pending_tree.xview)

        # Column headings
        self.pending_tree.heading("approval_id", text="Approval ID")
        self.pending_tree.heading("command", text="Command")
        self.pending_tree.heading("agent", text="Agent")
        self.pending_tree.heading("workflow", text="Workflow")
        self.pending_tree.heading("requested_at", text="Requested At")

        # Column widths
        self.pending_tree.column("approval_id", width=120)
        self.pending_tree.column("command", width=300)
        self.pending_tree.column("agent", width=150)
        self.pending_tree.column("workflow", width=150)
        self.pending_tree.column("requested_at", width=150)

        self.pending_tree.pack(fill=tk.BOTH, expand=True)

        # Double-click to review
        self.pending_tree.bind("<Double-1>", self._on_pending_double_click)

        # Action buttons for pending approvals
        action_frame = ttk.Frame(pending_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        self.review_button = ttk.Button(
            action_frame,
            text="Review Selected",
            command=self._review_selected,
            state=tk.DISABLED
        )
        self.review_button.pack(side=tk.LEFT, padx=(0, 5))

        self.quick_approve_button = ttk.Button(
            action_frame,
            text="Quick Approve",
            command=self._quick_approve_selected,
            state=tk.DISABLED
        )
        self.quick_approve_button.pack(side=tk.LEFT, padx=(0, 5))

        self.quick_deny_button = ttk.Button(
            action_frame,
            text="Quick Deny",
            command=self._quick_deny_selected,
            state=tk.DISABLED
        )
        self.quick_deny_button.pack(side=tk.LEFT)

        # Approval History section
        history_frame = ttk.LabelFrame(self, text="Approval History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Treeview for approval history
        history_tree_frame = ttk.Frame(history_frame)
        history_tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        history_scroll_y = ttk.Scrollbar(history_tree_frame)
        history_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        history_scroll_x = ttk.Scrollbar(history_tree_frame, orient=tk.HORIZONTAL)
        history_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview
        self.history_tree = ttk.Treeview(
            history_tree_frame,
            columns=("timestamp", "command", "decision", "result", "workflow"),
            show="headings",
            yscrollcommand=history_scroll_y.set,
            xscrollcommand=history_scroll_x.set
        )

        history_scroll_y.config(command=self.history_tree.yview)
        history_scroll_x.config(command=self.history_tree.xview)

        # Column headings
        self.history_tree.heading("timestamp", text="Timestamp")
        self.history_tree.heading("command", text="Command")
        self.history_tree.heading("decision", text="Decision")
        self.history_tree.heading("result", text="Result")
        self.history_tree.heading("workflow", text="Workflow")

        # Column widths
        self.history_tree.column("timestamp", width=150)
        self.history_tree.column("command", width=300)
        self.history_tree.column("decision", width=150)
        self.history_tree.column("result", width=100)
        self.history_tree.column("workflow", width=150)

        self.history_tree.pack(fill=tk.BOTH, expand=True)

        # Action buttons for history
        history_action_frame = ttk.Frame(history_frame)
        history_action_frame.pack(fill=tk.X, pady=(10, 0))

        self.refresh_history_button = ttk.Button(
            history_action_frame,
            text="Refresh History",
            command=self._refresh_history,
            state=tk.DISABLED
        )
        self.refresh_history_button.pack(side=tk.LEFT, padx=(0, 5))

        history_info_label = ttk.Label(
            history_action_frame,
            text="Showing last 50 approvals",
            font=("TkDefaultFont", 8)
        )
        history_info_label.pack(side=tk.LEFT)

    def _enable_features(self):
        """Enable approval features when system is running."""
        self.refresh_button.config(state=tk.NORMAL)
        self.review_button.config(state=tk.NORMAL)
        self.quick_approve_button.config(state=tk.NORMAL)
        self.quick_deny_button.config(state=tk.NORMAL)
        self.refresh_history_button.config(state=tk.NORMAL)

        # Enable auto-refresh checkbox
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        child.config(state=tk.NORMAL)

        self.status_label.config(text="Ready", foreground="green")

        # Start auto-refresh if enabled
        if self.auto_refresh_var.get():
            self._start_polling()

    def _disable_features(self):
        """Disable approval features when system is not running."""
        self.refresh_button.config(state=tk.DISABLED)
        self.review_button.config(state=tk.DISABLED)
        self.quick_approve_button.config(state=tk.DISABLED)
        self.quick_deny_button.config(state=tk.DISABLED)
        self.refresh_history_button.config(state=tk.DISABLED)

        # Disable auto-refresh checkbox
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        child.config(state=tk.DISABLED)

        self.status_label.config(text="System not running", foreground="gray")

        # Stop polling
        self._stop_polling()

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh polling."""
        if self.auto_refresh_var.get():
            self._start_polling()
        else:
            self._stop_polling()

    def _start_polling(self):
        """Start polling for pending approvals."""
        if not self.polling_active:
            self.polling_active = True
            self._poll_approvals()

    def _stop_polling(self):
        """Stop polling for pending approvals."""
        self.polling_active = False

    def _poll_approvals(self):
        """Poll for pending approvals and update UI."""
        if not self.polling_active:
            return

        # Refresh approvals list
        self._refresh_approvals()

        # Schedule next poll
        self.after(self.poll_interval, self._poll_approvals)

    def _refresh_approvals(self):
        """Refresh the pending approvals list and history."""
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            pending_approvals = central_post.get_pending_actions()

            # Clear current items
            for item in self.pending_tree.get_children():
                self.pending_tree.delete(item)

            # Add pending approvals
            for approval in pending_approvals:
                requested_at = datetime.fromtimestamp(approval['requested_at']).strftime("%Y-%m-%d %H:%M:%S")

                self.pending_tree.insert(
                    "",
                    tk.END,
                    values=(
                        approval['approval_id'],
                        approval['command'][:50] + "..." if len(approval['command']) > 50 else approval['command'],
                        approval['agent_id'],
                        approval.get('workflow_id', 'N/A'),
                        requested_at
                    )
                )

            # Update status
            count = len(pending_approvals)
            if count > 0:
                self.status_label.config(
                    text=f"{count} pending approval{'s' if count != 1 else ''}",
                    foreground="orange"
                )
            else:
                self.status_label.config(
                    text="No pending approvals",
                    foreground="green"
                )

            # Also refresh history
            self._refresh_history()

        except Exception as e:
            logger.error(f"Error refreshing approvals: {e}", exc_info=True)
            self.status_label.config(
                text="Error refreshing approvals",
                foreground="red"
            )

    def _refresh_history(self):
        """Refresh the approval history list."""
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post

            # Get approval manager (it's part of central_post)
            if not hasattr(central_post, 'approval_manager') or not central_post.approval_manager:
                logger.warning("Approval manager not available")
                return

            approval_history = central_post.approval_manager.get_approval_history(limit=50)

            # Clear current items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            # Add history entries
            for approval in approval_history:
                # Format timestamp
                timestamp = approval.decided_at if approval.decided_at else approval.expires_at
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

                # Format decision
                decision_str = approval.decision.value if approval.decision else "expired"

                # Format result based on status
                if approval.status.value == "approved":
                    result_str = "✓ Approved"
                elif approval.status.value == "denied":
                    result_str = "✗ Denied"
                elif approval.status.value == "expired":
                    result_str = "⏱ Expired"
                elif approval.status.value == "auto_approved":
                    result_str = "⚡ Auto"
                else:
                    result_str = approval.status.value

                # Truncate command for display
                command_display = approval.command[:50] + "..." if len(approval.command) > 50 else approval.command

                self.history_tree.insert(
                    "",
                    tk.END,
                    values=(
                        timestamp_str,
                        command_display,
                        decision_str,
                        result_str,
                        approval.workflow_id or "N/A"
                    )
                )

            logger.info(f"Refreshed approval history: {len(approval_history)} entries")

        except Exception as e:
            logger.error(f"Error refreshing approval history: {e}", exc_info=True)

    def _on_pending_double_click(self, event):
        """Handle double-click on pending approval."""
        self._review_selected()

    def _review_selected(self):
        """Open approval dialog for selected pending approval."""
        selection = self.pending_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection",
                "Please select a pending approval to review."
            )
            return

        item = self.pending_tree.item(selection[0])
        approval_id = item['values'][0]

        # Get full approval request details
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            pending_approvals = central_post.get_pending_actions()

            approval_request = None
            for approval in pending_approvals:
                if approval['approval_id'] == approval_id:
                    approval_request = approval
                    break

            if not approval_request:
                messagebox.showerror(
                    "Error",
                    "Approval request not found. It may have expired or been processed."
                )
                return

            # Open approval dialog
            dialog = ApprovalDialog(
                self,
                approval_request,
                self._on_approval_decision
            )
            dialog.wait_window()

            # Refresh after dialog closes
            self._refresh_approvals()

        except Exception as e:
            logger.error(f"Error opening approval dialog: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to open approval dialog:\n{str(e)}"
            )

    def _quick_approve_selected(self):
        """Quick approve selected pending approval (approve once)."""
        from src.execution.approval_manager import ApprovalDecision

        selection = self.pending_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection",
                "Please select a pending approval to approve."
            )
            return

        item = self.pending_tree.item(selection[0])
        approval_id = item['values'][0]
        command = item['values'][1]

        confirm = messagebox.askyesno(
            "Confirm Quick Approve",
            f"This will approve and execute the command once:\n\n{command}\n\nContinue?"
        )

        if confirm:
            self._on_approval_decision(approval_id, ApprovalDecision.APPROVE_ONCE)
            self._refresh_approvals()

    def _quick_deny_selected(self):
        """Quick deny selected pending approval."""
        from src.execution.approval_manager import ApprovalDecision

        selection = self.pending_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection",
                "Please select a pending approval to deny."
            )
            return

        item = self.pending_tree.item(selection[0])
        approval_id = item['values'][0]
        command = item['values'][1]

        confirm = messagebox.askyesno(
            "Confirm Quick Deny",
            f"This will deny the command:\n\n{command}\n\nContinue?"
        )

        if confirm:
            self._on_approval_decision(approval_id, ApprovalDecision.DENY)
            self._refresh_approvals()

    def _on_approval_decision(self, approval_id: str, decision):
        """
        Handle approval decision from dialog or quick actions.

        Args:
            approval_id: Approval request ID
            decision: ApprovalDecision enum value
        """
        try:
            if not self.main_app or not self.main_app.felix_system:
                raise Exception("Felix system not available")

            central_post = self.main_app.felix_system.central_post

            # Process decision
            success = central_post.approve_system_action(
                approval_id=approval_id,
                decision=decision,
                decided_by="user"
            )

            if success:
                logger.info(f"Approval decision processed: {approval_id} -> {decision.value}")
            else:
                raise Exception("Failed to process approval decision")

        except Exception as e:
            logger.error(f"Error processing approval decision: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to process approval decision:\n{str(e)}"
            )

    def apply_theme(self):
        """Apply current theme to the frame."""
        if not self.theme_manager:
            return

        # Theme application handled by ThemeManager
        pass
