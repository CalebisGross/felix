"""
Approvals Tab for Felix GUI (CustomTkinter Edition)

Provides system command approval management with:
- Pending approvals section with risk-level color coding
- Quick action buttons (Review, Approve All, Deny All)
- Approval history with filtering and search
- Real-time polling for new pending approvals
"""

import customtkinter as ctk
from tkinter import messagebox
import logging
from datetime import datetime
from typing import Dict, Any

from ..utils import logger, ThreadManager
from ..theme_manager import get_theme_manager
from ..components.themed_treeview import ThemedTreeview
from ..components.search_entry import SearchEntry
from ..dialogs.approval_dialog import ApprovalDialog

# Import Felix modules (with fallback)
try:
    from src.execution.approval_manager import ApprovalDecision
except ImportError:
    ApprovalDecision = None
    logger.warning("ApprovalDecision not available")


class ApprovalsTab(ctk.CTkFrame):
    """
    Tab showing approval request history and current pending approvals.

    Features:
    - Real-time display of pending approval requests
    - History of all approval decisions with search/filter
    - Quick approve/deny actions from list
    - Risk-level color coding
    - Automatic polling for new approvals
    """

    def __init__(self, master, thread_manager: ThreadManager, main_app=None, **kwargs):
        """
        Initialize Approvals tab.

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
        self.polling_active = False
        self.poll_interval = 2000  # Poll every 2 seconds

        self._setup_ui()

    def _setup_ui(self):
        """Set up the approvals tab UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Pending section
        self.grid_rowconfigure(3, weight=1)  # History section

        # Header section
        self._create_header()

        # Pending approvals section
        self._create_pending_section()

        # Separator
        separator = ctk.CTkFrame(self, height=2, fg_color=self.theme_manager.get_color("border"))
        separator.grid(row=2, column=0, sticky="ew", padx=20, pady=15)

        # History section
        self._create_history_section()

    def _create_header(self):
        """Create the header section with title and controls."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_frame,
            text="System Command Approvals",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_frame,
            text="Manage pending approvals and view approval history",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        ).pack(anchor="w")

        # Control buttons
        control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_frame.grid(row=0, column=2, sticky="e")

        self.refresh_button = ctk.CTkButton(
            control_frame,
            text="Refresh",
            command=self._refresh_approvals,
            width=90,
            height=32,
            state="disabled",
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.refresh_button.pack(side="left", padx=5)

        self.auto_refresh_var = ctk.BooleanVar(value=False)
        self.auto_refresh_check = ctk.CTkCheckBox(
            control_frame,
            text="Auto-refresh (2s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh,
            state="disabled"
        )
        self.auto_refresh_check.pack(side="left", padx=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="System not running",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.status_label.pack(side="left", padx=(15, 0))

    def _create_pending_section(self):
        """Create the pending approvals section."""
        pending_frame = ctk.CTkFrame(self)
        pending_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 5))
        pending_frame.grid_columnconfigure(0, weight=1)
        pending_frame.grid_rowconfigure(1, weight=1)

        # Section header
        header = ctk.CTkFrame(pending_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header,
            text="Pending Approvals",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left")

        self.pending_count_label = ctk.CTkLabel(
            header,
            text="(0)",
            font=ctk.CTkFont(size=12),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.pending_count_label.pack(side="left", padx=(5, 0))

        # TreeView for pending approvals
        self.pending_tree = ThemedTreeview(
            pending_frame,
            columns=["approval_id", "command", "agent", "trust_level", "workflow", "requested_at"],
            headings=["Approval ID", "Command", "Agent", "Trust Level", "Workflow", "Requested At"],
            widths=[120, 300, 150, 100, 150, 150],
            height=8
        )
        self.pending_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # Double-click to review
        self.pending_tree.bind_tree("<Double-1>", self._on_pending_double_click)

        # Action buttons
        action_frame = ctk.CTkFrame(pending_frame, fg_color="transparent")
        action_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.review_button = ctk.CTkButton(
            action_frame,
            text="Review Selected",
            command=self._review_selected,
            width=130,
            height=32,
            state="disabled"
        )
        self.review_button.pack(side="left", padx=(0, 5))

        self.approve_all_button = ctk.CTkButton(
            action_frame,
            text="Approve All",
            command=self._approve_all,
            width=110,
            height=32,
            state="disabled",
            fg_color=self.theme_manager.get_color("success"),
            hover_color="#1e8449"
        )
        self.approve_all_button.pack(side="left", padx=5)

        self.deny_all_button = ctk.CTkButton(
            action_frame,
            text="Deny All",
            command=self._deny_all,
            width=110,
            height=32,
            state="disabled",
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226"
        )
        self.deny_all_button.pack(side="left", padx=5)

    def _create_history_section(self):
        """Create the approval history section."""
        history_frame = ctk.CTkFrame(self)
        history_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(5, 20))
        history_frame.grid_columnconfigure(0, weight=1)
        history_frame.grid_rowconfigure(2, weight=1)

        # Section header with search
        header = ctk.CTkFrame(history_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Approval History",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        # Search entry
        self.search_entry = SearchEntry(
            header,
            placeholder="Search history...",
            width=250
        )
        self.search_entry.grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.search_entry.bind("<Return>", lambda e: self._search_history())

        # Filter buttons
        filter_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
        filter_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            filter_frame,
            text="Filter:",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(0, 10))

        self.filter_var = ctk.StringVar(value="all")

        filters = [
            ("all", "All"),
            ("approved", "Approved"),
            ("denied", "Denied"),
            ("expired", "Expired"),
            ("auto_approved", "Auto-Approved")
        ]

        for value, text in filters:
            radio = ctk.CTkRadioButton(
                filter_frame,
                text=text,
                variable=self.filter_var,
                value=value,
                command=self._refresh_history
            )
            radio.pack(side="left", padx=5)

        # TreeView for history
        self.history_tree = ThemedTreeview(
            history_frame,
            columns=["timestamp", "command", "decision", "result", "workflow"],
            headings=["Timestamp", "Command", "Decision", "Result", "Workflow"],
            widths=[150, 300, 180, 100, 150],
            height=8
        )
        self.history_tree.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # History action buttons
        history_action_frame = ctk.CTkFrame(history_frame, fg_color="transparent")
        history_action_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.refresh_history_button = ctk.CTkButton(
            history_action_frame,
            text="Refresh History",
            command=self._refresh_history,
            width=130,
            height=32,
            state="disabled",
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray90")
        )
        self.refresh_history_button.pack(side="left", padx=(0, 5))

        ctk.CTkLabel(
            history_action_frame,
            text="Showing last 50 approvals",
            font=ctk.CTkFont(size=10),
            text_color=self.theme_manager.get_color("fg_muted")
        ).pack(side="left", padx=(10, 0))

        # Configure tag colors for different statuses
        self._configure_tree_tags()

    def _configure_tree_tags(self):
        """Configure color tags for tree items."""
        # Trust level colors for pending tree
        self.pending_tree.tag_configure("trust_high", foreground=self.theme_manager.get_color("success"))
        self.pending_tree.tag_configure("trust_review", foreground=self.theme_manager.get_color("warning"))
        self.pending_tree.tag_configure("trust_low", foreground=self.theme_manager.get_color("error"))

        # Result colors for history tree
        self.history_tree.tag_configure("approved", foreground=self.theme_manager.get_color("success"))
        self.history_tree.tag_configure("denied", foreground=self.theme_manager.get_color("error"))
        self.history_tree.tag_configure("expired", foreground=self.theme_manager.get_color("fg_muted"))
        self.history_tree.tag_configure("auto_approved", foreground=self.theme_manager.get_color("accent"))

    def _enable_features(self):
        """Enable approval features when system is running."""
        self.refresh_button.configure(state="normal")
        self.auto_refresh_check.configure(state="normal")
        self.review_button.configure(state="normal")
        self.approve_all_button.configure(state="normal")
        self.deny_all_button.configure(state="normal")
        self.refresh_history_button.configure(state="normal")

        self.status_label.configure(
            text="Ready",
            text_color=self.theme_manager.get_color("success")
        )

        # Start auto-refresh if enabled
        if self.auto_refresh_var.get():
            self._start_polling()

        # Initial data load
        self._refresh_approvals()

    def _disable_features(self):
        """Disable approval features when system is not running."""
        self.refresh_button.configure(state="disabled")
        self.auto_refresh_check.configure(state="disabled")
        self.review_button.configure(state="disabled")
        self.approve_all_button.configure(state="disabled")
        self.deny_all_button.configure(state="disabled")
        self.refresh_history_button.configure(state="disabled")

        self.status_label.configure(
            text="System not running",
            text_color=self.theme_manager.get_color("fg_muted")
        )

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
            self.pending_tree.clear()

            # Add pending approvals
            for approval in pending_approvals:
                requested_at = datetime.fromtimestamp(approval['requested_at']).strftime("%Y-%m-%d %H:%M:%S")
                command_display = approval['command'][:50] + "..." if len(approval['command']) > 50 else approval['command']

                # Determine tag based on trust level
                trust_level = approval.get('trust_level', 'review')
                tag = f"trust_{trust_level}"

                item_id = self.pending_tree.insert(
                    "",
                    "end",
                    values=(
                        approval['approval_id'],
                        command_display,
                        approval['agent_id'],
                        trust_level.upper(),
                        approval.get('workflow_id', 'N/A'),
                        requested_at
                    ),
                    tags=(tag,)
                )

            # Update count and status
            count = len(pending_approvals)
            self.pending_count_label.configure(text=f"({count})")

            if count > 0:
                self.status_label.configure(
                    text=f"{count} pending approval{'s' if count != 1 else ''}",
                    text_color=self.theme_manager.get_color("warning")
                )
            else:
                self.status_label.configure(
                    text="No pending approvals",
                    text_color=self.theme_manager.get_color("success")
                )

            # Also refresh history
            self._refresh_history()

        except Exception as e:
            logger.error(f"Error refreshing approvals: {e}", exc_info=True)
            self.status_label.configure(
                text="Error refreshing approvals",
                text_color=self.theme_manager.get_color("error")
            )

    def _refresh_history(self):
        """Refresh the approval history list."""
        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post

            # Get approval manager
            if not hasattr(central_post, 'approval_manager') or not central_post.approval_manager:
                logger.warning("Approval manager not available")
                return

            approval_history = central_post.approval_manager.get_approval_history(limit=50)

            # Clear current items
            self.history_tree.clear()

            # Filter by selected filter
            filter_value = self.filter_var.get()

            # Add history entries
            for approval in approval_history:
                # Apply filter
                if filter_value != "all" and approval.status.value != filter_value:
                    continue

                # Format timestamp
                timestamp = approval.decided_at if approval.decided_at else approval.expires_at
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

                # Format decision
                decision_str = approval.decision.value if approval.decision else "expired"

                # Format result based on status
                status_value = approval.status.value
                if status_value == "approved":
                    result_str = "✓ Approved"
                    tag = "approved"
                elif status_value == "denied":
                    result_str = "✗ Denied"
                    tag = "denied"
                elif status_value == "expired":
                    result_str = "⏱ Expired"
                    tag = "expired"
                elif status_value == "auto_approved":
                    result_str = "⚡ Auto"
                    tag = "auto_approved"
                else:
                    result_str = status_value
                    tag = "approved"

                # Truncate command for display
                command_display = approval.command[:50] + "..." if len(approval.command) > 50 else approval.command

                self.history_tree.insert(
                    "",
                    "end",
                    values=(
                        timestamp_str,
                        command_display,
                        decision_str,
                        result_str,
                        approval.workflow_id or "N/A"
                    ),
                    tags=(tag,)
                )

            logger.debug(f"Refreshed approval history: {len(approval_history)} entries")

        except Exception as e:
            logger.error(f"Error refreshing approval history: {e}", exc_info=True)

    def _search_history(self):
        """Search approval history based on search entry."""
        # For now, just refresh - in future could implement actual filtering
        self._refresh_history()

    def _on_pending_double_click(self, event):
        """Handle double-click on pending approval."""
        self._review_selected()

    def _review_selected(self):
        """Open approval dialog for selected pending approval."""
        selection = self.pending_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Selection",
                "Please select a pending approval to review.",
                parent=self
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
                    "Approval request not found. It may have expired or been processed.",
                    parent=self
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
                f"Failed to open approval dialog:\n{str(e)}",
                parent=self
            )

    def _approve_all(self):
        """Quick approve all pending approvals (approve once)."""
        if not ApprovalDecision:
            messagebox.showerror("Error", "Approval system not available", parent=self)
            return

        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            pending_approvals = central_post.get_pending_actions()

            if not pending_approvals:
                messagebox.showinfo(
                    "No Approvals",
                    "There are no pending approvals.",
                    parent=self
                )
                return

            count = len(pending_approvals)
            confirm = messagebox.askyesno(
                "Confirm Approve All",
                f"This will approve and execute {count} pending command(s).\n\n"
                f"Each command will be approved ONCE only.\n\n"
                f"Continue?",
                parent=self
            )

            if not confirm:
                return

            # Approve all
            for approval in pending_approvals:
                self._on_approval_decision(
                    approval['approval_id'],
                    ApprovalDecision.APPROVE_ONCE
                )

            self._refresh_approvals()
            messagebox.showinfo(
                "Success",
                f"Approved {count} command(s).",
                parent=self
            )

        except Exception as e:
            logger.error(f"Error approving all: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to approve all:\n{str(e)}",
                parent=self
            )

    def _deny_all(self):
        """Quick deny all pending approvals."""
        if not ApprovalDecision:
            messagebox.showerror("Error", "Approval system not available", parent=self)
            return

        try:
            if not self.main_app or not self.main_app.felix_system:
                return

            central_post = self.main_app.felix_system.central_post
            pending_approvals = central_post.get_pending_actions()

            if not pending_approvals:
                messagebox.showinfo(
                    "No Approvals",
                    "There are no pending approvals.",
                    parent=self
                )
                return

            count = len(pending_approvals)
            confirm = messagebox.askyesno(
                "Confirm Deny All",
                f"This will deny {count} pending command(s).\n\n"
                f"The workflow will continue without executing these commands.\n\n"
                f"Continue?",
                parent=self
            )

            if not confirm:
                return

            # Deny all
            for approval in pending_approvals:
                self._on_approval_decision(
                    approval['approval_id'],
                    ApprovalDecision.DENY
                )

            self._refresh_approvals()
            messagebox.showinfo(
                "Success",
                f"Denied {count} command(s).",
                parent=self
            )

        except Exception as e:
            logger.error(f"Error denying all: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to deny all:\n{str(e)}",
                parent=self
            )

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
                f"Failed to process approval decision:\n{str(e)}",
                parent=self
            )
