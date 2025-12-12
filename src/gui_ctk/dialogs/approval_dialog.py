"""
Approval Dialog for Felix GUI (CustomTkinter Edition)

Popup dialog for user to approve/deny system commands with multiple approval options.
"""

import customtkinter as ctk
from tkinter import messagebox
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


class ApprovalDialog(ctk.CTkToplevel):
    """
    Popup dialog for approving system commands.

    Displays command details, risk assessment, and approval options:
    - Approve Once
    - Approve Always (Exact)
    - Approve Always (Command Type)
    - Approve Always (Path Pattern)
    - Deny
    """

    def __init__(self, parent, approval_request: Dict[str, Any], on_decision_callback: Callable):
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
        self.geometry("750x750")
        self.resizable(True, True)

        # Make modal
        self.transient(parent)

        # Build UI
        self._build_ui()

        # Center and focus
        self.update_idletasks()
        self.minsize(750, 750)
        self._center_window()

        # Delay grab to prevent "window not viewable" error
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
        """Grab focus after window is fully rendered."""
        try:
            self.grab_set()
            self.focus_set()
            logger.debug("✓ Dialog focus grabbed successfully")
        except Exception as e:
            logger.warning(f"Could not grab dialog focus: {e}")

    def _build_ui(self):
        """Build the approval dialog UI."""

        # Main container with scrolling
        main_container = ctk.CTkScrollableFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_label = ctk.CTkLabel(
            main_container,
            text="⚠️ System Command Requires Approval",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        header_label.pack(anchor="w", pady=(0, 5))

        subtitle_label = ctk.CTkLabel(
            main_container,
            text="An agent has requested permission to execute a system command.",
            font=ctk.CTkFont(size=11)
        )
        subtitle_label.pack(anchor="w", pady=(0, 15))

        # Command details section
        details_frame = ctk.CTkFrame(main_container)
        details_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            details_frame,
            text="Command Details",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        # Command
        ctk.CTkLabel(
            details_frame,
            text="Command:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 0))

        command_text = ctk.CTkTextbox(details_frame, height=60, wrap="word")
        command_text.pack(fill="x", padx=10, pady=(0, 10))
        command_text.insert("1.0", self.approval_request['command'])
        command_text.configure(state="disabled")

        # Agent
        agent_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        agent_frame.pack(fill="x", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            agent_frame,
            text="Requested by:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", padx=(0, 10))

        ctk.CTkLabel(
            agent_frame,
            text=self.approval_request['agent_id']
        ).pack(side="left")

        # Trust level
        trust_frame = ctk.CTkFrame(details_frame, fg_color="transparent")
        trust_frame.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            trust_frame,
            text="Trust Level:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(side="left", padx=(0, 10))

        trust_level = self.approval_request['trust_level'].upper()
        trust_color = "#ff8c00" if self.approval_request['trust_level'] == 'review' else None
        ctk.CTkLabel(
            trust_frame,
            text=trust_level,
            text_color=trust_color
        ).pack(side="left")

        # Context (if available)
        if self.approval_request.get('context'):
            ctk.CTkLabel(
                details_frame,
                text="Context:",
                font=ctk.CTkFont(size=11, weight="bold")
            ).pack(anchor="w", padx=10, pady=(0, 0))

            context_text = ctk.CTkTextbox(details_frame, height=50, wrap="word")
            context_text.pack(fill="x", padx=10, pady=(0, 10))
            context_text.insert("1.0", self.approval_request['context'])
            context_text.configure(state="disabled")

        # Risk assessment section
        risk_frame = ctk.CTkFrame(main_container)
        risk_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            risk_frame,
            text="Risk Assessment",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        risk_text = ctk.CTkTextbox(risk_frame, height=50, wrap="word")
        risk_text.pack(fill="x", padx=10, pady=(0, 10))
        risk_assessment = self.approval_request.get('risk_assessment', 'Modifies system state - requires user confirmation.')
        risk_text.insert("1.0", risk_assessment)
        risk_text.configure(state="disabled")

        # Approval options section
        options_frame = ctk.CTkFrame(main_container)
        options_frame.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(
            options_frame,
            text="Approval Options",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            options_frame,
            text="Choose how to handle this command:",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Decision variable
        self.decision_var = ctk.StringVar(value="approve_once")

        # Radio buttons with descriptions
        decisions = [
            ("approve_once", "Approve Once", "Execute this command one time only"),
            ("approve_always_exact", "Always Approve (Exact)", "Always approve this exact command in this workflow"),
            ("approve_always_command", "Always Approve (Command Type)", "Always approve this command type (e.g., all 'mkdir') in this workflow"),
            ("approve_always_path_pattern", "Always Approve (Path Pattern)", "Always approve commands in this path pattern in this workflow"),
            ("deny", "Deny", "Reject this command and continue workflow")
        ]

        for value, label, description in decisions:
            radio = ctk.CTkRadioButton(
                options_frame,
                text=label,
                variable=self.decision_var,
                value=value
            )
            radio.pack(anchor="w", padx=10, pady=(0, 2))

            desc_label = ctk.CTkLabel(
                options_frame,
                text=f"  └─ {description}",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            desc_label.pack(anchor="w", padx=30, pady=(0, 5))

        # Button frame (fixed at bottom)
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", side="bottom", padx=10, pady=10)

        # Cancel button (left)
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel (Deny)",
            command=self._on_cancel,
            width=140,
            fg_color="gray40",
            hover_color="gray30"
        )
        cancel_btn.pack(side="left", padx=(0, 5))

        # Submit button (right)
        submit_btn = ctk.CTkButton(
            button_frame,
            text="Submit Decision",
            command=self._on_submit,
            width=140
        )
        submit_btn.pack(side="right")

        # Focus submit button
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
