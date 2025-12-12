"""
Action Bubble Component for Felix Chat Interface

Displays inline system action approval requests with:
- Command preview (monospace)
- Trust level badge (SAFE/REVIEW/BLOCKED)
- Status indicator (pending/executing/complete/denied)
- Approve/Deny buttons
- Expandable output area after execution
"""

import customtkinter as ctk
from datetime import datetime
from typing import Optional, Callable, Any
from enum import Enum
import logging

from ...theme_manager import get_theme_manager
from ...styles import (
    FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD,
    RADIUS_MD, RADIUS_LG,
    BUTTON_SM
)

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an action request."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETE = "complete"
    DENIED = "denied"
    BLOCKED = "blocked"


# Trust level colors
TRUST_COLORS = {
    "safe": "#10B981",      # Green
    "review": "#F59E0B",    # Amber
    "blocked": "#EF4444",   # Red
}

# Status indicators
STATUS_INDICATORS = {
    ActionStatus.PENDING: ("Pending", "#9CA3AF"),      # Gray
    ActionStatus.EXECUTING: ("Executing...", "#F59E0B"),  # Amber
    ActionStatus.COMPLETE: ("Complete", "#10B981"),    # Green
    ActionStatus.DENIED: ("Denied", "#EF4444"),        # Red
    ActionStatus.BLOCKED: ("Blocked", "#EF4444"),      # Red
}


class ActionBubble(ctk.CTkFrame):
    """
    Inline system action approval widget for chat interface.

    Displays a command that Felix wants to run, with options to
    approve or deny based on trust level.

    Usage:
        bubble = ActionBubble(
            parent,
            command="cat /path/to/file",
            trust_level="review",
            on_approve=handle_approve,
            on_deny=handle_deny
        )
    """

    def __init__(
        self,
        master,
        command: str,
        trust_level: str,
        on_approve: Optional[Callable[["ActionBubble", str], None]] = None,
        on_deny: Optional[Callable[["ActionBubble", str], None]] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        """
        Initialize action bubble.

        Args:
            master: Parent widget
            command: The command Felix wants to execute
            trust_level: Trust classification ('safe', 'review', 'blocked')
            on_approve: Callback when Approve is clicked (bubble, command)
            on_deny: Callback when Deny is clicked (bubble, command)
            timestamp: When the action was requested
            **kwargs: Additional arguments for CTkFrame
        """
        self.theme_manager = get_theme_manager()
        self.command = command
        self.trust_level = trust_level.lower()
        self.on_approve = on_approve
        self.on_deny = on_deny
        self.timestamp = timestamp or datetime.now()
        self._status = ActionStatus.BLOCKED if self.trust_level == "blocked" else ActionStatus.PENDING
        self._output_expanded = False

        # Get colors
        colors = self.theme_manager.colors
        bg_color = colors["bg_tertiary"]

        super().__init__(
            master,
            fg_color=bg_color,
            corner_radius=RADIUS_LG,
            **kwargs
        )

        self._bg_color = bg_color
        self._setup_ui()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the UI components."""
        self.grid_columnconfigure(0, weight=1)
        current_row = 0

        # Header: "Felix wants to run:"
        header_frame = ctk.CTkFrame(self, fg_color=self._bg_color, corner_radius=0)
        header_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header_frame.grid_columnconfigure(0, weight=1)

        header_label = ctk.CTkLabel(
            header_frame,
            text="Felix wants to run:",
            font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            fg_color=self._bg_color,
            anchor="w"
        )
        header_label.grid(row=0, column=0, sticky="w")

        time_label = ctk.CTkLabel(
            header_frame,
            text=self.timestamp.strftime("%H:%M"),
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self.theme_manager.get_color("fg_muted"),
            fg_color=self._bg_color,
            anchor="e"
        )
        time_label.grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))
        current_row += 1

        # Command display (monospace, dark background)
        command_frame = ctk.CTkFrame(
            self,
            fg_color=self.theme_manager.get_color("bg_primary"),
            corner_radius=RADIUS_MD
        )
        command_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_XS)
        command_frame.grid_columnconfigure(0, weight=1)

        # Truncate long commands for display
        display_command = self.command
        if len(display_command) > 100:
            display_command = display_command[:97] + "..."

        self._command_label = ctk.CTkLabel(
            command_frame,
            text=display_command,
            font=ctk.CTkFont(family="monospace", size=FONT_BODY),
            text_color="#FFFFFF",
            fg_color=self.theme_manager.get_color("bg_primary"),
            anchor="w",
            justify="left",
            padx=SPACE_SM,
            pady=SPACE_XS
        )
        self._command_label.grid(row=0, column=0, sticky="ew")
        current_row += 1

        # Trust badge and status row
        status_frame = ctk.CTkFrame(self, fg_color=self._bg_color, corner_radius=0)
        status_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_XS)
        status_frame.grid_columnconfigure(1, weight=1)

        # Trust level badge
        trust_color = TRUST_COLORS.get(self.trust_level, TRUST_COLORS["review"])
        self._trust_badge = ctk.CTkLabel(
            status_frame,
            text=f"[{self.trust_level.upper()}]",
            font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
            text_color=trust_color,
            fg_color=self._bg_color,
            anchor="w"
        )
        self._trust_badge.grid(row=0, column=0, sticky="w")

        # Status indicator
        status_text, status_color = STATUS_INDICATORS[self._status]
        self._status_label = ctk.CTkLabel(
            status_frame,
            text=f"  {status_text}",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=status_color,
            fg_color=self._bg_color,
            anchor="w"
        )
        self._status_label.grid(row=0, column=1, sticky="w", padx=(SPACE_XS, 0))
        current_row += 1

        # Button row
        button_frame = ctk.CTkFrame(self, fg_color=self._bg_color, corner_radius=0)
        button_frame.grid(row=current_row, column=0, sticky="e", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        # Deny button (always shown except after completion)
        self._deny_btn = ctk.CTkButton(
            button_frame,
            text="Deny",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_CAPTION),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            border_width=1,
            border_color=self.theme_manager.get_color("border"),
            command=self._on_deny_clicked
        )
        self._deny_btn.grid(row=0, column=0, padx=(0, SPACE_XS))

        # Approve button (hidden for BLOCKED)
        self._approve_btn = ctk.CTkButton(
            button_frame,
            text="Approve",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_CAPTION),
            fg_color=TRUST_COLORS.get(self.trust_level, TRUST_COLORS["review"]),
            hover_color=self._darken_color(TRUST_COLORS.get(self.trust_level, TRUST_COLORS["review"])),
            text_color="#FFFFFF",
            command=self._on_approve_clicked
        )
        self._approve_btn.grid(row=0, column=1)

        # Hide approve button for blocked commands
        if self.trust_level == "blocked":
            self._approve_btn.grid_remove()
            self._deny_btn.configure(text="Dismiss")

        current_row += 1

        # Output area (hidden initially)
        self._output_frame = ctk.CTkFrame(
            self,
            fg_color=self.theme_manager.get_color("bg_primary"),
            corner_radius=RADIUS_MD
        )
        self._output_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._output_frame.grid_columnconfigure(0, weight=1)
        self._output_frame.grid_remove()  # Hidden by default

        # Output toggle header
        self._output_toggle = ctk.CTkButton(
            self._output_frame,
            text="Output",
            font=ctk.CTkFont(size=FONT_SMALL),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            anchor="w",
            command=self._toggle_output
        )
        self._output_toggle.grid(row=0, column=0, sticky="w", padx=SPACE_XS, pady=SPACE_XS)

        # Output content
        self._output_content = ctk.CTkTextbox(
            self._output_frame,
            height=100,
            font=ctk.CTkFont(family="monospace", size=FONT_SMALL),
            fg_color=self.theme_manager.get_color("bg_primary"),
            text_color=self.theme_manager.get_color("fg_primary"),
            wrap="word",
            state="disabled"
        )
        self._output_content.grid(row=1, column=0, sticky="ew", padx=SPACE_XS, pady=(0, SPACE_XS))

        # Bind mousewheel events for scroll passthrough
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel)
        self.bind("<Button-5>", self._on_mousewheel)

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color for hover state."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
        factor = 0.8
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_approve_clicked(self):
        """Handle approve button click."""
        if self._status != ActionStatus.PENDING:
            return

        logger.info(f"ActionBubble: Approve clicked for: {self.command[:50]}")

        if self.on_approve:
            self.on_approve(self, self.command)

    def _on_deny_clicked(self):
        """Handle deny button click."""
        if self._status not in (ActionStatus.PENDING, ActionStatus.BLOCKED):
            return

        logger.info(f"ActionBubble: Deny clicked for: {self.command[:50]}")

        if self.trust_level == "blocked":
            self.set_status(ActionStatus.BLOCKED)
        else:
            self.set_status(ActionStatus.DENIED)

        if self.on_deny:
            self.on_deny(self, self.command)

    def set_status(self, status: ActionStatus, output: Optional[str] = None, exit_code: Optional[int] = None):
        """
        Update the action status and optionally show output.

        Args:
            status: New status
            output: Command output to display (for COMPLETE status)
            exit_code: Command exit code (for COMPLETE status)
        """
        self._status = status
        status_text, status_color = STATUS_INDICATORS[status]

        # Add exit code to status if provided
        if exit_code is not None and status == ActionStatus.COMPLETE:
            if exit_code == 0:
                status_text = f"Complete (exit 0)"
            else:
                status_text = f"Failed (exit {exit_code})"
                status_color = "#EF4444"  # Red for non-zero exit

        self._status_label.configure(text=f"  {status_text}", text_color=status_color)

        # Update button visibility
        if status in (ActionStatus.EXECUTING, ActionStatus.COMPLETE, ActionStatus.DENIED, ActionStatus.BLOCKED):
            self._approve_btn.grid_remove()
            self._deny_btn.grid_remove()

        # Show output if provided
        if output is not None and status == ActionStatus.COMPLETE:
            self._show_output(output)

    def _show_output(self, output: str):
        """Display command output."""
        # Truncate very long output
        if len(output) > 5000:
            output = output[:5000] + "\n... (output truncated)"

        self._output_content.configure(state="normal")
        self._output_content.delete("1.0", "end")
        self._output_content.insert("1.0", output or "(no output)")
        self._output_content.configure(state="disabled")

        # Show output frame
        self._output_frame.grid()
        self._output_expanded = True

    def _toggle_output(self):
        """Toggle output visibility."""
        self._output_expanded = not self._output_expanded

        if self._output_expanded:
            self._output_content.grid()
            self._output_toggle.configure(text="Output")
        else:
            self._output_content.grid_remove()
            self._output_toggle.configure(text="Output")

    def _on_mousewheel(self, event):
        """Pass mousewheel events to parent for scrolling."""
        # Find the scrollable parent and forward the event
        widget = self.master
        while widget:
            if hasattr(widget, '_parent_canvas'):
                widget._parent_canvas.event_generate('<MouseWheel>', delta=event.delta if hasattr(event, 'delta') else 0)
                break
            widget = getattr(widget, 'master', None)

    def _on_theme_change(self, mode: str):
        """Handle theme change.

        Args:
            mode: New theme mode ('dark' or 'light')
        """
        colors = self.theme_manager.colors
        self._bg_color = colors["bg_tertiary"]

        self.configure(fg_color=self._bg_color)
        # Update child widget colors as needed

    @property
    def status(self) -> ActionStatus:
        """Get current action status."""
        return self._status

    def destroy(self):
        """Cleanup when destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass  # Ignore errors during cleanup
        super().destroy()
