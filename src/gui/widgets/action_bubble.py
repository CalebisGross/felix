"""Action bubble widget for command approvals."""

from typing import Optional
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton, QSizePolicy
)

from ..core.theme import Colors


class ActionStatus(Enum):
    """Status of an action/command."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionBubble(QFrame):
    """Bubble for displaying and approving system commands.

    Shows a command with Approve/Deny buttons when pending,
    and status updates as execution progresses.

    Signals:
        approved: Emitted when Approve is clicked (action_id)
        denied: Emitted when Deny is clicked (action_id)
    """

    approved = Signal(str)  # action_id
    denied = Signal(str)  # action_id

    def __init__(
        self,
        action_id: str,
        command: str,
        description: str = "",
        risk_level: str = "low",
        status: ActionStatus = ActionStatus.PENDING,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._action_id = action_id
        self._command = command
        self._description = description
        self._risk_level = risk_level
        self._status = status
        self._output = ""
        self._setup_ui()

    def _setup_ui(self):
        """Set up the action bubble UI."""
        self.setObjectName("actionBubble")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Determine colors based on risk level
        if self._risk_level == "high":
            border_color = Colors.ERROR
            bg_color = "#2d1f1f"
        elif self._risk_level == "medium":
            border_color = Colors.WARNING
            bg_color = "#2d2a1f"
        else:
            border_color = Colors.ACCENT
            bg_color = Colors.SURFACE

        self.setStyleSheet(f"""
            QFrame#actionBubble {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header with icon and title
        header = QHBoxLayout()
        header.setSpacing(8)

        # Action icon/indicator
        icon_label = QLabel("⚡")
        icon_label.setStyleSheet(f"font-size: 14px;")
        header.addWidget(icon_label)

        title = QLabel("System Action")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-weight: 600;
            font-size: 13px;
        """)
        header.addWidget(title)

        # Risk badge
        risk_colors = {
            "high": Colors.ERROR,
            "medium": Colors.WARNING,
            "low": Colors.SUCCESS
        }
        risk_color = risk_colors.get(self._risk_level, Colors.TEXT_MUTED)

        risk_badge = QLabel(self._risk_level.upper())
        risk_badge.setStyleSheet(f"""
            color: {risk_color};
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border: 1px solid {risk_color};
            border-radius: 4px;
        """)
        header.addWidget(risk_badge)

        header.addStretch()

        # Status indicator
        self._status_label = QLabel()
        self._update_status_label()
        header.addWidget(self._status_label)

        layout.addLayout(header)

        # Command display
        command_frame = QFrame()
        command_frame.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-radius: 4px;
            padding: 4px;
        """)
        command_layout = QVBoxLayout(command_frame)
        command_layout.setContentsMargins(8, 6, 8, 6)

        command_label = QLabel(self._command)
        command_label.setWordWrap(True)
        command_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        command_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
        """)
        command_layout.addWidget(command_label)

        layout.addWidget(command_frame)

        # Description if provided
        if self._description:
            desc_label = QLabel(self._description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet(f"""
                color: {Colors.TEXT_SECONDARY};
                font-size: 12px;
            """)
            layout.addWidget(desc_label)

        # Output area (shown after execution)
        self._output_frame = QFrame()
        self._output_frame.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-radius: 4px;
        """)
        self._output_frame.hide()

        output_layout = QVBoxLayout(self._output_frame)
        output_layout.setContentsMargins(8, 6, 8, 6)

        output_header = QLabel("Output:")
        output_header.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        output_layout.addWidget(output_header)

        self._output_label = QLabel()
        self._output_label.setWordWrap(True)
        self._output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._output_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 11px;
        """)
        output_layout.addWidget(self._output_label)

        layout.addWidget(self._output_frame)

        # Action buttons (shown when pending)
        self._buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(self._buttons_widget)
        buttons_layout.setContentsMargins(0, 4, 0, 0)
        buttons_layout.setSpacing(8)

        buttons_layout.addStretch()

        self._deny_btn = QPushButton("Deny")
        self._deny_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._deny_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.ERROR};
                border: 1px solid {Colors.ERROR};
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.ERROR};
                color: white;
            }}
        """)
        self._deny_btn.clicked.connect(self._on_deny)
        buttons_layout.addWidget(self._deny_btn)

        self._approve_btn = QPushButton("Approve")
        self._approve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._approve_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: #1a1b26;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #a8d86a;
            }}
        """)
        self._approve_btn.clicked.connect(self._on_approve)
        buttons_layout.addWidget(self._approve_btn)

        layout.addWidget(self._buttons_widget)

        # Update visibility based on status
        self._update_ui_for_status()

    def _update_status_label(self):
        """Update the status indicator."""
        status_config = {
            ActionStatus.PENDING: ("Pending", Colors.WARNING),
            ActionStatus.APPROVED: ("Approved", Colors.SUCCESS),
            ActionStatus.DENIED: ("Denied", Colors.ERROR),
            ActionStatus.EXECUTING: ("Executing...", Colors.ACCENT),
            ActionStatus.COMPLETED: ("Completed", Colors.SUCCESS),
            ActionStatus.FAILED: ("Failed", Colors.ERROR),
        }

        text, color = status_config.get(
            self._status,
            ("Unknown", Colors.TEXT_MUTED)
        )

        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"""
            color: {color};
            font-size: 11px;
            font-weight: 500;
        """)

    def _update_ui_for_status(self):
        """Update UI elements based on current status."""
        # Show/hide buttons based on status
        show_buttons = self._status == ActionStatus.PENDING
        self._buttons_widget.setVisible(show_buttons)

        # Show output if completed or failed
        show_output = self._status in (
            ActionStatus.COMPLETED,
            ActionStatus.FAILED
        ) and self._output

        self._output_frame.setVisible(show_output)

    def _on_approve(self):
        """Handle approve button click."""
        self.set_status(ActionStatus.APPROVED)
        self.approved.emit(self._action_id)

    def _on_deny(self):
        """Handle deny button click."""
        self.set_status(ActionStatus.DENIED)
        self.denied.emit(self._action_id)

    def set_status(self, status: ActionStatus):
        """Update the action status."""
        self._status = status
        self._update_status_label()
        self._update_ui_for_status()

    def set_output(self, output: str):
        """Set the command output."""
        self._output = output
        self._output_label.setText(output[:500] + "..." if len(output) > 500 else output)
        self._update_ui_for_status()

    def get_action_id(self) -> str:
        """Get the action ID."""
        return self._action_id

    def get_status(self) -> ActionStatus:
        """Get current status."""
        return self._status
