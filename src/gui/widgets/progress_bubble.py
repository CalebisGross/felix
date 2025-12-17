"""Progress bubble widget for workflow progress display."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Signal, Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QProgressBar, QPushButton, QSizePolicy
)
from PySide6.QtGui import QFont

from ..core.theme import Colors


class WorkflowStepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a workflow step."""
    name: str
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    description: str = ""
    agent: str = ""
    duration: float = 0.0


class ProgressBubble(QFrame):
    """Widget for displaying workflow progress inline.

    Shows:
    - Current workflow step
    - Progress bar
    - Step list with status indicators
    - Agent activity
    """

    cancelled = Signal()  # Emitted when user cancels

    def __init__(self, title: str = "Processing...", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = title
        self._steps: List[WorkflowStep] = []
        self._current_step_index = -1
        self._is_expanded = False
        self._is_indeterminate = True

        self._setup_ui()
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_indicator)
        self._pulse_timer.start(100)

    def _setup_ui(self):
        """Set up the progress bubble UI."""
        self.setObjectName("progressBubble")
        self.setStyleSheet(f"""
            QFrame#progressBubble {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
                margin: 4px 60px 4px 16px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header row
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        # Activity indicator
        self._indicator = QLabel("●")
        self._indicator.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 12px;")
        header_row.addWidget(self._indicator)

        # Title
        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-weight: 600;
            font-size: 14px;
        """)
        header_row.addWidget(self._title_label)

        header_row.addStretch()

        # Expand/collapse button
        self._expand_btn = QPushButton("▼")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: none;
                font-size: 10px;
            }}
            QPushButton:hover {{
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self._expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._expand_btn.clicked.connect(self._toggle_expand)
        header_row.addWidget(self._expand_btn)

        layout.addLayout(header_row)

        # Current step label
        self._step_label = QLabel("Initializing...")
        self._step_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self._step_label)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BACKGROUND_LIGHT};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self._progress_bar)

        # Expandable steps container
        self._steps_container = QFrame()
        self._steps_container.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND_LIGHT};
            border-radius: 6px;
            padding: 4px;
        """)
        self._steps_layout = QVBoxLayout(self._steps_container)
        self._steps_layout.setContentsMargins(8, 8, 8, 8)
        self._steps_layout.setSpacing(4)
        self._steps_container.hide()
        layout.addWidget(self._steps_container)

        # Agent activity label
        self._agent_label = QLabel("")
        self._agent_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        self._agent_label.hide()
        layout.addWidget(self._agent_label)

    def _toggle_expand(self):
        """Toggle expanded/collapsed state."""
        self._is_expanded = not self._is_expanded
        self._steps_container.setVisible(self._is_expanded)
        self._expand_btn.setText("▲" if self._is_expanded else "▼")

    def _pulse_indicator(self):
        """Pulse the activity indicator."""
        current = self._indicator.styleSheet()
        if Colors.ACCENT in current:
            self._indicator.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px;")
        else:
            self._indicator.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 12px;")

    def set_title(self, title: str):
        """Set the progress title."""
        self._title = title
        self._title_label.setText(title)

    def set_current_step(self, step: str, description: str = ""):
        """Set the current step being executed."""
        self._step_label.setText(step)
        if description:
            self._step_label.setToolTip(description)

    def set_agent_activity(self, agent: str, activity: str = ""):
        """Show agent activity."""
        if agent:
            text = f"Agent: {agent}"
            if activity:
                text += f" - {activity}"
            self._agent_label.setText(text)
            self._agent_label.show()
        else:
            self._agent_label.hide()

    def set_progress(self, current: int, total: int):
        """Set determinate progress."""
        if self._is_indeterminate:
            self._progress_bar.setRange(0, total)
            self._is_indeterminate = False
        self._progress_bar.setValue(current)

    def set_indeterminate(self):
        """Set indeterminate progress."""
        self._progress_bar.setRange(0, 0)
        self._is_indeterminate = True

    def add_step(self, name: str, description: str = "", agent: str = ""):
        """Add a workflow step."""
        step = WorkflowStep(name=name, description=description, agent=agent)
        self._steps.append(step)
        self._update_steps_display()

    def update_step(self, index: int, status: WorkflowStepStatus, duration: float = 0.0):
        """Update a step's status."""
        if 0 <= index < len(self._steps):
            self._steps[index].status = status
            self._steps[index].duration = duration
            self._update_steps_display()

    def start_step(self, index: int):
        """Mark a step as running."""
        if 0 <= index < len(self._steps):
            self._current_step_index = index
            self._steps[index].status = WorkflowStepStatus.RUNNING
            self.set_current_step(
                self._steps[index].name,
                self._steps[index].description
            )
            if self._steps[index].agent:
                self.set_agent_activity(self._steps[index].agent)
            self._update_steps_display()

    def complete_step(self, index: int, duration: float = 0.0):
        """Mark a step as completed."""
        self.update_step(index, WorkflowStepStatus.COMPLETED, duration)

    def fail_step(self, index: int):
        """Mark a step as failed."""
        self.update_step(index, WorkflowStepStatus.FAILED)

    def _update_steps_display(self):
        """Update the expanded steps display."""
        # Clear existing
        while self._steps_layout.count():
            item = self._steps_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add step rows
        for i, step in enumerate(self._steps):
            row = QHBoxLayout()
            row.setSpacing(8)

            # Status icon
            status_icons = {
                WorkflowStepStatus.PENDING: ("○", Colors.TEXT_MUTED),
                WorkflowStepStatus.RUNNING: ("●", Colors.ACCENT),
                WorkflowStepStatus.COMPLETED: ("✓", Colors.SUCCESS),
                WorkflowStepStatus.FAILED: ("✗", Colors.ERROR),
                WorkflowStepStatus.SKIPPED: ("−", Colors.TEXT_MUTED),
            }
            icon, color = status_icons.get(step.status, ("○", Colors.TEXT_MUTED))

            icon_label = QLabel(icon)
            icon_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            icon_label.setFixedWidth(16)
            row.addWidget(icon_label)

            # Step name
            name_label = QLabel(step.name)
            name_color = Colors.TEXT_PRIMARY if step.status == WorkflowStepStatus.RUNNING else Colors.TEXT_SECONDARY
            name_label.setStyleSheet(f"color: {name_color}; font-size: 12px;")
            row.addWidget(name_label, 1)

            # Duration (if completed)
            if step.status == WorkflowStepStatus.COMPLETED and step.duration > 0:
                duration_label = QLabel(f"{step.duration:.1f}s")
                duration_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
                row.addWidget(duration_label)

            # Agent (if specified)
            if step.agent:
                agent_label = QLabel(f"[{step.agent}]")
                agent_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
                row.addWidget(agent_label)

            row_widget = QWidget()
            row_widget.setLayout(row)
            self._steps_layout.addWidget(row_widget)

    def set_complete(self, success: bool = True):
        """Mark the entire workflow as complete."""
        self._pulse_timer.stop()

        if success:
            self._indicator.setText("✓")
            self._indicator.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px;")
            self._title_label.setText("Completed")
            self._step_label.setText("Workflow finished successfully")
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(100)
        else:
            self._indicator.setText("✗")
            self._indicator.setStyleSheet(f"color: {Colors.ERROR}; font-size: 14px;")
            self._title_label.setText("Failed")
            self._step_label.setText("Workflow encountered an error")
            self._progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {Colors.BACKGROUND_LIGHT};
                    border: none;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background-color: {Colors.ERROR};
                    border-radius: 2px;
                }}
            """)

        self._agent_label.hide()

    def cleanup(self):
        """Clean up resources."""
        self._pulse_timer.stop()


class CompactProgressIndicator(QFrame):
    """Compact inline progress indicator for simple status updates."""

    def __init__(self, text: str = "Processing...", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui(text)

        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(500)

    def _setup_ui(self, text: str):
        """Set up UI."""
        self.setStyleSheet(f"""
            background-color: {Colors.SURFACE_LIGHT};
            border-radius: 8px;
            margin: 4px 16px;
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Animated dots
        self._dots = QLabel("●●●")
        self._dots.setStyleSheet(f"color: {Colors.ACCENT}; font-size: 10px;")
        layout.addWidget(self._dots)

        # Text
        self._label = QLabel(text)
        self._label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        layout.addWidget(self._label)

        layout.addStretch()

        self._dot_state = 0

    def _pulse(self):
        """Animate the dots."""
        dots = ["●○○", "○●○", "○○●", "○●○"]
        self._dot_state = (self._dot_state + 1) % len(dots)
        self._dots.setText(dots[self._dot_state])

    def set_text(self, text: str):
        """Update the text."""
        self._label.setText(text)

    def cleanup(self):
        """Clean up resources."""
        self._pulse_timer.stop()
