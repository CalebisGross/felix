"""Unified activity view showing workflow, agents, and pending actions."""

import logging
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QProgressBar, QTreeWidget, QTreeWidgetItem
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class CollapsibleSection(QFrame):
    """A collapsible section with header and content."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._is_collapsed = False
        self._title = title
        self._setup_ui()

    def _setup_ui(self):
        """Set up the section UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        self._header = QFrame()
        self._header.setFixedHeight(32)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE_LIGHT};
                border: none;
                border-radius: 6px 6px 0 0;
            }}
        """)

        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 0, 8, 0)

        self._toggle_btn = QLabel("▼")
        self._toggle_btn.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        header_layout.addWidget(self._toggle_btn)

        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 12px;
            font-weight: 600;
        """)
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        # Badge for counts
        self._badge = QLabel("")
        self._badge.setStyleSheet(f"""
            background-color: {Colors.ACCENT};
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 8px;
        """)
        self._badge.hide()
        header_layout.addWidget(self._badge)

        layout.addWidget(self._header)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(4)
        layout.addWidget(self._content)

        # Click handling
        self._header.mousePressEvent = self._on_header_clicked

    def _on_header_clicked(self, event):
        """Toggle collapse on header click."""
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)
        self._toggle_btn.setText("▶" if self._is_collapsed else "▼")

    def get_content_layout(self) -> QVBoxLayout:
        """Get the content layout for adding widgets."""
        return self._content_layout

    def set_badge(self, count: int):
        """Set badge count. Hide if 0."""
        if count > 0:
            self._badge.setText(str(count))
            self._badge.show()
        else:
            self._badge.hide()

    def set_badge_color(self, color: str):
        """Set badge background color."""
        self._badge.setStyleSheet(f"""
            background-color: {color};
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 8px;
        """)


class WorkflowProgressWidget(QWidget):
    """Widget showing current workflow progress."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up progress UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Phase indicator
        phase_layout = QHBoxLayout()

        self._phase_label = QLabel("Idle")
        self._phase_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 12px;
            font-weight: 500;
        """)
        phase_layout.addWidget(self._phase_label)

        phase_layout.addStretch()

        self._step_label = QLabel("")
        self._step_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        phase_layout.addWidget(self._step_label)

        layout.addLayout(phase_layout)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BACKGROUND_LIGHT};
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT};
                border-radius: 3px;
            }}
        """)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        # Status message
        self._status_label = QLabel("No active workflow")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._status_label)

    def set_workflow_active(self, active: bool):
        """Set whether a workflow is active."""
        if not active:
            self._phase_label.setText("Idle")
            self._step_label.setText("")
            self._progress_bar.setValue(0)
            self._status_label.setText("No active workflow")

    def update_progress(self, phase: str, current_step: int, total_steps: int, status: str = ""):
        """Update workflow progress."""
        self._phase_label.setText(phase.title())
        self._step_label.setText(f"Step {current_step}/{total_steps}")

        if total_steps > 0:
            progress = int((current_step / total_steps) * 100)
            self._progress_bar.setValue(progress)
        else:
            self._progress_bar.setValue(0)

        if status:
            self._status_label.setText(status)


class AgentListWidget(QWidget):
    """Widget showing active agents."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up agent list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Phase summary bar
        self._phase_bar = QFrame()
        self._phase_bar.setFixedHeight(20)
        phase_layout = QHBoxLayout(self._phase_bar)
        phase_layout.setContentsMargins(0, 0, 0, 0)
        phase_layout.setSpacing(4)

        self._exploration_count = QLabel("E: 0")
        self._exploration_count.setStyleSheet(f"""
            background-color: {Colors.STATUS_STARTING};
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        phase_layout.addWidget(self._exploration_count)

        self._analysis_count = QLabel("A: 0")
        self._analysis_count.setStyleSheet(f"""
            background-color: {Colors.ACCENT};
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        phase_layout.addWidget(self._analysis_count)

        self._synthesis_count = QLabel("S: 0")
        self._synthesis_count.setStyleSheet(f"""
            background-color: {Colors.STATUS_RUNNING};
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        phase_layout.addWidget(self._synthesis_count)

        phase_layout.addStretch()
        layout.addWidget(self._phase_bar)

        # Agent tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        self._tree.setMaximumHeight(150)
        self._tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: transparent;
                border: none;
                font-size: 11px;
            }}
            QTreeWidget::item {{
                padding: 2px 0;
            }}
            QTreeWidget::item:selected {{
                background-color: {Colors.SURFACE_LIGHT};
            }}
        """)
        layout.addWidget(self._tree)

        # Empty state
        self._empty_label = QLabel("No active agents")
        self._empty_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

    def update_agents(self, agents: List[Dict[str, Any]]):
        """Update agent list."""
        self._tree.clear()
        self._agents.clear()

        exploration = 0
        analysis = 0
        synthesis = 0

        for agent in agents:
            agent_id = agent.get("id", "unknown")
            agent_type = agent.get("type", "Agent")
            phase = agent.get("phase", "exploration").lower()
            confidence = agent.get("confidence", 0.0)

            self._agents[agent_id] = agent

            # Count by phase
            if phase == "exploration":
                exploration += 1
            elif phase == "analysis":
                analysis += 1
            elif phase == "synthesis":
                synthesis += 1

            # Create tree item
            item = QTreeWidgetItem()
            phase_icon = {"exploration": "🔍", "analysis": "📊", "synthesis": "✨"}.get(phase, "•")
            item.setText(0, f"{phase_icon} {agent_type} ({confidence:.0%})")
            self._tree.addTopLevelItem(item)

        # Update counts
        self._exploration_count.setText(f"E: {exploration}")
        self._analysis_count.setText(f"A: {analysis}")
        self._synthesis_count.setText(f"S: {synthesis}")

        # Show/hide empty state
        has_agents = len(agents) > 0
        self._tree.setVisible(has_agents)
        self._empty_label.setVisible(not has_agents)


class PendingApprovalsWidget(QWidget):
    """Widget showing pending command approvals."""

    approved = Signal(str)  # approval_id
    denied = Signal(str)    # approval_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._approvals: Dict[str, Dict[str, Any]] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up approvals UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Approvals container
        self._approvals_layout = QVBoxLayout()
        self._approvals_layout.setSpacing(4)
        layout.addLayout(self._approvals_layout)

        # Empty state
        self._empty_label = QLabel("No pending approvals")
        self._empty_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

    def add_approval(self, approval_id: str, command: str, risk_level: str = "medium"):
        """Add a pending approval."""
        if approval_id in self._approvals:
            return

        # Create approval card
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BACKGROUND_LIGHT};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        # Command preview
        cmd_label = QLabel(command[:50] + "..." if len(command) > 50 else command)
        cmd_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-family: monospace;
            font-size: 11px;
        """)
        cmd_label.setToolTip(command)
        card_layout.addWidget(cmd_label)

        # Risk level indicator
        risk_colors = {
            "low": Colors.STATUS_RUNNING,
            "medium": Colors.STATUS_STARTING,
            "high": Colors.STATUS_STOPPED
        }
        risk_label = QLabel(f"Risk: {risk_level.upper()}")
        risk_label.setStyleSheet(f"""
            color: {risk_colors.get(risk_level, Colors.TEXT_MUTED)};
            font-size: 10px;
            font-weight: 600;
        """)
        card_layout.addWidget(risk_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        approve_btn = QPushButton("Approve")
        approve_btn.setFixedHeight(24)
        approve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        approve_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.STATUS_RUNNING};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                padding: 0 8px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SUCCESS};
            }}
        """)
        approve_btn.clicked.connect(lambda: self._on_approve(approval_id))
        btn_layout.addWidget(approve_btn)

        deny_btn = QPushButton("Deny")
        deny_btn.setFixedHeight(24)
        deny_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        deny_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.STATUS_STOPPED};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 11px;
                padding: 0 8px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ERROR};
            }}
        """)
        deny_btn.clicked.connect(lambda: self._on_deny(approval_id))
        btn_layout.addWidget(deny_btn)

        card_layout.addLayout(btn_layout)

        self._approvals[approval_id] = {"card": card, "command": command}
        self._approvals_layout.addWidget(card)
        self._update_empty_state()

    def remove_approval(self, approval_id: str):
        """Remove an approval."""
        if approval_id in self._approvals:
            card = self._approvals[approval_id]["card"]
            self._approvals_layout.removeWidget(card)
            card.deleteLater()
            del self._approvals[approval_id]
            self._update_empty_state()

    def _on_approve(self, approval_id: str):
        """Handle approve button click."""
        self.approved.emit(approval_id)
        self.remove_approval(approval_id)

    def _on_deny(self, approval_id: str):
        """Handle deny button click."""
        self.denied.emit(approval_id)
        self.remove_approval(approval_id)

    def _update_empty_state(self):
        """Update empty state visibility."""
        has_approvals = len(self._approvals) > 0
        self._empty_label.setVisible(not has_approvals)

    def get_count(self) -> int:
        """Get number of pending approvals."""
        return len(self._approvals)


class ActivityView(QWidget):
    """Unified activity view showing workflow, agents, and pending actions.

    Signals:
        approval_approved: Emitted when user approves a command
        approval_denied: Emitted when user denies a command
    """

    approval_approved = Signal(str)  # approval_id
    approval_denied = Signal(str)    # approval_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._felix_system = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up activity view UI."""
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.BACKGROUND};
                border: none;
            }}
        """)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Workflow Progress Section
        self._workflow_section = CollapsibleSection("Workflow Progress")
        self._workflow_widget = WorkflowProgressWidget()
        self._workflow_section.get_content_layout().addWidget(self._workflow_widget)
        layout.addWidget(self._workflow_section)

        # Active Agents Section
        self._agents_section = CollapsibleSection("Active Agents")
        self._agents_widget = AgentListWidget()
        self._agents_section.get_content_layout().addWidget(self._agents_widget)
        layout.addWidget(self._agents_section)

        # Pending Approvals Section
        self._approvals_section = CollapsibleSection("Pending Approvals")
        self._approvals_section.set_badge_color(Colors.STATUS_STOPPED)
        self._approvals_widget = PendingApprovalsWidget()
        self._approvals_widget.approved.connect(self.approval_approved.emit)
        self._approvals_widget.denied.connect(self.approval_denied.emit)
        self._approvals_section.get_content_layout().addWidget(self._approvals_widget)
        layout.addWidget(self._approvals_section)

        layout.addStretch()
        scroll.setWidget(content)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def set_felix_system(self, felix_system):
        """Set Felix system reference.

        Note: Polling has been removed. ActivityView now receives updates
        via the update_from_status() slot connected to adapter.status_updated.
        """
        self._felix_system = felix_system
        if not felix_system:
            # Clear display when system disconnected
            self._agents_widget.update_agents([])
            self._agents_section.set_badge(0)
            self._workflow_widget.set_workflow_active(False)

    @Slot(dict)
    def update_from_status(self, status: dict):
        """Update activity view from unified system status.

        This is the single source of truth for agent data, called via
        adapter.status_updated signal to ensure consistency with sidebar.

        Args:
            status: Status dict from FelixSystem.get_system_status()
        """
        if not status.get("running"):
            self._agents_widget.update_agents([])
            self._agents_section.set_badge(0)
            return

        # Update agents from unified agents_list field
        agents_list = status.get("agents_list", [])
        self._agents_widget.update_agents(agents_list)
        self._agents_section.set_badge(len(agents_list))

    # ========== Public API ==========

    @Slot(str, int, int, str)
    def update_workflow_progress(self, phase: str, current: int, total: int, status: str = ""):
        """Update workflow progress display."""
        self._workflow_widget.update_progress(phase, current, total, status)
        self._workflow_section.set_badge(1 if current < total else 0)

    @Slot()
    def clear_workflow(self):
        """Clear workflow progress."""
        self._workflow_widget.set_workflow_active(False)
        self._workflow_section.set_badge(0)

    @Slot(str, str, str)
    def add_pending_approval(self, approval_id: str, command: str, risk_level: str = "medium"):
        """Add a pending approval."""
        self._approvals_widget.add_approval(approval_id, command, risk_level)
        self._approvals_section.set_badge(self._approvals_widget.get_count())

    @Slot(str)
    def remove_pending_approval(self, approval_id: str):
        """Remove a pending approval."""
        self._approvals_widget.remove_approval(approval_id)
        self._approvals_section.set_badge(self._approvals_widget.get_count())

    def cleanup(self):
        """Clean up resources."""
        pass  # No resources to clean up - polling removed
