"""Sidebar panel with system status, controls, and session list."""

from enum import Enum
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Signal, Slot, Qt, QPoint
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QListWidget, QListWidgetItem, QComboBox, QMenu
)

from ..core.theme import Colors
from ..models.session_model import Session


class UserMode(Enum):
    """User interface complexity modes."""
    CASUAL = "casual"
    POWER = "power"
    DEVELOPER = "developer"


class AgentPhaseBar(QFrame):
    """Visual indicator of agent distribution across phases."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._exploration = 0
        self._analysis = 0
        self._synthesis = 0
        self._setup_ui()

    def _setup_ui(self):
        """Set up the phase bar UI."""
        self.setFixedHeight(24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Exploration count
        self._exploration_badge = QLabel("E: 0")
        self._exploration_badge.setStyleSheet(f"""
            background-color: {Colors.STATUS_STARTING};
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        layout.addWidget(self._exploration_badge)

        # Analysis count
        self._analysis_badge = QLabel("A: 0")
        self._analysis_badge.setStyleSheet(f"""
            background-color: {Colors.ACCENT};
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        layout.addWidget(self._analysis_badge)

        # Synthesis count
        self._synthesis_badge = QLabel("S: 0")
        self._synthesis_badge.setStyleSheet(f"""
            background-color: {Colors.STATUS_RUNNING};
            color: white;
            font-size: 10px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
        """)
        layout.addWidget(self._synthesis_badge)

        layout.addStretch()

    def update_counts(self, exploration: int, analysis: int, synthesis: int):
        """Update phase counts."""
        self._exploration = exploration
        self._analysis = analysis
        self._synthesis = synthesis
        self._exploration_badge.setText(f"E: {exploration}")
        self._analysis_badge.setText(f"A: {analysis}")
        self._synthesis_badge.setText(f"S: {synthesis}")

    def clear(self):
        """Clear all counts."""
        self.update_counts(0, 0, 0)


class StatusCard(QFrame):
    """Card displaying system status information."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("statusCard")
        self._setup_ui()
        self._set_stopped_state()

    def _setup_ui(self):
        """Set up the status card UI."""
        self.setStyleSheet(f"""
            QFrame#statusCard {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                padding: 4px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Status indicator row
        status_row = QHBoxLayout()

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(10, 10)
        self._status_dot.setStyleSheet(f"""
            background-color: {Colors.STATUS_STOPPED};
            border-radius: 5px;
        """)
        status_row.addWidget(self._status_dot)

        self._status_label = QLabel("Stopped")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: 500;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()

        layout.addLayout(status_row)

        # Agent phase bar (shows when running)
        self._phase_bar = AgentPhaseBar()
        self._phase_bar.hide()  # Hidden when stopped
        layout.addWidget(self._phase_bar)

        # Stats grid
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(4)

        self._agents_label = QLabel("Agents: 0")
        self._agents_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        stats_layout.addWidget(self._agents_label)

        self._messages_label = QLabel("Messages: 0")
        self._messages_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        stats_layout.addWidget(self._messages_label)

        self._provider_label = QLabel("Provider: --")
        self._provider_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 12px;")
        stats_layout.addWidget(self._provider_label)

        layout.addLayout(stats_layout)

    def _set_stopped_state(self):
        """Set UI to stopped state."""
        self._status_dot.setStyleSheet(f"""
            background-color: {Colors.STATUS_STOPPED};
            border-radius: 5px;
        """)
        self._status_label.setText("Stopped")
        self._agents_label.setText("Agents: --")
        self._messages_label.setText("Messages: --")
        self._provider_label.setText("Provider: --")
        self._phase_bar.hide()
        self._phase_bar.clear()

    def _set_starting_state(self):
        """Set UI to starting state."""
        self._status_dot.setStyleSheet(f"""
            background-color: {Colors.STATUS_STARTING};
            border-radius: 5px;
        """)
        self._status_label.setText("Starting...")

    def _set_running_state(self):
        """Set UI to running state."""
        self._status_dot.setStyleSheet(f"""
            background-color: {Colors.STATUS_RUNNING};
            border-radius: 5px;
        """)
        self._status_label.setText("Running")
        self._phase_bar.show()

    @Slot(dict)
    def update_status(self, status: Dict[str, Any]):
        """Update status card with new data."""
        if status.get("running"):
            self._set_running_state()
            self._agents_label.setText(f"Agents: {status.get('agents', 0)}")
            self._messages_label.setText(f"Messages: {status.get('messages_processed', 0)}")

            provider = status.get("llm_provider", "unknown")
            if provider == "multi_provider_router":
                provider = "Router"
            elif provider == "lm_studio":
                provider = "LM Studio"
            self._provider_label.setText(f"Provider: {provider}")

            # Update agent phase counts if available
            exploration = status.get("agents_exploration", 0)
            analysis = status.get("agents_analysis", 0)
            synthesis = status.get("agents_synthesis", 0)
            self._phase_bar.update_counts(exploration, analysis, synthesis)
        else:
            self._set_stopped_state()

    def set_starting(self):
        """Set to starting state externally."""
        self._set_starting_state()


class SessionList(QWidget):
    """List of chat sessions."""

    session_selected = Signal(str)  # session_id
    new_session_requested = Signal()
    rename_requested = Signal(str)  # session_id
    delete_requested = Signal(str)  # session_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._sessions: Dict[str, Session] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the session list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # New Chat button (full width, prominent)
        new_btn = QPushButton("+ New Chat")
        new_btn.setFixedHeight(32)
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_HOVER};
            }}
        """)
        new_btn.clicked.connect(self.new_session_requested.emit)
        layout.addWidget(new_btn)

        # Sessions header
        header = QHBoxLayout()
        title = QLabel("Recent")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Session list
        self._list = QListWidget()
        self._list.setObjectName("sessionList")
        self._list.setStyleSheet(f"""
            QListWidget#sessionList {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget#sessionList::item {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border-radius: 6px;
                padding: 8px;
                margin: 2px 0;
            }}
            QListWidget#sessionList::item:hover {{
                background-color: {Colors.SURFACE};
            }}
            QListWidget#sessionList::item:selected {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._list)

        # Empty state
        self._empty_label = QLabel("No sessions yet")
        self._empty_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle session item click."""
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self.session_selected.emit(session_id)

    def _show_context_menu(self, pos: QPoint):
        """Show context menu for session item."""
        item = self._list.itemAt(pos)
        if not item:
            return

        session_id = item.data(Qt.ItemDataRole.UserRole)
        if not session_id:
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                color: {Colors.TEXT_PRIMARY};
                padding: 6px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.BACKGROUND_LIGHT};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER};
                margin: 4px 8px;
            }}
        """)

        rename_action = menu.addAction("Rename...")
        rename_action.triggered.connect(lambda: self.rename_requested.emit(session_id))

        menu.addSeparator()

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(session_id))

        menu.exec(self._list.mapToGlobal(pos))

    def add_session(self, session: Session):
        """Add a session to the list."""
        self._sessions[session.id] = session

        item = QListWidgetItem(session.title)
        item.setData(Qt.ItemDataRole.UserRole, session.id)
        item.setToolTip(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")

        self._list.insertItem(0, item)  # Insert at top
        self._update_empty_state()

    def update_session(self, session: Session):
        """Update a session in the list."""
        self._sessions[session.id] = session

        # Find and update the item
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == session.id:
                item.setText(session.title)
                # Move to top
                self._list.takeItem(i)
                self._list.insertItem(0, item)
                break

    def remove_session(self, session_id: str):
        """Remove a session from the list."""
        if session_id in self._sessions:
            del self._sessions[session_id]

        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                self._list.takeItem(i)
                break

        self._update_empty_state()

    def select_session(self, session_id: str):
        """Select a session by ID."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                self._list.setCurrentItem(item)
                break

    def load_sessions(self, sessions: List[Session]):
        """Load multiple sessions."""
        self._list.clear()
        self._sessions.clear()

        for session in sessions:
            self._sessions[session.id] = session
            item = QListWidgetItem(session.title)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            item.setToolTip(f"Created: {session.created_at.strftime('%Y-%m-%d %H:%M')}")
            self._list.addItem(item)

        self._update_empty_state()

    def _update_empty_state(self):
        """Show/hide empty state."""
        has_sessions = self._list.count() > 0
        self._list.setVisible(has_sessions)
        self._empty_label.setVisible(not has_sessions)

    def clear_selection(self):
        """Clear the current selection."""
        self._list.clearSelection()


class Sidebar(QWidget):
    """Left sidebar with status, controls, and sessions.

    Signals:
        start_requested: Emitted when Start button clicked
        stop_requested: Emitted when Stop button clicked
        session_selected: Emitted when a session is selected (session_id)
        new_session_requested: Emitted when New Session is requested
        rename_requested: Emitted when session rename is requested (session_id)
        delete_requested: Emitted when session delete is requested (session_id)
        mode_changed: Emitted when user mode changes (casual/power/developer)
    """

    start_requested = Signal()
    stop_requested = Signal()
    session_selected = Signal(str)  # session_id
    new_session_requested = Signal()
    rename_requested = Signal(str)  # session_id
    delete_requested = Signal(str)  # session_id
    mode_changed = Signal(str)  # "casual" | "power" | "developer"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)
        self._current_mode = UserMode.POWER  # Default to Power mode
        self._setup_ui()

    def _setup_ui(self):
        """Set up sidebar UI."""
        self.setStyleSheet(f"""
            QWidget#sidebar {{
                background-color: {Colors.BACKGROUND_LIGHT};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header row with title and mode selector
        header_layout = QHBoxLayout()

        # Logo/Title
        title = QLabel("Felix")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 20px;
            font-weight: 600;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Status card
        self._status_card = StatusCard()
        layout.addWidget(self._status_card)

        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self._start_btn = QPushButton("Start")
        self._start_btn.setProperty("success", True)
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.clicked.connect(self._on_start_clicked)
        controls_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        controls_layout.addWidget(self._stop_btn)

        layout.addLayout(controls_layout)

        # Separator before sessions
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {Colors.BORDER};")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        # Session list
        self._session_list = SessionList()
        self._session_list.session_selected.connect(self.session_selected.emit)
        self._session_list.new_session_requested.connect(self.new_session_requested.emit)
        self._session_list.rename_requested.connect(self.rename_requested.emit)
        self._session_list.delete_requested.connect(self.delete_requested.emit)
        layout.addWidget(self._session_list, 1)  # Give it stretch

    def _on_start_clicked(self):
        """Handle Start button click."""
        self._start_btn.setEnabled(False)
        self._status_card.set_starting()
        self.start_requested.emit()

    def _on_stop_clicked(self):
        """Handle Stop button click."""
        self._stop_btn.setEnabled(False)
        self.stop_requested.emit()

    @Slot()
    def on_system_started(self):
        """Handle system started event."""
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    @Slot()
    def on_system_stopped(self):
        """Handle system stopped event."""
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_card.update_status({"running": False})

    @Slot(str)
    def on_system_error(self, error: str):
        """Handle system error."""
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    @Slot(dict)
    def update_status(self, status: Dict[str, Any]):
        """Update status display."""
        self._status_card.update_status(status)

    def get_session_list(self) -> SessionList:
        """Get the session list widget."""
        return self._session_list

    def add_session(self, session: Session):
        """Add a session to the sidebar."""
        self._session_list.add_session(session)

    def update_session(self, session: Session):
        """Update a session in the sidebar."""
        self._session_list.update_session(session)

    def load_sessions(self, sessions: List[Session]):
        """Load sessions into the sidebar."""
        self._session_list.load_sessions(sessions)

    def select_session(self, session_id: str):
        """Select a session."""
        self._session_list.select_session(session_id)

    def remove_session(self, session_id: str):
        """Remove a session from the sidebar."""
        self._session_list.remove_session(session_id)

    # ========== Mode Management ==========

    def get_mode(self) -> UserMode:
        """Get current user mode."""
        return self._current_mode

    def set_mode(self, mode: UserMode):
        """Set user mode programmatically."""
        if self._current_mode != mode:
            self._current_mode = mode
            self.mode_changed.emit(mode.value)
