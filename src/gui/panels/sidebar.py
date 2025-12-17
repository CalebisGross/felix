"""Sidebar panel with system status, controls, and session list."""

from typing import Optional, Dict, Any, List

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QSizePolicy,
    QListWidget, QListWidgetItem
)

from ..core.theme import Colors
from ..models.session_model import Session


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
        else:
            self._set_stopped_state()

    def set_starting(self):
        """Set to starting state externally."""
        self._set_starting_state()


class SessionList(QWidget):
    """List of chat sessions."""

    session_selected = Signal(str)  # session_id
    new_session_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._sessions: Dict[str, Session] = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up the session list UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header with New button
        header = QHBoxLayout()

        title = QLabel("Sessions")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        header.addWidget(title)

        header.addStretch()

        new_btn = QPushButton("+")
        new_btn.setFixedSize(24, 24)
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setToolTip("New Chat")
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-size: 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        new_btn.clicked.connect(self.new_session_requested.emit)
        header.addWidget(new_btn)

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
    """

    start_requested = Signal()
    stop_requested = Signal()
    session_selected = Signal(str)  # session_id
    new_session_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)
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
        layout.setSpacing(16)

        # Logo/Title
        title = QLabel("Felix")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 20px;
            font-weight: 600;
        """)
        layout.addWidget(title)

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

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {Colors.BORDER};")
        separator.setFixedHeight(1)
        layout.addWidget(separator)

        # Navigation section
        nav_label = QLabel("Navigation")
        nav_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(nav_label)

        # Nav buttons
        self._chat_btn = self._create_nav_button("Chat", active=True)
        layout.addWidget(self._chat_btn)

        # Separator before sessions
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet(f"background-color: {Colors.BORDER};")
        separator2.setFixedHeight(1)
        layout.addWidget(separator2)

        # Session list
        self._session_list = SessionList()
        self._session_list.session_selected.connect(self.session_selected.emit)
        self._session_list.new_session_requested.connect(self.new_session_requested.emit)
        layout.addWidget(self._session_list, 1)  # Give it stretch

    def _create_nav_button(self, text: str, active: bool = False) -> QPushButton:
        """Create a navigation button."""
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setCheckable(True)
        btn.setChecked(active)

        base_style = f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_SECONDARY};
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                text-align: left;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE};
            }}
            QPushButton:checked {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
            }}
        """
        btn.setStyleSheet(base_style)
        return btn

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
