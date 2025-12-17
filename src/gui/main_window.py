"""Main window with 3-panel layout."""

import logging
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Slot, Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QLabel, QMessageBox, QStatusBar,
    QMenuBar, QMenu
)
from PySide6.QtGui import QShortcut, QKeySequence, QAction

from .core.theme import Colors, get_theme_manager
from .adapters.felix_adapter import FelixAdapter
from .panels.sidebar import Sidebar
from .panels.workspace import Workspace
from .panels.context_panel import ContextPanel
from .models.session_model import Session, SessionStore
from .models.message_model import Message, MessageRole

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window.

    Layout:
    +------------------+--------------------------------+------------------+
    |     SIDEBAR      |        MAIN WORKSPACE          |  CONTEXT PANEL   |
    |                  |                                |   (collapsible)  |
    | [System Status]  |  +-------------------------+   |                  |
    |  Running: Yes    |  |                         |   | [Terminal]       |
    |  Agents: 3       |  |    CHAT / WORKFLOW      |   |  cmd history     |
    |                  |  |                         |   |                  |
    | [Navigation]     |  |                         |   | [Knowledge]      |
    |  * Chat          |  |  streaming responses    |   |  KB browser      |
    |  * Knowledge     |  |                         |   |                  |
    | [Sessions]       |  +-------------------------+   | [Settings]       |
    |                  |  | INPUT + Mode Selector   |   |  quick access    |
    +------------------+--------------------------------+------------------+
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Felix")
        self.setMinimumSize(1024, 700)
        self.resize(1400, 850)

        # Create adapter
        self._adapter = FelixAdapter(self)

        # Session management
        self._session_store = SessionStore(parent=self)
        self._current_session: Optional[Session] = None

        # Set up UI
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_shortcuts()
        self._connect_signals()

        # Status bar
        self._setup_status_bar()

        # Load sessions
        self._load_sessions()

        logger.info("Main window initialized")

    def _setup_ui(self):
        """Set up the main window UI."""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar (fixed width)
        self._sidebar = Sidebar()
        layout.addWidget(self._sidebar)

        # Main splitter for workspace and context panel
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
        """)

        # Workspace (main area)
        self._workspace = Workspace()
        splitter.addWidget(self._workspace)

        # Context panel (collapsible right side)
        self._context_panel = ContextPanel()
        splitter.addWidget(self._context_panel)

        # Set splitter sizes (workspace gets more space)
        splitter.setSizes([900, 250])

        # Allow workspace to stretch, context to stay smaller
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout.addWidget(splitter, 1)

    def _setup_menu_bar(self):
        """Set up menu bar with View menu for developer mode."""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 1px solid {Colors.BORDER};
                padding: 2px;
            }}
            QMenuBar::item {{
                padding: 4px 8px;
                background-color: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: {Colors.SURFACE};
            }}
            QMenu {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
            }}
            QMenu::item {{
                padding: 6px 24px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER};
                margin: 4px 8px;
            }}
        """)

        # File menu
        file_menu = menubar.addMenu("&File")

        new_chat_action = QAction("&New Chat", self)
        new_chat_action.setShortcut(QKeySequence("Ctrl+N"))
        new_chat_action.triggered.connect(self._on_new_session)
        file_menu.addAction(new_chat_action)

        file_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._show_settings_dialog)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        toggle_context_action = QAction("Toggle &Context Panel", self)
        toggle_context_action.setShortcut(QKeySequence("Ctrl+T"))
        toggle_context_action.triggered.connect(self._toggle_context_panel)
        view_menu.addAction(toggle_context_action)

        view_menu.addSeparator()

        # Context panel tabs
        show_terminal_action = QAction("Show &Terminal", self)
        show_terminal_action.triggered.connect(self._context_panel.show_terminal)
        view_menu.addAction(show_terminal_action)

        show_knowledge_action = QAction("Show &Knowledge", self)
        show_knowledge_action.triggered.connect(self._context_panel.show_knowledge)
        view_menu.addAction(show_knowledge_action)

        show_settings_action = QAction("Show &Quick Settings", self)
        show_settings_action.triggered.connect(self._context_panel.show_settings)
        view_menu.addAction(show_settings_action)

        view_menu.addSeparator()

        # Theme toggle
        self._theme_action = QAction("Switch to &Light Theme", self)
        self._theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._theme_action)

        view_menu.addSeparator()

        # Developer mode submenu
        dev_menu = view_menu.addMenu("&Developer")

        self._agents_action = QAction("&Agents View", self)
        self._agents_action.triggered.connect(self._show_agents_view)
        dev_menu.addAction(self._agents_action)

        self._memory_action = QAction("&Memory View", self)
        self._memory_action.triggered.connect(self._show_memory_view)
        dev_menu.addAction(self._memory_action)

        self._prompts_action = QAction("&Prompts View", self)
        self._prompts_action.triggered.connect(self._show_prompts_view)
        dev_menu.addAction(self._prompts_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About Felix", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self):
        """Set up global keyboard shortcuts."""
        # Ctrl+N - New chat (already in menu)
        # Ctrl+, - Settings (already in menu)
        # Ctrl+T - Toggle context (already in menu)
        # Ctrl+Q - Quit (already in menu)

        # Additional shortcuts
        focus_input = QShortcut(QKeySequence("Ctrl+L"), self)
        focus_input.activated.connect(self._workspace.focus_input)

    def _toggle_context_panel(self):
        """Toggle context panel collapse state."""
        if self._context_panel.is_collapsed():
            self._context_panel._expand()
        else:
            self._context_panel._collapse()

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        theme_manager = get_theme_manager()
        theme_manager.toggle_theme()

        # Update menu text based on current theme
        if theme_manager.is_dark:
            self._theme_action.setText("Switch to &Light Theme")
        else:
            self._theme_action.setText("Switch to &Dark Theme")

    def _show_settings_dialog(self):
        """Show the full settings dialog."""
        from .dialogs.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self._adapter.felix_system, self)
        dialog.exec()

    def _show_agents_view(self):
        """Show agents developer view."""
        from .dialogs.developer_views import AgentsDialog
        dialog = AgentsDialog(self._adapter.felix_system, self)
        dialog.show()

    def _show_memory_view(self):
        """Show memory developer view."""
        from .dialogs.developer_views import MemoryDialog
        dialog = MemoryDialog(self._adapter.felix_system, self)
        dialog.show()

    def _show_prompts_view(self):
        """Show prompts developer view."""
        from .dialogs.developer_views import PromptsDialog
        dialog = PromptsDialog(self._adapter.felix_system, self)
        dialog.show()

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Felix",
            "Felix - Air-Gapped Multi-Agent AI Framework\n\n"
            "A production-ready framework for organizations\n"
            "requiring complete data isolation.\n\n"
            "Built with PySide6"
        )

    def _setup_status_bar(self):
        """Set up status bar."""
        status_bar = QStatusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT_MUTED};
                border-top: 1px solid {Colors.BORDER};
            }}
        """)
        self.setStatusBar(status_bar)
        status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect all signals."""
        # Sidebar controls -> Adapter
        self._sidebar.start_requested.connect(self._on_start_requested)
        self._sidebar.stop_requested.connect(self._adapter.stop_system)

        # Adapter -> Sidebar
        self._adapter.system_started.connect(self._sidebar.on_system_started)
        self._adapter.system_stopped.connect(self._sidebar.on_system_stopped)
        self._adapter.system_error.connect(self._sidebar.on_system_error)
        self._adapter.status_updated.connect(self._sidebar.update_status)

        # Adapter -> Workspace
        self._adapter.chunk_received.connect(self._on_chunk_received)
        self._adapter.response_complete.connect(self._on_response_complete)
        self._adapter.request_failed.connect(self._on_request_failed)

        # Workspace -> Adapter
        self._workspace.message_submitted.connect(self._on_message_submitted)
        self._workspace.stop_requested.connect(self._on_stop_generation)
        self._workspace.new_chat_requested.connect(self._on_new_chat)

        # Session management
        self._sidebar.session_selected.connect(self._on_session_selected)
        self._sidebar.new_session_requested.connect(self._on_new_session)

        # Adapter -> Context panel
        self._adapter.system_started.connect(self._on_system_started)
        self._adapter.system_stopped.connect(self._on_system_stopped)

        # Adapter -> Status bar
        self._adapter.system_started.connect(
            lambda: self.statusBar().showMessage("Felix running")
        )
        self._adapter.system_stopped.connect(
            lambda: self.statusBar().showMessage("Felix stopped")
        )

    @Slot()
    def _on_start_requested(self):
        """Handle start request from sidebar."""
        self.statusBar().showMessage("Starting Felix...")
        self._adapter.start_system()

    @Slot()
    def _on_system_started(self):
        """Handle system started - wire up context panel."""
        if self._adapter.felix_system:
            self._context_panel.set_felix_system(self._adapter.felix_system)

    @Slot()
    def _on_system_stopped(self):
        """Handle system stopped - clear context panel."""
        self._context_panel.clear_system()

    @Slot(str, str)
    def _on_message_submitted(self, message: str, mode: str):
        """Handle message submission from workspace."""
        if not self._adapter.is_running:
            QMessageBox.warning(
                self,
                "Not Running",
                "Please start Felix first using the Start button in the sidebar."
            )
            return

        # Create session if needed
        if not self._current_session:
            self._create_new_session()

        # Add user message to display and session
        self._workspace.add_message("user", message)
        self._save_message_to_session("user", message)
        self._workspace.clear_input()

        # Set processing state (shows stop button, disables input)
        self._workspace.set_processing(True)

        # Show typing indicator briefly, then start streaming
        self._workspace.show_typing()

        # Start streaming display
        self._workspace.start_streaming()

        # Send to adapter
        self._adapter.send_message(message, mode)

        self.statusBar().showMessage("Processing...")

    @Slot(str)
    def _on_chunk_received(self, chunk: str):
        """Handle streaming chunk from adapter."""
        self._workspace.append_chunk(chunk)

    @Slot(dict)
    def _on_response_complete(self, result: Dict[str, Any]):
        """Handle completed response."""
        content = self._workspace.end_streaming()
        self._workspace.set_processing(False)
        self._workspace.focus_input()

        # Save assistant message to session
        if content:
            self._save_message_to_session("assistant", content)

        # Update session title from first message
        if self._current_session:
            new_title = self._current_session.generate_title()
            if new_title != self._current_session.title:
                self._current_session.title = new_title
                self._session_store.update_session(self._current_session)
                self._sidebar.update_session(self._current_session)
                self._workspace.set_title(new_title)

        # Update status bar with result info
        mode_used = result.get("mode_used", "unknown")
        confidence = result.get("confidence", 0)
        self.statusBar().showMessage(
            f"Completed (mode: {mode_used}, confidence: {confidence:.0%})"
        )

    @Slot(str)
    def _on_request_failed(self, error: str):
        """Handle request failure."""
        self._workspace.end_streaming()
        self._workspace.set_processing(False)
        self._workspace.add_message("system", f"Error: {error}")

        self.statusBar().showMessage(f"Error: {error}")

    def closeEvent(self, event):
        """Handle window close."""
        if self._adapter.is_running:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Felix is still running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._adapter.stop_system()
                self._context_panel.cleanup()
                self._session_store.close()
                # Give it a moment to shut down
                from PySide6.QtCore import QTimer
                QTimer.singleShot(500, self.close)
                event.ignore()
                return
            else:
                event.ignore()
                return

        self._context_panel.cleanup()
        self._session_store.close()
        event.accept()

    # ========== Session Management ==========

    def _load_sessions(self):
        """Load sessions from database into sidebar."""
        sessions = self._session_store.get_all_sessions()
        self._sidebar.load_sessions(sessions)

        # If there are sessions, select the most recent one
        if sessions:
            self._on_session_selected(sessions[0].id)
        else:
            # Start with a fresh workspace
            self._workspace.set_title("New Chat")

    def _create_new_session(self) -> Session:
        """Create a new session and set it as current."""
        session = self._session_store.create_session()
        self._current_session = session
        self._sidebar.add_session(session)
        self._sidebar.select_session(session.id)
        self._workspace.set_title(session.title)
        return session

    @Slot()
    def _on_new_session(self):
        """Handle new session request from sidebar."""
        self._create_new_session()
        self._workspace.get_model().clear()
        # Clear the message area manually since we're not using new_chat_requested signal
        self._workspace._message_area.clear()

    @Slot()
    def _on_new_chat(self):
        """Handle new chat request from workspace."""
        self._create_new_session()

    @Slot(str)
    def _on_session_selected(self, session_id: str):
        """Handle session selection from sidebar."""
        session = self._session_store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        self._current_session = session
        self._workspace.set_title(session.title)

        # Clear and repopulate workspace
        self._workspace._message_area.clear()
        self._workspace.get_model().clear()

        for message in session.messages:
            if message.role == MessageRole.USER:
                self._workspace._message_area.add_user_bubble(message.content)
            elif message.role == MessageRole.ASSISTANT:
                self._workspace._message_area.add_assistant_bubble(message.content)
            else:
                self._workspace._message_area.add_system_bubble(message.content)

            # Also add to model for conversation history
            self._workspace.get_model()._messages.append(message)

    def _save_message_to_session(self, role: str, content: str):
        """Save a message to the current session."""
        if not self._current_session:
            return

        message = Message(
            role=MessageRole(role),
            content=content
        )
        self._current_session.messages.append(message)
        self._session_store.add_message(self._current_session.id, message)

    @Slot()
    def _on_stop_generation(self):
        """Handle stop generation request."""
        self._adapter.cancel_request()
        self._workspace.end_streaming()
        self._workspace.set_processing(False)
        self._workspace.add_message("system", "Generation stopped by user")
        self.statusBar().showMessage("Generation stopped")
