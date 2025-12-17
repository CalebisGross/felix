"""Main window with 3-panel layout."""

import logging
from enum import Enum
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Slot, Qt, QSettings, Signal
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QLabel, QMessageBox, QStatusBar,
    QMenuBar, QMenu, QInputDialog, QLineEdit
)
from PySide6.QtGui import QShortcut, QKeySequence, QAction, QResizeEvent

from .core.theme import Colors, get_theme_manager
from .core.signals import get_signals
from .adapters.felix_adapter import FelixAdapter
from .panels.sidebar import Sidebar
from .panels.workspace import Workspace
from .panels.context_panel import ContextPanel
from .models.session_model import Session, SessionStore
from .models.message_model import Message, MessageRole

logger = logging.getLogger(__name__)


class LayoutMode(Enum):
    """Layout modes based on window width."""
    COMPACT = "compact"      # < 900px - auto-collapse context panel
    STANDARD = "standard"    # 900-1400px - normal layout
    WIDE = "wide"            # > 1400px - extra space available


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

    # Signal emitted when layout mode changes due to window resize
    layout_mode_changed = Signal(str)  # "compact" | "standard" | "wide"

    # Breakpoints for responsive layout
    COMPACT_BREAKPOINT = 900
    WIDE_BREAKPOINT = 1400

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Felix")
        self.setMinimumSize(800, 600)  # Allow smaller windows

        # Settings for persistence
        self._settings = QSettings("Felix", "FelixGUI")

        # Layout tracking
        self._current_layout_mode = LayoutMode.STANDARD
        self._context_was_collapsed_by_resize = False  # Track if we auto-collapsed

        # Create adapter
        self._adapter = FelixAdapter(self)

        # Session management
        self._session_store = SessionStore(parent=self)
        self._current_session: Optional[Session] = None

        # Pending config from settings (applied when Felix starts)
        self._pending_config: Dict[str, Any] = {}

        # Set up UI
        self._setup_ui()
        self._setup_menu_bar()
        self._setup_shortcuts()
        self._connect_signals()

        # Status bar
        self._setup_status_bar()

        # Restore window state from settings
        self._restore_layout()

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
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(6)  # Wider handle for easier grabbing
        self._splitter.setChildrenCollapsible(False)  # Prevent accidental collapse
        self._splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {Colors.ACCENT};
            }}
            QSplitter::handle:pressed {{
                background-color: {Colors.ACCENT_HOVER};
            }}
        """)

        # Workspace (main area)
        self._workspace = Workspace()
        self._splitter.addWidget(self._workspace)

        # Context panel (collapsible right side)
        self._context_panel = ContextPanel()
        self._splitter.addWidget(self._context_panel)

        # Set splitter sizes (workspace gets more space)
        self._splitter.setSizes([900, 320])

        # Allow workspace to stretch, context panel can be resized by user
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        layout.addWidget(self._splitter, 1)

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

        # Context panel tabs with keyboard shortcuts
        show_activity_action = QAction("Show &Activity", self)
        show_activity_action.setShortcut(QKeySequence("Ctrl+1"))
        show_activity_action.triggered.connect(self._context_panel.show_activity)
        view_menu.addAction(show_activity_action)

        show_terminal_action = QAction("Show &Terminal", self)
        show_terminal_action.setShortcut(QKeySequence("Ctrl+2"))
        show_terminal_action.triggered.connect(self._context_panel.show_terminal)
        view_menu.addAction(show_terminal_action)

        show_knowledge_action = QAction("Show &Knowledge", self)
        show_knowledge_action.setShortcut(QKeySequence("Ctrl+3"))
        show_knowledge_action.triggered.connect(self._context_panel.show_knowledge)
        view_menu.addAction(show_knowledge_action)

        show_learning_action = QAction("Show &Learning", self)
        show_learning_action.setShortcut(QKeySequence("Ctrl+4"))
        show_learning_action.triggered.connect(self._context_panel.show_learning)
        view_menu.addAction(show_learning_action)

        show_settings_action = QAction("Show &Quick Settings", self)
        show_settings_action.setShortcut(QKeySequence("Ctrl+5"))
        show_settings_action.triggered.connect(self._context_panel.show_settings)
        view_menu.addAction(show_settings_action)

        view_menu.addSeparator()

        # Theme toggle
        self._theme_action = QAction("Switch to &Light Theme", self)
        self._theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._theme_action)

        view_menu.addSeparator()

        # Developer mode submenu with shortcuts
        dev_menu = view_menu.addMenu("&Developer")

        self._agents_action = QAction("&Agents View", self)
        self._agents_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self._agents_action.triggered.connect(self._show_agents_view)
        dev_menu.addAction(self._agents_action)

        self._memory_action = QAction("&Memory View", self)
        self._memory_action.setShortcut(QKeySequence("Ctrl+Shift+M"))
        self._memory_action.triggered.connect(self._show_memory_view)
        dev_menu.addAction(self._memory_action)

        self._prompts_action = QAction("&Prompts View", self)
        self._prompts_action.setShortcut(QKeySequence("Ctrl+Shift+P"))
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
        self._sidebar.rename_requested.connect(self._on_rename_session)
        self._sidebar.delete_requested.connect(self._on_delete_session)

        # Adapter -> Context panel
        self._adapter.system_started.connect(self._on_system_started)
        self._adapter.system_stopped.connect(self._on_system_stopped)
        # Unified status updates to ActivityView (same data source as sidebar)
        self._adapter.status_updated.connect(
            self._context_panel.get_activity_view().update_from_status
        )

        # Settings panel -> Store pending config for next start
        self._context_panel.settings_changed.connect(self._on_settings_changed)

        # Adapter -> Status bar
        self._adapter.system_started.connect(
            lambda: self.statusBar().showMessage("Felix running")
        )
        self._adapter.system_stopped.connect(
            lambda: self.statusBar().showMessage("Felix stopped")
        )

        # Approval handling - FelixSignals -> MainWindow -> Workspace and ActivityView
        signals = get_signals()
        signals.approval_requested.connect(self._on_approval_requested)
        signals.approval_resolved.connect(self._on_approval_resolved)

        # Workspace approval signals -> Adapter response
        # Use lowercase to match ApprovalDecision enum values
        self._workspace.action_approved.connect(
            lambda aid: self._adapter.respond_to_approval(aid, 'approve_once')
        )
        self._workspace.action_denied.connect(
            lambda aid: self._adapter.respond_to_approval(aid, 'deny')
        )

        # Status updates with workflow state -> ActivityView
        self._adapter.status_updated.connect(self._on_status_updated_for_activity)

    @Slot(str, dict)
    def _on_approval_requested(self, approval_id: str, details: dict):
        """Handle incoming approval request - show in both Workspace and ActivityView."""
        logger.info(f"Approval requested: {approval_id} - {details.get('command', 'unknown')}")
        # Show in workspace
        self._workspace.add_action(
            action_id=approval_id,
            command=details.get('command', ''),
            description=details.get('context', ''),
            risk_level=details.get('risk_level', 'MEDIUM')
        )
        # Also show in activity view
        self._context_panel.get_activity_view().add_pending_approval(
            approval_id,
            details.get('command', ''),
            details.get('risk_level', 'MEDIUM')
        )

    @Slot(str, str)
    def _on_approval_resolved(self, approval_id: str, decision: str):
        """Handle approval resolution - update ActivityView."""
        logger.info(f"Approval resolved: {approval_id} -> {decision}")
        self._context_panel.get_activity_view().remove_pending_approval(approval_id)

    @Slot(dict)
    def _on_status_updated_for_activity(self, status: Dict[str, Any]):
        """Handle status updates for ActivityView workflow progress."""
        workflow_state = status.get('workflow_state', {})
        phase = workflow_state.get('phase', 'idle')

        if phase != 'idle':
            self._context_panel.get_activity_view().update_workflow_progress(
                phase,
                workflow_state.get('current_step', 0),
                workflow_state.get('total_steps', 0),
                workflow_state.get('status_message', '')
            )

    @Slot(dict)
    def _on_settings_changed(self, settings: Dict[str, Any]):
        """Handle settings changes from SettingsView.

        Maps GUI settings keys to FelixConfig parameter names and stores
        for use when Felix is started/restarted.

        Args:
            settings: Dict from SettingsView._get_current_settings()
        """
        # Map SettingsView keys to FelixConfig parameter names
        # Only include keys that FelixConfig actually accepts
        config_mapping = {
            'knowledge_enabled': 'enable_knowledge_brain',  # Key mapping!
            'web_search_enabled': 'web_search_enabled',
            'command_approval_required': 'auto_approve_system_actions',  # Inverted!
            'streaming_enabled': 'enable_streaming',
            'max_agents': 'max_agents',
            # Note: llm_provider, model, timeout are GUI-only and not passed to FelixConfig
        }

        # Build FelixConfig-compatible dict
        self._pending_config = {}
        for gui_key, config_key in config_mapping.items():
            if gui_key in settings:
                value = settings[gui_key]
                # Handle inversions
                if gui_key == 'command_approval_required':
                    value = not value  # GUI asks "require approval?", config is "auto approve"
                self._pending_config[config_key] = value

        logger.debug(f"Pending config updated: {self._pending_config}")

    @Slot()
    def _on_start_requested(self):
        """Handle start request from sidebar."""
        self.statusBar().showMessage("Starting Felix...")

        # Always read current settings from SettingsView, not just pending changes
        # This ensures settings are applied on first start, not just after changes
        config = self._get_current_felix_config()

        logger.info(f"Starting Felix with config: {config}")
        self._adapter.start_system(config)

    def _get_current_felix_config(self) -> Dict[str, Any]:
        """Get FelixConfig-compatible dict from current SettingsView values.

        This reads the actual checkbox/spinner values, ensuring settings
        are applied even if the user didn't change anything from defaults.
        """
        settings_view = self._context_panel.get_settings_view()
        gui_settings = settings_view._get_current_settings()

        # Map GUI settings keys to FelixConfig parameter names
        config_mapping = {
            'knowledge_enabled': 'enable_knowledge_brain',
            'web_search_enabled': 'web_search_enabled',
            'command_approval_required': 'auto_approve_system_actions',  # Inverted!
            'streaming_enabled': 'enable_streaming',
            'max_agents': 'max_agents',
        }

        config = {}
        for gui_key, config_key in config_mapping.items():
            if gui_key in gui_settings:
                value = gui_settings[gui_key]
                # Handle inversions
                if gui_key == 'command_approval_required':
                    value = not value  # GUI asks "require approval?", config is "auto approve"
                config[config_key] = value

        return config

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
                self._save_layout()  # Save before stopping
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

        self._save_layout()  # Save layout on close
        self._context_panel.cleanup()
        self._session_store.close()
        event.accept()

    # ========== Session Management ==========

    def _load_sessions(self):
        """Load sessions from database into sidebar."""
        sessions = self._session_store.get_all_sessions()
        self._sidebar.load_sessions(sessions)

        # Always start with a fresh workspace (user can select previous sessions)
        self._workspace.set_title("New Chat")
        self._current_session = None

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
        # Clear adapter conversation history for fresh start
        self._adapter.clear_conversation_history()

    @Slot()
    def _on_new_chat(self):
        """Handle new chat request from workspace."""
        self._create_new_session()
        self._adapter.clear_conversation_history()

    @Slot(str)
    def _on_session_selected(self, session_id: str):
        """Handle session selection from sidebar."""
        session = self._session_store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        self._current_session = session
        self._workspace.set_title(session.title)

        # Sync conversation history to adapter so Felix has context
        history = []
        for msg in session.messages:
            history.append({
                "role": msg.role.value,
                "content": msg.content
            })
        self._adapter.set_conversation_history(history)

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

    @Slot(str)
    def _on_rename_session(self, session_id: str):
        """Handle session rename request."""
        session = self._session_store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found for rename: {session_id}")
            return

        new_title, ok = QInputDialog.getText(
            self,
            "Rename Session",
            "New name:",
            QLineEdit.EchoMode.Normal,
            session.title
        )

        if ok and new_title.strip():
            session.title = new_title.strip()
            self._session_store.update_session(session)
            self._sidebar.update_session(session)

            # Update workspace title if this is the current session
            if self._current_session and self._current_session.id == session_id:
                self._current_session.title = session.title
                self._workspace.set_title(session.title)

            logger.info(f"Session renamed: {session_id} -> {new_title}")

    @Slot(str)
    def _on_delete_session(self, session_id: str):
        """Handle session delete request."""
        session = self._session_store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found for delete: {session_id}")
            return

        reply = QMessageBox.warning(
            self,
            "Delete Session",
            f"Delete '{session.title}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Yes:
            # If deleting current session, create a new one first
            if self._current_session and session_id == self._current_session.id:
                self._current_session = None
                self._workspace._message_area.clear()
                self._workspace.get_model().clear()
                self._workspace.set_title("New Chat")

            self._session_store.delete_session(session_id)
            self._sidebar.remove_session(session_id)
            logger.info(f"Session deleted: {session_id}")

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

    # ========== Layout Persistence & Responsive Handling ==========

    def _save_layout(self):
        """Save layout state to settings."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        self._settings.setValue("splitterSizes", self._splitter.sizes())
        self._settings.setValue("contextPanelWidth", self._context_panel.get_user_width())
        self._settings.setValue("contextPanelCollapsed", self._context_panel.is_collapsed())
        logger.debug("Layout saved to settings")

    def _restore_layout(self):
        """Restore layout state from settings."""
        # Restore window geometry
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            # Default size if no saved geometry
            self.resize(1400, 850)

        # Restore window state
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)

        # Restore splitter sizes
        splitter_sizes = self._settings.value("splitterSizes")
        if splitter_sizes:
            # QSettings may return strings, convert to ints
            if isinstance(splitter_sizes, list):
                sizes = [int(s) for s in splitter_sizes]
                self._splitter.setSizes(sizes)

        # Restore context panel width
        context_width = self._settings.value("contextPanelWidth", type=int)
        if context_width:
            self._context_panel.set_user_width(context_width)

        # Restore collapsed state
        was_collapsed = self._settings.value("contextPanelCollapsed", type=bool)
        if was_collapsed:
            self._context_panel._collapse()

        logger.debug("Layout restored from settings")

    def resizeEvent(self, event: QResizeEvent):
        """Handle window resize for responsive layout."""
        super().resizeEvent(event)

        new_width = event.size().width()
        old_mode = self._current_layout_mode
        new_mode = self._calculate_layout_mode(new_width)

        if new_mode != old_mode:
            self._current_layout_mode = new_mode
            self._handle_layout_mode_change(old_mode, new_mode)
            self.layout_mode_changed.emit(new_mode.value)
            logger.debug(f"Layout mode changed: {old_mode.value} -> {new_mode.value}")

    def _calculate_layout_mode(self, width: int) -> LayoutMode:
        """Calculate layout mode based on window width."""
        if width < self.COMPACT_BREAKPOINT:
            return LayoutMode.COMPACT
        elif width > self.WIDE_BREAKPOINT:
            return LayoutMode.WIDE
        else:
            return LayoutMode.STANDARD

    def _handle_layout_mode_change(self, old_mode: LayoutMode, new_mode: LayoutMode):
        """Handle layout mode transition."""
        # Auto-collapse context panel when going to COMPACT mode
        if new_mode == LayoutMode.COMPACT and not self._context_panel.is_collapsed():
            self._context_was_collapsed_by_resize = True
            self._context_panel._collapse()
            logger.debug("Auto-collapsed context panel for compact mode")

        # Auto-expand context panel when leaving COMPACT mode (if we collapsed it)
        elif old_mode == LayoutMode.COMPACT and self._context_was_collapsed_by_resize:
            self._context_was_collapsed_by_resize = False
            self._context_panel._expand()
            logger.debug("Auto-expanded context panel after leaving compact mode")

    def get_layout_mode(self) -> LayoutMode:
        """Get current layout mode."""
        return self._current_layout_mode
