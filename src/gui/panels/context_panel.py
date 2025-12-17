"""Collapsible context panel with terminal, knowledge, and settings views."""

import logging
from typing import Optional

from PySide6.QtCore import Signal, Slot, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTabWidget, QSizePolicy
)

from ..core.theme import Colors
from ..context.activity_view import ActivityView
from ..context.terminal_view import TerminalView
from ..context.knowledge_view import KnowledgeView
from ..context.learning_view import LearningView
from ..context.settings_view import SettingsView

logger = logging.getLogger(__name__)


class ContextPanel(QFrame):
    """Collapsible right context panel.

    Features:
    - Terminal view for command monitoring
    - Knowledge browser for KB exploration
    - Quick settings access
    - Collapse/expand functionality

    Signals:
        collapsed: Emitted when panel is collapsed
        expanded: Emitted when panel is expanded
        settings_changed: Forwarded from settings view
    """

    collapsed = Signal()
    expanded = Signal()
    settings_changed = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("contextPanel")

        self._is_collapsed = False
        self._default_width = 320
        self._min_width = 200
        self._max_width = 800
        self._collapsed_width = 32
        self._user_width = self._default_width  # Remember user's preferred width
        self._felix_system = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up context panel UI."""
        # Use flexible sizing instead of fixed width
        self.setMinimumWidth(self._collapsed_width)
        self.setMaximumWidth(self._max_width)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.setStyleSheet(f"""
            QFrame#contextPanel {{
                background-color: {Colors.BACKGROUND_LIGHT};
                border-left: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with collapse button
        header = QFrame()
        header.setFixedHeight(40)
        header.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-bottom: 1px solid {Colors.BORDER};
        """)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(8)

        # Collapse/expand button
        self._toggle_btn = QPushButton("<")
        self._toggle_btn.setFixedSize(24, 24)
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.setToolTip("Collapse panel")
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: none;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {Colors.TEXT_PRIMARY};
                background-color: {Colors.SURFACE};
                border-radius: 4px;
            }}
        """)
        self._toggle_btn.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self._toggle_btn)

        # Title
        self._title = QLabel("Context")
        self._title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 13px;
            font-weight: 600;
        """)
        header_layout.addWidget(self._title)
        header_layout.addStretch()

        layout.addWidget(header)

        # Tab widget for views
        self._tabs = QTabWidget()
        self._tabs.setObjectName("contextTabs")
        self._tabs.setStyleSheet(f"""
            QTabWidget#contextTabs::pane {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
            QTabWidget#contextTabs > QTabBar::tab {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_SECONDARY};
                padding: 8px 12px;
                border: none;
                border-bottom: 2px solid transparent;
                font-size: 12px;
            }}
            QTabWidget#contextTabs > QTabBar::tab:selected {{
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
            QTabWidget#contextTabs > QTabBar::tab:hover {{
                background-color: {Colors.SURFACE};
            }}
        """)

        # Activity view (first tab - unified workflow/agents/approvals)
        self._activity_view = ActivityView()
        self._tabs.addTab(self._activity_view, "Activity")

        # Terminal view
        self._terminal_view = TerminalView()
        self._tabs.addTab(self._terminal_view, "Terminal")

        # Knowledge view
        self._knowledge_view = KnowledgeView()
        self._tabs.addTab(self._knowledge_view, "Knowledge")

        # Learning view (self-improvement systems)
        self._learning_view = LearningView()
        self._tabs.addTab(self._learning_view, "Learning")

        # Settings view
        self._settings_view = SettingsView()
        self._settings_view.settings_changed.connect(self.settings_changed.emit)
        self._tabs.addTab(self._settings_view, "Settings")

        layout.addWidget(self._tabs, 1)

        # Set initial width (not fixed - allows resizing via splitter)
        self.resize(self._default_width, self.height())

    def _toggle_collapse(self):
        """Toggle collapse state."""
        if self._is_collapsed:
            self._expand()
        else:
            self._collapse()

    def _collapse(self):
        """Collapse the panel."""
        # Remember current width before collapsing
        if not self._is_collapsed and self.width() > self._collapsed_width:
            self._user_width = self.width()

        self._is_collapsed = True
        self._toggle_btn.setText(">")
        self._toggle_btn.setToolTip("Expand panel")
        self._title.hide()
        self._tabs.hide()
        # Use fixed width only when collapsed
        self.setMinimumWidth(self._collapsed_width)
        self.setMaximumWidth(self._collapsed_width)
        self.collapsed.emit()
        logger.debug("Context panel collapsed")

    def _expand(self):
        """Expand the panel."""
        self._is_collapsed = False
        self._toggle_btn.setText("<")
        self._toggle_btn.setToolTip("Collapse panel")
        self._title.show()
        self._tabs.show()
        # Restore flexible sizing
        self.setMinimumWidth(self._min_width)
        self.setMaximumWidth(self._max_width)
        # Restore user's preferred width
        self.resize(self._user_width, self.height())
        self.expanded.emit()
        logger.debug(f"Context panel expanded to {self._user_width}px")

    def set_felix_system(self, felix_system):
        """Set Felix system reference for all views."""
        self._felix_system = felix_system
        self._activity_view.set_felix_system(felix_system)
        self._terminal_view.set_felix_system(felix_system)
        self._learning_view.set_felix_system(felix_system)
        self._settings_view.set_felix_system(felix_system)

        if felix_system:
            # Set knowledge references
            knowledge_store = getattr(felix_system, 'knowledge_store', None)
            knowledge_retriever = getattr(felix_system, 'knowledge_retriever', None)
            knowledge_daemon = getattr(felix_system, 'knowledge_daemon', None)

            # Determine daemon status reason for better UX
            daemon_status_reason = None
            if knowledge_daemon is None:
                config = getattr(felix_system, 'config', None)
                if config and not getattr(config, 'enable_knowledge_brain', False):
                    daemon_status_reason = "disabled_in_config"
                else:
                    daemon_status_reason = "initialization_failed"

            logger.debug(f"Setting knowledge refs: store={knowledge_store is not None}, "
                        f"retriever={knowledge_retriever is not None}, "
                        f"daemon={knowledge_daemon is not None}, reason={daemon_status_reason}")

            self._knowledge_view.set_knowledge_refs(
                knowledge_store=knowledge_store,
                knowledge_retriever=knowledge_retriever,
                knowledge_daemon=knowledge_daemon,
                daemon_status_reason=daemon_status_reason
            )

    def clear_system(self):
        """Clear Felix system reference."""
        self._felix_system = None
        self._activity_view.set_felix_system(None)
        self._terminal_view.set_felix_system(None)
        self._learning_view.set_felix_system(None)
        self._settings_view.set_felix_system(None)
        self._knowledge_view.set_knowledge_refs(None, None)

    def get_activity_view(self) -> ActivityView:
        """Get activity view widget."""
        return self._activity_view

    def get_terminal_view(self) -> TerminalView:
        """Get terminal view widget."""
        return self._terminal_view

    def get_knowledge_view(self) -> KnowledgeView:
        """Get knowledge view widget."""
        return self._knowledge_view

    def get_learning_view(self) -> LearningView:
        """Get learning view widget."""
        return self._learning_view

    def get_settings_view(self) -> SettingsView:
        """Get settings view widget."""
        return self._settings_view

    def show_activity(self):
        """Switch to activity tab."""
        self._tabs.setCurrentWidget(self._activity_view)
        if self._is_collapsed:
            self._expand()

    def show_terminal(self):
        """Switch to terminal tab."""
        self._tabs.setCurrentWidget(self._terminal_view)
        if self._is_collapsed:
            self._expand()

    def show_knowledge(self):
        """Switch to knowledge tab."""
        self._tabs.setCurrentWidget(self._knowledge_view)
        if self._is_collapsed:
            self._expand()

    def show_learning(self):
        """Switch to learning tab."""
        self._tabs.setCurrentWidget(self._learning_view)
        if self._is_collapsed:
            self._expand()

    def show_settings(self):
        """Switch to settings tab."""
        self._tabs.setCurrentWidget(self._settings_view)
        if self._is_collapsed:
            self._expand()

    def is_collapsed(self) -> bool:
        """Check if panel is collapsed."""
        return self._is_collapsed

    def cleanup(self):
        """Clean up resources."""
        self._activity_view.cleanup()
        self._terminal_view.cleanup()
        self._knowledge_view.cleanup()
        self._learning_view.cleanup()
        self._settings_view.cleanup()

    def get_user_width(self) -> int:
        """Get the user's preferred expanded width."""
        if not self._is_collapsed:
            return self.width()
        return self._user_width

    def set_user_width(self, width: int):
        """Set the user's preferred expanded width."""
        self._user_width = max(self._min_width, min(width, self._max_width))
        if not self._is_collapsed:
            self.resize(self._user_width, self.height())
