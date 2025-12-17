"""Collapsible context panel with terminal, knowledge, and settings views."""

import logging
from typing import Optional

from PySide6.QtCore import Signal, Slot, Qt, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTabWidget, QSizePolicy
)

from ..core.theme import Colors
from ..context.terminal_view import TerminalView
from ..context.knowledge_view import KnowledgeView
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
        self._expanded_width = 320
        self._collapsed_width = 32
        self._felix_system = None

        self._setup_ui()

    def _setup_ui(self):
        """Set up context panel UI."""
        self.setMinimumWidth(self._collapsed_width)
        self.setMaximumWidth(450)
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

        # Terminal view
        self._terminal_view = TerminalView()
        self._tabs.addTab(self._terminal_view, "Terminal")

        # Knowledge view
        self._knowledge_view = KnowledgeView()
        self._tabs.addTab(self._knowledge_view, "Knowledge")

        # Settings view
        self._settings_view = SettingsView()
        self._settings_view.settings_changed.connect(self.settings_changed.emit)
        self._tabs.addTab(self._settings_view, "Settings")

        layout.addWidget(self._tabs, 1)

        # Set initial width
        self.setFixedWidth(self._expanded_width)

    def _toggle_collapse(self):
        """Toggle collapse state."""
        if self._is_collapsed:
            self._expand()
        else:
            self._collapse()

    def _collapse(self):
        """Collapse the panel."""
        self._is_collapsed = True
        self._toggle_btn.setText(">")
        self._toggle_btn.setToolTip("Expand panel")
        self._title.hide()
        self._tabs.hide()
        self.setFixedWidth(self._collapsed_width)
        self.collapsed.emit()
        logger.debug("Context panel collapsed")

    def _expand(self):
        """Expand the panel."""
        self._is_collapsed = False
        self._toggle_btn.setText("<")
        self._toggle_btn.setToolTip("Collapse panel")
        self._title.show()
        self._tabs.show()
        self.setFixedWidth(self._expanded_width)
        self.expanded.emit()
        logger.debug("Context panel expanded")

    def set_felix_system(self, felix_system):
        """Set Felix system reference for all views."""
        self._felix_system = felix_system
        self._terminal_view.set_felix_system(felix_system)
        self._settings_view.set_felix_system(felix_system)

        if felix_system:
            # Set knowledge references
            knowledge_store = getattr(felix_system, 'knowledge_store', None)
            knowledge_retriever = getattr(felix_system, 'knowledge_retriever', None)
            self._knowledge_view.set_knowledge_refs(
                knowledge_store=knowledge_store,
                knowledge_retriever=knowledge_retriever
            )

    def clear_system(self):
        """Clear Felix system reference."""
        self._felix_system = None
        self._terminal_view.set_felix_system(None)
        self._settings_view.set_felix_system(None)
        self._knowledge_view.set_knowledge_refs(None, None)

    def get_terminal_view(self) -> TerminalView:
        """Get terminal view widget."""
        return self._terminal_view

    def get_knowledge_view(self) -> KnowledgeView:
        """Get knowledge view widget."""
        return self._knowledge_view

    def get_settings_view(self) -> SettingsView:
        """Get settings view widget."""
        return self._settings_view

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
        self._terminal_view.cleanup()
        self._knowledge_view.cleanup()
        self._settings_view.cleanup()
