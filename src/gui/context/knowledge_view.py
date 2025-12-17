"""Knowledge browser view for viewing documents and concepts.

Enhanced with full daemon control features matching the CTK GUI:
- Process Directory, Process Pending, Manage Directories, Force Refinement
- Extended statistics dashboard with completion/processing/failed counts
- Processing queue progress display
- Activity feed with filtering, export, and auto-scroll
"""

import logging
import sqlite3
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTextEdit, QSplitter,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit, QTabWidget, QMessageBox, QScrollArea,
    QProgressBar, QDialog, QFileDialog, QDialogButtonBox,
    QComboBox, QCheckBox, QListWidget, QListWidgetItem,
    QGridLayout
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class StatusCard(QFrame):
    """Small status card for knowledge stats."""

    def __init__(self, title: str, value: str = "--", subtitle: str = "", parent=None):
        super().__init__(parent)
        self._subtitle = subtitle
        self._setup_ui(title, value)

    def _setup_ui(self, title: str, value: str):
        self.setFixedWidth(100)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        layout.addWidget(self._title_label)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        layout.addWidget(self._value_label)

        if self._subtitle:
            self._subtitle_label = QLabel(self._subtitle)
            self._subtitle_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 8px;")
            layout.addWidget(self._subtitle_label)
        else:
            self._subtitle_label = None

    def set_value(self, value: str):
        self._value_label.setText(value)

    def set_subtitle(self, subtitle: str):
        if self._subtitle_label:
            self._subtitle_label.setText(subtitle)

    def set_color(self, color: str):
        self._value_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")


class KnowledgeView(QWidget):
    """Knowledge browser with full control capabilities.

    Features:
    - Control: Daemon control, directory processing, refinement
    - Documents: Browser with search and preview
    - Concepts: Browser with domain filtering
    - Maintenance: Quality metrics, audit, cleanup
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._knowledge_store = None
        self._knowledge_retriever = None
        self._knowledge_daemon = None
        self._daemon_status_reason = None
        self._refresh_timer = None
        self._auto_refresh_enabled = False
        self._auto_scroll_enabled = True
        self._activity_filter = "All"
        self._setup_ui()

    def _setup_ui(self):
        """Set up knowledge view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget for documents/concepts
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_SECONDARY};
                padding: 8px 16px;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
            QTabBar::tab:hover {{
                background-color: {Colors.BACKGROUND_LIGHT};
            }}
        """)

        # Control tab (daemon status and controls)
        control_widget = self._create_control_tab()
        self._tabs.addTab(control_widget, "Control")

        # Documents tab
        docs_widget = self._create_documents_tab()
        self._tabs.addTab(docs_widget, "Docs")

        # Concepts tab
        concepts_widget = self._create_concepts_tab()
        self._tabs.addTab(concepts_widget, "Concepts")

        # Maintenance tab
        maint_widget = self._create_maintenance_tab()
        self._tabs.addTab(maint_widget, "Maint")

        layout.addWidget(self._tabs)

    def _create_control_tab(self) -> QWidget:
        """Create the control/daemon status tab with full features."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
        """)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ============ Statistics Dashboard ============
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(8, 8, 8, 8)
        stats_layout.setSpacing(6)

        stats_header = QHBoxLayout()
        stats_label = QLabel("Statistics Dashboard")
        stats_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        stats_header.addWidget(stats_label)
        stats_header.addStretch()

        refresh_stats_btn = QPushButton("Refresh")
        refresh_stats_btn.clicked.connect(self._refresh_all_stats)
        stats_header.addWidget(refresh_stats_btn)
        stats_layout.addLayout(stats_header)

        # First row of status cards
        cards_row1 = QHBoxLayout()
        cards_row1.setSpacing(6)

        self._daemon_status = StatusCard("Daemon", "Stopped", "Not running")
        cards_row1.addWidget(self._daemon_status)

        self._docs_card = StatusCard("Documents", "0", "Total sources")
        cards_row1.addWidget(self._docs_card)

        self._concepts_card = StatusCard("Entries", "0", "Knowledge entries")
        cards_row1.addWidget(self._concepts_card)

        self._relationships_card = StatusCard("Relations", "0", "Graph connections")
        cards_row1.addWidget(self._relationships_card)

        cards_row1.addStretch()
        stats_layout.addLayout(cards_row1)

        # Second row of status cards
        cards_row2 = QHBoxLayout()
        cards_row2.setSpacing(6)

        self._completed_card = StatusCard("Completed", "0", "Successfully processed")
        self._completed_card.set_color(Colors.SUCCESS)
        cards_row2.addWidget(self._completed_card)

        self._processing_card = StatusCard("Processing", "0", "Currently processing")
        self._processing_card.set_color(Colors.WARNING)
        cards_row2.addWidget(self._processing_card)

        self._failed_card = StatusCard("Failed", "0", "Processing errors")
        self._failed_card.set_color(Colors.ERROR)
        cards_row2.addWidget(self._failed_card)

        self._high_conf_card = StatusCard("High Conf", "0", "Reliable entries")
        cards_row2.addWidget(self._high_conf_card)

        cards_row2.addStretch()
        stats_layout.addLayout(cards_row2)

        layout.addWidget(stats_frame)

        # ============ Daemon Control Section ============
        controls_frame = QFrame()
        controls_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(6)

        ctrl_label = QLabel("Daemon Control")
        ctrl_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        controls_layout.addWidget(ctrl_label)

        # Row 1: Start/Stop buttons
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(6)

        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedWidth(70)
        self._start_btn.clicked.connect(self._start_daemon)
        self._start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background-color: #059669; }}
            QPushButton:disabled {{ background-color: {Colors.BORDER}; }}
        """)
        btn_row1.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.clicked.connect(self._stop_daemon)
        self._stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{ background-color: #dc2626; }}
            QPushButton:disabled {{ background-color: {Colors.BORDER}; }}
        """)
        btn_row1.addWidget(self._stop_btn)

        # Separator
        sep1 = QFrame()
        sep1.setFixedWidth(2)
        sep1.setFixedHeight(24)
        sep1.setStyleSheet(f"background-color: {Colors.BORDER};")
        btn_row1.addWidget(sep1)

        # Process Directory button
        self._process_dir_btn = QPushButton("Process Directory")
        self._process_dir_btn.clicked.connect(self._process_directory)
        btn_row1.addWidget(self._process_dir_btn)

        # Process Pending button
        self._process_pending_btn = QPushButton("Process Pending")
        self._process_pending_btn.clicked.connect(self._process_pending_now)
        btn_row1.addWidget(self._process_pending_btn)

        btn_row1.addStretch()
        controls_layout.addLayout(btn_row1)

        # Row 2: Additional controls
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(6)

        # Manage Directories button
        self._manage_dirs_btn = QPushButton("Manage Directories")
        self._manage_dirs_btn.clicked.connect(self._manage_directories)
        btn_row2.addWidget(self._manage_dirs_btn)

        # Force Refinement button
        self._refine_btn = QPushButton("Force Refinement")
        self._refine_btn.clicked.connect(self._force_refinement)
        btn_row2.addWidget(self._refine_btn)

        # Separator
        sep2 = QFrame()
        sep2.setFixedWidth(2)
        sep2.setFixedHeight(24)
        sep2.setStyleSheet(f"background-color: {Colors.BORDER};")
        btn_row2.addWidget(sep2)

        # Auto-refresh checkbox
        self._auto_refresh_cb = QCheckBox("Auto-Refresh")
        self._auto_refresh_cb.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        self._auto_refresh_cb.toggled.connect(self._toggle_auto_refresh)
        btn_row2.addWidget(self._auto_refresh_cb)

        btn_row2.addStretch()
        controls_layout.addLayout(btn_row2)

        layout.addWidget(controls_frame)

        # ============ Processing Queue Section ============
        queue_frame = QFrame()
        queue_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        queue_layout = QVBoxLayout(queue_frame)
        queue_layout.setContentsMargins(8, 8, 8, 8)
        queue_layout.setSpacing(6)

        queue_label = QLabel("Processing Queue")
        queue_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        queue_layout.addWidget(queue_label)

        # Progress bar
        self._queue_progress = QProgressBar()
        self._queue_progress.setMinimum(0)
        self._queue_progress.setMaximum(100)
        self._queue_progress.setValue(0)
        self._queue_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                text-align: center;
                background-color: {Colors.BACKGROUND};
            }}
            QProgressBar::chunk {{
                background-color: {Colors.ACCENT};
                border-radius: 3px;
            }}
        """)
        queue_layout.addWidget(self._queue_progress)

        # Queue status text
        self._queue_text = QTextEdit()
        self._queue_text.setReadOnly(True)
        self._queue_text.setMaximumHeight(80)
        self._queue_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }}
        """)
        self._queue_text.setPlainText("Processing queue status will appear here...")
        queue_layout.addWidget(self._queue_text)

        layout.addWidget(queue_frame)

        # ============ Activity Feed Section ============
        activity_frame = QFrame()
        activity_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        activity_layout = QVBoxLayout(activity_frame)
        activity_layout.setContentsMargins(8, 8, 8, 8)
        activity_layout.setSpacing(6)

        # Activity header with controls
        activity_header = QHBoxLayout()
        activity_header.setSpacing(6)

        activity_label = QLabel("Activity Feed")
        activity_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        activity_header.addWidget(activity_label)

        activity_header.addStretch()

        # Filter combo
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        activity_header.addWidget(filter_label)

        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "INFO", "WARNING", "ERROR"])
        self._filter_combo.setFixedWidth(80)
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self._filter_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 2px 6px;
            }}
        """)
        activity_header.addWidget(self._filter_combo)

        # Control buttons
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_activity)
        activity_header.addWidget(refresh_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_activity_log)
        activity_header.addWidget(clear_btn)

        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_activity_log)
        activity_header.addWidget(export_btn)

        activity_layout.addLayout(activity_header)

        # Auto-scroll checkbox
        scroll_row = QHBoxLayout()
        scroll_row.addStretch()
        self._auto_scroll_cb = QCheckBox("Auto-scroll")
        self._auto_scroll_cb.setChecked(True)
        self._auto_scroll_cb.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        self._auto_scroll_cb.toggled.connect(lambda v: setattr(self, '_auto_scroll_enabled', v))
        scroll_row.addWidget(self._auto_scroll_cb)
        activity_layout.addLayout(scroll_row)

        # Activity log
        self._activity_log = QTextEdit()
        self._activity_log.setReadOnly(True)
        self._activity_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BACKGROUND};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }}
        """)
        self._activity_log.setPlainText("Knowledge daemon activity will appear here...")
        activity_layout.addWidget(self._activity_log, 1)

        layout.addWidget(activity_frame, 1)

        scroll.setWidget(widget)
        return scroll

    def _create_documents_tab(self) -> QWidget:
        """Create the documents browser tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Search row
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self._doc_search = QLineEdit()
        self._doc_search.setPlaceholderText("Search documents...")
        self._doc_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px 8px;
            }}
        """)
        self._doc_search.returnPressed.connect(self._search_documents)
        search_row.addWidget(self._doc_search, 1)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._search_documents)
        search_row.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_documents)
        search_row.addWidget(refresh_btn)

        layout.addLayout(search_row)

        # Splitter for list and preview
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
        """)

        # Document list
        self._doc_tree = QTreeWidget()
        self._doc_tree.setHeaderLabels(["Document", "Type", "Added"])
        self._doc_tree.setRootIsDecorated(False)
        self._doc_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QTreeWidget::item {{
                padding: 4px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Colors.ACCENT};
            }}
            QHeaderView::section {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        self._doc_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._doc_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._doc_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._doc_tree.setColumnWidth(1, 60)
        self._doc_tree.setColumnWidth(2, 80)
        self._doc_tree.setSortingEnabled(True)
        self._doc_tree.itemSelectionChanged.connect(self._on_document_selected)
        self._doc_tree.itemDoubleClicked.connect(self._on_document_double_clicked)
        splitter.addWidget(self._doc_tree)

        # Document preview
        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 8, 0, 0)
        preview_layout.setSpacing(4)

        preview_label = QLabel("Preview:")
        preview_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        preview_layout.addWidget(preview_label)

        self._doc_preview = QTextEdit()
        self._doc_preview.setReadOnly(True)
        self._doc_preview.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        preview_layout.addWidget(self._doc_preview)

        splitter.addWidget(preview_frame)
        splitter.setSizes([200, 150])

        layout.addWidget(splitter, 1)

        # Stats row
        self._doc_stats = QLabel("Documents: --")
        self._doc_stats.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._doc_stats)

        return widget

    def _create_concepts_tab(self) -> QWidget:
        """Create the concepts browser tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Search row
        search_row = QHBoxLayout()
        search_row.setSpacing(8)

        self._concept_search = QLineEdit()
        self._concept_search.setPlaceholderText("Search concepts...")
        self._concept_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px 8px;
            }}
        """)
        self._concept_search.returnPressed.connect(self._search_concepts)
        search_row.addWidget(self._concept_search, 1)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._search_concepts)
        search_row.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_concepts)
        search_row.addWidget(refresh_btn)

        layout.addLayout(search_row)

        # Concept list
        self._concept_tree = QTreeWidget()
        self._concept_tree.setHeaderLabels(["Concept", "Type", "Occurrences"])
        self._concept_tree.setRootIsDecorated(False)
        self._concept_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QTreeWidget::item {{
                padding: 4px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Colors.ACCENT};
            }}
            QHeaderView::section {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        self._concept_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._concept_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._concept_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._concept_tree.setColumnWidth(1, 80)
        self._concept_tree.setColumnWidth(2, 80)
        self._concept_tree.setSortingEnabled(True)
        self._concept_tree.itemDoubleClicked.connect(self._on_concept_double_clicked)
        layout.addWidget(self._concept_tree, 1)

        # Stats row
        self._concept_stats = QLabel("Concepts: --")
        self._concept_stats.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._concept_stats)

        return widget

    def _create_maintenance_tab(self) -> QWidget:
        """Create the maintenance/cleanup tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Quality metrics section
        quality_frame = QFrame()
        quality_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        quality_layout = QVBoxLayout(quality_frame)
        quality_layout.setContentsMargins(8, 8, 8, 8)
        quality_layout.setSpacing(4)

        quality_label = QLabel("Quality Metrics")
        quality_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        quality_layout.addWidget(quality_label)

        self._quality_text = QLabel("Database health: --\nOrphan entries: --\nDuplicate concepts: --")
        self._quality_text.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        self._quality_text.setWordWrap(True)
        quality_layout.addWidget(self._quality_text)

        layout.addWidget(quality_frame)

        # Cleanup actions
        actions_frame = QFrame()
        actions_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setContentsMargins(8, 8, 8, 8)
        actions_layout.setSpacing(6)

        actions_label = QLabel("Maintenance Actions")
        actions_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: bold;")
        actions_layout.addWidget(actions_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self._analyze_database)
        btn_row.addWidget(analyze_btn)

        vacuum_btn = QPushButton("Vacuum")
        vacuum_btn.clicked.connect(self._vacuum_database)
        btn_row.addWidget(vacuum_btn)

        cleanup_btn = QPushButton("Cleanup")
        cleanup_btn.clicked.connect(self._cleanup_orphans)
        btn_row.addWidget(cleanup_btn)

        btn_row.addStretch()
        actions_layout.addLayout(btn_row)

        layout.addWidget(actions_frame)

        # Audit log
        audit_label = QLabel("Maintenance Log")
        audit_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(audit_label)

        self._audit_log = QTextEdit()
        self._audit_log.setReadOnly(True)
        self._audit_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }}
        """)
        self._audit_log.setPlainText("Maintenance operations will be logged here...")
        layout.addWidget(self._audit_log, 1)

        return widget

    # ============ Public API ============

    def set_knowledge_refs(self, knowledge_store=None, knowledge_retriever=None, knowledge_daemon=None, **kwargs):
        """Set knowledge brain references."""
        logger.debug(f"set_knowledge_refs called: store={knowledge_store is not None}, "
                    f"retriever={knowledge_retriever is not None}, daemon={knowledge_daemon is not None}")
        self._knowledge_store = knowledge_store
        self._knowledge_retriever = knowledge_retriever
        self._knowledge_daemon = knowledge_daemon
        self._daemon_status_reason = kwargs.get('daemon_status_reason')

        if knowledge_store:
            logger.debug(f"Knowledge store path: {getattr(knowledge_store, 'storage_path', 'unknown')}")
            self._refresh_documents()
            self._refresh_concepts()
            self._refresh_daemon_status()
            self._refresh_all_stats()

    # ============ Daemon Control Methods ============

    @Slot()
    def _start_daemon(self):
        """Start the knowledge daemon."""
        if not self._knowledge_daemon:
            self._log_activity("Daemon not available", "WARNING")
            return

        try:
            if hasattr(self._knowledge_daemon, 'start'):
                self._knowledge_daemon.start()
                self._log_activity("Daemon started", "INFO")
                self._refresh_daemon_status()
        except Exception as e:
            self._log_activity(f"Error starting daemon: {e}", "ERROR")
            logger.error(f"Failed to start daemon: {e}")

    @Slot()
    def _stop_daemon(self):
        """Stop the knowledge daemon."""
        if not self._knowledge_daemon:
            self._log_activity("Daemon not available", "WARNING")
            return

        try:
            if hasattr(self._knowledge_daemon, 'stop'):
                self._knowledge_daemon.stop()
                self._log_activity("Daemon stopped", "INFO")
                self._refresh_daemon_status()
        except Exception as e:
            self._log_activity(f"Error stopping daemon: {e}", "ERROR")
            logger.error(f"Failed to stop daemon: {e}")

    @Slot()
    def _process_directory(self):
        """Add a directory for one-time processing."""
        if not self._knowledge_daemon:
            QMessageBox.warning(self, "Not Available", "Knowledge daemon not available")
            return

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Documents",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            try:
                if hasattr(self._knowledge_daemon, 'process_directory_now'):
                    result = self._knowledge_daemon.process_directory_now(directory)
                    queued = result.get('queued', 0) if isinstance(result, dict) else 0
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Queued {queued} documents from:\n{directory}"
                    )
                    self._log_activity(f"Processed directory: {directory} ({queued} documents)", "INFO")
                    self._refresh_all_stats()
                else:
                    self._log_activity("process_directory_now not available on daemon", "WARNING")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to process directory:\n{str(e)}")
                self._log_activity(f"Error processing directory: {e}", "ERROR")

    @Slot()
    def _process_pending_now(self):
        """Manually trigger processing of all pending documents."""
        if not self._knowledge_daemon or not self._knowledge_store:
            QMessageBox.warning(self, "Not Available", "Knowledge daemon or store not available")
            return

        try:
            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if not storage_path:
                self._log_activity("Storage path not available", "WARNING")
                return

            conn = sqlite3.connect(storage_path)
            cursor = conn.execute("SELECT COUNT(*) FROM document_sources WHERE ingestion_status='pending'")
            pending_count = cursor.fetchone()[0]

            if pending_count == 0:
                conn.close()
                QMessageBox.information(self, "No Pending Documents", "All documents have been processed")
                return

            # Get all pending document paths
            cursor = conn.execute("SELECT file_path FROM document_sources WHERE ingestion_status='pending'")
            pending_paths = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Add all pending documents to the daemon's processing queue
            if hasattr(self._knowledge_daemon, 'document_queue'):
                for path in pending_paths:
                    self._knowledge_daemon.document_queue.add(path)

                QMessageBox.information(
                    self,
                    "Processing Started",
                    f"Queued {pending_count} pending documents for immediate processing."
                )
                self._log_activity(f"Queued {pending_count} pending documents for processing", "INFO")
                self._refresh_all_stats()
            else:
                self._log_activity("document_queue not available on daemon", "WARNING")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process pending:\n{str(e)}")
            self._log_activity(f"Error processing pending: {e}", "ERROR")

    @Slot()
    def _manage_directories(self):
        """Open dialog to manage watched directories."""
        if not self._knowledge_daemon:
            QMessageBox.warning(self, "Not Available", "Knowledge daemon not available")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Watch Directories")
        dialog.resize(600, 400)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BACKGROUND};
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("Currently Watched Directories")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Directory list
        dir_list = QListWidget()
        dir_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {Colors.ACCENT};
            }}
        """)

        # Get current directories
        current_dirs = []
        if hasattr(self._knowledge_daemon, 'config') and hasattr(self._knowledge_daemon.config, 'watch_directories'):
            current_dirs = self._knowledge_daemon.config.watch_directories.copy()

        for d in current_dirs:
            dir_list.addItem(d)

        layout.addWidget(dir_list, 1)

        # Info label
        info_label = QLabel(f"{len(current_dirs)} director{'y' if len(current_dirs) == 1 else 'ies'} configured")
        info_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        layout.addWidget(info_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        def add_directory():
            new_dir = QFileDialog.getExistingDirectory(dialog, "Select Directory to Watch")
            if new_dir:
                try:
                    if hasattr(self._knowledge_daemon, 'add_watch_directory'):
                        result = self._knowledge_daemon.add_watch_directory(new_dir)
                        if result.get('success'):
                            dir_list.addItem(new_dir)
                            current_dirs.append(new_dir)
                            info_label.setText(f"{len(current_dirs)} director{'y' if len(current_dirs) == 1 else 'ies'} configured")
                            self._log_activity(f"Added watch directory: {new_dir}", "INFO")
                            QMessageBox.information(dialog, "Success", f"Added to watch list:\n{new_dir}")
                        else:
                            QMessageBox.warning(dialog, "Cannot Add", result.get('error', 'Unknown error'))
                except Exception as e:
                    QMessageBox.critical(dialog, "Error", f"Failed to add directory:\n{str(e)}")

        def remove_directory():
            current_item = dir_list.currentItem()
            if not current_item:
                QMessageBox.warning(dialog, "No Selection", "Please select a directory to remove")
                return

            directory = current_item.text()
            if QMessageBox.question(
                dialog,
                "Confirm Removal",
                f"Remove this directory from watch list?\n\n{directory}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                try:
                    if hasattr(self._knowledge_daemon, 'remove_watch_directory'):
                        result = self._knowledge_daemon.remove_watch_directory(directory)
                        if result.get('success'):
                            dir_list.takeItem(dir_list.row(current_item))
                            if directory in current_dirs:
                                current_dirs.remove(directory)
                            info_label.setText(f"{len(current_dirs)} director{'y' if len(current_dirs) == 1 else 'ies'} configured")
                            self._log_activity(f"Removed watch directory: {directory}", "INFO")
                        else:
                            QMessageBox.warning(dialog, "Cannot Remove", result.get('error', 'Unknown error'))
                except Exception as e:
                    QMessageBox.critical(dialog, "Error", f"Failed to remove directory:\n{str(e)}")

        add_btn = QPushButton("Add Directory")
        add_btn.clicked.connect(add_directory)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: {Colors.ACCENT_HOVER}; }}
        """)
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(remove_directory)
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ERROR};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #dc2626; }}
        """)
        btn_layout.addWidget(remove_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        dialog.exec()

    @Slot()
    def _force_refinement(self):
        """Manually trigger a refinement cycle."""
        if not self._knowledge_daemon:
            QMessageBox.warning(self, "Not Available", "Knowledge daemon not available")
            return

        self._log_activity("Starting manual refinement (this may take a while)...", "INFO")

        try:
            if hasattr(self._knowledge_daemon, 'trigger_refinement'):
                result = self._knowledge_daemon.trigger_refinement()
                total = result.get('total_relationships', 0) if isinstance(result, dict) else 0
                QMessageBox.information(self, "Refinement Complete", f"Created {total} relationships")
                self._log_activity(f"Refinement complete: {total} relationships", "INFO")
                self._refresh_all_stats()
            else:
                self._log_activity("trigger_refinement not available on daemon", "WARNING")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Refinement failed:\n{str(e)}")
            self._log_activity(f"Refinement failed: {e}", "ERROR")

    @Slot(bool)
    def _toggle_auto_refresh(self, enabled: bool):
        """Toggle auto-refresh."""
        self._auto_refresh_enabled = enabled
        if enabled:
            if not self._refresh_timer:
                self._refresh_timer = QTimer(self)
                self._refresh_timer.timeout.connect(self._auto_refresh_tick)
            self._refresh_timer.start(5000)  # 5 seconds
            self._log_activity("Auto-refresh enabled (5s interval)", "INFO")
        else:
            if self._refresh_timer:
                self._refresh_timer.stop()
            self._log_activity("Auto-refresh disabled", "INFO")

    def _auto_refresh_tick(self):
        """Auto-refresh timer tick."""
        if self._auto_refresh_enabled:
            self._refresh_daemon_status()
            self._refresh_queue_status()

    # ============ Status Refresh Methods ============

    @Slot()
    def _refresh_daemon_status(self):
        """Refresh daemon status display."""
        if not self._knowledge_daemon:
            reason = getattr(self, '_daemon_status_reason', None)
            if reason == "disabled_in_config":
                self._daemon_status.set_value("Disabled")
                self._daemon_status.set_subtitle("In config")
                self._daemon_status.set_color(Colors.TEXT_MUTED)
            else:
                self._daemon_status.set_value("N/A")
                self._daemon_status.set_subtitle("Not initialized")
                self._daemon_status.set_color(Colors.WARNING)

            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            self._process_dir_btn.setEnabled(False)
            self._process_pending_btn.setEnabled(False)
            self._manage_dirs_btn.setEnabled(False)
            self._refine_btn.setEnabled(False)
            return

        try:
            is_running = getattr(self._knowledge_daemon, 'running', False)

            if is_running:
                # Get uptime if available
                uptime_str = "Running"
                if hasattr(self._knowledge_daemon, 'get_status'):
                    try:
                        status = self._knowledge_daemon.get_status()
                        if hasattr(status, 'uptime_seconds'):
                            hours = status.uptime_seconds / 3600
                            uptime_str = f"{hours:.1f}h"
                    except:
                        pass

                self._daemon_status.set_value("Running")
                self._daemon_status.set_subtitle(uptime_str)
                self._daemon_status.set_color(Colors.SUCCESS)
                self._start_btn.setEnabled(False)
                self._stop_btn.setEnabled(True)
            else:
                self._daemon_status.set_value("Stopped")
                self._daemon_status.set_subtitle("Not running")
                self._daemon_status.set_color(Colors.ERROR)
                self._start_btn.setEnabled(True)
                self._stop_btn.setEnabled(False)

            # Enable control buttons
            self._process_dir_btn.setEnabled(True)
            self._process_pending_btn.setEnabled(True)
            self._manage_dirs_btn.setEnabled(True)
            self._refine_btn.setEnabled(True)

        except Exception as e:
            logger.error(f"Failed to refresh daemon status: {e}")

    @Slot()
    def _refresh_all_stats(self):
        """Refresh all statistics cards."""
        self._refresh_daemon_status()
        self._refresh_queue_status()
        self._refresh_document_stats()

    def _refresh_queue_status(self):
        """Refresh processing queue status."""
        if not self._knowledge_store:
            return

        try:
            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if not storage_path:
                return

            conn = sqlite3.connect(storage_path)

            # Get counts by status
            cursor = conn.execute("""
                SELECT ingestion_status, COUNT(*)
                FROM document_sources
                GROUP BY ingestion_status
            """)
            status_counts = dict(cursor.fetchall())
            conn.close()

            pending = status_counts.get('pending', 0)
            processing = status_counts.get('processing', 0)
            completed = status_counts.get('completed', 0)
            failed = status_counts.get('failed', 0)
            total = pending + processing + completed + failed

            # Update cards
            self._completed_card.set_value(str(completed))
            self._processing_card.set_value(str(processing))
            self._failed_card.set_value(str(failed))

            # Update progress bar
            if total > 0:
                pct = int((completed / total) * 100)
                self._queue_progress.setValue(pct)
            else:
                self._queue_progress.setValue(0)

            # Update queue text
            queue_text = f"Total Documents: {total}\n"
            queue_text += f"Completed: {completed}  |  Processing: {processing}\n"
            queue_text += f"Pending: {pending}  |  Failed: {failed}"
            if total > 0:
                queue_text += f"\nCompletion: {(completed/total)*100:.1f}%"

            self._queue_text.setPlainText(queue_text)

        except Exception as e:
            logger.error(f"Failed to refresh queue status: {e}")

    def _refresh_document_stats(self):
        """Refresh document and concept statistics."""
        if not self._knowledge_store:
            return

        try:
            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if not storage_path:
                return

            conn = sqlite3.connect(storage_path)

            # Total documents
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM document_sources")
                doc_count = cursor.fetchone()[0]
                self._docs_card.set_value(str(doc_count))
            except:
                pass

            # Knowledge entries
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
                entry_count = cursor.fetchone()[0]
                self._concepts_card.set_value(str(entry_count))
            except:
                pass

            # Relationships
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
                rel_count = cursor.fetchone()[0]
                self._relationships_card.set_value(str(rel_count))
            except:
                self._relationships_card.set_value("--")

            # High confidence entries
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries WHERE confidence > 0.8")
                high_conf = cursor.fetchone()[0]
                self._high_conf_card.set_value(str(high_conf))
            except:
                pass

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh document stats: {e}")

    # ============ Activity Log Methods ============

    def _log_activity(self, message: str, level: str = "INFO"):
        """Log activity to the control tab with severity level."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Apply filter
        if self._activity_filter != "All" and level != self._activity_filter:
            return

        # Add emoji based on level
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
        log_message = f"[{timestamp}] {emoji} {level}: {message}"

        current = self._activity_log.toPlainText()
        lines = current.split('\n')
        # Keep last 100 lines
        if len(lines) > 100:
            lines = lines[-100:]
        lines.append(log_message)
        self._activity_log.setPlainText('\n'.join(lines))

        # Auto-scroll if enabled
        if self._auto_scroll_enabled:
            scrollbar = self._activity_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(str)
    def _on_filter_changed(self, filter_value: str):
        """Handle activity filter change."""
        self._activity_filter = filter_value
        self._refresh_activity()

    @Slot()
    def _refresh_activity(self):
        """Refresh activity log from database."""
        if not self._knowledge_store:
            return

        try:
            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if not storage_path:
                return

            conn = sqlite3.connect(storage_path)
            cursor = conn.execute("""
                SELECT file_path, ingestion_completed, ingestion_status
                FROM document_sources
                WHERE ingestion_completed IS NOT NULL
                ORDER BY ingestion_completed DESC
                LIMIT 30
            """)

            logs = []
            for row in cursor.fetchall():
                file_path, ingestion_completed, status = row
                filename = Path(file_path).name if file_path else 'unknown'
                try:
                    timestamp = datetime.fromtimestamp(ingestion_completed).strftime('%H:%M:%S')
                except:
                    timestamp = "??:??:??"

                if status == 'completed':
                    level = "INFO"
                    message = f"Completed: {filename}"
                elif status == 'failed':
                    level = "ERROR"
                    message = f"Failed: {filename}"
                else:
                    level = "INFO"
                    message = f"{status}: {filename}"

                # Apply filter
                if self._activity_filter == "All" or self._activity_filter == level:
                    emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
                    logs.append(f"[{timestamp}] {emoji} {level}: {message}")

            conn.close()

            # Add daemon status
            if self._knowledge_daemon:
                is_running = getattr(self._knowledge_daemon, 'running', False)
                timestamp = datetime.now().strftime('%H:%M:%S')
                if is_running:
                    if self._activity_filter in ["All", "INFO"]:
                        logs.insert(0, f"[{timestamp}] ℹ️ INFO: Daemon is running")
                else:
                    if self._activity_filter in ["All", "WARNING"]:
                        logs.insert(0, f"[{timestamp}] ⚠️ WARNING: Daemon is not running")

            if logs:
                self._activity_log.setPlainText('\n'.join(logs))
            else:
                self._activity_log.setPlainText("No activity to display")

        except Exception as e:
            logger.error(f"Failed to refresh activity: {e}")

    @Slot()
    def _clear_activity_log(self):
        """Clear activity log."""
        self._activity_log.setPlainText("Activity log cleared.")

    @Slot()
    def _export_activity_log(self):
        """Export activity log to a text file."""
        log_content = self._activity_log.toPlainText()
        if not log_content.strip() or log_content == "Activity log cleared.":
            QMessageBox.warning(self, "Empty Log", "Activity log is empty, nothing to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Activity Log",
            f"felix_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Felix Knowledge Brain Activity Log\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Filter: {self._activity_filter}\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(log_content)

                QMessageBox.information(self, "Export Complete", f"Activity log exported to:\n{file_path}")
                self._log_activity(f"Exported activity log to {file_path}", "INFO")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export log:\n{str(e)}")

    # ============ Document Methods ============

    @Slot()
    def _refresh_documents(self):
        """Refresh document list."""
        if not self._knowledge_store:
            self._doc_stats.setText("Documents: -- (not connected)")
            return

        try:
            entries = self._knowledge_store.advanced_search(limit=100)
            self._doc_tree.clear()

            for entry in entries:
                title = entry.domain or entry.source_doc_id or entry.knowledge_id or 'Unknown'
                if len(title) > 40:
                    title = title[:37] + "..."

                doc_type = entry.knowledge_type.value if entry.knowledge_type else 'unknown'
                added = str(entry.created_at)[:10] if entry.created_at else ''

                doc = {
                    'title': entry.domain,
                    'source': entry.source_agent or entry.source_doc_id or 'unknown',
                    'doc_type': doc_type,
                    'created_at': str(entry.created_at) if entry.created_at else '',
                    'content': entry.content.get('text', str(entry.content)) if isinstance(entry.content, dict) else str(entry.content)
                }

                item = QTreeWidgetItem([title, doc_type, added])
                item.setData(0, Qt.ItemDataRole.UserRole, doc)
                self._doc_tree.addTopLevelItem(item)

            self._doc_stats.setText(f"Documents: {len(entries)}")

        except Exception as e:
            logger.error(f"Error refreshing documents: {e}")
            self._doc_stats.setText(f"Error: {e}")

    @Slot()
    def _search_documents(self):
        """Search documents."""
        query = self._doc_search.text().strip()
        if not query or not self._knowledge_retriever:
            self._refresh_documents()
            return

        try:
            context = self._knowledge_retriever.search(query, top_k=20)
            self._doc_tree.clear()

            for result in context.results:
                entry = result.knowledge_entry
                if not entry:
                    continue

                title = entry.domain or entry.source_doc_id or entry.knowledge_id or 'Unknown'
                if len(title) > 40:
                    title = title[:37] + "..."

                score = result.relevance_score
                doc_type = entry.knowledge_type.value if entry.knowledge_type else 'unknown'

                doc = {
                    'title': entry.domain,
                    'source': entry.source_agent or entry.source_doc_id or 'unknown',
                    'doc_type': doc_type,
                    'created_at': str(entry.created_at) if entry.created_at else '',
                    'content': entry.content.get('text', str(entry.content)) if isinstance(entry.content, dict) else str(entry.content)
                }

                item = QTreeWidgetItem([title, doc_type, f"{score:.2f}"])
                item.setData(0, Qt.ItemDataRole.UserRole, doc)
                self._doc_tree.addTopLevelItem(item)

            self._doc_stats.setText(f"Results: {len(context.results)}")

        except Exception as e:
            logger.error(f"Error searching documents: {e}")

    def _on_document_selected(self):
        """Handle document selection."""
        item = self._doc_tree.currentItem()
        if not item:
            self._doc_preview.clear()
            return

        doc = item.data(0, Qt.ItemDataRole.UserRole)
        if doc:
            preview = f"Title: {doc.get('title', 'N/A')}\n"
            preview += f"Source: {doc.get('source', 'N/A')}\n"
            preview += f"Type: {doc.get('doc_type', 'N/A')}\n"
            preview += f"Created: {doc.get('created_at', 'N/A')}\n"
            preview += "\n---\n\n"

            content = doc.get('content', doc.get('text', ''))
            if len(content) > 1000:
                content = content[:1000] + "...\n\n[Content truncated - double-click to view full]"
            preview += content

            self._doc_preview.setPlainText(preview)

    def _on_document_double_clicked(self, item, column):
        """Show full document content in a dialog."""
        doc = item.data(0, Qt.ItemDataRole.UserRole)
        if not doc:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Document: {doc.get('title', 'Unknown')}")
        dialog.resize(700, 500)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {Colors.BACKGROUND}; }}
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel(f"<b>{doc.get('title', 'Unknown')}</b><br>"
                       f"<span style='color: {Colors.TEXT_MUTED};'>"
                       f"Source: {doc.get('source', 'N/A')} | "
                       f"Type: {doc.get('doc_type', 'N/A')} | "
                       f"Created: {doc.get('created_at', 'N/A')}</span>")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(header)

        content_edit = QTextEdit()
        content_edit.setReadOnly(True)
        content = doc.get('content', doc.get('text', ''))
        content_edit.setPlainText(content)
        layout.addWidget(content_edit, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        export_btn = QPushButton("Export to File")
        export_btn.clicked.connect(lambda: self._export_document(doc, dialog))
        btn_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
        dialog.exec()

    def _export_document(self, doc: dict, parent_dialog: QDialog = None):
        """Export document content to a file."""
        title = doc.get('title', 'document').replace('/', '_').replace('\\', '_')
        default_name = f"{title}.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            parent_dialog or self,
            "Export Document",
            default_name,
            "Text Files (*.txt);;Markdown Files (*.md);;All Files (*)"
        )

        if file_path:
            try:
                content = doc.get('content', doc.get('text', ''))
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {doc.get('title', 'N/A')}\n")
                    f.write(f"Source: {doc.get('source', 'N/A')}\n")
                    f.write(f"Type: {doc.get('doc_type', 'N/A')}\n")
                    f.write(f"Created: {doc.get('created_at', 'N/A')}\n")
                    f.write("\n---\n\n")
                    f.write(content)
                QMessageBox.information(parent_dialog or self, "Export Successful", f"Document exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(parent_dialog or self, "Export Failed", f"Failed to export document:\n{str(e)}")

    # ============ Concept Methods ============

    @Slot()
    def _refresh_concepts(self):
        """Refresh concept list."""
        if not self._knowledge_store:
            self._concept_stats.setText("Concepts: -- (not connected)")
            return

        try:
            entries = self._knowledge_store.advanced_search(tags=['concept'], limit=100)
            self._concept_tree.clear()

            for entry in entries:
                if isinstance(entry.content, dict):
                    name = entry.content.get('concept', entry.content.get('topic', entry.domain or 'Unknown'))
                else:
                    name = entry.domain or 'Unknown'

                if len(name) > 40:
                    name = name[:37] + "..."

                concept_type = 'concept'
                count = entry.access_count or 0

                concept = {
                    'name': name,
                    'type': concept_type,
                    'count': count,
                    'domain': entry.domain,
                    'content': entry.content
                }

                item = QTreeWidgetItem([name, concept_type, str(count)])
                item.setData(0, Qt.ItemDataRole.UserRole, concept)
                self._concept_tree.addTopLevelItem(item)

            self._concept_stats.setText(f"Concepts: {len(entries)}")

        except Exception as e:
            logger.error(f"Error refreshing concepts: {e}")
            self._concept_stats.setText(f"Error: {e}")

    @Slot()
    def _search_concepts(self):
        """Search concepts."""
        query = self._concept_search.text().strip()
        if not query or not self._knowledge_store:
            self._refresh_concepts()
            return

        try:
            entries = self._knowledge_store.advanced_search(content=query, tags=['concept'], limit=50)
            self._concept_tree.clear()

            for entry in entries:
                if isinstance(entry.content, dict):
                    name = entry.content.get('concept', entry.content.get('topic', entry.domain or 'Unknown'))
                else:
                    name = entry.domain or 'Unknown'

                if len(name) > 40:
                    name = name[:37] + "..."

                concept_type = 'concept'
                count = entry.access_count or 0

                concept = {
                    'name': name,
                    'type': concept_type,
                    'count': count,
                    'domain': entry.domain,
                    'content': entry.content
                }

                item = QTreeWidgetItem([name, concept_type, str(count)])
                item.setData(0, Qt.ItemDataRole.UserRole, concept)
                self._concept_tree.addTopLevelItem(item)

            self._concept_stats.setText(f"Results: {len(entries)}")

        except Exception as e:
            logger.error(f"Error searching concepts: {e}")

    def _on_concept_double_clicked(self, item, column):
        """Show concept details in a dialog."""
        concept = item.data(0, Qt.ItemDataRole.UserRole)
        if not concept:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Concept: {concept.get('name', 'Unknown')}")
        dialog.resize(500, 400)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {Colors.BACKGROUND}; }}
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel(f"<b>{concept.get('name', 'Unknown')}</b>")
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px;")
        layout.addWidget(header)

        meta_text = f"""
<table>
<tr><td style='color: {Colors.TEXT_MUTED};'>Type:</td><td>{concept.get('type', 'N/A')}</td></tr>
<tr><td style='color: {Colors.TEXT_MUTED};'>Domain:</td><td>{concept.get('domain', 'N/A')}</td></tr>
<tr><td style='color: {Colors.TEXT_MUTED};'>Occurrences:</td><td>{concept.get('count', 0)}</td></tr>
</table>
"""
        meta_label = QLabel(meta_text)
        meta_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(meta_label)

        content_label = QLabel("Content:")
        content_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(content_label)

        content_edit = QTextEdit()
        content_edit.setReadOnly(True)
        content = concept.get('content', {})
        if isinstance(content, dict):
            content_text = "\n".join(f"{k}: {v}" for k, v in content.items())
        else:
            content_text = str(content)
        content_edit.setPlainText(content_text)
        layout.addWidget(content_edit, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
        """)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)
        dialog.exec()

    # ============ Maintenance Methods ============

    @Slot()
    def _analyze_database(self):
        """Analyze database quality."""
        if not self._knowledge_store:
            self._log_audit("Knowledge store not available")
            return

        try:
            self._log_audit("Analyzing database...")

            entries = self._knowledge_store.advanced_search(limit=10000)
            total = len(entries)

            by_type = {}
            orphans = 0
            for e in entries:
                t = str(e.knowledge_type) if e.knowledge_type else 'unknown'
                by_type[t] = by_type.get(t, 0) + 1
                if not e.content:
                    orphans += 1

            quality_text = f"Total entries: {total}\n"
            quality_text += f"Orphan entries: {orphans}\n"
            for t, c in sorted(by_type.items()):
                quality_text += f"{t}: {c}\n"

            self._quality_text.setText(quality_text.strip())
            self._log_audit(f"Analysis complete: {total} entries, {orphans} orphans")

        except Exception as e:
            self._log_audit(f"Analysis failed: {e}")
            logger.error(f"Database analysis failed: {e}")

    @Slot()
    def _vacuum_database(self):
        """Vacuum/optimize the database."""
        if not self._knowledge_store:
            self._log_audit("Knowledge store not available")
            return

        try:
            self._log_audit("Vacuuming database...")

            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if storage_path:
                conn = sqlite3.connect(storage_path)
                conn.execute("VACUUM")
                conn.close()
                self._log_audit("Vacuum complete")
            else:
                self._log_audit("Storage path not available")

        except Exception as e:
            self._log_audit(f"Vacuum failed: {e}")
            logger.error(f"Database vacuum failed: {e}")

    @Slot()
    def _cleanup_orphans(self):
        """Clean up orphan entries."""
        if not self._knowledge_store:
            self._log_audit("Knowledge store not available")
            return

        reply = QMessageBox.question(
            self, "Confirm Cleanup",
            "This will remove orphan entries with no content.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self._log_audit("Cleaning up orphans...")

            storage_path = getattr(self._knowledge_store, 'storage_path', None)
            if storage_path:
                conn = sqlite3.connect(storage_path)
                cursor = conn.execute(
                    "DELETE FROM knowledge_entries WHERE content_json IS NULL AND content_compressed IS NULL"
                )
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                self._log_audit(f"Cleaned up {deleted} orphan entries")
            else:
                self._log_audit("Storage path not available")

        except Exception as e:
            self._log_audit(f"Cleanup failed: {e}")
            logger.error(f"Orphan cleanup failed: {e}")

    def _log_audit(self, message: str):
        """Log audit message to maintenance tab."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        current = self._audit_log.toPlainText()
        lines = current.split('\n')[-50:]
        lines.append(f"[{timestamp}] {message}")
        self._audit_log.setPlainText('\n'.join(lines))
        scrollbar = self._audit_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def cleanup(self):
        """Clean up resources."""
        if self._refresh_timer:
            self._refresh_timer.stop()
            self._refresh_timer = None
        self._knowledge_store = None
        self._knowledge_retriever = None
        self._knowledge_daemon = None
