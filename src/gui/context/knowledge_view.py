"""Knowledge browser view for viewing documents and concepts."""

import logging
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTextEdit, QSplitter,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit, QTabWidget, QMessageBox
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class KnowledgeView(QWidget):
    """Knowledge browser with documents and concepts.

    Features:
    - Document browser with search
    - Concept browser
    - Document content preview
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._knowledge_store = None
        self._knowledge_retriever = None
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

        # Documents tab
        docs_widget = self._create_documents_tab()
        self._tabs.addTab(docs_widget, "Documents")

        # Concepts tab
        concepts_widget = self._create_concepts_tab()
        self._tabs.addTab(concepts_widget, "Concepts")

        layout.addWidget(self._tabs)

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
        search_btn.setFixedWidth(60)
        search_btn.clicked.connect(self._search_documents)
        search_row.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(60)
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
        search_btn.setFixedWidth(60)
        search_btn.clicked.connect(self._search_concepts)
        search_row.addWidget(search_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(60)
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
        layout.addWidget(self._concept_tree, 1)

        # Stats row
        self._concept_stats = QLabel("Concepts: --")
        self._concept_stats.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._concept_stats)

        return widget

    def set_knowledge_refs(self, knowledge_store=None, knowledge_retriever=None, **kwargs):
        """Set knowledge brain references."""
        logger.debug(f"set_knowledge_refs called: store={knowledge_store is not None}, retriever={knowledge_retriever is not None}")
        self._knowledge_store = knowledge_store
        self._knowledge_retriever = knowledge_retriever
        if knowledge_store:
            logger.debug(f"Knowledge store path: {getattr(knowledge_store, 'storage_path', 'unknown')}")
            self._refresh_documents()
            self._refresh_concepts()

    @Slot()
    def _refresh_documents(self):
        """Refresh document list."""
        if not self._knowledge_store:
            logger.debug("Documents refresh: knowledge_store is None")
            self._doc_stats.setText("Documents: -- (not connected)")
            return

        try:
            # Get documents from knowledge store using advanced_search
            logger.debug(f"Documents refresh: querying knowledge store at {getattr(self._knowledge_store, 'storage_path', 'unknown')}")
            entries = self._knowledge_store.advanced_search(limit=100)
            logger.debug(f"Documents refresh: got {len(entries)} entries")

            self._doc_tree.clear()

            for entry in entries:
                # Convert KnowledgeEntry to display format
                # KnowledgeEntry has: domain, source_agent, source_doc_id, knowledge_id
                title = entry.domain or entry.source_doc_id or entry.knowledge_id or 'Unknown'
                if len(title) > 40:
                    title = title[:37] + "..."

                doc_type = entry.knowledge_type.value if entry.knowledge_type else 'unknown'
                added = str(entry.created_at)[:10] if entry.created_at else ''

                # Build doc dict for preview
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
            # Search using retriever - returns RetrievalContext with .results
            context = self._knowledge_retriever.search(query, top_k=20)

            self._doc_tree.clear()

            for result in context.results:
                # result is SearchResult with knowledge_entry and relevance_score
                entry = result.knowledge_entry
                if not entry:
                    continue

                # KnowledgeEntry has: domain, source_agent, source_doc_id, knowledge_id
                title = entry.domain or entry.source_doc_id or entry.knowledge_id or 'Unknown'
                if len(title) > 40:
                    title = title[:37] + "..."

                score = result.relevance_score
                doc_type = entry.knowledge_type.value if entry.knowledge_type else 'unknown'

                # Build doc dict for preview
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
            # Show document preview
            preview = f"Title: {doc.get('title', 'N/A')}\n"
            preview += f"Source: {doc.get('source', 'N/A')}\n"
            preview += f"Type: {doc.get('doc_type', 'N/A')}\n"
            preview += f"Created: {doc.get('created_at', 'N/A')}\n"
            preview += "\n---\n\n"

            content = doc.get('content', doc.get('text', ''))
            if len(content) > 1000:
                content = content[:1000] + "...\n\n[Content truncated]"
            preview += content

            self._doc_preview.setPlainText(preview)

    @Slot()
    def _refresh_concepts(self):
        """Refresh concept list."""
        if not self._knowledge_store:
            self._concept_stats.setText("Concepts: -- (not connected)")
            return

        try:
            # Get concepts from knowledge store using advanced_search with concept tag
            entries = self._knowledge_store.advanced_search(tags=['concept'], limit=100)

            self._concept_tree.clear()

            for entry in entries:
                # Extract concept name from content or domain
                if isinstance(entry.content, dict):
                    name = entry.content.get('concept', entry.content.get('topic', entry.domain or 'Unknown'))
                else:
                    name = entry.domain or 'Unknown'

                if len(name) > 40:
                    name = name[:37] + "..."

                concept_type = 'concept'
                count = entry.access_count or 0

                # Build concept dict for storage
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
            # Search concepts using advanced_search with content query and concept tag
            entries = self._knowledge_store.advanced_search(content=query, tags=['concept'], limit=50)

            self._concept_tree.clear()

            for entry in entries:
                # Extract concept name from content or domain
                if isinstance(entry.content, dict):
                    name = entry.content.get('concept', entry.content.get('topic', entry.domain or 'Unknown'))
                else:
                    name = entry.domain or 'Unknown'

                if len(name) > 40:
                    name = name[:37] + "..."

                concept_type = 'concept'
                count = entry.access_count or 0

                # Build concept dict for storage
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

    def cleanup(self):
        """Clean up resources."""
        self._knowledge_store = None
        self._knowledge_retriever = None
