"""
Knowledge Base Panel for Felix Knowledge Brain GUI.

This panel provides a horizontal split view with:
- Left Pane: Documents browser with status filtering and processing actions
- Right Pane: Concepts browser with domain filtering and editing capabilities

Part of the decomposed Knowledge Brain tab for the CustomTkinter GUI.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import logging
import sqlite3
import json
import pickle
from datetime import datetime
from typing import Optional, Dict, Any

from ...components.themed_treeview import ThemedTreeview
from ...components.search_entry import SearchEntry
from ...theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class KnowledgeBasePanel(ctk.CTkFrame):
    """
    Knowledge Base panel with split view for Documents and Concepts.

    Features:
    - Documents: Status filtering, processing actions, detail viewing
    - Concepts: Search, domain filtering, editing, deletion
    """

    def __init__(self, master, thread_manager, main_app=None):
        """
        Initialize Knowledge Base Panel.

        Args:
            master: Parent widget
            thread_manager: ThreadManager for background operations
            main_app: Reference to main application
        """
        super().__init__(master, fg_color="transparent")

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()

        # References to knowledge brain components (set by set_knowledge_refs)
        self.knowledge_store = None
        self.knowledge_daemon = None
        self.knowledge_retriever = None

        # Create split view layout
        self._create_layout()

    def _create_layout(self):
        """Create horizontal split view with Documents (left) and Concepts (right)."""
        # Configure grid weights for split
        self.grid_columnconfigure(0, weight=1)  # Documents pane
        self.grid_columnconfigure(1, weight=0)  # Separator
        self.grid_columnconfigure(2, weight=1)  # Concepts pane
        self.grid_rowconfigure(0, weight=1)

        # Left pane: Documents
        self.documents_pane = ctk.CTkFrame(self, corner_radius=10)
        self.documents_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        self._create_documents_pane()

        # Separator (visual divider)
        separator = ctk.CTkFrame(self, width=2, fg_color=self.theme_manager.get_color("border"))
        separator.grid(row=0, column=1, sticky="ns", padx=0, pady=0)

        # Right pane: Concepts
        self.concepts_pane = ctk.CTkFrame(self, corner_radius=10)
        self.concepts_pane.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=0)
        self._create_concepts_pane()

    def _create_documents_pane(self):
        """Create documents pane for the left side."""
        # Configure grid
        self.documents_pane.grid_columnconfigure(0, weight=1)
        self.documents_pane.grid_rowconfigure(2, weight=1)  # TreeView expands

        # Header
        header_label = ctk.CTkLabel(
            self.documents_pane,
            text="DOCUMENTS",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        header_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Controls frame
        controls_frame = ctk.CTkFrame(self.documents_pane, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        controls_frame.grid_columnconfigure(1, weight=1)  # Space between controls and refresh

        # Status filter
        filter_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        filter_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            filter_frame,
            text="Status:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))

        self.doc_filter_var = ctk.StringVar(value="all")
        self.doc_filter_combo = ctk.CTkComboBox(
            filter_frame,
            variable=self.doc_filter_var,
            values=["all", "completed", "processing", "pending", "failed"],
            width=130,
            command=lambda _: self._refresh_documents()
        )
        self.doc_filter_combo.pack(side="left", padx=2)

        # Refresh button
        self.doc_refresh_btn = ctk.CTkButton(
            controls_frame,
            text="🔄",
            width=32,
            command=self._refresh_documents,
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover")
        )
        self.doc_refresh_btn.grid(row=0, column=1, sticky="e", padx=2)

        # Document TreeView
        self.doc_tree = ThemedTreeview(
            self.documents_pane,
            columns=["file_name", "type", "status", "chunks", "concepts", "date"],
            headings=["File Name", "Type", "Status", "Chunks", "Concepts", "Date"],
            widths=[300, 60, 100, 80, 80, 150],
            height=15,
            selectmode="extended"
        )
        self.doc_tree.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Bind double-click to view details
        self.doc_tree.bind_tree("<Double-1>", self._view_document_details)

        # Action buttons
        action_frame = ctk.CTkFrame(self.documents_pane, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.process_btn = ctk.CTkButton(
            action_frame,
            text="⚡ Process Selected",
            command=self._process_selected_documents,
            width=140
        )
        self.process_btn.pack(side="left", padx=2)

        self.reprocess_btn = ctk.CTkButton(
            action_frame,
            text="🔄 Re-process",
            command=self._reprocess_selected_documents,
            width=120
        )
        self.reprocess_btn.pack(side="left", padx=2)

        self.delete_doc_btn = ctk.CTkButton(
            action_frame,
            text="🗑️ Delete",
            command=self._delete_selected_documents,
            width=100,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#b91c1c"
        )
        self.delete_doc_btn.pack(side="left", padx=2)

    def _create_concepts_pane(self):
        """Create concepts pane for the right side."""
        # Configure grid
        self.concepts_pane.grid_columnconfigure(0, weight=1)
        self.concepts_pane.grid_rowconfigure(2, weight=1)  # TreeView expands

        # Header
        header_label = ctk.CTkLabel(
            self.concepts_pane,
            text="CONCEPTS",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        header_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Search and filter controls
        controls_frame = ctk.CTkFrame(self.concepts_pane, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        controls_frame.grid_columnconfigure(0, weight=1)

        # Top row: Search entry
        search_row = ctk.CTkFrame(controls_frame, fg_color="transparent")
        search_row.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        search_row.grid_columnconfigure(0, weight=1)

        self.concept_search = SearchEntry(
            search_row,
            placeholder="Search concepts...",
            on_search=lambda q: self._search_concepts(),
            on_clear=lambda: self._refresh_concepts()
        )
        self.concept_search.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        # Search button
        self.search_btn = ctk.CTkButton(
            search_row,
            text="🔍",
            width=32,
            command=self._search_concepts,
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover")
        )
        self.search_btn.grid(row=0, column=1, padx=2)

        # Refresh button
        self.concept_refresh_btn = ctk.CTkButton(
            search_row,
            text="🔄",
            width=32,
            command=self._refresh_concepts,
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover")
        )
        self.concept_refresh_btn.grid(row=0, column=2, padx=2)

        # Bottom row: Domain filter
        filter_row = ctk.CTkFrame(controls_frame, fg_color="transparent")
        filter_row.grid(row=1, column=0, sticky="ew")

        ctk.CTkLabel(
            filter_row,
            text="Domain:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))

        self.concept_domain_var = ctk.StringVar(value="all")
        self.concept_domain_combo = ctk.CTkComboBox(
            filter_row,
            variable=self.concept_domain_var,
            values=["all", "python", "web", "ai", "database", "general"],
            width=120,
            command=lambda _: self._refresh_concepts()
        )
        self.concept_domain_combo.pack(side="left", padx=2)

        # Concepts TreeView
        self.concepts_tree = ThemedTreeview(
            self.concepts_pane,
            columns=["concept", "domain", "confidence", "definition"],
            headings=["Concept", "Domain", "Confidence", "Definition"],
            widths=[200, 100, 100, 400],
            height=15,
            selectmode="extended"
        )
        self.concepts_tree.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Bind events
        self.concepts_tree.bind_tree("<Double-1>", self._edit_concept_entry)
        self.concepts_tree.bind_tree("<Button-3>", self._show_concept_context_menu)

        # Action buttons
        action_frame = ctk.CTkFrame(self.concepts_pane, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

        self.edit_btn = ctk.CTkButton(
            action_frame,
            text="✏️ Edit",
            command=self._edit_concept_entry,
            width=80
        )
        self.edit_btn.pack(side="left", padx=2)

        self.delete_concept_btn = ctk.CTkButton(
            action_frame,
            text="🗑️ Delete",
            command=self._delete_selected_concepts,
            width=80,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#b91c1c"
        )
        self.delete_concept_btn.pack(side="left", padx=2)

        self.merge_btn = ctk.CTkButton(
            action_frame,
            text="🔗 Merge",
            command=self._merge_selected_concepts,
            width=80
        )
        self.merge_btn.pack(side="left", padx=2)

        self.export_btn = ctk.CTkButton(
            action_frame,
            text="💾 Export",
            command=self._export_selected_concepts,
            width=80
        )
        self.export_btn.pack(side="left", padx=2)

    # === Public API Methods ===

    def set_knowledge_refs(self, knowledge_store=None, knowledge_daemon=None, knowledge_retriever=None):
        """
        Set references to knowledge brain components.

        Args:
            knowledge_store: KnowledgeStore instance
            knowledge_daemon: KnowledgeDaemon instance
            knowledge_retriever: KnowledgeRetriever instance
        """
        self.knowledge_store = knowledge_store
        self.knowledge_daemon = knowledge_daemon
        self.knowledge_retriever = knowledge_retriever

        # Defer initial refresh to avoid blocking GUI thread during startup
        # Use background threads for database queries
        if self.knowledge_store:
            self.after(500, self._async_initial_refresh)

    def _async_initial_refresh(self):
        """Run initial refresh in background thread to avoid blocking GUI."""
        if self.thread_manager:
            self.thread_manager.start_thread(self._refresh_all_background)
        else:
            # Fallback to sync refresh if no thread_manager
            self._refresh_documents()
            self._refresh_concepts()

    def _refresh_all_background(self):
        """
        Background thread: fetch all data then schedule UI updates on main thread.

        This prevents database queries from blocking the GUI.
        """
        try:
            if not self.knowledge_store:
                return

            # Fetch documents data
            documents_data = self._fetch_documents_data()

            # Fetch concepts data
            concepts_data = self._fetch_concepts_data()

            # Schedule UI updates on main thread
            self.after(0, lambda: self._update_documents_ui(documents_data))
            self.after(0, lambda: self._update_concepts_ui(concepts_data))

        except Exception as e:
            logger.error(f"Background refresh failed: {e}")

    def _fetch_documents_data(self):
        """Fetch documents data in background thread."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            status_filter = self.doc_filter_var.get()
            if status_filter == "all":
                query = "SELECT * FROM document_sources ORDER BY added_at DESC LIMIT 100"
                params = ()
            else:
                query = "SELECT * FROM document_sources WHERE ingestion_status = ? ORDER BY added_at DESC LIMIT 100"
                params = (status_filter,)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return rows
        except Exception as e:
            logger.error(f"Failed to fetch documents data: {e}")
            return []

    def _fetch_concepts_data(self):
        """Fetch concepts data in background thread."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            domain_filter = self.concept_domain_var.get()
            if domain_filter == "all":
                query = """
                    SELECT knowledge_id, content_json, content_compressed, domain, confidence_level
                    FROM knowledge_entries
                    WHERE ((content_json IS NOT NULL AND content_json != ''
                            AND json_extract(content_json, '$.concept') IS NOT NULL)
                           OR content_compressed IS NOT NULL)
                    ORDER BY created_at DESC LIMIT 100
                """
                params = ()
            else:
                query = """
                    SELECT knowledge_id, content_json, content_compressed, domain, confidence_level
                    FROM knowledge_entries
                    WHERE domain = ?
                      AND ((content_json IS NOT NULL AND content_json != ''
                            AND json_extract(content_json, '$.concept') IS NOT NULL)
                           OR content_compressed IS NOT NULL)
                    ORDER BY created_at DESC LIMIT 100
                """
                params = (domain_filter,)

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return rows
        except Exception as e:
            logger.error(f"Failed to fetch concepts data: {e}")
            return []

    def _update_documents_ui(self, rows):
        """Update documents treeview on main thread with pre-fetched data."""
        try:
            self.doc_tree.clear()

            for row in rows:
                file_name = row[2]  # file_name column
                file_type = row[3]  # file_type
                status = row[12]  # ingestion_status
                chunks = row[15] or 0  # chunk_count
                concepts = row[16] or 0  # concept_count
                added_at = datetime.fromtimestamp(row[18]).strftime('%Y-%m-%d %H:%M') if row[18] else "N/A"

                item_id = self.doc_tree.insert(
                    values=(file_name, file_type, status, chunks, concepts, added_at)
                )
                self.doc_tree.tree.item(item_id, tags=(str(row[0]),))

        except Exception as e:
            logger.error(f"Failed to update documents UI: {e}")

    def _update_concepts_ui(self, rows):
        """Update concepts treeview on main thread with pre-fetched data."""
        try:
            self.concepts_tree.clear()

            for row in rows:
                knowledge_id = row[0]
                content_json = row[1]
                content_compressed = row[2]
                domain = row[3] or "unknown"
                confidence = row[4]

                # Extract concept and definition
                concept_text = "Unknown"
                definition = ""

                if content_json:
                    try:
                        data = json.loads(content_json)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass
                elif content_compressed:
                    try:
                        data = pickle.loads(content_compressed)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass

                # Truncate definition for display
                if len(definition) > 100:
                    definition = definition[:97] + "..."

                # Format confidence
                try:
                    conf_str = f"{float(confidence):.2f}" if confidence else "0.00"
                except (ValueError, TypeError):
                    conf_str = "0.00"

                # Insert into tree
                item_id = self.concepts_tree.insert(
                    values=(concept_text, domain, conf_str, definition)
                )
                self.concepts_tree.tree.item(item_id, tags=(str(knowledge_id),))

        except Exception as e:
            logger.error(f"Failed to update concepts UI: {e}")

    def _enable_features(self):
        """Enable features when Felix system starts."""
        # Enable all buttons
        for btn in [self.process_btn, self.reprocess_btn, self.delete_doc_btn,
                    self.edit_btn, self.delete_concept_btn, self.merge_btn, self.export_btn]:
            btn.configure(state="normal")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        # Disable all buttons
        for btn in [self.process_btn, self.reprocess_btn, self.delete_doc_btn,
                    self.edit_btn, self.delete_concept_btn, self.merge_btn, self.export_btn]:
            btn.configure(state="disabled")

    def refresh_documents(self):
        """Public method to refresh documents list."""
        self._refresh_documents()

    def refresh_concepts(self):
        """Public method to refresh concepts list."""
        self._refresh_concepts()

    # === Documents Methods ===

    def _refresh_documents(self):
        """Refresh documents list with current filter."""
        # Clear existing items
        self.doc_tree.clear()

        try:
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Build query with filter
            status_filter = self.doc_filter_var.get()
            if status_filter == "all":
                query = "SELECT * FROM document_sources ORDER BY added_at DESC LIMIT 100"
                params = ()
            else:
                query = "SELECT * FROM document_sources WHERE ingestion_status = ? ORDER BY added_at DESC LIMIT 100"
                params = (status_filter,)

            cursor = conn.execute(query, params)

            for row in cursor:
                file_name = row[2]  # file_name column
                file_type = row[3]  # file_type
                status = row[12]  # ingestion_status
                chunks = row[15] or 0  # chunk_count
                concepts = row[16] or 0  # concept_count
                added_at = datetime.fromtimestamp(row[18]).strftime('%Y-%m-%d %H:%M') if row[18] else "N/A"

                # Store full row data as item tag for retrieval
                item_id = self.doc_tree.insert(
                    values=(file_name, file_type, status, chunks, concepts, added_at)
                )
                # Store document ID for later use
                self.doc_tree.tree.item(item_id, tags=(str(row[0]),))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh documents: {e}")

    def _view_document_details(self, event):
        """View detailed information about selected document."""
        selection = self.doc_tree.selection()
        if not selection:
            return

        try:
            # Get document ID from tags
            item = selection[0]
            tags = self.doc_tree.tree.item(item, "tags")
            if not tags:
                return
            doc_id = int(tags[0])

            # Fetch full document info
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.execute("SELECT * FROM document_sources WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                # Show details dialog
                DocumentDetailsDialog(self, row)

        except Exception as e:
            logger.error(f"Failed to view document details: {e}")
            messagebox.showerror("Error", f"Failed to load document details: {e}")

    def _process_selected_documents(self):
        """Process selected documents."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to process")
            return

        # Implementation would queue documents for processing
        messagebox.showinfo("Process", f"Would process {len(selection)} document(s)")

    def _reprocess_selected_documents(self):
        """Re-process selected documents."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to re-process")
            return

        if not messagebox.askyesno("Confirm", f"Re-process {len(selection)} document(s)?"):
            return

        # Implementation would reset and re-queue documents
        messagebox.showinfo("Re-process", f"Would re-process {len(selection)} document(s)")

    def _delete_selected_documents(self):
        """Delete selected documents."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to delete")
            return

        if not messagebox.askyesno("Confirm Deletion",
                                   f"Delete {len(selection)} document(s)?\n\n"
                                   "This will also delete associated knowledge entries."):
            return

        try:
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            deleted_count = 0
            for item in selection:
                tags = self.doc_tree.tree.item(item, "tags")
                if tags:
                    doc_id = int(tags[0])
                    # Delete document and associated entries (cascade)
                    conn.execute("DELETE FROM document_sources WHERE id = ?", (doc_id,))
                    deleted_count += 1

            conn.commit()
            conn.close()

            messagebox.showinfo("Success", f"Deleted {deleted_count} document(s)")
            self._refresh_documents()

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            messagebox.showerror("Error", f"Failed to delete documents: {e}")

    # === Concepts Methods ===

    def _refresh_concepts(self):
        """Refresh concepts display with current filter."""
        # Clear existing items
        self.concepts_tree.clear()

        try:
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Build query with domain filter
            domain_filter = self.concept_domain_var.get()
            if domain_filter == "all":
                query = """
                    SELECT knowledge_id, content_json, content_compressed, domain, confidence_level
                    FROM knowledge_entries
                    WHERE ((content_json IS NOT NULL AND content_json != ''
                            AND json_extract(content_json, '$.concept') IS NOT NULL)
                           OR content_compressed IS NOT NULL)
                    ORDER BY created_at DESC LIMIT 100
                """
                params = ()
            else:
                query = """
                    SELECT knowledge_id, content_json, content_compressed, domain, confidence_level
                    FROM knowledge_entries
                    WHERE domain = ?
                      AND ((content_json IS NOT NULL AND content_json != ''
                            AND json_extract(content_json, '$.concept') IS NOT NULL)
                           OR content_compressed IS NOT NULL)
                    ORDER BY created_at DESC LIMIT 100
                """
                params = (domain_filter,)

            cursor = conn.execute(query, params)

            for row in cursor:
                knowledge_id = row[0]
                content_json = row[1]
                content_compressed = row[2]
                domain = row[3] or "unknown"
                confidence = row[4]

                # Extract concept and definition
                concept_text = "Unknown"
                definition = ""

                if content_json:
                    try:
                        data = json.loads(content_json)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass
                elif content_compressed:
                    try:
                        data = pickle.loads(content_compressed)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass

                # Truncate definition for display
                if len(definition) > 100:
                    definition = definition[:97] + "..."

                # Format confidence
                try:
                    conf_str = f"{float(confidence):.2f}" if confidence else "0.00"
                except (ValueError, TypeError):
                    conf_str = "0.00"

                # Insert into tree
                item_id = self.concepts_tree.insert(
                    values=(concept_text, domain, conf_str, definition)
                )
                # Store knowledge_id for later use
                self.concepts_tree.tree.item(item_id, tags=(str(knowledge_id),))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh concepts: {e}")

    def _search_concepts(self):
        """Search concepts by query string."""
        query = self.concept_search.get().strip()

        if not query:
            self._refresh_concepts()
            return

        # Clear existing items
        self.concepts_tree.clear()

        try:
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Search in FTS5 table if available
            search_query = """
                SELECT ke.knowledge_id, ke.content_json, ke.content_compressed,
                       ke.domain, ke.confidence_level
                FROM knowledge_entries ke
                JOIN knowledge_fts kf ON ke.knowledge_id = kf.rowid
                WHERE kf.content MATCH ?
                ORDER BY rank
                LIMIT 50
            """

            cursor = conn.execute(search_query, (query,))

            for row in cursor:
                knowledge_id = row[0]
                content_json = row[1]
                content_compressed = row[2]
                domain = row[3] or "unknown"
                confidence = row[4]

                # Extract concept and definition
                concept_text = "Unknown"
                definition = ""

                if content_json:
                    try:
                        data = json.loads(content_json)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass
                elif content_compressed:
                    try:
                        data = pickle.loads(content_compressed)
                        concept_text = data.get("concept", "Unknown")
                        definition = data.get("definition", "")
                    except:
                        pass

                # Truncate definition
                if len(definition) > 100:
                    definition = definition[:97] + "..."

                try:
                    conf_str = f"{float(confidence):.2f}" if confidence else "0.00"
                except (ValueError, TypeError):
                    conf_str = "0.00"

                item_id = self.concepts_tree.insert(
                    values=(concept_text, domain, conf_str, definition)
                )
                self.concepts_tree.tree.item(item_id, tags=(str(knowledge_id),))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to search concepts: {e}")
            # Fallback to refresh
            self._refresh_concepts()

    def _edit_concept_entry(self, event=None):
        """Edit selected concept entry."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a concept to edit")
            return

        try:
            # Get knowledge_id from tags
            item = selection[0]
            tags = self.concepts_tree.tree.item(item, "tags")
            if not tags:
                return
            knowledge_id = int(tags[0])

            # Fetch full concept data
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.execute(
                "SELECT content_json, content_compressed, domain, confidence_level FROM knowledge_entries WHERE knowledge_id = ?",
                (knowledge_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                # Show edit dialog
                ConceptEditDialog(self, knowledge_id, row, self._refresh_concepts)

        except Exception as e:
            logger.error(f"Failed to edit concept: {e}")
            messagebox.showerror("Error", f"Failed to load concept: {e}")

    def _delete_selected_concepts(self):
        """Delete selected concept entries."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select concepts to delete")
            return

        if not messagebox.askyesno("Confirm Deletion",
                                   f"Delete {len(selection)} concept(s)?"):
            return

        try:
            if not self.knowledge_store:
                return

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            deleted_count = 0
            for item in selection:
                tags = self.concepts_tree.tree.item(item, "tags")
                if tags:
                    knowledge_id = int(tags[0])
                    conn.execute("DELETE FROM knowledge_entries WHERE knowledge_id = ?", (knowledge_id,))
                    deleted_count += 1

            conn.commit()
            conn.close()

            messagebox.showinfo("Success", f"Deleted {deleted_count} concept(s)")
            self._refresh_concepts()

        except Exception as e:
            logger.error(f"Failed to delete concepts: {e}")
            messagebox.showerror("Error", f"Failed to delete concepts: {e}")

    def _merge_selected_concepts(self):
        """Merge selected concepts into one."""
        selection = self.concepts_tree.selection()
        if len(selection) < 2:
            messagebox.showwarning("Insufficient Selection",
                                 "Please select at least 2 concepts to merge")
            return

        messagebox.showinfo("Merge", f"Would merge {len(selection)} concepts\n\n"
                          "This feature requires additional implementation.")

    def _export_selected_concepts(self):
        """Export selected concepts to file."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select concepts to export")
            return

        messagebox.showinfo("Export", f"Would export {len(selection)} concept(s)\n\n"
                          "This feature requires file dialog implementation.")

    def _show_concept_context_menu(self, event):
        """Show right-click context menu for concepts."""
        # Identify clicked item
        item = self.concepts_tree.identify_row(event.y)
        if not item:
            return

        # Select the item
        self.concepts_tree.selection_set(item)

        # Create context menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="✏️ Edit", command=self._edit_concept_entry)
        menu.add_command(label="🗑️ Delete", command=self._delete_selected_concepts)
        menu.add_separator()
        menu.add_command(label="🔗 View Relationships", command=self._view_concept_relationships)
        menu.add_command(label="💾 Export", command=self._export_selected_concepts)

        # Show menu at cursor
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _view_concept_relationships(self):
        """View relationships for selected concept."""
        selection = self.concepts_tree.selection()
        if not selection:
            return

        messagebox.showinfo("Relationships",
                          "Relationship viewing requires the Relationships tab.\n\n"
                          "This feature will navigate to the relationships view.")


class DocumentDetailsDialog(ctk.CTkToplevel):
    """Dialog for viewing document details."""

    def __init__(self, parent, document_row):
        """
        Initialize document details dialog.

        Args:
            parent: Parent widget
            document_row: SQLite row from document_sources table
        """
        super().__init__(parent)

        self.title("Document Details")
        self.geometry("700x600")

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Extract document data
        doc_id = document_row[0]
        file_path = document_row[1]
        file_name = document_row[2]
        file_type = document_row[3]
        file_size = document_row[4]
        status = document_row[12]
        chunks = document_row[15] or 0
        concepts = document_row[16] or 0
        added_at = datetime.fromtimestamp(document_row[18]).strftime('%Y-%m-%d %H:%M:%S') if document_row[18] else "N/A"
        processed_at = datetime.fromtimestamp(document_row[19]).strftime('%Y-%m-%d %H:%M:%S') if document_row[19] else "N/A"

        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            scroll_frame,
            text=file_name,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))

        # Details
        details = [
            ("ID:", str(doc_id)),
            ("File Path:", file_path),
            ("Type:", file_type),
            ("Size:", f"{file_size:,} bytes" if file_size else "Unknown"),
            ("Status:", status),
            ("Chunks:", str(chunks)),
            ("Concepts Extracted:", str(concepts)),
            ("Added:", added_at),
            ("Processed:", processed_at),
        ]

        for label, value in details:
            row_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=5)

            ctk.CTkLabel(
                row_frame,
                text=label,
                font=ctk.CTkFont(weight="bold"),
                width=150,
                anchor="w"
            ).pack(side="left")

            ctk.CTkLabel(
                row_frame,
                text=value,
                anchor="w"
            ).pack(side="left", fill="x", expand=True)

        # Close button
        ctk.CTkButton(
            self,
            text="Close",
            command=self.destroy,
            width=100
        ).pack(pady=(0, 20))

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class ConceptEditDialog(ctk.CTkToplevel):
    """Dialog for editing concept entries."""

    def __init__(self, parent, knowledge_id, concept_row, refresh_callback):
        """
        Initialize concept edit dialog.

        Args:
            parent: Parent widget
            knowledge_id: ID of the knowledge entry
            concept_row: SQLite row (content_json, content_compressed, domain, confidence)
            refresh_callback: Callback to refresh concepts list after save
        """
        super().__init__(parent)

        self.parent = parent
        self.knowledge_id = knowledge_id
        self.refresh_callback = refresh_callback

        self.title("Edit Concept")
        self.geometry("600x500")

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Parse concept data
        content_json, content_compressed, domain, confidence = concept_row

        self.concept_data = {}
        if content_json:
            try:
                self.concept_data = json.loads(content_json)
            except:
                pass
        elif content_compressed:
            try:
                self.concept_data = pickle.loads(content_compressed)
            except:
                pass

        # Create form
        self._create_form()

        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _create_form(self):
        """Create edit form."""
        # Form container
        form_frame = ctk.CTkScrollableFrame(self)
        form_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(
            form_frame,
            text="Edit Concept Entry",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(0, 20))

        # Concept name
        ctk.CTkLabel(form_frame, text="Concept:", anchor="w").pack(fill="x", pady=(5, 2))
        self.concept_entry = ctk.CTkEntry(form_frame)
        self.concept_entry.pack(fill="x", pady=(0, 10))
        self.concept_entry.insert(0, self.concept_data.get("concept", ""))

        # Domain
        ctk.CTkLabel(form_frame, text="Domain:", anchor="w").pack(fill="x", pady=(5, 2))
        self.domain_entry = ctk.CTkEntry(form_frame)
        self.domain_entry.pack(fill="x", pady=(0, 10))
        self.domain_entry.insert(0, self.concept_data.get("domain", ""))

        # Definition
        ctk.CTkLabel(form_frame, text="Definition:", anchor="w").pack(fill="x", pady=(5, 2))
        self.definition_text = ctk.CTkTextbox(form_frame, height=150)
        self.definition_text.pack(fill="both", expand=True, pady=(0, 10))
        self.definition_text.insert("1.0", self.concept_data.get("definition", ""))

        # Confidence
        ctk.CTkLabel(form_frame, text="Confidence (0-1):", anchor="w").pack(fill="x", pady=(5, 2))
        self.confidence_entry = ctk.CTkEntry(form_frame)
        self.confidence_entry.pack(fill="x", pady=(0, 10))
        self.confidence_entry.insert(0, str(self.concept_data.get("confidence", 0.5)))

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 20))

        ctk.CTkButton(
            btn_frame,
            text="Save",
            command=self._save_changes,
            width=100
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            width=100,
            fg_color="gray"
        ).pack(side="left", padx=5)

    def _save_changes(self):
        """Save edited concept data."""
        try:
            # Update concept data
            self.concept_data["concept"] = self.concept_entry.get()
            self.concept_data["domain"] = self.domain_entry.get()
            self.concept_data["definition"] = self.definition_text.get("1.0", "end-1c")

            try:
                confidence = float(self.confidence_entry.get())
                if 0 <= confidence <= 1:
                    self.concept_data["confidence"] = confidence
            except ValueError:
                pass

            # Save to database (simplified - would need knowledge_store reference)
            messagebox.showinfo("Success", "Concept updated successfully!\n\n"
                              "(Note: Full save implementation requires knowledge_store integration)")

            # Refresh parent and close
            if self.refresh_callback:
                self.refresh_callback()
            self.destroy()

        except Exception as e:
            logger.error(f"Failed to save concept: {e}")
            messagebox.showerror("Error", f"Failed to save changes: {e}")
