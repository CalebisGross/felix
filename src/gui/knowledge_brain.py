"""
Knowledge Brain GUI Tab for Felix

Provides monitoring and control interface for the autonomous knowledge brain:
- Overview: Status, statistics, daemon control
- Documents: Browse ingested sources, view status
- Concepts: Explore extracted knowledge
- Activity: Real-time processing log
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from datetime import datetime
from pathlib import Path

from .utils import ThreadManager
from .themes import ThemeManager

logger = logging.getLogger(__name__)


class KnowledgeBrainFrame(ttk.Frame):
    """Knowledge Brain monitoring and control interface."""

    def __init__(self, parent, thread_manager: ThreadManager, main_app=None, theme_manager: ThemeManager = None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager

        # References to knowledge brain components (set by main_app)
        self.knowledge_daemon = None
        self.knowledge_retriever = None
        self.knowledge_store = None

        # Auto-refresh state
        self.auto_refresh_enabled = False
        self.refresh_interval = 5000  # 5 seconds

        # Create notebook with 4 tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Overview
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="Overview")
        self._create_overview_tab()

        # Tab 2: Documents
        self.documents_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.documents_frame, text="Documents")
        self._create_documents_tab()

        # Tab 3: Concepts
        self.concepts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.concepts_frame, text="Concepts")
        self._create_concepts_tab()

        # Tab 4: Activity
        self.activity_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.activity_frame, text="Activity")
        self._create_activity_tab()

    def _enable_features(self):
        """Enable features when Felix system starts."""
        if self.main_app and self.main_app.felix_system:
            # Wire up references to knowledge brain components
            self.knowledge_store = self.main_app.felix_system.knowledge_store
            self.knowledge_retriever = self.main_app.felix_system.knowledge_retriever
            self.knowledge_daemon = self.main_app.felix_system.knowledge_daemon

            # Update display to show current status
            self._refresh_overview()

            logger.info("Knowledge Brain tab features enabled")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        # Clear references
        self.knowledge_store = None
        self.knowledge_retriever = None
        self.knowledge_daemon = None

        # Update display to show disconnected state
        self._refresh_overview()

        logger.info("Knowledge Brain tab features disabled")

    def _create_overview_tab(self):
        """Create overview tab with status and statistics."""
        # Control buttons
        control_frame = ttk.Frame(self.overview_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Daemon Control:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.start_daemon_btn = ttk.Button(control_frame, text="▶ Start Daemon", command=self._start_daemon)
        self.start_daemon_btn.pack(side=tk.LEFT, padx=2)

        self.stop_daemon_btn = ttk.Button(control_frame, text="■ Stop Daemon", command=self._stop_daemon, state=tk.DISABLED)
        self.stop_daemon_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(control_frame, text="📂 Add Directory", command=self._add_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🗑️ Manage Directories", command=self._manage_directories).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🔄 Force Refinement", command=self._force_refinement).pack(side=tk.LEFT, padx=2)

        # Auto-refresh toggle
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Auto-Refresh", variable=self.auto_refresh_var,
                       command=self._toggle_auto_refresh).pack(side=tk.LEFT, padx=2)

        # Status display
        status_frame = ttk.LabelFrame(self.overview_frame, text="Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Scrolled text for status
        text_frame = ttk.Frame(status_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.overview_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80)
        overview_scrollbar = ttk.Scrollbar(text_frame, command=self.overview_text.yview)
        self.overview_text.config(yscrollcommand=overview_scrollbar.set)

        self.overview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        overview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Refresh button
        ttk.Button(status_frame, text="🔄 Refresh", command=self._refresh_overview).pack(pady=(5, 0))

        # Initial refresh
        self._refresh_overview()

    def _create_documents_tab(self):
        """Create documents tab for browsing ingested sources."""
        # Controls
        control_frame = ttk.Frame(self.documents_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Filter:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))

        self.doc_filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.doc_filter_var, state="readonly", width=15)
        filter_combo['values'] = ("all", "completed", "processing", "pending", "failed")
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_documents())

        ttk.Button(control_frame, text="🔄 Refresh", command=self._refresh_documents).pack(side=tk.LEFT, padx=5)

        # Document list (TreeView)
        list_frame = ttk.Frame(self.documents_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create treeview
        columns = ("file_name", "type", "status", "chunks", "concepts", "date")
        self.doc_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)

        self.doc_tree.heading("file_name", text="File Name")
        self.doc_tree.heading("type", text="Type")
        self.doc_tree.heading("status", text="Status")
        self.doc_tree.heading("chunks", text="Chunks")
        self.doc_tree.heading("concepts", text="Concepts")
        self.doc_tree.heading("date", text="Date")

        self.doc_tree.column("file_name", width=300)
        self.doc_tree.column("type", width=60)
        self.doc_tree.column("status", width=100)
        self.doc_tree.column("chunks", width=80)
        self.doc_tree.column("concepts", width=80)
        self.doc_tree.column("date", width=150)

        doc_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.doc_tree.yview)
        self.doc_tree.config(yscrollcommand=doc_scrollbar.set)

        self.doc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        doc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Double-click to view details
        self.doc_tree.bind("<Double-1>", self._view_document_details)

        # Initial refresh
        self._refresh_documents()

    def _create_concepts_tab(self):
        """Create concepts tab for exploring extracted knowledge."""
        # Search controls
        search_frame = ttk.Frame(self.concepts_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(search_frame, text="Search:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))

        self.concept_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.concept_search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<Return>", lambda e: self._search_concepts())

        ttk.Button(search_frame, text="🔍 Search", command=self._search_concepts).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="🔄 Show All", command=self._refresh_concepts).pack(side=tk.LEFT, padx=5)

        # Domain filter
        ttk.Label(search_frame, text="Domain:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(20, 5))
        self.concept_domain_var = tk.StringVar(value="all")
        domain_combo = ttk.Combobox(search_frame, textvariable=self.concept_domain_var, state="readonly", width=15)
        domain_combo['values'] = ("all", "python", "web", "ai", "database", "general")
        domain_combo.pack(side=tk.LEFT, padx=5)
        domain_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_concepts())

        # Concepts display
        display_frame = ttk.Frame(self.concepts_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.concepts_text = tk.Text(display_frame, wrap=tk.WORD, height=20, width=80)
        concepts_scrollbar = ttk.Scrollbar(display_frame, command=self.concepts_text.yview)
        self.concepts_text.config(yscrollcommand=concepts_scrollbar.set)

        self.concepts_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        concepts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial refresh
        self._refresh_concepts()

    def _create_activity_tab(self):
        """Create activity tab for real-time processing log."""
        # Controls
        control_frame = ttk.Frame(self.activity_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="🔄 Refresh", command=self._refresh_activity).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑️ Clear Log", command=self._clear_activity_log).pack(side=tk.LEFT, padx=5)

        # Activity log
        log_frame = ttk.Frame(self.activity_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.activity_text = tk.Text(log_frame, wrap=tk.WORD, height=20, width=80)
        activity_scrollbar = ttk.Scrollbar(log_frame, command=self.activity_text.yview)
        self.activity_text.config(yscrollcommand=activity_scrollbar.set)

        self.activity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        activity_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial activity
        self._refresh_activity()

    # === Command Handlers ===

    def _start_daemon(self):
        """Start the knowledge daemon."""
        if not self.main_app or not hasattr(self.main_app, 'felix_system'):
            messagebox.showwarning("Not Available", "Knowledge Brain not initialized")
            return

        try:
            # Get or create daemon from main_app
            if not self.knowledge_daemon:
                messagebox.showinfo("Starting...", "Initializing Knowledge Daemon...\nThis may take a moment.")

            # Enable daemon in felix_system (main_app should handle this)
            if hasattr(self.main_app, 'start_knowledge_daemon'):
                success = self.main_app.start_knowledge_daemon()

                if success:
                    self.start_daemon_btn.config(state=tk.DISABLED)
                    self.stop_daemon_btn.config(state=tk.NORMAL)
                    self._log_activity("Daemon started")
                    self._refresh_overview()
                else:
                    messagebox.showerror("Start Failed", "Failed to start daemon. Check logs for details.")
            else:
                messagebox.showerror("Error", "start_knowledge_daemon method not found in main app")
                logger.error("MainApp missing start_knowledge_daemon method")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start daemon: {e}")
            logger.error(f"Failed to start daemon: {e}")

    def _stop_daemon(self):
        """Stop the knowledge daemon."""
        if self.knowledge_daemon:
            try:
                self.knowledge_daemon.stop()
                self.start_daemon_btn.config(state=tk.NORMAL)
                self.stop_daemon_btn.config(state=tk.DISABLED)
                self._log_activity("Daemon stopped")
                self._refresh_overview()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop daemon: {e}")

    def _add_directory(self):
        """Add a directory for processing."""
        directory = filedialog.askdirectory(title="Select Directory with Documents")
        if directory:
            if self.knowledge_daemon:
                try:
                    result = self.knowledge_daemon.process_directory_now(directory)
                    messagebox.showinfo("Success",
                                       f"Queued {result.get('queued', 0)} documents from:\n{directory}")
                    self._log_activity(f"Added directory: {directory} ({result.get('queued', 0)} documents)")
                    self._refresh_overview()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process directory: {e}")
            else:
                messagebox.showwarning("Daemon Not Running",
                                      "Please start the daemon first")

    def _manage_directories(self):
        """Open dialog to manage watched directories."""
        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        # Create popup window
        dialog = tk.Toplevel(self)
        dialog.title("Manage Watch Directories")
        dialog.geometry("700x450")
        dialog.transient(self)
        dialog.grab_set()

        # Title label
        ttk.Label(dialog, text="Currently Watched Directories:",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=10)

        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        dir_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=15, font=("TkDefaultFont", 10))
        dir_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=dir_listbox.yview)

        # Populate with current directories
        current_dirs = self.knowledge_daemon.config.watch_directories.copy()
        for directory in current_dirs:
            dir_listbox.insert(tk.END, directory)

        # Info label
        info_label = ttk.Label(dialog, text="", foreground="gray")
        info_label.pack(pady=5)

        def update_info():
            """Update info label with selection count."""
            selected = dir_listbox.curselection()
            if selected:
                info_label.config(text=f"{len(selected)} director{'y' if len(selected) == 1 else 'ies'} selected")
            else:
                info_label.config(text="Select directories to remove")

        dir_listbox.bind('<<ListboxSelect>>', lambda e: update_info())
        update_info()

        # Buttons frame
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=15)

        def remove_selected():
            """Remove selected directories from watch list."""
            selection = dir_listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select directories to remove")
                return

            # Get selected directories
            to_remove = [dir_listbox.get(idx) for idx in selection]

            # Confirm removal
            if len(to_remove) == 1:
                message = f"Remove this directory from watch list?\n\n{to_remove[0]}"
            else:
                message = f"Remove {len(to_remove)} directories from watch list?"

            if not messagebox.askyesno("Confirm Removal", message):
                return

            # Remove each directory
            removed_count = 0
            for directory in to_remove:
                try:
                    result = self.knowledge_daemon.remove_watch_directory(directory)
                    if result.get('success'):
                        removed_count += 1
                        self._log_activity(f"Removed directory: {directory}")
                    else:
                        logger.warning(f"Failed to remove {directory}: {result.get('error')}")
                except Exception as e:
                    logger.error(f"Error removing {directory}: {e}")

            # Update listbox
            for idx in reversed(selection):
                dir_listbox.delete(idx)

            # Show result
            if removed_count > 0:
                messagebox.showinfo("Success", f"Removed {removed_count} director{'y' if removed_count == 1 else 'ies'}")
                self._refresh_overview()
            update_info()

        ttk.Button(btn_frame, text="🗑️ Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✖ Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        # Center dialog on parent
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _force_refinement(self):
        """Manually trigger a refinement cycle."""
        if self.knowledge_daemon:
            try:
                self._log_activity("Starting manual refinement...")
                result = self.knowledge_daemon.trigger_refinement()
                messagebox.showinfo("Refinement Complete",
                                   f"Created {result.get('total_relationships', 0)} relationships")
                self._log_activity(f"Refinement complete: {result.get('total_relationships', 0)} relationships")
                self._refresh_overview()
            except Exception as e:
                messagebox.showerror("Error", f"Refinement failed: {e}")
        else:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh."""
        self.auto_refresh_enabled = self.auto_refresh_var.get()
        if self.auto_refresh_enabled:
            self._auto_refresh_loop()

    def _auto_refresh_loop(self):
        """Auto-refresh loop."""
        if self.auto_refresh_enabled:
            self._refresh_overview()
            self.after(self.refresh_interval, self._auto_refresh_loop)

    # === Refresh Methods ===

    def _refresh_overview(self):
        """Refresh overview statistics."""
        text = "KNOWLEDGE BRAIN STATUS\n"
        text += "=" * 70 + "\n\n"

        try:
            # Daemon status
            if self.knowledge_daemon:
                status = self.knowledge_daemon.get_status()

                daemon_status = "● Running" if status.running else "○ Stopped"
                text += f"Daemon Status: {daemon_status}\n"

                if status.running:
                    uptime_hours = status.uptime_seconds / 3600
                    text += f"Uptime: {uptime_hours:.1f} hours\n"
                    text += f"\nModes Active:\n"
                    text += f"  Batch Processing: {'✓' if status.batch_processor_active else '✗'}\n"
                    text += f"  Refinement: {'✓' if status.refiner_active else '✗'}\n"
                    text += f"  File Watching: {'✓' if status.file_watcher_active else '✗'}\n"

                    text += f"\nCurrent Session Queue:\n"
                    text += f"  Pending in Queue: {status.documents_pending}\n"

                    if status.last_refinement:
                        last_ref = datetime.fromtimestamp(status.last_refinement).strftime('%Y-%m-%d %H:%M:%S')
                        text += f"\nLast Refinement: {last_ref}\n"
            else:
                text += "Daemon Status: ○ Not Initialized\n"

            # Document statistics
            if self.knowledge_store:
                text += "\n" + "=" * 70 + "\n"
                text += "DOCUMENTS\n"
                text += "=" * 70 + "\n\n"

                doc_stats = self._get_document_stats()
                text += f"Total Ingested: {doc_stats.get('total', 0)}\n"
                text += f"Completed: {doc_stats.get('completed', 0)}\n"
                text += f"Processing: {doc_stats.get('processing', 0)}\n"
                text += f"Failed: {doc_stats.get('failed', 0)}\n"

            # Knowledge statistics
            if self.knowledge_store:
                text += "\n" + "=" * 70 + "\n"
                text += "KNOWLEDGE\n"
                text += "=" * 70 + "\n\n"

                summary = self.knowledge_store.get_knowledge_summary()
                text += f"Total Entries: {summary.get('total_entries', 0)}\n"

                # Show breakdown if available
                concept_count = summary.get('concept_count')
                entity_count = summary.get('entity_count')
                if concept_count is not None and entity_count is not None:
                    text += f"  Concepts: {concept_count}\n"
                    text += f"  Entities: {entity_count}\n"

                text += f"High Confidence: {summary.get('high_confidence_entries', 0)}\n"

                domain_dist = summary.get('domain_distribution', {})
                if domain_dist:
                    text += f"\nDomains:\n"
                    for domain, count in sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
                        text += f"  {domain}: {count}\n"

            text += "\n" + "=" * 70 + "\n"
            text += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        except Exception as e:
            text += f"\nError loading status: {e}\n"
            logger.error(f"Failed to refresh overview: {e}")

        # Update display
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, text)
        self.overview_text.config(state=tk.DISABLED)

    def _refresh_documents(self):
        """Refresh documents list."""
        # Clear existing items
        for item in self.doc_tree.get_children():
            self.doc_tree.delete(item)

        try:
            import sqlite3
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

                self.doc_tree.insert("", tk.END, values=(file_name, file_type, status, chunks, concepts, added_at))

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh documents: {e}")

    def _refresh_concepts(self):
        """Refresh concepts display."""
        text = "EXTRACTED CONCEPTS\n"
        text += "=" * 70 + "\n\n"

        try:
            if not self.knowledge_store:
                text += "Knowledge store not available\n"
            else:
                import sqlite3
                import json
                import pickle

                conn = sqlite3.connect(self.knowledge_store.storage_path)

                # Build query with domain filter (fetch both content columns for backward compatibility)
                domain_filter = self.concept_domain_var.get()
                if domain_filter == "all":
                    query = """
                        SELECT content_json, content_compressed, domain, confidence_level
                        FROM knowledge_entries
                        WHERE ((content_json IS NOT NULL AND content_json != ''
                                AND json_extract(content_json, '$.concept') IS NOT NULL)
                               OR content_compressed IS NOT NULL)
                        ORDER BY created_at DESC LIMIT 50
                    """
                    params = ()
                else:
                    query = """
                        SELECT content_json, content_compressed, domain, confidence_level
                        FROM knowledge_entries
                        WHERE ((content_json IS NOT NULL AND content_json != ''
                                AND json_extract(content_json, '$.concept') IS NOT NULL)
                               OR content_compressed IS NOT NULL)
                        AND domain = ?
                        ORDER BY created_at DESC LIMIT 50
                    """
                    params = (domain_filter,)

                cursor = conn.execute(query, params)

                # Diagnostic counters
                examined = 0
                concepts_found = 0
                entities_skipped = 0
                parse_errors = 0
                empty_entries = 0

                for row in cursor:
                    examined += 1
                    # Handle both compressed (legacy) and JSON content
                    content_json = row[0]
                    content_compressed = row[1]
                    domain = row[2]
                    confidence = row[3]

                    try:
                        if content_compressed:
                            # Legacy compressed entry
                            content = pickle.loads(content_compressed)
                        elif content_json and content_json.strip():
                            content = json.loads(content_json)
                        else:
                            empty_entries += 1
                            continue  # Skip invalid entries

                        concept_name = content.get('concept', 'Unknown')
                        if concept_name == 'Unknown':
                            # Check if it's an entity instead
                            if 'entity' in content:
                                entities_skipped += 1
                            continue

                        definition = content.get('definition', 'No definition')

                        text += f"• {concept_name}\n"
                        text += f"  Domain: {domain} | Confidence: {confidence}\n"
                        text += f"  {definition}\n\n"
                        concepts_found += 1
                    except Exception as parse_error:
                        parse_errors += 1
                        logger.warning(f"Failed to parse knowledge entry: {parse_error}")
                        continue

                if concepts_found == 0:
                    text += "No concepts found\n"
                    text += f"(Examined {examined} entries: {entities_skipped} entities, "
                    text += f"{parse_errors} parse errors, {empty_entries} empty)\n"
                else:
                    text += f"\nShowing {concepts_found} concepts ({examined} entries examined)\n"

                conn.close()

        except Exception as e:
            text += f"Error loading concepts: {e}\n"
            logger.error(f"Failed to refresh concepts: {e}")

        self.concepts_text.config(state=tk.NORMAL)
        self.concepts_text.delete(1.0, tk.END)
        self.concepts_text.insert(1.0, text)
        self.concepts_text.config(state=tk.DISABLED)

    def _search_concepts(self):
        """Search concepts by query."""
        query = self.concept_search_var.get().strip()
        if not query:
            self._refresh_concepts()
            return

        text = f"SEARCH RESULTS: '{query}'\n"
        text += "=" * 70 + "\n\n"

        try:
            if self.knowledge_retriever:
                result = self.knowledge_retriever.search(query, top_k=20)

                if result.results:
                    for i, search_result in enumerate(result.results, 1):
                        concept = search_result.content.get('concept', 'Unknown')
                        definition = search_result.content.get('definition', 'No definition')

                        text += f"{i}. {concept} (relevance: {search_result.relevance_score:.2f})\n"
                        text += f"   Domain: {search_result.domain}\n"
                        text += f"   {definition}\n\n"

                    text += f"\nFound {len(result.results)} results in {result.processing_time:.2f}s\n"
                    text += f"Method: {result.retrieval_method}\n"
                else:
                    text += "No results found\n"
            else:
                text += "Knowledge retriever not available\n"

        except Exception as e:
            text += f"Search failed: {e}\n"
            logger.error(f"Concept search failed: {e}")

        self.concepts_text.config(state=tk.NORMAL)
        self.concepts_text.delete(1.0, tk.END)
        self.concepts_text.insert(1.0, text)
        self.concepts_text.config(state=tk.DISABLED)

    def _refresh_activity(self):
        """Refresh activity log with timestamp notification."""
        # Don't clear the log - just append a refresh notification
        timestamp = datetime.now().strftime('%H:%M:%S')
        self._log_activity(f"Activity view refreshed")

    def _clear_activity_log(self):
        """Clear activity log."""
        self.activity_text.config(state=tk.NORMAL)
        self.activity_text.delete(1.0, tk.END)
        self.activity_text.config(state=tk.DISABLED)

    def _log_activity(self, message: str):
        """Add message to activity log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"

        self.activity_text.config(state=tk.NORMAL)
        self.activity_text.insert(tk.END, log_message)
        self.activity_text.see(tk.END)
        self.activity_text.config(state=tk.DISABLED)

    # === Helper Methods ===

    def _get_document_stats(self) -> dict:
        """Get document statistics."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            cursor = conn.execute("SELECT COUNT(*) FROM document_sources")
            total = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM document_sources WHERE ingestion_status = 'completed'")
            completed = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM document_sources WHERE ingestion_status = 'processing'")
            processing = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM document_sources WHERE ingestion_status = 'failed'")
            failed = cursor.fetchone()[0]

            conn.close()

            return {
                'total': total,
                'completed': completed,
                'processing': processing,
                'failed': failed
            }
        except:
            return {}

    def _view_document_details(self, event):
        """View document details on double-click."""
        selection = self.doc_tree.selection()
        if selection:
            item = self.doc_tree.item(selection[0])
            values = item['values']

            details = f"Document: {values[0]}\n"
            details += f"Type: {values[1]}\n"
            details += f"Status: {values[2]}\n"
            details += f"Chunks: {values[3]}\n"
            details += f"Concepts: {values[4]}\n"
            details += f"Date: {values[5]}\n"

            messagebox.showinfo("Document Details", details)

    def set_knowledge_components(self, daemon, retriever, knowledge_store):
        """Set knowledge brain components (called by main_app)."""
        self.knowledge_daemon = daemon
        self.knowledge_retriever = retriever
        self.knowledge_store = knowledge_store
        self._refresh_overview()
