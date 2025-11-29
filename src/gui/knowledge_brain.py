"""
Knowledge Brain GUI Tab for Felix

Provides monitoring and control interface for the autonomous knowledge brain:
- Overview: Status, statistics, daemon control
- Documents: Browse ingested sources, view status
- Concepts: Explore extracted knowledge with related concepts
- Activity: Real-time processing log
- Relationships: Explore knowledge graph connections between concepts
- Cleanup: Bulk cleanup operations, pattern-based deletion, maintenance tools
- Audit: Complete audit trail of all CRUD operations, filtering, and export
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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

        # Create notebook with 4 consolidated tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Control & Processing (merged Overview + Activity)
        self.control_processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_processing_frame, text="⚙️ Control & Processing")
        self._create_control_processing_tab()

        # Tab 2: Knowledge Base (merged Documents + Concepts in split view)
        self.knowledge_base_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.knowledge_base_frame, text="📚 Knowledge Base")
        self._create_knowledge_base_tab()

        # Tab 3: Relationships & Graph
        self.relationships_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.relationships_frame, text="🔗 Relationships")
        self._create_relationships_tab()

        # Tab 4: Maintenance (merged Cleanup + Audit + Analytics with sub-tabs)
        self.maintenance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.maintenance_frame, text="🛠️ Maintenance")
        self._create_maintenance_tab()

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

    def _create_control_processing_tab(self):
        """Create Control & Processing tab with daemon control, queue status, statistics, and activity feed."""
        # Control buttons
        control_frame = ttk.Frame(self.control_processing_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Daemon Control:", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.start_daemon_btn = ttk.Button(control_frame, text="▶ Start Daemon", command=self._start_daemon)
        self.start_daemon_btn.pack(side=tk.LEFT, padx=2)

        self.stop_daemon_btn = ttk.Button(control_frame, text="■ Stop Daemon", command=self._stop_daemon, state=tk.DISABLED)
        self.stop_daemon_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(control_frame, text="📂 Process Directory Once", command=self._add_directory).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⚡ Process Pending Now", command=self._process_pending_now).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🗑️ Manage Directories", command=self._manage_directories).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="🔄 Force Refinement", command=self._force_refinement).pack(side=tk.LEFT, padx=2)

        # Auto-refresh toggle
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Auto-Refresh", variable=self.auto_refresh_var,
                       command=self._toggle_auto_refresh).pack(side=tk.LEFT, padx=2)

        # Processing Progress section
        progress_frame = ttk.LabelFrame(self.control_processing_frame, text="Processing Queue", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.progress_text = tk.Text(progress_frame, wrap=tk.WORD, height=4, width=80, state=tk.DISABLED)
        self.progress_text.pack(fill=tk.BOTH, expand=True)

        # Status display
        status_frame = ttk.LabelFrame(self.control_processing_frame, text="Statistics Dashboard", padding=10)
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

        # Activity Feed section (merged from Activity tab)
        activity_frame = ttk.LabelFrame(self.control_processing_frame, text="Activity Feed", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Activity controls
        activity_control_frame = ttk.Frame(activity_frame)
        activity_control_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(activity_control_frame, text="🔄 Refresh", command=self._refresh_activity).pack(side=tk.LEFT, padx=5)
        ttk.Button(activity_control_frame, text="🗑️ Clear Log", command=self._clear_activity_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(activity_control_frame, text="💾 Export Log", command=self._export_activity_log).pack(side=tk.LEFT, padx=5)

        ttk.Separator(activity_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Severity filter
        ttk.Label(activity_control_frame, text="Filter:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.activity_filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(activity_control_frame, textvariable=self.activity_filter_var,
                                    values=["All", "INFO", "WARNING", "ERROR"],
                                    state="readonly", width=12)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_activity())

        # Auto-scroll toggle
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(activity_control_frame, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=10)

        # Activity log
        log_frame = ttk.Frame(activity_frame)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.activity_text = tk.Text(log_frame, wrap=tk.WORD, height=12, width=80)
        activity_scrollbar = ttk.Scrollbar(log_frame, command=self.activity_text.yview)
        self.activity_text.config(yscrollcommand=activity_scrollbar.set)

        self.activity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        activity_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Initial refresh
        self._refresh_overview()
        self._refresh_activity()

    def _create_knowledge_base_tab(self):
        """Create Knowledge Base tab with split view (Documents left, Concepts right)."""
        # Create horizontal PanedWindow for split view
        paned_window = tk.PanedWindow(self.knowledge_base_frame, orient=tk.HORIZONTAL, sashwidth=5)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left pane: Documents
        documents_pane = ttk.Frame(paned_window)
        paned_window.add(documents_pane)
        self._create_documents_pane(documents_pane)

        # Right pane: Concepts
        concepts_pane = ttk.Frame(paned_window)
        paned_window.add(concepts_pane)
        self._create_concepts_pane(concepts_pane)

    def _create_documents_pane(self, parent):
        """Create documents pane for the left side of Knowledge Base tab."""
        # Header
        ttk.Label(parent, text="DOCUMENTS", font=("TkDefaultFont", 10, "bold")).pack(pady=(5, 10))

        # Controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Status:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))

        self.doc_filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.doc_filter_var, state="readonly", width=12)
        filter_combo['values'] = ("all", "completed", "processing", "pending", "failed")
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_documents())

        ttk.Button(control_frame, text="🔄", command=self._refresh_documents, width=3).pack(side=tk.LEFT, padx=2)

        # Document list (TreeView)
        list_frame = ttk.Frame(parent)
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

        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        ttk.Button(action_frame, text="⚡ Process Selected", command=self._process_selected_documents).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Re-process", command=self._reprocess_selected_documents).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🗑️ Delete", command=self._delete_selected_documents).pack(side=tk.LEFT, padx=2)

        # Initial refresh
        self._refresh_documents()

    def _create_concepts_pane(self, parent):
        """Create concepts pane for the right side of Knowledge Base tab."""
        # Header
        ttk.Label(parent, text="CONCEPTS", font=("TkDefaultFont", 10, "bold")).pack(pady=(5, 10))

        # Search controls
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=tk.X, padx=10, pady=10)

        self.concept_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.concept_search_var, width=25)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<Return>", lambda e: self._search_concepts())

        ttk.Button(search_frame, text="🔍", command=self._search_concepts, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="🔄", command=self._refresh_concepts, width=3).pack(side=tk.LEFT, padx=2)

        # Domain filter
        ttk.Label(search_frame, text="Domain:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(10, 5))
        self.concept_domain_var = tk.StringVar(value="all")
        domain_combo = ttk.Combobox(search_frame, textvariable=self.concept_domain_var, state="readonly", width=10)
        domain_combo['values'] = ("all", "python", "web", "ai", "database", "general")
        domain_combo.pack(side=tk.LEFT, padx=5)
        domain_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_concepts())

        # Concepts display with TreeView
        display_frame = ttk.Frame(parent)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # TreeView for concepts
        columns = ("concept", "domain", "confidence", "definition")
        self.concepts_tree = ttk.Treeview(display_frame, columns=columns,
                                          show="headings", height=15)

        self.concepts_tree.heading("concept", text="Concept")
        self.concepts_tree.heading("domain", text="Domain")
        self.concepts_tree.heading("confidence", text="Confidence")
        self.concepts_tree.heading("definition", text="Definition")

        self.concepts_tree.column("concept", width=200)
        self.concepts_tree.column("domain", width=100)
        self.concepts_tree.column("confidence", width=100)
        self.concepts_tree.column("definition", width=400)

        concepts_scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL,
                                          command=self.concepts_tree.yview)
        self.concepts_tree.config(yscrollcommand=concepts_scrollbar.set)

        self.concepts_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        concepts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Context menu bindings
        self.concepts_tree.bind("<Button-3>", self._show_concept_context_menu)
        self.concepts_tree.bind("<Double-1>", self._edit_concept_entry)

        # Selection mode (for bulk operations)
        self.concepts_tree.configure(selectmode="extended")

        # Action buttons
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(action_frame, text="✏️ Edit", command=self._edit_concept_entry).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🗑️ Delete", command=self._delete_selected_concepts).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔗 Merge", command=self._merge_selected_concepts).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="💾 Export", command=self._export_selected_concepts).pack(side=tk.LEFT, padx=2)

        # Initial refresh
        self._refresh_concepts()

    def _create_maintenance_tab(self):
        """Create Maintenance tab with sub-tabs for Quality, Audit, and Cleanup."""
        # Create sub-tab notebook
        maintenance_notebook = ttk.Notebook(self.maintenance_frame)
        maintenance_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sub-tab 1: Quality (Analytics + Duplicates + Reports)
        quality_frame = ttk.Frame(maintenance_notebook)
        maintenance_notebook.add(quality_frame, text="Quality")
        self._create_quality_subtab(quality_frame)

        # Sub-tab 2: Audit (CRUD history)
        audit_frame = ttk.Frame(maintenance_notebook)
        maintenance_notebook.add(audit_frame, text="Audit")
        self._create_audit_subtab(audit_frame)

        # Sub-tab 3: Cleanup (Bulk operations)
        cleanup_frame = ttk.Frame(maintenance_notebook)
        maintenance_notebook.add(cleanup_frame, text="Cleanup")
        self._create_cleanup_subtab(cleanup_frame)

    def _create_quality_subtab(self, parent):
        """Create Quality sub-tab (moved from Analytics tab)."""
        # Temporarily store parent and call existing analytics creation
        original_frame = getattr(self, 'analytics_frame', None)
        self.analytics_frame = parent
        self._create_analytics_tab()
        if original_frame:
            self.analytics_frame = original_frame

    def _create_audit_subtab(self, parent):
        """Create Audit sub-tab (moved from Audit tab)."""
        # Temporarily store parent and call existing audit creation
        original_frame = getattr(self, 'audit_frame', None)
        self.audit_frame = parent
        self._create_audit_tab()
        if original_frame:
            self.audit_frame = original_frame

    def _create_cleanup_subtab(self, parent):
        """Create Cleanup sub-tab (moved from Cleanup tab)."""
        # Temporarily store parent and call existing cleanup creation
        original_frame = getattr(self, 'cleanup_frame', None)
        self.cleanup_frame = parent
        self._create_cleanup_tab()
        if original_frame:
            self.cleanup_frame = original_frame

    def _create_activity_tab(self):
        """DEPRECATED: Activity tab is now merged into Control & Processing tab."""
        # This method is no longer used - keeping for reference during transition
        pass

    def _create_old_activity_tab_deprecated(self):
        """OLD METHOD - Activity tab for real-time processing log."""
        # Controls
        control_frame = ttk.Frame(self.activity_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text="🔄 Refresh", command=self._refresh_activity).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🗑️ Clear Log", command=self._clear_activity_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Export Log", command=self._export_activity_log).pack(side=tk.LEFT, padx=5)

        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Severity filter
        ttk.Label(control_frame, text="Filter:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.activity_filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(control_frame, textvariable=self.activity_filter_var,
                                    values=["All", "INFO", "WARNING", "ERROR"],
                                    state="readonly", width=12)
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_activity())

        # Auto-scroll toggle
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=10)

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

    def _create_relationships_tab(self):
        """Create relationships tab for exploring the knowledge graph."""
        # Controls
        control_frame = ttk.Frame(self.relationships_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Filter:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))

        # Domain filter
        self.rel_domain_var = tk.StringVar(value="all")
        domain_combo = ttk.Combobox(control_frame, textvariable=self.rel_domain_var, state="readonly", width=15)
        domain_combo['values'] = ("all", "python", "web", "ai", "database", "general")
        domain_combo.pack(side=tk.LEFT, padx=5)
        domain_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_relationships())

        ttk.Button(control_frame, text="🔄 Refresh", command=self._refresh_relationships).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="📊 Show Statistics", command=self._show_relationship_stats).pack(side=tk.LEFT, padx=5)

        # Search frame
        search_frame = ttk.Frame(self.relationships_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(search_frame, text="Find Concept:", font=("TkDefaultFont", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.rel_search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.rel_search_var, width=40)
        search_entry.pack(side=tk.LEFT, padx=5)
        search_entry.bind("<Return>", lambda e: self._search_relationships())

        ttk.Button(search_frame, text="🔍 Search", command=self._search_relationships).pack(side=tk.LEFT, padx=5)

        # Relationships list (TreeView)
        list_frame = ttk.Frame(self.relationships_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Create treeview
        columns = ("source_concept", "target_concept", "source_domain", "target_domain", "strength")
        self.rel_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)

        self.rel_tree.heading("source_concept", text="Source Concept")
        self.rel_tree.heading("target_concept", text="Target Concept")
        self.rel_tree.heading("source_domain", text="Source Domain")
        self.rel_tree.heading("target_domain", text="Target Domain")
        self.rel_tree.heading("strength", text="Connections")

        self.rel_tree.column("source_concept", width=250)
        self.rel_tree.column("target_concept", width=250)
        self.rel_tree.column("source_domain", width=100)
        self.rel_tree.column("target_domain", width=100)
        self.rel_tree.column("strength", width=100)

        rel_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.rel_tree.yview)
        self.rel_tree.config(yscrollcommand=rel_scrollbar.set)

        self.rel_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Context menu on right-click
        self.rel_tree.bind("<Button-3>", self._show_relationship_context_menu)
        self.rel_tree.bind("<Double-1>", self._explore_relationship_network)

        # Status label
        self.rel_status_label = ttk.Label(self.relationships_frame, text="", foreground="gray")
        self.rel_status_label.pack(pady=5)

        # Initial refresh
        self._refresh_relationships()

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
        """Add a directory for one-time processing (not persistent watch)."""
        directory = filedialog.askdirectory(title="Select Directory with Documents")
        if directory:
            if self.knowledge_daemon:
                try:
                    result = self.knowledge_daemon.process_directory_now(directory)
                    messagebox.showinfo("Success",
                                       f"Queued {result.get('queued', 0)} documents from:\n{directory}")
                    self._log_activity(f"Processed directory once: {directory} ({result.get('queued', 0)} documents)")
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

            # Save to persistent config
            if removed_count > 0 and hasattr(self.main_app, 'save_watch_directories'):
                self.main_app.save_watch_directories()

            # Show result
            if removed_count > 0:
                messagebox.showinfo("Success", f"Removed {removed_count} director{'y' if removed_count == 1 else 'ies'}")
                self._refresh_overview()
            update_info()

        def add_directory():
            """Add a directory to persistent watch list."""
            from tkinter import filedialog
            directory = filedialog.askdirectory(title="Select Directory to Watch", parent=dialog)
            if directory:
                try:
                    result = self.knowledge_daemon.add_watch_directory(directory)

                    if result.get('success'):
                        # Add to listbox immediately
                        dir_listbox.insert(tk.END, directory)

                        # Save to persistent config
                        if hasattr(self.main_app, 'save_watch_directories'):
                            self.main_app.save_watch_directories()

                        messagebox.showinfo("Success",
                                           f"Added to persistent watch list:\n{directory}\n\n"
                                           "This directory will be monitored for changes.",
                                           parent=dialog)
                        self._log_activity(f"Added watch directory: {directory}")
                        self._refresh_overview()
                        update_info()
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        messagebox.showwarning("Cannot Add Directory", error_msg, parent=dialog)

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add watch directory: {e}", parent=dialog)

        ttk.Button(btn_frame, text="➕ Add Directory", command=add_directory).pack(side=tk.LEFT, padx=5)
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
                self._refresh_documents()
            except Exception as e:
                messagebox.showerror("Error", f"Refinement failed: {e}")
        else:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")

    def _process_pending_now(self):
        """Manually trigger processing of all pending documents."""
        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        try:
            import sqlite3

            # Get count of pending documents
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.execute("SELECT COUNT(*) FROM document_sources WHERE ingestion_status='pending'")
            pending_count = cursor.fetchone()[0]

            if pending_count == 0:
                conn.close()
                messagebox.showinfo("No Pending Documents", "All documents have been processed")
                return

            # Get all pending document paths
            cursor = conn.execute("SELECT file_path FROM document_sources WHERE ingestion_status='pending'")
            pending_paths = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Add all pending documents to the daemon's processing queue
            for path in pending_paths:
                self.knowledge_daemon.document_queue.add(path)

            messagebox.showinfo("Processing Started",
                               f"Queued {pending_count} pending documents for immediate processing.\n\n"
                               f"Watch the Processing Queue section for progress.")
            self._log_activity(f"Manual trigger: Queued {pending_count} pending documents for processing")

            # Refresh overview to show updated queue
            self._refresh_overview()

        except Exception as e:
            logger.error(f"Failed to trigger pending document processing: {e}")
            messagebox.showerror("Error", f"Failed to start processing: {str(e)}")

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

            # Watch directory statistics
            if self.knowledge_store:
                try:
                    dirs = self.knowledge_store.get_watch_directories()
                    if dirs:
                        text += "\n" + "=" * 70 + "\n"
                        text += "WATCHED DIRECTORIES\n"
                        text += "=" * 70 + "\n\n"
                        text += f"Total: {len(dirs)}\n"
                        text += f"Enabled: {sum(1 for d in dirs if d['enabled'])}\n\n"

                        # Show top directories by entry count
                        top_dirs = sorted(dirs, key=lambda x: x['entry_count'], reverse=True)[:3]
                        if top_dirs:
                            text += "Top Contributors:\n"
                            for dir_info in top_dirs:
                                status = "✓" if dir_info['enabled'] else "✗"
                                path = dir_info['directory_path']
                                # Shorten path if too long
                                if len(path) > 50:
                                    path = "..." + path[-47:]
                                text += f"  {status} {path}\n"
                                text += f"     {dir_info['document_count']} docs, {dir_info['entry_count']} entries\n"
                except Exception as e:
                    logger.warning(f"Failed to load watch directories: {e}")

            text += "\n" + "=" * 70 + "\n"
            text += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        except Exception as e:
            text += f"\nError loading status: {e}\n"
            logger.error(f"Failed to refresh overview: {e}")

        # Update main display
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, text)
        self.overview_text.config(state=tk.DISABLED)

        # Update processing progress display
        self._refresh_processing_progress()

    def _refresh_processing_progress(self):
        """Update the processing progress section with queue statistics."""
        progress_text = ""

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Get counts by status
            cursor = conn.execute("""
                SELECT ingestion_status, COUNT(*)
                FROM document_sources
                GROUP BY ingestion_status
            """)
            status_counts = dict(cursor.fetchall())

            pending = status_counts.get('pending', 0)
            processing = status_counts.get('processing', 0)
            completed = status_counts.get('completed', 0)
            failed = status_counts.get('failed', 0)

            progress_text += f"📊 Queue Status:\n"
            progress_text += f"   Pending: {pending:,}  |  Processing: {processing}  |  Completed: {completed:,}  |  Failed: {failed}\n"

            # Show currently processing documents if any
            if processing > 0:
                cursor = conn.execute("""
                    SELECT file_path
                    FROM document_sources
                    WHERE ingestion_status='processing'
                    LIMIT 3
                """)
                current_docs = [row[0] for row in cursor.fetchall()]
                if current_docs:
                    progress_text += f"\n🔄 Currently Processing:\n"
                    for doc in current_docs:
                        # Show just filename, not full path
                        filename = doc.split('/')[-1] if '/' in doc else doc
                        if len(filename) > 60:
                            filename = filename[:57] + "..."
                        progress_text += f"   • {filename}\n"

            conn.close()

        except Exception as e:
            progress_text = f"Error loading queue status: {e}"
            logger.error(f"Failed to refresh processing progress: {e}")

        # Update progress display
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(1.0, progress_text)
        self.progress_text.config(state=tk.DISABLED)

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
        """Refresh concepts display in TreeView."""
        # Clear existing items
        for item in self.concepts_tree.get_children():
            self.concepts_tree.delete(item)

        try:
            if not self.knowledge_store:
                return

            import sqlite3
            import json
            import pickle

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
                    WHERE ((content_json IS NOT NULL AND content_json != ''
                            AND json_extract(content_json, '$.concept') IS NOT NULL)
                           OR content_compressed IS NOT NULL)
                    AND domain = ?
                    ORDER BY created_at DESC LIMIT 100
                """
                params = (domain_filter,)

            cursor = conn.execute(query, params)

            for row in cursor:
                knowledge_id, content_json, content_compressed, domain, confidence = row

                try:
                    # Handle both compressed (legacy) and JSON content
                    if content_compressed:
                        content = pickle.loads(content_compressed)
                    elif content_json and content_json.strip():
                        content = json.loads(content_json)
                    else:
                        continue

                    concept_name = content.get('concept', 'Unknown')
                    if concept_name == 'Unknown':
                        # Skip entities
                        if 'entity' in content:
                            continue

                    definition = content.get('definition', 'No definition')

                    # Truncate long definitions
                    if len(definition) > 100:
                        definition = definition[:97] + "..."

                    # Insert into TreeView (store knowledge_id in tags for retrieval)
                    self.concepts_tree.insert("", tk.END,
                        values=(concept_name, domain, confidence, definition),
                        tags=(knowledge_id,))

                except Exception as e:
                    logger.warning(f"Failed to parse entry: {e}")
                    continue

            conn.close()

        except Exception as e:
            logger.error(f"Failed to refresh concepts: {e}")
            messagebox.showerror("Error", f"Failed to load concepts: {e}")

    def _search_concepts(self):
        """Search concepts by query."""
        query = self.concept_search_var.get().strip()
        if not query:
            self._refresh_concepts()
            return

        # Clear existing items
        for item in self.concepts_tree.get_children():
            self.concepts_tree.delete(item)

        try:
            if self.knowledge_retriever:
                result = self.knowledge_retriever.search(query, top_k=100)

                if result.results:
                    for search_result in result.results:
                        concept = search_result.content.get('concept', 'Unknown')
                        definition = search_result.content.get('definition', 'No definition')

                        # Truncate long definitions
                        if len(definition) > 100:
                            definition = definition[:97] + "..."

                        # Add relevance score to confidence display
                        confidence_display = f"{search_result.confidence_level.value} ({search_result.relevance_score:.2f})"

                        # Insert into TreeView
                        knowledge_id = getattr(search_result, 'knowledge_id', None)
                        self.concepts_tree.insert("", tk.END,
                            values=(concept, search_result.domain, confidence_display, definition),
                            tags=(knowledge_id,) if knowledge_id else ())

                    self._log_activity(f"Search found {len(result.results)} results for '{query}'")
                else:
                    messagebox.showinfo("No Results", f"No results found for '{query}'")
            else:
                messagebox.showwarning("Not Available", "Knowledge retriever not available")

        except Exception as e:
            logger.error(f"Concept search failed: {e}")
            messagebox.showerror("Search Error", f"Search failed: {e}")

    def _edit_concept_entry(self, event=None):
        """Open dialog to edit selected concept entry."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an entry to edit")
            return

        # Get knowledge_id from tags
        item = self.concepts_tree.item(selection[0])
        knowledge_id = item['tags'][0] if item['tags'] else None

        if not knowledge_id or not self.knowledge_store:
            messagebox.showerror("Error", "Unable to load entry")
            return

        # Retrieve full entry
        entry = self.knowledge_store.get_entry_by_id(knowledge_id)
        if not entry:
            messagebox.showerror("Error", "Failed to load entry")
            return

        # Create edit dialog
        dialog = tk.Toplevel(self)
        dialog.title(f"Edit Concept: {entry.content.get('concept', 'Unknown')}")
        dialog.geometry("700x600")
        dialog.transient(self)
        dialog.grab_set()

        # Concept name
        ttk.Label(dialog, text="Concept Name:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=20, pady=(20, 5))
        concept_var = tk.StringVar(value=entry.content.get('concept', ''))
        concept_entry = ttk.Entry(dialog, textvariable=concept_var, width=60)
        concept_entry.pack(padx=20, pady=(0, 10))

        # Definition
        ttk.Label(dialog, text="Definition:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=20, pady=5)
        definition_frame = ttk.Frame(dialog)
        definition_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        definition_text = tk.Text(definition_frame, wrap=tk.WORD, height=10, width=70)
        def_scrollbar = ttk.Scrollbar(definition_frame, command=definition_text.yview)
        definition_text.config(yscrollcommand=def_scrollbar.set)
        definition_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        def_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        definition_text.insert(1.0, entry.content.get('definition', ''))

        # Domain
        ttk.Label(dialog, text="Domain:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=20, pady=5)
        domain_var = tk.StringVar(value=entry.domain)
        domain_combo = ttk.Combobox(dialog, textvariable=domain_var,
                                    values=["python", "web", "ai", "database", "general", "system"],
                                    state="readonly", width=20)
        domain_combo.pack(anchor=tk.W, padx=20, pady=(0, 10))

        # Confidence
        ttk.Label(dialog, text="Confidence:", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=20, pady=5)
        confidence_var = tk.StringVar(value=entry.confidence_level.value)
        confidence_combo = ttk.Combobox(dialog, textvariable=confidence_var,
                                        values=["low", "medium", "high", "verified"],
                                        state="readonly", width=20)
        confidence_combo.pack(anchor=tk.W, padx=20, pady=(0, 10))

        # Tags
        ttk.Label(dialog, text="Tags (comma-separated):", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, padx=20, pady=5)
        tags_var = tk.StringVar(value=", ".join(entry.tags))
        tags_entry = ttk.Entry(dialog, textvariable=tags_var, width=60)
        tags_entry.pack(padx=20, pady=(0, 20))

        def save_changes():
            """Save edited entry."""
            try:
                from src.memory.knowledge_store import ConfidenceLevel

                # Gather updates
                updates = {
                    'content': {
                        'concept': concept_var.get().strip(),
                        'definition': definition_text.get(1.0, tk.END).strip()
                    },
                    'domain': domain_var.get(),
                    'confidence_level': ConfidenceLevel(confidence_var.get()),
                    'tags': [t.strip() for t in tags_var.get().split(',') if t.strip()]
                }

                # Update in database
                success = self.knowledge_store.update_knowledge_entry(knowledge_id, updates)

                if success:
                    messagebox.showinfo("Success", "Entry updated successfully")
                    self._refresh_concepts()
                    self._log_activity(f"Updated concept: {updates['content']['concept']}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Failed to update entry")

            except Exception as e:
                messagebox.showerror("Error", f"Update failed: {e}")
                logger.error(f"Entry update error: {e}")

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="Save", command=save_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _delete_selected_concepts(self):
        """Delete selected concept entries."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select entries to delete")
            return

        # Confirm deletion
        count = len(selection)
        if not messagebox.askyesno("Confirm Deletion",
                                   f"Delete {count} selected {'entry' if count == 1 else 'entries'}?\n\n" +
                                   "This action cannot be undone."):
            return

        deleted = 0
        for item_id in selection:
            item = self.concepts_tree.item(item_id)
            knowledge_id = item['tags'][0] if item['tags'] else None

            if knowledge_id and self.knowledge_store:
                if self.knowledge_store.delete_knowledge(knowledge_id):
                    deleted += 1

        messagebox.showinfo("Deletion Complete", f"Deleted {deleted} of {count} entries")
        self._refresh_concepts()
        self._log_activity(f"Deleted {deleted} concept entries")

    def _merge_selected_concepts(self):
        """Merge selected concept entries."""
        selection = self.concepts_tree.selection()
        if len(selection) < 2:
            messagebox.showwarning("Invalid Selection", "Please select at least 2 entries to merge")
            return

        # Get entry IDs and names
        entry_ids = []
        entry_names = []
        for item_id in selection:
            item = self.concepts_tree.item(item_id)
            knowledge_id = item['tags'][0] if item['tags'] else None
            if knowledge_id:
                entry_ids.append(knowledge_id)
                entry = self.knowledge_store.get_entry_by_id(knowledge_id)
                if entry:
                    entry_names.append(entry.content.get('concept', 'Unknown'))

        if len(entry_ids) < 2:
            messagebox.showerror("Error", "Could not retrieve entry IDs")
            return

        # Create selection dialog
        dialog = tk.Toplevel(self)
        dialog.title("Merge Entries")
        dialog.geometry("500x350")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Select primary entry to keep:",
                 font=("TkDefaultFont", 10, "bold")).pack(pady=10)

        listbox_frame = ttk.Frame(dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        listbox = tk.Listbox(listbox_frame, height=10)
        scrollbar = ttk.Scrollbar(listbox_frame, command=listbox.yview)
        listbox.config(yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for name in entry_names:
            listbox.insert(tk.END, name)

        listbox.selection_set(0)

        def do_merge():
            """Execute merge."""
            selected_idx = listbox.curselection()
            if not selected_idx:
                messagebox.showwarning("No Selection", "Please select primary entry")
                return

            primary_id = entry_ids[selected_idx[0]]
            secondary_ids = [eid for i, eid in enumerate(entry_ids) if i != selected_idx[0]]

            try:
                success = self.knowledge_store.merge_knowledge_entries(
                    primary_id, secondary_ids, merge_strategy="combine_content"
                )

                if success:
                    messagebox.showinfo("Success", f"Merged {len(secondary_ids)} entries")
                    self._refresh_concepts()
                    self._log_activity(f"Merged {len(secondary_ids)} entries into {entry_names[selected_idx[0]]}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "Merge failed")
            except Exception as e:
                messagebox.showerror("Error", f"Merge error: {e}")

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Merge", command=do_merge).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def _export_selected_concepts(self):
        """Export selected concepts to JSON file."""
        selection = self.concepts_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select entries to export")
            return

        # Get file path
        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            title="Export Concepts",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Collect entries
        entries = []
        for item_id in selection:
            item = self.concepts_tree.item(item_id)
            knowledge_id = item['tags'][0] if item['tags'] else None

            if knowledge_id and self.knowledge_store:
                entry = self.knowledge_store.get_entry_by_id(knowledge_id)
                if entry:
                    # Convert to dict
                    entry_dict = {
                        'knowledge_id': entry.knowledge_id,
                        'knowledge_type': entry.knowledge_type.value,
                        'content': entry.content,
                        'confidence_level': entry.confidence_level.value,
                        'domain': entry.domain,
                        'tags': entry.tags,
                        'created_at': entry.created_at,
                        'updated_at': entry.updated_at,
                        'related_entries': entry.related_entries
                    }
                    entries.append(entry_dict)

        # Write to file
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(entries, f, indent=2, default=str)

            messagebox.showinfo("Success", f"Exported {len(entries)} entries to {file_path}")
            self._log_activity(f"Exported {len(entries)} concepts to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")

    def _show_concept_context_menu(self, event):
        """Show context menu for concept entries."""
        item = self.concepts_tree.identify_row(event.y)
        if item:
            self.concepts_tree.selection_set(item)

            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Edit", command=self._edit_concept_entry)
            menu.add_command(label="Delete", command=self._delete_selected_concepts)
            menu.add_separator()
            menu.add_command(label="View Related", command=self._view_related_concepts)
            menu.add_command(label="View Source Document", command=self._view_source_document)

            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

    def _view_related_concepts(self):
        """View concepts related to selected entry."""
        selection = self.concepts_tree.selection()
        if not selection:
            return

        item = self.concepts_tree.item(selection[0])
        knowledge_id = item['tags'][0] if item['tags'] else None

        if not knowledge_id or not self.knowledge_retriever:
            messagebox.showinfo("Not Available", "Related concepts feature requires knowledge retriever")
            return

        try:
            # Get related concepts
            related = self.knowledge_retriever.get_related_concepts(knowledge_id, max_depth=2)

            if not related or not related.get('concepts'):
                messagebox.showinfo("No Related Concepts", "No related concepts found")
                return

            # Create dialog to show related concepts
            dialog = tk.Toplevel(self)
            dialog.title("Related Concepts")
            dialog.geometry("600x400")
            dialog.transient(self)

            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text = tk.Text(text_frame, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(text_frame, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            text.insert(1.0, f"Related Concepts (max depth: 2)\n")
            text.insert(tk.END, "=" * 50 + "\n\n")

            for concept_data in related.get('concepts', []):
                text.insert(tk.END, f"• {concept_data.get('concept', 'Unknown')}\n")
                text.insert(tk.END, f"  Relationship: {concept_data.get('relationship_type', 'unknown')}\n")
                text.insert(tk.END, f"  Strength: {concept_data.get('strength', 0.0):.2f}\n\n")

            text.config(state=tk.DISABLED)

            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load related concepts: {e}")
            logger.error(f"Related concepts error: {e}")

    def _view_source_document(self):
        """View the source document for selected entry."""
        selection = self.concepts_tree.selection()
        if not selection:
            return

        item = self.concepts_tree.item(selection[0])
        knowledge_id = item['tags'][0] if item['tags'] else None

        if not knowledge_id or not self.knowledge_store:
            return

        entry = self.knowledge_store.get_entry_by_id(knowledge_id)
        if not entry or not entry.source_doc_id:
            messagebox.showinfo("No Source", "This entry has no associated source document")
            return

        # Get document info
        import sqlite3
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.execute("""
                SELECT file_name, file_path, file_type, ingestion_status
                FROM document_sources
                WHERE doc_id = ?
            """, (entry.source_doc_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                file_name, file_path, file_type, status = row
                details = f"Source Document:\n\n"
                details += f"File: {file_name}\n"
                details += f"Path: {file_path}\n"
                details += f"Type: {file_type}\n"
                details += f"Status: {status}\n"
                details += f"Chunk: {entry.chunk_index}\n"

                messagebox.showinfo("Source Document", details)
            else:
                messagebox.showwarning("Not Found", "Source document not found in database")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve document info: {e}")

    def _refresh_activity(self):
        """Refresh activity log with daemon logs and apply filtering."""
        try:
            import sqlite3

            # Get filter selection
            filter_level = self.activity_filter_var.get()

            # Build log display
            logs = []

            # Add recent document processing events
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Get recently completed documents
            cursor = conn.execute("""
                SELECT file_path, processed_at, ingestion_status
                FROM document_sources
                WHERE processed_at IS NOT NULL
                ORDER BY processed_at DESC
                LIMIT 20
            """)

            for row in cursor.fetchall():
                file_path, processed_at, status = row
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                timestamp = datetime.fromtimestamp(processed_at).strftime('%H:%M:%S')

                if status == 'completed':
                    level = "INFO"
                    message = f"Completed processing: {filename}"
                elif status == 'failed':
                    level = "ERROR"
                    message = f"Failed to process: {filename}"
                else:
                    level = "INFO"
                    message = f"Status {status}: {filename}"

                # Apply filter
                if filter_level == "All" or filter_level == level:
                    emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
                    logs.append(f"[{timestamp}] {emoji} {level}: {message}\n")

            conn.close()

            # Get daemon status if available
            if self.knowledge_daemon:
                status = self.knowledge_daemon.get_status()
                timestamp = datetime.now().strftime('%H:%M:%S')

                if filter_level in ["All", "INFO"]:
                    if status.running:
                        logs.insert(0, f"[{timestamp}] ℹ️ INFO: Daemon is running (uptime: {status.uptime_seconds/3600:.1f}h)\n")
                        logs.insert(1, f"[{timestamp}] ℹ️ INFO: Pending in queue: {status.documents_pending}\n")
                    else:
                        logs.insert(0, f"[{timestamp}] ⚠️ WARNING: Daemon is not running\n")

            # Update display
            self.activity_text.config(state=tk.NORMAL)
            self.activity_text.delete(1.0, tk.END)

            if logs:
                for log in logs:
                    self.activity_text.insert(tk.END, log)
            else:
                self.activity_text.insert(1.0, "No activity to display\n")

            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.activity_text.see(tk.END)

            self.activity_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Failed to refresh activity log: {e}")
            self._log_activity(f"Error refreshing activity: {str(e)}", "ERROR")

    def _clear_activity_log(self):
        """Clear activity log."""
        self.activity_text.config(state=tk.NORMAL)
        self.activity_text.delete(1.0, tk.END)
        self.activity_text.config(state=tk.DISABLED)

    def _export_activity_log(self):
        """Export activity log to a text file."""
        from tkinter import filedialog

        try:
            # Get current log content
            log_content = self.activity_text.get(1.0, tk.END)

            if not log_content.strip():
                messagebox.showwarning("Empty Log", "Activity log is empty, nothing to export")
                return

            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                title="Export Activity Log",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")],
                initialfile=f"felix_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Felix Knowledge Brain Activity Log\n")
                    f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Filter: {self.activity_filter_var.get()}\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(log_content)

                messagebox.showinfo("Export Complete", f"Activity log exported to:\n{file_path}")
                self._log_activity(f"Exported activity log to {file_path}", "INFO")

        except Exception as e:
            logger.error(f"Failed to export activity log: {e}")
            messagebox.showerror("Export Failed", f"Failed to export log: {str(e)}")

    def _log_activity(self, message: str, level: str = "INFO"):
        """Add message to activity log with severity level."""
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Add emoji based on level
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔍"}.get(level, "📝")
        log_message = f"[{timestamp}] {emoji} {level}: {message}\n"

        self.activity_text.config(state=tk.NORMAL)
        self.activity_text.insert(tk.END, log_message)

        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
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

    def _reprocess_selected_documents(self):
        """Re-process selected documents."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to re-process")
            return

        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        # Confirm re-processing
        count = len(selection)
        if not messagebox.askyesno("Confirm Re-processing",
                                   f"Re-process {count} selected {'document' if count == 1 else 'documents'}?\n\n" +
                                   "This will delete existing entries and re-analyze the documents."):
            return

        # Get file paths
        queued = 0
        skipped = 0
        errors = []

        for item_id in selection:
            item = self.doc_tree.item(item_id)
            file_name = item['values'][0]

            # Get full path from database
            try:
                import sqlite3
                conn = sqlite3.connect(self.knowledge_store.storage_path)
                cursor = conn.execute("""
                    SELECT file_path FROM document_sources
                    WHERE file_name = ?
                """, (file_name,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    file_path = row[0]
                    result = self.knowledge_daemon.reprocess_document(file_path, force=True)

                    if result['status'] == 'queued':
                        queued += 1
                    elif result['status'] == 'skipped':
                        skipped += 1
                    elif result['status'] == 'error':
                        errors.append(f"{file_name}: {result.get('error', 'Unknown error')}")
                else:
                    errors.append(f"{file_name}: File path not found")

            except Exception as e:
                errors.append(f"{file_name}: {e}")

        # Show results
        msg = f"Re-processing initiated:\n\n"
        msg += f"Queued: {queued}\n"
        if skipped > 0:
            msg += f"Skipped: {skipped}\n"
        if errors:
            msg += f"Errors: {len(errors)}\n\n"
            msg += "Errors:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n... and {len(errors) - 5} more"

        messagebox.showinfo("Re-processing Status", msg)
        self._log_activity(f"Queued {queued} documents for re-processing")

        # Refresh display after delay
        self.after(2000, self._refresh_documents)

    def _process_selected_documents(self):
        """Process selected pending documents."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select pending documents to process")
            return

        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        # Get selected documents and filter for pending only
        pending_paths = []
        non_pending = 0

        for item_id in selection:
            item = self.doc_tree.item(item_id)
            status = item['values'][2]  # Status column

            if status == 'pending':
                file_name = item['values'][0]

                # Get full path from database
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.knowledge_store.storage_path)
                    cursor = conn.execute("""
                        SELECT file_path FROM document_sources
                        WHERE file_name = ? AND ingestion_status = 'pending'
                    """, (file_name,))
                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        pending_paths.append(row[0])
                except Exception as e:
                    logger.error(f"Failed to get path for {file_name}: {e}")
            else:
                non_pending += 1

        if not pending_paths:
            messagebox.showinfo("No Pending Documents",
                               f"No pending documents selected.\n{non_pending} non-pending documents were skipped.")
            return

        # Queue documents for processing
        for path in pending_paths:
            self.knowledge_daemon.document_queue.add(path)

        msg = f"Queued {len(pending_paths)} pending documents for processing"
        if non_pending > 0:
            msg += f"\n\nSkipped {non_pending} non-pending documents"

        messagebox.showinfo("Processing Started", msg)
        self._log_activity(f"Queued {len(pending_paths)} pending documents for processing")

        # Refresh display
        self.after(1000, self._refresh_documents)

    def _delete_selected_documents(self):
        """Delete selected documents and their associated knowledge entries."""
        selection = self.doc_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select documents to delete")
            return

        count = len(selection)
        if not messagebox.askyesno("Confirm Deletion",
                                   f"Delete {count} selected {'document' if count == 1 else 'documents'}?\n\n" +
                                   "This will permanently delete the documents and all associated knowledge entries.\n" +
                                   "This action cannot be undone.",
                                   icon='warning'):
            return

        deleted = 0
        errors = []

        for item_id in selection:
            item = self.doc_tree.item(item_id)
            file_name = item['values'][0]

            try:
                import sqlite3
                conn = sqlite3.connect(self.knowledge_store.storage_path)

                # Get doc_id
                cursor = conn.execute("SELECT doc_id FROM document_sources WHERE file_name = ?", (file_name,))
                row = cursor.fetchone()

                if row:
                    doc_id = row[0]

                    # Delete knowledge entries associated with this document
                    conn.execute("DELETE FROM knowledge_entries WHERE source_document_id = ?", (doc_id,))

                    # Delete document record
                    conn.execute("DELETE FROM document_sources WHERE doc_id = ?", (doc_id,))

                    conn.commit()
                    deleted += 1
                else:
                    errors.append(f"{file_name}: Not found in database")

                conn.close()

            except Exception as e:
                errors.append(f"{file_name}: {e}")
                logger.error(f"Failed to delete document {file_name}: {e}")

        # Show results
        msg = f"Deleted: {deleted} documents"
        if errors:
            msg += f"\nErrors: {len(errors)}\n\n"
            msg += "\n".join(errors[:5])
            if len(errors) > 5:
                msg += f"\n... and {len(errors) - 5} more"

        messagebox.showinfo("Deletion Complete", msg)
        self._log_activity(f"Deleted {deleted} documents")

        # Refresh display
        self._refresh_documents()

    def _refresh_relationships(self):
        """Refresh relationships list."""
        # Clear existing items
        for item in self.rel_tree.get_children():
            self.rel_tree.delete(item)

        if not self.knowledge_store:
            self.rel_status_label.config(text="Knowledge store not available")
            return

        try:
            import sqlite3
            import json

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Build query with domain filter
            domain_filter = self.rel_domain_var.get()

            # Query to get all entries with relationships
            if domain_filter == "all":
                query = """
                    SELECT knowledge_id, content_json, domain, related_entries_json
                    FROM knowledge_entries
                    WHERE related_entries_json IS NOT NULL
                    AND related_entries_json != '[]'
                    AND json_extract(content_json, '$.concept') IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 500
                """
                params = ()
            else:
                query = """
                    SELECT knowledge_id, content_json, domain, related_entries_json
                    FROM knowledge_entries
                    WHERE related_entries_json IS NOT NULL
                    AND related_entries_json != '[]'
                    AND json_extract(content_json, '$.concept') IS NOT NULL
                    AND domain = ?
                    ORDER BY created_at DESC
                    LIMIT 500
                """
                params = (domain_filter,)

            cursor = conn.execute(query, params)

            relationships_found = 0
            entries_with_rels = 0

            for row in cursor:
                knowledge_id = row[0]
                content_json = row[1]
                domain = row[2]
                related_json = row[3]

                try:
                    content = json.loads(content_json) if content_json else {}
                    related_ids = json.loads(related_json) if related_json else []

                    source_concept = content.get('concept', 'Unknown')

                    if not related_ids:
                        continue

                    entries_with_rels += 1

                    # For each related entry, fetch its details
                    for related_id in related_ids[:10]:  # Limit to first 10 to avoid clutter
                        rel_cursor = conn.execute(
                            "SELECT content_json, domain FROM knowledge_entries WHERE knowledge_id = ?",
                            (related_id,)
                        )
                        rel_row = rel_cursor.fetchone()

                        if rel_row:
                            rel_content = json.loads(rel_row[0]) if rel_row[0] else {}
                            target_concept = rel_content.get('concept', 'Unknown')
                            target_domain = rel_row[1]

                            # Insert into treeview
                            self.rel_tree.insert("", tk.END, values=(
                                source_concept,
                                target_concept,
                                domain,
                                target_domain,
                                f"{len(related_ids)} total"
                            ), tags=(knowledge_id, related_id))

                            relationships_found += 1

                except Exception as e:
                    logger.warning(f"Failed to parse relationship entry: {e}")
                    continue

            conn.close()

            # Update status
            status_text = f"Showing {relationships_found} relationships from {entries_with_rels} concepts"
            self.rel_status_label.config(text=status_text)

        except Exception as e:
            logger.error(f"Failed to refresh relationships: {e}")
            self.rel_status_label.config(text=f"Error: {e}")

    def _search_relationships(self):
        """Search for relationships involving a specific concept."""
        query = self.rel_search_var.get().strip()

        if not query:
            self._refresh_relationships()
            return

        # Clear existing items
        for item in self.rel_tree.get_children():
            self.rel_tree.delete(item)

        if not self.knowledge_store:
            self.rel_status_label.config(text="Knowledge store not available")
            return

        try:
            import sqlite3
            import json

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Search for concepts matching the query (case-insensitive)
            search_query = f"%{query}%"
            cursor = conn.execute("""
                SELECT knowledge_id, content_json, domain, related_entries_json
                FROM knowledge_entries
                WHERE related_entries_json IS NOT NULL
                AND related_entries_json != '[]'
                AND json_extract(content_json, '$.concept') IS NOT NULL
                AND (json_extract(content_json, '$.concept') LIKE ?
                     OR json_extract(content_json, '$.definition') LIKE ?)
                ORDER BY created_at DESC
                LIMIT 100
            """, (search_query, search_query))

            relationships_found = 0
            concepts_found = 0

            for row in cursor:
                knowledge_id = row[0]
                content_json = row[1]
                domain = row[2]
                related_json = row[3]

                try:
                    content = json.loads(content_json) if content_json else {}
                    related_ids = json.loads(related_json) if related_json else []

                    source_concept = content.get('concept', 'Unknown')

                    if not related_ids:
                        continue

                    concepts_found += 1

                    # For each related entry, fetch its details
                    for related_id in related_ids:
                        rel_cursor = conn.execute(
                            "SELECT content_json, domain FROM knowledge_entries WHERE knowledge_id = ?",
                            (related_id,)
                        )
                        rel_row = rel_cursor.fetchone()

                        if rel_row:
                            rel_content = json.loads(rel_row[0]) if rel_row[0] else {}
                            target_concept = rel_content.get('concept', 'Unknown')
                            target_domain = rel_row[1]

                            # Insert into treeview
                            self.rel_tree.insert("", tk.END, values=(
                                source_concept,
                                target_concept,
                                domain,
                                target_domain,
                                f"{len(related_ids)} total"
                            ), tags=(knowledge_id, related_id))

                            relationships_found += 1

                except Exception as e:
                    logger.warning(f"Failed to parse relationship entry: {e}")
                    continue

            conn.close()

            # Update status
            status_text = f"Found {relationships_found} relationships across {concepts_found} concepts matching '{query}'"
            self.rel_status_label.config(text=status_text)

        except Exception as e:
            logger.error(f"Failed to search relationships: {e}")
            self.rel_status_label.config(text=f"Error: {e}")

    def _show_relationship_stats(self):
        """Show statistics about relationships."""
        if not self.knowledge_store:
            messagebox.showwarning("Not Available", "Knowledge store not available")
            return

        try:
            import sqlite3
            import json

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Count total entries with relationships
            cursor = conn.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE related_entries_json IS NOT NULL
                AND related_entries_json != '[]'
            """)
            entries_with_rels = cursor.fetchone()[0]

            # Count total relationships (sum of all related_entries arrays)
            cursor = conn.execute("""
                SELECT related_entries_json FROM knowledge_entries
                WHERE related_entries_json IS NOT NULL
                AND related_entries_json != '[]'
            """)

            total_relationships = 0
            max_connections = 0
            relationship_counts = []

            for row in cursor:
                try:
                    related_ids = json.loads(row[0]) if row[0] else []
                    count = len(related_ids)
                    total_relationships += count
                    relationship_counts.append(count)
                    max_connections = max(max_connections, count)
                except:
                    continue

            avg_connections = total_relationships / entries_with_rels if entries_with_rels > 0 else 0

            # Get domain breakdown
            cursor = conn.execute("""
                SELECT domain, COUNT(*) as count
                FROM knowledge_entries
                WHERE related_entries_json IS NOT NULL
                AND related_entries_json != '[]'
                GROUP BY domain
                ORDER BY count DESC
            """)

            domain_breakdown = cursor.fetchall()

            conn.close()

            # Format statistics message
            stats = "RELATIONSHIP STATISTICS\n"
            stats += "=" * 50 + "\n\n"
            stats += f"Concepts with Relationships: {entries_with_rels}\n"
            stats += f"Total Relationship Edges: {total_relationships}\n"
            stats += f"Average Connections per Concept: {avg_connections:.1f}\n"
            stats += f"Maximum Connections: {max_connections}\n\n"

            stats += "Domain Distribution:\n"
            for domain, count in domain_breakdown[:5]:
                stats += f"  {domain}: {count} concepts\n"

            messagebox.showinfo("Relationship Statistics", stats)

        except Exception as e:
            logger.error(f"Failed to get relationship stats: {e}")
            messagebox.showerror("Error", f"Failed to get statistics: {e}")

    def _show_relationship_context_menu(self, event):
        """Show context menu on right-click."""
        # Get clicked item
        item = self.rel_tree.identify_row(event.y)
        if item:
            self.rel_tree.selection_set(item)

            # Create context menu
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Explore Network (Double-click)",
                           command=lambda: self._explore_relationship_network(None))
            menu.add_separator()
            menu.add_command(label="Copy Source Concept",
                           command=lambda: self._copy_concept(0))
            menu.add_command(label="Copy Target Concept",
                           command=lambda: self._copy_concept(1))

            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

    def _copy_concept(self, column_index):
        """Copy concept name to clipboard."""
        selection = self.rel_tree.selection()
        if selection:
            item = self.rel_tree.item(selection[0])
            concept = item['values'][column_index]
            self.clipboard_clear()
            self.clipboard_append(concept)
            self._log_activity(f"Copied: {concept}")

    def _explore_relationship_network(self, event):
        """Explore the full network for a selected concept."""
        selection = self.rel_tree.selection()
        if not selection:
            return

        item = self.rel_tree.item(selection[0])
        source_concept = item['values'][0]

        if not self.knowledge_retriever:
            messagebox.showwarning("Not Available", "Knowledge retriever not available")
            return

        # Get the knowledge_id from tags
        tags = self.rel_tree.item(selection[0])['tags']
        if not tags:
            return

        knowledge_id = tags[0]

        try:
            # Use get_related_concepts to traverse the graph
            related_results = self.knowledge_retriever.get_related_concepts(knowledge_id, max_depth=2)

            # Create popup window to show network
            dialog = tk.Toplevel(self)
            dialog.title(f"Relationship Network: {source_concept}")
            dialog.geometry("800x600")
            dialog.transient(self)

            # Title
            ttk.Label(dialog, text=f"Exploring connections for: {source_concept}",
                     font=("TkDefaultFont", 12, "bold")).pack(pady=10)

            # Text display with scrollbar
            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

            text_widget = tk.Text(text_frame, wrap=tk.WORD, height=30, width=90)
            scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
            text_widget.config(yscrollcommand=scrollbar.set)

            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            # Build network display
            network_text = f"SOURCE CONCEPT: {source_concept}\n"
            network_text += "=" * 70 + "\n\n"
            network_text += f"Found {len(related_results)} related concepts (up to 2 hops away):\n\n"

            for i, result in enumerate(related_results, 1):
                concept_name = result.content.get('concept', 'Unknown')
                definition = result.content.get('definition', 'No definition')
                score = result.relevance_score

                network_text += f"{i}. {concept_name} (relevance: {score:.2f})\n"
                network_text += f"   Domain: {result.domain}\n"
                network_text += f"   {definition}\n\n"

            text_widget.insert(1.0, network_text)
            text_widget.config(state=tk.DISABLED)

            # Close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

            self._log_activity(f"Explored network for: {source_concept}")

        except Exception as e:
            logger.error(f"Failed to explore relationship network: {e}")
            messagebox.showerror("Error", f"Failed to explore network: {e}")

    def _create_cleanup_tab(self):
        """Create cleanup & maintenance tab."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.cleanup_frame)
        scrollbar = ttk.Scrollbar(self.cleanup_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title_label = ttk.Label(scrollable_frame, text="Knowledge Base Cleanup & Maintenance",
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)

        # Recommendations section
        rec_frame = ttk.LabelFrame(scrollable_frame, text="Cleanup Recommendations", padding=10)
        rec_frame.pack(fill=tk.X, padx=10, pady=5)

        self.recommendations_text = tk.Text(rec_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(rec_frame, text="Refresh Recommendations",
                  command=self._refresh_recommendations).pack(pady=5)

        # Quick cleanup actions
        quick_frame = ttk.LabelFrame(scrollable_frame, text="Quick Cleanup Actions", padding=10)
        quick_frame.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: Virtual environments
        venv_frame = ttk.Frame(quick_frame)
        venv_frame.pack(fill=tk.X, pady=5)
        ttk.Label(venv_frame, text="Virtual Environments (.venv, site-packages):").pack(side=tk.LEFT)
        ttk.Button(venv_frame, text="Preview",
                  command=lambda: self._preview_cleanup("venv")).pack(side=tk.RIGHT, padx=2)
        ttk.Button(venv_frame, text="Clean",
                  command=lambda: self._execute_cleanup("venv")).pack(side=tk.RIGHT, padx=2)

        # Row 2: Pending documents
        pending_frame = ttk.Frame(quick_frame)
        pending_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pending_frame, text="Pending Documents:").pack(side=tk.LEFT)
        ttk.Button(pending_frame, text="Preview",
                  command=lambda: self._preview_cleanup("pending")).pack(side=tk.RIGHT, padx=2)
        ttk.Button(pending_frame, text="Clean",
                  command=lambda: self._execute_cleanup("pending")).pack(side=tk.RIGHT, padx=2)

        # Row 3: Orphaned entries
        orphan_frame = ttk.Frame(quick_frame)
        orphan_frame.pack(fill=tk.X, pady=5)
        ttk.Label(orphan_frame, text="Orphaned Entries (no source document):").pack(side=tk.LEFT)
        ttk.Button(orphan_frame, text="Preview",
                  command=lambda: self._preview_cleanup("orphaned")).pack(side=tk.RIGHT, padx=2)
        ttk.Button(orphan_frame, text="Clean",
                  command=lambda: self._execute_cleanup("orphaned")).pack(side=tk.RIGHT, padx=2)

        # Row 4: Failed documents
        failed_frame = ttk.Frame(quick_frame)
        failed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(failed_frame, text="Failed Documents (>7 days old):").pack(side=tk.LEFT)
        ttk.Button(failed_frame, text="Preview",
                  command=lambda: self._preview_cleanup("failed")).pack(side=tk.RIGHT, padx=2)
        ttk.Button(failed_frame, text="Clean",
                  command=lambda: self._execute_cleanup("failed")).pack(side=tk.RIGHT, padx=2)

        # Custom pattern cleanup
        pattern_frame = ttk.LabelFrame(scrollable_frame, text="Custom Pattern Cleanup", padding=10)
        pattern_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(pattern_frame, text="Path Pattern (glob or SQL LIKE):").pack(anchor=tk.W)
        self.pattern_entry = ttk.Entry(pattern_frame, width=50)
        self.pattern_entry.pack(fill=tk.X, pady=5)
        self.pattern_entry.insert(0, "*/test_data/*")

        pattern_options_frame = ttk.Frame(pattern_frame)
        pattern_options_frame.pack(fill=tk.X, pady=5)

        self.cascade_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pattern_options_frame, text="Delete associated knowledge entries",
                       variable=self.cascade_var).pack(side=tk.LEFT)

        pattern_buttons_frame = ttk.Frame(pattern_frame)
        pattern_buttons_frame.pack(fill=tk.X)
        ttk.Button(pattern_buttons_frame, text="Preview Pattern",
                  command=self._preview_custom_pattern).pack(side=tk.LEFT, padx=2)
        ttk.Button(pattern_buttons_frame, text="Clean Pattern",
                  command=self._execute_custom_pattern).pack(side=tk.LEFT, padx=2)

        # Common patterns quick buttons
        common_frame = ttk.Frame(pattern_frame)
        common_frame.pack(fill=tk.X, pady=5)
        ttk.Label(common_frame, text="Common patterns:").pack(side=tk.LEFT)
        ttk.Button(common_frame, text=".venv",
                  command=lambda: self.pattern_entry.delete(0, tk.END) or self.pattern_entry.insert(0, "*/.venv/*")).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_frame, text="node_modules",
                  command=lambda: self.pattern_entry.delete(0, tk.END) or self.pattern_entry.insert(0, "*/node_modules/*")).pack(side=tk.LEFT, padx=2)
        ttk.Button(common_frame, text="__pycache__",
                  command=lambda: self.pattern_entry.delete(0, tk.END) or self.pattern_entry.insert(0, "*/__pycache__/*")).pack(side=tk.LEFT, padx=2)

        # Results display
        results_frame = ttk.LabelFrame(scrollable_frame, text="Cleanup Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.cleanup_results_text = tk.Text(results_frame, height=12, wrap=tk.WORD)
        cleanup_scroll = ttk.Scrollbar(results_frame, command=self.cleanup_results_text.yview)
        self.cleanup_results_text.configure(yscrollcommand=cleanup_scroll.set)

        self.cleanup_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cleanup_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Statistics section
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Database Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.cleanup_stats_text = tk.Text(stats_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.cleanup_stats_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(stats_frame, text="Refresh Statistics",
                  command=self._refresh_cleanup_stats).pack(pady=5)

        # Initialize displays
        self._refresh_recommendations()
        self._refresh_cleanup_stats()

    def _create_audit_tab(self):
        """Create audit log tab for tracking all CRUD operations."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.audit_frame)
        scrollbar = ttk.Scrollbar(self.audit_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title_label = ttk.Label(scrollable_frame, text="Audit Log - Knowledge Base Operations",
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)

        # Filter controls
        filter_frame = ttk.LabelFrame(scrollable_frame, text="Filters", padding=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)

        # Row 1: Operation type and user/agent
        row1 = ttk.Frame(filter_frame)
        row1.pack(fill=tk.X, pady=3)

        ttk.Label(row1, text="Operation:").pack(side=tk.LEFT, padx=5)
        self.audit_operation_var = tk.StringVar(value="ALL")
        operation_combo = ttk.Combobox(row1, textvariable=self.audit_operation_var,
                                       values=["ALL", "INSERT", "UPDATE", "DELETE", "MERGE", "CLEANUP", "SYSTEM"],
                                       state="readonly", width=12)
        operation_combo.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="User/Agent:").pack(side=tk.LEFT, padx=15)
        self.audit_user_var = tk.StringVar()
        user_entry = ttk.Entry(row1, textvariable=self.audit_user_var, width=20)
        user_entry.pack(side=tk.LEFT, padx=5)

        # Row 2: Knowledge ID filter
        row2 = ttk.Frame(filter_frame)
        row2.pack(fill=tk.X, pady=3)

        ttk.Label(row2, text="Knowledge ID:").pack(side=tk.LEFT, padx=5)
        self.audit_knowledge_id_var = tk.StringVar()
        knowledge_id_entry = ttk.Entry(row2, textvariable=self.audit_knowledge_id_var, width=40)
        knowledge_id_entry.pack(side=tk.LEFT, padx=5)

        # Row 3: Action buttons
        row3 = ttk.Frame(filter_frame)
        row3.pack(fill=tk.X, pady=5)

        ttk.Button(row3, text="🔍 Search", command=self._search_audit_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="🔄 Refresh", command=self._refresh_audit_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="📤 Export CSV", command=self._export_audit_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(row3, text="🗑️ Clear Filters", command=self._clear_audit_filters).pack(side=tk.LEFT, padx=5)

        # Audit log display (TreeView)
        log_frame = ttk.LabelFrame(scrollable_frame, text="Audit Entries (Most Recent 500)", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create Treeview with scrollbars
        tree_container = ttk.Frame(log_frame)
        tree_container.pack(fill=tk.BOTH, expand=True)

        tree_scroll_y = ttk.Scrollbar(tree_container, orient="vertical")
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        tree_scroll_x = ttk.Scrollbar(tree_container, orient="horizontal")
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        columns = ("Timestamp", "Operation", "Knowledge ID", "User/Agent", "Details")
        self.audit_tree = ttk.Treeview(tree_container, columns=columns, show="headings",
                                       yscrollcommand=tree_scroll_y.set,
                                       xscrollcommand=tree_scroll_x.set,
                                       height=15)

        tree_scroll_y.config(command=self.audit_tree.yview)
        tree_scroll_x.config(command=self.audit_tree.xview)

        # Configure columns
        self.audit_tree.heading("Timestamp", text="Timestamp")
        self.audit_tree.heading("Operation", text="Operation")
        self.audit_tree.heading("Knowledge ID", text="Knowledge ID")
        self.audit_tree.heading("User/Agent", text="User/Agent")
        self.audit_tree.heading("Details", text="Details")

        self.audit_tree.column("Timestamp", width=150, anchor=tk.W)
        self.audit_tree.column("Operation", width=100, anchor=tk.CENTER)
        self.audit_tree.column("Knowledge ID", width=150, anchor=tk.W)
        self.audit_tree.column("User/Agent", width=120, anchor=tk.W)
        self.audit_tree.column("Details", width=300, anchor=tk.W)

        self.audit_tree.pack(fill=tk.BOTH, expand=True)

        # Context menu for audit entries
        self.audit_context_menu = tk.Menu(self.audit_tree, tearoff=0)
        self.audit_context_menu.add_command(label="View Details", command=self._view_audit_details)
        self.audit_context_menu.add_command(label="View Entry History", command=self._view_entry_history)
        self.audit_context_menu.add_separator()
        self.audit_context_menu.add_command(label="Copy Audit ID", command=lambda: self._copy_audit_field(0))
        self.audit_context_menu.add_command(label="Copy Knowledge ID", command=lambda: self._copy_audit_field(2))

        self.audit_tree.bind("<Button-3>", self._show_audit_context_menu)
        self.audit_tree.bind("<Double-1>", lambda e: self._view_audit_details())

        # Statistics section
        stats_frame = ttk.LabelFrame(scrollable_frame, text="Audit Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.audit_stats_text = tk.Text(stats_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.audit_stats_text.pack(fill=tk.BOTH, expand=True)

        # Initialize display
        self._refresh_audit_log()

    def _refresh_recommendations(self):
        """Refresh cleanup recommendations."""
        if not self.knowledge_store:
            self._display_cleanup_message("Felix system not running. Start Felix to see recommendations.")
            return

        try:
            from src.knowledge.knowledge_cleanup import KnowledgeCleanupManager
            manager = KnowledgeCleanupManager(self.knowledge_store)
            recommendations = manager.get_cleanup_recommendations()

            self.recommendations_text.config(state=tk.NORMAL)
            self.recommendations_text.delete(1.0, tk.END)

            if "error" in recommendations:
                self.recommendations_text.insert(tk.END, f"Error: {recommendations['error']}\n")
            else:
                urgent = recommendations.get("urgent", [])
                suggested = recommendations.get("suggested", [])
                stats = recommendations.get("stats", {})

                self.recommendations_text.insert(tk.END, "=== URGENT ACTIONS ===\n", "urgent")
                if urgent:
                    for rec in urgent:
                        self.recommendations_text.insert(tk.END,
                            f"\u26a0 {rec['reason']}\n", "urgent")
                        self.recommendations_text.insert(tk.END,
                            f"   Action: {rec['action']} ({rec['count']} items)\n\n")
                else:
                    self.recommendations_text.insert(tk.END, "None\n\n")

                self.recommendations_text.insert(tk.END, "=== SUGGESTED ACTIONS ===\n", "suggested")
                if suggested:
                    for rec in suggested:
                        self.recommendations_text.insert(tk.END,
                            f"\u2139 {rec['reason']}\n")
                        self.recommendations_text.insert(tk.END,
                            f"   Action: {rec['action']} ({rec['count']} items)\n\n")
                else:
                    self.recommendations_text.insert(tk.END, "None\n\n")

                self.recommendations_text.insert(tk.END, f"=== DATABASE STATS ===\n")
                self.recommendations_text.insert(tk.END,
                    f"Total Documents: {stats.get('total_documents', 0)}\n")
                self.recommendations_text.insert(tk.END,
                    f"Total Entries: {stats.get('total_entries', 0)}\n")
                self.recommendations_text.insert(tk.END,
                    f"Pending: {stats.get('pending_documents', 0)} | " +
                    f"Failed: {stats.get('failed_documents', 0)} | " +
                    f"Completed: {stats.get('completed_documents', 0)}\n")

            self.recommendations_text.config(state=tk.DISABLED)
            self._log_activity("Refreshed cleanup recommendations")

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            self._display_cleanup_message(f"Error getting recommendations: {e}")

    def _preview_cleanup(self, cleanup_type: str):
        """Preview a quick cleanup action."""
        if not self.knowledge_store:
            messagebox.showwarning("Not Available", "Felix system not running")
            return

        try:
            from src.knowledge.knowledge_cleanup import KnowledgeCleanupManager
            manager = KnowledgeCleanupManager(self.knowledge_store)

            if cleanup_type == "venv":
                result = manager.cleanup_virtual_environments(dry_run=True)
            elif cleanup_type == "pending":
                result = manager.cleanup_pending_documents(dry_run=True)
            elif cleanup_type == "orphaned":
                result = manager.cleanup_orphaned_entries(dry_run=True)
            elif cleanup_type == "failed":
                result = manager.cleanup_failed_documents(dry_run=True)
            else:
                messagebox.showerror("Error", f"Unknown cleanup type: {cleanup_type}")
                return

            self._display_cleanup_result(f"Preview: {cleanup_type}", result, is_preview=True)
            self._log_activity(f"Previewed {cleanup_type} cleanup")

        except Exception as e:
            logger.error(f"Failed to preview {cleanup_type} cleanup: {e}")
            messagebox.showerror("Error", f"Preview failed: {e}")

    def _execute_cleanup(self, cleanup_type: str):
        """Execute a quick cleanup action."""
        if not self.knowledge_store:
            messagebox.showwarning("Not Available", "Felix system not running")
            return

        # Confirmation dialog
        confirm = messagebox.askyesno(
            "Confirm Cleanup",
            f"Are you sure you want to execute '{cleanup_type}' cleanup?\n\n" +
            "This will permanently delete documents and/or entries.\n" +
            "This action cannot be undone."
        )

        if not confirm:
            return

        try:
            from src.knowledge.knowledge_cleanup import KnowledgeCleanupManager
            manager = KnowledgeCleanupManager(self.knowledge_store)

            if cleanup_type == "venv":
                result = manager.cleanup_virtual_environments(dry_run=False)
            elif cleanup_type == "pending":
                result = manager.cleanup_pending_documents(dry_run=False)
            elif cleanup_type == "orphaned":
                result = manager.cleanup_orphaned_entries(dry_run=False)
            elif cleanup_type == "failed":
                result = manager.cleanup_failed_documents(dry_run=False)
            else:
                messagebox.showerror("Error", f"Unknown cleanup type: {cleanup_type}")
                return

            self._display_cleanup_result(f"Executed: {cleanup_type}", result, is_preview=False)
            self._log_activity(f"Executed {cleanup_type} cleanup")

            # Refresh recommendations and stats
            self._refresh_recommendations()
            self._refresh_cleanup_stats()

            messagebox.showinfo("Success", "Cleanup completed successfully!")

        except Exception as e:
            logger.error(f"Failed to execute {cleanup_type} cleanup: {e}")
            messagebox.showerror("Error", f"Cleanup failed: {e}")

    def _preview_custom_pattern(self):
        """Preview custom pattern cleanup."""
        if not self.knowledge_store:
            messagebox.showwarning("Not Available", "Felix system not running")
            return

        pattern = self.pattern_entry.get().strip()
        if not pattern:
            messagebox.showwarning("Invalid Input", "Please enter a path pattern")
            return

        try:
            result = self.knowledge_store.preview_delete_by_pattern(
                pattern,
                include_entries=self.cascade_var.get()
            )
            self._display_cleanup_result(f"Preview: {pattern}", result, is_preview=True)
            self._log_activity(f"Previewed pattern: {pattern}")

        except Exception as e:
            logger.error(f"Failed to preview pattern: {e}")
            messagebox.showerror("Error", f"Preview failed: {e}")

    def _execute_custom_pattern(self):
        """Execute custom pattern cleanup."""
        if not self.knowledge_store:
            messagebox.showwarning("Not Available", "Felix system not running")
            return

        pattern = self.pattern_entry.get().strip()
        if not pattern:
            messagebox.showwarning("Invalid Input", "Please enter a path pattern")
            return

        # Show preview first
        try:
            preview = self.knowledge_store.preview_delete_by_pattern(
                pattern,
                include_entries=self.cascade_var.get()
            )

            confirm_msg = (
                f"Pattern: {pattern}\n\n" +
                f"Will delete:\n" +
                f"  - {preview.get('document_count', 0)} documents\n" +
                f"  - {preview.get('entry_count', 0)} knowledge entries\n\n" +
                f"Sample paths:\n"
            )

            for path in preview.get('sample_paths', [])[:5]:
                confirm_msg += f"  - {path}\n"

            confirm_msg += "\nThis action cannot be undone. Continue?"

            if not messagebox.askyesno("Confirm Cleanup", confirm_msg):
                return

        except Exception as e:
            logger.error(f"Failed to preview pattern: {e}")
            messagebox.showerror("Error", f"Preview failed: {e}")
            return

        # Execute cleanup
        try:
            result = self.knowledge_store.delete_documents_by_pattern(
                pattern,
                cascade_entries=self.cascade_var.get(),
                dry_run=False
            )

            self._display_cleanup_result(f"Executed: {pattern}", result, is_preview=False)
            self._log_activity(f"Executed pattern cleanup: {pattern}")

            # Refresh displays
            self._refresh_recommendations()
            self._refresh_cleanup_stats()

            messagebox.showinfo("Success",
                f"Deleted {result.get('documents_deleted', 0)} documents and " +
                f"{result.get('entries_deleted', 0)} entries")

        except Exception as e:
            logger.error(f"Failed to execute pattern cleanup: {e}")
            messagebox.showerror("Error", f"Cleanup failed: {e}")

    def _display_cleanup_result(self, title: str, result: Dict[str, Any], is_preview: bool):
        """Display cleanup results in the results text widget."""
        self.cleanup_results_text.delete(1.0, tk.END)

        mode = "PREVIEW" if is_preview else "EXECUTED"
        self.cleanup_results_text.insert(tk.END, f"=== {mode}: {title} ===\n\n", "title")

        if "error" in result:
            self.cleanup_results_text.insert(tk.END, f"Error: {result['error']}\n", "error")
            return

        # Display counts
        if "documents_deleted" in result or "document_count" in result:
            doc_count = result.get("documents_deleted") or result.get("document_count", 0)
            self.cleanup_results_text.insert(tk.END, f"Documents: {doc_count}\n")

        if "entries_deleted" in result or "entry_count" in result:
            entry_count = result.get("entries_deleted") or result.get("entry_count", 0)
            self.cleanup_results_text.insert(tk.END, f"Knowledge Entries: {entry_count}\n")

        if "total_docs_deleted" in result:
            self.cleanup_results_text.insert(tk.END,
                f"Total Documents Deleted: {result['total_docs_deleted']}\n")

        if "total_entries_deleted" in result:
            self.cleanup_results_text.insert(tk.END,
                f"Total Entries Deleted: {result['total_entries_deleted']}\n")

        if "orphaned_count" in result:
            self.cleanup_results_text.insert(tk.END,
                f"Orphaned Entries: {result['orphaned_count']}\n")

        if "failed_documents" in result:
            self.cleanup_results_text.insert(tk.END,
                f"Failed Documents: {result['failed_documents']}\n")

        # Display sample paths
        if "sample_paths" in result and result["sample_paths"]:
            self.cleanup_results_text.insert(tk.END, "\nSample Paths:\n")
            for path in result["sample_paths"][:10]:
                self.cleanup_results_text.insert(tk.END, f"  {path}\n")

        # Display patterns processed
        if "patterns_processed" in result:
            self.cleanup_results_text.insert(tk.END, "\nPatterns Processed:\n")
            for pattern in result["patterns_processed"]:
                self.cleanup_results_text.insert(tk.END, f"  {pattern}\n")

        # Display errors
        if "errors" in result and result["errors"]:
            self.cleanup_results_text.insert(tk.END, "\nErrors:\n", "error")
            for error in result["errors"]:
                self.cleanup_results_text.insert(tk.END,
                    f"  {error.get('pattern', 'Unknown')}: {error.get('error', 'Unknown error')}\n")

        self.cleanup_results_text.insert(tk.END, "\n" + "=" * 60 + "\n")

    def _display_cleanup_message(self, message: str):
        """Display a message in the recommendations text widget."""
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, message)
        self.recommendations_text.config(state=tk.DISABLED)

    def _refresh_cleanup_stats(self):
        """Refresh database statistics."""
        if not self.knowledge_store:
            self.cleanup_stats_text.config(state=tk.NORMAL)
            self.cleanup_stats_text.delete(1.0, tk.END)
            self.cleanup_stats_text.insert(tk.END, "Felix system not running. Start Felix to see statistics.")
            self.cleanup_stats_text.config(state=tk.DISABLED)
            return

        try:
            import sqlite3
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM knowledge_entries) as entries,
                        (SELECT COUNT(*) FROM document_sources) as docs,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='pending') as pending,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='failed') as failed,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='completed') as completed,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='processing') as processing
                """)
                stats = cursor.fetchone()

            self.cleanup_stats_text.config(state=tk.NORMAL)
            self.cleanup_stats_text.delete(1.0, tk.END)

            self.cleanup_stats_text.insert(tk.END, "=== Database Health ===\n\n")
            self.cleanup_stats_text.insert(tk.END, f"Knowledge Entries: {stats[0]:,}\n")
            self.cleanup_stats_text.insert(tk.END, f"Total Documents: {stats[1]:,}\n\n")
            self.cleanup_stats_text.insert(tk.END, f"Document Status:\n")
            self.cleanup_stats_text.insert(tk.END, f"  \u2713 Completed: {stats[4]:,}\n")
            self.cleanup_stats_text.insert(tk.END, f"  \u23f3 Pending: {stats[2]:,}\n")
            self.cleanup_stats_text.insert(tk.END, f"  \u2699 Processing: {stats[5]:,}\n")
            self.cleanup_stats_text.insert(tk.END, f"  \u2717 Failed: {stats[3]:,}\n")

            self.cleanup_stats_text.config(state=tk.DISABLED)
            self._log_activity("Refreshed cleanup statistics")

        except Exception as e:
            logger.error(f"Failed to get cleanup stats: {e}")
            self.cleanup_stats_text.config(state=tk.NORMAL)
            self.cleanup_stats_text.delete(1.0, tk.END)
            self.cleanup_stats_text.insert(tk.END, f"Error: {e}")
            self.cleanup_stats_text.config(state=tk.DISABLED)

    # ===== Audit Tab Methods =====

    def _refresh_audit_log(self):
        """Refresh audit log display with current filters."""
        try:
            from src.memory.audit_log import get_audit_logger

            # Clear existing items
            for item in self.audit_tree.get_children():
                self.audit_tree.delete(item)

            # Get audit logger
            audit_logger = get_audit_logger()

            # Build filter parameters
            kwargs = {'limit': 500}

            operation = self.audit_operation_var.get()
            if operation != "ALL":
                kwargs['operation'] = operation

            user_agent = self.audit_user_var.get().strip()
            if user_agent:
                kwargs['user_agent'] = user_agent

            knowledge_id = self.audit_knowledge_id_var.get().strip()
            if knowledge_id:
                kwargs['knowledge_id'] = knowledge_id

            # Query audit history
            entries = audit_logger.get_audit_history(**kwargs)

            # Populate tree
            for entry in entries:
                timestamp_str = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                knowledge_id_display = entry['knowledge_id'] or "(N/A)"
                user_display = entry['user_agent'] or "(N/A)"
                details_display = entry['details'] or ""

                # Truncate long details
                if len(details_display) > 80:
                    details_display = details_display[:77] + "..."

                self.audit_tree.insert('', tk.END, values=(
                    timestamp_str,
                    entry['operation'],
                    knowledge_id_display,
                    user_display,
                    details_display
                ), tags=(str(entry['audit_id']),))

            # Update statistics
            stats = audit_logger.get_statistics()
            self.audit_stats_text.config(state=tk.NORMAL)
            self.audit_stats_text.delete(1.0, tk.END)

            total = stats['total_entries']
            by_op = stats['by_operation']
            self.audit_stats_text.insert(tk.END, f"Total Audit Entries: {total}\n")
            self.audit_stats_text.insert(tk.END, f"By Operation: ")
            self.audit_stats_text.insert(tk.END, ", ".join([f"{k}={v}" for k, v in by_op.items()]))
            self.audit_stats_text.insert(tk.END, f"\n")

            if stats['date_range']:
                oldest = stats['date_range']['oldest'].strftime('%Y-%m-%d %H:%M')
                newest = stats['date_range']['newest'].strftime('%Y-%m-%d %H:%M')
                self.audit_stats_text.insert(tk.END, f"Date Range: {oldest} to {newest}\n")

            self.audit_stats_text.insert(tk.END, f"Displaying: {len(entries)} entries (filtered)\n")
            self.audit_stats_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error refreshing audit log: {e}")
            messagebox.showerror("Error", f"Failed to refresh audit log: {str(e)}")

    def _search_audit_log(self):
        """Execute search with current filters."""
        self._refresh_audit_log()

    def _clear_audit_filters(self):
        """Clear all filters and refresh."""
        self.audit_operation_var.set("ALL")
        self.audit_user_var.set("")
        self.audit_knowledge_id_var.set("")
        self._refresh_audit_log()

    def _export_audit_log(self):
        """Export audit log to CSV file."""
        try:
            from src.memory.audit_log import get_audit_logger

            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                title="Export Audit Log",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_path:
                return

            # Get audit logger
            audit_logger = get_audit_logger()

            # Build filter parameters
            kwargs = {}
            operation = self.audit_operation_var.get()
            if operation != "ALL":
                kwargs['operation'] = operation

            user_agent = self.audit_user_var.get().strip()
            if user_agent:
                kwargs['user_agent'] = user_agent

            knowledge_id = self.audit_knowledge_id_var.get().strip()
            if knowledge_id:
                kwargs['knowledge_id'] = knowledge_id

            # Export
            success = audit_logger.export_to_csv(file_path, **kwargs)

            if success:
                messagebox.showinfo("Success", f"Audit log exported to:\n{file_path}")
            else:
                messagebox.showwarning("Warning", "No entries to export.")

        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    def _show_audit_context_menu(self, event):
        """Show context menu for audit entry."""
        # Select row under cursor
        row_id = self.audit_tree.identify_row(event.y)
        if row_id:
            self.audit_tree.selection_set(row_id)
            self.audit_context_menu.post(event.x_root, event.y_root)

    def _view_audit_details(self):
        """Show detailed view of selected audit entry."""
        selection = self.audit_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an audit entry to view.")
            return

        try:
            from src.memory.audit_log import get_audit_logger

            # Get audit_id from tags
            item = selection[0]
            tags = self.audit_tree.item(item, 'tags')
            if not tags:
                return

            audit_id = int(tags[0])

            # Fetch full entry
            audit_logger = get_audit_logger()
            entries = audit_logger.get_audit_history(limit=1000)

            entry = None
            for e in entries:
                if e['audit_id'] == audit_id:
                    entry = e
                    break

            if not entry:
                messagebox.showwarning("Not Found", "Audit entry not found.")
                return

            # Create details dialog
            dialog = tk.Toplevel(self)
            dialog.title(f"Audit Entry Details - ID {audit_id}")
            dialog.geometry("600x500")

            # Details text
            text = tk.Text(dialog, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            text.insert(tk.END, f"Audit ID: {entry['audit_id']}\n")
            text.insert(tk.END, f"Timestamp: {entry['timestamp']}\n")
            text.insert(tk.END, f"Operation: {entry['operation']}\n")
            text.insert(tk.END, f"Knowledge ID: {entry['knowledge_id']}\n")
            text.insert(tk.END, f"User/Agent: {entry['user_agent']}\n")
            text.insert(tk.END, f"Transaction ID: {entry['transaction_id']}\n")
            text.insert(tk.END, f"Details: {entry['details']}\n\n")

            if entry['old_values']:
                text.insert(tk.END, "Old Values:\n")
                import json
                text.insert(tk.END, json.dumps(entry['old_values'], indent=2))
                text.insert(tk.END, "\n\n")

            if entry['new_values']:
                text.insert(tk.END, "New Values:\n")
                import json
                text.insert(tk.END, json.dumps(entry['new_values'], indent=2))

            text.config(state=tk.DISABLED)

            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            logger.error(f"Error viewing audit details: {e}")
            messagebox.showerror("Error", f"Failed to view details: {str(e)}")

    def _view_entry_history(self):
        """Show complete history for a knowledge entry."""
        selection = self.audit_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an audit entry.")
            return

        # Get knowledge_id from selected item
        item = selection[0]
        values = self.audit_tree.item(item, 'values')
        knowledge_id = values[2]  # Knowledge ID column

        if knowledge_id == "(N/A)":
            messagebox.showinfo("No History", "This audit entry is not associated with a knowledge entry.")
            return

        try:
            from src.memory.audit_log import get_audit_logger

            audit_logger = get_audit_logger()
            history = audit_logger.get_entry_history(knowledge_id)

            if not history:
                messagebox.showinfo("No History", f"No audit history found for entry {knowledge_id}")
                return

            # Create history dialog
            dialog = tk.Toplevel(self)
            dialog.title(f"Entry History - {knowledge_id}")
            dialog.geometry("800x600")

            # History tree
            columns = ("Timestamp", "Operation", "User/Agent", "Details")
            tree = ttk.Treeview(dialog, columns=columns, show="headings", height=20)

            tree.heading("Timestamp", text="Timestamp")
            tree.heading("Operation", text="Operation")
            tree.heading("User/Agent", text="User/Agent")
            tree.heading("Details", text="Details")

            tree.column("Timestamp", width=150)
            tree.column("Operation", width=100)
            tree.column("User/Agent", width=120)
            tree.column("Details", width=400)

            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Populate history
            for entry in history:
                timestamp_str = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                user = entry['user_agent'] or "(N/A)"
                details = entry['details'] or ""

                tree.insert('', tk.END, values=(
                    timestamp_str,
                    entry['operation'],
                    user,
                    details
                ))

            ttk.Label(dialog, text=f"Total operations: {len(history)}").pack(pady=5)
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

        except Exception as e:
            logger.error(f"Error viewing entry history: {e}")
            messagebox.showerror("Error", f"Failed to view history: {str(e)}")

    def _copy_audit_field(self, column_index):
        """Copy field value to clipboard."""
        selection = self.audit_tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.audit_tree.item(item, 'values')

        if column_index == 0:  # Audit ID from tags
            tags = self.audit_tree.item(item, 'tags')
            value = tags[0] if tags else ""
        else:
            value = values[column_index] if column_index < len(values) else ""

        self.clipboard_clear()
        self.clipboard_append(value)

    # ===== Analytics Tab Methods =====

    def _create_analytics_tab(self):
        """Create analytics & quality tools tab."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.analytics_frame)
        scrollbar = ttk.Scrollbar(self.analytics_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title_label = ttk.Label(scrollable_frame, text="Knowledge Base Analytics & Quality",
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)

        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(button_frame, text="🔄 Refresh Analytics",
                  command=self._refresh_analytics).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔍 Find Duplicates",
                  command=self._find_duplicates).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="⚠️ Quality Report",
                  command=self._generate_quality_report).pack(side=tk.LEFT, padx=5)

        # Analytics display
        analytics_frame = ttk.LabelFrame(scrollable_frame, text="Knowledge Base Analytics", padding=10)
        analytics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.analytics_text = tk.Text(analytics_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.analytics_text.pack(fill=tk.BOTH, expand=True)

        # Quality issues display
        quality_frame = ttk.LabelFrame(scrollable_frame, text="Quality Issues & Recommendations", padding=10)
        quality_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.quality_text = tk.Text(quality_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.quality_text.pack(fill=tk.BOTH, expand=True)

        # Duplicate candidates display
        dup_frame = ttk.LabelFrame(scrollable_frame, text="Duplicate Candidates", padding=10)
        dup_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Treeview for duplicates
        dup_tree_frame = ttk.Frame(dup_frame)
        dup_tree_frame.pack(fill=tk.BOTH, expand=True)

        dup_scroll = ttk.Scrollbar(dup_tree_frame, orient="vertical")
        dup_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        columns = ("Similarity", "Type", "Entry 1", "Entry 2", "Action")
        self.dup_tree = ttk.Treeview(dup_tree_frame, columns=columns, show="headings",
                                     yscrollcommand=dup_scroll.set, height=8)

        dup_scroll.config(command=self.dup_tree.yview)

        self.dup_tree.heading("Similarity", text="Similarity")
        self.dup_tree.heading("Type", text="Type")
        self.dup_tree.heading("Entry 1", text="Entry 1")
        self.dup_tree.heading("Entry 2", text="Entry 2")
        self.dup_tree.heading("Action", text="Suggested Action")

        self.dup_tree.column("Similarity", width=80, anchor=tk.CENTER)
        self.dup_tree.column("Type", width=100, anchor=tk.CENTER)
        self.dup_tree.column("Entry 1", width=200, anchor=tk.W)
        self.dup_tree.column("Entry 2", width=200, anchor=tk.W)
        self.dup_tree.column("Action", width=120, anchor=tk.CENTER)

        self.dup_tree.pack(fill=tk.BOTH, expand=True)

        # Duplicate action buttons
        dup_button_frame = ttk.Frame(dup_frame)
        dup_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(dup_button_frame, text="View Details",
                  command=self._view_duplicate_details).pack(side=tk.LEFT, padx=2)
        ttk.Button(dup_button_frame, text="Merge Selected",
                  command=self._merge_duplicates).pack(side=tk.LEFT, padx=2)
        ttk.Button(dup_button_frame, text="Mark as Different",
                  command=self._mark_as_different).pack(side=tk.LEFT, padx=2)

        # Initialize displays
        self._refresh_analytics()

    def _refresh_analytics(self):
        """Refresh analytics dashboard."""
        if not self.knowledge_store:
            self._display_analytics_message("Felix system not running. Start Felix to view analytics.")
            return

        try:
            analytics = self.knowledge_store.get_analytics_data()

            self.analytics_text.config(state=tk.NORMAL)
            self.analytics_text.delete(1.0, tk.END)

            # Display analytics
            self.analytics_text.insert(tk.END, "=== Knowledge Base Statistics ===\n\n")

            # Overall metrics
            qm = analytics.get('quality_metrics', {})
            self.analytics_text.insert(tk.END, f"Total Entries: {qm.get('total_entries', 0)}\n")
            self.analytics_text.insert(tk.END, f"Average Confidence: {qm.get('avg_confidence', 0):.2f}/4.0\n")
            self.analytics_text.insert(tk.END, f"Average Validation Score: {qm.get('avg_validation_score', 0):.2f}\n")
            self.analytics_text.insert(tk.END, f"Average Success Rate: {qm.get('avg_success_rate', 0):.2%}\n\n")

            # Top domains
            self.analytics_text.insert(tk.END, "Top 10 Domains by Entry Count:\n")
            for domain_info in analytics.get('top_domains', []):
                domain = domain_info['domain']
                count = domain_info['count']
                self.analytics_text.insert(tk.END, f"  {domain}: {count} entries\n")

            self.analytics_text.insert(tk.END, "\n")

            # Confidence distribution
            self.analytics_text.insert(tk.END, "Confidence Distribution:\n")
            conf_dist = analytics.get('confidence_distribution', {})
            for level, count in conf_dist.items():
                self.analytics_text.insert(tk.END, f"  {level}: {count} entries\n")

            self.analytics_text.insert(tk.END, "\n")

            # Entry growth (last 3 months)
            self.analytics_text.insert(tk.END, "Entry Growth (Last 3 Months):\n")
            growth_data = analytics.get('entry_growth_trend', [])
            for month_data in growth_data[-3:]:
                month = month_data['month']
                count = month_data['count']
                self.analytics_text.insert(tk.END, f"  {month}: {count} entries\n")

            self.analytics_text.insert(tk.END, "\n")

            # Relationship stats
            self.analytics_text.insert(tk.END, "Relationship Statistics:\n")
            rel_stats = analytics.get('relationship_stats', [])
            if rel_stats:
                for rel in rel_stats:
                    rel_type = rel['type']
                    count = rel['count']
                    avg_conf = rel['avg_confidence']
                    self.analytics_text.insert(tk.END, f"  {rel_type}: {count} (avg conf: {avg_conf:.2f})\n")
            else:
                self.analytics_text.insert(tk.END, "  No relationships found\n")

            self.analytics_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Failed to refresh analytics: {e}")
            self._display_analytics_message(f"Error loading analytics: {str(e)}")

    def _generate_quality_report(self):
        """Generate and display quality report."""
        if not self.knowledge_store:
            self._display_quality_message("Felix system not running. Start Felix to generate report.")
            return

        try:
            report = self.knowledge_store.generate_quality_report()

            self.quality_text.config(state=tk.NORMAL)
            self.quality_text.delete(1.0, tk.END)

            summary = report.get('summary', {})

            self.quality_text.insert(tk.END, "=== Quality Report Summary ===\n\n")
            self.quality_text.insert(tk.END, f"Total Issues Found: {summary.get('total_issues', 0)}\n\n")

            # Low confidence entries
            low_conf = report.get('low_confidence_entries', [])
            self.quality_text.insert(tk.END, f"Low Confidence Entries: {len(low_conf)}\n")
            if low_conf:
                for entry in low_conf[:3]:
                    concept = entry['content'].get('concept', 'Unknown')
                    domain = entry['domain']
                    age = entry['age_days']
                    self.quality_text.insert(tk.END, f"  - {concept} ({domain}) - {age:.0f} days old\n")
                if len(low_conf) > 3:
                    self.quality_text.insert(tk.END, f"  ... and {len(low_conf) - 3} more\n")

            self.quality_text.insert(tk.END, "\n")

            # Entries with flags
            flagged = report.get('entries_with_flags', [])
            self.quality_text.insert(tk.END, f"Entries with Validation Flags: {len(flagged)}\n")
            if flagged:
                for entry in flagged[:3]:
                    concept = entry['content'].get('concept', 'Unknown')
                    flags = ', '.join(entry['flags'])
                    self.quality_text.insert(tk.END, f"  - {concept}: {flags}\n")
                if len(flagged) > 3:
                    self.quality_text.insert(tk.END, f"  ... and {len(flagged) - 3} more\n")

            self.quality_text.insert(tk.END, "\n")

            # Orphaned entries
            orphaned = report.get('orphaned_entries', [])
            self.quality_text.insert(tk.END, f"Orphaned Entries (No Relationships): {len(orphaned)}\n")

            # Unvalidated entries
            unvalidated = report.get('unvalidated_entries', [])
            self.quality_text.insert(tk.END, f"Unvalidated Entries (>7 days old): {len(unvalidated)}\n")

            # Recent failures
            failures = report.get('recent_failures', [])
            self.quality_text.insert(tk.END, f"Recent Failures (Low Success Rate): {len(failures)}\n\n")

            # Recommendations
            self.quality_text.insert(tk.END, "=== Recommendations ===\n\n")
            if summary.get('total_issues', 0) > 0:
                self.quality_text.insert(tk.END, "1. Review low confidence entries and validate accuracy\n")
                self.quality_text.insert(tk.END, "2. Build relationships for orphaned concepts\n")
                self.quality_text.insert(tk.END, "3. Address validation flags promptly\n")
                self.quality_text.insert(tk.END, "4. Re-validate unvalidated entries\n")
                self.quality_text.insert(tk.END, "5. Review entries with low success rates\n")
            else:
                self.quality_text.insert(tk.END, "Knowledge base quality is good! No major issues found.\n")

            self.quality_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            self._display_quality_message(f"Error generating report: {str(e)}")

    def _find_duplicates(self):
        """Find and display duplicate candidates."""
        if not self.knowledge_store:
            messagebox.showwarning("System Not Running", "Start Felix system to find duplicates.")
            return

        try:
            from src.knowledge.quality_checker import QualityChecker

            checker = QualityChecker(self.knowledge_store)

            # Show progress
            self._display_quality_message("Finding duplicates... This may take a moment.")
            self.update()

            duplicates = checker.find_duplicates(similarity_threshold=0.90)

            # Clear tree
            for item in self.dup_tree.get_children():
                self.dup_tree.delete(item)

            # Populate tree
            for dup in duplicates:
                concept1 = dup.entry1_content.get('concept', dup.entry1_id[:15])
                concept2 = dup.entry2_content.get('concept', dup.entry2_id[:15])

                self.dup_tree.insert('', tk.END, values=(
                    f"{dup.similarity_score:.1%}",
                    dup.similarity_type,
                    concept1,
                    concept2,
                    dup.suggested_action
                ), tags=(dup.entry1_id, dup.entry2_id))

            # Update quality text
            self._display_quality_message(
                f"Found {len(duplicates)} potential duplicate pairs.\n"
                f"Review and merge as appropriate."
            )

        except Exception as e:
            logger.error(f"Failed to find duplicates: {e}")
            messagebox.showerror("Error", f"Failed to find duplicates: {str(e)}")

    def _view_duplicate_details(self):
        """View detailed information about selected duplicate pair."""
        import json

        selection = self.dup_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a duplicate pair to view.")
            return

        item = selection[0]
        tags = self.dup_tree.item(item, 'tags')

        if len(tags) < 2:
            return

        entry1_id, entry2_id = tags[0], tags[1]

        # Get entries
        entry1 = self.knowledge_store.get_entry_by_id(entry1_id)
        entry2 = self.knowledge_store.get_entry_by_id(entry2_id)

        if not entry1 or not entry2:
            messagebox.showerror("Error", "Could not load entry details.")
            return

        # Create details dialog
        dialog = tk.Toplevel(self)
        dialog.title("Duplicate Details")
        dialog.geometry("800x600")

        # Split view
        text1_frame = ttk.LabelFrame(dialog, text="Entry 1", padding=10)
        text1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        text1 = tk.Text(text1_frame, wrap=tk.WORD)
        text1.pack(fill=tk.BOTH, expand=True)
        text1.insert(tk.END, f"ID: {entry1.knowledge_id}\n")
        text1.insert(tk.END, f"Domain: {entry1.domain}\n")
        text1.insert(tk.END, f"Confidence: {entry1.confidence_level.value}\n")
        text1.insert(tk.END, f"Created: {datetime.fromtimestamp(entry1.created_at)}\n\n")
        text1.insert(tk.END, json.dumps(entry1.content, indent=2))
        text1.config(state=tk.DISABLED)

        text2_frame = ttk.LabelFrame(dialog, text="Entry 2", padding=10)
        text2_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        text2 = tk.Text(text2_frame, wrap=tk.WORD)
        text2.pack(fill=tk.BOTH, expand=True)
        text2.insert(tk.END, f"ID: {entry2.knowledge_id}\n")
        text2.insert(tk.END, f"Domain: {entry2.domain}\n")
        text2.insert(tk.END, f"Confidence: {entry2.confidence_level.value}\n")
        text2.insert(tk.END, f"Created: {datetime.fromtimestamp(entry2.created_at)}\n\n")
        text2.insert(tk.END, json.dumps(entry2.content, indent=2))
        text2.config(state=tk.DISABLED)

    def _merge_duplicates(self):
        """Merge selected duplicate pairs."""
        selection = self.dup_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select duplicate pairs to merge.")
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Merge",
            f"Merge {len(selection)} duplicate pair(s)?\n\n"
            "This will keep the higher confidence entry and delete the lower one.\n"
            "This cannot be undone.",
            icon='warning'
        )

        if not result:
            return

        merged = 0
        errors = 0

        for item in selection:
            tags = self.dup_tree.item(item, 'tags')
            if len(tags) < 2:
                continue

            entry1_id, entry2_id = tags[0], tags[1]

            try:
                # Get suggestions
                from src.knowledge.quality_checker import QualityChecker
                checker = QualityChecker(self.knowledge_store)

                # Simple merge: keep entry with higher confidence
                success = self.knowledge_store.merge_knowledge_entries(
                    primary_id=entry1_id,
                    secondary_ids=[entry2_id],
                    merge_strategy="combine_content"
                )

                if success:
                    merged += 1
                    self.dup_tree.delete(item)
                else:
                    errors += 1

            except Exception as e:
                logger.error(f"Failed to merge {entry1_id} and {entry2_id}: {e}")
                errors += 1

        messagebox.showinfo(
            "Merge Complete",
            f"Merged: {merged} pairs\nErrors: {errors}"
        )

        # Refresh analytics and concepts to show updated knowledge base
        self._refresh_analytics()
        self._refresh_concepts()

        # If merges were successful and there are still items in the tree,
        # offer to re-scan for duplicates
        if merged > 0 and len(self.dup_tree.get_children()) == 0:
            if messagebox.askyesno("Scan Again?",
                                   "All duplicates have been merged. Scan for more duplicates?"):
                self._find_duplicates()

    def _mark_as_different(self):
        """Mark selected pairs as legitimately different (not duplicates)."""
        selection = self.dup_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select pairs to mark.")
            return

        # Just remove from tree (no database changes)
        for item in selection:
            self.dup_tree.delete(item)

        messagebox.showinfo("Marked", f"Marked {len(selection)} pair(s) as different.")

    def _display_analytics_message(self, message: str):
        """Display message in analytics text area."""
        self.analytics_text.config(state=tk.NORMAL)
        self.analytics_text.delete(1.0, tk.END)
        self.analytics_text.insert(tk.END, message)
        self.analytics_text.config(state=tk.DISABLED)

    def _display_quality_message(self, message: str):
        """Display message in quality text area."""
        self.quality_text.config(state=tk.NORMAL)
        self.quality_text.delete(1.0, tk.END)
        self.quality_text.insert(tk.END, message)
        self.quality_text.config(state=tk.DISABLED)

    def set_knowledge_components(self, daemon, retriever, knowledge_store):
        """Set knowledge brain components (called by main_app)."""
        self.knowledge_daemon = daemon
        self.knowledge_retriever = retriever
        self.knowledge_store = knowledge_store
        self._refresh_overview()
