"""
Control Panel for Knowledge Brain Tab (CustomTkinter Edition)

Provides comprehensive daemon control, processing queue monitoring,
statistics dashboard, and activity feed for the autonomous knowledge brain.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from datetime import datetime
from typing import Optional
import sqlite3
import logging

from ...theme_manager import get_theme_manager
from ...components.status_card import StatusCard

logger = logging.getLogger(__name__)


class ControlPanel(ctk.CTkScrollableFrame):
    """
    Control Panel for Knowledge Brain with daemon control, queue status,
    statistics, and activity feed.

    Features:
    - Daemon start/stop controls with status indicators
    - One-time directory processing
    - Processing queue monitoring with progress
    - Statistics dashboard with key metrics (using StatusCards)
    - Real-time activity feed with severity filtering
    - Auto-scroll and log export functionality
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize Control Panel.

        Args:
            master: Parent widget
            thread_manager: ThreadManager instance for background operations
            main_app: Reference to main FelixApp for system access
            **kwargs: Additional arguments passed to CTkScrollableFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()

        # References to knowledge brain components (set by main_app)
        self.knowledge_daemon = None
        self.knowledge_retriever = None
        self.knowledge_store = None

        # Auto-refresh state
        self.auto_refresh_enabled = False
        self.refresh_interval = 5000  # 5 seconds

        # Activity log state
        self.auto_scroll_enabled = True

        self._setup_ui()

    def _setup_ui(self):
        """Set up the control panel UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)

        # Section 1: Daemon Control
        self._create_daemon_control_section()

        # Section 2: Processing Queue
        self._create_processing_queue_section()

        # Section 3: Statistics Dashboard
        self._create_statistics_section()

        # Section 4: Activity Feed
        self._create_activity_feed_section()

    def _create_daemon_control_section(self):
        """Create daemon control section with start/stop and processing buttons."""
        section_frame = ctk.CTkFrame(self, corner_radius=10)
        section_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        section_frame.grid_columnconfigure(0, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            section_frame,
            text="Daemon Control",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Button container
        button_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        button_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        button_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=0)
        button_frame.grid_columnconfigure(6, weight=1)  # Spacer

        # Start daemon button (green)
        self.start_daemon_btn = ctk.CTkButton(
            button_frame,
            text="▶ Start Daemon",
            command=self._start_daemon,
            fg_color=self.theme_manager.get_color("success"),
            hover_color="#1f8554",
            width=130
        )
        self.start_daemon_btn.grid(row=0, column=0, padx=(0, 8), pady=5)

        # Stop daemon button (red)
        self.stop_daemon_btn = ctk.CTkButton(
            button_frame,
            text="■ Stop Daemon",
            command=self._stop_daemon,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#b91c1c",
            width=130,
            state="disabled"
        )
        self.stop_daemon_btn.grid(row=0, column=1, padx=8, pady=5)

        # Separator
        separator = ctk.CTkFrame(button_frame, width=2, height=30, fg_color=self.theme_manager.get_color("border"))
        separator.grid(row=0, column=2, padx=15, pady=5)

        # Process directory button
        self.process_dir_btn = ctk.CTkButton(
            button_frame,
            text="📂 Process Directory",
            command=self._add_directory,
            width=160
        )
        self.process_dir_btn.grid(row=0, column=3, padx=(0, 8), pady=5)

        # Process pending button
        self.process_pending_btn = ctk.CTkButton(
            button_frame,
            text="⚡ Process Pending Now",
            command=self._process_pending_now,
            width=180
        )
        self.process_pending_btn.grid(row=0, column=4, padx=8, pady=5)

        # Manage directories button
        manage_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Manage Directories",
            command=self._manage_directories,
            width=170
        )
        manage_btn.grid(row=1, column=0, padx=(0, 8), pady=5, columnspan=2)

        # Force refinement button
        refine_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Force Refinement",
            command=self._force_refinement,
            width=160
        )
        refine_btn.grid(row=1, column=3, padx=(0, 8), pady=5)

        # Auto-refresh toggle
        self.auto_refresh_var = ctk.BooleanVar(value=False)
        auto_refresh_switch = ctk.CTkSwitch(
            button_frame,
            text="Auto-Refresh",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh
        )
        auto_refresh_switch.grid(row=1, column=4, padx=8, pady=5, sticky="w")

    def _create_processing_queue_section(self):
        """Create processing queue section showing current queue status."""
        section_frame = ctk.CTkFrame(self, corner_radius=10)
        section_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        section_frame.grid_columnconfigure(0, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            section_frame,
            text="Processing Queue",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Progress textbox
        self.progress_text = ctk.CTkTextbox(
            section_frame,
            height=80,
            font=ctk.CTkFont(family="SF Mono", size=12),
            wrap="word"
        )
        self.progress_text.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.progress_text.insert("1.0", "Processing queue status will appear here...\n")
        self.progress_text.configure(state="disabled")

    def _create_statistics_section(self):
        """Create statistics dashboard with StatusCards for key metrics."""
        section_frame = ctk.CTkFrame(self, corner_radius=10)
        section_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        section_frame.grid_columnconfigure(0, weight=1)

        # Section title
        header_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        header_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Statistics Dashboard",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(side="left")

        # Refresh button
        refresh_btn = ctk.CTkButton(
            header_frame,
            text="🔄 Refresh",
            command=self.refresh,
            width=100
        )
        refresh_btn.pack(side="right")

        # Status cards container
        cards_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Create status cards
        self.daemon_status_card = StatusCard(
            cards_frame,
            title="Daemon Status",
            value="Stopped",
            subtitle="Not running",
            width=180
        )
        self.daemon_status_card.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.documents_card = StatusCard(
            cards_frame,
            title="Total Documents",
            value="--",
            subtitle="Ingested sources",
            width=180
        )
        self.documents_card.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.concepts_card = StatusCard(
            cards_frame,
            title="Knowledge Entries",
            value="--",
            subtitle="Total concepts",
            width=180
        )
        self.concepts_card.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.relationships_card = StatusCard(
            cards_frame,
            title="Relationships",
            value="--",
            subtitle="Graph connections",
            width=180
        )
        self.relationships_card.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Second row of cards
        self.completed_card = StatusCard(
            cards_frame,
            title="Completed Docs",
            value="--",
            subtitle="Successfully processed",
            status_color=self.theme_manager.get_color("success"),
            width=180
        )
        self.completed_card.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.processing_card = StatusCard(
            cards_frame,
            title="Processing",
            value="--",
            subtitle="Currently processing",
            status_color=self.theme_manager.get_color("warning"),
            width=180
        )
        self.processing_card.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.failed_card = StatusCard(
            cards_frame,
            title="Failed",
            value="--",
            subtitle="Processing errors",
            status_color=self.theme_manager.get_color("error"),
            width=180
        )
        self.failed_card.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        self.high_confidence_card = StatusCard(
            cards_frame,
            title="High Confidence",
            value="--",
            subtitle="Reliable knowledge",
            width=180
        )
        self.high_confidence_card.grid(row=1, column=3, padx=5, pady=5, sticky="ew")

    def _create_activity_feed_section(self):
        """Create activity feed section with filtering and export."""
        section_frame = ctk.CTkFrame(self, corner_radius=10)
        section_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(10, 15))
        section_frame.grid_columnconfigure(0, weight=1)

        # Section title
        title_label = ctk.CTkLabel(
            section_frame,
            text="Activity Feed",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Controls
        controls_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 10))
        controls_frame.grid_columnconfigure(5, weight=1)  # Spacer

        # Refresh button
        refresh_btn = ctk.CTkButton(
            controls_frame,
            text="🔄 Refresh",
            command=self._refresh_activity,
            width=100
        )
        refresh_btn.grid(row=0, column=0, padx=(0, 8), pady=5)

        # Clear button
        clear_btn = ctk.CTkButton(
            controls_frame,
            text="🗑️ Clear Log",
            command=self._clear_activity_log,
            width=100
        )
        clear_btn.grid(row=0, column=1, padx=8, pady=5)

        # Export button
        export_btn = ctk.CTkButton(
            controls_frame,
            text="💾 Export Log",
            command=self._export_activity_log,
            width=110
        )
        export_btn.grid(row=0, column=2, padx=8, pady=5)

        # Separator
        separator = ctk.CTkFrame(controls_frame, width=2, height=30, fg_color=self.theme_manager.get_color("border"))
        separator.grid(row=0, column=3, padx=15, pady=5)

        # Filter label
        filter_label = ctk.CTkLabel(
            controls_frame,
            text="Filter:",
            font=ctk.CTkFont(size=12)
        )
        filter_label.grid(row=0, column=4, padx=(0, 8), pady=5)

        # Severity filter
        self.activity_filter_var = ctk.StringVar(value="All")
        filter_combo = ctk.CTkComboBox(
            controls_frame,
            variable=self.activity_filter_var,
            values=["All", "INFO", "WARNING", "ERROR"],
            command=lambda _: self._refresh_activity(),
            width=120,
            state="readonly"
        )
        filter_combo.grid(row=0, column=6, padx=8, pady=5)

        # Auto-scroll toggle
        self.auto_scroll_var = ctk.BooleanVar(value=True)
        auto_scroll_switch = ctk.CTkSwitch(
            controls_frame,
            text="Auto-scroll",
            variable=self.auto_scroll_var
        )
        auto_scroll_switch.grid(row=0, column=7, padx=(15, 0), pady=5)

        # Activity log textbox
        self.activity_text = ctk.CTkTextbox(
            section_frame,
            height=250,
            font=ctk.CTkFont(family="SF Mono", size=11),
            wrap="word"
        )
        self.activity_text.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.activity_text.insert("1.0", "Activity log will appear here...\n")
        self.activity_text.configure(state="disabled")

    # === Public Methods ===

    def set_knowledge_refs(self, knowledge_store=None, knowledge_retriever=None, knowledge_daemon=None):
        """
        Set references to knowledge brain components.

        Args:
            knowledge_store: KnowledgeStore instance
            knowledge_retriever: KnowledgeRetriever instance
            knowledge_daemon: KnowledgeDaemon instance
        """
        self.knowledge_store = knowledge_store
        self.knowledge_retriever = knowledge_retriever
        self.knowledge_daemon = knowledge_daemon

        # Refresh display with new references
        self.refresh()

    def _enable_features(self):
        """Enable features when Felix system starts."""
        felix_system = getattr(self.main_app, 'felix_system', None) if self.main_app else None
        if felix_system:
            # Wire up references to knowledge brain components
            self.knowledge_store = getattr(felix_system, 'knowledge_store', None)
            self.knowledge_retriever = getattr(felix_system, 'knowledge_retriever', None)
            self.knowledge_daemon = getattr(felix_system, 'knowledge_daemon', None)

            # Defer refresh to avoid blocking GUI thread during startup
            # Use background thread to run database queries
            self.after(500, self._async_initial_refresh)
            logger.info("Knowledge Brain Control Panel features enabled")

    def _async_initial_refresh(self):
        """Run initial refresh in background thread to avoid blocking GUI."""
        if self.thread_manager:
            self.thread_manager.start_thread(self._refresh_background)
        else:
            # Fallback to sync refresh if no thread_manager
            self.refresh()

    def _refresh_background(self):
        """
        Background thread: fetch all data then schedule UI updates on main thread.

        This prevents database queries from blocking the GUI.
        """
        try:
            # Fetch all data in background thread
            daemon_data = None
            doc_stats = None
            summary = None
            rel_count = None
            progress_data = None
            activity_data = None

            # Daemon status
            if self.knowledge_daemon:
                try:
                    daemon_data = self.knowledge_daemon.get_status()
                except Exception as e:
                    logger.debug(f"Could not get daemon status: {e}")

            # Document and knowledge statistics
            if self.knowledge_store:
                try:
                    doc_stats = self._get_document_stats()
                    summary = self.knowledge_store.get_knowledge_summary()
                except Exception as e:
                    logger.debug(f"Could not get stats: {e}")

                # Relationship count
                try:
                    conn = sqlite3.connect(self.knowledge_store.storage_path)
                    cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
                    rel_count = cursor.fetchone()[0]
                    conn.close()
                except Exception:
                    rel_count = None

                # Processing progress data
                try:
                    conn = sqlite3.connect(self.knowledge_store.storage_path)
                    cursor = conn.execute("""
                        SELECT ingestion_status, COUNT(*)
                        FROM document_sources
                        GROUP BY ingestion_status
                    """)
                    progress_data = dict(cursor.fetchall())
                    conn.close()
                except Exception as e:
                    logger.debug(f"Could not get progress data: {e}")

                # Activity data
                try:
                    conn = sqlite3.connect(self.knowledge_store.storage_path)
                    cursor = conn.execute("""
                        SELECT file_path, ingestion_completed, ingestion_status
                        FROM document_sources
                        WHERE ingestion_completed IS NOT NULL
                        ORDER BY ingestion_completed DESC
                        LIMIT 20
                    """)
                    activity_data = cursor.fetchall()
                    conn.close()
                except Exception as e:
                    logger.debug(f"Could not get activity data: {e}")

            # Schedule UI updates on main thread
            self.after(0, lambda: self._update_statistics_ui(daemon_data, doc_stats, summary, rel_count))
            self.after(0, lambda: self._update_progress_ui(progress_data))
            self.after(0, lambda: self._update_activity_ui(activity_data, daemon_data))

        except Exception as e:
            logger.error(f"Background refresh failed: {e}")

    def _update_statistics_ui(self, daemon_data, doc_stats, summary, rel_count):
        """Update statistics UI on main thread with pre-fetched data."""
        try:
            # Daemon status
            if daemon_data:
                if daemon_data.running:
                    uptime_hours = daemon_data.uptime_seconds / 3600
                    self.daemon_status_card.set_value("Running")
                    self.daemon_status_card.set_subtitle(f"Uptime: {uptime_hours:.1f}h")
                    self.daemon_status_card.set_status_color(self.theme_manager.get_color("success"))
                else:
                    self.daemon_status_card.set_value("Stopped")
                    self.daemon_status_card.set_subtitle("Not running")
                    if hasattr(self.daemon_status_card, 'status_bar') and self.daemon_status_card.status_bar:
                        self.daemon_status_card.status_bar.grid_remove()
            elif self.knowledge_daemon:
                self.daemon_status_card.set_value("Unknown")
                self.daemon_status_card.set_subtitle("Status unavailable")
            else:
                self.daemon_status_card.set_value("Not Init")
                self.daemon_status_card.set_subtitle("Not initialized")

            # Document statistics
            if doc_stats:
                self.documents_card.set_value(str(doc_stats.get('total', 0)))
                self.completed_card.set_value(str(doc_stats.get('completed', 0)))
                self.processing_card.set_value(str(doc_stats.get('processing', 0)))
                self.failed_card.set_value(str(doc_stats.get('failed', 0)))

            # Knowledge statistics
            if summary:
                self.concepts_card.set_value(str(summary.get('total_entries', 0)))
                self.high_confidence_card.set_value(str(summary.get('high_confidence_entries', 0)))

            # Relationship count
            if rel_count is not None:
                self.relationships_card.set_value(str(rel_count))
            else:
                self.relationships_card.set_value("--")

        except Exception as e:
            logger.error(f"Failed to update statistics UI: {e}")

    def _update_progress_ui(self, progress_data):
        """Update progress UI on main thread with pre-fetched data."""
        try:
            if progress_data is None:
                return

            pending = progress_data.get('pending', 0)
            processing = progress_data.get('processing', 0)
            completed = progress_data.get('completed', 0)
            failed = progress_data.get('failed', 0)
            total = sum(progress_data.values())

            # Build progress text
            progress_lines = []
            progress_lines.append(f"Total Documents: {total}")
            progress_lines.append(f"✓ Completed: {completed}")
            progress_lines.append(f"⚙ Processing: {processing}")
            progress_lines.append(f"⏳ Pending: {pending}")
            progress_lines.append(f"✗ Failed: {failed}")

            if total > 0:
                completion_pct = (completed / total) * 100
                progress_lines.append(f"\nCompletion: {completion_pct:.1f}%")

            progress_text = "\n".join(progress_lines)

            self.progress_text.configure(state="normal")
            self.progress_text.delete("1.0", "end")
            self.progress_text.insert("1.0", progress_text)
            self.progress_text.configure(state="disabled")

        except Exception as e:
            logger.error(f"Failed to update progress UI: {e}")

    def _update_activity_ui(self, activity_data, daemon_data):
        """Update activity UI on main thread with pre-fetched data."""
        try:
            filter_level = self.activity_filter_var.get()
            logs = []

            # Process activity data
            if activity_data:
                for row in activity_data:
                    file_path, ingestion_completed, status = row
                    filename = file_path.split('/')[-1] if '/' in file_path else file_path
                    timestamp = datetime.fromtimestamp(ingestion_completed).strftime('%H:%M:%S')

                    if status == 'completed':
                        level = "INFO"
                        message = f"Completed processing: {filename}"
                    elif status == 'failed':
                        level = "ERROR"
                        message = f"Failed to process: {filename}"
                    else:
                        level = "INFO"
                        message = f"Status {status}: {filename}"

                    if filter_level == "All" or filter_level == level:
                        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
                        logs.append(f"[{timestamp}] {emoji} {level}: {message}\n")

            # Add daemon status
            if daemon_data and filter_level in ["All", "INFO"]:
                timestamp = datetime.now().strftime('%H:%M:%S')
                if daemon_data.running:
                    logs.insert(0, f"[{timestamp}] ℹ️ INFO: Daemon is running (uptime: {daemon_data.uptime_seconds/3600:.1f}h)\n")
                    logs.insert(1, f"[{timestamp}] ℹ️ INFO: Pending in queue: {daemon_data.documents_pending}\n")
                else:
                    logs.insert(0, f"[{timestamp}] ⚠️ WARNING: Daemon is not running\n")

            # Update display
            self.activity_text.configure(state="normal")
            self.activity_text.delete("1.0", "end")

            if logs:
                for log in logs:
                    self.activity_text.insert("end", log)
            else:
                self.activity_text.insert("1.0", "No activity to display\n")

            if self.auto_scroll_var.get():
                self.activity_text.see("end")

            self.activity_text.configure(state="disabled")

        except Exception as e:
            logger.error(f"Failed to update activity UI: {e}")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        # Clear references
        self.knowledge_store = None
        self.knowledge_retriever = None
        self.knowledge_daemon = None

        # Update display to show disconnected state
        self.refresh()
        logger.info("Knowledge Brain Control Panel features disabled")

    def refresh(self):
        """Refresh all displays (statistics, queue, activity)."""
        self._refresh_statistics()
        self._refresh_processing_progress()
        self._refresh_activity()

    # === Daemon Control Handlers ===

    def _start_daemon(self):
        """Start the knowledge daemon."""
        # Check if Felix system is running
        if not self.main_app:
            messagebox.showwarning("Not Available", "Main app not available")
            return

        felix_system = getattr(self.main_app, 'felix_system', None)
        if not felix_system:
            messagebox.showwarning("Felix Not Running",
                "Please start Felix first:\n\n"
                "1. Go to the Dashboard tab\n"
                "2. Click the green 'Start Felix' button\n"
                "3. Wait for 'System Running' status\n"
                "4. Return here to start the daemon")
            return

        if not getattr(felix_system, 'running', False):
            messagebox.showwarning("Felix Not Running",
                "Felix system is not running.\n\n"
                "Go to Dashboard and click 'Start Felix'.")
            return

        try:
            # Get daemon reference
            daemon = getattr(felix_system, 'knowledge_daemon', None)
            if daemon:
                daemon.start()
                self.knowledge_daemon = daemon
                self.start_daemon_btn.configure(state="disabled")
                self.stop_daemon_btn.configure(state="normal")
                self._log_activity("Daemon started", "INFO")
                self.refresh()
            else:
                messagebox.showinfo("Knowledge Brain Not Enabled",
                    "The Knowledge Brain daemon is not available.\n\n"
                    "To enable it:\n"
                    "1. Go to Settings tab\n"
                    "2. Enable 'Knowledge Brain'\n"
                    "3. Stop and restart Felix")
                logger.info("Knowledge daemon not available - needs to be enabled in settings")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start daemon: {e}")
            logger.error(f"Failed to start daemon: {e}")

    def _stop_daemon(self):
        """Stop the knowledge daemon."""
        if self.knowledge_daemon:
            try:
                self.knowledge_daemon.stop()
                self.start_daemon_btn.configure(state="normal")
                self.stop_daemon_btn.configure(state="disabled")
                self._log_activity("Daemon stopped", "INFO")
                self.refresh()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop daemon: {e}")
                logger.error(f"Failed to stop daemon: {e}")

    def _add_directory(self):
        """Add a directory for one-time processing (not persistent watch)."""
        directory = filedialog.askdirectory(title="Select Directory with Documents")
        if directory:
            if self.knowledge_daemon:
                try:
                    result = self.knowledge_daemon.process_directory_now(directory)
                    messagebox.showinfo("Success",
                                       f"Queued {result.get('queued', 0)} documents from:\n{directory}")
                    self._log_activity(f"Processed directory once: {directory} ({result.get('queued', 0)} documents)", "INFO")
                    self.refresh()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process directory: {e}")
                    logger.error(f"Failed to process directory: {e}")
            else:
                messagebox.showwarning("Daemon Not Running", "Please start the daemon first")

    def _manage_directories(self):
        """Open dialog to manage watched directories."""
        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        # Create popup window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Manage Watch Directories")
        dialog.geometry("700x500")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        # Title
        title_label = ctk.CTkLabel(
            dialog,
            text="Currently Watched Directories",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(20, 10))

        # Listbox frame
        list_frame = ctk.CTkFrame(dialog)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Textbox for directory list (read-only, simulating listbox)
        dir_textbox = ctk.CTkTextbox(list_frame, font=ctk.CTkFont(size=12))
        dir_textbox.pack(fill="both", expand=True)

        # Populate with current directories
        current_dirs = self.knowledge_daemon.config.watch_directories.copy()
        for i, directory in enumerate(current_dirs):
            dir_textbox.insert("end", f"{i+1}. {directory}\n")
        dir_textbox.configure(state="disabled")

        # Info label
        info_label = ctk.CTkLabel(
            dialog,
            text=f"{len(current_dirs)} director{'y' if len(current_dirs) == 1 else 'ies'} configured",
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        def add_directory():
            """Add a directory to persistent watch list."""
            directory = filedialog.askdirectory(title="Select Directory to Watch", parent=dialog)
            if directory:
                try:
                    result = self.knowledge_daemon.add_watch_directory(directory)

                    if result.get('success'):
                        # Update textbox
                        dir_textbox.configure(state="normal")
                        dir_textbox.insert("end", f"{len(current_dirs)+1}. {directory}\n")
                        dir_textbox.configure(state="disabled")
                        current_dirs.append(directory)

                        # Save to persistent config
                        if hasattr(self.main_app, 'save_watch_directories'):
                            self.main_app.save_watch_directories()

                        messagebox.showinfo("Success",
                                           f"Added to persistent watch list:\n{directory}\n\n"
                                           "This directory will be monitored for changes.",
                                           parent=dialog)
                        self._log_activity(f"Added watch directory: {directory}", "INFO")
                        self.refresh()
                        info_label.configure(text=f"{len(current_dirs)} director{'y' if len(current_dirs) == 1 else 'ies'} configured")
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        messagebox.showwarning("Cannot Add Directory", error_msg, parent=dialog)

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to add watch directory: {e}", parent=dialog)
                    logger.error(f"Failed to add watch directory: {e}")

        add_btn = ctk.CTkButton(
            btn_frame,
            text="➕ Add Directory",
            command=add_directory,
            width=150
        )
        add_btn.pack(side="left", padx=5)

        close_btn = ctk.CTkButton(
            btn_frame,
            text="✖ Close",
            command=dialog.destroy,
            width=100
        )
        close_btn.pack(side="left", padx=5)

        # Center dialog
        dialog.update_idletasks()
        x = self.winfo_toplevel().winfo_x() + (self.winfo_toplevel().winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_toplevel().winfo_y() + (self.winfo_toplevel().winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _force_refinement(self):
        """Manually trigger a refinement cycle."""
        if self.knowledge_daemon:
            try:
                self._log_activity("Starting manual refinement...", "INFO")
                result = self.knowledge_daemon.trigger_refinement()
                messagebox.showinfo("Refinement Complete",
                                   f"Created {result.get('total_relationships', 0)} relationships")
                self._log_activity(f"Refinement complete: {result.get('total_relationships', 0)} relationships", "INFO")
                self.refresh()
            except Exception as e:
                messagebox.showerror("Error", f"Refinement failed: {e}")
                logger.error(f"Refinement failed: {e}")
        else:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")

    def _process_pending_now(self):
        """Manually trigger processing of all pending documents."""
        if not self.knowledge_daemon:
            messagebox.showwarning("Daemon Not Running", "Please start the daemon first")
            return

        try:
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
            self._log_activity(f"Manual trigger: Queued {pending_count} pending documents for processing", "INFO")

            # Refresh to show updated queue
            self.refresh()

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
            self.refresh()
            self.after(self.refresh_interval, self._auto_refresh_loop)

    # === Refresh Methods ===

    def _refresh_statistics(self):
        """Refresh statistics dashboard cards."""
        try:
            # Daemon status
            if self.knowledge_daemon:
                status = self.knowledge_daemon.get_status()

                if status.running:
                    uptime_hours = status.uptime_seconds / 3600
                    self.daemon_status_card.set_value("Running")
                    self.daemon_status_card.set_subtitle(f"Uptime: {uptime_hours:.1f}h")
                    self.daemon_status_card.set_status_color(self.theme_manager.get_color("success"))
                else:
                    self.daemon_status_card.set_value("Stopped")
                    self.daemon_status_card.set_subtitle("Not running")
                    if hasattr(self.daemon_status_card, 'status_bar') and self.daemon_status_card.status_bar:
                        self.daemon_status_card.status_bar.grid_remove()
            else:
                self.daemon_status_card.set_value("Not Init")
                self.daemon_status_card.set_subtitle("Not initialized")

            # Document statistics
            if self.knowledge_store:
                doc_stats = self._get_document_stats()
                self.documents_card.set_value(str(doc_stats.get('total', 0)))
                self.completed_card.set_value(str(doc_stats.get('completed', 0)))
                self.processing_card.set_value(str(doc_stats.get('processing', 0)))
                self.failed_card.set_value(str(doc_stats.get('failed', 0)))

                # Knowledge statistics
                summary = self.knowledge_store.get_knowledge_summary()
                self.concepts_card.set_value(str(summary.get('total_entries', 0)))
                self.high_confidence_card.set_value(str(summary.get('high_confidence_entries', 0)))

                # Relationship count (if available)
                try:
                    conn = sqlite3.connect(self.knowledge_store.storage_path)
                    cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
                    rel_count = cursor.fetchone()[0]
                    conn.close()
                    self.relationships_card.set_value(str(rel_count))
                except:
                    self.relationships_card.set_value("--")

        except Exception as e:
            logger.error(f"Failed to refresh statistics: {e}")

    def _refresh_processing_progress(self):
        """Update the processing progress section with queue statistics."""
        try:
            if self.knowledge_store is None:
                return

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
            total = sum(status_counts.values())

            conn.close()

            # Build progress text
            progress_lines = []
            progress_lines.append(f"Total Documents: {total}")
            progress_lines.append(f"✓ Completed: {completed}")
            progress_lines.append(f"⚙ Processing: {processing}")
            progress_lines.append(f"⏳ Pending: {pending}")
            progress_lines.append(f"✗ Failed: {failed}")

            # Calculate completion percentage
            if total > 0:
                completion_pct = (completed / total) * 100
                progress_lines.append(f"\nCompletion: {completion_pct:.1f}%")

            progress_text = "\n".join(progress_lines)

            # Update textbox
            self.progress_text.configure(state="normal")
            self.progress_text.delete("1.0", "end")
            self.progress_text.insert("1.0", progress_text)
            self.progress_text.configure(state="disabled")

        except Exception as e:
            logger.error(f"Failed to refresh processing progress: {e}")

    def _refresh_activity(self):
        """Refresh activity log with daemon logs and apply filtering."""
        try:
            if self.knowledge_store is None:
                return

            # Get filter selection
            filter_level = self.activity_filter_var.get()

            # Build log display
            logs = []

            # Add recent document processing events
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            cursor = conn.execute("""
                SELECT file_path, ingestion_completed, ingestion_status
                FROM document_sources
                WHERE ingestion_completed IS NOT NULL
                ORDER BY ingestion_completed DESC
                LIMIT 20
            """)

            for row in cursor.fetchall():
                file_path, ingestion_completed, status = row
                filename = file_path.split('/')[-1] if '/' in file_path else file_path
                timestamp = datetime.fromtimestamp(ingestion_completed).strftime('%H:%M:%S')

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
            self.activity_text.configure(state="normal")
            self.activity_text.delete("1.0", "end")

            if logs:
                for log in logs:
                    self.activity_text.insert("end", log)
            else:
                self.activity_text.insert("1.0", "No activity to display\n")

            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.activity_text.see("end")

            self.activity_text.configure(state="disabled")

        except Exception as e:
            logger.error(f"Failed to refresh activity log: {e}")
            self._log_activity(f"Error refreshing activity: {str(e)}", "ERROR")

    def _clear_activity_log(self):
        """Clear activity log."""
        self.activity_text.configure(state="normal")
        self.activity_text.delete("1.0", "end")
        self.activity_text.insert("1.0", "Activity log cleared.\n")
        self.activity_text.configure(state="disabled")

    def _export_activity_log(self):
        """Export activity log to a text file."""
        try:
            # Get current log content
            log_content = self.activity_text.get("1.0", "end-1c")

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

        self.activity_text.configure(state="normal")
        self.activity_text.insert("end", log_message)

        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.activity_text.see("end")

        self.activity_text.configure(state="disabled")

    # === Helper Methods ===

    def _get_document_stats(self) -> dict:
        """Get document statistics."""
        try:
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
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {}
