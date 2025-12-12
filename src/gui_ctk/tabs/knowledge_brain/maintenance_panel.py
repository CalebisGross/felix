"""
Maintenance Panel for Knowledge Brain GUI.

Provides comprehensive maintenance tools including:
- Quality monitoring (duplicates, low confidence, orphaned concepts)
- Audit trail (CRUD operation history with filtering)
- Cleanup operations (pattern-based, domain, date range, bulk operations)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import csv
import json
import re

from ...components.themed_treeview import ThemedTreeview
from ...components.status_card import StatusCard
from ...theme_manager import get_theme_manager
from ...styles import (
    BUTTON_SM, BUTTON_MD, BUTTON_LG,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    CARD_MD, TEXTBOX_MD
)

logger = logging.getLogger(__name__)


class MaintenancePanel(ctk.CTkFrame):
    """
    Maintenance panel with Quality, Audit, and Cleanup sub-tabs.

    Modern Apple-like design with comprehensive maintenance tools for
    the Knowledge Brain system.
    """

    def __init__(
        self,
        master,
        thread_manager=None,
        main_app=None,
        **kwargs
    ):
        """
        Initialize Maintenance Panel.

        Args:
            master: Parent widget
            thread_manager: Thread manager for background operations
            main_app: Reference to main application for system access
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()

        # References to knowledge brain components (set via set_knowledge_refs)
        self.knowledge_store = None
        self.knowledge_retriever = None
        self.knowledge_daemon = None

        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main tabview with 3 sub-tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=SPACE_MD, pady=SPACE_MD)

        # Add tabs
        self.tabview.add("Quality")
        self.tabview.add("Audit")
        self.tabview.add("Cleanup")

        # Create each sub-tab
        self._create_quality_tab()
        self._create_audit_tab()
        self._create_cleanup_tab()

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

        # Don't auto-refresh on init - quality checks are expensive O(n²) operations
        # that block the GUI thread. User can manually refresh via the Quality tab.

    # ==================== Quality Sub-tab ====================

    def _create_quality_tab(self):
        """Create Quality sub-tab for monitoring knowledge quality."""
        quality_frame = self.tabview.tab("Quality")

        # Configure grid
        quality_frame.grid_columnconfigure(0, weight=1)
        quality_frame.grid_rowconfigure(1, weight=1)

        # Header with title and refresh button
        header = ctk.CTkFrame(quality_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header,
            text="Knowledge Quality Monitor",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        refresh_btn = ctk.CTkButton(
            header,
            text="Refresh",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._refresh_quality_stats
        )
        refresh_btn.grid(row=0, column=2, sticky="e", padx=(SPACE_SM, 0))

        # Main content area with scrolling
        content = ctk.CTkScrollableFrame(quality_frame)
        content.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        content.grid_columnconfigure(0, weight=1)

        # Quality scores section
        scores_label = ctk.CTkLabel(
            content,
            text="Quality Metrics",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold"),
            anchor="w"
        )
        scores_label.grid(row=0, column=0, sticky="w", pady=(0, SPACE_SM))

        # Status cards container
        cards_frame = ctk.CTkFrame(content, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="ew", pady=(0, SPACE_MD))

        # Create status cards
        self.quality_cards = {}
        card_configs = [
            ("duplicates", "Potential Duplicates", "0", "", self.theme_manager.get_color("warning")),
            ("low_confidence", "Low Confidence", "0", "entries below 0.5", self.theme_manager.get_color("error")),
            ("orphaned", "Orphaned Concepts", "0", "no relationships", self.theme_manager.get_color("warning")),
            ("overall_score", "Overall Quality", "0%", "health score", self.theme_manager.get_color("success"))
        ]

        for i, (key, title, value, subtitle, color) in enumerate(card_configs):
            card = StatusCard(
                cards_frame,
                title=title,
                value=value,
                subtitle=subtitle,
                status_color=color,
                width=CARD_MD
            )
            card.grid(row=0, column=i, padx=SPACE_XS, sticky="ew")
            cards_frame.grid_columnconfigure(i, weight=1)
            self.quality_cards[key] = card

        # Duplicates section
        duplicates_label = ctk.CTkLabel(
            content,
            text="Potential Duplicates",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold"),
            anchor="w"
        )
        duplicates_label.grid(row=2, column=0, sticky="w", pady=(SPACE_SM, SPACE_XS))

        # Duplicates tree
        self.duplicates_tree = ThemedTreeview(
            content,
            columns=["Entry 1", "Entry 2", "Similarity", "Type", "Action"],
            headings=["Entry 1", "Entry 2", "Similarity", "Type", "Suggested Action"],
            widths=[200, 200, 100, 120, 150],
            height=8
        )
        self.duplicates_tree.grid(row=3, column=0, sticky="ew", pady=(0, SPACE_SM))

        # Low confidence section
        low_conf_label = ctk.CTkLabel(
            content,
            text="Low Confidence Entries",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold"),
            anchor="w"
        )
        low_conf_label.grid(row=4, column=0, sticky="w", pady=(SPACE_SM, SPACE_XS))

        # Low confidence tree
        self.low_confidence_tree = ThemedTreeview(
            content,
            columns=["ID", "Content", "Confidence", "Domain", "Created"],
            headings=["Entry ID", "Content Preview", "Confidence", "Domain", "Created Date"],
            widths=[200, 250, 100, 120, 130],
            height=8
        )
        self.low_confidence_tree.grid(row=5, column=0, sticky="ew", pady=(0, SPACE_SM))

        # Actions section
        actions_frame = ctk.CTkFrame(content, fg_color="transparent")
        actions_frame.grid(row=6, column=0, sticky="ew", pady=(SPACE_SM, 0))

        ctk.CTkButton(
            actions_frame,
            text="Run Full Quality Check",
            command=self._run_quality_check,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        ).pack(side="left", padx=SPACE_XS)

        ctk.CTkButton(
            actions_frame,
            text="Generate Quality Report",
            command=self._generate_quality_report,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        ).pack(side="left", padx=SPACE_XS)

        ctk.CTkButton(
            actions_frame,
            text="View Orphaned Concepts",
            command=self._view_orphaned_concepts,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        ).pack(side="left", padx=SPACE_XS)

    def _refresh_quality_stats(self):
        """Refresh quality statistics and update display."""
        if not self.knowledge_store:
            self._show_warning("Quality Check", "Felix system not running. Start Felix to check quality.")
            return

        try:
            from src.knowledge.quality_checker import QualityChecker

            checker = QualityChecker(self.knowledge_store)

            # Get duplicate candidates
            duplicates = checker.find_duplicates(similarity_threshold=0.85)
            self.quality_cards["duplicates"].set_value(str(len(duplicates)))

            # Update duplicates tree
            self.duplicates_tree.clear()
            for dup in duplicates[:20]:  # Show first 20
                self.duplicates_tree.insert(
                    values=(
                        dup.entry1_id[:30] + "...",
                        dup.entry2_id[:30] + "...",
                        f"{dup.similarity_score:.2f}",
                        dup.similarity_type,
                        dup.suggested_action
                    )
                )

            # Get low confidence entries
            low_conf = self._get_low_confidence_entries()
            self.quality_cards["low_confidence"].set_value(str(len(low_conf)))

            # Update low confidence tree
            self.low_confidence_tree.clear()
            for entry in low_conf[:20]:  # Show first 20
                content_preview = entry.get("content", "")[:50] + "..."
                self.low_confidence_tree.insert(
                    values=(
                        entry.get("knowledge_id", "")[:30] + "...",
                        content_preview,
                        f"{entry.get('confidence', 0.0):.2f}",
                        entry.get("domain", ""),
                        entry.get("timestamp", "")[:10]
                    )
                )

            # Get orphaned concepts
            orphaned = self._get_orphaned_concepts()
            self.quality_cards["orphaned"].set_value(str(len(orphaned)))

            # Calculate overall quality score
            total_entries = self._get_total_entries_count()
            if total_entries > 0:
                quality_score = (
                    (1 - len(duplicates) / max(total_entries, 1)) * 0.3 +
                    (1 - len(low_conf) / max(total_entries, 1)) * 0.4 +
                    (1 - len(orphaned) / max(total_entries, 1)) * 0.3
                )
                self.quality_cards["overall_score"].set_value(f"{quality_score * 100:.0f}%")

                # Update color based on score
                if quality_score >= 0.8:
                    color = self.theme_manager.get_color("success")
                elif quality_score >= 0.6:
                    color = self.theme_manager.get_color("warning")
                else:
                    color = self.theme_manager.get_color("error")
                self.quality_cards["overall_score"].set_status_color(color)

            logger.info("Quality statistics refreshed successfully")

        except Exception as e:
            logger.error(f"Error refreshing quality stats: {e}")
            self._show_error("Quality Check Error", f"Failed to refresh quality stats: {str(e)}")

    def _get_low_confidence_entries(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Get entries with low confidence level (LOW or SPECULATIVE)."""
        if not self.knowledge_store:
            return []

        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # confidence_level is TEXT enum: 'VERIFIED', 'HIGH', 'MEDIUM', 'LOW', 'SPECULATIVE'
            cursor.execute("""
                SELECT knowledge_id, content_json, confidence_level, domain, created_at
                FROM knowledge_entries
                WHERE confidence_level IN ('LOW', 'SPECULATIVE')
                ORDER BY created_at DESC
                LIMIT 100
            """)

            entries = []
            for row in cursor.fetchall():
                # Parse content_json to get displayable content
                try:
                    content_data = json.loads(row[1]) if row[1] else {}
                    content = content_data.get("concept", content_data.get("content", str(row[1])[:100]))
                except (json.JSONDecodeError, TypeError):
                    content = str(row[1])[:100] if row[1] else ""

                # Convert Unix timestamp to ISO string
                created_at = row[4]
                if isinstance(created_at, (int, float)):
                    from datetime import datetime
                    created_at = datetime.fromtimestamp(created_at).isoformat()

                entries.append({
                    "knowledge_id": row[0],
                    "content": content,
                    "confidence": row[2],  # This is the confidence_level text
                    "domain": row[3] or "general",
                    "timestamp": created_at
                })

            conn.close()
            return entries

        except Exception as e:
            logger.error(f"Error getting low confidence entries: {e}")
            return []

    def _get_orphaned_concepts(self) -> List[Dict[str, Any]]:
        """Get concepts with no relationships."""
        if not self.knowledge_store:
            return []

        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Correct column names: source_id and target_id (not from_concept_id/to_concept_id)
            cursor.execute("""
                SELECT k.knowledge_id, k.content_json, k.domain
                FROM knowledge_entries k
                LEFT JOIN knowledge_relationships r1 ON k.knowledge_id = r1.source_id
                LEFT JOIN knowledge_relationships r2 ON k.knowledge_id = r2.target_id
                WHERE r1.source_id IS NULL AND r2.target_id IS NULL
                LIMIT 100
            """)

            orphaned = []
            for row in cursor.fetchall():
                # Parse content_json to get displayable content
                try:
                    content_data = json.loads(row[1]) if row[1] else {}
                    content = content_data.get("concept", content_data.get("content", str(row[1])[:100]))
                except (json.JSONDecodeError, TypeError):
                    content = str(row[1])[:100] if row[1] else ""

                orphaned.append({
                    "knowledge_id": row[0],
                    "content": content,
                    "domain": row[2] or "general"
                })

            conn.close()
            return orphaned

        except Exception as e:
            logger.error(f"Error getting orphaned concepts: {e}")
            return []

    def _get_total_entries_count(self) -> int:
        """Get total number of knowledge entries."""
        if not self.knowledge_store:
            return 0

        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            count = cursor.fetchone()[0]

            conn.close()
            return count

        except Exception as e:
            logger.error(f"Error getting total entries count: {e}")
            return 0

    def _run_quality_check(self):
        """Run comprehensive quality check in background."""
        if not self.knowledge_store:
            self._show_warning("Quality Check", "Felix system not running")
            return

        # Show progress dialog
        self._show_info("Quality Check", "Running comprehensive quality check...\nThis may take a few moments.")

        # Run in background thread
        if self.thread_manager:
            self.thread_manager.run_in_thread(
                self._quality_check_worker,
                on_complete=self._quality_check_complete
            )
        else:
            self._refresh_quality_stats()

    def _quality_check_worker(self):
        """Worker function for quality check."""
        try:
            from src.knowledge.quality_checker import QualityChecker

            checker = QualityChecker(self.knowledge_store)

            # Run checks
            duplicates = checker.find_duplicates(similarity_threshold=0.85)
            contradictions = checker.find_contradictions() if hasattr(checker, 'find_contradictions') else []

            return {
                "success": True,
                "duplicates": len(duplicates),
                "contradictions": len(contradictions)
            }

        except Exception as e:
            logger.error(f"Quality check error: {e}")
            return {"success": False, "error": str(e)}

    def _quality_check_complete(self, result):
        """Handle quality check completion."""
        if result.get("success"):
            self._show_info(
                "Quality Check Complete",
                f"Found:\n"
                f"- {result['duplicates']} potential duplicates\n"
                f"- {result.get('contradictions', 0)} potential contradictions"
            )
            self._refresh_quality_stats()
        else:
            self._show_error("Quality Check Failed", result.get("error", "Unknown error"))

    def _generate_quality_report(self):
        """Generate and export quality report."""
        if not self.knowledge_store:
            self._show_warning("Quality Report", "Felix system not running")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown files", "*.md"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            from src.knowledge.quality_checker import QualityChecker

            checker = QualityChecker(self.knowledge_store)

            # Generate report
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Knowledge Base Quality Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Summary stats
                total = self._get_total_entries_count()
                duplicates = len(checker.find_duplicates(similarity_threshold=0.85))
                low_conf = len(self._get_low_confidence_entries())
                orphaned = len(self._get_orphaned_concepts())

                f.write("## Summary Statistics\n\n")
                f.write(f"- Total Entries: {total}\n")
                f.write(f"- Potential Duplicates: {duplicates}\n")
                f.write(f"- Low Confidence Entries: {low_conf}\n")
                f.write(f"- Orphaned Concepts: {orphaned}\n\n")

                # Quality score
                if total > 0:
                    quality_score = (
                        (1 - duplicates / max(total, 1)) * 0.3 +
                        (1 - low_conf / max(total, 1)) * 0.4 +
                        (1 - orphaned / max(total, 1)) * 0.3
                    )
                    f.write(f"## Overall Quality Score: {quality_score * 100:.1f}%\n\n")

                f.write("## Recommendations\n\n")
                if duplicates > 0:
                    f.write(f"- Review and merge {duplicates} potential duplicate entries\n")
                if low_conf > 0:
                    f.write(f"- Validate or remove {low_conf} low confidence entries\n")
                if orphaned > 0:
                    f.write(f"- Create relationships for {orphaned} orphaned concepts\n")

                if duplicates == 0 and low_conf == 0 and orphaned == 0:
                    f.write("- Knowledge base is in good health!\n")

            self._show_info("Quality Report", f"Report saved to:\n{filename}")
            logger.info(f"Quality report saved to {filename}")

        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            self._show_error("Report Error", f"Failed to generate report: {str(e)}")

    def _view_orphaned_concepts(self):
        """Show dialog with orphaned concepts details."""
        if not self.knowledge_store:
            self._show_warning("Orphaned Concepts", "Felix system not running")
            return

        orphaned = self._get_orphaned_concepts()

        if not orphaned:
            self._show_info("Orphaned Concepts", "No orphaned concepts found!")
            return

        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Orphaned Concepts")
        dialog.geometry("600x400")

        # Make modal
        dialog.transient(self)
        dialog.after(100, lambda: self._safe_grab(dialog))

        # Title
        title = ctk.CTkLabel(
            dialog,
            text=f"Found {len(orphaned)} Orphaned Concepts",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title.pack(pady=SPACE_SM)

        # List
        tree = ThemedTreeview(
            dialog,
            columns=["ID", "Content", "Domain"],
            headings=["Entry ID", "Content Preview", "Domain"],
            widths=[200, 250, 120],
            height=15
        )
        tree.pack(fill="both", expand=True, padx=SPACE_SM, pady=SPACE_SM)

        for entry in orphaned:
            content_preview = entry["content"][:60] + "..."
            tree.insert(
                values=(
                    entry["knowledge_id"][:30] + "...",
                    content_preview,
                    entry["domain"]
                )
            )

        # Close button
        ctk.CTkButton(
            dialog,
            text="Close",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=dialog.destroy
        ).pack(pady=SPACE_SM)

    # ==================== Audit Sub-tab ====================

    def _create_audit_tab(self):
        """Create Audit sub-tab for CRUD operation history."""
        audit_frame = self.tabview.tab("Audit")

        # Configure grid
        audit_frame.grid_columnconfigure(0, weight=1)
        audit_frame.grid_rowconfigure(2, weight=1)

        # Header
        header = ctk.CTkFrame(audit_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        title_label = ctk.CTkLabel(
            header,
            text="Audit Log - Knowledge Base Operations",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title_label.pack(side="left")

        # Filters section
        filters_frame = ctk.CTkFrame(audit_frame)
        filters_frame.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_XS)
        filters_frame.grid_columnconfigure(1, weight=1)

        # Row 1: Operation type and User/Agent
        row1 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", pady=SPACE_XS, padx=SPACE_SM)

        ctk.CTkLabel(row1, text="Operation:").pack(side="left", padx=(0, SPACE_XS))
        self.audit_operation_var = tk.StringVar(value="ALL")
        operation_combo = ctk.CTkComboBox(
            row1,
            variable=self.audit_operation_var,
            values=["ALL", "INSERT", "UPDATE", "DELETE", "MERGE", "CLEANUP", "SYSTEM"],
            width=BUTTON_MD[0]
        )
        operation_combo.pack(side="left", padx=SPACE_XS)

        ctk.CTkLabel(row1, text="User/Agent:").pack(side="left", padx=(SPACE_MD, SPACE_XS))
        self.audit_user_var = tk.StringVar()
        user_entry = ctk.CTkEntry(row1, textvariable=self.audit_user_var, width=TEXTBOX_MD)
        user_entry.pack(side="left", padx=SPACE_XS)

        # Row 2: Knowledge ID filter
        row2 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", pady=SPACE_XS, padx=SPACE_SM)

        ctk.CTkLabel(row2, text="Knowledge ID:").pack(side="left", padx=(0, SPACE_XS))
        self.audit_knowledge_id_var = tk.StringVar()
        knowledge_id_entry = ctk.CTkEntry(row2, textvariable=self.audit_knowledge_id_var, width=TEXTBOX_MD * 2)
        knowledge_id_entry.pack(side="left", padx=SPACE_XS)

        # Row 3: Action buttons
        row3 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row3.grid(row=2, column=0, sticky="ew", pady=SPACE_XS, padx=SPACE_SM)

        ctk.CTkButton(row3, text="Search", width=BUTTON_SM[0], height=BUTTON_SM[1], command=self._search_audit_log).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(row3, text="Refresh", width=BUTTON_SM[0], height=BUTTON_SM[1], command=self._refresh_audit_log).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(row3, text="Export CSV", width=BUTTON_MD[0], height=BUTTON_MD[1], command=self._export_audit_log).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(row3, text="Clear Filters", width=BUTTON_MD[0], height=BUTTON_MD[1], command=self._clear_audit_filters).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(row3, text="Clear Old Entries", width=BUTTON_LG[0], height=BUTTON_LG[1], command=self._clear_old_audit_entries).pack(side="left", padx=SPACE_XS)

        # Audit log tree
        self.audit_tree = ThemedTreeview(
            audit_frame,
            columns=["Timestamp", "Operation", "Target", "Details", "User"],
            headings=["Timestamp", "Operation", "Knowledge ID", "Details", "User/Agent"],
            widths=[150, 100, 200, 250, 120],
            height=15
        )
        self.audit_tree.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)

        # Bind double-click to view details
        self.audit_tree.bind_tree("<Double-1>", self._view_audit_details)

        # Statistics section
        stats_frame = ctk.CTkFrame(audit_frame)
        stats_frame.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        self.audit_stats_label = ctk.CTkLabel(
            stats_frame,
            text="Statistics: Loading...",
            font=ctk.CTkFont(size=FONT_CAPTION)
        )
        self.audit_stats_label.pack(pady=SPACE_SM)

    def _refresh_audit_log(self):
        """Refresh audit log display."""
        if not self.knowledge_store:
            self.audit_tree.clear()
            self.audit_stats_label.configure(text="Statistics: Felix system not running")
            return

        try:
            import sqlite3
            from src.memory.audit_log import AuditLogger

            audit_logger = AuditLogger(self.knowledge_store.storage_path)

            # Build query based on filters
            query = """
                SELECT timestamp, operation, knowledge_id, details, user_agent
                FROM knowledge_audit_log
                WHERE 1=1
            """
            params = []

            # Apply filters
            if self.audit_operation_var.get() != "ALL":
                query += " AND operation = ?"
                params.append(self.audit_operation_var.get())

            if self.audit_user_var.get():
                query += " AND user_agent LIKE ?"
                params.append(f"%{self.audit_user_var.get()}%")

            if self.audit_knowledge_id_var.get():
                query += " AND knowledge_id LIKE ?"
                params.append(f"%{self.audit_knowledge_id_var.get()}%")

            query += " ORDER BY timestamp DESC LIMIT 500"

            # Execute query
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()
            cursor.execute(query, params)

            # Clear and populate tree
            self.audit_tree.clear()
            for row in cursor.fetchall():
                timestamp, operation, knowledge_id, details, user_agent = row

                # Format timestamp (handles both Unix float and ISO string formats)
                try:
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp)
                    else:
                        dt = datetime.fromisoformat(str(timestamp))
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    timestamp_str = str(timestamp)[:19] if timestamp else "N/A"

                # Truncate long values
                knowledge_id_str = (knowledge_id[:30] + "...") if knowledge_id and len(knowledge_id) > 30 else (knowledge_id or "")
                details_str = (details[:50] + "...") if details and len(details) > 50 else (details or "")

                self.audit_tree.insert(
                    values=(
                        timestamp_str,
                        operation,
                        knowledge_id_str,
                        details_str,
                        user_agent or "System"
                    )
                )

            # Update statistics
            cursor.execute("SELECT COUNT(*) FROM knowledge_audit_log")
            total_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT operation, COUNT(*)
                FROM knowledge_audit_log
                GROUP BY operation
            """)
            operation_counts = dict(cursor.fetchall())

            conn.close()

            stats_text = f"Total Entries: {total_count} | "
            stats_text += " | ".join([f"{op}: {count}" for op, count in operation_counts.items()])
            self.audit_stats_label.configure(text=f"Statistics: {stats_text}")

            logger.info("Audit log refreshed successfully")

        except Exception as e:
            logger.error(f"Error refreshing audit log: {e}")
            self.audit_tree.clear()
            self.audit_stats_label.configure(text=f"Error: {str(e)}")

    def _search_audit_log(self):
        """Apply filters and refresh audit log."""
        self._refresh_audit_log()

    def _clear_audit_filters(self):
        """Clear all audit filters."""
        self.audit_operation_var.set("ALL")
        self.audit_user_var.set("")
        self.audit_knowledge_id_var.set("")
        self._refresh_audit_log()

    def _export_audit_log(self):
        """Export audit log to CSV file."""
        if not self.knowledge_store:
            self._show_warning("Export Audit Log", "Felix system not running")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, operation, knowledge_id, details, user_agent,
                       old_values, new_values, transaction_id
                FROM knowledge_audit_log
                ORDER BY timestamp DESC
            """)

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Operation", "Knowledge ID", "Details",
                    "User/Agent", "Old Values", "New Values", "Transaction ID"
                ])
                writer.writerows(cursor.fetchall())

            conn.close()

            self._show_info("Export Complete", f"Audit log exported to:\n{filename}")
            logger.info(f"Audit log exported to {filename}")

        except Exception as e:
            logger.error(f"Error exporting audit log: {e}")
            self._show_error("Export Error", f"Failed to export audit log: {str(e)}")

    def _clear_old_audit_entries(self):
        """Clear audit entries older than specified days."""
        if not self.knowledge_store:
            self._show_warning("Clear Old Entries", "Felix system not running")
            return

        # Ask for confirmation with days input
        dialog = ctk.CTkInputDialog(
            text="Delete audit entries older than how many days?",
            title="Clear Old Audit Entries"
        )
        days_str = dialog.get_input()

        if not days_str:
            return

        try:
            days = int(days_str)
            if days < 1:
                self._show_warning("Invalid Input", "Days must be at least 1")
                return

            # Confirm
            result = messagebox.askyesno(
                "Confirm Deletion",
                f"This will permanently delete audit entries older than {days} days.\n\n"
                f"Continue?"
            )

            if not result:
                return

            import sqlite3

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM knowledge_audit_log
                WHERE timestamp < ?
            """, (cutoff_date,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            self._show_info(
                "Cleanup Complete",
                f"Deleted {deleted_count} audit entries older than {days} days"
            )

            self._refresh_audit_log()
            logger.info(f"Cleared {deleted_count} old audit entries (older than {days} days)")

        except ValueError:
            self._show_warning("Invalid Input", "Please enter a valid number of days")
        except Exception as e:
            logger.error(f"Error clearing old audit entries: {e}")
            self._show_error("Cleanup Error", f"Failed to clear old entries: {str(e)}")

    def _view_audit_details(self, event=None):
        """Show detailed view of selected audit entry."""
        selection = self.audit_tree.selection()
        if not selection:
            return

        # Get full entry data
        item_values = self.audit_tree.item(selection[0])["values"]

        # Create dialog
        dialog = ctk.CTkToplevel(self)
        dialog.title("Audit Entry Details")
        dialog.geometry("600x500")

        # Make modal
        dialog.transient(self)
        dialog.after(100, lambda: self._safe_grab(dialog))

        # Title
        title = ctk.CTkLabel(
            dialog,
            text="Audit Entry Details",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title.pack(pady=SPACE_SM)

        # Details text
        details_frame = ctk.CTkScrollableFrame(dialog)
        details_frame.pack(fill="both", expand=True, padx=SPACE_SM, pady=SPACE_SM)

        details_text = ctk.CTkTextbox(details_frame, wrap="word")
        details_text.pack(fill="both", expand=True)

        # Format and display details
        details_content = f"""Timestamp: {item_values[0]}
Operation: {item_values[1]}
Knowledge ID: {item_values[2]}
User/Agent: {item_values[4]}

Details:
{item_values[3]}

(Full entry data available in CSV export)
"""
        details_text.insert("1.0", details_content)
        details_text.configure(state="disabled")

        # Close button
        ctk.CTkButton(
            dialog,
            text="Close",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=dialog.destroy
        ).pack(pady=SPACE_SM)

    # ==================== Cleanup Sub-tab ====================

    def _create_cleanup_tab(self):
        """Create Cleanup sub-tab for maintenance operations."""
        cleanup_frame = self.tabview.tab("Cleanup")

        # Configure grid
        cleanup_frame.grid_columnconfigure(0, weight=1)
        cleanup_frame.grid_rowconfigure(0, weight=1)

        # Scrollable content
        content = ctk.CTkScrollableFrame(cleanup_frame)
        content.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        content.grid_columnconfigure(0, weight=1)

        # ===== SYSTEM RESET SECTION (at top) =====
        self._create_system_reset_section(content, row=0)

        # Title for knowledge-specific cleanup
        title_label = ctk.CTkLabel(
            content,
            text="Knowledge Base Cleanup",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        title_label.grid(row=1, column=0, sticky="w", pady=(SPACE_MD, SPACE_MD))

        # Pattern-based cleanup section
        pattern_frame = ctk.CTkFrame(content)
        pattern_frame.grid(row=2, column=0, sticky="ew", pady=(0, SPACE_MD))
        pattern_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            pattern_frame,
            text="Pattern-Based Deletion",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        ctk.CTkLabel(
            pattern_frame,
            text="Path Pattern (glob or SQL LIKE):",
            font=ctk.CTkFont(size=FONT_BODY)
        ).grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_XS))

        self.pattern_entry = ctk.CTkEntry(pattern_frame, placeholder_text="*/test_data/*")
        self.pattern_entry.grid(row=2, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_SM))

        # Cascade option
        self.cascade_var = tk.BooleanVar(value=True)
        cascade_check = ctk.CTkCheckBox(
            pattern_frame,
            text="Delete associated knowledge entries",
            variable=self.cascade_var
        )
        cascade_check.grid(row=3, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_SM))

        # Pattern buttons
        pattern_buttons = ctk.CTkFrame(pattern_frame, fg_color="transparent")
        pattern_buttons.grid(row=4, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_SM))

        ctk.CTkButton(
            pattern_buttons,
            text="Preview Pattern",
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            command=self._preview_custom_pattern
        ).pack(side="left", padx=(0, SPACE_XS))

        ctk.CTkButton(
            pattern_buttons,
            text="Clean Pattern",
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            command=self._execute_custom_pattern,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).pack(side="left", padx=SPACE_XS)

        # Common patterns
        common_frame = ctk.CTkFrame(pattern_frame, fg_color="transparent")
        common_frame.grid(row=5, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_MD))

        ctk.CTkLabel(common_frame, text="Quick patterns:").pack(side="left", padx=(0, SPACE_SM))

        for pattern_name, pattern in [
            (".venv", "*/.venv/*"),
            ("node_modules", "*/node_modules/*"),
            ("__pycache__", "*/__pycache__/*"),
            ("test_data", "*/test_data/*")
        ]:
            ctk.CTkButton(
                common_frame,
                text=pattern_name,
                width=BUTTON_SM[0],
                height=BUTTON_SM[1],
                command=lambda p=pattern: self.pattern_entry.delete(0, "end") or self.pattern_entry.insert(0, p)
            ).pack(side="left", padx=2)

        # Delete by domain section
        domain_frame = ctk.CTkFrame(content)
        domain_frame.grid(row=3, column=0, sticky="ew", pady=(0, SPACE_MD))
        domain_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            domain_frame,
            text="Delete by Domain",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        ctk.CTkLabel(domain_frame, text="Domain:").grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_XS)

        self.domain_entry = ctk.CTkEntry(domain_frame, placeholder_text="e.g., test, deprecated")
        self.domain_entry.grid(row=1, column=1, sticky="ew", padx=SPACE_XS, pady=SPACE_XS)

        ctk.CTkButton(
            domain_frame,
            text="Delete",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._delete_by_domain,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).grid(row=1, column=2, padx=(SPACE_XS, SPACE_MD), pady=SPACE_XS)

        # Delete by date range section
        date_frame = ctk.CTkFrame(content)
        date_frame.grid(row=4, column=0, sticky="ew", pady=(0, SPACE_MD))
        date_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            date_frame,
            text="Delete by Date Range",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        ctk.CTkLabel(date_frame, text="Delete entries older than:").grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_XS)

        self.days_entry = ctk.CTkEntry(date_frame, placeholder_text="Days (e.g., 30, 90, 365)", width=TEXTBOX_MD)
        self.days_entry.grid(row=1, column=1, sticky="w", padx=SPACE_XS, pady=SPACE_XS)

        ctk.CTkButton(
            date_frame,
            text="Delete",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._delete_by_date_range,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).grid(row=1, column=2, padx=(SPACE_XS, SPACE_MD), pady=SPACE_XS)

        # Delete low confidence section
        confidence_frame = ctk.CTkFrame(content)
        confidence_frame.grid(row=5, column=0, sticky="ew", pady=(0, SPACE_MD))
        confidence_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            confidence_frame,
            text="Delete Low Confidence Entries",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        ctk.CTkLabel(
            confidence_frame,
            text="Confidence Threshold:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_XS))

        slider_frame = ctk.CTkFrame(confidence_frame, fg_color="transparent")
        slider_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_SM))
        slider_frame.grid_columnconfigure(0, weight=1)

        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.confidence_slider = ctk.CTkSlider(
            slider_frame,
            from_=0.0,
            to=1.0,
            variable=self.confidence_threshold,
            command=self._update_confidence_label
        )
        self.confidence_slider.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_SM))

        self.confidence_label = ctk.CTkLabel(slider_frame, text="0.30")
        self.confidence_label.grid(row=0, column=1, sticky="e")

        confidence_buttons = ctk.CTkFrame(confidence_frame, fg_color="transparent")
        confidence_buttons.grid(row=3, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_MD))

        ctk.CTkButton(
            confidence_buttons,
            text="Preview",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._preview_low_confidence
        ).pack(side="left", padx=(0, SPACE_XS))

        ctk.CTkButton(
            confidence_buttons,
            text="Delete",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._delete_low_confidence,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).pack(side="left", padx=SPACE_XS)

        # Database maintenance section
        maintenance_frame = ctk.CTkFrame(content)
        maintenance_frame.grid(row=6, column=0, sticky="ew", pady=(0, SPACE_MD))

        ctk.CTkLabel(
            maintenance_frame,
            text="Database Maintenance",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        # Backup before cleanup
        self.backup_var = tk.BooleanVar(value=True)
        backup_check = ctk.CTkCheckBox(
            maintenance_frame,
            text="Create backup before destructive operations",
            variable=self.backup_var
        )
        backup_check.grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_SM))

        maintenance_buttons = ctk.CTkFrame(maintenance_frame, fg_color="transparent")
        maintenance_buttons.grid(row=2, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_MD))

        ctk.CTkButton(
            maintenance_buttons,
            text="Vacuum Database",
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            command=self._vacuum_database
        ).pack(side="left", padx=(0, SPACE_XS))

        ctk.CTkButton(
            maintenance_buttons,
            text="Create Backup",
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            command=self._create_backup
        ).pack(side="left", padx=SPACE_XS)

        ctk.CTkButton(
            maintenance_buttons,
            text="Refresh Statistics",
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            command=self._refresh_cleanup_stats
        ).pack(side="left", padx=SPACE_XS)

        # Results section
        results_frame = ctk.CTkFrame(content)
        results_frame.grid(row=7, column=0, sticky="ew")
        results_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            results_frame,
            text="Cleanup Results",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        self.cleanup_results = ctk.CTkTextbox(results_frame, height=TEXTBOX_MD, wrap="word")
        self.cleanup_results.grid(row=1, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_MD))

    # ==================== System Reset Section ====================

    def _create_system_reset_section(self, parent, row: int):
        """Create System Reset section at the top of Cleanup tab."""
        # Main frame with warning color border
        reset_frame = ctk.CTkFrame(parent, border_width=2, border_color="#e74c3c")
        reset_frame.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))
        reset_frame.grid_columnconfigure(0, weight=1)

        # Header with warning icon
        header = ctk.CTkFrame(reset_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))

        ctk.CTkLabel(
            header,
            text="System Reset",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold"),
            text_color="#e74c3c"
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="Refresh",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            command=self._refresh_system_reset_stats
        ).pack(side="right")

        # Description
        ctk.CTkLabel(
            reset_frame,
            text="Backup and/or wipe ALL Felix databases to start fresh.",
            font=ctk.CTkFont(size=FONT_BODY),
            text_color="gray"
        ).grid(row=1, column=0, sticky="w", padx=SPACE_MD, pady=(0, SPACE_SM))

        # Database status list
        self.db_status_frame = ctk.CTkFrame(reset_frame)
        self.db_status_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_SM))
        self.db_status_frame.grid_columnconfigure(0, weight=1)

        # Create labels for each database (will be populated by refresh)
        self.db_status_labels = {}

        # Last backup info
        self.last_backup_label = ctk.CTkLabel(
            reset_frame,
            text="Last Backup: Never",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color="gray"
        )
        self.last_backup_label.grid(row=3, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_XS, SPACE_SM))

        # Buttons frame
        buttons_frame = ctk.CTkFrame(reset_frame, fg_color="transparent")
        buttons_frame.grid(row=4, column=0, sticky="ew", padx=SPACE_MD, pady=(0, SPACE_MD))

        ctk.CTkButton(
            buttons_frame,
            text="Backup All Databases",
            command=self._backup_all_databases,
            width=BUTTON_LG[0],
            height=BUTTON_LG[1]
        ).pack(side="left", padx=(0, SPACE_SM))

        ctk.CTkButton(
            buttons_frame,
            text="Reset All (Fresh Start)",
            command=self._reset_all_databases,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=BUTTON_LG[0],
            height=BUTTON_LG[1]
        ).pack(side="left")

        # Initial refresh
        self.after(100, self._refresh_system_reset_stats)

    def _refresh_system_reset_stats(self):
        """Refresh the system reset database status display."""
        try:
            from src.knowledge.system_reset import SystemResetManager

            manager = SystemResetManager()
            stats = manager.get_database_stats()

            # Clear old labels
            for widget in self.db_status_frame.winfo_children():
                widget.destroy()
            self.db_status_labels.clear()

            # Create header row
            header_frame = ctk.CTkFrame(self.db_status_frame, fg_color="transparent")
            header_frame.grid(row=0, column=0, sticky="ew", pady=(SPACE_XS, SPACE_XS))
            header_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                header_frame,
                text="Database",
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                width=250,
                anchor="w"
            ).grid(row=0, column=0, sticky="w")

            ctk.CTkLabel(
                header_frame,
                text="Size",
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                width=BUTTON_SM[0],
                anchor="e"
            ).grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

            ctk.CTkLabel(
                header_frame,
                text="Status",
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                width=BUTTON_SM[0],
                anchor="e"
            ).grid(row=0, column=2, sticky="e", padx=(SPACE_SM, 0))

            # Create row for each database
            total_size = 0
            for i, db_stat in enumerate(stats):
                row_frame = ctk.CTkFrame(self.db_status_frame, fg_color="transparent")
                row_frame.grid(row=i + 1, column=0, sticky="ew", pady=2)
                row_frame.grid_columnconfigure(0, weight=1)

                # Database name
                ctk.CTkLabel(
                    row_frame,
                    text=db_stat["name"],
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    width=250,
                    anchor="w"
                ).grid(row=0, column=0, sticky="w")

                # Size
                ctk.CTkLabel(
                    row_frame,
                    text=db_stat["size_display"] if db_stat["exists"] else "-",
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    width=BUTTON_SM[0],
                    anchor="e"
                ).grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

                # Status
                status_text = "exists" if db_stat["exists"] else "missing"
                status_color = "green" if db_stat["exists"] else "gray"
                ctk.CTkLabel(
                    row_frame,
                    text=status_text,
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    text_color=status_color,
                    width=BUTTON_SM[0],
                    anchor="e"
                ).grid(row=0, column=2, sticky="e", padx=(SPACE_SM, 0))

                if db_stat["exists"]:
                    total_size += db_stat["size_bytes"]

            # Total row
            total_frame = ctk.CTkFrame(self.db_status_frame, fg_color="transparent")
            total_frame.grid(row=len(stats) + 1, column=0, sticky="ew", pady=(SPACE_XS, SPACE_XS))
            total_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                total_frame,
                text="Total",
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                width=250,
                anchor="w"
            ).grid(row=0, column=0, sticky="w")

            total_display = f"{total_size / (1024 * 1024):.1f} MB" if total_size >= 1024 * 1024 else f"{total_size / 1024:.1f} KB"
            ctk.CTkLabel(
                total_frame,
                text=total_display,
                font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
                width=BUTTON_SM[0],
                anchor="e"
            ).grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

            # Update last backup time
            last_backup = manager.get_last_backup_time()
            if last_backup:
                self.last_backup_label.configure(
                    text=f"Last Backup: {last_backup.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                self.last_backup_label.configure(text="Last Backup: Never")

        except Exception as e:
            logger.error(f"Error refreshing system reset stats: {e}")

    def _backup_all_databases(self):
        """Backup all Felix databases."""
        try:
            from src.knowledge.system_reset import SystemResetManager

            manager = SystemResetManager()

            # Confirm
            stats = manager.get_database_stats()
            existing = [s["name"] for s in stats if s["exists"]]

            if not existing:
                self._show_warning("Backup", "No databases found to backup.")
                return

            result = messagebox.askyesno(
                "Confirm Backup",
                f"This will backup {len(existing)} database(s):\n\n"
                + "\n".join(f"  - {name}" for name in existing)
                + "\n\nContinue?"
            )

            if not result:
                return

            # Perform backup
            backup_result = manager.backup_all_databases()

            if backup_result["success"]:
                self._show_info(
                    "Backup Complete",
                    f"Successfully backed up {len(backup_result['backed_up'])} database(s) to:\n\n"
                    f"{backup_result['backup_path']}"
                )
                self._refresh_system_reset_stats()
            else:
                self._show_error("Backup Failed", backup_result.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Error backing up databases: {e}")
            self._show_error("Backup Error", f"Failed to backup databases: {str(e)}")

    def _reset_all_databases(self):
        """Reset (wipe) all Felix databases."""
        # Check if Felix system is running
        if self.main_app and hasattr(self.main_app, 'felix_system') and self.main_app.felix_system:
            self._show_warning(
                "System Running",
                "Please stop the Felix system before resetting databases.\n\n"
                "Go to Dashboard and click 'Stop Felix' first."
            )
            return

        try:
            from src.knowledge.system_reset import SystemResetManager

            manager = SystemResetManager()
            stats = manager.get_database_stats()
            existing = [s["name"] for s in stats if s["exists"]]

            if not existing:
                self._show_info("Reset", "No databases found. System is already clean.")
                return

            # First confirmation
            result = messagebox.askyesno(
                "Confirm Reset",
                "WARNING: This will permanently delete ALL Felix data!\n\n"
                f"Databases to delete ({len(existing)}):\n"
                + "\n".join(f"  - {name}" for name in existing)
                + "\n\nIndex files (.felix_index.json) will also be deleted.\n\n"
                "This action cannot be undone!\n\n"
                "Do you want to backup first?"
            )

            if result:
                # User wants to backup first
                backup_result = manager.backup_all_databases()
                if not backup_result["success"]:
                    self._show_error("Backup Failed", "Backup failed. Reset aborted.")
                    return
                self._show_info("Backup Created", f"Backup saved to:\n{backup_result['backup_path']}")

            # Final confirmation - require typing RESET
            dialog = ctk.CTkInputDialog(
                text="Type RESET to confirm deletion of all databases:",
                title="Final Confirmation"
            )
            confirmation = dialog.get_input()

            if confirmation != "RESET":
                self._show_info("Cancelled", "Reset cancelled. No changes were made.")
                return

            # Perform reset
            wipe_result = manager.wipe_all_databases(delete_indexes=True)

            if wipe_result["success"]:
                deleted_count = len(wipe_result["deleted"])
                index_count = len(wipe_result["index_files_deleted"])

                # Re-initialize databases with empty schemas
                init_result = manager.initialize_all_databases()

                if init_result["success"]:
                    self._show_info(
                        "Reset Complete",
                        f"Successfully deleted:\n\n"
                        f"  - {deleted_count} database(s)\n"
                        f"  - {index_count} index file(s)\n\n"
                        f"Re-initialized {len(init_result['initialized'])} database(s) with fresh schemas.\n\n"
                        "You may need to restart the application."
                    )
                else:
                    error_list = "\n".join(init_result["errors"][:5])
                    self._show_warning(
                        "Reset Partial",
                        f"Databases deleted but some failed to re-initialize:\n\n"
                        f"{error_list}\n\n"
                        "Please restart the application."
                    )
                self._refresh_system_reset_stats()
            else:
                self._show_error("Reset Failed", wipe_result.get("error", "Unknown error"))

        except Exception as e:
            logger.error(f"Error resetting databases: {e}")
            self._show_error("Reset Error", f"Failed to reset databases: {str(e)}")

    def _update_confidence_label(self, value):
        """Update confidence threshold label."""
        self.confidence_label.configure(text=f"{float(value):.2f}")

    def _preview_custom_pattern(self):
        """Preview cleanup by pattern."""
        if not self.knowledge_store:
            self._show_warning("Preview Cleanup", "Felix system not running")
            return

        pattern = self.pattern_entry.get().strip()
        if not pattern:
            self._show_warning("Invalid Input", "Please enter a path pattern")
            return

        try:
            from src.knowledge.knowledge_cleanup import KnowledgeCleanupManager

            manager = KnowledgeCleanupManager(self.knowledge_store)
            result = manager.preview_cleanup_by_pattern(pattern)

            self._display_cleanup_result("Preview - Pattern Cleanup", result, is_preview=True)

        except Exception as e:
            logger.error(f"Error previewing pattern cleanup: {e}")
            self._show_error("Preview Error", f"Failed to preview cleanup: {str(e)}")

    def _execute_custom_pattern(self):
        """Execute cleanup by pattern."""
        if not self.knowledge_store:
            self._show_warning("Clean Pattern", "Felix system not running")
            return

        pattern = self.pattern_entry.get().strip()
        if not pattern:
            self._show_warning("Invalid Input", "Please enter a path pattern")
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Cleanup",
            f"This will permanently delete all documents and entries matching:\n\n{pattern}\n\n"
            f"Cascade entries: {self.cascade_var.get()}\n\n"
            f"Continue?"
        )

        if not result:
            return

        try:
            # Create backup if enabled
            if self.backup_var.get():
                self._create_backup()

            from src.knowledge.knowledge_cleanup import KnowledgeCleanupManager

            manager = KnowledgeCleanupManager(self.knowledge_store)
            result = self.knowledge_store.delete_documents_by_pattern(
                pattern,
                cascade_entries=self.cascade_var.get(),
                dry_run=False
            )

            self._display_cleanup_result("Pattern Cleanup Complete", result, is_preview=False)

            # Refresh displays
            self.refresh()

        except Exception as e:
            logger.error(f"Error executing pattern cleanup: {e}")
            self._show_error("Cleanup Error", f"Failed to execute cleanup: {str(e)}")

    def _delete_by_domain(self):
        """Delete all entries in a domain."""
        if not self.knowledge_store:
            self._show_warning("Delete by Domain", "Felix system not running")
            return

        domain = self.domain_entry.get().strip()
        if not domain:
            self._show_warning("Invalid Input", "Please enter a domain name")
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete all knowledge entries in domain:\n\n{domain}\n\n"
            f"Continue?"
        )

        if not result:
            return

        try:
            # Create backup if enabled
            if self.backup_var.get():
                self._create_backup()

            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM knowledge_entries
                WHERE domain = ?
            """, (domain,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0", f"Domain Cleanup Complete\n\n" +
                                                f"Deleted {deleted_count} entries from domain '{domain}'")

            self._show_info("Cleanup Complete", f"Deleted {deleted_count} entries from domain '{domain}'")

            # Refresh displays
            self.refresh()
            logger.info(f"Deleted {deleted_count} entries from domain '{domain}'")

        except Exception as e:
            logger.error(f"Error deleting by domain: {e}")
            self._show_error("Deletion Error", f"Failed to delete by domain: {str(e)}")

    def _delete_by_date_range(self):
        """Delete entries older than specified days."""
        if not self.knowledge_store:
            self._show_warning("Delete by Date", "Felix system not running")
            return

        days_str = self.days_entry.get().strip()
        if not days_str:
            self._show_warning("Invalid Input", "Please enter number of days")
            return

        try:
            days = int(days_str)
            if days < 1:
                self._show_warning("Invalid Input", "Days must be at least 1")
                return
        except ValueError:
            self._show_warning("Invalid Input", "Please enter a valid number")
            return

        # Confirm
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete all knowledge entries older than {days} days.\n\n"
            f"Continue?"
        )

        if not result:
            return

        try:
            # Create backup if enabled
            if self.backup_var.get():
                self._create_backup()

            import sqlite3

            # created_at is stored as Unix timestamp (REAL), not ISO string
            cutoff_timestamp = (datetime.now() - timedelta(days=days)).timestamp()

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM knowledge_entries
                WHERE created_at < ?
            """, (cutoff_timestamp,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0", f"Date Range Cleanup Complete\n\n" +
                                                f"Deleted {deleted_count} entries older than {days} days")

            self._show_info("Cleanup Complete", f"Deleted {deleted_count} entries older than {days} days")

            # Refresh displays
            self.refresh()
            logger.info(f"Deleted {deleted_count} entries older than {days} days")

        except Exception as e:
            logger.error(f"Error deleting by date range: {e}")
            self._show_error("Deletion Error", f"Failed to delete by date range: {str(e)}")

    def _get_confidence_levels_for_threshold(self, threshold: float) -> List[str]:
        """Map numeric threshold to confidence level names.

        confidence_level is TEXT: 'VERIFIED', 'HIGH', 'MEDIUM', 'LOW', 'SPECULATIVE'
        Lower threshold = delete fewer entries (only lowest confidence)
        Higher threshold = delete more entries (include higher confidence levels)
        """
        if threshold <= 0.2:
            return ['SPECULATIVE']
        elif threshold <= 0.4:
            return ['SPECULATIVE', 'LOW']
        elif threshold <= 0.6:
            return ['SPECULATIVE', 'LOW', 'MEDIUM']
        elif threshold <= 0.8:
            return ['SPECULATIVE', 'LOW', 'MEDIUM', 'HIGH']
        else:
            return ['SPECULATIVE', 'LOW', 'MEDIUM', 'HIGH', 'VERIFIED']

    def _preview_low_confidence(self):
        """Preview deletion of low confidence entries."""
        if not self.knowledge_store:
            self._show_warning("Preview Deletion", "Felix system not running")
            return

        threshold = self.confidence_threshold.get()
        levels_to_delete = self._get_confidence_levels_for_threshold(threshold)

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # confidence_level is TEXT enum, not a float
            placeholders = ','.join('?' * len(levels_to_delete))
            cursor.execute(f"""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE confidence_level IN ({placeholders})
            """, levels_to_delete)

            count = cursor.fetchone()[0]
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Preview - Low Confidence Cleanup\n\n"
                f"Threshold: {threshold:.2f}\n"
                f"Confidence levels to delete: {', '.join(levels_to_delete)}\n"
                f"Entries to delete: {count}\n\n"
                f"This is a preview. Click 'Delete' to execute."
            )

        except Exception as e:
            logger.error(f"Error previewing low confidence cleanup: {e}")
            self._show_error("Preview Error", f"Failed to preview cleanup: {str(e)}")

    def _delete_low_confidence(self):
        """Delete entries below confidence threshold."""
        if not self.knowledge_store:
            self._show_warning("Delete Low Confidence", "Felix system not running")
            return

        threshold = self.confidence_threshold.get()
        levels_to_delete = self._get_confidence_levels_for_threshold(threshold)

        # Confirm
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete all entries with confidence levels:\n"
            f"{', '.join(levels_to_delete)}\n\n"
            f"Continue?"
        )

        if not result:
            return

        try:
            # Create backup if enabled
            if self.backup_var.get():
                self._create_backup()

            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # confidence_level is TEXT enum, not a float
            placeholders = ','.join('?' * len(levels_to_delete))
            cursor.execute(f"""
                DELETE FROM knowledge_entries
                WHERE confidence_level IN ({placeholders})
            """, levels_to_delete)

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Low Confidence Cleanup Complete\n\n"
                f"Confidence levels deleted: {', '.join(levels_to_delete)}\n"
                f"Deleted: {deleted_count} entries"
            )

            self._show_info("Cleanup Complete", f"Deleted {deleted_count} low confidence entries")

            # Refresh displays
            self.refresh()
            logger.info(f"Deleted {deleted_count} entries with confidence_level IN {levels_to_delete}")

        except Exception as e:
            logger.error(f"Error deleting low confidence entries: {e}")
            self._show_error("Deletion Error", f"Failed to delete entries: {str(e)}")

    def _vacuum_database(self):
        """Vacuum database to reclaim space."""
        if not self.knowledge_store:
            self._show_warning("Vacuum Database", "Felix system not running")
            return

        # Confirm
        result = messagebox.askyesno(
            "Vacuum Database",
            "This will optimize the database and reclaim unused space.\n\n"
            "This may take a few moments.\n\n"
            "Continue?"
        )

        if not result:
            return

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Get size before
            cursor = conn.cursor()
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            size_before = cursor.fetchone()[0]

            # Vacuum
            conn.execute("VACUUM")

            # Get size after
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            size_after = cursor.fetchone()[0]

            conn.close()

            saved_mb = (size_before - size_after) / (1024 * 1024)

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Database Vacuum Complete\n\n"
                f"Size before: {size_before / (1024 * 1024):.2f} MB\n"
                f"Size after: {size_after / (1024 * 1024):.2f} MB\n"
                f"Space saved: {saved_mb:.2f} MB"
            )

            self._show_info("Vacuum Complete", f"Database optimized. Saved {saved_mb:.2f} MB")
            logger.info(f"Database vacuumed, saved {saved_mb:.2f} MB")

        except Exception as e:
            logger.error(f"Error vacuuming database: {e}")
            self._show_error("Vacuum Error", f"Failed to vacuum database: {str(e)}")

    def _create_backup(self):
        """Create database backup."""
        if not self.knowledge_store:
            self._show_warning("Create Backup", "Felix system not running")
            return

        try:
            from src.knowledge.backup_manager_extended import KnowledgeBackupManager

            backup_manager = KnowledgeBackupManager(self.knowledge_store)
            backup_path = backup_manager.export_to_json()

            if backup_path:
                self._show_info("Backup Created", f"Backup saved to:\n{backup_path}")
                logger.info(f"Backup created: {backup_path}")
            else:
                self._show_warning("Backup Failed", "Failed to create backup")

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            self._show_error("Backup Error", f"Failed to create backup: {str(e)}")

    def _refresh_cleanup_stats(self):
        """Refresh cleanup statistics."""
        if not self.knowledge_store:
            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0", "Felix system not running. Start Felix to see statistics.")
            return

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Get various counts
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM document_sources")
            total_docs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM knowledge_relationships")
            total_relationships = cursor.fetchone()[0]

            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0] / (1024 * 1024)

            # confidence_level is TEXT enum: 'VERIFIED', 'HIGH', 'MEDIUM', 'LOW', 'SPECULATIVE'
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE confidence_level IN ('LOW', 'SPECULATIVE')")
            low_conf = cursor.fetchone()[0]

            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Database Statistics\n\n"
                f"Total Entries: {total_entries}\n"
                f"Total Documents: {total_docs}\n"
                f"Total Relationships: {total_relationships}\n"
                f"Database Size: {db_size:.2f} MB\n"
                f"Low Confidence Entries (LOW/SPECULATIVE): {low_conf}\n\n"
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except Exception as e:
            logger.error(f"Error refreshing cleanup stats: {e}")
            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0", f"Error: {str(e)}")

    def _display_cleanup_result(self, title: str, result: Dict[str, Any], is_preview: bool):
        """Display cleanup result in results textbox."""
        self.cleanup_results.delete("1.0", "end")

        if "error" in result:
            self.cleanup_results.insert("1.0", f"{title}\n\nError: {result['error']}")
            return

        text = f"{title}\n\n"

        if is_preview:
            text += "THIS IS A PREVIEW - NO CHANGES MADE\n\n"

        if "documents_deleted" in result:
            text += f"Documents: {result['documents_deleted']}\n"
        if "entries_deleted" in result:
            text += f"Entries: {result['entries_deleted']}\n"
        if "affected_documents" in result:
            text += f"\nAffected Documents:\n"
            for doc in result["affected_documents"][:10]:
                text += f"  - {doc}\n"
            if len(result["affected_documents"]) > 10:
                text += f"  ... and {len(result['affected_documents']) - 10} more\n"

        self.cleanup_results.insert("1.0", text)

    # ==================== Common Methods ====================

    def _enable_features(self):
        """Enable features when Felix system starts."""
        if self.main_app and hasattr(self.main_app, 'felix_system') and self.main_app.felix_system:
            self.set_knowledge_refs(
                knowledge_store=getattr(self.main_app.felix_system, 'knowledge_store', None),
                knowledge_retriever=getattr(self.main_app.felix_system, 'knowledge_retriever', None),
                knowledge_daemon=getattr(self.main_app.felix_system, 'knowledge_daemon', None)
            )
            logger.info("Maintenance Panel features enabled")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        self.set_knowledge_refs(None, None, None)
        logger.info("Maintenance Panel features disabled")

    def refresh(self):
        """Refresh all displays."""
        if self.knowledge_store:
            self._refresh_quality_stats()
            self._refresh_audit_log()
            self._refresh_cleanup_stats()

    def _show_info(self, title: str, message: str):
        """Show info message."""
        messagebox.showinfo(title, message)

    def _show_warning(self, title: str, message: str):
        """Show warning message."""
        messagebox.showwarning(title, message)

    def _show_error(self, title: str, message: str):
        """Show error message."""
        messagebox.showerror(title, message)

    def _safe_grab(self, dialog):
        """Safely grab focus after window is rendered."""
        try:
            dialog.grab_set()
            dialog.focus_set()
        except Exception as e:
            logger.warning(f"Could not grab dialog focus: {e}")
