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

        # Create main tabview with 3 sub-tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

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

        # Refresh displays if available
        if knowledge_store:
            self.refresh()

    # ==================== Quality Sub-tab ====================

    def _create_quality_tab(self):
        """Create Quality sub-tab for monitoring knowledge quality."""
        quality_frame = self.tabview.tab("Quality")

        # Configure grid
        quality_frame.grid_columnconfigure(0, weight=1)
        quality_frame.grid_rowconfigure(1, weight=1)

        # Header with title and refresh button
        header = ctk.CTkFrame(quality_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header.grid_columnconfigure(1, weight=1)

        title_label = ctk.CTkLabel(
            header,
            text="Knowledge Quality Monitor",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w")

        refresh_btn = ctk.CTkButton(
            header,
            text="Refresh",
            width=100,
            command=self._refresh_quality_stats
        )
        refresh_btn.grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Main content area with scrolling
        content = ctk.CTkScrollableFrame(quality_frame)
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        content.grid_columnconfigure(0, weight=1)

        # Quality scores section
        scores_label = ctk.CTkLabel(
            content,
            text="Quality Metrics",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        scores_label.grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Status cards container
        cards_frame = ctk.CTkFrame(content, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))

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
                width=160,
                height=110
            )
            card.grid(row=0, column=i, padx=5, sticky="ew")
            cards_frame.grid_columnconfigure(i, weight=1)
            self.quality_cards[key] = card

        # Duplicates section
        duplicates_label = ctk.CTkLabel(
            content,
            text="Potential Duplicates",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        duplicates_label.grid(row=2, column=0, sticky="w", pady=(10, 5))

        # Duplicates tree
        self.duplicates_tree = ThemedTreeview(
            content,
            columns=["Entry 1", "Entry 2", "Similarity", "Type", "Action"],
            headings=["Entry 1", "Entry 2", "Similarity", "Type", "Suggested Action"],
            widths=[200, 200, 100, 120, 150],
            height=8
        )
        self.duplicates_tree.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        # Low confidence section
        low_conf_label = ctk.CTkLabel(
            content,
            text="Low Confidence Entries",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        low_conf_label.grid(row=4, column=0, sticky="w", pady=(10, 5))

        # Low confidence tree
        self.low_confidence_tree = ThemedTreeview(
            content,
            columns=["ID", "Content", "Confidence", "Domain", "Created"],
            headings=["Entry ID", "Content Preview", "Confidence", "Domain", "Created Date"],
            widths=[200, 250, 100, 120, 130],
            height=8
        )
        self.low_confidence_tree.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        # Actions section
        actions_frame = ctk.CTkFrame(content, fg_color="transparent")
        actions_frame.grid(row=6, column=0, sticky="ew", pady=(10, 0))

        ctk.CTkButton(
            actions_frame,
            text="Run Full Quality Check",
            command=self._run_quality_check
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            actions_frame,
            text="Generate Quality Report",
            command=self._generate_quality_report
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            actions_frame,
            text="View Orphaned Concepts",
            command=self._view_orphaned_concepts
        ).pack(side="left", padx=5)

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
        """Get entries with confidence below threshold."""
        if not self.knowledge_store:
            return []

        try:
            import sqlite3
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT knowledge_id, content, confidence, domain, timestamp
                FROM knowledge_entries
                WHERE confidence < ?
                ORDER BY confidence ASC
                LIMIT 100
            """, (threshold,))

            entries = []
            for row in cursor.fetchall():
                entries.append({
                    "knowledge_id": row[0],
                    "content": row[1],
                    "confidence": row[2],
                    "domain": row[3] or "general",
                    "timestamp": row[4]
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
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT k.knowledge_id, k.content, k.domain
                FROM knowledge_entries k
                LEFT JOIN knowledge_relationships r1 ON k.knowledge_id = r1.from_concept_id
                LEFT JOIN knowledge_relationships r2 ON k.knowledge_id = r2.to_concept_id
                WHERE r1.from_concept_id IS NULL AND r2.to_concept_id IS NULL
                LIMIT 100
            """)

            orphaned = []
            for row in cursor.fetchall():
                orphaned.append({
                    "knowledge_id": row[0],
                    "content": row[1],
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
        dialog.grab_set()

        # Title
        title = ctk.CTkLabel(
            dialog,
            text=f"Found {len(orphaned)} Orphaned Concepts",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=10)

        # List
        tree = ThemedTreeview(
            dialog,
            columns=["ID", "Content", "Domain"],
            headings=["Entry ID", "Content Preview", "Domain"],
            widths=[200, 250, 120],
            height=15
        )
        tree.pack(fill="both", expand=True, padx=10, pady=10)

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
            command=dialog.destroy
        ).pack(pady=10)

    # ==================== Audit Sub-tab ====================

    def _create_audit_tab(self):
        """Create Audit sub-tab for CRUD operation history."""
        audit_frame = self.tabview.tab("Audit")

        # Configure grid
        audit_frame.grid_columnconfigure(0, weight=1)
        audit_frame.grid_rowconfigure(2, weight=1)

        # Header
        header = ctk.CTkFrame(audit_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        title_label = ctk.CTkLabel(
            header,
            text="Audit Log - Knowledge Base Operations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(side="left")

        # Filters section
        filters_frame = ctk.CTkFrame(audit_frame)
        filters_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        filters_frame.grid_columnconfigure(1, weight=1)

        # Row 1: Operation type and User/Agent
        row1 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", pady=5, padx=10)

        ctk.CTkLabel(row1, text="Operation:").pack(side="left", padx=(0, 5))
        self.audit_operation_var = tk.StringVar(value="ALL")
        operation_combo = ctk.CTkComboBox(
            row1,
            variable=self.audit_operation_var,
            values=["ALL", "INSERT", "UPDATE", "DELETE", "MERGE", "CLEANUP", "SYSTEM"],
            width=120
        )
        operation_combo.pack(side="left", padx=5)

        ctk.CTkLabel(row1, text="User/Agent:").pack(side="left", padx=(15, 5))
        self.audit_user_var = tk.StringVar()
        user_entry = ctk.CTkEntry(row1, textvariable=self.audit_user_var, width=150)
        user_entry.pack(side="left", padx=5)

        # Row 2: Knowledge ID filter
        row2 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row2.grid(row=1, column=0, sticky="ew", pady=5, padx=10)

        ctk.CTkLabel(row2, text="Knowledge ID:").pack(side="left", padx=(0, 5))
        self.audit_knowledge_id_var = tk.StringVar()
        knowledge_id_entry = ctk.CTkEntry(row2, textvariable=self.audit_knowledge_id_var, width=300)
        knowledge_id_entry.pack(side="left", padx=5)

        # Row 3: Action buttons
        row3 = ctk.CTkFrame(filters_frame, fg_color="transparent")
        row3.grid(row=2, column=0, sticky="ew", pady=5, padx=10)

        ctk.CTkButton(row3, text="Search", width=100, command=self._search_audit_log).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="Refresh", width=100, command=self._refresh_audit_log).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="Export CSV", width=120, command=self._export_audit_log).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="Clear Filters", width=120, command=self._clear_audit_filters).pack(side="left", padx=5)
        ctk.CTkButton(row3, text="Clear Old Entries", width=140, command=self._clear_old_audit_entries).pack(side="left", padx=5)

        # Audit log tree
        self.audit_tree = ThemedTreeview(
            audit_frame,
            columns=["Timestamp", "Operation", "Target", "Details", "User"],
            headings=["Timestamp", "Operation", "Knowledge ID", "Details", "User/Agent"],
            widths=[150, 100, 200, 250, 120],
            height=15
        )
        self.audit_tree.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # Bind double-click to view details
        self.audit_tree.bind_tree("<Double-1>", self._view_audit_details)

        # Statistics section
        stats_frame = ctk.CTkFrame(audit_frame)
        stats_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))

        self.audit_stats_label = ctk.CTkLabel(
            stats_frame,
            text="Statistics: Loading...",
            font=ctk.CTkFont(size=11)
        )
        self.audit_stats_label.pack(pady=10)

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
        dialog.grab_set()

        # Title
        title = ctk.CTkLabel(
            dialog,
            text="Audit Entry Details",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=10)

        # Details text
        details_frame = ctk.CTkScrollableFrame(dialog)
        details_frame.pack(fill="both", expand=True, padx=10, pady=10)

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
            command=dialog.destroy
        ).pack(pady=10)

    # ==================== Cleanup Sub-tab ====================

    def _create_cleanup_tab(self):
        """Create Cleanup sub-tab for maintenance operations."""
        cleanup_frame = self.tabview.tab("Cleanup")

        # Configure grid
        cleanup_frame.grid_columnconfigure(0, weight=1)
        cleanup_frame.grid_rowconfigure(0, weight=1)

        # Scrollable content
        content = ctk.CTkScrollableFrame(cleanup_frame)
        content.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        content.grid_columnconfigure(0, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            content,
            text="Knowledge Base Cleanup & Maintenance",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 15))

        # Pattern-based cleanup section
        pattern_frame = ctk.CTkFrame(content)
        pattern_frame.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        pattern_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            pattern_frame,
            text="Pattern-Based Deletion",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            pattern_frame,
            text="Path Pattern (glob or SQL LIKE):",
            font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", padx=15, pady=(0, 5))

        self.pattern_entry = ctk.CTkEntry(pattern_frame, placeholder_text="*/test_data/*")
        self.pattern_entry.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))

        # Cascade option
        self.cascade_var = tk.BooleanVar(value=True)
        cascade_check = ctk.CTkCheckBox(
            pattern_frame,
            text="Delete associated knowledge entries",
            variable=self.cascade_var
        )
        cascade_check.grid(row=3, column=0, sticky="w", padx=15, pady=(0, 10))

        # Pattern buttons
        pattern_buttons = ctk.CTkFrame(pattern_frame, fg_color="transparent")
        pattern_buttons.grid(row=4, column=0, sticky="ew", padx=15, pady=(0, 10))

        ctk.CTkButton(
            pattern_buttons,
            text="Preview Pattern",
            command=self._preview_custom_pattern
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            pattern_buttons,
            text="Clean Pattern",
            command=self._execute_custom_pattern,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).pack(side="left", padx=5)

        # Common patterns
        common_frame = ctk.CTkFrame(pattern_frame, fg_color="transparent")
        common_frame.grid(row=5, column=0, sticky="ew", padx=15, pady=(0, 15))

        ctk.CTkLabel(common_frame, text="Quick patterns:").pack(side="left", padx=(0, 10))

        for pattern_name, pattern in [
            (".venv", "*/.venv/*"),
            ("node_modules", "*/node_modules/*"),
            ("__pycache__", "*/__pycache__/*"),
            ("test_data", "*/test_data/*")
        ]:
            ctk.CTkButton(
                common_frame,
                text=pattern_name,
                width=100,
                command=lambda p=pattern: self.pattern_entry.delete(0, "end") or self.pattern_entry.insert(0, p)
            ).pack(side="left", padx=2)

        # Delete by domain section
        domain_frame = ctk.CTkFrame(content)
        domain_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        domain_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            domain_frame,
            text="Delete by Domain",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(domain_frame, text="Domain:").grid(row=1, column=0, sticky="w", padx=15, pady=5)

        self.domain_entry = ctk.CTkEntry(domain_frame, placeholder_text="e.g., test, deprecated")
        self.domain_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ctk.CTkButton(
            domain_frame,
            text="Delete",
            width=100,
            command=self._delete_by_domain,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).grid(row=1, column=2, padx=(5, 15), pady=5)

        # Delete by date range section
        date_frame = ctk.CTkFrame(content)
        date_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        date_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            date_frame,
            text="Delete by Date Range",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(date_frame, text="Delete entries older than:").grid(row=1, column=0, sticky="w", padx=15, pady=5)

        self.days_entry = ctk.CTkEntry(date_frame, placeholder_text="Days (e.g., 30, 90, 365)", width=200)
        self.days_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ctk.CTkButton(
            date_frame,
            text="Delete",
            width=100,
            command=self._delete_by_date_range,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).grid(row=1, column=2, padx=(5, 15), pady=5)

        # Delete low confidence section
        confidence_frame = ctk.CTkFrame(content)
        confidence_frame.grid(row=4, column=0, sticky="ew", pady=(0, 15))
        confidence_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            confidence_frame,
            text="Delete Low Confidence Entries",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        ctk.CTkLabel(
            confidence_frame,
            text="Confidence Threshold:",
            font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", padx=15, pady=(0, 5))

        slider_frame = ctk.CTkFrame(confidence_frame, fg_color="transparent")
        slider_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))
        slider_frame.grid_columnconfigure(0, weight=1)

        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.confidence_slider = ctk.CTkSlider(
            slider_frame,
            from_=0.0,
            to=1.0,
            variable=self.confidence_threshold,
            command=self._update_confidence_label
        )
        self.confidence_slider.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.confidence_label = ctk.CTkLabel(slider_frame, text="0.30")
        self.confidence_label.grid(row=0, column=1, sticky="e")

        confidence_buttons = ctk.CTkFrame(confidence_frame, fg_color="transparent")
        confidence_buttons.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

        ctk.CTkButton(
            confidence_buttons,
            text="Preview",
            command=self._preview_low_confidence
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            confidence_buttons,
            text="Delete",
            command=self._delete_low_confidence,
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self.theme_manager.get_color("error")
        ).pack(side="left", padx=5)

        # Database maintenance section
        maintenance_frame = ctk.CTkFrame(content)
        maintenance_frame.grid(row=5, column=0, sticky="ew", pady=(0, 15))

        ctk.CTkLabel(
            maintenance_frame,
            text="Database Maintenance",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        # Backup before cleanup
        self.backup_var = tk.BooleanVar(value=True)
        backup_check = ctk.CTkCheckBox(
            maintenance_frame,
            text="Create backup before destructive operations",
            variable=self.backup_var
        )
        backup_check.grid(row=1, column=0, sticky="w", padx=15, pady=(0, 10))

        maintenance_buttons = ctk.CTkFrame(maintenance_frame, fg_color="transparent")
        maintenance_buttons.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))

        ctk.CTkButton(
            maintenance_buttons,
            text="Vacuum Database",
            command=self._vacuum_database
        ).pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            maintenance_buttons,
            text="Create Backup",
            command=self._create_backup
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            maintenance_buttons,
            text="Refresh Statistics",
            command=self._refresh_cleanup_stats
        ).pack(side="left", padx=5)

        # Results section
        results_frame = ctk.CTkFrame(content)
        results_frame.grid(row=6, column=0, sticky="ew")
        results_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            results_frame,
            text="Cleanup Results",
            font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 10))

        self.cleanup_results = ctk.CTkTextbox(results_frame, height=150, wrap="word")
        self.cleanup_results.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))

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

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM knowledge_entries
                WHERE timestamp < ?
            """, (cutoff_date,))

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

    def _preview_low_confidence(self):
        """Preview deletion of low confidence entries."""
        if not self.knowledge_store:
            self._show_warning("Preview Deletion", "Felix system not running")
            return

        threshold = self.confidence_threshold.get()

        try:
            import sqlite3

            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE confidence < ?
            """, (threshold,))

            count = cursor.fetchone()[0]
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Preview - Low Confidence Cleanup\n\n"
                f"Threshold: {threshold:.2f}\n"
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

        # Confirm
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"This will permanently delete all entries with confidence below {threshold:.2f}\n\n"
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
                WHERE confidence < ?
            """, (threshold,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Low Confidence Cleanup Complete\n\n"
                f"Threshold: {threshold:.2f}\n"
                f"Deleted: {deleted_count} entries"
            )

            self._show_info("Cleanup Complete", f"Deleted {deleted_count} low confidence entries")

            # Refresh displays
            self.refresh()
            logger.info(f"Deleted {deleted_count} entries with confidence < {threshold:.2f}")

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
            from src.knowledge.backup_manager_extended import BackupManagerExtended

            backup_manager = BackupManagerExtended()
            backup_path = backup_manager.create_backup(self.knowledge_store.storage_path)

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

            cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE confidence < 0.5")
            low_conf = cursor.fetchone()[0]

            conn.close()

            self.cleanup_results.delete("1.0", "end")
            self.cleanup_results.insert("1.0",
                f"Database Statistics\n\n"
                f"Total Entries: {total_entries}\n"
                f"Total Documents: {total_docs}\n"
                f"Total Relationships: {total_relationships}\n"
                f"Database Size: {db_size:.2f} MB\n"
                f"Low Confidence Entries (<0.5): {low_conf}\n\n"
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
