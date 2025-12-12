"""
Relationships Panel for Knowledge Brain GUI.

Visualizes the knowledge graph relationships between concepts with:
- Search and filtering by relationship type and strength
- Color-coded treeview display
- Graph traversal and exploration
- Statistics dashboard
- Actions for managing relationships
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import logging
import json
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path

from ...components.themed_treeview import ThemedTreeview
from ...components.search_entry import SearchEntry
from ...components.status_card import StatusCard
from ...theme_manager import get_theme_manager
from ...styles import (
    BUTTON_SM, BUTTON_MD, BUTTON_LG,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    CARD_MD, TEXTBOX_MD
)

logger = logging.getLogger(__name__)


class RelationshipsPanel(ctk.CTkFrame):
    """
    Relationships Panel for exploring knowledge graph connections.

    Features:
    - Search concepts by name
    - Filter by relationship type and minimum strength
    - Color-coded relationship display
    - Graph traversal (1-hop, 2-hop, 3-hop)
    - Path finding between concepts
    - Statistics breakdown by type
    - Relationship management (delete, adjust strength)
    - Export graph data to JSON
    """

    def __init__(
        self,
        master,
        thread_manager,
        main_app=None,
        **kwargs
    ):
        """
        Initialize RelationshipsPanel.

        Args:
            master: Parent widget
            thread_manager: Thread manager for async operations
            main_app: Reference to main application
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

        # Configure grid for main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container with padding
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(3, weight=1)  # TreeView section expands

        # Create UI sections within container
        self._create_search_section(main_container)
        self._create_filter_section(main_container)
        self._create_statistics_section(main_container)
        self._create_results_section(main_container)
        self._create_actions_section(main_container)

    def _create_search_section(self, parent):
        """Create search and relationship type filter section."""
        search_frame = ctk.CTkFrame(parent)
        search_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, SPACE_SM))
        search_frame.grid_columnconfigure(1, weight=1)

        # Search label
        ctk.CTkLabel(
            search_frame,
            text="Search Concept:",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=(SPACE_SM, SPACE_SM), pady=SPACE_XS)

        # Search entry
        self.search_entry = SearchEntry(
            search_frame,
            placeholder="Enter concept name...",
            on_search=self._on_search,
            on_clear=self._on_clear_search,
            width=TEXTBOX_MD * 2
        )
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_SM), pady=SPACE_XS)

        # Relationship type filter
        ctk.CTkLabel(
            search_frame,
            text="Type:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).grid(row=0, column=2, sticky="w", padx=(SPACE_SM, SPACE_XS), pady=SPACE_XS)

        self.type_var = ctk.StringVar(value="all")
        self.type_dropdown = ctk.CTkOptionMenu(
            search_frame,
            variable=self.type_var,
            values=["all", "explicit_mention", "embedding_similarity", "co_occurrence"],
            command=self._on_filter_change,
            width=CARD_MD
        )
        self.type_dropdown.grid(row=0, column=3, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

    def _create_filter_section(self, parent):
        """Create minimum strength slider and depth selector."""
        filter_frame = ctk.CTkFrame(parent)
        filter_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, SPACE_SM))
        filter_frame.grid_columnconfigure(1, weight=1)

        # Minimum strength slider
        ctk.CTkLabel(
            filter_frame,
            text="Min Strength:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).grid(row=0, column=0, sticky="w", padx=(SPACE_SM, SPACE_SM), pady=SPACE_XS)

        self.strength_var = ctk.DoubleVar(value=0.0)
        self.strength_slider = ctk.CTkSlider(
            filter_frame,
            from_=0.0,
            to=1.0,
            variable=self.strength_var,
            command=self._on_strength_change,
            width=TEXTBOX_MD
        )
        self.strength_slider.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_SM), pady=SPACE_XS)

        self.strength_label = ctk.CTkLabel(
            filter_frame,
            text="0.0",
            font=ctk.CTkFont(size=FONT_BODY),
            width=40
        )
        self.strength_label.grid(row=0, column=2, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

        # Traversal depth dropdown
        ctk.CTkLabel(
            filter_frame,
            text="Explore Depth:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).grid(row=0, column=3, sticky="w", padx=(SPACE_SM, SPACE_XS), pady=SPACE_XS)

        self.depth_var = ctk.StringVar(value="1-hop")
        self.depth_dropdown = ctk.CTkOptionMenu(
            filter_frame,
            variable=self.depth_var,
            values=["1-hop", "2-hop", "3-hop"],
            width=BUTTON_MD[0]
        )
        self.depth_dropdown.grid(row=0, column=4, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

    def _create_statistics_section(self, parent):
        """Create statistics cards showing relationship counts by type."""
        stats_frame = ctk.CTkFrame(parent)
        stats_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, SPACE_SM))

        # Title
        ctk.CTkLabel(
            stats_frame,
            text="Relationship Statistics",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).pack(anchor="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Cards container
        cards_frame = ctk.CTkFrame(stats_frame, fg_color="transparent")
        cards_frame.pack(fill="x", padx=SPACE_SM, pady=(0, SPACE_SM))

        # Create status cards for each relationship type
        self.total_card = StatusCard(
            cards_frame,
            title="Total Relationships",
            value="--",
            width=CARD_MD,
            status_color=self.theme_manager.get_color("accent")
        )
        self.total_card.pack(side="left", padx=(0, SPACE_SM))

        self.explicit_card = StatusCard(
            cards_frame,
            title="Explicit Mentions",
            value="--",
            width=CARD_MD,
            status_color=self.theme_manager.get_color("success")
        )
        self.explicit_card.pack(side="left", padx=(0, SPACE_SM))

        self.similarity_card = StatusCard(
            cards_frame,
            title="Similarity Links",
            value="--",
            width=CARD_MD,
            status_color=self.theme_manager.get_color("warning")
        )
        self.similarity_card.pack(side="left", padx=(0, SPACE_SM))

        self.cooccurrence_card = StatusCard(
            cards_frame,
            title="Co-occurrences",
            value="--",
            width=CARD_MD,
            status_color="#8b5cf6"  # Purple
        )
        self.cooccurrence_card.pack(side="left", padx=(0, SPACE_SM))

    def _create_results_section(self, parent):
        """Create treeview for displaying relationships."""
        results_frame = ctk.CTkFrame(parent)
        results_frame.grid(row=3, column=0, sticky="nsew", padx=0, pady=(0, SPACE_SM))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        # Label
        ctk.CTkLabel(
            results_frame,
            text="Relationships",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # TreeView
        columns = ["source_concept", "rel_type", "target_concept", "strength", "evidence"]
        headings = ["Source Concept", "Relationship Type", "Target Concept", "Strength", "Evidence"]
        widths = [250, 150, 250, 80, 200]

        self.tree = ThemedTreeview(
            results_frame,
            columns=columns,
            headings=headings,
            widths=widths,
            height=15,
            selectmode="extended"
        )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))

        # Configure tags for color-coding
        colors = self.theme_manager.colors
        self.tree.tag_configure("explicit_mention", background=colors["success"], foreground="#ffffff")
        self.tree.tag_configure("embedding_similarity", background=colors["warning"], foreground="#ffffff")
        self.tree.tag_configure("co_occurrence", background="#8b5cf6", foreground="#ffffff")

        # Bind events
        self.tree.bind_tree("<Double-1>", self._on_double_click)
        self.tree.bind_tree("<Button-3>", self._on_right_click)
        self.tree.bind_tree("<<TreeviewSelect>>", self._on_selection_change)

        # Status label
        self.status_label = ctk.CTkLabel(
            results_frame,
            text="Ready",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.status_label.grid(row=2, column=0, sticky="w", padx=SPACE_SM, pady=(0, SPACE_SM))

    def _create_actions_section(self, parent):
        """Create action buttons for graph operations."""
        actions_frame = ctk.CTkFrame(parent)
        actions_frame.grid(row=4, column=0, sticky="ew", padx=0, pady=0)

        # Left side buttons
        left_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
        left_frame.pack(side="left", fill="x", expand=True, padx=SPACE_SM, pady=SPACE_SM)

        self.refresh_btn = ctk.CTkButton(
            left_frame,
            text="🔄 Refresh",
            command=self.refresh,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        )
        self.refresh_btn.pack(side="left", padx=(0, SPACE_SM))

        self.explore_btn = ctk.CTkButton(
            left_frame,
            text="🔍 Explore Connected",
            command=self._explore_connected,
            width=BUTTON_LG[0],
            height=BUTTON_LG[1],
            state="disabled"
        )
        self.explore_btn.pack(side="left", padx=(0, SPACE_SM))

        # Path finding button
        self.path_btn = ctk.CTkButton(
            left_frame,
            text="🧭 Find Path",
            command=self._find_path,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            state="disabled"
        )
        self.path_btn.pack(side="left", padx=(0, SPACE_SM))

        # Right side buttons
        right_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
        right_frame.pack(side="right", padx=SPACE_SM, pady=SPACE_SM)

        self.export_btn = ctk.CTkButton(
            right_frame,
            text="💾 Export JSON",
            command=self._export_graph,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        )
        self.export_btn.pack(side="left", padx=(0, SPACE_SM))

        self.delete_btn = ctk.CTkButton(
            right_frame,
            text="🗑️ Delete",
            command=self._delete_relationship,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            state="disabled",
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#b91c1c"
        )
        self.delete_btn.pack(side="left")

    # Event Handlers

    def _on_selection_change(self, event=None):
        """Handle selection change in treeview."""
        self._update_button_states()

    def _on_search(self, query: str):
        """Handle search query."""
        if not query.strip():
            self.refresh()
        else:
            self._search_relationships(query)

    def _on_clear_search(self):
        """Handle search clear."""
        self.refresh()

    def _on_filter_change(self, _=None):
        """Handle filter change."""
        self.refresh()

    def _on_strength_change(self, value):
        """Handle strength slider change."""
        self.strength_label.configure(text=f"{float(value):.1f}")
        self.refresh()

    def _on_double_click(self, event):
        """Handle double-click on relationship."""
        selection = self.tree.selection()
        if selection:
            self._explore_connected()

    def _on_right_click(self, event):
        """Handle right-click context menu."""
        # Identify the clicked item
        item = self.tree.identify_row(event.y)
        if not item:
            return

        # Select the item
        self.tree.selection_set(item)

        # Create context menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Explore Connected Concepts", command=self._explore_connected)
        menu.add_command(label="Find Path to Another Concept", command=self._find_path)
        menu.add_separator()
        menu.add_command(label="Copy Source Concept", command=lambda: self._copy_to_clipboard(0))
        menu.add_command(label="Copy Target Concept", command=lambda: self._copy_to_clipboard(2))
        menu.add_separator()
        menu.add_command(label="Adjust Strength", command=self._adjust_strength)
        menu.add_command(label="Delete Relationship", command=self._delete_relationship)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # Core Operations

    def refresh(self):
        """Refresh the relationships display."""
        if not self.knowledge_store:
            self.status_label.configure(text="Knowledge store not available")
            return

        try:
            # Clear existing items
            self.tree.clear()

            # Get filter values
            rel_type = self.type_var.get()
            min_strength = self.strength_var.get()
            search_query = self.search_entry.get().strip()

            # Query relationships from knowledge_relationships table
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Check if knowledge_relationships table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)

            if not cursor.fetchone():
                # Fallback to related_entries_json method
                self._refresh_legacy_relationships(conn)
                conn.close()
                return

            # Build query with filters
            query = """
                SELECT
                    kr.source_id,
                    kr.target_id,
                    kr.relationship_type,
                    kr.confidence,
                    ke1.content_json as source_content,
                    ke2.content_json as target_content
                FROM knowledge_relationships kr
                JOIN knowledge_entries ke1 ON kr.source_id = ke1.knowledge_id
                JOIN knowledge_entries ke2 ON kr.target_id = ke2.knowledge_id
                WHERE kr.confidence >= ?
            """
            params = [min_strength]

            if rel_type != "all":
                query += " AND kr.relationship_type = ?"
                params.append(rel_type)

            if search_query:
                query += """ AND (
                    json_extract(ke1.content_json, '$.concept') LIKE ? OR
                    json_extract(ke2.content_json, '$.concept') LIKE ?
                )"""
                search_pattern = f"%{search_query}%"
                params.extend([search_pattern, search_pattern])

            query += " ORDER BY kr.confidence DESC LIMIT 500"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Statistics counters
            stats = {
                "total": 0,
                "explicit_mention": 0,
                "embedding_similarity": 0,
                "co_occurrence": 0
            }

            # Populate treeview
            for row in rows:
                source_id, target_id, rel_type_db, strength, source_json, target_json = row

                try:
                    source_content = json.loads(source_json) if source_json else {}
                    target_content = json.loads(target_json) if target_json else {}

                    source_concept = source_content.get("concept", "Unknown")
                    target_concept = target_content.get("concept", "Unknown")

                    # Determine evidence/basis
                    evidence = self._get_evidence_label(rel_type_db, strength)

                    # Insert into treeview
                    item_id = self.tree.insert(
                        "",
                        "end",
                        values=(
                            source_concept,
                            self._format_rel_type(rel_type_db),
                            target_concept,
                            f"{strength:.2f}",
                            evidence
                        ),
                        tags=[rel_type_db, source_id, target_id]
                    )

                    # Update statistics
                    stats["total"] += 1
                    if rel_type_db in stats:
                        stats[rel_type_db] += 1

                except Exception as e:
                    logger.warning(f"Failed to parse relationship row: {e}")
                    continue

            conn.close()

            # Update statistics cards
            self._update_statistics(stats)

            # Update status
            self.status_label.configure(
                text=f"Showing {stats['total']} relationships"
            )

        except Exception as e:
            logger.error(f"Failed to refresh relationships: {e}")
            self.status_label.configure(text=f"Error: {str(e)}")

    def _refresh_legacy_relationships(self, conn):
        """Fallback method using related_entries_json for older databases."""
        try:
            rel_type_filter = self.type_var.get()
            min_strength = self.strength_var.get()

            # Query entries with relationships
            query = """
                SELECT knowledge_id, content_json, domain, related_entries_json
                FROM knowledge_entries
                WHERE related_entries_json IS NOT NULL
                AND related_entries_json != '[]'
                AND json_extract(content_json, '$.concept') IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 500
            """

            cursor = conn.execute(query)
            relationships_found = 0

            for row in cursor:
                knowledge_id, content_json, domain, related_json = row

                try:
                    content = json.loads(content_json) if content_json else {}
                    related_ids = json.loads(related_json) if related_json else []

                    source_concept = content.get("concept", "Unknown")

                    if not related_ids:
                        continue

                    # For each related entry
                    for related_id in related_ids[:10]:
                        rel_cursor = conn.execute(
                            "SELECT content_json, domain FROM knowledge_entries WHERE knowledge_id = ?",
                            (related_id,)
                        )
                        rel_row = rel_cursor.fetchone()

                        if rel_row:
                            rel_content = json.loads(rel_row[0]) if rel_row[0] else {}
                            target_concept = rel_content.get("concept", "Unknown")

                            # Insert into treeview (legacy format)
                            self.tree.insert(
                                "",
                                "end",
                                values=(
                                    source_concept,
                                    "related_to",
                                    target_concept,
                                    "N/A",
                                    f"{len(related_ids)} total connections"
                                ),
                                tags=["legacy", knowledge_id, related_id]
                            )

                            relationships_found += 1

                except Exception as e:
                    logger.warning(f"Failed to parse legacy relationship: {e}")
                    continue

            self.status_label.configure(
                text=f"Showing {relationships_found} relationships (legacy format)"
            )

        except Exception as e:
            logger.error(f"Failed to refresh legacy relationships: {e}")
            self.status_label.configure(text=f"Error: {str(e)}")

    def _search_relationships(self, query: str):
        """Search for relationships involving a specific concept."""
        # The refresh method already handles search queries
        self.refresh()

    def _explore_connected(self):
        """Explore all concepts connected to the selected concept."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a relationship to explore")
            return

        if not self.knowledge_retriever:
            messagebox.showwarning("Not Available", "Knowledge retriever not available")
            return

        try:
            # Get source knowledge_id from tags
            item = self.tree.item(selection[0])
            tags = item["tags"]

            if not tags or len(tags) < 2:
                messagebox.showwarning("Error", "Invalid relationship data")
                return

            source_id = tags[1]  # tags[0] is rel_type
            source_concept = item["values"][0]

            # Get depth
            depth_str = self.depth_var.get()
            depth = int(depth_str.split("-")[0])

            # Retrieve connected concepts
            related_results = self.knowledge_retriever.get_related_concepts(source_id, max_depth=depth)

            # Create popup window
            dialog = tk.Toplevel(self)
            dialog.title(f"Connected Concepts: {source_concept}")
            dialog.geometry("900x600")
            dialog.transient(self)
            dialog.after(100, lambda: self._safe_grab(dialog))

            # Title
            title_label = ctk.CTkLabel(
                dialog,
                text=f"Exploring connections for: {source_concept}",
                font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
            )
            title_label.pack(pady=SPACE_MD)

            # Subtitle
            subtitle_label = ctk.CTkLabel(
                dialog,
                text=f"Found {len(related_results)} concepts within {depth} hop(s)",
                font=ctk.CTkFont(size=FONT_BODY),
                text_color=self.theme_manager.get_color("fg_muted")
            )
            subtitle_label.pack(pady=(0, SPACE_SM))

            # Text display
            text_frame = ctk.CTkFrame(dialog)
            text_frame.pack(fill="both", expand=True, padx=SPACE_LG, pady=(0, SPACE_SM))

            text_widget = tk.Text(
                text_frame,
                wrap="word",
                font=("Monospace", 10),
                bg=self.theme_manager.colors["bg_primary"],
                fg=self.theme_manager.colors["fg_primary"]
            )
            scrollbar = ctk.CTkScrollbar(text_frame, orientation="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Build network display
            network_text = f"SOURCE CONCEPT: {source_concept}\n"
            network_text += "=" * 80 + "\n\n"

            if related_results:
                for i, result in enumerate(related_results, 1):
                    concept_name = result.content.get("concept", "Unknown")
                    definition = result.content.get("definition", "No definition")
                    score = result.relevance_score
                    domain = result.domain

                    network_text += f"{i}. {concept_name} (relevance: {score:.2f})\n"
                    network_text += f"   Domain: {domain}\n"
                    network_text += f"   {definition[:200]}{'...' if len(definition) > 200 else ''}\n\n"
            else:
                network_text += "No connected concepts found within the specified depth.\n\n"
                network_text += "This could mean:\n"
                network_text += "  • This concept has no relationships in the knowledge graph\n"
                network_text += "  • Relationships haven't been built yet (run refinement)\n"
                network_text += "  • Try increasing the hop depth setting\n\n"
                network_text += "Tip: Go to the Control tab and click 'Force Refinement'\n"
                network_text += "to rebuild the knowledge graph relationships.\n"

            text_widget.insert("1.0", network_text)
            text_widget.configure(state="disabled")

            # Close button
            close_btn = ctk.CTkButton(
                dialog,
                text="Close",
                command=dialog.destroy,
                width=BUTTON_SM[0],
                height=BUTTON_SM[1]
            )
            close_btn.pack(pady=SPACE_SM)

        except Exception as e:
            logger.error(f"Failed to explore connected concepts: {e}")
            messagebox.showerror("Error", f"Failed to explore connections:\n{str(e)}")

    def _find_path(self):
        """Find path between two concepts."""
        # Get selected source
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a source relationship first")
            return

        item = self.tree.item(selection[0])
        source_concept = item["values"][0]

        # Prompt for target concept
        from tkinter import simpledialog
        target_concept = simpledialog.askstring(
            "Find Path",
            f"Find path from '{source_concept}' to:",
            parent=self
        )

        if not target_concept:
            return

        # TODO: Implement path finding algorithm
        # For now, show placeholder message
        messagebox.showinfo(
            "Path Finding",
            f"Path finding from '{source_concept}' to '{target_concept}'\n\n"
            "This feature is coming soon!"
        )

    def _adjust_strength(self):
        """Adjust the strength of selected relationship."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a relationship to adjust")
            return

        item = self.tree.item(selection[0])
        current_strength = float(item["values"][3]) if item["values"][3] != "N/A" else 0.5

        # Prompt for new strength
        from tkinter import simpledialog
        new_strength = simpledialog.askfloat(
            "Adjust Strength",
            f"Current strength: {current_strength:.2f}\n\nEnter new strength (0.0 - 1.0):",
            minvalue=0.0,
            maxvalue=1.0,
            initialvalue=current_strength,
            parent=self
        )

        if new_strength is None:
            return

        try:
            # Update in database
            tags = item["tags"]
            if len(tags) >= 3:
                source_id = tags[1]
                target_id = tags[2]

                conn = sqlite3.connect(self.knowledge_store.storage_path)
                conn.execute("""
                    UPDATE knowledge_relationships
                    SET confidence = ?
                    WHERE source_id = ? AND target_id = ?
                """, (new_strength, source_id, target_id))
                conn.commit()
                conn.close()

                messagebox.showinfo("Success", "Relationship strength updated")
                self.refresh()
            else:
                messagebox.showwarning("Error", "Cannot adjust legacy relationship format")

        except Exception as e:
            logger.error(f"Failed to adjust strength: {e}")
            messagebox.showerror("Error", f"Failed to update strength:\n{str(e)}")

    def _delete_relationship(self):
        """Delete selected relationship(s)."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select relationship(s) to delete")
            return

        count = len(selection)
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {count} relationship(s)?\n\nThis action cannot be undone."
        )

        if not confirm:
            return

        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            deleted = 0

            for item_id in selection:
                item = self.tree.item(item_id)
                tags = item["tags"]

                if len(tags) >= 3:
                    source_id = tags[1]
                    target_id = tags[2]

                    conn.execute("""
                        DELETE FROM knowledge_relationships
                        WHERE source_id = ? AND target_id = ?
                    """, (source_id, target_id))
                    deleted += 1

            conn.commit()
            conn.close()

            messagebox.showinfo("Success", f"Deleted {deleted} relationship(s)")
            self.refresh()

        except Exception as e:
            logger.error(f"Failed to delete relationships: {e}")
            messagebox.showerror("Error", f"Failed to delete:\n{str(e)}")

    def _export_graph(self):
        """Export graph data to JSON."""
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            title="Export Graph Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        try:
            # Collect all relationships
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Check for modern table
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)

            if cursor.fetchone():
                cursor.execute("""
                    SELECT
                        kr.source_id,
                        kr.target_id,
                        kr.relationship_type,
                        kr.confidence,
                        ke1.content_json as source_content,
                        ke2.content_json as target_content
                    FROM knowledge_relationships kr
                    JOIN knowledge_entries ke1 ON kr.source_id = ke1.knowledge_id
                    JOIN knowledge_entries ke2 ON kr.target_id = ke2.knowledge_id
                """)

                relationships = []
                for row in cursor.fetchall():
                    source_id, target_id, rel_type, confidence, source_json, target_json = row

                    source_content = json.loads(source_json) if source_json else {}
                    target_content = json.loads(target_json) if target_json else {}

                    relationships.append({
                        "source_id": source_id,
                        "source_concept": source_content.get("concept", "Unknown"),
                        "target_id": target_id,
                        "target_concept": target_content.get("concept", "Unknown"),
                        "relationship_type": rel_type,
                        "strength": confidence
                    })

                export_data = {
                    "format": "felix_knowledge_graph",
                    "version": "1.0",
                    "relationship_count": len(relationships),
                    "relationships": relationships
                }

            else:
                export_data = {
                    "format": "felix_knowledge_graph",
                    "version": "legacy",
                    "note": "This database uses legacy relationship format",
                    "relationships": []
                }

            conn.close()

            # Write to file
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            messagebox.showinfo("Success", f"Exported {len(export_data.get('relationships', []))} relationships to:\n{filepath}")

        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            messagebox.showerror("Error", f"Failed to export:\n{str(e)}")

    def _copy_to_clipboard(self, column_index: int):
        """Copy concept name to clipboard."""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            value = item["values"][column_index]
            self.clipboard_clear()
            self.clipboard_append(value)

    # Helper Methods

    def _update_statistics(self, stats: Dict[str, int]):
        """Update statistics cards."""
        self.total_card.set_value(str(stats["total"]))
        self.explicit_card.set_value(str(stats.get("explicit_mention", 0)))
        self.similarity_card.set_value(str(stats.get("embedding_similarity", 0)))
        self.cooccurrence_card.set_value(str(stats.get("co_occurrence", 0)))

    def _format_rel_type(self, rel_type: str) -> str:
        """Format relationship type for display."""
        return rel_type.replace("_", " ").title()

    def _get_evidence_label(self, rel_type: str, strength: float) -> str:
        """Generate evidence label based on relationship type and strength."""
        if rel_type == "explicit_mention":
            return "Direct mention in text"
        elif rel_type == "embedding_similarity":
            return f"Semantic similarity ({strength:.0%})"
        elif rel_type == "co_occurrence":
            return "Co-occurs in documents"
        else:
            return "Related concept"

    # Public Interface

    def set_knowledge_refs(
        self,
        knowledge_store=None,
        knowledge_retriever=None,
        knowledge_daemon=None
    ):
        """Set references to knowledge brain components."""
        self.knowledge_store = knowledge_store
        self.knowledge_retriever = knowledge_retriever
        self.knowledge_daemon = knowledge_daemon

        # Enable/disable buttons based on availability
        self._update_button_states()

        # Defer initial refresh to avoid blocking GUI thread
        if self.knowledge_store:
            self.after(300, self._async_initial_refresh)

    def _async_initial_refresh(self):
        """Run initial refresh in background thread to avoid blocking GUI."""
        if self.thread_manager:
            self.thread_manager.start_thread(self._refresh_background)
        else:
            # Fallback to sync refresh if no thread_manager
            self.refresh()

    def _refresh_background(self):
        """
        Background thread: fetch relationship data then schedule UI updates on main thread.

        This prevents heavy JOIN queries from blocking the GUI.
        """
        if not self.knowledge_store:
            self.after(0, lambda: self.status_label.configure(text="Knowledge store not available"))
            return

        try:
            # Get filter values (these are thread-safe reads)
            rel_type = self.type_var.get()
            min_strength = self.strength_var.get()
            search_query = self.search_entry.get().strip()

            rows = []
            is_legacy = False
            stats = {
                "total": 0,
                "explicit_mention": 0,
                "embedding_similarity": 0,
                "co_occurrence": 0
            }

            # Query relationships from database
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Check if knowledge_relationships table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)

            if not cursor.fetchone():
                # Legacy mode - fetch from related_entries_json
                is_legacy = True
                rows = self._fetch_legacy_relationships(conn, rel_type, min_strength)
            else:
                # Modern mode - use knowledge_relationships table
                query = """
                    SELECT
                        kr.source_id,
                        kr.target_id,
                        kr.relationship_type,
                        kr.confidence,
                        ke1.content_json as source_content,
                        ke2.content_json as target_content
                    FROM knowledge_relationships kr
                    JOIN knowledge_entries ke1 ON kr.source_id = ke1.knowledge_id
                    JOIN knowledge_entries ke2 ON kr.target_id = ke2.knowledge_id
                    WHERE kr.confidence >= ?
                """
                params = [min_strength]

                if rel_type != "all":
                    query += " AND kr.relationship_type = ?"
                    params.append(rel_type)

                if search_query:
                    query += """ AND (
                        json_extract(ke1.content_json, '$.concept') LIKE ? OR
                        json_extract(ke2.content_json, '$.concept') LIKE ?
                    )"""
                    search_pattern = f"%{search_query}%"
                    params.extend([search_pattern, search_pattern])

                query += " ORDER BY kr.confidence DESC LIMIT 500"

                cursor.execute(query, params)
                rows = cursor.fetchall()

            conn.close()

            # Schedule UI update on main thread
            self.after(0, lambda: self._update_relationships_ui(rows, is_legacy, stats))

        except Exception as e:
            logger.error(f"Background refresh failed: {e}")
            self.after(0, lambda: self.status_label.configure(text=f"Error: {str(e)}"))

    def _fetch_legacy_relationships(self, conn, rel_type_filter, min_strength):
        """Fetch relationships from legacy related_entries_json format."""
        query = """
            SELECT knowledge_id, content_json, domain, related_entries_json
            FROM knowledge_entries
            WHERE related_entries_json IS NOT NULL
            AND related_entries_json != '[]'
            AND json_extract(content_json, '$.concept') IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 500
        """
        cursor = conn.execute(query)
        return list(cursor.fetchall())

    def _update_relationships_ui(self, rows, is_legacy, stats):
        """Update relationships treeview on main thread with pre-fetched data."""
        try:
            # Clear existing items
            self.tree.clear()

            if is_legacy:
                # Process legacy format
                relationships_found = 0
                for row in rows:
                    knowledge_id, content_json, domain, related_json = row
                    try:
                        content = json.loads(content_json) if content_json else {}
                        related_ids = json.loads(related_json) if related_json else []
                        source_concept = content.get("concept", "Unknown")

                        if not related_ids:
                            continue

                        for related_id in related_ids[:10]:
                            # For legacy, we can't easily fetch target without another query
                            # Show simplified view
                            self.tree.insert(
                                "",
                                "end",
                                values=(
                                    source_concept,
                                    "related_to",
                                    f"(ID: {related_id[:8]}...)" if len(related_id) > 8 else related_id,
                                    "N/A",
                                    f"{len(related_ids)} total connections"
                                ),
                                tags=["legacy", knowledge_id, related_id]
                            )
                            relationships_found += 1

                    except Exception as e:
                        logger.warning(f"Failed to parse legacy relationship: {e}")
                        continue

                self.status_label.configure(
                    text=f"Showing {relationships_found} relationships (legacy format)"
                )
            else:
                # Process modern format
                for row in rows:
                    source_id, target_id, rel_type_db, strength, source_json, target_json = row

                    try:
                        source_content = json.loads(source_json) if source_json else {}
                        target_content = json.loads(target_json) if target_json else {}

                        source_concept = source_content.get("concept", "Unknown")
                        target_concept = target_content.get("concept", "Unknown")

                        evidence = self._get_evidence_label(rel_type_db, strength)

                        self.tree.insert(
                            "",
                            "end",
                            values=(
                                source_concept,
                                self._format_rel_type(rel_type_db),
                                target_concept,
                                f"{strength:.2f}",
                                evidence
                            ),
                            tags=[rel_type_db, source_id, target_id]
                        )

                        stats["total"] += 1
                        if rel_type_db in stats:
                            stats[rel_type_db] += 1

                    except Exception as e:
                        logger.warning(f"Failed to parse relationship row: {e}")
                        continue

                # Update statistics cards
                self._update_statistics(stats)

                self.status_label.configure(
                    text=f"Showing {stats['total']} relationships"
                )

        except Exception as e:
            logger.error(f"Failed to update relationships UI: {e}")
            self.status_label.configure(text=f"Error: {str(e)}")

    def _update_button_states(self):
        """Update button states based on component availability."""
        has_store = self.knowledge_store is not None
        has_retriever = self.knowledge_retriever is not None

        # Update selection-dependent buttons
        selection = self.tree.selection()
        has_selection = len(selection) > 0

        state = "normal" if (has_retriever and has_selection) else "disabled"
        self.explore_btn.configure(state=state)
        self.path_btn.configure(state=state)

        state = "normal" if (has_store and has_selection) else "disabled"
        self.delete_btn.configure(state=state)

    def _enable_features(self):
        """Enable features when Felix system starts."""
        felix_system = getattr(self.main_app, 'felix_system', None) if self.main_app else None
        if felix_system:
            # Wire up references to knowledge brain components
            self.knowledge_store = getattr(felix_system, 'knowledge_store', None)
            self.knowledge_retriever = getattr(felix_system, 'knowledge_retriever', None)
            self.knowledge_daemon = getattr(felix_system, 'knowledge_daemon', None)

        self._update_button_states()
        # Defer refresh to avoid blocking GUI thread during startup
        self.after(500, self._async_initial_refresh)
        logger.info("Relationships panel features enabled")

    def _disable_features(self):
        """Disable features when Felix system stops."""
        self.knowledge_store = None
        self.knowledge_retriever = None
        self.knowledge_daemon = None
        self._update_button_states()
        self.tree.clear()
        self.status_label.configure(text="Knowledge Brain not available")
        logger.info("Relationships panel features disabled")

    def _safe_grab(self, dialog):
        """Safely grab focus after window is rendered."""
        try:
            dialog.grab_set()
            dialog.focus_set()
        except Exception as e:
            logger.warning(f"Could not grab dialog focus: {e}")
