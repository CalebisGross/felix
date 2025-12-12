"""
Memory Tab for Felix GUI (CustomTkinter Edition)

The Memory tab provides:
- Nested tabs for Memory (tasks), Knowledge, and Workflow History
- Entry browsing with search and filter capabilities
- Detail view and edit/delete operations
- Thread-safe database operations
"""

import customtkinter as ctk
from tkinter import ttk, messagebox
import tkinter as tk
from typing import Optional, List, Tuple
import json
import queue
import logging
import sqlite3
from datetime import datetime

from ..utils import ThreadManager, DBHelper, logger
from ..theme_manager import get_theme_manager
from ..components.themed_treeview import ThemedTreeview
from ..styles import (
    BUTTON_SM, BUTTON_MD,
    FONT_SECTION, FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_MD
)

# Try to import Felix memory modules
try:
    from src.memory import knowledge_store, task_memory
except ImportError as e:
    logger.error(f"Failed to import memory modules: {e}")
    knowledge_store = None
    task_memory = None

try:
    from src.memory.workflow_history import WorkflowHistory
except ImportError as e:
    logger.error(f"Failed to import workflow history: {e}")
    WorkflowHistory = None


class MemoryTab(ctk.CTkFrame):
    """
    Main Memory tab with nested sub-tabs for different memory types.
    """

    def __init__(self, master, thread_manager, db_helper, main_app=None, **kwargs):
        """
        Initialize Memory tab.

        Args:
            master: Parent widget
            thread_manager: ThreadManager instance
            db_helper: DBHelper instance
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.db_helper = db_helper
        self.main_app = main_app
        self.theme_manager = get_theme_manager()
        self._layout_manager = None

        self._setup_ui()

    def set_layout_manager(self, layout_manager):
        """
        Set the layout manager for responsive updates.
        MemoryTab handles its own internal responsive layouts.

        Args:
            layout_manager: ResponsiveLayoutManager instance from main app
        """
        self._layout_manager = layout_manager
        # Register callback for breakpoint changes
        layout_manager.register_callback(self._on_breakpoint_change)
        # Trigger initial layout
        current_breakpoint = layout_manager.get_current_breakpoint()
        current_config = layout_manager.get_current_config()
        self._on_breakpoint_change(current_breakpoint, current_config)

    def _on_breakpoint_change(self, breakpoint, config):
        """
        Handle breakpoint changes.
        Forward to subtabs that support responsive layouts.
        """
        # Forward breakpoint changes to subtabs if they support it
        if hasattr(self, 'memory_subtab') and hasattr(self.memory_subtab, '_build_layout'):
            mode = breakpoint.value if hasattr(breakpoint, 'value') else str(breakpoint)
            self.memory_subtab._build_layout(mode)
        if hasattr(self, 'knowledge_subtab') and hasattr(self.knowledge_subtab, '_build_layout'):
            mode = breakpoint.value if hasattr(breakpoint, 'value') else str(breakpoint)
            self.knowledge_subtab._build_layout(mode)

    def _setup_ui(self):
        """Set up the UI with nested tabs."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create nested tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)

        # Add sub-tabs
        self.tabview.add("Memory")
        self.tabview.add("Knowledge")
        self.tabview.add("Workflow History")

        # Create sub-tab content
        self.memory_subtab = MemorySubTab(
            self.tabview.tab("Memory"),
            self.thread_manager,
            self.db_helper,
            "felix_memory.db",
            "tasks"
        )
        self.memory_subtab.main_app = self.main_app  # Set reference to main app
        self.memory_subtab.pack(fill="both", expand=True)

        self.knowledge_subtab = MemorySubTab(
            self.tabview.tab("Knowledge"),
            self.thread_manager,
            self.db_helper,
            "felix_knowledge.db",
            "knowledge"
        )
        self.knowledge_subtab.main_app = self.main_app  # Set reference to main app
        self.knowledge_subtab.pack(fill="both", expand=True)

        self.workflow_history_subtab = WorkflowHistorySubTab(
            self.tabview.tab("Workflow History"),
            self.thread_manager
        )
        self.workflow_history_subtab.main_app = self.main_app  # Set reference to main app
        self.workflow_history_subtab.pack(fill="both", expand=True)

    def refresh_all(self):
        """Refresh all sub-tabs (called after workflow completion)."""
        self.memory_subtab.refresh_entries()
        self.knowledge_subtab.refresh_entries()
        self.workflow_history_subtab.refresh_workflows()


class MemorySubTab(ctk.CTkFrame):
    """
    Sub-tab for displaying memory entries (tasks or knowledge) with master-detail-sidebar layout.
    Responsive layout adapts from single column (compact) to 3-column (ultrawide).
    """

    def __init__(self, master, thread_manager, db_helper, db_name, table_name, **kwargs):
        """
        Initialize Memory sub-tab.

        Args:
            master: Parent widget
            thread_manager: ThreadManager instance
            db_helper: DBHelper instance
            db_name: Database file name
            table_name: Table name ("tasks" or "knowledge")
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.db_helper = db_helper
        self.db_name = db_name
        self.table_name = table_name
        self.theme_manager = get_theme_manager()
        self.entries = []  # List of (id, content, display) tuples
        self.main_app = None  # Will be set by MemoryTab
        self.selected_entry_id = None

        # Queue for thread-safe communication
        self.result_queue = queue.Queue()

        # Lazy loading flag - don't load until tab is visible
        self._data_loaded = False

        # Determine initial layout (fallback if no responsive manager)
        self.current_layout = "standard"  # compact, standard, wide, ultrawide

        self._setup_ui()
        self._start_polling()
        # Removed immediate load_entries() - now loads on first visibility

    def _setup_ui(self):
        """Set up the UI components with responsive master-detail-sidebar layout."""
        # Configure grid for main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew")

        # Build initial layout (will be updated by responsive manager if available)
        self._build_layout("standard")

    def _build_layout(self, layout_mode: str):
        """Build the layout based on breakpoint mode."""
        self.current_layout = layout_mode

        # Clear existing layout
        for widget in self.main_container.winfo_children():
            widget.destroy()

        if layout_mode == "ultrawide":
            self._build_three_column_layout()
        elif layout_mode in ("wide", "standard"):
            self._build_two_column_layout()
        else:  # compact
            self._build_single_column_layout()

    def _build_three_column_layout(self):
        """Build 3-column layout: list | detail | related sidebar."""
        self.main_container.grid_columnconfigure(0, weight=0, minsize=300)  # List (fixed width)
        self.main_container.grid_columnconfigure(1, weight=1)  # Detail (expands)
        self.main_container.grid_columnconfigure(2, weight=0, minsize=250)  # Related (fixed width)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Left: List panel
        list_panel = self._create_list_panel(self.main_container)
        list_panel.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_XS), pady=0)

        # Center: Detail panel
        detail_panel = self._create_detail_panel(self.main_container)
        detail_panel.grid(row=0, column=1, sticky="nsew", padx=SPACE_XS, pady=0)

        # Right: Related sidebar
        related_panel = self._create_related_panel(self.main_container)
        related_panel.grid(row=0, column=2, sticky="nsew", padx=(SPACE_XS, 0), pady=0)

    def _build_two_column_layout(self):
        """Build 2-column layout: list | detail (related in expandable section)."""
        self.main_container.grid_columnconfigure(0, weight=0, minsize=300)  # List
        self.main_container.grid_columnconfigure(1, weight=1)  # Detail
        self.main_container.grid_rowconfigure(0, weight=1)

        # Left: List panel
        list_panel = self._create_list_panel(self.main_container)
        list_panel.grid(row=0, column=0, sticky="nsew", padx=(0, SPACE_XS), pady=0)

        # Right: Detail panel (with related section at bottom)
        detail_panel = self._create_detail_panel_with_related(self.main_container)
        detail_panel.grid(row=0, column=1, sticky="nsew", padx=(SPACE_XS, 0), pady=0)

    def _build_single_column_layout(self):
        """Build single column layout for compact screens."""
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Create tabbed interface for compact mode
        compact_tabview = ctk.CTkTabview(self.main_container)
        compact_tabview.grid(row=0, column=0, sticky="nsew")

        compact_tabview.add("List")
        compact_tabview.add("Detail")

        # List tab
        list_panel = self._create_list_panel(compact_tabview.tab("List"))
        list_panel.pack(fill="both", expand=True)

        # Detail tab
        detail_panel = self._create_detail_panel_with_related(compact_tabview.tab("Detail"))
        detail_panel.pack(fill="both", expand=True)

    def _create_list_panel(self, parent):
        """Create list panel with search and entries."""
        panel = ctk.CTkFrame(parent)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Search section
        search_frame = ctk.CTkFrame(panel, fg_color="transparent")
        search_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        search_frame.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search entries...")
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_XS))
        self.search_entry.bind('<Return>', lambda e: self.search())

        button_row = ctk.CTkFrame(search_frame, fg_color="transparent")
        button_row.grid(row=1, column=0, sticky="ew", pady=(SPACE_XS, 0))

        self.search_button = ctk.CTkButton(
            button_row,
            text="Search",
            command=self.search,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.search_button.pack(side="left", padx=(0, SPACE_XS))

        self.clear_button = ctk.CTkButton(
            button_row,
            text="Clear",
            command=self.clear_search,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color="transparent",
            border_width=1
        )
        self.clear_button.pack(side="left")

        # Count label
        self.count_label = ctk.CTkLabel(button_row, text="0 entries", font=ctk.CTkFont(size=FONT_CAPTION))
        self.count_label.pack(side="right", padx=SPACE_XS)

        # Entry list
        self.list_scrollable = ctk.CTkScrollableFrame(panel)
        self.list_scrollable.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        self.list_scrollable.grid_columnconfigure(0, weight=1)

        self.entry_buttons = []

        return panel

    def _create_detail_panel(self, parent):
        """Create detail panel with full preview."""
        panel = ctk.CTkFrame(parent)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(header_frame, text="Details", font=ctk.CTkFont(size=FONT_SECTION, weight="bold")).pack(side="left")

        # Action buttons on right
        actions = ctk.CTkFrame(header_frame, fg_color="transparent")
        actions.pack(side="right")

        self.delete_button = ctk.CTkButton(
            actions,
            text="Delete",
            command=self.delete_entry,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226"
        )
        self.delete_button.pack(side="left", padx=SPACE_XS)

        self.refresh_button = ctk.CTkButton(
            actions,
            text="Refresh",
            command=self.refresh_entries,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.refresh_button.pack(side="left")

        # Details textbox
        self.details_textbox = ctk.CTkTextbox(
            panel,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.details_textbox.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))

        return panel

    def _create_detail_panel_with_related(self, parent):
        """Create detail panel with related section at bottom (2-column mode)."""
        panel = ctk.CTkFrame(parent)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header
        header_frame = ctk.CTkFrame(panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(header_frame, text="Details", font=ctk.CTkFont(size=FONT_SECTION, weight="bold")).pack(side="left")

        # Action buttons
        actions = ctk.CTkFrame(header_frame, fg_color="transparent")
        actions.pack(side="right")

        self.delete_button = ctk.CTkButton(
            actions,
            text="Delete",
            command=self.delete_entry,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226"
        )
        self.delete_button.pack(side="left", padx=SPACE_XS)

        self.refresh_button = ctk.CTkButton(
            actions,
            text="Refresh",
            command=self.refresh_entries,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.refresh_button.pack(side="left")

        # Details textbox
        self.details_textbox = ctk.CTkTextbox(
            panel,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.details_textbox.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_XS))

        # Related entries section (expandable)
        related_frame = ctk.CTkFrame(panel)
        related_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))
        related_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            related_frame,
            text="Related Entries (coming soon)",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color="gray"
        ).grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=SPACE_SM)

        return panel

    def _create_related_panel(self, parent):
        """Create related entries sidebar (ultrawide mode only)."""
        panel = ctk.CTkFrame(parent)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header
        ctk.CTkLabel(
            panel,
            text="Related",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Placeholder for related entries
        placeholder = ctk.CTkTextbox(panel, height=100, font=ctk.CTkFont(size=FONT_CAPTION))
        placeholder.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        placeholder.insert("1.0", "Related entries will appear here when available.")
        placeholder.configure(state="disabled")

        return panel

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                # Check if main Memory tab is visible
                if self.main_app.tabview.get() != "Memory":
                    return False
                # Check if this specific sub-tab is visible
                parent_memory_tab = self.winfo_parent()
                if hasattr(self.master, 'master') and hasattr(self.master.master, 'get'):
                    # Get the nested tabview's current tab
                    current_subtab = self.master.master.get()
                    # Match table_name to subtab name
                    if self.table_name == 'tasks' and current_subtab != "Memory":
                        return False
                    elif self.table_name == 'knowledge' and current_subtab != "Knowledge":
                        return False
        except Exception:
            pass
        return True

    def _start_polling(self):
        """Start polling the result queue."""
        self._poll_results()

    def _poll_results(self):
        """Poll the result queue and update GUI (runs on main thread)."""
        # Check if shutdown signaled
        if not self.thread_manager.is_active:
            return  # Stop polling

        try:
            is_visible = self._is_visible()

            # Lazy load on first visibility
            if is_visible and not self._data_loaded:
                self._data_loaded = True
                self.load_entries()

            # Only process results if this tab is visible
            if is_visible:
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    action = result.get('action')

                    if action == 'update_list':
                        self._update_entry_list()
                    elif action == 'display_details':
                        self._display_details(result['entry_id'], result['content'])
                    elif action == 'show_no_entries':
                        self._show_no_entries_message()
                    elif action == 'show_error':
                        messagebox.showerror("Error", result['message'])
                    elif action == 'show_warning':
                        messagebox.showwarning(result['title'], result['message'])
                    elif action == 'show_info':
                        messagebox.showinfo(result['title'], result['message'])
                    elif action == 'reload':
                        self.load_entries()

        except Exception as e:
            logger.error(f"Error in poll_results: {e}", exc_info=True)

        # Schedule next poll - slower when not visible
        if self.thread_manager.is_active:
            poll_interval = 100 if self._is_visible() else 500
            self.after(poll_interval, self._poll_results)

    def load_entries(self):
        """Load entries from database."""
        self.thread_manager.start_thread(self._load_entries_thread)

    def refresh_entries(self):
        """Manually refresh the entries list."""
        self.load_entries()

    def _load_entries_thread(self):
        """Background thread to load entries."""
        try:
            if self.table_name == 'knowledge' and knowledge_store:
                # Use KnowledgeStore API
                ks = knowledge_store.KnowledgeStore(self.db_name)
                query = knowledge_store.KnowledgeQuery(limit=100)
                entries = ks.retrieve_knowledge(query)

                self.entries = []
                for e in entries:
                    content_str = json.dumps(e.content) if isinstance(e.content, dict) else str(e.content)
                    display = f"{e.domain}: {content_str[:100]}..."
                    self.entries.append((e.knowledge_id, content_str, display))

            elif self.table_name == 'tasks' and task_memory:
                # Use TaskMemory API
                tm = task_memory.TaskMemory(self.db_name)
                patterns = tm.get_patterns(task_memory.TaskMemoryQuery(limit=100))

                self.entries = []
                for p in patterns:
                    keywords_str = ", ".join(p.keywords) if p.keywords else "No keywords"
                    full_content = json.dumps({
                        "keywords": p.keywords,
                        "task_type": p.task_type,
                        "success_rate": p.success_rate,
                        "typical_duration": p.typical_duration
                    })
                    display = f"{p.task_type}: {keywords_str}"
                    self.entries.append((p.pattern_id, full_content, display))
            else:
                # Fallback: try generic SQL query
                self.entries = []
                logger.warning(f"Memory modules not available for {self.table_name}")

            # Update UI
            self.result_queue.put({'action': 'update_list'})

            if not self.entries:
                self.result_queue.put({'action': 'show_no_entries'})

        except Exception as e:
            logger.error(f"Error loading entries: {e}", exc_info=True)
            self.result_queue.put({'action': 'show_error', 'message': f"Failed to load entries: {str(e)}"})

    def _update_entry_list(self):
        """Update the entry list display."""
        # Clear existing buttons
        for btn in self.entry_buttons:
            btn.destroy()
        self.entry_buttons.clear()

        # Create new entry buttons
        for i, entry in enumerate(self.entries):
            if len(entry) >= 3:
                entry_id, content, display = entry[:3]

                btn = ctk.CTkButton(
                    self.list_scrollable,
                    text=f"#{entry_id}: {display[:80]}",
                    command=lambda eid=entry_id, c=content: self._on_entry_click(eid, c),
                    anchor="w",
                    fg_color="transparent",
                    border_width=1,
                    height=32
                )
                btn.grid(row=i, column=0, sticky="ew", pady=2, padx=5)
                self.entry_buttons.append(btn)

        # Update count
        self.count_label.configure(text=f"{len(self.entries)} entries")

    def _on_entry_click(self, entry_id, content):
        """Handle entry button click."""
        self.result_queue.put({'action': 'display_details', 'entry_id': entry_id, 'content': content})

    def _display_details(self, entry_id, content):
        """Display entry details in textbox."""
        self.details_textbox.delete("1.0", "end")

        # Pretty print JSON if possible
        try:
            parsed = json.loads(content)
            pretty_content = json.dumps(parsed, indent=2)
            self.details_textbox.insert("1.0", pretty_content)
        except:
            self.details_textbox.insert("1.0", content)

        # Store selected entry ID
        self.selected_entry_id = entry_id

    def _show_no_entries_message(self):
        """Show message when no entries are found."""
        self.details_textbox.delete("1.0", "end")
        self.details_textbox.insert("1.0", "No entries found. Add data via workflows or manually.")

    def search(self):
        """Search entries by keyword."""
        query = self.search_entry.get().strip()
        if not query:
            self.load_entries()
            return

        self.thread_manager.start_thread(self._search_thread, args=(query,))

    def _search_thread(self, query):
        """Background thread to search entries."""
        try:
            if self.table_name == 'knowledge' and knowledge_store:
                ks = knowledge_store.KnowledgeStore(self.db_name)
                query_obj = knowledge_store.KnowledgeQuery(content_keywords=[query], limit=100)
                entries = ks.retrieve_knowledge(query_obj)

                self.entries = []
                for e in entries:
                    content_str = json.dumps(e.content) if isinstance(e.content, dict) else str(e.content)
                    display = f"{e.domain}: {content_str[:100]}..."
                    self.entries.append((e.knowledge_id, content_str, display))

            elif self.table_name == 'tasks' and task_memory:
                tm = task_memory.TaskMemory(self.db_name)
                query_obj = task_memory.TaskMemoryQuery(keywords=[query], limit=100)
                patterns = tm.get_patterns(query_obj)

                self.entries = []
                for p in patterns:
                    keywords_str = ", ".join(p.keywords) if p.keywords else "No keywords"
                    full_content = json.dumps({
                        "keywords": p.keywords,
                        "task_type": p.task_type,
                        "success_rate": p.success_rate,
                        "typical_duration": p.typical_duration
                    })
                    display = f"{p.task_type}: {keywords_str}"
                    self.entries.append((p.pattern_id, full_content, display))
            else:
                self.entries = []

            # Update UI
            self.result_queue.put({'action': 'update_list'})

        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            self.result_queue.put({'action': 'show_error', 'message': f"Failed to search: {str(e)}"})

    def clear_search(self):
        """Clear search and reload all entries."""
        self.search_entry.delete(0, "end")
        self.load_entries()

    def delete_entry(self):
        """Delete the selected entry after confirmation."""
        if not hasattr(self, 'selected_entry_id'):
            messagebox.showwarning("No Selection", "Please select an entry to delete.")
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete entry #{self.selected_entry_id}?\n\n"
            "This will remove it from the Felix memory system.",
            icon='warning'
        )

        if not result:
            return

        self.thread_manager.start_thread(self._delete_thread, args=(self.selected_entry_id,))

    def _delete_thread(self, entry_id):
        """Background thread to delete entry."""
        try:
            # Use SQL for deletion since memory APIs don't provide delete methods
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Validate table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                          (self.table_name,))
            if not cursor.fetchone():
                self.result_queue.put({
                    'action': 'show_error',
                    'message': f"Table '{self.table_name}' not found in database"
                })
                conn.close()
                return

            # Determine correct column name and delete
            if self.table_name == 'knowledge':
                cursor.execute("DELETE FROM knowledge_entries WHERE knowledge_id = ?", (entry_id,))
                success_msg = "Knowledge entry deleted."
            elif self.table_name == 'tasks':
                # Check if table is 'tasks' or 'task_patterns'
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_patterns'")
                if cursor.fetchone():
                    cursor.execute("DELETE FROM task_patterns WHERE pattern_id = ?", (entry_id,))
                else:
                    cursor.execute("DELETE FROM tasks WHERE task_id = ?", (entry_id,))
                success_msg = "Task entry deleted."
            else:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': f"Unknown table type: {self.table_name}"
                })
                conn.close()
                return

            # Check if anything was deleted
            if cursor.rowcount == 0:
                self.result_queue.put({
                    'action': 'show_warning',
                    'title': 'Warning',
                    'message': "No entry found with that ID. It may have already been deleted."
                })
            else:
                self.result_queue.put({
                    'action': 'show_info',
                    'title': 'Success',
                    'message': success_msg
                })

            conn.commit()
            conn.close()

            # Reload entries
            self.result_queue.put({'action': 'reload'})

        except sqlite3.Error as e:
            logger.error(f"SQLite error deleting entry: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to delete entry: {str(e)}"
            })
        except Exception as e:
            logger.error(f"Error deleting: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to delete: {str(e)}"
            })


class WorkflowHistorySubTab(ctk.CTkFrame):
    """
    Sub-tab for displaying workflow execution history.
    """

    def __init__(self, master, thread_manager, **kwargs):
        """
        Initialize Workflow History sub-tab.

        Args:
            master: Parent widget
            thread_manager: ThreadManager instance
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.theme_manager = get_theme_manager()
        self.workflow_history = None  # Lazy initialization
        self.selected_workflow_id = None
        self.main_app = None  # Will be set by MemoryTab

        # Queue for thread-safe communication
        self.result_queue = queue.Queue()

        # Lazy loading flag - don't load until tab is visible
        self._data_loaded = False

        self._setup_ui()
        self._start_polling()
        # Removed immediate load_workflows() - now loads on first visibility

    def _setup_ui(self):
        """Set up the UI components."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # Tree expands
        self.grid_rowconfigure(3, weight=1)  # Details expands

        # Search and filter section
        control_frame = ctk.CTkFrame(self, fg_color="transparent")
        control_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        control_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(control_frame, text="Search:").grid(row=0, column=0, padx=(0, SPACE_XS), sticky="w")

        self.search_entry = ctk.CTkEntry(control_frame, placeholder_text="Search workflows...")
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=SPACE_XS)
        self.search_entry.bind('<Return>', lambda e: self.search_workflows())

        self.search_button = ctk.CTkButton(
            control_frame,
            text="Search",
            command=self.search_workflows,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.search_button.grid(row=0, column=2, padx=SPACE_XS)

        ctk.CTkLabel(control_frame, text="Status:").grid(row=0, column=3, padx=(SPACE_SM, SPACE_XS), sticky="w")

        self.status_var = ctk.StringVar(value="all")
        self.status_dropdown = ctk.CTkOptionMenu(
            control_frame,
            variable=self.status_var,
            values=["all", "completed", "failed"],
            command=lambda _: self.load_workflows(),
            width=BUTTON_SM[0]
        )
        self.status_dropdown.grid(row=0, column=4, padx=SPACE_XS)

        self.refresh_button = ctk.CTkButton(
            control_frame,
            text="Refresh",
            command=self.load_workflows,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.refresh_button.grid(row=0, column=5, padx=SPACE_XS)

        self.clear_button = ctk.CTkButton(
            control_frame,
            text="Clear",
            command=self.clear_search,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color="transparent",
            border_width=1
        )
        self.clear_button.grid(row=0, column=6, padx=(0, SPACE_XS))

        # Count label
        self.count_label = ctk.CTkLabel(control_frame, text="0 workflows")
        self.count_label.grid(row=0, column=7, padx=SPACE_XS)

        # Workflow tree section
        tree_frame = ctk.CTkFrame(self)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        # Use ThemedTreeview for workflow list
        columns = ["id", "task", "status", "confidence", "date"]
        headings = ["ID", "Task", "Status", "Conf.", "Date"]
        widths = [50, 400, 100, 80, 150]

        self.tree = ThemedTreeview(
            tree_frame,
            columns=columns,
            headings=headings,
            widths=widths,
            height=12
        )
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Bind selection event
        self.tree.bind_tree('<<TreeviewSelect>>', self.on_select)

        # Details section
        details_label = ctk.CTkLabel(self, text="Details:", anchor="w")
        details_label.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        details_frame = ctk.CTkFrame(self)
        details_frame.grid(row=3, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        details_frame.grid_columnconfigure(0, weight=1)
        details_frame.grid_rowconfigure(0, weight=1)

        self.details_textbox = ctk.CTkTextbox(
            details_frame,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.details_textbox.grid(row=0, column=0, sticky="nsew", padx=SPACE_XS, pady=SPACE_XS)

        # Action buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=4, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        self.delete_button = ctk.CTkButton(
            button_frame,
            text="Delete",
            command=self.delete_workflow,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226"
        )
        self.delete_button.pack(side="left", padx=SPACE_XS)

        self.view_full_button = ctk.CTkButton(
            button_frame,
            text="View Full Details",
            command=self.view_full_details,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        )
        self.view_full_button.pack(side="left", padx=SPACE_XS)

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                # Check if main Memory tab is visible
                if self.main_app.tabview.get() != "Memory":
                    return False
                # Check if this specific sub-tab (Workflow History) is visible
                if hasattr(self.master, 'master') and hasattr(self.master.master, 'get'):
                    current_subtab = self.master.master.get()
                    if current_subtab != "Workflow History":
                        return False
        except Exception:
            pass
        return True

    def _start_polling(self):
        """Start polling the result queue."""
        self._poll_results()

    def _poll_results(self):
        """Poll the result queue and update GUI (runs on main thread)."""
        # Check if shutdown signaled
        if not self.thread_manager.is_active:
            return  # Stop polling

        try:
            is_visible = self._is_visible()

            # Lazy load on first visibility
            if is_visible and not self._data_loaded:
                self._data_loaded = True
                self.load_workflows()

            # Only process results if this tab is visible
            if is_visible:
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    action = result.get('action')

                    if action == 'update_tree':
                        self._update_tree(result['workflows'])
                    elif action == 'display_details':
                        self._display_details(result['workflow'])
                    elif action == 'show_full_details':
                        self._show_full_details_popup(result['workflow'])
                    elif action == 'show_error':
                        messagebox.showerror("Error", result['message'])
                    elif action == 'show_warning':
                        messagebox.showwarning(result['title'], result['message'])
                    elif action == 'show_info':
                        messagebox.showinfo(result['title'], result['message'])
                    elif action == 'reload':
                        self.load_workflows()
                    elif action == 'clear_selection':
                        self.selected_workflow_id = None

        except Exception as e:
            logger.error(f"Error in poll_results: {e}", exc_info=True)

        # Schedule next poll - slower when not visible
        if self.thread_manager.is_active:
            poll_interval = 100 if self._is_visible() else 500
            self.after(poll_interval, self._poll_results)

    def _get_workflow_history(self):
        """Lazy initialization of WorkflowHistory."""
        if self.workflow_history is None and WorkflowHistory:
            self.workflow_history = WorkflowHistory()
        return self.workflow_history

    def load_workflows(self):
        """Load workflows from database with current filters."""
        status_filter = self.status_var.get()
        if status_filter == "all":
            status_filter = None
        self.thread_manager.start_thread(self._load_workflows_thread, args=(status_filter,))

    def refresh_workflows(self):
        """Manually refresh workflows list."""
        self.load_workflows()

    def _load_workflows_thread(self, status_filter):
        """Background thread to load workflows."""
        try:
            wh = self._get_workflow_history()
            if not wh:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "WorkflowHistory not available"
                })
                return

            # Retrieve workflows
            workflows = wh.get_workflow_outputs(status_filter=status_filter, limit=200)

            # Update UI
            self.result_queue.put({'action': 'update_tree', 'workflows': workflows})

        except Exception as e:
            logger.error(f"Error loading workflows: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load workflows: {str(e)}"
            })

    def _update_tree(self, workflows):
        """Update the treeview with workflow data."""
        # Clear existing items
        self.tree.clear()

        # Add workflows to tree
        for workflow in workflows:
            # Truncate task for display
            task_preview = workflow.task_input[:80] + "..." if len(workflow.task_input) > 80 else workflow.task_input

            # Format date
            try:
                dt = datetime.fromisoformat(workflow.created_at)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = workflow.created_at[:16] if workflow.created_at else "N/A"

            # Format confidence
            conf_str = f"{workflow.confidence:.2f}" if workflow.confidence > 0 else "N/A"

            # Status with indicator
            status_display = {
                "completed": "✓ Completed",
                "failed": "✗ Failed"
            }.get(workflow.status, workflow.status)

            self.tree.insert(
                "",
                "end",
                iid=str(workflow.workflow_id),
                values=(
                    workflow.workflow_id,
                    task_preview,
                    status_display,
                    conf_str,
                    date_str
                )
            )

        # Update count
        count = len(workflows)
        self.count_label.configure(text=f"{count} workflows")

        # Show message if no workflows
        if count == 0:
            self.details_textbox.delete("1.0", "end")
            self.details_textbox.insert(
                "1.0",
                "No workflows found. Run a workflow from the Workflows tab to see history here."
            )

    def search_workflows(self):
        """Search workflows by keyword."""
        keyword = self.search_entry.get().strip()
        if not keyword:
            self.load_workflows()
            return

        self.thread_manager.start_thread(self._search_workflows_thread, args=(keyword,))

    def _search_workflows_thread(self, keyword):
        """Background thread to search workflows."""
        try:
            wh = self._get_workflow_history()
            if not wh:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "WorkflowHistory not available"
                })
                return

            workflows = wh.search_workflows(keyword, limit=200)

            # Update UI
            self.result_queue.put({'action': 'update_tree', 'workflows': workflows})

        except Exception as e:
            logger.error(f"Error searching workflows: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to search workflows: {str(e)}"
            })

    def clear_search(self):
        """Clear search box and reload all workflows."""
        self.search_entry.delete(0, "end")
        self.status_var.set("all")
        self.load_workflows()

    def on_select(self, event):
        """Handle workflow selection in tree."""
        selection = self.tree.selection()
        if not selection:
            return

        workflow_id = int(selection[0])
        self.selected_workflow_id = workflow_id

        # Load workflow details
        self.thread_manager.start_thread(self._load_details_thread, args=(workflow_id,))

    def _load_details_thread(self, workflow_id):
        """Background thread to load workflow details."""
        try:
            wh = self._get_workflow_history()
            if not wh:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "WorkflowHistory not available"
                })
                return

            workflow = wh.get_workflow_by_id(workflow_id)

            if workflow:
                self.result_queue.put({'action': 'display_details', 'workflow': workflow})
            else:
                self.result_queue.put({
                    'action': 'show_warning',
                    'title': 'Not Found',
                    'message': f"Workflow {workflow_id} not found"
                })

        except Exception as e:
            logger.error(f"Error loading workflow details: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load details: {str(e)}"
            })

    def _display_details(self, workflow):
        """Display workflow details in the details panel."""
        self.details_textbox.delete("1.0", "end")

        # Format and display details
        details = f"""WORKFLOW #{workflow.workflow_id}
{'=' * 60}

Task Input:
{workflow.task_input}

{'=' * 60}
Status: {workflow.status.upper()}
Confidence: {workflow.confidence:.2f}
Agents: {workflow.agents_count}
Tokens: {workflow.tokens_used} / {workflow.max_tokens}
Temperature: {workflow.temperature:.2f}
Processing Time: {workflow.processing_time:.2f}s
Date: {workflow.created_at}
{'=' * 60}

Final Synthesis:
{workflow.final_synthesis if workflow.final_synthesis else 'No synthesis available'}

{'=' * 60}
"""

        self.details_textbox.insert("1.0", details)

    def view_full_details(self):
        """Open a popup window with full workflow details."""
        if not self.selected_workflow_id:
            messagebox.showwarning("No Selection", "Please select a workflow to view details")
            return

        self.thread_manager.start_thread(self._view_full_details_thread, args=(self.selected_workflow_id,))

    def _view_full_details_thread(self, workflow_id):
        """Background thread to load full workflow details."""
        try:
            wh = self._get_workflow_history()
            if not wh:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "WorkflowHistory not available"
                })
                return

            workflow = wh.get_workflow_by_id(workflow_id)

            if workflow:
                self.result_queue.put({'action': 'show_full_details', 'workflow': workflow})
            else:
                self.result_queue.put({
                    'action': 'show_warning',
                    'title': 'Not Found',
                    'message': f"Workflow {workflow_id} not found"
                })

        except Exception as e:
            logger.error(f"Error loading full details: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load details: {str(e)}"
            })

    def _show_full_details_popup(self, workflow):
        """Show a popup window with comprehensive workflow details."""
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Workflow #{workflow.workflow_id} - Full Details")
        popup.geometry("800x600")

        # Configure grid
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)

        # Create textbox with comprehensive details
        text_widget = ctk.CTkTextbox(
            popup,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        text_widget.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)

        # Format comprehensive details
        details = f"""WORKFLOW #{workflow.workflow_id} - COMPREHENSIVE DETAILS
{'=' * 80}

TASK INPUT:
{workflow.task_input}

{'=' * 80}
STATUS & METRICS:
{'=' * 80}
Status: {workflow.status.upper()}
Confidence: {workflow.confidence:.2f}
Agents Spawned: {workflow.agents_count}
Tokens Used: {workflow.tokens_used} / {workflow.max_tokens}
Temperature: {workflow.temperature:.2f}
Processing Time: {workflow.processing_time:.2f} seconds
Created: {workflow.created_at}
Completed: {workflow.completed_at if workflow.completed_at else 'N/A'}

{'=' * 80}
FINAL SYNTHESIS OUTPUT:
{'=' * 80}
{workflow.final_synthesis if workflow.final_synthesis else 'No synthesis available'}

{'=' * 80}
METADATA (JSON):
{'=' * 80}
{json.dumps(workflow.metadata, indent=2) if workflow.metadata else 'No metadata'}

{'=' * 80}
"""

        text_widget.insert("1.0", details)

        # Close button
        close_button = ctk.CTkButton(
            popup,
            text="Close",
            command=popup.destroy,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        close_button.grid(row=1, column=0, pady=SPACE_SM)

    def delete_workflow(self):
        """Delete the selected workflow after confirmation."""
        if not self.selected_workflow_id:
            messagebox.showwarning("No Selection", "Please select a workflow to delete")
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete workflow #{self.selected_workflow_id}?\n\n"
            "This action cannot be undone.",
            icon='warning'
        )

        if not result:
            return

        self.thread_manager.start_thread(self._delete_workflow_thread, args=(self.selected_workflow_id,))

    def _delete_workflow_thread(self, workflow_id):
        """Background thread to delete workflow."""
        try:
            wh = self._get_workflow_history()
            if not wh:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "WorkflowHistory not available"
                })
                return

            success = wh.delete_workflow(workflow_id)

            if success:
                self.result_queue.put({
                    'action': 'show_info',
                    'title': 'Success',
                    'message': f"Workflow #{workflow_id} deleted"
                })
                self.result_queue.put({'action': 'reload'})
                self.result_queue.put({'action': 'clear_selection'})
            else:
                self.result_queue.put({
                    'action': 'show_warning',
                    'title': 'Not Found',
                    'message': f"Workflow {workflow_id} not found"
                })

        except Exception as e:
            logger.error(f"Error deleting workflow: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to delete workflow: {str(e)}"
            })
