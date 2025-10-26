"""
Workflow History GUI Component

This module provides a user-friendly interface for viewing, searching,
and managing workflow execution history.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import json
from typing import Optional
from .utils import ThreadManager, logger


class WorkflowHistoryFrame(ttk.Frame):
    """
    Frame for displaying and managing workflow history.

    Uses Treeview for a professional table display with sorting,
    searching, and filtering capabilities.
    """

    def __init__(self, parent, thread_manager, theme_manager=None):
        """
        Initialize the workflow history frame.

        Args:
            parent: Parent widget
            thread_manager: ThreadManager for async operations
            theme_manager: Optional theme manager for styling
        """
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.theme_manager = theme_manager
        self.workflow_history = None  # Will be initialized when first accessed
        self.selected_workflow_id = None

        # Create UI components
        self._create_widgets()

        # Apply initial theme
        self.apply_theme()

        # Load initial data
        self.load_workflows()

    def _create_widgets(self):
        """Create all widgets for the workflow history interface."""
        # Top control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Search controls
        ttk.Label(control_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        self.search_entry = ttk.Entry(control_frame, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.search_entry.bind('<Return>', lambda e: self.search_workflows())

        search_button = ttk.Button(control_frame, text="Search", command=self.search_workflows)
        search_button.pack(side=tk.LEFT, padx=(0, 10))

        # Status filter
        ttk.Label(control_frame, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        self.status_var = tk.StringVar(value="all")
        status_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.status_var,
            values=["all", "completed", "failed"],
            state="readonly",
            width=12
        )
        status_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        status_dropdown.bind('<<ComboboxSelected>>', lambda e: self.load_workflows())

        # Refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh", command=self.load_workflows)
        refresh_button.pack(side=tk.LEFT, padx=(0, 5))

        # Clear button
        clear_button = ttk.Button(control_frame, text="Clear Search", command=self.clear_search)
        clear_button.pack(side=tk.LEFT)

        # Result count label
        self.count_label = ttk.Label(control_frame, text="")
        self.count_label.pack(side=tk.RIGHT, padx=5)

        # Treeview for workflow list
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Define columns
        columns = ("id", "task", "status", "confidence", "date")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)

        # Configure columns
        self.tree.heading("id", text="ID")
        self.tree.heading("task", text="Task")
        self.tree.heading("status", text="Status")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("date", text="Date")

        self.tree.column("id", width=50, anchor="center")
        self.tree.column("task", width=400, anchor="w")
        self.tree.column("status", width=100, anchor="center")
        self.tree.column("confidence", width=100, anchor="center")
        self.tree.column("date", width=150, anchor="center")

        # Add scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout for tree and scrollbars
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_select)

        # Details panel
        details_label = ttk.Label(self, text="Details:")
        details_label.pack(anchor=tk.W, padx=5, pady=(5, 0))

        details_frame = ttk.Frame(self)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.details_text = tk.Text(details_frame, wrap=tk.WORD, height=10)
        details_scrollbar = ttk.Scrollbar(details_frame, command=self.details_text.yview)
        self.details_text.config(yscrollcommand=details_scrollbar.set)

        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Button panel
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.delete_button = ttk.Button(button_frame, text="Delete", command=self.delete_workflow)
        self.delete_button.pack(side=tk.LEFT, padx=(0, 5))

        self.view_full_button = ttk.Button(button_frame, text="View Full Details", command=self.view_full_details)
        self.view_full_button.pack(side=tk.LEFT)

    def _get_workflow_history(self):
        """Lazy initialization of WorkflowHistory."""
        if self.workflow_history is None:
            from src.memory.workflow_history import WorkflowHistory
            self.workflow_history = WorkflowHistory()
        return self.workflow_history

    def load_workflows(self):
        """Load workflows from database with current filters."""
        self.thread_manager.start_thread(self._load_workflows_thread)

    def _load_workflows_thread(self):
        """Background thread to load workflows."""
        try:
            wh = self._get_workflow_history()

            # Get status filter
            status_filter = self.status_var.get()
            if status_filter == "all":
                status_filter = None

            # Retrieve workflows
            workflows = wh.get_workflow_outputs(status_filter=status_filter, limit=200)

            # Update UI on main thread
            self.after(0, lambda: self._update_tree(workflows))

        except Exception as e:
            logger.error(f"Error loading workflows: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load workflows: {error_msg}"))

    def _update_tree(self, workflows):
        """Update the treeview with workflow data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add workflows to tree
        for workflow in workflows:
            # Truncate task for display
            task_preview = workflow.task_input[:80] + "..." if len(workflow.task_input) > 80 else workflow.task_input

            # Format date
            try:
                dt = datetime.fromisoformat(workflow.created_at)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = workflow.created_at[:16]

            # Format confidence
            conf_str = f"{workflow.confidence:.2f}" if workflow.confidence > 0 else "N/A"

            # Status with emoji/indicator
            status_display = {
                "completed": "✓ Completed",
                "failed": "✗ Failed"
            }.get(workflow.status, workflow.status)

            self.tree.insert("", tk.END, iid=str(workflow.workflow_id), values=(
                workflow.workflow_id,
                task_preview,
                status_display,
                conf_str,
                date_str
            ))

        # Update count
        count = len(workflows)
        self.count_label.config(text=f"{count} workflow(s)")

        # Show message if no workflows
        if count == 0:
            self.details_text.config(state='normal')
            self.details_text.delete(1.0, tk.END)
            self.details_text.insert(tk.END, "No workflows found. Run a workflow from the Workflows tab to see history here.")
            self.details_text.config(state='disabled')

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
            workflows = wh.search_workflows(keyword, limit=200)

            self.after(0, lambda: self._update_tree(workflows))

        except Exception as e:
            logger.error(f"Error searching workflows: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to search workflows: {error_msg}"))

    def clear_search(self):
        """Clear search box and reload all workflows."""
        self.search_entry.delete(0, tk.END)
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
            workflow = wh.get_workflow_by_id(workflow_id)

            if workflow:
                self.after(0, lambda: self._display_details(workflow))
            else:
                self.after(0, lambda: messagebox.showwarning("Not Found", f"Workflow {workflow_id} not found"))

        except Exception as e:
            logger.error(f"Error loading workflow details: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load details: {error_msg}"))

    def _display_details(self, workflow):
        """Display workflow details in the details panel."""
        self.details_text.config(state='normal')
        self.details_text.delete(1.0, tk.END)

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

        self.details_text.insert(tk.END, details)
        self.details_text.config(state='disabled')

    def view_full_details(self):
        """Open a popup window with full workflow details including metadata."""
        if not self.selected_workflow_id:
            messagebox.showwarning("No Selection", "Please select a workflow to view details")
            return

        self.thread_manager.start_thread(self._view_full_details_thread, args=(self.selected_workflow_id,))

    def _view_full_details_thread(self, workflow_id):
        """Background thread to load full workflow details."""
        try:
            wh = self._get_workflow_history()
            workflow = wh.get_workflow_by_id(workflow_id)

            if workflow:
                self.after(0, lambda: self._show_full_details_popup(workflow))
            else:
                self.after(0, lambda: messagebox.showwarning("Not Found", f"Workflow {workflow_id} not found"))

        except Exception as e:
            logger.error(f"Error loading full details: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load details: {error_msg}"))

    def _show_full_details_popup(self, workflow):
        """Show a popup window with comprehensive workflow details."""
        popup = tk.Toplevel(self)
        popup.title(f"Workflow #{workflow.workflow_id} - Full Details")
        popup.geometry("800x600")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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
Completed: {workflow.completed_at}

{'=' * 80}
FINAL SYNTHESIS OUTPUT:
{'=' * 80}
{workflow.final_synthesis if workflow.final_synthesis else 'No synthesis available'}

{'=' * 80}
METADATA (JSON):
{'=' * 80}
{json.dumps(workflow.metadata, indent=2)}

{'=' * 80}
"""

        text_widget.insert(tk.END, details)
        text_widget.config(state='disabled')

        # Close button
        close_button = ttk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=5)

        # Apply theme to popup
        if self.theme_manager:
            self.theme_manager.apply_to_text_widget(text_widget)

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
            success = wh.delete_workflow(workflow_id)

            if success:
                self.after(0, lambda: messagebox.showinfo("Success", f"Workflow #{workflow_id} deleted"))
                self.after(0, self.load_workflows)
                self.selected_workflow_id = None
            else:
                self.after(0, lambda: messagebox.showwarning("Not Found", f"Workflow {workflow_id} not found"))

        except Exception as e:
            logger.error(f"Error deleting workflow: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to delete workflow: {error_msg}"))

    def refresh_workflows(self):
        """Public method to refresh workflows list (called from workflows tab after save)."""
        self.load_workflows()

    def apply_theme(self):
        """Apply current theme to workflow history widgets."""
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()

            # Apply theme to details text widget
            self.theme_manager.apply_to_text_widget(self.details_text)

            # Apply theme to treeview
            style = ttk.Style()
            style.configure("Treeview",
                          background=theme["text_bg"],
                          foreground=theme["text_fg"],
                          fieldbackground=theme["text_bg"])
            style.map('Treeview',
                     background=[('selected', theme["text_select_bg"])],
                     foreground=[('selected', theme["text_select_fg"])])
