"""
Prompts Tab for Felix GUI.

Allows viewing and editing agent prompts:
- View default prompts from YAML
- Edit and save custom prompts to database
- Reset prompts to defaults
- View prompt history and versions
- Preview rendered prompts with test data
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, Dict, Any
import logging

from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class PromptsTab(ttk.Frame):
    """GUI tab for viewing and editing agent prompts."""

    def __init__(self, parent, prompt_manager: Optional[PromptManager] = None, theme_manager=None):
        """
        Initialize Prompts tab.

        Args:
            parent: Parent widget
            prompt_manager: PromptManager instance (created if None)
            theme_manager: Optional theme manager for styling
        """
        super().__init__(parent)

        self.prompt_manager = prompt_manager or PromptManager()
        self.theme_manager = theme_manager
        self.current_prompt_key: Optional[str] = None
        self.is_modified = False

        self._setup_ui()
        self._load_prompt_tree()

        # Apply initial theme
        self.apply_theme()

    def _setup_ui(self):
        """Setup the UI layout."""
        # Create main layout with paned window
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Prompt tree
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        # Tree header
        tree_header = ttk.Frame(left_frame)
        tree_header.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(tree_header, text="Prompt Categories", font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT)

        # Prompt tree
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame, selectmode="browse", show="tree")
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right panel: Editor
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        # Editor header
        header_frame = ttk.Frame(right_frame)
        header_frame.pack(fill=tk.X, pady=(0, 5))

        self.prompt_label = ttk.Label(header_frame, text="Select a prompt", font=("TkDefaultFont", 10, "bold"))
        self.prompt_label.pack(side=tk.LEFT)

        self.source_label = ttk.Label(header_frame, text="", foreground="gray")
        self.source_label.pack(side=tk.LEFT, padx=(10, 0))

        self.status_label = ttk.Label(header_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.RIGHT)

        # Description
        self.desc_label = ttk.Label(right_frame, text="", foreground="gray", wraplength=600)
        self.desc_label.pack(fill=tk.X, pady=(0, 5))

        # Text editor
        editor_frame = ttk.Frame(right_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.text_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Courier", 10)
        )
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        self.text_editor.bind("<<Modified>>", self._on_text_modified)

        # Variables help
        vars_frame = ttk.LabelFrame(right_frame, text="Available Variables", padding=5)
        vars_frame.pack(fill=tk.X, pady=(0, 5))

        self.vars_label = tk.Text(vars_frame, height=2, wrap=tk.WORD, font=("Courier", 9), relief=tk.FLAT, background="white")
        self.vars_label.pack(fill=tk.X)
        self.vars_label.configure(state=tk.DISABLED)

        # Button bar
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X)

        self.save_btn = ttk.Button(button_frame, text="Save Custom", command=self._save_custom)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_btn = ttk.Button(button_frame, text="Reset to Default", command=self._reset_to_default)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.preview_btn = ttk.Button(button_frame, text="Preview", command=self._preview_prompt)
        self.preview_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.history_btn = ttk.Button(button_frame, text="View History", command=self._view_history)
        self.history_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.reload_btn = ttk.Button(button_frame, text="Reload YAML", command=self._reload_yaml)
        self.reload_btn.pack(side=tk.LEFT)

        # Initially disable buttons
        self._set_buttons_enabled(False)

    def _load_prompt_tree(self):
        """Load prompts into tree view."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Get all prompts
        all_prompts = self.prompt_manager.list_all_prompts()

        # Organize by agent type
        categories: Dict[str, Dict[str, tuple]] = {}

        for prompt_key, description, source in all_prompts:
            # Parse prompt_key like "research_exploration_normal"
            parts = prompt_key.split('_')
            if len(parts) < 2:
                continue

            agent_type = parts[0]  # "research", "analysis", etc.
            sub_category = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]  # "exploration", "mid_phase", etc.
            mode = parts[-1] if parts[-1] in ['normal', 'strict', 'header', 'context'] else sub_category

            if agent_type not in categories:
                categories[agent_type] = {}

            if sub_category not in categories[agent_type]:
                categories[agent_type][sub_category] = []

            # Add indicator for custom prompts
            indicator = "✎" if source == "custom" else "✓"
            display_name = f"{indicator} {mode}"

            categories[agent_type][sub_category].append((prompt_key, display_name, description))

        # Add to tree
        type_names = {
            "research": "Research Prompts",
            "analysis": "Analysis Prompts",
            "critic": "Critic Prompts",
            "llm_agent": "Generic Agent Prompts",
            "metadata": "Metadata Templates"
        }

        for agent_type, subcategories in sorted(categories.items()):
            type_node = self.tree.insert("", tk.END, text=type_names.get(agent_type, agent_type.title()), open=False)

            for sub_cat, prompts in sorted(subcategories.items()):
                # Format subcategory name nicely
                sub_cat_name = sub_cat.replace('_', ' ').title()
                sub_node = self.tree.insert(type_node, tk.END, text=sub_cat_name, open=False)

                for prompt_key, display_name, description in prompts:
                    # Store prompt_key in item values
                    self.tree.insert(sub_node, tk.END, text=display_name, values=(prompt_key,))

    def _on_tree_select(self, event):
        """Handle tree selection."""
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.tree.item(item, "values")

        if not values:
            # Category selected, not a prompt
            return

        prompt_key = values[0]
        self._load_prompt(prompt_key)

    def _load_prompt(self, prompt_key: str):
        """Load and display a prompt."""
        self.current_prompt_key = prompt_key

        # Get prompt from manager
        prompt_template = self.prompt_manager.get_prompt(prompt_key)

        if not prompt_template:
            messagebox.showerror("Error", f"Could not load prompt: {prompt_key}")
            return

        # Update header
        self.prompt_label.config(text=f"Prompt: {prompt_key}")
        self.source_label.config(
            text=f"Source: {prompt_template.source.upper()}",
            foreground="green" if prompt_template.source == "yaml" else "orange"
        )
        self.desc_label.config(text=prompt_template.description)

        # Update status
        if prompt_template.source == "database":
            self.status_label.config(text=f"Custom (v{prompt_template.version})", foreground="orange")
        else:
            self.status_label.config(text="Default", foreground="green")

        # Load template into editor
        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert("1.0", prompt_template.template)
        self.text_editor.edit_modified(False)
        self.is_modified = False

        # Update variables help
        self._update_variables_help(prompt_key)

        # Enable buttons
        self._set_buttons_enabled(True)

    def _update_variables_help(self, prompt_key: str):
        """Update the variables help text based on prompt type."""
        # Determine which variables are relevant
        variables = []

        if "research" in prompt_key:
            variables = ["{research_domain}", "{depth_ratio}"]
        elif "analysis" in prompt_key:
            variables = ["{analysis_type}", "{depth_ratio}"]
        elif "critic" in prompt_key:
            variables = ["{review_focus}", "{depth_ratio}"]

        if "context" in prompt_key or "footer" in prompt_key:
            variables.append("{context}")

        vars_text = "Variables: " + ", ".join(variables) if variables else "No variables for this template"

        self.vars_label.configure(state=tk.NORMAL)
        self.vars_label.delete("1.0", tk.END)
        self.vars_label.insert("1.0", vars_text)
        self.vars_label.configure(state=tk.DISABLED)

    def _on_text_modified(self, event=None):
        """Handle text modification."""
        if self.text_editor.edit_modified():
            self.is_modified = True
            self.status_label.config(text="Modified (unsaved)", foreground="red")

    def _save_custom(self):
        """Save current prompt as custom version."""
        if not self.current_prompt_key:
            return

        if not self.is_modified:
            messagebox.showinfo("No Changes", "No changes to save")
            return

        # Get edited text
        custom_template = self.text_editor.get("1.0", tk.END).rstrip()

        # Prompt for notes
        notes_dialog = tk.Toplevel(self)
        notes_dialog.title("Save Custom Prompt")
        notes_dialog.geometry("400x150")

        ttk.Label(notes_dialog, text="Optional notes about this version:").pack(pady=10)

        notes_entry = tk.Text(notes_dialog, height=3, width=50)
        notes_entry.pack(padx=10, pady=5)

        def save():
            notes = notes_entry.get("1.0", tk.END).strip()
            version = self.prompt_manager.save_custom_prompt(
                prompt_key=self.current_prompt_key,
                template=custom_template,
                notes=notes
            )
            messagebox.showinfo("Saved", f"Custom prompt saved as version {version}")
            notes_dialog.destroy()

            # Reload prompt to show new status
            self._load_prompt(self.current_prompt_key)
            self._load_prompt_tree()  # Refresh tree to show custom indicator

        def cancel():
            notes_dialog.destroy()

        btn_frame = ttk.Frame(notes_dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

    def _reset_to_default(self):
        """Reset prompt to YAML default."""
        if not self.current_prompt_key:
            return

        # Confirm
        if not messagebox.askyesno("Reset to Default",
                                   f"Reset '{self.current_prompt_key}' to YAML default?\n\n"
                                   "This will deactivate all custom versions."):
            return

        self.prompt_manager.reset_to_default(self.current_prompt_key)
        messagebox.showinfo("Reset", "Prompt reset to default")

        # Reload
        self._load_prompt(self.current_prompt_key)
        self._load_prompt_tree()

    def _preview_prompt(self):
        """Preview rendered prompt with test data."""
        if not self.current_prompt_key:
            return

        # Get current template
        template = self.text_editor.get("1.0", tk.END).rstrip()

        # Create test variables based on prompt type
        test_vars = {}
        if "research" in self.current_prompt_key:
            test_vars = {"research_domain": "technical", "depth_ratio": 0.25}
        elif "analysis" in self.current_prompt_key:
            test_vars = {"analysis_type": "technical", "depth_ratio": 0.55}
        elif "critic" in self.current_prompt_key:
            test_vars = {"review_focus": "accuracy", "depth_ratio": 0.75}

        if "context" in self.current_prompt_key or "footer" in self.current_prompt_key:
            test_vars["context"] = "Sample task context about building a multi-agent system"

        # Render
        try:
            rendered = self.prompt_manager.render_template(template, **test_vars)
        except Exception as e:
            messagebox.showerror("Render Error", f"Failed to render template:\n{e}")
            return

        # Show in dialog
        preview_dialog = tk.Toplevel(self)
        preview_dialog.title(f"Preview: {self.current_prompt_key}")
        preview_dialog.geometry("700x500")

        # Test vars display
        vars_frame = ttk.LabelFrame(preview_dialog, text="Test Variables", padding=5)
        vars_frame.pack(fill=tk.X, padx=10, pady=5)

        vars_text = "\n".join(f"{k} = {v}" for k, v in test_vars.items())
        vars_label = ttk.Label(vars_frame, text=vars_text, font=("Courier", 9))
        vars_label.pack()

        # Rendered output
        output_frame = ttk.LabelFrame(preview_dialog, text="Rendered Prompt", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Courier", 10))
        output_text.pack(fill=tk.BOTH, expand=True)
        output_text.insert("1.0", rendered)
        output_text.configure(state=tk.DISABLED)

        ttk.Button(preview_dialog, text="Close", command=preview_dialog.destroy).pack(pady=10)

    def _view_history(self):
        """View prompt version history."""
        if not self.current_prompt_key:
            return

        history = self.prompt_manager.get_prompt_history(self.current_prompt_key)

        if not history:
            messagebox.showinfo("No History", "No version history for this prompt")
            return

        # Show in dialog
        history_dialog = tk.Toplevel(self)
        history_dialog.title(f"History: {self.current_prompt_key}")
        history_dialog.geometry("800x400")

        # Create treeview for history
        tree_frame = ttk.Frame(history_dialog)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        columns = ("version", "action", "timestamp", "notes")
        history_tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        history_tree.heading("version", text="Version")
        history_tree.heading("action", text="Action")
        history_tree.heading("timestamp", text="Timestamp")
        history_tree.heading("notes", text="Notes")

        history_tree.column("version", width=80)
        history_tree.column("action", width=120)
        history_tree.column("timestamp", width=200)
        history_tree.column("notes", width=300)

        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=history_tree.yview)
        history_tree.configure(yscrollcommand=scroll.set)

        history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate history
        for entry in history:
            history_tree.insert("", tk.END, values=(
                entry['version'],
                entry['action'],
                entry['timestamp'],
                entry['notes']
            ))

        ttk.Button(history_dialog, text="Close", command=history_dialog.destroy).pack(pady=10)

    def _reload_yaml(self):
        """Reload YAML file."""
        self.prompt_manager.reload_yaml()
        self._load_prompt_tree()
        messagebox.showinfo("Reloaded", "YAML configuration reloaded")

    def _set_buttons_enabled(self, enabled: bool):
        """Enable or disable editor buttons."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.save_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.preview_btn.config(state=state)
        self.history_btn.config(state=state)

    def apply_theme(self):
        """Apply current theme to all prompts tab widgets."""
        if not self.theme_manager:
            return

        theme = self.theme_manager.get_current_theme()

        # Apply to text editor (ScrolledText contains Text widget)
        try:
            # ScrolledText has an internal text widget
            if hasattr(self.text_editor, 'vbar'):
                text_widget = self.text_editor
                # ScrolledText uses internal frame, apply to the actual text widget
                for child in text_widget.winfo_children():
                    if isinstance(child, tk.Text):
                        self.theme_manager.apply_to_text_widget(child)
            else:
                self.theme_manager.apply_to_text_widget(self.text_editor)
        except Exception as e:
            logger.warning(f"Could not theme text_editor: {e}")

        # Apply to vars_label (tk.Text widget)
        try:
            self.vars_label.configure(
                bg=theme["text_bg"],
                fg=theme["text_fg"]
            )
        except Exception as e:
            logger.warning(f"Could not theme vars_label: {e}")

        # Apply to treeview
        try:
            style = ttk.Style()
            style.configure("Treeview",
                          background=theme["text_bg"],
                          foreground=theme["text_fg"],
                          fieldbackground=theme["text_bg"])
            style.map('Treeview',
                     background=[('selected', theme["text_select_bg"])],
                     foreground=[('selected', theme["text_select_fg"])])
        except Exception as e:
            logger.warning(f"Could not theme treeview: {e}")

        # Recursively apply theme to all children (frames, labels, etc.)
        try:
            self.theme_manager.apply_to_all_children(self)
        except Exception as e:
            logger.warning(f"Could not recursively apply theme: {e}")
