"""
Prompts Tab for Felix GUI (CustomTkinter Edition)

The Prompts tab provides:
- Browse prompt templates organized by category
- View and edit prompt templates
- Save custom versions with version history
- Preview rendered prompts with test variables
- Reset prompts to YAML defaults
- Reload YAML configuration
"""

import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
from typing import Optional, Dict, List, Tuple
import queue
import logging

from ..utils import ThreadManager, logger
from ..theme_manager import get_theme_manager
from ..styles import (
    BUTTON_SM, BUTTON_MD,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    SIDEBAR_WIDTH, TEXTBOX_SM
)

# Import PromptManager
try:
    from src.prompts.prompt_manager import PromptManager
except ImportError as e:
    logger.error(f"Failed to import PromptManager: {e}")
    PromptManager = None


class PromptsTab(ctk.CTkFrame):
    """
    Main Prompts tab for managing agent prompts.
    """

    def __init__(self, master, thread_manager: ThreadManager, main_app=None, **kwargs):
        """
        Initialize Prompts tab.

        Args:
            master: Parent widget
            thread_manager: ThreadManager instance
            main_app: Reference to main application (optional)
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()
        self.prompt_manager = None
        self.current_prompt_key: Optional[str] = None
        self.is_modified = False
        self.selected_category = None
        self._layout_manager = None

        # Queue for thread-safe communication
        self.result_queue = queue.Queue()

        # Lazy loading flag - don't load until tab is visible
        self._data_loaded = False

        # Initialize PromptManager
        if PromptManager:
            try:
                self.prompt_manager = PromptManager()
            except Exception as e:
                logger.error(f"Failed to initialize PromptManager: {e}")

        self._setup_ui()
        self._start_polling()

    def set_layout_manager(self, layout_manager):
        """Set the layout manager (interface compliance)."""
        self._layout_manager = layout_manager
        # Removed immediate _load_prompt_tree() - now loads on first visibility

    def _setup_ui(self):
        """Set up the UI components."""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)  # Editor column expands
        self.grid_rowconfigure(0, weight=1)

        # Left panel: Prompt categories and list
        left_panel = ctk.CTkFrame(self, width=SIDEBAR_WIDTH)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(SPACE_SM, SPACE_XS), pady=SPACE_SM)
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_propagate(False)

        # Header
        header_label = ctk.CTkLabel(
            left_panel,
            text="Prompt Categories",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        header_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Scrollable frame for prompt list
        self.prompt_list_frame = ctk.CTkScrollableFrame(left_panel)
        self.prompt_list_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_XS, pady=SPACE_XS)
        self.prompt_list_frame.grid_columnconfigure(0, weight=1)

        # Placeholder for prompt buttons
        self.prompt_buttons = []

        # Right panel: Editor
        right_panel = ctk.CTkFrame(self)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(SPACE_XS, SPACE_SM), pady=SPACE_SM)
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(2, weight=1)  # Editor expands

        # Editor header
        header_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header_frame.grid_columnconfigure(0, weight=1)

        self.prompt_label = ctk.CTkLabel(
            header_frame,
            text="Select a prompt",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold"),
            anchor="w"
        )
        self.prompt_label.grid(row=0, column=0, sticky="w")

        self.source_label = ctk.CTkLabel(
            header_frame,
            text="",
            text_color=self.theme_manager.get_color("fg_muted"),
            anchor="w"
        )
        self.source_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.status_label = ctk.CTkLabel(
            header_frame,
            text="",
            text_color=self.theme_manager.get_color("accent"),
            anchor="e"
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        # Description
        self.desc_label = ctk.CTkLabel(
            right_panel,
            text="",
            text_color=self.theme_manager.get_color("fg_secondary"),
            anchor="w",
            wraplength=600
        )
        self.desc_label.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        # Text editor
        editor_frame = ctk.CTkFrame(right_panel)
        editor_frame.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(0, weight=1)

        self.text_editor = ctk.CTkTextbox(
            editor_frame,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.text_editor.grid(row=0, column=0, sticky="nsew")
        self.text_editor.bind("<<Modified>>", self._on_text_modified)

        # Variables help section
        vars_label = ctk.CTkLabel(
            right_panel,
            text="Available Variables:",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
            anchor="w"
        )
        vars_label.grid(row=3, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, 2))

        self.vars_textbox = ctk.CTkTextbox(
            right_panel,
            height=TEXTBOX_SM,
            font=ctk.CTkFont(family="Courier", size=FONT_SMALL)
        )
        self.vars_textbox.grid(row=4, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        # Button bar
        button_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        button_frame.grid(row=5, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        self.save_btn = ctk.CTkButton(
            button_frame,
            text="Save Custom",
            command=self._save_custom,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        )
        self.save_btn.pack(side="left", padx=SPACE_XS)

        self.reset_btn = ctk.CTkButton(
            button_frame,
            text="Reset to Default",
            command=self._reset_to_default,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            fg_color="transparent",
            border_width=1
        )
        self.reset_btn.pack(side="left", padx=SPACE_XS)

        self.preview_btn = ctk.CTkButton(
            button_frame,
            text="Preview",
            command=self._preview_prompt,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.preview_btn.pack(side="left", padx=SPACE_XS)

        self.history_btn = ctk.CTkButton(
            button_frame,
            text="View History",
            command=self._view_history,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1]
        )
        self.history_btn.pack(side="left", padx=SPACE_XS)

        self.reload_btn = ctk.CTkButton(
            button_frame,
            text="Reload YAML",
            command=self._reload_yaml,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.reload_btn.pack(side="right", padx=SPACE_XS)

        # Initially disable buttons
        self._set_buttons_enabled(False)

    def _start_polling(self):
        """Start polling the result queue."""
        self._poll_results()

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                return self.main_app.tabview.get() == "Prompts"
        except Exception:
            pass
        return False

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
                self._load_prompt_tree()

            # Skip expensive updates if tab is not visible
            if not is_visible:
                # Schedule slower poll when not visible
                if self.thread_manager.is_active and self.winfo_exists():
                    self.after(500, self._poll_results)
                return

            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                action = result.get('action')

                if action == 'update_tree':
                    self._update_prompt_tree(result['categories'])
                elif action == 'load_prompt':
                    self._display_prompt(result['prompt_key'], result['template'])
                elif action == 'show_error':
                    messagebox.showerror("Error", result['message'])
                elif action == 'show_warning':
                    messagebox.showwarning(result['title'], result['message'])
                elif action == 'show_info':
                    messagebox.showinfo(result['title'], result['message'])
                elif action == 'reload_tree':
                    self._load_prompt_tree()
                elif action == 'show_preview':
                    self._show_preview_popup(result['rendered'], result['variables'])
                elif action == 'show_history':
                    self._show_history_popup(result['history'])
                elif action == 'show_save_dialog':
                    self._show_save_dialog(result['template'])

        except Exception as e:
            logger.error(f"Error in poll_results: {e}", exc_info=True)

        # Schedule next poll
        if self.thread_manager.is_active and self.winfo_exists():
            self.after(100, self._poll_results)

    def _load_prompt_tree(self):
        """Load prompts into tree view."""
        if not self.prompt_manager:
            logger.warning("PromptManager not available")
            return

        self.thread_manager.start_thread(self._load_prompt_tree_thread)

    def _load_prompt_tree_thread(self):
        """Background thread to load prompt categories."""
        try:
            # Get all prompts
            all_prompts = self.prompt_manager.list_all_prompts()

            # Organize by agent type
            categories: Dict[str, List[Tuple[str, str, str]]] = {}

            for prompt_key, description, source in all_prompts:
                # Parse prompt_key like "research_exploration_normal"
                parts = prompt_key.split('_')
                if len(parts) < 2:
                    continue

                agent_type = parts[0]  # "research", "analysis", etc.

                if agent_type not in categories:
                    categories[agent_type] = []

                # Add indicator for custom prompts
                indicator = "✎" if source == "custom" else "✓"
                display_name = f"{indicator} {prompt_key}"

                categories[agent_type].append((prompt_key, display_name, description))

            # Update UI
            self.result_queue.put({'action': 'update_tree', 'categories': categories})

        except Exception as e:
            logger.error(f"Error loading prompt tree: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load prompts: {str(e)}"
            })

    def _update_prompt_tree(self, categories: Dict[str, List[Tuple[str, str, str]]]):
        """Update the prompt list display."""
        # Clear existing buttons
        for btn in self.prompt_buttons:
            btn.destroy()
        self.prompt_buttons.clear()

        # Type names mapping
        type_names = {
            "research": "Research Prompts",
            "analysis": "Analysis Prompts",
            "critic": "Critic Prompts",
            "llm_agent": "Generic Agent Prompts",
            "metadata": "Metadata Templates"
        }

        row = 0
        for agent_type in sorted(categories.keys()):
            prompts = categories[agent_type]

            # Category header
            category_label = ctk.CTkLabel(
                self.prompt_list_frame,
                text=type_names.get(agent_type, agent_type.title()),
                font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
                anchor="w"
            )
            category_label.grid(row=row, column=0, sticky="ew", padx=SPACE_XS, pady=(SPACE_SM, SPACE_XS))
            self.prompt_buttons.append(category_label)
            row += 1

            # Prompt buttons
            for prompt_key, display_name, description in prompts:
                btn = ctk.CTkButton(
                    self.prompt_list_frame,
                    text=display_name,
                    command=lambda pk=prompt_key: self._on_prompt_click(pk),
                    anchor="w",
                    fg_color="transparent",
                    border_width=1,
                    height=BUTTON_MD[1]
                )
                btn.grid(row=row, column=0, sticky="ew", padx=SPACE_XS, pady=2)
                self.prompt_buttons.append(btn)
                row += 1

    def _on_prompt_click(self, prompt_key: str):
        """Handle prompt button click."""
        if self.is_modified:
            result = messagebox.askyesnocancel(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?"
            )
            if result is True:  # Yes - save
                self._save_custom()
            elif result is None:  # Cancel - don't switch
                return

        self.thread_manager.start_thread(self._load_prompt_thread, args=(prompt_key,))

    def _load_prompt_thread(self, prompt_key: str):
        """Background thread to load a prompt."""
        try:
            if not self.prompt_manager:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "PromptManager not available"
                })
                return

            # Get prompt from manager
            prompt_template = self.prompt_manager.get_prompt(prompt_key)

            if not prompt_template:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': f"Could not load prompt: {prompt_key}"
                })
                return

            # Update UI
            self.result_queue.put({
                'action': 'load_prompt',
                'prompt_key': prompt_key,
                'template': prompt_template
            })

        except Exception as e:
            logger.error(f"Error loading prompt: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load prompt: {str(e)}"
            })

    def _display_prompt(self, prompt_key: str, prompt_template):
        """Display a prompt in the editor."""
        self.current_prompt_key = prompt_key

        # Update header
        self.prompt_label.configure(text=f"Prompt: {prompt_key}")

        # Update source label
        source_text = f"Source: {prompt_template.source.upper()}"
        source_color = self.theme_manager.get_color("success") if prompt_template.source == "yaml" else self.theme_manager.get_color("warning")
        self.source_label.configure(text=source_text, text_color=source_color)

        # Update description
        self.desc_label.configure(text=prompt_template.description)

        # Update status
        if prompt_template.source == "database":
            self.status_label.configure(
                text=f"Custom (v{prompt_template.version})",
                text_color=self.theme_manager.get_color("warning")
            )
        else:
            self.status_label.configure(
                text="Default",
                text_color=self.theme_manager.get_color("success")
            )

        # Load template into editor
        self.text_editor.delete("1.0", "end")
        self.text_editor.insert("1.0", prompt_template.template)

        # Clear modified flag (CTkTextbox doesn't have edit_modified)
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

        vars_text = ", ".join(variables) if variables else "No variables for this template"

        self.vars_textbox.delete("1.0", "end")
        self.vars_textbox.insert("1.0", vars_text)

    def _on_text_modified(self, event=None):
        """Handle text modification."""
        # CTkTextbox doesn't have edit_modified, track manually
        if self.current_prompt_key:
            self.is_modified = True
            self.status_label.configure(
                text="Modified (unsaved)",
                text_color=self.theme_manager.get_color("error")
            )

    def _save_custom(self):
        """Save current prompt as custom version."""
        if not self.current_prompt_key:
            return

        if not self.is_modified:
            messagebox.showinfo("No Changes", "No changes to save")
            return

        # Get edited text
        custom_template = self.text_editor.get("1.0", "end").rstrip()

        # Show save dialog
        self.result_queue.put({
            'action': 'show_save_dialog',
            'template': custom_template
        })

    def _show_save_dialog(self, template: str):
        """Show dialog to save custom prompt."""
        # Create dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Save Custom Prompt")
        dialog.geometry("500x250")
        dialog.transient(self)
        dialog.after(100, lambda: self._safe_grab(dialog))

        # Configure grid
        dialog.grid_columnconfigure(0, weight=1)

        # Label
        label = ctk.CTkLabel(
            dialog,
            text="Optional notes about this version:",
            font=ctk.CTkFont(size=FONT_BODY),
            anchor="w"
        )
        label.grid(row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_LG, SPACE_XS))

        # Notes entry
        notes_textbox = ctk.CTkTextbox(dialog, height=TEXTBOX_SM)
        notes_textbox.grid(row=1, column=0, sticky="ew", padx=SPACE_LG, pady=SPACE_XS)

        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(row=2, column=0, pady=SPACE_LG)

        def save():
            notes = notes_textbox.get("1.0", "end").strip()
            dialog.destroy()
            self.thread_manager.start_thread(
                self._save_custom_thread,
                args=(template, notes)
            )

        def cancel():
            dialog.destroy()

        save_btn = ctk.CTkButton(button_frame, text="Save", command=save, width=BUTTON_SM[0], height=BUTTON_SM[1])
        save_btn.pack(side="left", padx=SPACE_XS)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=cancel,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color="transparent",
            border_width=1
        )
        cancel_btn.pack(side="left", padx=SPACE_XS)

    def _save_custom_thread(self, template: str, notes: str):
        """Background thread to save custom prompt."""
        try:
            if not self.prompt_manager:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "PromptManager not available"
                })
                return

            version = self.prompt_manager.save_custom_prompt(
                prompt_key=self.current_prompt_key,
                template=template,
                notes=notes
            )

            self.result_queue.put({
                'action': 'show_info',
                'title': 'Success',
                'message': f"Custom prompt saved as version {version}"
            })

            # Reload prompt
            self.result_queue.put({'action': 'reload_tree'})
            self.thread_manager.start_thread(
                self._load_prompt_thread,
                args=(self.current_prompt_key,)
            )

        except Exception as e:
            logger.error(f"Error saving prompt: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to save prompt: {str(e)}"
            })

    def _reset_to_default(self):
        """Reset prompt to YAML default."""
        if not self.current_prompt_key:
            return

        # Confirm
        result = messagebox.askyesno(
            "Reset to Default",
            f"Reset '{self.current_prompt_key}' to YAML default?\n\n"
            "This will deactivate all custom versions."
        )

        if not result:
            return

        self.thread_manager.start_thread(self._reset_to_default_thread)

    def _reset_to_default_thread(self):
        """Background thread to reset prompt."""
        try:
            if not self.prompt_manager:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "PromptManager not available"
                })
                return

            self.prompt_manager.reset_to_default(self.current_prompt_key)

            self.result_queue.put({
                'action': 'show_info',
                'title': 'Success',
                'message': "Prompt reset to default"
            })

            # Reload
            self.result_queue.put({'action': 'reload_tree'})
            self.thread_manager.start_thread(
                self._load_prompt_thread,
                args=(self.current_prompt_key,)
            )

        except Exception as e:
            logger.error(f"Error resetting prompt: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to reset prompt: {str(e)}"
            })

    def _preview_prompt(self):
        """Preview rendered prompt with test data."""
        if not self.current_prompt_key:
            return

        template = self.text_editor.get("1.0", "end").rstrip()

        self.thread_manager.start_thread(
            self._preview_prompt_thread,
            args=(template,)
        )

    def _preview_prompt_thread(self, template: str):
        """Background thread to preview prompt."""
        try:
            if not self.prompt_manager:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "PromptManager not available"
                })
                return

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
            rendered = self.prompt_manager.render_template(template, **test_vars)

            # Update UI
            self.result_queue.put({
                'action': 'show_preview',
                'rendered': rendered,
                'variables': test_vars
            })

        except Exception as e:
            logger.error(f"Error previewing prompt: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to render template: {str(e)}"
            })

    def _show_preview_popup(self, rendered: str, variables: Dict):
        """Show preview in popup window."""
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"Preview: {self.current_prompt_key}")
        popup.geometry("800x600")
        popup.transient(self)

        # Configure grid
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)

        # Test variables display
        vars_frame = ctk.CTkFrame(popup)
        vars_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        vars_frame.grid_columnconfigure(0, weight=1)

        vars_label = ctk.CTkLabel(
            vars_frame,
            text="Test Variables:",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
            anchor="w"
        )
        vars_label.pack(anchor="w", padx=SPACE_SM, pady=(SPACE_XS, 2))

        vars_text = "\n".join(f"{k} = {v}" for k, v in variables.items())
        vars_textbox = ctk.CTkTextbox(vars_frame, height=TEXTBOX_SM, font=ctk.CTkFont(family="Courier", size=FONT_SMALL))
        vars_textbox.pack(fill="x", padx=SPACE_SM, pady=(0, SPACE_SM))
        vars_textbox.insert("1.0", vars_text)

        # Rendered output
        output_frame = ctk.CTkFrame(popup)
        output_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_XS)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        output_label = ctk.CTkLabel(
            output_frame,
            text="Rendered Prompt:",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
            anchor="w"
        )
        output_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_XS, 2))

        output_textbox = ctk.CTkTextbox(
            output_frame,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        output_textbox.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        output_textbox.insert("1.0", rendered)

        # Close button
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy, width=BUTTON_SM[0], height=BUTTON_SM[1])
        close_btn.grid(row=2, column=0, pady=(SPACE_XS, SPACE_SM))

    def _view_history(self):
        """View prompt version history."""
        if not self.current_prompt_key:
            return

        self.thread_manager.start_thread(self._view_history_thread)

    def _view_history_thread(self):
        """Background thread to load prompt history."""
        try:
            if not self.prompt_manager:
                self.result_queue.put({
                    'action': 'show_error',
                    'message': "PromptManager not available"
                })
                return

            history = self.prompt_manager.get_prompt_history(self.current_prompt_key)

            if not history:
                self.result_queue.put({
                    'action': 'show_info',
                    'title': 'No History',
                    'message': "No version history for this prompt"
                })
                return

            # Update UI
            self.result_queue.put({
                'action': 'show_history',
                'history': history
            })

        except Exception as e:
            logger.error(f"Error loading history: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to load history: {str(e)}"
            })

    def _show_history_popup(self, history: List[Dict]):
        """Show history in popup window."""
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title(f"History: {self.current_prompt_key}")
        popup.geometry("900x500")
        popup.transient(self)

        # Configure grid
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(0, weight=1)

        # Scrollable frame for history entries
        scroll_frame = ctk.CTkScrollableFrame(popup)
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        scroll_frame.grid_columnconfigure(0, weight=1)

        # Display history entries
        for i, entry in enumerate(history):
            entry_frame = ctk.CTkFrame(scroll_frame, fg_color=self.theme_manager.get_color("bg_secondary"))
            entry_frame.grid(row=i, column=0, sticky="ew", pady=SPACE_XS)
            entry_frame.grid_columnconfigure(1, weight=1)

            # Version and action
            version_label = ctk.CTkLabel(
                entry_frame,
                text=f"v{entry['version']}",
                font=ctk.CTkFont(size=FONT_BODY, weight="bold"),
                width=50
            )
            version_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=SPACE_XS)

            action_label = ctk.CTkLabel(
                entry_frame,
                text=entry['action'].replace('_', ' ').title(),
                font=ctk.CTkFont(size=FONT_BODY),
                anchor="w"
            )
            action_label.grid(row=0, column=1, sticky="w", padx=SPACE_XS, pady=SPACE_XS)

            # Timestamp
            timestamp_label = ctk.CTkLabel(
                entry_frame,
                text=entry['timestamp'],
                font=ctk.CTkFont(size=FONT_CAPTION),
                text_color=self.theme_manager.get_color("fg_muted"),
                anchor="e"
            )
            timestamp_label.grid(row=0, column=2, sticky="e", padx=SPACE_SM, pady=SPACE_XS)

            # Notes
            if entry.get('notes'):
                notes_label = ctk.CTkLabel(
                    entry_frame,
                    text=entry['notes'],
                    font=ctk.CTkFont(size=FONT_CAPTION),
                    text_color=self.theme_manager.get_color("fg_secondary"),
                    anchor="w",
                    wraplength=700
                )
                notes_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=SPACE_SM, pady=(0, SPACE_XS))

        # Close button
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy, width=BUTTON_SM[0], height=BUTTON_SM[1])
        close_btn.grid(row=1, column=0, pady=(SPACE_XS, SPACE_SM))

    def _reload_yaml(self):
        """Reload YAML file."""
        if not self.prompt_manager:
            messagebox.showerror("Error", "PromptManager not available")
            return

        self.thread_manager.start_thread(self._reload_yaml_thread)

    def _reload_yaml_thread(self):
        """Background thread to reload YAML."""
        try:
            self.prompt_manager.reload_yaml()

            self.result_queue.put({
                'action': 'show_info',
                'title': 'Success',
                'message': "YAML configuration reloaded"
            })

            self.result_queue.put({'action': 'reload_tree'})

        except Exception as e:
            logger.error(f"Error reloading YAML: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to reload YAML: {str(e)}"
            })

    def _set_buttons_enabled(self, enabled: bool):
        """Enable or disable editor buttons."""
        state = "normal" if enabled else "disabled"
        self.save_btn.configure(state=state)
        self.reset_btn.configure(state=state)
        self.preview_btn.configure(state=state)
        self.history_btn.configure(state=state)

    def _enable_features(self):
        """Enable features when Felix system is running."""
        # Prompts tab doesn't depend on Felix system running
        pass

    def _disable_features(self):
        """Disable features when Felix system is stopped."""
        # Prompts tab doesn't depend on Felix system running
        pass

    def _safe_grab(self, dialog):
        """Safely grab focus after window is rendered."""
        try:
            dialog.grab_set()
            dialog.focus_set()
        except Exception:
            pass  # Silently ignore if grab fails
