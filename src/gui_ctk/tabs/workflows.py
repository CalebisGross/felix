"""
Workflows Tab for Felix GUI (CustomTkinter Edition)

Provides workflow execution interface with:
- Task input (multi-line text)
- Max steps configuration
- Continue from previous workflow
- Real-time progress display
- Approval dialog integration
- Markdown export
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import logging
import textwrap
from datetime import datetime
from typing import Optional, Dict, Any

from ..utils import ThreadManager, logger
from ..theme_manager import get_theme_manager
from ..dialogs import ApprovalDialog
from ..responsive import Breakpoint, BreakpointConfig
from .base_tab import ResponsiveTab
from ..components.resizable_separator import ResizableSeparator
from ..styles import (
    BUTTON_SM, BUTTON_MD, BUTTON_LG,
    FONT_SECTION, FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    INPUT_MD, INPUT_XL, TEXTBOX_MD, TEXTBOX_XL
)

logger = logging.getLogger(__name__)


class WorkflowsTab(ResponsiveTab):
    """
    Workflows tab for running Felix workflows with real-time feedback.

    Responsive layout:
    - COMPACT: Single column (input, output stacked vertically)
    - STANDARD: 2 columns stacked or tabbed (input | output)
    - WIDE/ULTRAWIDE: Side-by-side with resizable separator (input | separator | output)

    Features:
    - Multi-line task input
    - Max steps configuration (Auto/5/10/15/20)
    - Continue from previous workflow
    - Real-time progress tracking
    - Approval dialog integration during workflow
    - Save results to markdown
    - Workflow rating/feedback
    """

    def __init__(self, master, thread_manager: ThreadManager, main_app=None, **kwargs):
        """
        Initialize workflows tab.

        Args:
            master: Parent widget
            thread_manager: Thread manager for background work
            main_app: Reference to main application for Felix system access
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, thread_manager, main_app, **kwargs)

        self.theme_manager = get_theme_manager()

        # State variables
        self.last_workflow_result: Optional[Dict[str, Any]] = None
        self.last_workflow_id: Optional[str] = None
        self.workflow_running = False
        self.approval_polling_active = False
        self.approval_poll_interval = 1000  # 1 second
        self.last_approval_check = set()
        self.dialog_open = False
        self.feedback_timer = None

        # Workflow ID mapping for continuation
        self.workflow_id_map: Dict[str, str] = {}

        # Layout containers
        self._main_container = None
        self._input_panel = None
        self._output_panel = None
        self._separator = None
        self._current_layout = None

        # Setup UI
        self._setup_ui()

        # Initialize feedback system
        self._init_feedback()

        logger.info("Workflows tab initialized")

    def _init_feedback(self):
        """Initialize feedback system."""
        try:
            from src.feedback import FeedbackManager, FeedbackIntegrator
            self.feedback_manager = FeedbackManager()
            self.feedback_integrator = FeedbackIntegrator(self.feedback_manager)
            logger.debug("Feedback system initialized")
        except Exception as e:
            logger.warning(f"Could not initialize feedback system: {e}")
            self.feedback_manager = None
            self.feedback_integrator = None

    def _setup_ui(self):
        """Setup the UI components."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main container for responsive layout
        self._main_container = ctk.CTkFrame(self, fg_color="transparent")
        self._main_container.grid(row=0, column=0, sticky="nsew")

        # Create input and output panels
        self._create_input_panel()
        self._create_output_panel()

        # Progress bar (stays at bottom)
        self.progress = ctk.CTkProgressBar(self, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_SM))
        self.progress.set(0)

        # Initially disable features
        self._disable_features()

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Handle responsive layout changes based on breakpoint."""
        if not self._main_container:
            return

        # Skip redundant updates
        if self._current_layout == breakpoint:
            return

        self._current_layout = breakpoint

        # Clear existing layout
        for widget in self._main_container.winfo_children():
            widget.grid_forget()

        # Remove separator if it exists
        if self._separator:
            self._separator.destroy()
            self._separator = None

        # Apply breakpoint-specific layout
        if breakpoint == Breakpoint.COMPACT:
            self._layout_compact()
        elif breakpoint == Breakpoint.STANDARD:
            self._layout_standard()
        else:  # WIDE or ULTRAWIDE
            self._layout_wide()

    def _layout_compact(self):
        """Single column: input and output stacked vertically."""
        self._main_container.grid_columnconfigure(0, weight=1)
        self._main_container.grid_rowconfigure(0, weight=0)  # Input
        self._main_container.grid_rowconfigure(1, weight=1)  # Output expands

        if self._input_panel:
            self._input_panel.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_SM, SPACE_MD))

        if self._output_panel:
            self._output_panel.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=(0, SPACE_SM))

    def _layout_standard(self):
        """2 columns: input on left, output on right (no separator)."""
        self._main_container.grid_columnconfigure(0, weight=1)  # Input
        self._main_container.grid_columnconfigure(1, weight=1)  # Output
        self._main_container.grid_rowconfigure(0, weight=1)

        if self._input_panel:
            self._input_panel.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_SM)

        if self._output_panel:
            self._output_panel.grid(row=0, column=1, sticky="nsew", padx=(SPACE_SM, SPACE_LG), pady=SPACE_SM)

    def _layout_wide(self):
        """Side-by-side with resizable separator: input | separator | output."""
        self._main_container.grid_columnconfigure(0, weight=1)  # Input
        self._main_container.grid_columnconfigure(1, weight=0)  # Separator
        self._main_container.grid_columnconfigure(2, weight=1)  # Output
        self._main_container.grid_rowconfigure(0, weight=1)

        if self._input_panel:
            self._input_panel.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, 0), pady=SPACE_SM)

        # Create separator
        self._separator = ResizableSeparator(
            self._main_container,
            orientation="vertical",
            on_drag_complete=self._on_separator_drag
        )
        self._separator.grid(row=0, column=1, sticky="ns", pady=SPACE_SM)

        if self._output_panel:
            self._output_panel.grid(row=0, column=2, sticky="nsew", padx=(0, SPACE_LG), pady=SPACE_SM)

    def _on_separator_drag(self, ratio: float):
        """Handle separator drag to resize panels."""
        # Adjust column weights based on drag ratio
        if self._main_container and self._current_layout in (Breakpoint.WIDE, Breakpoint.ULTRAWIDE):
            left_weight = max(1, int(ratio * 10))
            right_weight = max(1, int((1 - ratio) * 10))
            self._main_container.grid_columnconfigure(0, weight=left_weight)
            self._main_container.grid_columnconfigure(2, weight=right_weight)

    def _create_input_panel(self):
        """Create the input panel with task entry and controls."""
        self._input_panel = ctk.CTkFrame(self._main_container)
        self._input_panel.grid_columnconfigure(0, weight=1)
        self._input_panel.grid_rowconfigure(1, weight=1)  # Task entry expands

        # Task label
        task_label = ctk.CTkLabel(
            self._input_panel,
            text="Task:",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        task_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Task entry
        self.task_entry = ctk.CTkTextbox(self._input_panel, height=TEXTBOX_MD, wrap="word")
        self.task_entry.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_MD))

        # Options row (max steps and continue from)
        options_frame = ctk.CTkFrame(self._input_panel, fg_color="transparent")
        options_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_MD))

        # Max steps
        max_steps_container = ctk.CTkFrame(options_frame, fg_color="transparent")
        max_steps_container.pack(side="left", padx=(0, SPACE_LG))

        ctk.CTkLabel(
            max_steps_container,
            text="Max Steps:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).pack(side="left", padx=(0, SPACE_XS))

        self.max_steps_var = ctk.StringVar(value="Auto")
        self.max_steps_dropdown = ctk.CTkComboBox(
            max_steps_container,
            values=["Auto", "5", "10", "15", "20"],
            variable=self.max_steps_var,
            width=INPUT_MD,
            state="readonly"
        )
        self.max_steps_dropdown.pack(side="left", padx=(0, SPACE_XS))

        ctk.CTkLabel(
            max_steps_container,
            text="(Auto = adaptive)",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color="gray"
        ).pack(side="left")

        # Continue from workflow
        continue_container = ctk.CTkFrame(options_frame, fg_color="transparent")
        continue_container.pack(side="left")

        ctk.CTkLabel(
            continue_container,
            text="Continue from:",
            font=ctk.CTkFont(size=FONT_BODY)
        ).pack(side="left", padx=(0, SPACE_XS))

        self.parent_workflow_var = ctk.StringVar(value="New Workflow")
        self.parent_workflow_dropdown = ctk.CTkComboBox(
            continue_container,
            values=["New Workflow"],
            variable=self.parent_workflow_var,
            width=INPUT_XL,
            state="readonly",
            command=self._on_parent_workflow_selected
        )
        self.parent_workflow_dropdown.pack(side="left", padx=(0, SPACE_XS))

        self.refresh_workflows_button = ctk.CTkButton(
            continue_container,
            text="Refresh",
            command=self._refresh_workflow_list,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1]
        )
        self.refresh_workflows_button.pack(side="left")

        # Button row
        button_frame = ctk.CTkFrame(self._input_panel, fg_color="transparent")
        button_frame.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))

        self.run_button = ctk.CTkButton(
            button_frame,
            text="Run Workflow",
            command=self.run_workflow,
            width=BUTTON_LG[0],
            height=BUTTON_LG[1],
            fg_color="#2fa572",
            hover_color="#25835e",
            state="disabled"
        )
        self.run_button.pack(side="left", padx=(0, SPACE_XS))

        self.save_button = ctk.CTkButton(
            button_frame,
            text="Save Results",
            command=self.save_results,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            state="disabled"
        )
        self.save_button.pack(side="left", padx=(0, SPACE_XS))

        self.rate_button = ctk.CTkButton(
            button_frame,
            text="⭐ Rate Workflow",
            command=self._show_feedback_dialog_from_button,
            width=BUTTON_LG[0],
            height=BUTTON_LG[1],
            state="disabled"
        )
        self.rate_button.pack(side="left")

    def _create_output_panel(self):
        """Create the output panel with results display."""
        self._output_panel = ctk.CTkFrame(self._main_container)
        self._output_panel.grid_columnconfigure(0, weight=1)
        self._output_panel.grid_rowconfigure(1, weight=1)  # Output text expands

        # Output label
        output_label = ctk.CTkLabel(
            self._output_panel,
            text="Workflow Output:",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        )
        output_label.grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        # Output textbox
        self.output_text = ctk.CTkTextbox(self._output_panel, height=TEXTBOX_XL, wrap="word")
        self.output_text.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))

    def _enable_features(self):
        """Enable workflow features when system is running."""
        self.run_button.configure(state="normal")
        self.refresh_workflows_button.configure(state="normal")
        self._refresh_workflow_list()
        logger.debug("Workflow features enabled")

    def _disable_features(self):
        """Disable workflow features when system is not running."""
        self.run_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.rate_button.configure(state="disabled")
        self.refresh_workflows_button.configure(state="disabled")

        # Cancel any pending feedback timer
        if self.feedback_timer:
            self.after_cancel(self.feedback_timer)
            self.feedback_timer = None

        logger.debug("Workflow features disabled")

    def _refresh_workflow_list(self):
        """Refresh the list of recent workflows for continuation."""
        try:
            from src.memory.workflow_history import WorkflowHistory

            workflow_history = WorkflowHistory()
            recent_workflows = workflow_history.get_workflow_outputs(
                status_filter="completed",
                limit=20,
                offset=0
            )

            # Build dropdown options
            workflow_options = ["New Workflow"]
            self.workflow_id_map = {}

            for wf in recent_workflows:
                # Truncate task for display
                task_preview = wf.task_input[:60] + "..." if len(wf.task_input) > 60 else wf.task_input
                display_text = f"#{wf.workflow_id}: {task_preview} ({wf.confidence:.2f})"
                workflow_options.append(display_text)
                self.workflow_id_map[display_text] = wf.workflow_id

            self.parent_workflow_dropdown.configure(values=workflow_options)
            logger.debug(f"Refreshed workflow list: {len(recent_workflows)} workflows")

        except Exception as e:
            logger.error(f"Failed to refresh workflow list: {e}", exc_info=True)

    def _on_parent_workflow_selected(self, choice: str):
        """Handle parent workflow selection."""
        if choice == "New Workflow":
            return

        try:
            from src.memory.workflow_history import WorkflowHistory

            workflow_id = self.workflow_id_map.get(choice)
            if not workflow_id:
                return

            workflow_history = WorkflowHistory()
            parent_wf = workflow_history.get_workflow_by_id(workflow_id)

            if parent_wf:
                # Show parent info in output
                self._write_output("=" * 60)
                self._write_output(f"CONTINUING FROM WORKFLOW #{workflow_id}")
                self._write_output(f"Parent Task: {parent_wf.task_input}")
                self._write_output(f"Parent Confidence: {parent_wf.confidence:.2f}")
                self._write_output(f"Agents Used: {parent_wf.agents_count}")
                self._write_output("")
                self._write_output("Parent Synthesis:")
                synthesis_preview = parent_wf.final_synthesis[:300] + "..." if len(parent_wf.final_synthesis) > 300 else parent_wf.final_synthesis
                self._write_output(synthesis_preview)
                self._write_output("=" * 60)
                self._write_output("")
                self._write_output("Enter your follow-up question in the task field above.")

        except Exception as e:
            logger.error(f"Failed to load parent workflow: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load workflow: {str(e)}")

    def _write_output(self, message: str):
        """Write a message to the output text widget."""
        self.output_text.configure(state="normal")
        self.output_text.insert("end", message + '\n')
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def run_workflow(self):
        """Run a workflow with the given task input."""
        task_input = self.task_entry.get("1.0", "end").strip()
        if not task_input:
            messagebox.showwarning("Input Error", "Please enter a task description.")
            return

        # Get max steps override
        max_steps_value = self.max_steps_var.get()
        max_steps_override = None if max_steps_value == "Auto" else int(max_steps_value)

        # Get parent workflow ID if continuing
        parent_workflow_id = None
        selected = self.parent_workflow_var.get()
        if selected != "New Workflow":
            parent_workflow_id = self.workflow_id_map.get(selected)

        # Disable run button and start workflow
        self.run_button.configure(state="disabled")
        self.progress.set(0)
        self.progress.start()

        # Clear output
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

        # Start workflow in background thread
        self.thread_manager.start_thread(
            self._run_workflow_thread,
            args=(task_input, max_steps_override, parent_workflow_id)
        )

        logger.info(f"Started workflow: {task_input[:50]}...")

    def _run_workflow_thread(self, task_input: str, max_steps_override: Optional[int] = None, parent_workflow_id: Optional[str] = None):
        """Run workflow in background thread."""

        def progress_callback(status: str, progress_percentage: float):
            """Callback to update GUI progress from pipeline thread."""
            self.after(0, lambda: self._update_progress(status, progress_percentage))
            self.after(0, lambda: self._write_output(f"[{progress_percentage:.0f}%] {status}"))

        # Record start time
        start_time = datetime.now()

        # Mark workflow as running and start approval polling
        self.workflow_running = True
        logger.info("=== WORKFLOW STARTED: Scheduling approval polling ===")
        self._write_output("🔍 Approval polling: Starting (will check for approvals every 1s)")
        self.after(0, self._start_approval_polling)

        try:
            if parent_workflow_id:
                self.after(0, lambda: self._write_output(f"Continuing from workflow #{parent_workflow_id}"))
                self.after(0, lambda: self._write_output(f"Follow-up question: {task_input}"))
            else:
                self.after(0, lambda: self._write_output(f"Starting workflow for task: {task_input}"))

            if max_steps_override is not None:
                self.after(0, lambda: self._write_output(f"Max steps override: {max_steps_override}"))
            else:
                self.after(0, lambda: self._write_output(f"Max steps: Auto (adaptive based on complexity)"))
            self.after(0, lambda: self._write_output("=" * 60))

            # Use Felix system's integrated workflow runner if available
            if self.main_app and self.main_app.felix_system and self.main_app.system_running:
                # Signal user activity to pause background processing (e.g., Knowledge Brain batch processing)
                # This gives priority to user-initiated workflows over background LLM requests
                if hasattr(self.main_app.felix_system.lm_client, 'signal_user_activity'):
                    self.main_app.felix_system.lm_client.signal_user_activity(True)
                    logger.debug("User activity signaled - background processing paused")

                self.after(0, lambda: self._write_output("Running through Felix system..."))
                result = self.main_app.felix_system.run_workflow(
                    task_input,
                    progress_callback=progress_callback,
                    max_steps_override=max_steps_override,
                    parent_workflow_id=parent_workflow_id
                )
            else:
                # Felix system not running
                self.after(0, lambda: self._write_output("ERROR: Felix system not running"))
                self.after(0, lambda: self._write_output("Please start the Felix system from the Dashboard tab first."))
                result = {
                    "status": "failed",
                    "error": "Felix system not running. Start the system from Dashboard first."
                }

            # Add task_input and timestamps to result
            end_time = datetime.now()
            result["task_input"] = task_input
            result["start_time"] = start_time.isoformat()
            result["end_time"] = end_time.isoformat()
            result["processing_time"] = (end_time - start_time).total_seconds()

            # Display summary results
            self.after(0, lambda: self._write_output("\n" + "=" * 60))
            self.after(0, lambda: self._write_output("WORKFLOW SUMMARY"))
            self.after(0, lambda: self._write_output("=" * 60))

            if result.get("status") == "completed":
                self.after(0, lambda: self._write_output(f"Status: COMPLETED"))
                self.after(0, lambda: self._write_output(f"Agents spawned: {len(result.get('agents_spawned', []))}"))
                self.after(0, lambda: self._write_output(f"LLM responses: {len(result.get('llm_responses', []))}"))
                self.after(0, lambda: self._write_output(f"Completed agents: {result.get('completed_agents', 0)}"))

                # Display final synthesis
                final_synthesis = result.get("centralpost_synthesis")
                if final_synthesis:
                    self.after(0, lambda: self._write_output("\n" + "=" * 60))
                    self.after(0, lambda: self._write_output("FINAL SYNTHESIS"))
                    self.after(0, lambda: self._write_output("=" * 60))

                    confidence = final_synthesis.get("confidence", 0.0)
                    self.after(0, lambda c=confidence: self._write_output(f"Confidence: {c:.2f}"))

                    synthesis_time = final_synthesis.get("synthesis_time", 0)
                    self.after(0, lambda st=synthesis_time: self._write_output(f"Synthesis Time: {st:.2f}s"))

                    tokens_used = final_synthesis.get("tokens_used", 0)
                    max_tokens = final_synthesis.get("max_tokens", 0)
                    self.after(0, lambda tu=tokens_used, mt=max_tokens: self._write_output(f"Tokens: {tu} / {mt}"))

                    temperature = final_synthesis.get("temperature", 0)
                    self.after(0, lambda temp=temperature: self._write_output(f"Temperature: {temp:.2f}"))

                    agents_count = final_synthesis.get("agents_synthesized", 0)
                    self.after(0, lambda ac=agents_count: self._write_output(f"Agents Synthesized: {ac}"))

                    self.after(0, lambda: self._write_output(""))

                    # Wrap synthesis content
                    synthesis_content = final_synthesis.get("synthesis_content", "")
                    wrapped_lines = textwrap.wrap(synthesis_content, width=80, break_long_words=False, replace_whitespace=False)

                    for line in wrapped_lines:
                        self.after(0, lambda l=line: self._write_output(l))

                    self.after(0, lambda: self._write_output(""))
                    self.after(0, lambda: self._write_output("=" * 60))

                self.after(0, lambda: self._write_output("\nWorkflow completed successfully!"))

                # Store result and enable save button
                self.last_workflow_result = result
                self.after(0, lambda: self.save_button.configure(state="normal"))

                # Save workflow to database
                self._save_workflow_to_history(result, parent_workflow_id)

                # Enable rate button
                if self.last_workflow_id:
                    self.after(0, lambda: self.rate_button.configure(state="normal"))
                    # Auto-show feedback dialog after 10 seconds
                    self.feedback_timer = self.after(10000, lambda: self._show_feedback_dialog_auto())

            else:
                error_msg = result.get("error", "Unknown error")
                self.after(0, lambda: self._write_output(f"Status: FAILED"))
                self.after(0, lambda: self._write_output(f"Error: {error_msg}"))
                self.after(0, lambda: messagebox.showerror("Workflow Failed", f"Workflow failed: {error_msg}"))

        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._write_output(f"\nERROR: {error_msg}"))
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to run workflow: {error_msg}"))
            logger.error(f"Workflow error: {e}", exc_info=True)

        finally:
            # Resume background processing (e.g., Knowledge Brain batch processing)
            if self.main_app and self.main_app.felix_system and hasattr(self.main_app.felix_system.lm_client, 'signal_user_activity'):
                self.main_app.felix_system.lm_client.signal_user_activity(False)
                logger.debug("User activity ended - background processing resumed")

            # Stop approval polling and mark workflow as not running
            self.workflow_running = False
            self._stop_approval_polling()

            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.progress.set(0))
            self.after(0, lambda: self.run_button.configure(state="normal"))

    def _save_workflow_to_history(self, result: Dict[str, Any], parent_workflow_id: Optional[str] = None):
        """Save workflow result to history database."""
        try:
            from src.memory.workflow_history import WorkflowHistory

            workflow_history = WorkflowHistory()
            workflow_id = workflow_history.save_workflow_output(result, parent_workflow_id=parent_workflow_id)

            if workflow_id:
                self.last_workflow_id = workflow_id

                if parent_workflow_id:
                    self.after(0, lambda wid=workflow_id, pid=parent_workflow_id:
                              self._write_output(f"Workflow saved to history (ID: {wid}, continuing from #{pid})"))
                else:
                    self.after(0, lambda wid=workflow_id:
                              self._write_output(f"Workflow saved to history (ID: {wid})"))
                logger.info(f"Saved workflow to history (ID: {workflow_id})")
            else:
                self.after(0, lambda: self._write_output("Warning: Failed to save workflow to history"))
                logger.warning("Failed to save workflow to history database")

        except Exception as e:
            self.after(0, lambda: self._write_output(f"Warning: Could not save to history: {e}"))
            logger.error(f"Error saving workflow to history: {e}", exc_info=True)

    def _update_progress(self, status: str, progress_percentage: float):
        """Update progress bar and status display."""
        # Clamp progress to 0-100%
        progress = max(0.0, min(1.0, progress_percentage / 100.0))
        self.progress.set(progress)

    def _start_approval_polling(self):
        """Start polling for pending approvals during workflow execution."""
        try:
            logger.info("=== APPROVAL POLLING: START REQUESTED ===")
            logger.info(f"  Current state: active={self.approval_polling_active}, workflow_running={self.workflow_running}")

            if not self.approval_polling_active:
                self.approval_polling_active = True
                self.last_approval_check.clear()
                logger.info("✓ Approval polling activated, starting poll loop")
                self._write_output("✓ Approval polling activated")
                self._poll_for_approvals()
            else:
                logger.warning("⚠ Approval polling already active, skipping start")
                self._write_output("⚠ Approval polling already active")

        except Exception as e:
            logger.error(f"❌ Error starting approval polling: {e}", exc_info=True)
            self._write_output(f"❌ Error starting approval polling: {e}")

    def _stop_approval_polling(self):
        """Stop polling for pending approvals."""
        self.approval_polling_active = False
        logger.info("Approval polling stopped")

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                return self.main_app.tabview.get() == "Workflows"
        except Exception:
            pass
        return False

    def _poll_for_approvals(self):
        """Poll for pending approvals and show dialog if found."""
        logger.debug(f"POLL CHECK: active={self.approval_polling_active}, workflow_running={self.workflow_running}, dialog_open={self.dialog_open}")

        if not self.approval_polling_active or not self.workflow_running:
            logger.debug("POLL SKIPPED: Conditions not met")
            return

        # Skip expensive operations if tab is not visible
        if not self._is_visible():
            logger.debug("POLL SKIPPED: Tab not visible")
            # Schedule next poll
            if self.approval_polling_active and self.workflow_running:
                self.after(self.approval_poll_interval, self._poll_for_approvals)
            return

        try:
            # Only poll if no dialog is currently open
            if not self.dialog_open:
                # Check for pending approvals
                if self.main_app and self.main_app.felix_system:
                    central_post = self.main_app.felix_system.central_post
                    pending_approvals = central_post.get_pending_actions()

                    logger.debug(f"POLL RESULT: Found {len(pending_approvals)} pending approvals")

                    # Show only the FIRST pending approval
                    if pending_approvals:
                        first_approval = pending_approvals[0]
                        approval_id = first_approval['approval_id']

                        logger.info(f"📋 Approval detected: {approval_id}")
                        logger.info(f"   Command: {first_approval.get('command', 'N/A')[:50]}...")

                        # Only show if not already shown
                        if approval_id not in self.last_approval_check:
                            self.dialog_open = True
                            self.last_approval_check.add(approval_id)

                            logger.info(f"✓ Opening approval dialog for: {approval_id}")
                            cmd_preview = first_approval.get('command', 'N/A')[:60]
                            self._write_output(f"⏸️  Workflow paused - approval required for: {cmd_preview}")
                            self._write_output(f"   Opening approval dialog...")

                            # Show approval dialog on main thread
                            self.after(0, lambda a=first_approval: self._show_approval_dialog(a))
                        else:
                            logger.debug(f"POLL SKIPPED: Approval {approval_id} already shown")
            else:
                logger.debug("POLL SKIPPED: Dialog already open")

        except Exception as e:
            logger.error(f"Error polling for approvals: {e}", exc_info=True)
            self._write_output(f"❌ Error checking for approvals: {e}")

        # Schedule next poll if still active
        if self.approval_polling_active and self.workflow_running and not self.dialog_open:
            logger.debug(f"POLL SCHEDULED: Next poll in {self.approval_poll_interval}ms")
            self.after(self.approval_poll_interval, self._poll_for_approvals)
        else:
            logger.debug(f"POLL NOT SCHEDULED: active={self.approval_polling_active}, running={self.workflow_running}, dialog_open={self.dialog_open}")

    def _show_approval_dialog(self, approval_request: Dict[str, Any]):
        """Show approval dialog for pending approval request."""
        logger.info("=== SHOWING APPROVAL DIALOG ===")
        logger.info(f"  Approval ID: {approval_request.get('approval_id')}")
        logger.info(f"  Command: {approval_request.get('command')}")
        self._write_output("✓ Approval dialog opened - waiting for decision...")

        try:
            # Create and show approval dialog
            dialog = ApprovalDialog(
                self,
                approval_request,
                self._on_approval_decision
            )

            # Wait for dialog to close
            logger.info("✓ Dialog created, waiting for user decision...")
            dialog.wait_window()
            logger.info("✓ Dialog closed")
            self._write_output("✓ Approval dialog closed")

        except Exception as e:
            logger.error(f"Error showing approval dialog: {e}", exc_info=True)
            self._write_output(f"❌ Error showing approval dialog: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to show approval dialog:\n{str(e)}"
            )

        finally:
            # Always reset dialog_open flag
            self.dialog_open = False
            logger.info("Approval dialog cleanup complete, resuming polling")
            self._write_output("▶️  Workflow resuming, checking for more approvals...")

            # Resume polling immediately
            if self.approval_polling_active and self.workflow_running:
                self.after(100, self._poll_for_approvals)

    def _on_approval_decision(self, approval_id: str, decision):
        """Handle approval decision from dialog."""
        try:
            if not self.main_app or not self.main_app.felix_system:
                raise Exception("Felix system not available")

            central_post = self.main_app.felix_system.central_post

            # Process decision
            success = central_post.approve_system_action(
                approval_id=approval_id,
                decision=decision,
                decided_by="user"
            )

            if success:
                logger.info(f"Approval decision processed: {approval_id} -> {decision.value}")
                self._write_output(f"\n✓ Approval processed: {decision.value}")
            else:
                raise Exception("Failed to process approval decision")

        except Exception as e:
            logger.error(f"Error processing approval decision: {e}", exc_info=True)
            messagebox.showerror(
                "Error",
                f"Failed to process approval decision:\n{str(e)}"
            )

    def save_results(self):
        """Save workflow results to a formatted markdown file."""
        if not self.last_workflow_result:
            messagebox.showwarning("No Results", "No workflow results to save. Run a workflow first.")
            return

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"felix_synthesis_{timestamp}.md"

        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if not file_path:
            return  # User cancelled

        try:
            from src.utils.markdown_formatter import format_synthesis_markdown_detailed

            result = self.last_workflow_result

            # Get agent_manager if available
            agent_manager = None
            if self.main_app and self.main_app.felix_system:
                agent_manager = self.main_app.felix_system.agent_manager

            # Format results as markdown
            markdown_content = format_synthesis_markdown_detailed(
                result=result,
                agent_manager=agent_manager,
                include_prompts=True
            )

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")
            logger.info(f"Saved workflow results to: {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
            logger.error(f"Error saving results: {e}", exc_info=True)

    def _show_feedback_dialog_from_button(self):
        """Show feedback dialog when user clicks the Rate Workflow button."""
        # Cancel auto-show timer if it exists
        if self.feedback_timer:
            self.after_cancel(self.feedback_timer)
            self.feedback_timer = None
            logger.info("Cancelled auto-show timer (user clicked button)")

        # Show the dialog
        if self.last_workflow_id:
            self._show_quick_feedback_dialog(self.last_workflow_id)
        else:
            logger.warning("No workflow ID available for feedback")

    def _show_feedback_dialog_auto(self):
        """Auto-show feedback dialog after delay if user hasn't clicked the button."""
        # Check if button is still enabled (not yet rated)
        if self.rate_button.cget("state") == "normal":
            logger.info("Auto-showing feedback dialog (10 second timer elapsed)")
            if self.last_workflow_id:
                self._show_quick_feedback_dialog(self.last_workflow_id)
        else:
            logger.info("Skipping auto-show (already rated)")

    def _show_quick_feedback_dialog(self, workflow_id: str):
        """Show quick feedback dialog for workflow rating (thumbs up/down)."""
        if not self.feedback_integrator:
            logger.warning("Feedback system not available")
            return

        # Create dialog window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Rate this workflow")
        dialog.geometry("450x250")
        dialog.transient(self)

        # Center dialog
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Question label
        ctk.CTkLabel(
            dialog,
            text="How would you rate this workflow?",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).pack(pady=(30, SPACE_SM))

        ctk.CTkLabel(
            dialog,
            text="Your feedback helps Felix learn and improve!",
            font=ctk.CTkFont(size=FONT_CAPTION)
        ).pack(pady=(0, 30))

        # Button frame
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)

        def submit_rating(positive: bool):
            """Submit rating and close dialog."""
            try:
                self.feedback_integrator.submit_workflow_rating_with_propagation(
                    str(workflow_id),
                    positive,
                    knowledge_ids_used=[]
                )
                logger.info(f"Feedback submitted for workflow {workflow_id}: {'positive' if positive else 'negative'}")

                # Disable rate button after submission
                self.rate_button.configure(state="disabled")

                # Cancel auto-show timer
                if self.feedback_timer:
                    self.after_cancel(self.feedback_timer)
                    self.feedback_timer = None

                # Show thank you message
                self._write_output(f"\n✓ Thank you for your feedback!")

            except Exception as e:
                logger.error(f"Failed to submit feedback: {e}")
                messagebox.showerror("Error", f"Failed to submit feedback: {e}")

            dialog.destroy()

        # Thumbs up button (green)
        thumbs_up_btn = ctk.CTkButton(
            button_frame,
            text="👍 Helpful",
            command=lambda: submit_rating(True),
            width=BUTTON_LG[0],
            height=40,
            fg_color="#2fa572",
            hover_color="#25835e",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold")
        )
        thumbs_up_btn.pack(side="left", padx=SPACE_SM)

        # Thumbs down button (red)
        thumbs_down_btn = ctk.CTkButton(
            button_frame,
            text="👎 Not Helpful",
            command=lambda: submit_rating(False),
            width=BUTTON_LG[0],
            height=40,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            font=ctk.CTkFont(size=FONT_BODY, weight="bold")
        )
        thumbs_down_btn.pack(side="left", padx=SPACE_SM)

        # Skip button
        skip_btn = ctk.CTkButton(
            dialog,
            text="Skip",
            command=dialog.destroy,
            fg_color="gray40",
            hover_color="gray30"
        )
        skip_btn.pack(pady=(10, 0))

        # Bind Escape key to close dialog
        dialog.bind('<Escape>', lambda e: dialog.destroy())
