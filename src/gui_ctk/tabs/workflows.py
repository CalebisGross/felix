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

logger = logging.getLogger(__name__)


class WorkflowsTab(ctk.CTkFrame):
    """
    Workflows tab for running Felix workflows with real-time feedback.

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
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
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

        # Main container with scrolling
        main_container = ctk.CTkScrollableFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Task input section
        task_label = ctk.CTkLabel(
            main_container,
            text="Task:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        task_label.pack(anchor="w", pady=(0, 5))

        self.task_entry = ctk.CTkTextbox(main_container, height=150, wrap="word")
        self.task_entry.pack(fill="x", pady=(0, 15))

        # Options row (max steps and continue from)
        options_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        options_frame.pack(fill="x", pady=(0, 15))

        # Max steps
        max_steps_container = ctk.CTkFrame(options_frame, fg_color="transparent")
        max_steps_container.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(
            max_steps_container,
            text="Max Steps:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))

        self.max_steps_var = ctk.StringVar(value="Auto")
        self.max_steps_dropdown = ctk.CTkComboBox(
            max_steps_container,
            values=["Auto", "5", "10", "15", "20"],
            variable=self.max_steps_var,
            width=100,
            state="readonly"
        )
        self.max_steps_dropdown.pack(side="left", padx=(0, 5))

        ctk.CTkLabel(
            max_steps_container,
            text="(Auto = adaptive)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left")

        # Continue from workflow
        continue_container = ctk.CTkFrame(options_frame, fg_color="transparent")
        continue_container.pack(side="left")

        ctk.CTkLabel(
            continue_container,
            text="Continue from:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 5))

        self.parent_workflow_var = ctk.StringVar(value="New Workflow")
        self.parent_workflow_dropdown = ctk.CTkComboBox(
            continue_container,
            values=["New Workflow"],
            variable=self.parent_workflow_var,
            width=400,
            state="readonly",
            command=self._on_parent_workflow_selected
        )
        self.parent_workflow_dropdown.pack(side="left", padx=(0, 5))

        self.refresh_workflows_button = ctk.CTkButton(
            continue_container,
            text="Refresh",
            command=self._refresh_workflow_list,
            width=80
        )
        self.refresh_workflows_button.pack(side="left")

        # Button row
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        button_frame.pack(fill="x", pady=(0, 15))

        self.run_button = ctk.CTkButton(
            button_frame,
            text="Run Workflow",
            command=self.run_workflow,
            width=130,
            fg_color="#2fa572",
            hover_color="#25835e",
            state="disabled"
        )
        self.run_button.pack(side="left", padx=(0, 5))

        self.save_button = ctk.CTkButton(
            button_frame,
            text="Save Results",
            command=self.save_results,
            width=130,
            state="disabled"
        )
        self.save_button.pack(side="left", padx=(0, 5))

        self.rate_button = ctk.CTkButton(
            button_frame,
            text="⭐ Rate Workflow",
            command=self._show_feedback_dialog_from_button,
            width=150,
            state="disabled"
        )
        self.rate_button.pack(side="left")

        # Progress bar
        self.progress = ctk.CTkProgressBar(main_container, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 15))
        self.progress.set(0)

        # Output display
        output_label = ctk.CTkLabel(
            main_container,
            text="Workflow Output:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        output_label.pack(anchor="w", pady=(0, 5))

        self.output_text = ctk.CTkTextbox(main_container, height=400, wrap="word")
        self.output_text.pack(fill="both", expand=True)

        # Initially disable features
        self._disable_features()

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

    def _poll_for_approvals(self):
        """Poll for pending approvals and show dialog if found."""
        logger.debug(f"POLL CHECK: active={self.approval_polling_active}, workflow_running={self.workflow_running}, dialog_open={self.dialog_open}")

        if not self.approval_polling_active or not self.workflow_running:
            logger.debug("POLL SKIPPED: Conditions not met")
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
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(30, 10))

        ctk.CTkLabel(
            dialog,
            text="Your feedback helps Felix learn and improve!",
            font=ctk.CTkFont(size=11)
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
            width=140,
            height=50,
            fg_color="#2fa572",
            hover_color="#25835e",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        thumbs_up_btn.pack(side="left", padx=10)

        # Thumbs down button (red)
        thumbs_down_btn = ctk.CTkButton(
            button_frame,
            text="👎 Not Helpful",
            command=lambda: submit_rating(False),
            width=140,
            height=50,
            fg_color="#dc2626",
            hover_color="#b91c1c",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        thumbs_down_btn.pack(side="left", padx=10)

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
