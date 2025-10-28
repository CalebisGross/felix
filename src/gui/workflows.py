import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import textwrap
from datetime import datetime
from .utils import ThreadManager, log_queue, logger
from src.utils.markdown_formatter import format_synthesis_markdown_detailed
from src.memory.workflow_history import WorkflowHistory
from src.feedback import FeedbackManager, FeedbackIntegrator

class WorkflowsFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for Felix system access
        self.theme_manager = theme_manager
        self.last_workflow_result = None  # Store last workflow result for saving
        self.last_workflow_id = None  # Store last workflow ID for feedback

        # Initialize feedback system
        self.feedback_manager = FeedbackManager()
        self.feedback_integrator = FeedbackIntegrator(self.feedback_manager)

        # Approval polling for workflows
        self.workflow_running = False
        self.approval_polling_active = False
        self.approval_poll_interval = 1000  # Poll every 1 second during workflow
        self.last_approval_check = set()  # Track already-shown approval IDs
        self.dialog_open = False  # Track if approval dialog is currently open

        # Task input (multi-line text widget)
        ttk.Label(self, text="Task:").pack(pady=(10, 0))

        # Frame to hold Text widget and scrollbar
        task_input_frame = ttk.Frame(self)
        task_input_frame.pack(pady=(0, 10), padx=10, fill=tk.X)

        self.task_entry = tk.Text(task_input_frame, wrap=tk.WORD, height=10, width=60)
        task_scrollbar = ttk.Scrollbar(task_input_frame, command=self.task_entry.yview)
        self.task_entry.config(yscrollcommand=task_scrollbar.set)

        self.task_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        task_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Max steps configuration
        max_steps_frame = ttk.Frame(self)
        max_steps_frame.pack(pady=(0, 10), padx=10, fill=tk.X)

        ttk.Label(max_steps_frame, text="Max Steps:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_steps_var = tk.StringVar(value="Auto")
        self.max_steps_dropdown = ttk.Combobox(
            max_steps_frame,
            textvariable=self.max_steps_var,
            values=["Auto", "5", "10", "15", "20"],
            state="readonly",
            width=10
        )
        self.max_steps_dropdown.pack(side=tk.LEFT, padx=(0, 5))

        # Tooltip for max steps
        ttk.Label(
            max_steps_frame,
            text="(Auto = adaptive based on task complexity)",
            foreground="gray",
            font=("TkDefaultFont", 8)
        ).pack(side=tk.LEFT)

        # Continue from previous workflow
        continue_frame = ttk.Frame(self)
        continue_frame.pack(pady=(0, 10), padx=10, fill=tk.X)

        ttk.Label(continue_frame, text="Continue from:").pack(side=tk.LEFT, padx=(0, 5))
        self.parent_workflow_var = tk.StringVar(value="New Workflow")
        self.parent_workflow_dropdown = ttk.Combobox(
            continue_frame,
            textvariable=self.parent_workflow_var,
            state="readonly",
            width=50
        )
        self.parent_workflow_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        self.parent_workflow_dropdown.bind("<<ComboboxSelected>>", self._on_parent_workflow_selected)

        # Refresh button for workflow list
        self.refresh_workflows_button = ttk.Button(continue_frame, text="Refresh", command=self._refresh_workflow_list)
        self.refresh_workflows_button.pack(side=tk.LEFT, padx=(0, 5))

        # Button frame for Run and Save buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=(0, 10))

        # Run button
        self.run_button = ttk.Button(button_frame, text="Run Workflow", command=self.run_workflow, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))

        # Save Results button
        self.save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(5, 0))

        # Rate Workflow button (initially disabled, shows after workflow completes)
        self.rate_button = ttk.Button(button_frame, text="⭐ Rate Workflow", command=self._show_feedback_dialog_from_button, state=tk.DISABLED)
        self.rate_button.pack(side=tk.LEFT, padx=(5, 0))

        # Track feedback timer for auto-show
        self.feedback_timer = None

        # Initially disable features
        self._disable_features()

        # Progress bar
        self.progress = ttk.Progressbar(self, mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Output text with scrollbar (properly contained in frame)
        output_frame = ttk.Frame(self)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, height=20)
        output_scrollbar = ttk.Scrollbar(output_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=output_scrollbar.set)

        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Setup logging handler for this output text widget
        self.log_handler = None
        self._setup_logging()

        # Apply initial theme
        self.apply_theme()

    def _setup_logging(self):
        """Setup logging to route pipeline logs to output text widget."""
        # Setup module-specific logger for workflows to avoid conflicts
        from .logging_handler import setup_gui_logging
        self.workflow_logger = setup_gui_logging(
            text_widget=self.output_text,
            log_queue=None,
            level=logging.INFO,
            module_name='felix_workflows'
        )

        # Also add a root logger handler to catch everything
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.Handler) for h in root_logger.handlers):
            # Only set if root logger has no handlers
            root_logger.setLevel(logging.INFO)

        # Confirm logging setup
        self.workflow_logger.info("Workflows logging initialized successfully")

    def _enable_features(self):
        """Enable workflow features when system is running."""
        self.run_button.config(state=tk.NORMAL)
        self._refresh_workflow_list()

    def _disable_features(self):
        """Disable workflow features when system is not running."""
        self.run_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.rate_button.config(state=tk.DISABLED)
        # Cancel any pending feedback timer
        if self.feedback_timer:
            self.after_cancel(self.feedback_timer)
            self.feedback_timer = None

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
            self.workflow_id_map = {}  # Map display strings to workflow IDs

            for wf in recent_workflows:
                # Truncate task for display
                task_preview = wf.task_input[:60] + "..." if len(wf.task_input) > 60 else wf.task_input
                display_text = f"#{wf.workflow_id}: {task_preview} ({wf.confidence:.2f})"
                workflow_options.append(display_text)
                self.workflow_id_map[display_text] = wf.workflow_id

            self.parent_workflow_dropdown['values'] = workflow_options

        except Exception as e:
            logging.error(f"Failed to refresh workflow list: {e}", exc_info=True)

    def _on_parent_workflow_selected(self, event=None):
        """Handle parent workflow selection."""
        selected = self.parent_workflow_var.get()

        if selected == "New Workflow":
            return

        try:
            from src.memory.workflow_history import WorkflowHistory

            workflow_id = self.workflow_id_map.get(selected)
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
            logging.error(f"Failed to load parent workflow: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load workflow: {str(e)}")

    def _write_output(self, message):
        """Write a message to the output text widget."""
        self.output_text.insert(tk.END, message + '\n')
        self.output_text.see(tk.END)

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
            result = self.last_workflow_result

            # Get agent_manager if available for detailed metrics
            agent_manager = None
            if self.main_app and self.main_app.felix_system:
                agent_manager = self.main_app.felix_system.agent_manager

            # Format results as markdown with detailed agent information
            markdown_content = format_synthesis_markdown_detailed(
                result=result,
                agent_manager=agent_manager,
                include_prompts=True  # Include system/user prompts in detailed view
            )

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

    def run_workflow(self):
        task_input = self.task_entry.get("1.0", tk.END).strip()
        if not task_input:
            messagebox.showwarning("Input Error", "Please enter a task description.")
            return

        # Get max steps override (None for Auto, otherwise integer)
        max_steps_value = self.max_steps_var.get()
        max_steps_override = None if max_steps_value == "Auto" else int(max_steps_value)

        # Get parent workflow ID if continuing
        parent_workflow_id = None
        selected = self.parent_workflow_var.get()
        if selected != "New Workflow" and hasattr(self, 'workflow_id_map'):
            parent_workflow_id = self.workflow_id_map.get(selected)

        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.thread_manager.start_thread(self._run_pipeline_thread, args=(task_input, max_steps_override, parent_workflow_id))

    def _run_pipeline_thread(self, task_input, max_steps_override=None, parent_workflow_id=None):
        def progress_callback(status, progress_percentage):
            """Callback to update GUI progress from pipeline thread."""
            self.after(0, lambda: self._update_progress(status, progress_percentage))
            # Also write status to output
            self.after(0, lambda: self._write_output(f"[{progress_percentage:.0f}%] {status}"))

        # Record start time for tracking
        start_time = datetime.now()

        # Mark workflow as running and start approval polling
        self.workflow_running = True

        # IMPORTANT: Start polling on main GUI thread using after()
        logger.info("=== WORKFLOW STARTED: Scheduling approval polling on GUI thread ===")
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
            self.after(0, lambda: self._write_output("="*60))

            # Use Felix system's integrated workflow runner if available
            if self.main_app and self.main_app.felix_system and self.main_app.system_running:
                self.after(0, lambda: self._write_output("Running through Felix system..."))
                result = self.main_app.felix_system.run_workflow(
                    task_input,
                    progress_callback=progress_callback,
                    max_steps_override=max_steps_override,
                    parent_workflow_id=parent_workflow_id
                )
            else:
                # Felix system not running - cannot run workflow
                self.after(0, lambda: self._write_output("ERROR: Felix system not running"))
                self.after(0, lambda: self._write_output("Please start the Felix system from the Dashboard tab first."))
                result = {
                    "status": "failed",
                    "error": "Felix system not running. Start the system from Dashboard first."
                }

            # Add task_input and timestamps to result for database storage
            end_time = datetime.now()
            result["task_input"] = task_input
            result["start_time"] = start_time.isoformat()
            result["end_time"] = end_time.isoformat()
            result["processing_time"] = (end_time - start_time).total_seconds()

            # Display summary results
            self.after(0, lambda: self._write_output("\n" + "="*60))
            self.after(0, lambda: self._write_output("WORKFLOW SUMMARY"))
            self.after(0, lambda: self._write_output("="*60))

            if result.get("status") == "completed":
                self.after(0, lambda: self._write_output(f"Status: COMPLETED"))
                self.after(0, lambda: self._write_output(f"Agents spawned: {len(result.get('agents_spawned', []))}"))
                self.after(0, lambda: self._write_output(f"LLM responses: {len(result.get('llm_responses', []))}"))
                self.after(0, lambda: self._write_output(f"Completed agents: {result.get('completed_agents', 0)}"))

                # Display final synthesis if available (CentralPost)
                final_synthesis = result.get("centralpost_synthesis")
                if final_synthesis:
                    self.after(0, lambda: self._write_output("\n" + "="*60))
                    self.after(0, lambda: self._write_output("FINAL SYNTHESIS"))
                    self.after(0, lambda: self._write_output("="*60))

                    # Display CentralPost synthesis metrics
                    self.after(0, lambda: self._write_output(f"Generated by: CentralPost (Smart Hub Synthesis)"))

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

                    avg_conf = final_synthesis.get("avg_agent_confidence", 0)
                    self.after(0, lambda avgc=avg_conf: self._write_output(f"Avg Agent Confidence: {avgc:.2f}"))

                    self.after(0, lambda: self._write_output(""))  # Empty line

                    # Wrap text at 80 characters for readability
                    synthesis_content = final_synthesis.get("synthesis_content", "")
                    wrapped_lines = textwrap.wrap(synthesis_content, width=80, break_long_words=False, replace_whitespace=False)

                    for line in wrapped_lines:
                        self.after(0, lambda l=line: self._write_output(l))

                    self.after(0, lambda: self._write_output(""))  # Empty line
                    self.after(0, lambda: self._write_output("="*60))

                self.after(0, lambda: self._write_output("\nWorkflow completed successfully!"))

                # Store result and enable save button
                self.last_workflow_result = result
                self.after(0, lambda: self.save_button.config(state=tk.NORMAL))

                # Automatically save workflow output to database
                try:
                    workflow_history = WorkflowHistory()
                    # Pass parent_workflow_id if this was a continuation
                    parent_id = result.get("parent_workflow_id")
                    workflow_id = workflow_history.save_workflow_output(result, parent_workflow_id=parent_id)
                    if workflow_id:
                        # Store workflow_id for feedback
                        self.last_workflow_id = workflow_id

                        if parent_id:
                            self.after(0, lambda wid=workflow_id, pid=parent_id:
                                      self._write_output(f"Workflow saved to history (ID: {wid}, continuing from #{pid})"))
                        else:
                            self.after(0, lambda wid=workflow_id:
                                      self._write_output(f"Workflow saved to history (ID: {wid})"))
                        logger.info(f"Automatically saved workflow result to database (ID: {workflow_id})")
                    else:
                        self.after(0, lambda: self._write_output("Warning: Failed to save workflow to history"))
                        logger.warning("Failed to save workflow to history database")
                except Exception as save_error:
                    # Don't fail workflow if save fails, just log warning
                    self.after(0, lambda: self._write_output(f"Warning: Could not save to history: {save_error}"))
                    logger.error(f"Error saving workflow to history: {save_error}", exc_info=True)

                # Refresh memory tab to show new knowledge entries and workflow history
                if self.main_app and hasattr(self.main_app, 'memory_frame'):
                    try:
                        # Get the notebook tabs from memory frame
                        for child in self.main_app.memory_frame.notebook.winfo_children():
                            if hasattr(child, 'refresh_entries'):
                                child.refresh_entries()
                            elif hasattr(child, 'refresh_workflows'):
                                # Refresh workflow history tab
                                child.refresh_workflows()
                    except Exception as refresh_error:
                        # Don't fail workflow if refresh fails
                        print(f"Warning: Could not refresh memory tab: {refresh_error}")

                # Enable rate button with highlight after successful workflow completion
                if self.last_workflow_id:
                    self.after(0, lambda: self._enable_rate_button_with_highlight())
                    # Optional: Auto-show dialog after 10 seconds if user doesn't click button
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
        finally:
            # Stop approval polling and mark workflow as not running
            self.workflow_running = False
            self._stop_approval_polling()

            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def _update_progress(self, status, progress_percentage):
        """Update progress bar and status display."""
        # Update progress bar
        self.progress['value'] = progress_percentage

    def _start_approval_polling(self):
        """Start polling for pending approvals during workflow execution."""
        try:
            logger.info("=== APPROVAL POLLING: START REQUESTED ===")
            logger.info(f"  Current state: active={self.approval_polling_active}, workflow_running={self.workflow_running}")

            if not self.approval_polling_active:
                self.approval_polling_active = True
                self.last_approval_check.clear()  # Clear previous approval tracking
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

                    # Show only the FIRST pending approval (workflow pauses for each)
                    if pending_approvals:
                        first_approval = pending_approvals[0]
                        approval_id = first_approval['approval_id']

                        logger.info(f"📋 Approval detected: {approval_id}")
                        logger.info(f"   Command: {first_approval.get('command', 'N/A')[:50]}...")

                        # Only show if not already shown
                        if approval_id not in self.last_approval_check:
                            self.dialog_open = True  # Mark dialog as open BEFORE scheduling
                            self.last_approval_check.add(approval_id)

                            logger.info(f"✓ Opening approval dialog for: {approval_id}")
                            # Write to workflow output so user sees it
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

        # Schedule next poll if still active (but not if dialog is open)
        if self.approval_polling_active and self.workflow_running and not self.dialog_open:
            logger.debug(f"POLL SCHEDULED: Next poll in {self.approval_poll_interval}ms")
            self.after(self.approval_poll_interval, self._poll_for_approvals)
        else:
            logger.debug(f"POLL NOT SCHEDULED: active={self.approval_polling_active}, running={self.workflow_running}, dialog_open={self.dialog_open}")

    def _show_approval_dialog(self, approval_request):
        """Show approval dialog for pending approval request."""
        from .approvals import ApprovalDialog

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

            # Center on parent window
            dialog.transient(self.winfo_toplevel())
            logger.info("✓ Dialog created, waiting for user decision...")
            dialog.wait_window()  # Blocks until dialog is closed
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
            # Always reset dialog_open flag when dialog closes
            self.dialog_open = False
            logger.info("Approval dialog cleanup complete, resuming polling")
            self._write_output("▶️  Workflow resuming, checking for more approvals...")

            # Resume polling immediately to check for next approval
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

    def apply_theme(self):
        """Apply current theme to the workflow widgets."""
        if not self.theme_manager:
            return

        theme = self.theme_manager.get_current_theme()

        # Apply to task entry text widget
        try:
            self.theme_manager.apply_to_text_widget(self.task_entry)
        except Exception as e:
            logger.warning(f"Could not theme task_entry: {e}")

        # Apply to output text widget
        try:
            self.theme_manager.apply_to_text_widget(self.output_text)
        except Exception as e:
            logger.warning(f"Could not theme output_text: {e}")

        # Apply theme to comboboxes
        try:
            style = ttk.Style()
            style.configure("TCombobox",
                          fieldbackground=theme["text_bg"],
                          foreground=theme["text_fg"],
                          selectbackground=theme["text_select_bg"],
                          selectforeground=theme["text_select_fg"])
        except Exception as e:
            logger.warning(f"Could not theme combobox: {e}")

        # Recursively apply theme to all children
        try:
            self.theme_manager.apply_to_all_children(self)
        except Exception as e:
            logger.warning(f"Could not recursively apply theme: {e}")

    def _show_quick_feedback_dialog(self, workflow_id):
        """
        Show quick feedback dialog for workflow rating (thumbs up/down).

        Args:
            workflow_id: ID of completed workflow
        """
        # Create dialog window
        dialog = tk.Toplevel(self)
        dialog.title("Rate this workflow")
        dialog.geometry("400x200")
        dialog.transient(self)
        # dialog.grab_set()  # REMOVED - allows scrolling/clicking workflow output behind dialog

        # Center dialog on parent window
        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() // 2) - (dialog.winfo_width() // 2)
        y = self.winfo_rooty() + (self.winfo_height() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Question label
        ttk.Label(
            dialog,
            text="How would you rate this workflow?",
            font=("TkDefaultFont", 12, "bold")
        ).pack(pady=(20, 10))

        ttk.Label(
            dialog,
            text="Your feedback helps Felix learn and improve!",
            font=("TkDefaultFont", 9)
        ).pack(pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)

        def submit_rating(positive):
            """Submit rating and close dialog."""
            try:
                self.feedback_integrator.submit_workflow_rating_with_propagation(
                    str(workflow_id),
                    positive,
                    knowledge_ids_used=[]  # TODO: Track knowledge IDs in workflow
                )
                logger.info(f"Feedback submitted for workflow {workflow_id}: {'positive' if positive else 'negative'}")

                # Disable rate button after submission
                self.rate_button.config(state=tk.DISABLED)

                # Cancel auto-show timer if it exists
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
        thumbs_up_btn = tk.Button(
            button_frame,
            text="👍 Helpful",
            command=lambda: submit_rating(True),
            bg="#4CAF50",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            width=12,
            height=2,
            relief=tk.RAISED,
            cursor="hand2"
        )
        thumbs_up_btn.pack(side=tk.LEFT, padx=10)

        # Thumbs down button (red)
        thumbs_down_btn = tk.Button(
            button_frame,
            text="👎 Not Helpful",
            command=lambda: submit_rating(False),
            bg="#f44336",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            width=12,
            height=2,
            relief=tk.RAISED,
            cursor="hand2"
        )
        thumbs_down_btn.pack(side=tk.LEFT, padx=10)

        # Skip button
        skip_btn = ttk.Button(
            dialog,
            text="Skip",
            command=dialog.destroy
        )
        skip_btn.pack(pady=(10, 0))

        # Bind Escape key to close dialog
        dialog.bind('<Escape>', lambda e: dialog.destroy())

    def _enable_rate_button_with_highlight(self):
        """
        Enable rate button and add visual highlight to draw attention.
        """
        self.rate_button.config(state=tk.NORMAL)

        # Try to add highlight styling (green background)
        try:
            # Use tk.Button styling since ttk.Button styling is theme-dependent
            # Create a custom style for the button
            style = ttk.Style()
            style.configure("Highlight.TButton",
                          foreground="#4CAF50",  # Green text
                          font=("TkDefaultFont", 10, "bold"))
            self.rate_button.config(style="Highlight.TButton")
            logger.info("Rate button enabled with highlight")
        except Exception as e:
            # Fallback: Just enable without special styling
            logger.warning(f"Could not apply button highlight: {e}")

    def _show_feedback_dialog_from_button(self):
        """
        Show feedback dialog when user clicks the Rate Workflow button.
        """
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
        """
        Auto-show feedback dialog after delay if user hasn't clicked the button.

        Only shows if button is still enabled (user hasn't rated yet).
        """
        # Check if button is still enabled (not yet rated)
        if str(self.rate_button['state']) == 'normal':
            logger.info("Auto-showing feedback dialog (10 second timer elapsed)")
            if self.last_workflow_id:
                self._show_quick_feedback_dialog(self.last_workflow_id)
        else:
            logger.info("Skipping auto-show (already rated)")