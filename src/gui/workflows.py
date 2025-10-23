import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import textwrap
from datetime import datetime
from .utils import ThreadManager, log_queue, logger
from src.utils.markdown_formatter import format_synthesis_markdown_detailed
from src.memory.workflow_history import WorkflowHistory

class WorkflowsFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for Felix system access
        self.theme_manager = theme_manager
        self.last_workflow_result = None  # Store last workflow result for saving

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

        # Button frame for Run and Save buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=(0, 10))

        # Run button
        self.run_button = ttk.Button(button_frame, text="Run Workflow", command=self.run_workflow, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=(0, 5))

        # Save Results button
        self.save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(5, 0))

        # Initially disable features
        self._disable_features()

        # Progress bar
        self.progress = ttk.Progressbar(self, mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Output text
        self.output_text = tk.Text(self, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(self, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

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

    def _disable_features(self):
        """Disable workflow features when system is not running."""
        self.run_button.config(state=tk.DISABLED)

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

        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.thread_manager.start_thread(self._run_pipeline_thread, args=(task_input, max_steps_override))

    def _run_pipeline_thread(self, task_input, max_steps_override=None):
        def progress_callback(status, progress_percentage):
            """Callback to update GUI progress from pipeline thread."""
            self.after(0, lambda: self._update_progress(status, progress_percentage))
            # Also write status to output
            self.after(0, lambda: self._write_output(f"[{progress_percentage:.0f}%] {status}"))

        # Record start time for tracking
        start_time = datetime.now()

        try:
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
                    max_steps_override=max_steps_override
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
                    workflow_id = workflow_history.save_workflow_output(result)
                    if workflow_id:
                        self.after(0, lambda wid=workflow_id: self._write_output(f"Workflow saved to history (ID: {wid})"))
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
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def _update_progress(self, status, progress_percentage):
        """Update progress bar and status display."""
        # Update progress bar
        self.progress['value'] = progress_percentage

    def apply_theme(self):
        """Apply current theme to the workflow widgets."""
        if self.theme_manager:
            self.theme_manager.apply_to_text_widget(self.task_entry)
            self.theme_manager.apply_to_text_widget(self.output_text)