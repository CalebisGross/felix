import tkinter as tk
from tkinter import ttk, messagebox
import logging
from .utils import ThreadManager, log_queue, logger

class WorkflowsFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, main_app=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for Felix system access

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

        # Run button
        self.run_button = ttk.Button(self, text="Run Workflow", command=self.run_workflow, state=tk.DISABLED)
        self.run_button.pack(pady=(0, 10))

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

    def run_workflow(self):
        task_input = self.task_entry.get("1.0", tk.END).strip()
        if not task_input:
            messagebox.showwarning("Input Error", "Please enter a task description.")
            return

        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.thread_manager.start_thread(self._run_pipeline_thread, args=(task_input,))

    def _run_pipeline_thread(self, task_input):
        def progress_callback(status, progress_percentage):
            """Callback to update GUI progress from pipeline thread."""
            self.after(0, lambda: self._update_progress(status, progress_percentage))
            # Also write status to output
            self.after(0, lambda: self._write_output(f"[{progress_percentage:.0f}%] {status}"))

        try:
            self.after(0, lambda: self._write_output(f"Starting workflow for task: {task_input}"))
            self.after(0, lambda: self._write_output("="*60))

            # Use Felix system's integrated workflow runner if available
            if self.main_app and self.main_app.felix_system and self.main_app.system_running:
                self.after(0, lambda: self._write_output("Running through Felix system..."))
                result = self.main_app.felix_system.run_workflow(task_input, progress_callback=progress_callback)
            else:
                # Felix system not running - cannot run workflow
                self.after(0, lambda: self._write_output("ERROR: Felix system not running"))
                self.after(0, lambda: self._write_output("Please start the Felix system from the Dashboard tab first."))
                result = {
                    "status": "failed",
                    "error": "Felix system not running. Start the system from Dashboard first."
                }

            # Display summary results
            self.after(0, lambda: self._write_output("\n" + "="*60))
            self.after(0, lambda: self._write_output("WORKFLOW SUMMARY"))
            self.after(0, lambda: self._write_output("="*60))

            if result.get("status") == "completed":
                self.after(0, lambda: self._write_output(f"Status: COMPLETED"))
                self.after(0, lambda: self._write_output(f"Agents spawned: {len(result.get('agents_spawned', []))}"))
                self.after(0, lambda: self._write_output(f"LLM responses: {len(result.get('llm_responses', []))}"))
                self.after(0, lambda: self._write_output(f"Completed agents: {result.get('completed_agents', 0)}"))

                # Display sample LLM responses
                llm_responses = result.get("llm_responses", [])
                if llm_responses:
                    self.after(0, lambda: self._write_output(f"\nSample LLM Responses (showing first 3):"))
                    for i, resp in enumerate(llm_responses[:3]):
                        if isinstance(resp, dict) and 'response' in resp:
                            agent_id = resp.get('agent_id', 'unknown')
                            response_text = resp['response'][:2000] + "..." if len(resp['response']) > 2000 else resp['response']
                            self.after(0, lambda aid=agent_id, rt=response_text: self._write_output(f"  [{aid}]: {rt}"))

                self.after(0, lambda: self._write_output("\nWorkflow completed successfully!"))

                # Refresh memory tab to show new knowledge entries
                if self.main_app and hasattr(self.main_app, 'memory_frame'):
                    try:
                        # Get the notebook tabs from memory frame
                        for child in self.main_app.memory_frame.notebook.winfo_children():
                            if hasattr(child, 'refresh_entries'):
                                child.refresh_entries()
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