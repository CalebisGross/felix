import tkinter as tk
from tkinter import ttk, messagebox
import logging
from .utils import ThreadManager, log_queue, logger
from src.pipeline import linear_pipeline

class WorkflowsFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, main_app=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for Felix system access

        # Task input
        ttk.Label(self, text="Task:").pack(pady=(10, 0))
        self.task_entry = ttk.Entry(self, width=50)
        self.task_entry.pack(pady=(0, 10))

        # Run button
        self.run_button = ttk.Button(self, text="Run Workflow", command=self.run_workflow, state=tk.DISABLED)
        self.run_button.pack(pady=(0, 10))

        # Initially disable features
        self._disable_features()

        # Progress bar
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Output text
        self.output_text = tk.Text(self, wrap=tk.WORD, height=20)
        scrollbar = ttk.Scrollbar(self, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Poll log queue
        self.poll_log_queue()

    def _enable_features(self):
        """Enable workflow features when system is running."""
        self.run_button.config(state=tk.NORMAL)

    def _disable_features(self):
        """Disable workflow features when system is not running."""
        self.run_button.config(state=tk.DISABLED)

    def poll_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.output_text.insert(tk.END, msg + '\n')
                self.output_text.see(tk.END)
        except:
            pass
        self.after(100, self.poll_log_queue)

    def run_workflow(self):
        task_input = self.task_entry.get().strip()
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

        try:
            logger.info("Starting workflow for task: %s", task_input)
            result = linear_pipeline.run(task_input, progress_callback=progress_callback)

            # Log summary results
            if result.get("status") == "completed":
                logger.info("Workflow completed successfully")
                logger.info("Agents spawned: %d", len(result.get("agents_spawned", [])))
                logger.info("LLM responses: %d", len(result.get("llm_responses", [])))
                logger.info("Completed agents: %d", result.get("completed_agents", 0))
            else:
                logger.error("Workflow failed: %s", result.get("error", "Unknown error"))

        except Exception as e:
            logger.error("Error running workflow: %s", str(e))
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to run workflow: {e}"))
        finally:
            self.after(0, lambda: self.progress.stop())
            self.after(0, lambda: self.run_button.config(state=tk.NORMAL))

    def _update_progress(self, status, progress_percentage):
        """Update progress bar and status display."""
        # Update progress bar
        self.progress['value'] = progress_percentage

        # Log status to output
        logger.info("Progress: %s (%.1f%%)", status, progress_percentage)