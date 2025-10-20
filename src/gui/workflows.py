import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import logging
import textwrap
from datetime import datetime
from .utils import ThreadManager, log_queue, logger

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
        """Save workflow results to a formatted text file."""
        if not self.last_workflow_result:
            messagebox.showwarning("No Results", "No workflow results to save. Run a workflow first.")
            return

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"workflow_results_{timestamp}.txt"

        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if not file_path:
            return  # User cancelled

        try:
            result = self.last_workflow_result
            with open(file_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("═" * 60 + "\n")
                f.write("FELIX WORKFLOW RESULTS\n")
                f.write("═" * 60 + "\n\n")

                # Task and metadata
                f.write(f"Task: {result.get('task', 'N/A')}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: {result.get('status', 'unknown').upper()}\n")
                f.write(f"Agents Spawned: {len(result.get('agents_spawned', []))}\n")
                f.write(f"Completed Agents: {result.get('completed_agents', 0)}\n\n")

                # Final Synthesis Section
                final_synthesis = result.get("final_synthesis")
                if final_synthesis:
                    f.write("═" * 60 + "\n")
                    f.write("FINAL SYNTHESIS\n")
                    f.write("═" * 60 + "\n")

                    # Get detailed metrics for synthesis agent
                    synthesis_agent_id = final_synthesis.get('agent_id', 'unknown')
                    synthesis_data = None
                    if self.main_app and self.main_app.felix_system:
                        synthesis_data = self.main_app.felix_system.agent_manager.get_agent_output(synthesis_agent_id)

                    f.write(f"Generated by: {synthesis_agent_id}\n")
                    f.write(f"Confidence: {final_synthesis.get('confidence', 0.0):.2f}\n")

                    if synthesis_data:
                        f.write(f"Processing Time: {synthesis_data.get('processing_time', 0):.2f}s\n")
                        f.write(f"Tokens: {synthesis_data.get('tokens_used', 0)} / {synthesis_data.get('token_budget', 0)}\n")
                        f.write(f"Temperature: {synthesis_data.get('temperature', 0):.2f}\n")
                        collab_count = synthesis_data.get('collaborative_count', 0)
                        if collab_count > 0:
                            f.write(f"Collaborative Context: {collab_count} agents\n")
                    f.write("\n")

                    # Wrap synthesis content at 80 characters
                    synthesis_content = final_synthesis.get("content", "")
                    wrapped_lines = textwrap.wrap(synthesis_content, width=80, break_long_words=False)
                    for line in wrapped_lines:
                        f.write(line + "\n")
                    f.write("\n")

                # Agent Outputs Section
                llm_responses = result.get("llm_responses", [])
                if llm_responses:
                    f.write("═" * 60 + "\n")
                    f.write("AGENT OUTPUTS\n")
                    f.write("═" * 60 + "\n\n")

                    for i, resp in enumerate(llm_responses, 1):
                        agent_id = resp.get('agent_id', 'unknown')
                        agent_type = resp.get('agent_type', 'unknown')
                        confidence = resp.get('confidence', 0.0)
                        response_text = resp.get('response', '')

                        # Get comprehensive metrics from agent_manager
                        agent_data = None
                        if self.main_app and self.main_app.felix_system:
                            agent_data = self.main_app.felix_system.agent_manager.get_agent_output(agent_id)

                        f.write(f"[{agent_type.capitalize()} Agent: {agent_id}] - Confidence: {confidence:.2f}\n")
                        f.write("─" * 60 + "\n")

                        # Add comprehensive metrics if available
                        if agent_data:
                            position_info = agent_data.get('position_info', {})
                            depth_ratio = position_info.get('depth_ratio', 0)
                            phase = "Exploration" if depth_ratio < 0.3 else ("Analysis" if depth_ratio < 0.7 else "Synthesis")

                            f.write(f"Type: {agent_type.capitalize()} | Position: {depth_ratio:.2f} ({phase})\n")
                            f.write(f"Processing: {agent_data.get('processing_time', 0):.2f}s | ")
                            f.write(f"Tokens: {agent_data.get('tokens_used', 0)}/{agent_data.get('token_budget', 0)} | ")
                            f.write(f"Temp: {agent_data.get('temperature', 0):.2f}\n")
                            f.write(f"Model: {agent_data.get('model', 'unknown')}\n")

                            collab_count = agent_data.get('collaborative_count', 0)
                            if collab_count > 0:
                                f.write(f"Collaborative: {collab_count} prior agent outputs\n")
                            else:
                                f.write("Collaborative: No prior context\n")

                            f.write("\n")

                            # Add prompts section
                            system_prompt = agent_data.get('system_prompt', '')
                            user_prompt = agent_data.get('user_prompt', '')

                            if system_prompt:
                                f.write("SYSTEM PROMPT:\n")
                                wrapped_system = textwrap.wrap(system_prompt, width=80, break_long_words=False)
                                for line in wrapped_system:
                                    f.write(line + "\n")
                                f.write("\n")

                            if user_prompt:
                                f.write("USER PROMPT:\n")
                                wrapped_user = textwrap.wrap(user_prompt, width=80, break_long_words=False)
                                for line in wrapped_user:
                                    f.write(line + "\n")
                                f.write("\n")

                        f.write("OUTPUT:\n")
                        # Wrap response text at 80 characters
                        wrapped_lines = textwrap.wrap(response_text, width=80, break_long_words=False)
                        for line in wrapped_lines:
                            f.write(line + "\n")
                        f.write("\n")

                # Footer
                f.write("═" * 60 + "\n")
                f.write("END OF REPORT\n")
                f.write("═" * 60 + "\n")

            messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

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

                # Display final synthesis if available
                final_synthesis = result.get("final_synthesis")
                if final_synthesis:
                    self.after(0, lambda: self._write_output("\n" + "="*60))
                    self.after(0, lambda: self._write_output("FINAL SYNTHESIS"))
                    self.after(0, lambda: self._write_output("="*60))

                    # Get detailed metrics from agent_manager if available
                    agent_id = final_synthesis.get("agent_id", "unknown")
                    agent_data = None
                    if self.main_app and self.main_app.felix_system:
                        agent_data = self.main_app.felix_system.agent_manager.get_agent_output(agent_id)

                    # Display metrics header
                    self.after(0, lambda aid=agent_id: self._write_output(f"Generated by: {aid}"))
                    confidence = final_synthesis.get("confidence", 0.0)
                    self.after(0, lambda c=confidence: self._write_output(f"Confidence: {c:.2f}"))

                    if agent_data:
                        processing_time = agent_data.get("processing_time", 0)
                        self.after(0, lambda pt=processing_time: self._write_output(f"Processing Time: {pt:.2f}s"))

                        tokens_used = agent_data.get("tokens_used", 0)
                        token_budget = agent_data.get("token_budget", 0)
                        self.after(0, lambda tu=tokens_used, tb=token_budget: self._write_output(f"Tokens: {tu} / {tb}"))

                        temperature = agent_data.get("temperature", 0)
                        self.after(0, lambda temp=temperature: self._write_output(f"Temperature: {temp:.2f}"))

                        collab_count = agent_data.get("collaborative_count", 0)
                        if collab_count > 0:
                            self.after(0, lambda cc=collab_count: self._write_output(f"Collaborative Context: {cc} agents"))

                    self.after(0, lambda: self._write_output(""))  # Empty line

                    # Wrap text at 80 characters for readability
                    synthesis_content = final_synthesis.get("content", "")
                    wrapped_lines = textwrap.wrap(synthesis_content, width=80, break_long_words=False, replace_whitespace=False)

                    for line in wrapped_lines:
                        self.after(0, lambda l=line: self._write_output(l))

                    self.after(0, lambda: self._write_output(""))  # Empty line
                    self.after(0, lambda: self._write_output("="*60))

                self.after(0, lambda: self._write_output("\nWorkflow completed successfully!"))

                # Store result and enable save button
                self.last_workflow_result = result
                self.after(0, lambda: self.save_button.config(state=tk.NORMAL))

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

    def apply_theme(self):
        """Apply current theme to the workflow widgets."""
        if self.theme_manager:
            self.theme_manager.apply_to_text_widget(self.task_entry)
            self.theme_manager.apply_to_text_widget(self.output_text)