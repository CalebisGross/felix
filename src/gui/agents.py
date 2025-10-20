import tkinter as tk
from tkinter import ttk, messagebox
import time
import textwrap
from .utils import ThreadManager, logger
try:
    from ..agents import dynamic_spawning
    from ..agents.agent import AgentState
except ImportError:
    dynamic_spawning = None
    AgentState = None
try:
    from ..communication import mesh
except ImportError:
    mesh = None
try:
    from ..agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent
except ImportError:
    ResearchAgent = AnalysisAgent = SynthesisAgent = CriticAgent = None
try:
    from ..communication.central_post import CentralPost, Message, MessageType
except ImportError:
    CentralPost = Message = MessageType = None

class AgentsFrame(ttk.Frame):
    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for core system access
        self.theme_manager = theme_manager
        self.agents = []  # Reference to main system's agent list
        self.agent_counter = 0
        self.recent_messages = []  # For dynamic spawning analysis
        self.polling_active = False
        self._updating_tree = False  # Flag to track programmatic treeview updates
        self._last_selected_agent_id = None  # Track last agent shown in monitor

        # Label + Combobox for type
        ttk.Label(self, text="Agent Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.type_combo = ttk.Combobox(self, values=["Research", "Analysis", "Synthesis", "Critic"], state="readonly")
        self.type_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        # Specialized parameters
        ttk.Label(self, text="Domain/Focus:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.domain_entry = ttk.Entry(self)
        self.domain_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.domain_entry.insert(0, "general")  # Default value

        # Spawn button
        self.spawn_button = ttk.Button(self, text="Spawn", command=self.spawn_agent, state=tk.DISABLED)
        self.spawn_button.grid(row=0, column=2, padx=5, pady=5)

        # Treeview for active agents
        self.tree = ttk.Treeview(self, columns=("Type", "Position", "State", "Progress", "Confidence", "Velocity"), show="headings", height=10)
        self.tree.heading("Type", text="Type")
        self.tree.heading("Position", text="Position")
        self.tree.heading("State", text="State")
        self.tree.heading("Progress", text="Progress")
        self.tree.heading("Confidence", text="Confidence")
        self.tree.heading("Velocity", text="Velocity")

        # Set column widths
        self.tree.column("Type", width=80)
        self.tree.column("Position", width=80)
        self.tree.column("State", width=80)
        self.tree.column("Progress", width=80)
        self.tree.column("Confidence", width=80)
        self.tree.column("Velocity", width=80)

        scrollbar = ttk.Scrollbar(self, command=self.tree.yview)
        self.tree.config(yscrollcommand=scrollbar.set)
        self.tree.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        scrollbar.grid(row=2, column=2, sticky='ns')

        self.tree.bind('<<TreeviewSelect>>', self.on_select)

        # Interact Entry + Send button
        ttk.Label(self, text="Message:").grid(row=3, column=0, sticky='w', padx=5)
        self.message_entry = ttk.Entry(self)
        self.message_entry.grid(row=3, column=1, sticky='ew', padx=5)
        self.send_button = ttk.Button(self, text="Send", command=self.send_message)
        self.send_button.grid(row=3, column=2, padx=5)

        # Initially disable features
        self._disable_features()

        # Start polling for updates if main_app is available
        self._start_polling()

        # Monitor Text
        ttk.Label(self, text="Monitor:").grid(row=4, column=0, sticky='w', padx=5)
        self.monitor_text = tk.Text(self, height=10, wrap=tk.WORD, state='disabled')
        monitor_scrollbar = ttk.Scrollbar(self, command=self.monitor_text.yview)
        self.monitor_text.config(yscrollcommand=monitor_scrollbar.set)
        self.monitor_text.grid(row=5, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        monitor_scrollbar.grid(row=5, column=2, sticky='ns')

        # Grid weights
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Apply initial theme
        self.apply_theme()

    def _enable_features(self):
        """Enable agent features when system is running."""
        if self.main_app and hasattr(self.main_app, 'felix_system') and self.main_app.felix_system:
            self.spawn_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self._append_monitor("System ready - GUI fully integrated with Felix architecture")
            self._start_polling()
        else:
            self._append_monitor("Main system components not available - some features disabled")

    def _start_polling(self):
        """Start polling for agent updates."""
        if self.polling_active:
            return
        self.polling_active = True
        self._poll_updates()

    def _stop_polling(self):
        """Stop polling for updates."""
        self.polling_active = False

    def _poll_updates(self):
        """Poll for agent updates every 1-2 seconds."""
        if not self.polling_active:
            return

        try:
            self._update_agents_from_main()
            self._update_treeview()
        except Exception as e:
            logger.warning(f"Error during polling update: {e}")

        # Schedule next poll in 1.5 seconds
        self.after(1500, self._poll_updates)

    def _update_agents_from_main(self):
        """Update local agent list from main system's agent manager."""
        if not self.main_app or not self.main_app.felix_system:
            return

        # Get agents from felix_system's agent_manager
        self.agents = self.main_app.felix_system.agent_manager.get_all_agents()

    def _disable_features(self):
        """Disable agent features when system is not running."""
        self.spawn_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        # Monitor text is created after this call, so delay the message
        self.after(100, lambda: self._append_monitor("System not ready - waiting for main Felix system initialization..."))

    def _is_system_ready(self):
        """Check if the system is running and ready for agent operations."""
        return (self.main_app and
                self.main_app.system_running and
                self.main_app.felix_system and
                self.main_app.felix_system.lm_client and
                self.main_app.felix_system.lm_client.test_connection())

    def spawn_agent(self):
        agent_type = self.type_combo.get()
        if not agent_type:
            messagebox.showwarning("Input Error", "Please select an agent type.")
            return

        # Check if Felix system is available
        if not self.main_app or not self.main_app.felix_system:
            messagebox.showerror("System Not Ready",
                                "Felix system not available. Please start the system first.")
            return

        # Get specialized parameters
        domain = self.domain_entry.get().strip() or "general"

        self.thread_manager.start_thread(self._spawn_thread, args=(agent_type, domain))

    def _spawn_thread(self, agent_type, domain):
        try:
            # Check if Felix system is available
            if not self.main_app or not self.main_app.felix_system:
                self.after(0, lambda: messagebox.showerror("Error", "Felix system not available"))
                return

            # Spawn agent through the unified Felix system
            agent = self.main_app.felix_system.spawn_agent(
                agent_type=agent_type,
                domain=domain
            )

            if agent:
                logger.info(f"Spawned {agent_type} agent: {agent.agent_id} (domain: {domain})")
                self.after(0, lambda: self._append_monitor(
                    f"Successfully spawned {agent_type} agent: {agent.agent_id}\n"
                    f"Domain: {domain}\n"
                    f"Spawn time: {agent.spawn_time:.2f}"
                ))
                # Refresh agent list
                self.after(0, self._update_treeview)
            else:
                logger.warning(f"Failed to spawn {agent_type} agent")
                self.after(0, lambda: self._append_monitor(
                    f"Failed to spawn {agent_type} agent - check system logs"
                ))

        except Exception as e:
            logger.error(f"Error spawning agent: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to spawn agent: {error_msg}"))

    def _update_treeview(self):
        """Update treeview with current agent states while preserving selection."""
        # Set flag to indicate programmatic update
        self._updating_tree = True

        # Remember current selection before clearing
        selected_items = self.tree.selection()
        selected_agent_id = None
        if selected_items:
            try:
                selected_agent_id = self.tree.item(selected_items[0], "tags")[0]
            except (IndexError, KeyError):
                selected_agent_id = None

        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add agents from core system
        for agent in self.agents:
            try:
                # Get agent properties with safe access
                agent_type = getattr(agent, 'agent_type', 'unknown')
                agent_id = getattr(agent, 'agent_id', 'unknown')
                # Convert AgentState enum to string value
                state = getattr(agent, 'state', 'unknown')
                if hasattr(state, 'value') and not isinstance(state, str):
                    state = state.value

                # Try to get position info
                position_str = "N/A"
                try:
                    if hasattr(agent, 'get_position_info'):
                        position_info = agent.get_position_info(0.1)
                        depth_ratio = position_info.get("depth_ratio", 0.0)
                        position_str = f"{depth_ratio:.2f}"
                except Exception:
                    pass

                # Get other properties with defaults
                confidence = getattr(agent, 'confidence', 0.0)
                velocity = getattr(agent, 'velocity', 0.0)
                progress = getattr(agent, 'progress', 0.0)

                # Format display values
                confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "N/A"
                velocity_str = f"{velocity:.2f}" if isinstance(velocity, (int, float)) else "N/A"
                progress_str = f"{progress:.1%}" if isinstance(progress, (int, float)) else "N/A"

                # Insert into treeview
                item_id = self.tree.insert("", tk.END, values=(
                    agent_type.capitalize() if isinstance(agent_type, str) else str(agent_type),
                    position_str,
                    state,
                    progress_str,
                    confidence_str,
                    velocity_str
                ), tags=(agent_id,))

                # Restore selection if this was the selected agent
                if selected_agent_id and agent_id == selected_agent_id:
                    self.tree.selection_set(item_id)
                    self.tree.see(item_id)  # Scroll into view if needed

            except Exception as e:
                logger.warning(f"Error updating treeview for agent: {e}")
                # Fallback display with minimal info
                try:
                    agent_id = getattr(agent, 'agent_id', 'unknown')
                    agent_type = getattr(agent, 'agent_type', 'unknown')
                    self.tree.insert("", tk.END, values=(
                        agent_type,
                        "N/A",
                        "error",
                        "N/A",
                        "N/A",
                        "N/A"
                    ), tags=(agent_id,))
                except Exception:
                    # Complete fallback
                    pass

        # Clear the updating flag after all updates are done
        self._updating_tree = False

    def on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        agent_id = self.tree.item(item, "tags")[0]

        # Skip monitor update if this is a programmatic update or same agent
        if self._updating_tree or agent_id == self._last_selected_agent_id:
            return

        # Update last selected agent
        self._last_selected_agent_id = agent_id

        # Find agent in list
        agent = None
        for a in self.agents:
            if getattr(a, 'agent_id', None) == agent_id:
                agent = a
                break

        if not agent:
            self._append_monitor(f"Agent {agent_id} not found")
            return

        # Get detailed agent information with robust error handling
        details = []

        # Basic info
        details.append(f"ID: {getattr(agent, 'agent_id', 'N/A')}")
        details.append(f"Type: {getattr(agent, 'agent_type', 'N/A')}")
        # Convert AgentState enum to string
        state = getattr(agent, 'state', 'N/A')
        if hasattr(state, 'value') and not isinstance(state, str):
            state = state.value
        details.append(f"State: {state}")

        # Position info
        try:
            if hasattr(agent, 'get_position_info'):
                position_info = agent.get_position_info(0.1)
                depth_ratio = position_info.get('depth_ratio', 'N/A')
                if isinstance(depth_ratio, (int, float)):
                    details.append(f"Position: Depth {depth_ratio:.2f}")
                else:
                    details.append(f"Position: {depth_ratio}")
        except Exception as e:
            details.append(f"Position: Error ({e})")

        # Performance metrics
        confidence = getattr(agent, 'confidence', None)
        if confidence is not None and isinstance(confidence, (int, float)):
            details.append(f"Confidence: {confidence:.2f}")

        progress = getattr(agent, 'progress', None)
        if progress is not None and isinstance(progress, (int, float)):
            details.append(f"Progress: {progress:.1%}")

        velocity = getattr(agent, 'velocity', None)
        if velocity is not None and isinstance(velocity, (int, float)):
            details.append(f"Velocity: {velocity:.2f}")

        # Spawn time
        spawn_time = getattr(agent, 'spawn_time', None)
        if spawn_time is not None:
            details.append(f"Spawn Time: {spawn_time}")

        # Add specialized information based on agent type
        agent_type = getattr(agent, 'agent_type', '').lower()

        if agent_type == 'research' and hasattr(agent, 'research_domain'):
            details.append(f"Research Domain: {agent.research_domain}")
        elif agent_type == 'analysis' and hasattr(agent, 'analysis_type'):
            details.append(f"Analysis Type: {agent.analysis_type}")
        elif agent_type == 'synthesis' and hasattr(agent, 'output_format'):
            details.append(f"Output Format: {agent.output_format}")
        elif agent_type == 'critic' and hasattr(agent, 'review_focus'):
            details.append(f"Review Focus: {agent.review_focus}")

        # Get agent output with comprehensive metrics if available
        output_text = ""
        prompts_text = ""
        if self.main_app and hasattr(self.main_app, 'felix_system') and self.main_app.felix_system:
            output_data = self.main_app.felix_system.agent_manager.get_agent_output(agent_id)
            if output_data:
                # Format prompts section
                prompts_text = "\n\n" + "="*60 + "\n"
                prompts_text += "PROMPTS\n"
                prompts_text += "="*60 + "\n\n"

                prompts_text += "System Prompt:\n"
                prompts_text += "-" * 60 + "\n"
                system_prompt = output_data.get("system_prompt", "N/A")
                wrapped_system = textwrap.fill(system_prompt, width=80, break_long_words=False)
                prompts_text += wrapped_system + "\n\n"

                prompts_text += "User Prompt:\n"
                prompts_text += "-" * 60 + "\n"
                user_prompt = output_data.get("user_prompt", "N/A")
                wrapped_user = textwrap.fill(user_prompt, width=80, break_long_words=False)
                prompts_text += wrapped_user + "\n"

                # Add collaborative context info
                collab_count = output_data.get("collaborative_count", 0)
                if collab_count > 0:
                    prompts_text += f"\nCollaborative Context: {collab_count} previous agent outputs"

                # Format output section
                output_text = "\n\n" + "="*60 + "\n"
                output_text += "OUTPUT\n"
                output_text += "="*60 + "\n\n"

                # Wrap the output text at 80 characters for readability
                wrapped_output = textwrap.fill(output_data["output"], width=80, break_long_words=False)
                output_text += wrapped_output + "\n\n"

                # Add comprehensive metrics
                output_text += f"Confidence: {output_data['confidence']:.2f}\n"
                output_text += f"Processing Time: {output_data.get('processing_time', 0):.2f}s\n"
                output_text += f"Temperature: {output_data.get('temperature', 0):.2f}\n"
                tokens_used = output_data.get('tokens_used', 0)
                token_budget = output_data.get('token_budget', 0)
                output_text += f"Tokens: {tokens_used} / {token_budget}\n"
                output_text += f"Model: {output_data.get('model', 'unknown')}\n"
                position_info = output_data.get('position_info', {})
                if position_info:
                    depth_ratio = position_info.get('depth_ratio', 0)
                    output_text += f"Helix Position: {depth_ratio:.2f} "
                    if depth_ratio < 0.3:
                        output_text += "(Exploration Phase)\n"
                    elif depth_ratio < 0.7:
                        output_text += "(Analysis Phase)\n"
                    else:
                        output_text += "(Synthesis Phase)\n"

        # Display in monitor
        self.monitor_text.config(state='normal')
        self.monitor_text.delete(1.0, tk.END)

        # Add section header for agent details
        self.monitor_text.insert(tk.END, "="*60 + "\n")
        self.monitor_text.insert(tk.END, "AGENT DETAILS\n")
        self.monitor_text.insert(tk.END, "="*60 + "\n")
        self.monitor_text.insert(tk.END, '\n'.join(details))

        # Add prompts section if available
        if prompts_text:
            self.monitor_text.insert(tk.END, prompts_text)

        # Add output section if available
        if output_text:
            self.monitor_text.insert(tk.END, output_text)

        self.monitor_text.config(state='disabled')

    def send_message(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Selection Error", "Please select an agent to interact with.")
            return
        message = self.message_entry.get().strip()
        if not message:
            messagebox.showwarning("Input Error", "Please enter a message.")
            return

        # Check if Felix system is available
        if not self.main_app or not self.main_app.felix_system:
            messagebox.showerror("System Not Ready",
                                "Felix system not available. Please start the system first.")
            return

        item = selection[0]
        agent_id = self.tree.item(item, "tags")[0]

        self.thread_manager.start_thread(self._send_thread, args=(agent_id, message))

    def _send_thread(self, agent_id, message):
        try:
            # Send task through the unified Felix system
            if not self.main_app or not self.main_app.felix_system:
                self.after(0, lambda: messagebox.showerror("Error", "Felix system not available"))
                return

            result = self.main_app.felix_system.send_task_to_agent(agent_id, message)

            if result:
                response = (
                    f"[{agent_id}] Task processed\n"
                    f"Content: {result.get('content', 'No response')[:200]}...\n"
                    f"Confidence: {result.get('confidence', 0.0):.2f}"
                )
                self.after(0, lambda: self._append_monitor(response))
            else:
                self.after(0, lambda: self._append_monitor(f"Failed to send task to {agent_id}"))

        except Exception as e:
            logger.error(f"Error sending message to agent: {e}", exc_info=True)
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to send message to agent: {error_msg}"))

    def _append_monitor(self, text):
        self.monitor_text.config(state='normal')
        self.monitor_text.insert(tk.END, text + '\n')
        self.monitor_text.config(state='disabled')
        self.monitor_text.see(tk.END)

    def apply_theme(self):
        """Apply current theme to the agents frame widgets."""
        if self.theme_manager:
            self.theme_manager.apply_to_text_widget(self.monitor_text)