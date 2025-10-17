import tkinter as tk
from tkinter import ttk, messagebox
import time
from .utils import ThreadManager, logger
try:
    from ..agents import dynamic_spawning
except ImportError:
    dynamic_spawning = None
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
    def __init__(self, parent, thread_manager, main_app=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app  # Reference to main application for core system access
        self.agents = []  # Reference to main system's agent list
        self.agent_counter = 0
        self.recent_messages = []  # For dynamic spawning analysis
        self.polling_active = False

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

    def _enable_features(self):
        """Enable agent features when system is running."""
        if self.main_app and hasattr(self.main_app, 'dynamic_spawner') and hasattr(self.main_app, 'central_post'):
            self.spawn_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self._append_monitor("System ready - GUI connected to main Felix architecture")
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
        if not self.main_app:
            return

        # Try to get agents from main_app's agent_manager
        if hasattr(self.main_app, 'agent_manager') and self.main_app.agent_manager:
            self.agents = list(self.main_app.agent_manager.agents.values())
        elif hasattr(self.main_app, 'agents'):
            self.agents = self.main_app.agents
        else:
            # Fallback: keep local list if no main system agents available
            pass

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
                self.main_app.lm_client and
                self.main_app.lm_client.test_connection())

    def spawn_agent(self):
        agent_type = self.type_combo.get()
        if not agent_type:
            messagebox.showwarning("Input Error", "Please select an agent type.")
            return

        # Check if main system is available
        if not self.main_app or not hasattr(self.main_app, 'dynamic_spawner'):
            messagebox.showerror("System Not Ready",
                                "Main Felix system not available or DynamicSpawning not initialized.\n"
                                "Please start the main system first.")
            return

        # Get specialized parameters
        domain = self.domain_entry.get().strip() or "general"

        self.thread_manager.start_thread(self._spawn_thread, args=(agent_type, domain))

    def _spawn_thread(self, agent_type, domain):
        try:
            # Check if main system is available
            if not self.main_app or not hasattr(self.main_app, 'dynamic_spawner'):
                self.after(0, lambda: messagebox.showerror("Error", "Main Felix system not available"))
                return

            # Send spawn request to main system's DynamicSpawning
            spawn_request = {
                'type': agent_type.lower(),
                'domain': domain,
                'source': 'gui'
            }

            # Try to call analyze_and_spawn with the request
            if hasattr(self.main_app.dynamic_spawner, 'analyze_and_spawn'):
                # Get current agents from main system
                current_agents = []
                if hasattr(self.main_app, 'agent_manager') and self.main_app.agent_manager:
                    current_agents = list(self.main_app.agent_manager.agents.values())

                # Call dynamic spawning with request parameters
                new_agents = self.main_app.dynamic_spawner.analyze_and_spawn(
                    processed_messages=self.recent_messages,
                    current_agents=current_agents,
                    current_time=time.time(),
                    spawn_request=spawn_request
                )

                if new_agents:
                    logger.info(f"Dynamic spawning created {len(new_agents)} agents from GUI request")
                    self.after(0, lambda: self._append_monitor(f"Spawned {len(new_agents)} {agent_type} agent(s)"))
                else:
                    self.after(0, lambda: self._append_monitor(f"No agents spawned - system may be at capacity or conditions not met"))
            else:
                logger.warning("DynamicSpawning does not have analyze_and_spawn method")
                self.after(0, lambda: self._append_monitor("Spawn request sent but spawning system unavailable"))

        except Exception as e:
            logger.error(f"Error spawning agent: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to spawn agent: {error_msg}"))

    # Manual agent creation removed - GUI should not create agents directly
    # All spawning should go through main system's DynamicSpawning

    def _update_treeview(self):
        """Update treeview with current agent states."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Add agents from core system
        for agent in self.agents:
            try:
                # Get agent properties
                position_info = agent.get_position_info(0.1)  # Use current time
                depth_ratio = position_info.get("depth_ratio", 0.0)
                confidence = getattr(agent, 'confidence', 0.0)
                velocity = getattr(agent, 'velocity', 0.0)
                progress = getattr(agent, 'progress', 0.0)
                state = getattr(agent, 'state', 'active')

                # Format display values
                position_str = f"{depth_ratio:.2f}"
                confidence_str = f"{confidence:.2f}"
                velocity_str = f"{velocity:.2f}"
                progress_str = f"{progress:.1%}"

                # Insert into treeview
                self.tree.insert("", tk.END, values=(
                    agent.agent_type.capitalize(),
                    position_str,
                    state,
                    progress_str,
                    confidence_str,
                    velocity_str
                ), tags=(agent.agent_id,))

            except Exception as e:
                logger.warning(f"Error updating treeview for agent {agent.agent_id}: {e}")
                # Fallback display
                self.tree.insert("", tk.END, values=(
                    getattr(agent, 'agent_type', 'unknown'),
                    "0.00",
                    "error",
                    "0.0%",
                    "0.00",
                    "0.00"
                ), tags=(agent.agent_id,))

    def on_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        agent_id = self.tree.item(item, "tags")[0]

        # Find agent in list
        agent = None
        for a in self.agents:
            if a.agent_id == agent_id:
                agent = a
                break

        if not agent:
            return

        # Get detailed agent information
        try:
            position_info = agent.get_position_info(0.1)
            details = f"ID: {agent.agent_id}\n"
            details += f"Type: {agent.agent_type}\n"
            details += f"Position: Depth {position_info.get('depth_ratio', 0.0):.2f}\n"
            details += f"State: {getattr(agent, 'state', 'active')}\n"
            details += f"Confidence: {getattr(agent, 'confidence', 0.0):.2f}\n"
            details += f"Progress: {getattr(agent, 'progress', 0.0):.1%}\n"
            details += f"Velocity: {getattr(agent, 'velocity', 0.0):.2f}\n"

            # Add specialized information
            if hasattr(agent, 'research_domain'):
                details += f"Research Domain: {agent.research_domain}\n"
            elif hasattr(agent, 'analysis_type'):
                details += f"Analysis Type: {agent.analysis_type}\n"
            elif hasattr(agent, 'output_format'):
                details += f"Output Format: {agent.output_format}\n"
            elif hasattr(agent, 'review_focus'):
                details += f"Review Focus: {agent.review_focus}\n"

        except Exception as e:
            details = f"ID: {agent.agent_id}\nType: {agent.agent_type}\nError getting details: {e}"

        self.monitor_text.config(state='normal')
        self.monitor_text.delete(1.0, tk.END)
        self.monitor_text.insert(tk.END, details)
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

        # Check if main system is available
        if not self.main_app or not hasattr(self.main_app, 'central_post'):
            messagebox.showerror("System Not Ready",
                                "Main Felix system not available or CentralPost not initialized.")
            return

        item = selection[0]
        agent_id = self.tree.item(item, "tags")[0]

        # Find agent in main system's agent list
        agent = None
        if hasattr(self.main_app, 'agent_manager') and self.main_app.agent_manager:
            agent = self.main_app.agent_manager.agents.get(agent_id)
        elif hasattr(self.main_app, 'agents'):
            for a in self.main_app.agents:
                if a.agent_id == agent_id:
                    agent = a
                    break

        if not agent:
            messagebox.showerror("Agent Not Found", f"Agent {agent_id} not found in main system.")
            return

        self.thread_manager.start_thread(self._send_thread, args=(agent, message))

    def _send_thread(self, agent, message):
        try:
            # Send task to main system's task processor
            if hasattr(self.main_app, 'task_processor') and self.main_app.task_processor:
                # Create task and send to main system's processor
                task_data = {
                    'agent_id': agent.agent_id,
                    'task_type': 'message',
                    'content': message,
                    'source': 'gui'
                }

                # Queue task through main system's task processor
                result = self.main_app.task_processor.process_task(task_data)

                if result:
                    response = f"[{agent.agent_type.upper()}] Task processed: {result.get('content', 'No response')}"
                    if 'confidence' in result:
                        response += f" (confidence: {result['confidence']:.2f})"
                else:
                    response = f"Task sent to {agent.agent_id} but no immediate response"

            # Send message through central post for communication
            elif hasattr(self.main_app, 'central_post') and self.main_app.central_post:
                if Message and MessageType:
                    msg = Message(
                        sender_id="gui",
                        message_type=MessageType.TASK_REQUEST,
                        content={
                            "target_agent": agent.agent_id,
                            "task": message,
                            "source": "gui"
                        },
                        timestamp=time.time()
                    )
                    self.main_app.central_post.queue_message(msg)
                    self.recent_messages.append(msg)
                    response = f"Message queued for {agent.agent_id}"
                else:
                    response = f"Message sent to {agent.agent_id}: {message}"
            else:
                response = f"Message sent to {agent.agent_id}: {message}"

            self.after(0, lambda: self._append_monitor(response))
        except Exception as e:
            logger.error(f"Error sending message to agent: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to send message to agent: {error_msg}"))

    def _append_monitor(self, text):
        self.monitor_text.config(state='normal')
        self.monitor_text.insert(tk.END, text + '\n')
        self.monitor_text.config(state='disabled')
        self.monitor_text.see(tk.END)