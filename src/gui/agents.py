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
        self.main_app = main_app  # Reference to main application for LM client access
        self.agents = []  # List of active agents from core system
        self.agent_counter = 0
        self.dynamic_spawning = None
        self.agent_factory = None
        self.central_post = None
        self.recent_messages = []  # For dynamic spawning analysis

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

        # Initialize dynamic spawning and communication when system starts
        self._initialize_core_components()

        # Initially disable features
        self._disable_features()

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
        if self.main_app and self.main_app.system_running:
            self.spawn_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self._initialize_core_components()

    def _initialize_core_components(self):
        """Initialize dynamic spawning and communication components."""
        if not self.main_app or not self.main_app.system_running:
            return

        try:
            # Initialize central post if available
            if self.main_app.cp and CentralPost:
                self.central_post = self.main_app.cp
            elif CentralPost:
                self.central_post = CentralPost()
                logger.info("Initialized local central post for GUI")

            # Initialize dynamic spawning if available
            if dynamic_spawning and self.main_app.lm_client:
                from ..agents.dynamic_spawning import DynamicSpawning
                from ..communication.central_post import AgentFactory

                # Create agent factory
                self.agent_factory = AgentFactory(
                    helix=self.main_app.helix if hasattr(self.main_app, 'helix') else None,
                    llm_client=self.main_app.lm_client,
                    token_budget_manager=getattr(self.main_app, 'token_budget_manager', None)
                )

                # Initialize dynamic spawning
                self.dynamic_spawning = DynamicSpawning(
                    agent_factory=self.agent_factory,
                    confidence_threshold=0.7,
                    max_agents=15,
                    token_budget_limit=10000
                )
                logger.info("Initialized dynamic spawning system")

        except Exception as e:
            logger.warning(f"Failed to initialize core components: {e}")

    def _disable_features(self):
        """Disable agent features when system is not running."""
        self.spawn_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

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

        # Check if system is ready for real LLM agents
        if not self._is_system_ready():
            messagebox.showerror("System Not Ready",
                                "Cannot spawn real LLM agents. Please ensure:\n"
                                "1. Felix system is running (Start System button)\n"
                                "2. LM Studio is connected and has a model loaded")
            return

        # Get specialized parameters
        domain = self.domain_entry.get().strip() or "general"

        self.thread_manager.start_thread(self._spawn_thread, args=(agent_type, domain))

    def _spawn_thread(self, agent_type, domain):
        try:
            # Check if system is running and LM client is available
            if not self.main_app or not self.main_app.system_running or not self.main_app.lm_client:
                self.after(0, lambda: messagebox.showerror("Error", "System not running or LM Studio not connected"))
                return

            # Use dynamic spawning if available, otherwise create manually
            if self.dynamic_spawning and self.agent_factory:
                # Use dynamic spawning system
                current_time = 0.1  # GUI demo time
                new_agents = self.dynamic_spawning.analyze_and_spawn(
                    processed_messages=self.recent_messages,
                    current_agents=self.agents,
                    current_time=current_time
                )

                if new_agents:
                    for agent in new_agents:
                        self.agents.append(agent)
                        # Register with central post if available
                        if self.central_post:
                            try:
                                self.central_post.register_agent(agent)
                                logger.info(f"Agent {agent.agent_id} registered with central post")
                            except Exception as e:
                                logger.warning(f"Failed to register agent {agent.agent_id}: {e}")

                    logger.info(f"Dynamic spawning created {len(new_agents)} agents")
                    self.after(0, self._update_treeview)
                    self.after(0, lambda: self._append_monitor(f"Dynamic spawning created {len(new_agents)} agents"))
                else:
                    # Fallback: create specific agent type manually
                    self._create_manual_agent(agent_type, domain)
            else:
                # Manual agent creation
                self._create_manual_agent(agent_type, domain)

        except Exception as e:
            logger.error(f"Error spawning agent: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to spawn agent: {error_msg}"))

    def _create_manual_agent(self, agent_type, domain):
        """Create agent manually using specialized classes."""
        try:
            # Get helix from main app or create default
            helix = getattr(self.main_app, 'helix', None)
            if not helix:
                try:
                    from ..core.helix_geometry import HelixGeometry
                    helix = HelixGeometry(
                        top_radius=5.0, bottom_radius=1.0, height=10.0, turns=3
                    )
                except ImportError:
                    self.after(0, lambda: messagebox.showerror("Error", "Helix geometry module not available"))
                    return

            # Get token budget manager from main app
            token_budget_manager = getattr(self.main_app, 'token_budget_manager', None)

            # Create specialized agent
            agent_id = f"{agent_type.lower()}_{self.agent_counter:03d}"
            self.agent_counter += 1
            spawn_time = 0.1

            if agent_type == "Research" and ResearchAgent:
                agent = ResearchAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=self.main_app.lm_client,
                    research_domain=domain,
                    token_budget_manager=token_budget_manager
                )
            elif agent_type == "Analysis" and AnalysisAgent:
                agent = AnalysisAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=self.main_app.lm_client,
                    analysis_type=domain,
                    token_budget_manager=token_budget_manager
                )
            elif agent_type == "Synthesis" and SynthesisAgent:
                agent = SynthesisAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=self.main_app.lm_client,
                    output_format=domain,
                    token_budget_manager=token_budget_manager
                )
            elif agent_type == "Critic" and CriticAgent:
                agent = CriticAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=self.main_app.lm_client,
                    review_focus=domain,
                    token_budget_manager=token_budget_manager
                )
            else:
                # Fallback to basic LLMAgent
                from ..agents.llm_agent import LLMAgent
                agent = LLMAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=self.main_app.lm_client,
                    agent_type=agent_type.lower()
                )

            # Add to agents list
            self.agents.append(agent)

            # Register with central post if available
            if self.central_post:
                try:
                    self.central_post.register_agent(agent)
                    logger.info(f"Agent {agent_id} registered with central post")
                except Exception as e:
                    logger.warning(f"Failed to register agent {agent_id}: {e}")

            logger.info(f"Specialized {agent_type} agent spawned with id {agent_id}")
            self.after(0, self._update_treeview)
            self.after(0, lambda: self._append_monitor(f"Specialized {agent_type} agent spawned with id {agent_id}"))

        except Exception as e:
            logger.error(f"Error creating manual agent: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to create {agent_type} agent: {str(e)}"))

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

        # Check if system is ready for LLM processing
        if not self._is_system_ready():
            messagebox.showerror("System Not Ready",
                                "Cannot send message to LLM agent. Please ensure:\n"
                                "1. Felix system is running (Start System button)\n"
                                "2. LM Studio is connected and has a model loaded")
            return

        item = selection[0]
        agent_id = self.tree.item(item, "tags")[0]

        # Find agent in list
        agent = None
        for a in self.agents:
            if a.agent_id == agent_id:
                agent = a
                break

        if not agent or not hasattr(agent, 'process_task_with_llm'):
            messagebox.showerror("Invalid Agent",
                                f"Agent {agent_id} is not a real LLM agent.\n"
                                "Please spawn a new LLM agent to interact with LM Studio.")
            return

        self.thread_manager.start_thread(self._send_thread, args=(agent, message))

    def _send_thread(self, agent, message):
        try:
            # Check if this is a real LLM agent
            if hasattr(agent, 'process_task_with_llm'):
                # Use real LLM agent processing
                from ..agents.llm_agent import LLMTask

                # Create a task for the LLM agent
                task = LLMTask(
                    task_id=f"gui_message_{agent.agent_id}_{len(self.recent_messages)}",
                    description=message,
                    context=f"GUI message from agents tab for {agent.agent_id}"
                )

                # Get current time for position calculation
                current_time = 0.1  # Use a small time value for GUI demo

                # Use type-specific processing methods
                if agent.agent_type == "research" and hasattr(agent, 'process_research_task'):
                    result = agent.process_research_task(task, current_time)
                elif agent.agent_type == "analysis" and hasattr(agent, 'process_analysis_task'):
                    result = agent.process_analysis_task(task, current_time)
                elif agent.agent_type == "synthesis" and hasattr(agent, 'process_synthesis_task'):
                    result = agent.process_synthesis_task(task, current_time)
                elif agent.agent_type == "critic" and hasattr(agent, 'process_critic_task'):
                    result = agent.process_critic_task(task, current_time)
                else:
                    # Default LLM processing
                    result = agent.process_task_with_llm(task, current_time)

                # Format response with agent info
                response = f"[{agent.agent_type.upper()}] {result.content}"
                if hasattr(result, 'confidence') and result.confidence > 0:
                    response += f" (confidence: {result.confidence:.2f})"

                # Create message for communication system
                if self.central_post and Message and MessageType:
                    msg = Message(
                        sender_id=agent.agent_id,
                        message_type=MessageType.TASK_COMPLETE,
                        content={
                            "result": result.content,
                            "confidence": getattr(result, 'confidence', 0.0),
                            "agent_type": agent.agent_type,
                            "position_info": result.position_info if hasattr(result, 'position_info') else {}
                        },
                        timestamp=time.time()
                    )
                    self.central_post.queue_message(msg)
                    self.recent_messages.append(msg)

                    # Store result as knowledge if central post supports it
                    if hasattr(self.central_post, 'store_agent_result_as_knowledge'):
                        self.central_post.store_agent_result_as_knowledge(
                            agent_id=agent.agent_id,
                            content=result.content,
                            confidence=getattr(result, 'confidence', 0.0),
                            domain=agent.agent_type
                        )

            elif hasattr(agent, 'process_message'):
                # Fallback for any remaining simple agents
                response = agent.process_message(message)
            else:
                # Final fallback
                response = f"Message sent to {agent.agent_id}: {message}"

            self.after(0, lambda: self._append_monitor(response))
            self.after(0, self._update_treeview)  # Update display after processing
        except Exception as e:
            logger.error(f"Error sending message to LLM agent: {e}")
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to send message to LLM agent: {error_msg}"))

    def _append_monitor(self, text):
        self.monitor_text.config(state='normal')
        self.monitor_text.insert(tk.END, text + '\n')
        self.monitor_text.config(state='disabled')
        self.monitor_text.see(tk.END)