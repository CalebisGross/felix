"""
Agents Tab for Felix GUI (CustomTkinter Edition)

The Agents tab provides:
- Agent spawning controls (type, domain/focus)
- Active agent list with TreeView
- Agent selection and detailed information
- Message sending to selected agents
- Real-time polling of agent states
"""

import customtkinter as ctk
from typing import Optional, List
import textwrap
import logging

from ..utils import ThreadManager, logger
from ..theme_manager import get_theme_manager
from ..components.themed_treeview import ThemedTreeview
from ..components.resizable_separator import ResizableSeparator
from ..responsive import Breakpoint, BreakpointConfig
from .base_tab import ResponsiveTab
from ..styles import (
    BUTTON_SM, BUTTON_MD,
    FONT_SECTION, FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_LG,
    INPUT_MD, INPUT_LG
)


class AgentsTab(ResponsiveTab):
    """
    Agents tab with spawning controls, agent list, and monitoring.
    Uses responsive master-detail layout.
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize Agents tab.

        Args:
            master: Parent widget (typically CTkTabview)
            thread_manager: ThreadManager instance
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, thread_manager, main_app, **kwargs)

        self.theme_manager = get_theme_manager()
        self.agents = []  # Reference to main system's agent list
        self.agent_counter = 0
        self.polling_active = False
        self._updating_tree = False  # Flag to track programmatic treeview updates
        self._last_selected_agent_id = None  # Track last agent shown in monitor

        # Layout state
        self.detail_visible = True
        self.master_detail_ratio = 0.5  # 50-50 split

        self._setup_ui()

        # Start polling for agent updates
        self._start_polling()

    def _setup_ui(self):
        """Set up the agents tab UI with responsive master-detail layout."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=0, sticky="nsew")

        # Create master frame (agent list + controls)
        self.master_frame = ctk.CTkFrame(self.main_container)
        self.master_frame.grid_columnconfigure(0, weight=1)
        self.master_frame.grid_rowconfigure(2, weight=1)  # TreeView expands

        # Create detail frame (agent monitor + message controls)
        self.detail_frame = ctk.CTkFrame(self.main_container)
        self.detail_frame.grid_columnconfigure(0, weight=1)
        self.detail_frame.grid_rowconfigure(1, weight=1)  # Monitor expands

        # Build sections
        self._create_spawn_controls()
        self._create_agent_treeview()
        self._create_message_controls()
        self._create_monitor_section()

        # Separator for resizing
        self.separator = None

        # Initially disable features until system starts
        self._disable_features()

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Handle breakpoint changes for responsive master-detail layout."""
        # Clear existing layout
        for widget in self.main_container.winfo_children():
            widget.grid_forget()

        if breakpoint == Breakpoint.COMPACT:
            # COMPACT: Show master only (list), detail as popup
            self._layout_compact()
        else:
            # STANDARD/WIDE/ULTRAWIDE: Master-detail side-by-side
            self._layout_master_detail()

    def _layout_compact(self):
        """Layout for compact screens: list only, details in selection callback."""
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Show only master frame (list + controls)
        self.master_frame.grid(row=0, column=0, sticky="nsew", padx=SPACE_LG, pady=SPACE_LG)
        self.detail_visible = False

        # Hide separator
        if self.separator:
            self.separator.grid_forget()

    def _layout_master_detail(self):
        """Layout for standard/wide screens: master-detail side-by-side."""
        self.main_container.grid_columnconfigure(0, weight=0, minsize=300)
        self.main_container.grid_columnconfigure(1, weight=0)
        self.main_container.grid_columnconfigure(2, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Master frame on left
        self.master_frame.grid(row=0, column=0, sticky="nsew", padx=(SPACE_LG, SPACE_SM), pady=SPACE_LG)

        # Create separator if needed
        if not self.separator:
            self.separator = ResizableSeparator(
                self.main_container,
                orientation="vertical",
                on_drag_complete=self._on_separator_drag
            )
        self.separator.grid(row=0, column=1, sticky="ns", pady=SPACE_LG)

        # Detail frame on right
        self.detail_frame.grid(row=0, column=2, sticky="nsew", padx=(0, SPACE_LG), pady=SPACE_LG)
        self.detail_visible = True

    def _on_separator_drag(self, ratio: float):
        """Handle separator drag completion."""
        self.master_detail_ratio = ratio
        # Update column weights based on ratio
        # Simplified: keep fixed master, expanding detail

    def _create_spawn_controls(self):
        """Create agent spawning controls."""
        controls_frame = ctk.CTkFrame(self.master_frame, fg_color="transparent")
        controls_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))
        controls_frame.grid_columnconfigure(1, weight=1)

        # Agent Type label and dropdown
        type_label = ctk.CTkLabel(
            controls_frame,
            text="Agent Type:",
            font=ctk.CTkFont(size=FONT_BODY)
        )
        type_label.grid(row=0, column=0, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

        self.type_combo = ctk.CTkComboBox(
            controls_frame,
            values=["Research", "Analysis", "Critic", "System"],
            state="readonly",
            width=INPUT_MD
        )
        self.type_combo.grid(row=0, column=1, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)
        self.type_combo.set("Research")  # Default selection

        # Domain/Focus label and entry
        domain_label = ctk.CTkLabel(
            controls_frame,
            text="Domain/Focus:",
            font=ctk.CTkFont(size=FONT_BODY)
        )
        domain_label.grid(row=1, column=0, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS)

        self.domain_entry = ctk.CTkEntry(
            controls_frame,
            placeholder_text="general",
            width=INPUT_LG
        )
        self.domain_entry.grid(row=1, column=1, sticky="ew", padx=(0, SPACE_SM), pady=SPACE_XS)
        self.domain_entry.insert(0, "general")

        # Spawn button
        self.spawn_button = ctk.CTkButton(
            controls_frame,
            text="Spawn Agent",
            command=self.spawn_agent,
            width=BUTTON_MD[0],
            height=BUTTON_MD[1],
            fg_color=self.theme_manager.get_color("accent"),
            hover_color=self.theme_manager.get_color("accent_hover"),
            state="disabled"
        )
        self.spawn_button.grid(row=0, column=2, rowspan=2, padx=(SPACE_SM, 0), pady=SPACE_XS)

    def _create_agent_treeview(self):
        """Create the agent TreeView."""
        tree_frame = ctk.CTkFrame(self.master_frame)
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=SPACE_LG, pady=SPACE_SM)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(1, weight=1)

        # TreeView header
        tree_header = ctk.CTkFrame(tree_frame, fg_color="transparent")
        tree_header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(
            tree_header,
            text="Active Agents",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).pack(side="left")

        # TreeView
        self.agent_tree = ThemedTreeview(
            tree_frame,
            columns=["type", "position", "state", "progress", "confidence", "velocity"],
            headings=["Type", "Position", "State", "Progress", "Confidence", "Velocity"],
            widths=[80, 100, 80, 80, 80, 80],
            height=10
        )
        self.agent_tree.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        # Bind selection event
        self.agent_tree.bind_tree('<<TreeviewSelect>>', self.on_agent_select)

    def _create_message_controls(self):
        """Create message sending controls."""
        message_frame = ctk.CTkFrame(self.detail_frame, fg_color="transparent")
        message_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))
        message_frame.grid_columnconfigure(1, weight=1)

        # Message label
        message_label = ctk.CTkLabel(
            message_frame,
            text="Message:",
            font=ctk.CTkFont(size=FONT_BODY)
        )
        message_label.grid(row=0, column=0, sticky="w", padx=(0, SPACE_SM))

        # Message entry
        self.message_entry = ctk.CTkEntry(
            message_frame,
            placeholder_text="Enter task message for selected agent..."
        )
        self.message_entry.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_SM))

        # Send button
        self.send_button = ctk.CTkButton(
            message_frame,
            text="Send",
            command=self.send_message,
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            fg_color=self.theme_manager.get_color("accent"),
            hover_color=self.theme_manager.get_color("accent_hover"),
            state="disabled"
        )
        self.send_button.grid(row=0, column=2)

    def _create_monitor_section(self):
        """Create the monitor display section."""
        monitor_frame = ctk.CTkFrame(self.detail_frame)
        monitor_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_LG, pady=(SPACE_SM, SPACE_LG))
        monitor_frame.grid_columnconfigure(0, weight=1)
        monitor_frame.grid_rowconfigure(1, weight=1)

        # Monitor header
        monitor_header = ctk.CTkFrame(monitor_frame, fg_color="transparent")
        monitor_header.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))

        ctk.CTkLabel(
            monitor_header,
            text="Agent Monitor",
            font=ctk.CTkFont(size=FONT_SECTION, weight="bold")
        ).pack(side="left")

        # Monitor textbox
        self.monitor_text = ctk.CTkTextbox(
            monitor_frame,
            font=ctk.CTkFont(family="Courier", size=FONT_CAPTION),
            wrap="word"
        )
        self.monitor_text.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(SPACE_XS, SPACE_SM))

        # Make monitor read-only by default
        self.monitor_text.configure(state="disabled")

    def _enable_features(self):
        """Enable agent features when system is running."""
        if self.main_app and hasattr(self.main_app, 'felix_system') and self.main_app.felix_system:
            self.spawn_button.configure(state="normal")
            self.send_button.configure(state="normal")
            self._append_monitor("System ready - GUI fully integrated with Felix architecture")
            self._start_polling()
        else:
            self._append_monitor("Main system components not available - some features disabled")

    def _disable_features(self):
        """Disable agent features when system is not running."""
        self.spawn_button.configure(state="disabled")
        self.send_button.configure(state="disabled")
        # Monitor text is created during setup, so delay the message
        self.after(100, lambda: self._append_monitor(
            "System not ready - waiting for main Felix system initialization..."
        ))

    def _start_polling(self):
        """Start polling for agent updates."""
        if self.polling_active:
            return
        self.polling_active = True
        self._poll_updates()

    def _stop_polling(self):
        """Stop polling for updates."""
        self.polling_active = False

    def _is_visible(self) -> bool:
        """Check if this tab is currently visible."""
        try:
            if self.main_app and hasattr(self.main_app, 'tabview'):
                return self.main_app.tabview.get() == "Agents"
        except Exception:
            pass
        return False

    def _poll_updates(self):
        """Poll for agent updates every 1.5 seconds (only when visible)."""
        if not self.polling_active:
            return

        # Only do expensive updates when tab is visible
        if self._is_visible():
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

    def _get_current_time(self) -> float:
        """Get current simulation time from Felix system, or default to 0.0."""
        if self.main_app and self.main_app.felix_system:
            return self.main_app.felix_system._current_time
        return 0.0

    def spawn_agent(self):
        """Spawn a new agent based on selected type and domain."""
        agent_type = self.type_combo.get()
        if not agent_type:
            self._show_warning("Input Error", "Please select an agent type.")
            return

        # Check if Felix system is available
        if not self.main_app or not self.main_app.felix_system:
            self._show_error(
                "System Not Ready",
                "Felix system not available. Please start the system first."
            )
            return

        # Get specialized parameters
        domain = self.domain_entry.get().strip() or "general"

        self.thread_manager.start_thread(self._spawn_thread, args=(agent_type, domain))

    def _spawn_thread(self, agent_type, domain):
        """Background thread for spawning agent."""
        try:
            # Check if Felix system is available
            if not self.main_app or not self.main_app.felix_system:
                self.after(0, lambda: self._show_error("Error", "Felix system not available"))
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
            self.after(0, lambda: self._show_error("Error", f"Failed to spawn agent: {error_msg}"))

    def _update_treeview(self):
        """Update treeview with current agent states while preserving selection."""
        # Set flag to indicate programmatic update
        self._updating_tree = True

        # Remember current selection before clearing
        selected_items = self.agent_tree.selection()
        selected_agent_id = None
        if selected_items:
            try:
                selected_agent_id = self.agent_tree.item(selected_items[0], "tags")[0]
            except (IndexError, KeyError):
                selected_agent_id = None

        # Clear existing items
        self.agent_tree.clear()

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

                # Try to get position info using actual system time
                # Display as "r=X.XX (YY%)" showing both radius and depth
                position_str = "N/A"
                try:
                    if hasattr(agent, 'get_position_info'):
                        current_time = self._get_current_time()
                        position_info = agent.get_position_info(current_time)
                        radius = position_info.get("radius", 0.0)
                        depth_ratio = position_info.get("depth_ratio", 0.0)
                        # Show radius (physical position on helix) and depth percentage
                        position_str = f"r={radius:.2f} ({depth_ratio*100:.0f}%)"
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
                item_id = self.agent_tree.insert("", "end", values=(
                    agent_type.capitalize() if isinstance(agent_type, str) else str(agent_type),
                    position_str,
                    state,
                    progress_str,
                    confidence_str,
                    velocity_str
                ), tags=(agent_id,))

                # Restore selection if this was the selected agent
                if selected_agent_id and agent_id == selected_agent_id:
                    self.agent_tree.selection_set(item_id)
                    self.agent_tree.see(item_id)  # Scroll into view if needed

            except Exception as e:
                logger.warning(f"Error updating treeview for agent: {e}")
                # Fallback display with minimal info
                try:
                    agent_id = getattr(agent, 'agent_id', 'unknown')
                    agent_type = getattr(agent, 'agent_type', 'unknown')
                    self.agent_tree.insert("", "end", values=(
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

    def on_agent_select(self, event):
        """Handle agent selection in TreeView."""
        selection = self.agent_tree.selection()
        if not selection:
            return

        item = selection[0]
        agent_id = self.agent_tree.item(item, "tags")[0]

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

        # Position info using actual system time
        try:
            if hasattr(agent, 'get_position_info'):
                current_time = self._get_current_time()
                position_info = agent.get_position_info(current_time)
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
        self.monitor_text.configure(state='normal')
        self.monitor_text.delete("1.0", "end")

        # Add section header for agent details
        self.monitor_text.insert("end", "="*60 + "\n")
        self.monitor_text.insert("end", "AGENT DETAILS\n")
        self.monitor_text.insert("end", "="*60 + "\n")
        self.monitor_text.insert("end", '\n'.join(details))

        # Add prompts section if available
        if prompts_text:
            self.monitor_text.insert("end", prompts_text)

        # Add output section if available
        if output_text:
            self.monitor_text.insert("end", output_text)

        self.monitor_text.configure(state='disabled')

    def send_message(self):
        """Send a message/task to the selected agent."""
        selection = self.agent_tree.selection()
        if not selection:
            self._show_warning("Selection Error", "Please select an agent to interact with.")
            return

        message = self.message_entry.get().strip()
        if not message:
            self._show_warning("Input Error", "Please enter a message.")
            return

        # Check if Felix system is available
        if not self.main_app or not self.main_app.felix_system:
            self._show_error(
                "System Not Ready",
                "Felix system not available. Please start the system first."
            )
            return

        item = selection[0]
        agent_id = self.agent_tree.item(item, "tags")[0]

        self.thread_manager.start_thread(self._send_thread, args=(agent_id, message))

    def _send_thread(self, agent_id, message):
        """Background thread for sending message to agent."""
        try:
            # Send task through the unified Felix system
            if not self.main_app or not self.main_app.felix_system:
                self.after(0, lambda: self._show_error("Error", "Felix system not available"))
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
            self.after(0, lambda: self._show_error("Error", f"Failed to send message to agent: {error_msg}"))

    def _append_monitor(self, text: str):
        """Append text to monitor display."""
        self.monitor_text.configure(state='normal')
        self.monitor_text.insert("end", text + '\n')
        self.monitor_text.configure(state='disabled')
        self.monitor_text.see("end")

    def _show_warning(self, title: str, message: str):
        """Show a warning dialog (stub - CustomTkinter uses different approach)."""
        logger.warning(f"{title}: {message}")
        self._append_monitor(f"WARNING: {message}")

    def _show_error(self, title: str, message: str):
        """Show an error dialog (stub - CustomTkinter uses different approach)."""
        logger.error(f"{title}: {message}")
        self._append_monitor(f"ERROR: {message}")
