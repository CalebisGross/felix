"""
Settings Tab for Felix GUI (CustomTkinter Edition)

Provides comprehensive configuration interface for all Felix settings:
- LM Studio connection
- Helix geometry parameters
- Agent configuration
- Dynamic spawning settings
- Memory configuration
- Web search settings
- Knowledge Brain settings
- Workflow early stopping
- Learning systems
- Feature toggles
- Appearance (dark/light mode)
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import os
import logging
from typing import Dict, Any, List, Union

from ..utils import ThreadManager, logger
from ..theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class SettingsTab(ctk.CTkFrame):
    """
    Settings tab for configuring Felix framework parameters.

    All settings are saved to felix_gui_config.json and automatically
    applied when the Felix system starts.
    """

    def __init__(self, master, thread_manager: ThreadManager, main_app=None, **kwargs):
        """
        Initialize settings tab.

        Args:
            master: Parent widget
            thread_manager: Thread manager for background work
            main_app: Reference to main application
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()
        self.config_file = "felix_gui_config.json"

        # Dictionary to store all setting widgets (key -> widget)
        self.setting_widgets: Dict[str, Union[ctk.CTkEntry, ctk.CTkComboBox, ctk.CTkSwitch, ctk.StringVar]] = {}

        # Setup UI
        self._setup_ui()

        # Load settings from file
        self.load_settings()

        logger.info("Settings tab initialized")

    def _setup_ui(self):
        """Setup the UI components."""

        # Main scrollable container
        main_container = ctk.CTkScrollableFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(
            main_container,
            text="Felix Framework Settings",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(anchor="w", pady=(0, 20))

        # Build all settings sections
        current_row = 0

        # LM Studio Connection
        self._create_section(main_container, "LM Studio Connection")
        self._create_text_field(main_container, "lm_host", "Host:", "127.0.0.1", "LM Studio server hostname or IP address")
        self._create_text_field(main_container, "lm_port", "Port:", "1234", "LM Studio server port")
        self._create_spacer(main_container)

        # Helix Geometry
        self._create_section(main_container, "Helix Geometry")
        self._create_float_slider(main_container, "helix_top_radius", "Top Radius:", 3.0, 0.5, 10.0, "Exploration breadth (wider = more diverse)")
        self._create_float_slider(main_container, "helix_bottom_radius", "Bottom Radius:", 0.5, 0.1, 5.0, "Focus precision (narrower = more focused)")
        self._create_float_slider(main_container, "helix_height", "Height:", 8.0, 1.0, 20.0, "Total progression depth")
        self._create_float_slider(main_container, "helix_turns", "Turns:", 2.0, 1.0, 5.0, "Number of helical spirals")
        self._create_spacer(main_container)

        # Agent Configuration
        self._create_section(main_container, "Agent Configuration")
        self._create_int_slider(main_container, "max_agents", "Max Agents:", 25, 1, 133, "Maximum team size")
        self._create_int_field(main_container, "base_token_budget", "Base Token Budget:", "20000", "Base tokens per agent")
        self._create_spacer(main_container)

        # Dynamic Spawning
        self._create_section(main_container, "Dynamic Spawning (Advanced)")
        self._create_float_slider(main_container, "confidence_threshold", "Confidence Threshold:", 0.8, 0.0, 1.0, "Trigger for dynamic spawning")
        self._create_float_slider(main_container, "volatility_threshold", "Volatility Threshold:", 0.15, 0.0, 1.0, "High volatility triggers stabilizing agents")
        self._create_float_slider(main_container, "time_window_minutes", "Time Window (min):", 5.0, 1.0, 30.0, "Window for trend analysis")
        self._create_int_field(main_container, "token_budget_limit", "Token Budget Limit:", "45000", "Max tokens per context")
        self._create_spacer(main_container)

        # Memory Configuration
        self._create_section(main_container, "Memory Configuration")
        self._create_text_field(main_container, "memory_db_path", "Memory DB Path:", "felix_memory.db", "Path to task memory database")
        self._create_text_field(main_container, "knowledge_db_path", "Knowledge DB Path:", "felix_knowledge.db", "Path to knowledge database")
        self._create_int_field(main_container, "compression_target_length", "Compression Target Length:", "100", "Target length for compressed context")
        self._create_float_slider(main_container, "compression_ratio", "Compression Ratio:", 0.3, 0.1, 1.0, "Target reduction ratio")
        self._create_dropdown_field(main_container, "compression_strategy", "Compression Strategy:",
                                    ["abstractive", "hierarchical"], "abstractive", "Context compression method")
        self._create_spacer(main_container)

        # Web Search Configuration
        self._create_section(main_container, "Web Search Configuration")
        self._create_switch_field(main_container, "web_search_enabled", "Enable Web Search", False,
                                 "Enable web search by CentralPost when confidence is low")
        self._create_dropdown_field(main_container, "web_search_provider", "Search Provider:",
                                    ["duckduckgo", "searxng"], "duckduckgo", "DuckDuckGo (free) or SearxNG (self-hosted)")
        self._create_int_field(main_container, "web_search_max_results", "Max Results per Query:", "5", "Maximum search results to fetch")
        self._create_int_field(main_container, "web_search_max_queries", "Max Queries per Search:", "3", "Maximum queries per search session")
        self._create_text_field(main_container, "searxng_url", "SearxNG URL (optional):", "", "URL for SearxNG instance")
        self._create_text_area_field(main_container, "web_search_blocked_domains", "Blocked Domains (one per line):",
                                     "wikipedia.org\nreddit.com", 4, "Search results from these domains will be filtered out")

        # Web Search Triggers
        self._create_float_slider(main_container, "web_search_confidence_threshold", "Search Confidence Threshold:", 0.7, 0.0, 1.0,
                                 "Trigger web search when avg confidence < this value")
        self._create_int_field(main_container, "web_search_min_samples", "Min Confidence Samples:", "1", "Minimum confidence scores before checking average")
        self._create_float_slider(main_container, "web_search_cooldown", "Search Cooldown (seconds):", 10.0, 0.0, 60.0,
                                 "Minimum time between web searches")
        self._create_spacer(main_container)

        # Workflow Early Stopping
        self._create_section(main_container, "Workflow Early Stopping (Adaptive Complexity)")
        self._create_int_field(main_container, "workflow_max_steps_simple", "Max Steps (Simple):", "5", "Maximum steps for simple tasks")
        self._create_int_field(main_container, "workflow_max_steps_medium", "Max Steps (Medium):", "10", "Maximum steps for medium tasks")
        self._create_int_field(main_container, "workflow_max_steps_complex", "Max Steps (Complex):", "20", "Maximum steps for complex tasks")
        self._create_float_slider(main_container, "workflow_simple_threshold", "Simple Threshold:", 0.75, 0.0, 1.0,
                                 "Confidence threshold for simple tasks")
        self._create_float_slider(main_container, "workflow_medium_threshold", "Medium Threshold:", 0.50, 0.0, 1.0,
                                 "Confidence threshold for medium tasks")
        self._create_spacer(main_container)

        # Learning Systems
        self._create_section(main_container, "Learning Systems")
        self._create_switch_field(main_container, "enable_learning", "Enable Learning", True,
                                 "Enable adaptive learning systems")
        self._create_switch_field(main_container, "learning_auto_apply", "Auto-Apply Recommendations", True,
                                 "Auto-apply high-confidence recommendations (≥95% success, ≥20 samples)")
        self._create_int_field(main_container, "learning_min_samples_patterns", "Min Samples (Patterns):", "10",
                              "Minimum samples before recommending patterns")
        self._create_int_field(main_container, "learning_min_samples_calibration", "Min Samples (Calibration):", "10",
                              "Minimum samples before calibrating agent confidence")
        self._create_int_field(main_container, "learning_min_samples_thresholds", "Min Samples (Thresholds):", "20",
                              "Minimum samples before learning optimal thresholds")
        self._create_spacer(main_container)

        # Knowledge Brain
        self._create_section(main_container, "Knowledge Brain (Autonomous Document Learning)")
        self._create_switch_field(main_container, "enable_knowledge_brain", "Enable Knowledge Brain", False,
                                 "Enable autonomous document ingestion and knowledge retrieval")
        self._create_switch_field(main_container, "knowledge_auto_augment", "Auto-Augment Workflows", True,
                                 "Automatically inject relevant knowledge into workflow context")
        self._create_switch_field(main_container, "knowledge_daemon_enabled", "Enable Daemon", True,
                                 "Run background daemon for continuous processing")
        self._create_text_field(main_container, "knowledge_watch_dirs", "Watch Directories (comma-separated):",
                               "./knowledge_sources", "Directories to monitor for new documents")
        self._create_dropdown_field(main_container, "knowledge_embedding_mode", "Embedding Mode:",
                                    ["auto", "lm_studio", "tfidf", "fts5"], "auto", "Embedding provider (auto=best available)")
        self._create_int_field(main_container, "knowledge_refinement_interval", "Refinement Interval (sec):", "3600",
                              "How often to discover new knowledge relationships")
        self._create_int_field(main_container, "knowledge_processing_threads", "Processing Threads:", "2",
                              "Number of concurrent document processing threads")
        self._create_int_field(main_container, "knowledge_chunk_size", "Chunk Size (chars):", "1000",
                              "Characters per document chunk for processing")
        self._create_spacer(main_container)

        # Feature Toggles
        self._create_section(main_container, "Feature Toggles")
        self._create_switch_field(main_container, "enable_metrics", "Enable Metrics", True, "Enable performance tracking")
        self._create_switch_field(main_container, "enable_memory", "Enable Memory", True, "Enable persistent memory systems")
        self._create_switch_field(main_container, "enable_dynamic_spawning", "Enable Dynamic Spawning", True,
                                 "Enable confidence-based agent spawning")
        self._create_switch_field(main_container, "enable_compression", "Enable Compression", True, "Enable context compression")
        self._create_switch_field(main_container, "enable_spoke_topology", "Enable Spoke Topology", True,
                                 "Use O(N) hub-spoke communication")
        self._create_switch_field(main_container, "verbose_llm_logging", "Verbose LLM Logging", True,
                                 "Log detailed LLM requests/responses")
        self._create_switch_field(main_container, "enable_streaming", "Enable Streaming", True,
                                 "Enable incremental token streaming")
        self._create_spacer(main_container)

        # Appearance
        self._create_section(main_container, "Appearance")
        self._create_switch_field(main_container, "dark_mode", "Dark Mode", True, "Use dark color scheme for the GUI")
        self._create_spacer(main_container)

        # Action Buttons
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        button_frame.pack(fill="x", pady=(20, 10))

        self.save_button = ctk.CTkButton(
            button_frame,
            text="Save Settings",
            command=self.save_settings,
            width=150,
            fg_color="#2fa572",
            hover_color="#25835e"
        )
        self.save_button.pack(side="left", padx=(0, 10))

        self.reset_button = ctk.CTkButton(
            button_frame,
            text="Reset to Defaults",
            command=self.reset_to_defaults,
            width=150,
            fg_color="#d97706",
            hover_color="#b45309"
        )
        self.reset_button.pack(side="left", padx=(0, 10))

        self.load_from_file_button = ctk.CTkButton(
            button_frame,
            text="Load from File",
            command=self.load_from_file,
            width=150
        )
        self.load_from_file_button.pack(side="left")

        # Test connection button
        self.test_connection_button = ctk.CTkButton(
            button_frame,
            text="Test LM Studio Connection",
            command=self.test_lm_connection,
            width=200
        )
        self.test_connection_button.pack(side="right")

        # Status label
        self.status_label = ctk.CTkLabel(
            main_container,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)

    def _create_section(self, parent, title: str):
        """Create a section header."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=(15, 10))

        # Separator line
        separator = ctk.CTkFrame(frame, height=2, fg_color=self.theme_manager.get_color("border"))
        separator.pack(fill="x", pady=(0, 10))

        # Section title
        label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w")

    def _create_spacer(self, parent, height: int = 10):
        """Create vertical spacing."""
        spacer = ctk.CTkFrame(parent, height=height, fg_color="transparent")
        spacer.pack()

    def _create_text_field(self, parent, key: str, label: str, default: str, tooltip: str = ""):
        """Create a text input field."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=3)

        # Label
        label_widget = ctk.CTkLabel(container, text=label, width=250, anchor="w")
        label_widget.pack(side="left", padx=(0, 10))

        # Entry
        entry = ctk.CTkEntry(container, width=300)
        entry.insert(0, str(default))
        entry.pack(side="left", padx=(0, 10))

        self.setting_widgets[key] = entry

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(side="left")

    def _create_int_field(self, parent, key: str, label: str, default: str, tooltip: str = ""):
        """Create an integer input field."""
        self._create_text_field(parent, key, label, default, tooltip)

    def _create_float_slider(self, parent, key: str, label: str, default: float, from_: float, to: float, tooltip: str = ""):
        """Create a float slider with value display."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=3)

        # Label
        label_widget = ctk.CTkLabel(container, text=label, width=250, anchor="w")
        label_widget.pack(side="left", padx=(0, 10))

        # Value display
        value_var = ctk.StringVar(value=f"{default:.2f}")
        value_label = ctk.CTkLabel(container, textvariable=value_var, width=60)
        value_label.pack(side="left", padx=(0, 10))

        # Slider
        slider = ctk.CTkSlider(
            container,
            from_=from_,
            to=to,
            width=300,
            command=lambda v: value_var.set(f"{float(v):.2f}")
        )
        slider.set(default)
        slider.pack(side="left", padx=(0, 10))

        # Store both slider and value_var for retrieval
        self.setting_widgets[key] = slider
        self.setting_widgets[f"{key}_display"] = value_var

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(side="left")

    def _create_int_slider(self, parent, key: str, label: str, default: int, from_: int, to: int, tooltip: str = ""):
        """Create an integer slider with value display."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=3)

        # Label
        label_widget = ctk.CTkLabel(container, text=label, width=250, anchor="w")
        label_widget.pack(side="left", padx=(0, 10))

        # Value display
        value_var = ctk.StringVar(value=str(default))
        value_label = ctk.CTkLabel(container, textvariable=value_var, width=60)
        value_label.pack(side="left", padx=(0, 10))

        # Slider
        slider = ctk.CTkSlider(
            container,
            from_=from_,
            to=to,
            number_of_steps=(to - from_),
            width=300,
            command=lambda v: value_var.set(str(int(float(v))))
        )
        slider.set(default)
        slider.pack(side="left", padx=(0, 10))

        # Store both slider and value_var for retrieval
        self.setting_widgets[key] = slider
        self.setting_widgets[f"{key}_display"] = value_var

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(side="left")

    def _create_switch_field(self, parent, key: str, label: str, default: bool, tooltip: str = ""):
        """Create a switch (toggle) field."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=3)

        # Switch
        switch = ctk.CTkSwitch(container, text=label, width=250)
        if default:
            switch.select()
        else:
            switch.deselect()
        switch.pack(side="left", padx=(0, 10))

        self.setting_widgets[key] = switch

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(side="left")

    def _create_dropdown_field(self, parent, key: str, label: str, options: List[str], default: str, tooltip: str = ""):
        """Create a dropdown (combobox) field."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=3)

        # Label
        label_widget = ctk.CTkLabel(container, text=label, width=250, anchor="w")
        label_widget.pack(side="left", padx=(0, 10))

        # Dropdown
        var = ctk.StringVar(value=default)
        dropdown = ctk.CTkComboBox(
            container,
            values=options,
            variable=var,
            width=300,
            state="readonly"
        )
        dropdown.pack(side="left", padx=(0, 10))

        self.setting_widgets[key] = var

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(side="left")

    def _create_text_area_field(self, parent, key: str, label: str, default_text: str, height: int = 4, tooltip: str = ""):
        """Create a multi-line text area field."""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=5)

        # Label above text area
        label_widget = ctk.CTkLabel(container, text=label, anchor="w")
        label_widget.pack(anchor="w", pady=(0, 5))

        # Text area
        text_area = ctk.CTkTextbox(container, height=height * 20, wrap="word")
        text_area.insert("1.0", default_text)
        text_area.pack(fill="x", padx=(0, 10))

        self.setting_widgets[key] = text_area

        # Tooltip
        if tooltip:
            tooltip_label = ctk.CTkLabel(
                container,
                text=f"({tooltip})",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            tooltip_label.pack(anchor="w", pady=(5, 0))

    def get_settings_dict(self) -> Dict[str, Any]:
        """Get current settings as a dictionary."""
        settings = {}

        for key, widget in self.setting_widgets.items():
            # Skip display variables (they're just for UI)
            if key.endswith("_display"):
                continue

            if isinstance(widget, ctk.CTkEntry):
                settings[key] = widget.get()
            elif isinstance(widget, ctk.CTkSwitch):
                settings[key] = widget.get() == 1
            elif isinstance(widget, ctk.StringVar):
                settings[key] = widget.get()
            elif isinstance(widget, ctk.CTkSlider):
                settings[key] = widget.get()
            elif isinstance(widget, ctk.CTkTextbox):
                settings[key] = widget.get("1.0", "end-1c")

        return settings

    def set_settings_dict(self, settings: Dict[str, Any]):
        """Set settings from a dictionary."""
        for key, value in settings.items():
            if key in self.setting_widgets:
                widget = self.setting_widgets[key]

                if isinstance(widget, ctk.CTkEntry):
                    widget.delete(0, "end")
                    widget.insert(0, str(value))
                elif isinstance(widget, ctk.CTkSwitch):
                    if bool(value):
                        widget.select()
                    else:
                        widget.deselect()
                elif isinstance(widget, ctk.StringVar):
                    widget.set(str(value))
                elif isinstance(widget, ctk.CTkSlider):
                    widget.set(float(value))
                    # Update display variable if it exists
                    display_key = f"{key}_display"
                    if display_key in self.setting_widgets:
                        display_var = self.setting_widgets[display_key]
                        if isinstance(display_var, ctk.StringVar):
                            # Check if it's an int or float slider
                            if key in ["max_agents"]:
                                display_var.set(str(int(float(value))))
                            else:
                                display_var.set(f"{float(value):.2f}")
                elif isinstance(widget, ctk.CTkTextbox):
                    widget.delete("1.0", "end")
                    widget.insert("1.0", str(value))

    def validate_settings(self) -> tuple[bool, str]:
        """Validate all settings. Returns (is_valid, error_message)."""
        try:
            settings = self.get_settings_dict()

            # Validate numeric ranges
            helix_top = float(settings["helix_top_radius"])
            helix_bottom = float(settings["helix_bottom_radius"])
            helix_height = float(settings["helix_height"])
            helix_turns = float(settings["helix_turns"])

            if helix_top <= helix_bottom:
                return False, "Top radius must be greater than bottom radius"
            if helix_height <= 0:
                return False, "Height must be positive"
            if helix_turns <= 0:
                return False, "Turns must be positive"

            # Validate max_agents
            max_agents = int(float(settings["max_agents"]))
            if max_agents < 1 or max_agents > 133:
                return False, "Max agents must be between 1 and 133"

            # Validate thresholds
            confidence = float(settings["confidence_threshold"])
            if not 0.0 <= confidence <= 1.0:
                return False, "Confidence threshold must be between 0.0 and 1.0"

            volatility = float(settings["volatility_threshold"])
            if not 0.0 <= volatility <= 1.0:
                return False, "Volatility threshold must be between 0.0 and 1.0"

            compression_ratio = float(settings["compression_ratio"])
            if not 0.0 <= compression_ratio <= 1.0:
                return False, "Compression ratio must be between 0.0 and 1.0"

            # Validate port
            port_str = settings.get("lm_port", "1234")
            if isinstance(port_str, (int, float)):
                port = int(port_str)
            else:
                port = int(port_str)
            if port < 1 or port > 65535:
                return False, "Port must be between 1 and 65535"

            # Validate web search thresholds
            if "web_search_confidence_threshold" in settings:
                ws_conf_threshold = float(settings["web_search_confidence_threshold"])
                if not 0.0 <= ws_conf_threshold <= 1.0:
                    return False, "Web search confidence threshold must be between 0.0 and 1.0"

            if "web_search_min_samples" in settings:
                ws_min_samples = int(settings["web_search_min_samples"])
                if ws_min_samples < 1:
                    return False, "Web search min samples must be at least 1"

            if "web_search_cooldown" in settings:
                ws_cooldown = float(settings["web_search_cooldown"])
                if ws_cooldown < 0:
                    return False, "Web search cooldown must be non-negative"

            # Validate workflow thresholds
            if "workflow_simple_threshold" in settings:
                wf_simple = float(settings["workflow_simple_threshold"])
                if not 0.0 <= wf_simple <= 1.0:
                    return False, "Workflow simple threshold must be between 0.0 and 1.0"

            if "workflow_medium_threshold" in settings:
                wf_medium = float(settings["workflow_medium_threshold"])
                if not 0.0 <= wf_medium <= 1.0:
                    return False, "Workflow medium threshold must be between 0.0 and 1.0"

            return True, ""

        except ValueError as e:
            return False, f"Invalid numeric value: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def save_settings(self):
        """Save settings to config file."""
        # Check if system is running
        if self.main_app and hasattr(self.main_app, 'system_running') and self.main_app.system_running:
            messagebox.showwarning(
                "System Running",
                "Cannot save settings while Felix system is running.\n"
                "Please stop the system first."
            )
            return

        # Validate settings
        is_valid, error_msg = self.validate_settings()
        if not is_valid:
            messagebox.showerror("Validation Error", f"Invalid settings:\n{error_msg}")
            return

        try:
            settings = self.get_settings_dict()

            # Convert string values to appropriate types
            typed_settings = self._convert_settings_types(settings)

            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(typed_settings, f, indent=2)

            # Update main app config
            if self.main_app:
                self.main_app.lm_host = typed_settings.get("lm_host", "127.0.0.1")
                self.main_app.lm_port = typed_settings.get("lm_port", 1234)

            # Apply dark mode if changed
            if "dark_mode" in typed_settings:
                new_mode = "dark" if typed_settings["dark_mode"] else "light"
                if self.theme_manager.mode != new_mode:
                    self.theme_manager.set_mode(new_mode)

            self.status_label.configure(text="✓ Settings saved successfully!", text_color="#2fa572")
            logger.info(f"Settings saved to {self.config_file}")

            # Auto-clear status after 3 seconds
            self.after(3000, lambda: self.status_label.configure(text=""))

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings:\n{e}")
            logger.error(f"Failed to save settings: {e}", exc_info=True)

    def load_settings(self):
        """Load settings from config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)

                self.set_settings_dict(settings)

                # Update main app config
                if self.main_app:
                    self.main_app.lm_host = settings.get("lm_host", "127.0.0.1")
                    self.main_app.lm_port = settings.get("lm_port", 1234)

                # Apply dark mode setting
                if "dark_mode" in settings:
                    mode = "dark" if settings["dark_mode"] else "light"
                    self.theme_manager.set_mode(mode)

                self.status_label.configure(text="Settings loaded from file", text_color=self.theme_manager.get_color("fg_secondary"))
                logger.info(f"Settings loaded from {self.config_file}")
            else:
                logger.info(f"No config file found, using defaults")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings:\n{e}")
            logger.error(f"Failed to load settings: {e}", exc_info=True)

    def load_from_file(self):
        """Load settings from a user-selected file."""
        filename = filedialog.askopenfilename(
            title="Load Settings",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)

                self.set_settings_dict(settings)
                self.status_label.configure(
                    text=f"Settings loaded from {os.path.basename(filename)}",
                    text_color=self.theme_manager.get_color("fg_secondary")
                )
                logger.info(f"Settings loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load settings:\n{e}")
                logger.error(f"Failed to load settings from {filename}: {e}", exc_info=True)

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        result = messagebox.askyesno(
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n"
            "This will overwrite your current configuration."
        )

        if not result:
            return

        defaults = {
            "lm_host": "127.0.0.1",
            "lm_port": "1234",
            "helix_top_radius": 3.0,
            "helix_bottom_radius": 0.5,
            "helix_height": 8.0,
            "helix_turns": 2.0,
            "max_agents": 25,
            "base_token_budget": "20000",
            "confidence_threshold": 0.8,
            "volatility_threshold": 0.15,
            "time_window_minutes": 5.0,
            "token_budget_limit": "45000",
            "memory_db_path": "felix_memory.db",
            "knowledge_db_path": "felix_knowledge.db",
            "compression_target_length": "100",
            "compression_ratio": 0.3,
            "compression_strategy": "abstractive",
            "web_search_enabled": False,
            "web_search_provider": "duckduckgo",
            "web_search_max_results": "5",
            "web_search_max_queries": "3",
            "searxng_url": "",
            "web_search_blocked_domains": "wikipedia.org\nreddit.com",
            "web_search_confidence_threshold": 0.7,
            "web_search_min_samples": "1",
            "web_search_cooldown": 10.0,
            "workflow_max_steps_simple": "5",
            "workflow_max_steps_medium": "10",
            "workflow_max_steps_complex": "20",
            "workflow_simple_threshold": 0.75,
            "workflow_medium_threshold": 0.50,
            "enable_learning": True,
            "learning_auto_apply": True,
            "learning_min_samples_patterns": "10",
            "learning_min_samples_calibration": "10",
            "learning_min_samples_thresholds": "20",
            "enable_knowledge_brain": False,
            "knowledge_auto_augment": True,
            "knowledge_daemon_enabled": True,
            "knowledge_watch_dirs": "./knowledge_sources",
            "knowledge_embedding_mode": "auto",
            "knowledge_refinement_interval": "3600",
            "knowledge_processing_threads": "2",
            "knowledge_chunk_size": "1000",
            "enable_metrics": True,
            "enable_memory": True,
            "enable_dynamic_spawning": True,
            "enable_compression": True,
            "enable_spoke_topology": True,
            "verbose_llm_logging": True,
            "enable_streaming": True,
            "dark_mode": True
        }

        self.set_settings_dict(defaults)
        self.status_label.configure(text="Settings reset to defaults", text_color=self.theme_manager.get_color("warning"))
        logger.info("Settings reset to defaults")

    def test_lm_connection(self):
        """Test connection to LM Studio."""
        settings = self.get_settings_dict()
        host = settings.get("lm_host", "127.0.0.1")
        port = settings.get("lm_port", "1234")

        if isinstance(port, str):
            port = int(port)

        self.status_label.configure(text="Testing LM Studio connection...", text_color=self.theme_manager.get_color("fg_secondary"))
        self.test_connection_button.configure(state="disabled")

        def test_thread():
            """Run connection test in background thread."""
            try:
                import httpx

                url = f"http://{host}:{port}/v1/models"
                response = httpx.get(url, timeout=5.0)

                if response.status_code == 200:
                    self.after(0, lambda: self._connection_test_success())
                else:
                    self.after(0, lambda: self._connection_test_failed(f"HTTP {response.status_code}"))

            except Exception as e:
                self.after(0, lambda: self._connection_test_failed(str(e)))

        self.thread_manager.start_thread(test_thread)

    def _connection_test_success(self):
        """Handle successful connection test."""
        self.status_label.configure(text="✓ LM Studio connection successful!", text_color="#2fa572")
        self.test_connection_button.configure(state="normal")
        logger.info("LM Studio connection test: SUCCESS")

        # Auto-clear after 3 seconds
        self.after(3000, lambda: self.status_label.configure(text=""))

    def _connection_test_failed(self, error: str):
        """Handle failed connection test."""
        self.status_label.configure(text=f"✗ Connection failed: {error}", text_color="#dc2626")
        self.test_connection_button.configure(state="normal")
        logger.warning(f"LM Studio connection test: FAILED - {error}")

        # Auto-clear after 5 seconds
        self.after(5000, lambda: self.status_label.configure(text=""))

    def _convert_settings_types(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string settings to appropriate types."""
        typed = {}

        # Integer fields
        int_fields = [
            "lm_port", "max_agents", "base_token_budget", "token_budget_limit",
            "compression_target_length", "web_search_max_results", "web_search_max_queries",
            "web_search_min_samples", "workflow_max_steps_simple", "workflow_max_steps_medium",
            "workflow_max_steps_complex", "learning_min_samples_patterns",
            "learning_min_samples_calibration", "learning_min_samples_thresholds",
            "knowledge_refinement_interval", "knowledge_processing_threads", "knowledge_chunk_size"
        ]

        # Float fields (these come from sliders, already floats)
        float_fields = [
            "helix_top_radius", "helix_bottom_radius", "helix_height", "helix_turns",
            "confidence_threshold", "volatility_threshold", "time_window_minutes",
            "compression_ratio", "web_search_confidence_threshold", "web_search_cooldown",
            "workflow_simple_threshold", "workflow_medium_threshold"
        ]

        for key, value in settings.items():
            if key in int_fields:
                typed[key] = int(float(value)) if isinstance(value, (int, float, str)) else int(value)
            elif key in float_fields:
                typed[key] = float(value)
            else:
                typed[key] = value

        return typed

    def _enable_features(self):
        """Enable settings editing when system is stopped."""
        # Actually, settings should be DISABLED when system is running
        # This is called when system STOPS
        self.save_button.configure(state="normal")
        self.reset_button.configure(state="normal")
        self.load_from_file_button.configure(state="normal")
        logger.debug("Settings editing enabled (system stopped)")

    def _disable_features(self):
        """Disable settings editing when system is running."""
        # This is called when system STARTS
        self.save_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        self.load_from_file_button.configure(state="disabled")
        logger.debug("Settings editing disabled (system running)")
