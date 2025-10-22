import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import logging
from typing import Dict, Any

from .utils import logger

class SettingsFrame(ttk.Frame):
    """Settings tab for configuring Felix framework parameters."""

    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager
        self.config_file = "felix_gui_config.json"

        # Dictionary to store all setting widgets
        self.setting_widgets = {}

        # Create scrollable frame
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Bind canvas resize to update inner frame width
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Build settings UI
        self._build_settings_ui()

        # Pack scrollbar and canvas with proper expansion
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux scroll down

        # Load settings from file
        self.load_settings()

    def _build_settings_ui(self):
        """Build the complete settings interface."""
        current_row = 0

        # Title
        title_label = ttk.Label(self.scrollable_frame, text="Felix Framework Settings",
                               font=("TkDefaultFont", 14, "bold"))
        title_label.grid(row=current_row, column=0, columnspan=3, pady=10, sticky="w")
        current_row += 1

        # LM Studio Connection Section
        current_row = self._create_section(current_row, "LM Studio Connection")
        current_row = self._create_text_field(current_row, "lm_host", "Host:", "127.0.0.1")
        current_row = self._create_text_field(current_row, "lm_port", "Port:", "1234")
        current_row += 1

        # Helix Geometry Section
        current_row = self._create_section(current_row, "Helix Geometry")
        current_row = self._create_float_field(current_row, "helix_top_radius", "Top Radius:", 3.0,
                                                "Exploration breadth (wider = more diverse)")
        current_row = self._create_float_field(current_row, "helix_bottom_radius", "Bottom Radius:", 0.5,
                                                "Focus precision (narrower = more focused)")
        current_row = self._create_float_field(current_row, "helix_height", "Height:", 8.0,
                                                "Total progression depth")
        current_row = self._create_float_field(current_row, "helix_turns", "Turns:", 2.0,
                                                "Number of helical spirals")
        current_row += 1

        # Agent Spawning Section
        current_row = self._create_section(current_row, "Agent Configuration")
        current_row = self._create_int_field(current_row, "max_agents", "Max Agents:", 25,
                                              "Maximum team size (1-133)")
        current_row = self._create_int_field(current_row, "base_token_budget", "Base Token Budget:", 2500,
                                              "Base tokens per agent")
        current_row += 1

        # Dynamic Spawning Section
        current_row = self._create_section(current_row, "Dynamic Spawning (Advanced)")
        current_row = self._create_float_field(current_row, "confidence_threshold", "Confidence Threshold:", 0.8,
                                                "Trigger for dynamic spawning (0.0-1.0)")
        current_row = self._create_float_field(current_row, "volatility_threshold", "Volatility Threshold:", 0.15,
                                                "High volatility triggers stabilizing agents")
        current_row = self._create_float_field(current_row, "time_window_minutes", "Time Window (min):", 5.0,
                                                "Window for trend analysis")
        current_row = self._create_int_field(current_row, "token_budget_limit", "Token Budget Limit:", 10000,
                                              "Total tokens across all agents")
        current_row += 1

        # Memory Settings Section
        current_row = self._create_section(current_row, "Memory Configuration")
        current_row = self._create_text_field(current_row, "memory_db_path", "Memory DB Path:", "felix_memory.db")
        current_row = self._create_text_field(current_row, "knowledge_db_path", "Knowledge DB Path:", "felix_knowledge.db")
        current_row = self._create_int_field(current_row, "compression_target_length", "Compression Target Length:", 100,
                                              "Target length for compressed context")
        current_row = self._create_float_field(current_row, "compression_ratio", "Compression Ratio:", 0.3,
                                                "Target reduction ratio (0.0-1.0)")
        current_row = self._create_dropdown_field(current_row, "compression_strategy", "Compression Strategy:",
                                                   ["abstractive", "hierarchical"], "abstractive")
        current_row += 1

        # Feature Toggles Section
        current_row = self._create_section(current_row, "Feature Toggles")
        current_row = self._create_checkbox_field(current_row, "enable_metrics", "Enable Metrics", True,
                                                   "Enable performance tracking")
        current_row = self._create_checkbox_field(current_row, "enable_memory", "Enable Memory", True,
                                                   "Enable persistent memory systems")
        current_row = self._create_checkbox_field(current_row, "enable_dynamic_spawning", "Enable Dynamic Spawning", True,
                                                   "Enable confidence-based agent spawning")
        current_row = self._create_checkbox_field(current_row, "enable_compression", "Enable Compression", True,
                                                   "Enable context compression")
        current_row = self._create_checkbox_field(current_row, "enable_spoke_topology", "Enable Spoke Topology", True,
                                                   "Use O(N) hub-spoke communication")
        current_row = self._create_checkbox_field(current_row, "verbose_llm_logging", "Verbose LLM Logging", True,
                                                   "Log detailed LLM requests/responses")
        current_row = self._create_checkbox_field(current_row, "enable_streaming", "Enable Streaming", True,
                                                   "Enable incremental token streaming for real-time agent communication")
        current_row += 1

        # Appearance Section
        current_row = self._create_section(current_row, "Appearance")
        current_row = self._create_checkbox_field(current_row, "dark_mode", "Dark Mode", False,
                                                   "Use dark color scheme for the GUI")
        current_row += 1

        # Action Buttons
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.grid(row=current_row, column=0, columnspan=3, pady=20)

        self.save_button = ttk.Button(button_frame, text="Save Settings", command=self.save_settings)
        self.save_button.pack(side="left", padx=5)

        self.reset_button = ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_to_defaults)
        self.reset_button.pack(side="left", padx=5)

        self.load_button = ttk.Button(button_frame, text="Load from File", command=self.load_from_file)
        self.load_button.pack(side="left", padx=5)

        # Status label
        self.status_label = ttk.Label(self.scrollable_frame, text="", foreground="blue")
        self.status_label.grid(row=current_row + 1, column=0, columnspan=3, pady=5)

    def _create_section(self, row: int, title: str) -> int:
        """Create a section header."""
        separator = ttk.Separator(self.scrollable_frame, orient="horizontal")
        separator.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 5))

        label = ttk.Label(self.scrollable_frame, text=title, font=("TkDefaultFont", 10, "bold"))
        label.grid(row=row + 1, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        return row + 2

    def _create_text_field(self, row: int, key: str, label: str, default: str,
                          tooltip: str = "") -> int:
        """Create a text input field."""
        label_widget = ttk.Label(self.scrollable_frame, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(20, 5), pady=2)

        entry = ttk.Entry(self.scrollable_frame, width=30)
        entry.insert(0, str(default))
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)

        if tooltip:
            tooltip_label = ttk.Label(self.scrollable_frame, text=f"({tooltip})",
                                     foreground="gray", font=("TkDefaultFont", 8))
            tooltip_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)

        self.setting_widgets[key] = entry
        return row + 1

    def _create_int_field(self, row: int, key: str, label: str, default: int,
                         tooltip: str = "") -> int:
        """Create an integer input field."""
        return self._create_text_field(row, key, label, str(default), tooltip)

    def _create_float_field(self, row: int, key: str, label: str, default: float,
                           tooltip: str = "") -> int:
        """Create a float input field."""
        return self._create_text_field(row, key, label, str(default), tooltip)

    def _create_checkbox_field(self, row: int, key: str, label: str, default: bool,
                              tooltip: str = "") -> int:
        """Create a checkbox field."""
        var = tk.BooleanVar(value=default)
        checkbox = ttk.Checkbutton(self.scrollable_frame, text=label, variable=var)
        checkbox.grid(row=row, column=0, columnspan=2, sticky="w", padx=(20, 5), pady=2)

        if tooltip:
            tooltip_label = ttk.Label(self.scrollable_frame, text=f"({tooltip})",
                                     foreground="gray", font=("TkDefaultFont", 8))
            tooltip_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)

        self.setting_widgets[key] = var
        return row + 1

    def _create_dropdown_field(self, row: int, key: str, label: str, options: list,
                              default: str, tooltip: str = "") -> int:
        """Create a dropdown field."""
        label_widget = ttk.Label(self.scrollable_frame, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(20, 5), pady=2)

        var = tk.StringVar(value=default)
        dropdown = ttk.Combobox(self.scrollable_frame, textvariable=var, values=options,
                               state="readonly", width=27)
        dropdown.grid(row=row, column=1, sticky="w", padx=5, pady=2)

        if tooltip:
            tooltip_label = ttk.Label(self.scrollable_frame, text=f"({tooltip})",
                                     foreground="gray", font=("TkDefaultFont", 8))
            tooltip_label.grid(row=row, column=2, sticky="w", padx=5, pady=2)

        self.setting_widgets[key] = var
        return row + 1

    def get_settings_dict(self) -> Dict[str, Any]:
        """Get current settings as a dictionary."""
        settings = {}

        for key, widget in self.setting_widgets.items():
            if isinstance(widget, tk.BooleanVar) or isinstance(widget, tk.StringVar):
                settings[key] = widget.get()
            elif isinstance(widget, ttk.Entry):
                settings[key] = widget.get()

        return settings

    def set_settings_dict(self, settings: Dict[str, Any]):
        """Set settings from a dictionary."""
        for key, value in settings.items():
            if key in self.setting_widgets:
                widget = self.setting_widgets[key]

                if isinstance(widget, tk.BooleanVar):
                    widget.set(bool(value))
                elif isinstance(widget, tk.StringVar):
                    widget.set(str(value))
                elif isinstance(widget, ttk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))

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
            max_agents = int(settings["max_agents"])
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
            port = int(settings["lm_port"])
            if port < 1 or port > 65535:
                return False, "Port must be between 1 and 65535"

            return True, ""

        except ValueError as e:
            return False, f"Invalid numeric value: {e}"

    def save_settings(self):
        """Save settings to config file."""
        # Check if system is running
        if self.main_app and self.main_app.system_running:
            messagebox.showwarning("System Running",
                                 "Cannot save settings while Felix system is running. "
                                 "Please stop the system first.")
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
                self.main_app.lm_host = typed_settings["lm_host"]
                self.main_app.lm_port = typed_settings["lm_port"]

            # Apply dark mode if theme manager is available
            if self.theme_manager and "dark_mode" in typed_settings:
                theme_name = "dark" if typed_settings["dark_mode"] else "light"
                self.theme_manager.set_theme(theme_name)

            self.status_label.config(text="Settings saved successfully!", foreground="green")
            logger.info(f"Settings saved to {self.config_file}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings:\n{e}")
            logger.error(f"Failed to save settings: {e}")

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

                self.status_label.config(text="Settings loaded from file", foreground="blue")
                logger.info(f"Settings loaded from {self.config_file}")
            else:
                logger.info(f"No config file found, using defaults")

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load settings:\n{e}")
            logger.error(f"Failed to load settings: {e}")

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
                self.status_label.config(text=f"Settings loaded from {os.path.basename(filename)}",
                                       foreground="blue")
                logger.info(f"Settings loaded from {filename}")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load settings:\n{e}")
                logger.error(f"Failed to load settings from {filename}: {e}")

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        if messagebox.askyesno("Reset Settings",
                              "Are you sure you want to reset all settings to defaults?"):
            defaults = {
                "lm_host": "127.0.0.1",
                "lm_port": "1234",
                "helix_top_radius": "3.0",
                "helix_bottom_radius": "0.5",
                "helix_height": "8.0",
                "helix_turns": "2.0",
                "max_agents": "25",
                "base_token_budget": "2500",
                "confidence_threshold": "0.8",
                "volatility_threshold": "0.15",
                "time_window_minutes": "5.0",
                "token_budget_limit": "10000",
                "memory_db_path": "felix_memory.db",
                "knowledge_db_path": "felix_knowledge.db",
                "compression_target_length": "100",
                "compression_ratio": "0.3",
                "compression_strategy": "abstractive",
                "enable_metrics": True,
                "enable_memory": True,
                "enable_dynamic_spawning": True,
                "enable_compression": True,
                "enable_spoke_topology": True,
                "verbose_llm_logging": True,
                "enable_streaming": True,
                "dark_mode": False
            }

            self.set_settings_dict(defaults)
            self.status_label.config(text="Settings reset to defaults", foreground="blue")
            logger.info("Settings reset to defaults")

    def _convert_settings_types(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string settings to appropriate types."""
        typed = {}

        # Integer fields
        int_fields = ["lm_port", "max_agents", "base_token_budget",
                     "token_budget_limit", "compression_target_length"]

        # Float fields
        float_fields = ["helix_top_radius", "helix_bottom_radius", "helix_height",
                       "helix_turns", "confidence_threshold", "volatility_threshold",
                       "time_window_minutes", "compression_ratio"]

        for key, value in settings.items():
            if key in int_fields:
                typed[key] = int(value)
            elif key in float_fields:
                typed[key] = float(value)
            else:
                typed[key] = value

        return typed

    def _enable_features(self):
        """Enable settings editing when system is running (actually disable for safety)."""
        # Disable editing when system is running
        self.save_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

        for widget in self.setting_widgets.values():
            if isinstance(widget, ttk.Entry):
                widget.config(state=tk.DISABLED)
            elif isinstance(widget, ttk.Combobox):
                widget.config(state=tk.DISABLED)
            elif isinstance(widget, ttk.Checkbutton):
                widget.config(state=tk.DISABLED)

    def _disable_features(self):
        """Disable settings editing when system is not running (actually enable)."""
        # Enable editing when system is not running
        self.save_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)

        for widget in self.setting_widgets.values():
            if isinstance(widget, ttk.Entry):
                widget.config(state=tk.NORMAL)
            elif isinstance(widget, ttk.Combobox):
                widget.config(state="readonly")
            elif isinstance(widget, ttk.Checkbutton):
                widget.config(state=tk.NORMAL)

    def _on_canvas_configure(self, event):
        """Update the inner frame width when canvas is resized."""
        # Make the scrollable frame match the canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        if event.num == 5 or event.delta < 0:
            # Scroll down
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0:
            # Scroll up
            self.canvas.yview_scroll(-1, "units")

    def apply_theme(self):
        """Apply current theme to settings widgets."""
        if self.theme_manager:
            theme = self.theme_manager.get_current_theme()
            # Apply theme to canvas
            self.canvas.configure(bg=theme["bg_primary"])
