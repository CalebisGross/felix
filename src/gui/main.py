import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
from typing import List
from .utils import ThreadManager, DBHelper, logger
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame
from .settings import SettingsFrame
from .approvals import ApprovalsFrame
from .terminal import TerminalFrame
from .felix_system import FelixSystem, FelixConfig
from .themes import ThemeManager

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Felix GUI")
        self.geometry("800x1200")

        # Felix system manager (unified integration)
        self.felix_system = None
        self.system_running = False

        # Configuration file
        self.config_file = "felix_gui_config.json"
        self.app_config = self._load_config()

        # LM connection settings (loaded from config)
        self.lm_host = self.app_config.get('lm_host', '127.0.0.1')
        self.lm_port = self.app_config.get('lm_port', 1234)

        # Initialize theme manager
        self.theme_manager = ThemeManager(self)

        # Load and apply saved theme
        dark_mode = self.app_config.get('dark_mode', False)
        theme_name = "dark" if dark_mode else "light"
        self.theme_manager.set_theme(theme_name)

        # Menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ThreadManager
        self.thread_manager = ThreadManager(self)

        # DBHelper
        self.db_helper = DBHelper()

        # Dashboard tab
        self.dashboard_frame = DashboardFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.dashboard_frame, text="Dashboard")

        # Workflows tab
        self.workflows_frame = WorkflowsFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.workflows_frame, text="Workflows")

        # Memory tab
        self.memory_frame = MemoryFrame(self.notebook, self.thread_manager, self.db_helper, theme_manager=self.theme_manager)
        self.notebook.add(self.memory_frame, text="Memory")

        # Agents tab
        self.agents_frame = AgentsFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.agents_frame, text="Agents")

        # Approvals tab
        self.approvals_frame = ApprovalsFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.approvals_frame, text="Approvals")

        # Terminal tab
        self.terminal_frame = TerminalFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.terminal_frame, text="Terminal")

        # Prompts tab
        from src.gui.prompts import PromptsTab
        self.prompts_frame = PromptsTab(self.notebook, theme_manager=self.theme_manager)
        self.notebook.add(self.prompts_frame, text="Prompts")

        # Settings tab
        self.settings_frame = SettingsFrame(self.notebook, self.thread_manager, main_app=self, theme_manager=self.theme_manager)
        self.notebook.add(self.settings_frame, text="Settings")

        # Register theme change callback to update all frames
        self.theme_manager.register_callback(self._on_theme_changed)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_about(self):
        messagebox.showinfo("About", "Felix GUI - Version 1.0")

    def _on_theme_changed(self, theme):
        """Called when theme changes to update all frames."""
        # Each frame will handle its own theme application
        if hasattr(self.dashboard_frame, 'apply_theme'):
            self.dashboard_frame.apply_theme()
        if hasattr(self.workflows_frame, 'apply_theme'):
            self.workflows_frame.apply_theme()
        if hasattr(self.memory_frame, 'apply_theme'):
            self.memory_frame.apply_theme()
        if hasattr(self.agents_frame, 'apply_theme'):
            self.agents_frame.apply_theme()
        if hasattr(self.approvals_frame, 'apply_theme'):
            self.approvals_frame.apply_theme()
        if hasattr(self.terminal_frame, 'apply_theme'):
            self.terminal_frame.apply_theme()
        if hasattr(self.prompts_frame, 'apply_theme'):
            self.prompts_frame.apply_theme()
        if hasattr(self.settings_frame, 'apply_theme'):
            self.settings_frame.apply_theme()

    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                logger.info("No config file found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _parse_blocked_domains(self, domains_str: str) -> List[str]:
        """Parse blocked domains from newline-separated string."""
        if not domains_str:
            return ['wikipedia.org', 'reddit.com']  # Default

        # Split by newlines and filter empty lines
        domains = [d.strip() for d in domains_str.split('\n') if d.strip()]
        return domains if domains else ['wikipedia.org', 'reddit.com']

    def start_system(self):
        """Start the Felix system with full integration."""
        if not self.system_running:
            self.thread_manager.start_thread(self._start_system_thread)

    def _start_system_thread(self):
        """Thread function to start system components."""
        try:
            # Reload config from settings
            self.app_config = self._load_config()
            self.lm_host = self.app_config.get('lm_host', '127.0.0.1')
            self.lm_port = self.app_config.get('lm_port', 1234)

            # Create Felix configuration from saved settings
            config = FelixConfig(
                lm_host=self.lm_host,
                lm_port=self.lm_port,
                helix_top_radius=self.app_config.get('helix_top_radius', 3.0),
                helix_bottom_radius=self.app_config.get('helix_bottom_radius', 0.5),
                helix_height=self.app_config.get('helix_height', 8.0),
                helix_turns=self.app_config.get('helix_turns', 2.0),
                max_agents=self.app_config.get('max_agents', 25),
                base_token_budget=self.app_config.get('base_token_budget', 2500),
                memory_db_path=self.app_config.get('memory_db_path', 'felix_memory.db'),
                knowledge_db_path=self.app_config.get('knowledge_db_path', 'felix_knowledge.db'),
                compression_target_length=self.app_config.get('compression_target_length', 100),
                compression_ratio=self.app_config.get('compression_ratio', 0.3),
                compression_strategy=self.app_config.get('compression_strategy', 'abstractive'),
                enable_metrics=self.app_config.get('enable_metrics', True),
                enable_memory=self.app_config.get('enable_memory', True),
                enable_dynamic_spawning=self.app_config.get('enable_dynamic_spawning', True),
                enable_compression=self.app_config.get('enable_compression', True),
                enable_spoke_topology=self.app_config.get('enable_spoke_topology', True),
                verbose_llm_logging=self.app_config.get('verbose_llm_logging', True),
                enable_streaming=self.app_config.get('enable_streaming', True),
                streaming_batch_interval=self.app_config.get('streaming_batch_interval', 0.1),
                # Web search configuration
                web_search_enabled=self.app_config.get('web_search_enabled', False),
                web_search_provider=self.app_config.get('web_search_provider', 'duckduckgo'),
                web_search_max_results=int(self.app_config.get('web_search_max_results', 5)),
                web_search_max_queries=int(self.app_config.get('web_search_max_queries', 3)),
                searxng_url=self.app_config.get('searxng_url') or None,
                web_search_blocked_domains=self._parse_blocked_domains(
                    self.app_config.get('web_search_blocked_domains', 'wikipedia.org\nreddit.com')
                ),
                # Web search trigger configuration
                web_search_confidence_threshold=float(self.app_config.get('web_search_confidence_threshold', 0.7)),
                web_search_min_samples=int(self.app_config.get('web_search_min_samples', 1)),
                web_search_cooldown=float(self.app_config.get('web_search_cooldown', 10.0)),
                # Workflow early stopping configuration
                workflow_max_steps_simple=int(self.app_config.get('workflow_max_steps_simple', 5)),
                workflow_max_steps_medium=int(self.app_config.get('workflow_max_steps_medium', 10)),
                workflow_max_steps_complex=int(self.app_config.get('workflow_max_steps_complex', 20)),
                workflow_simple_threshold=float(self.app_config.get('workflow_simple_threshold', 0.75)),
                workflow_medium_threshold=float(self.app_config.get('workflow_medium_threshold', 0.50))
            )

            # Initialize unified Felix system
            self.felix_system = FelixSystem(config)

            # Start the system
            if self.felix_system.start():
                self.system_running = True
                self.status_var.set("System Running")
                logger.info("Felix system fully integrated and running")
                self._enable_all_features()
            else:
                self.system_running = False
                self.felix_system = None
                self.status_var.set("System Start Failed")
                self.after(0, lambda: messagebox.showerror(
                    "System Start Failed",
                    "Failed to start Felix system. Check logs for details.\n"
                    "Ensure LM Studio is running with a model loaded."
                ))

        except Exception as e:
            logger.error(f"Error starting Felix system: {e}", exc_info=True)
            self.system_running = False
            self.felix_system = None
            self.status_var.set("System Start Failed")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to start system: {e}"))

    def _enable_all_features(self):
        """Enable features in all frames when system is running."""
        # Validate system health before enabling
        if not self._validate_system_health():
            logger.warning("System health check failed, not enabling features")
            return

        # Enable agents frame features
        for child in self.notebook.winfo_children():
            if hasattr(child, '_enable_features') and callable(getattr(child, '_enable_features')):
                child._enable_features()

    def _validate_system_health(self) -> bool:
        """Validate Felix system is healthy and ready."""
        if not self.felix_system:
            return False
        if not self.felix_system.running:
            return False
        if not self.felix_system.lm_client:
            return False
        # Test LM client connection
        try:
            if not self.felix_system.lm_client.test_connection():
                logger.error("LM client connection test failed")
                return False
        except Exception as e:
            logger.error(f"LM client health check exception: {e}")
            return False
        return True

    def _disable_all_features(self):
        """Disable features in all frames when system is not running."""
        # Disable agents frame features
        for child in self.notebook.winfo_children():
            if hasattr(child, '_disable_features') and callable(getattr(child, '_disable_features')):
                child._disable_features()

    def stop_system(self):
        """Stop the Felix system."""
        if self.system_running:
            self._disable_all_features()
            self.thread_manager.start_thread(self._stop_system_thread)

    def _stop_system_thread(self):
        """Thread function to stop system components."""
        try:
            if self.felix_system:
                self.felix_system.stop()
                self.felix_system = None

            self.system_running = False
            self.status_var.set("System Stopped")
            logger.info("Felix system stopped")

        except Exception as e:
            logger.error(f"Error stopping system: {e}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to stop system: {e}"))

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()