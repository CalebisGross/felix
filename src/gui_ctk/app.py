"""
Felix GUI Main Application (CustomTkinter Edition)

A modern, clean GUI for the Felix multi-agent AI framework.
"""

import customtkinter as ctk
import json
import os
import queue
from typing import List, Optional
import logging

from .utils import ThreadManager, DBHelper, logger, enable_mouse_scroll
from .theme_manager import ThemeManager, get_theme_manager
from .responsive import ResponsiveLayoutManager
from .tabs.dashboard import DashboardTab
from .tabs.workflows import WorkflowsTab
from .tabs.agents import AgentsTab
from .tabs.memory import MemoryTab
from .tabs.approvals import ApprovalsTab
from .tabs.terminal import TerminalTab
from .tabs.prompts import PromptsTab
from .tabs.settings import SettingsTab
from .tabs.learning import LearningTab
from .tabs.knowledge_brain import KnowledgeBrainTab

# Import Chat tab (with fallback)
try:
    from .tabs.chat import ChatTab
    CHAT_TAB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ChatTab not available: {e}")
    CHAT_TAB_AVAILABLE = False
    ChatTab = None

# Import Felix system (with fallback)
try:
    from src.gui.felix_system import FelixSystem, FelixConfig
    FELIX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FelixSystem not available: {e}")
    FELIX_AVAILABLE = False
    FelixSystem = None
    FelixConfig = None


class MessageDialog(ctk.CTkToplevel):
    """
    A proper message dialog for displaying information, warnings, and errors.

    CTkInputDialog is designed for user input, not message display.
    This class provides a proper modal dialog with title, message, and OK button.
    """

    def __init__(self, parent, title: str, message: str, dialog_type: str = "info"):
        """
        Initialize message dialog.

        Args:
            parent: Parent window
            title: Dialog title
            message: Message to display
            dialog_type: Type of dialog ("info", "warning", "error")
        """
        super().__init__(parent)

        self.title(title)
        self.geometry("500x200")
        self.resizable(False, False)

        # Make dialog modal
        self.transient(parent)
        self.after(100, self._delayed_grab)

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set icon color based on dialog type
        icon_colors = {
            "info": "#3B8ED0",      # Blue
            "warning": "#FFA500",   # Orange
            "error": "#D32F2F"      # Red
        }
        icon_color = icon_colors.get(dialog_type, "#3B8ED0")

        # Message frame
        message_frame = ctk.CTkFrame(self)
        message_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        message_frame.grid_columnconfigure(0, weight=1)
        message_frame.grid_rowconfigure(0, weight=1)

        # Message label with wrapping
        message_label = ctk.CTkLabel(
            message_frame,
            text=message,
            font=ctk.CTkFont(size=13),
            wraplength=450,
            justify="left",
            text_color=icon_color if dialog_type == "error" else None
        )
        message_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Button frame
        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        button_frame.grid_columnconfigure(0, weight=1)

        # OK button
        ok_button = ctk.CTkButton(
            button_frame,
            text="OK",
            width=100,
            command=self._on_ok,
            fg_color=icon_color if dialog_type == "error" else None
        )
        ok_button.grid(row=0, column=0, pady=5)

        # Center on parent window
        self._center_on_parent(parent)

        # Focus on OK button
        ok_button.focus()

        # Bind Enter key to OK
        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_ok())

    def _center_on_parent(self, parent):
        """Center dialog on parent window."""
        self.update_idletasks()

        # Get parent geometry
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()

        # Get dialog geometry
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()

        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2

        # Ensure dialog is on screen
        x = max(0, x)
        y = max(0, y)

        self.geometry(f"+{x}+{y}")

    def _on_ok(self):
        """Handle OK button click."""
        self.grab_release()
        self.destroy()

    def _delayed_grab(self):
        """Grab focus after window is rendered."""
        try:
            self.grab_set()
            self.focus_set()
        except Exception:
            pass  # Silently ignore if grab fails


class FelixApp(ctk.CTk):
    """
    Main Felix GUI application.

    Features:
    - Modern CustomTkinter interface
    - Tabbed navigation
    - Dark/light mode support
    - Integration with Felix multi-agent system
    """

    def __init__(self):
        """Initialize Felix GUI application."""
        super().__init__()

        # Window configuration
        self.title("Felix - Multi-Agent AI Framework")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Felix system manager
        self.felix_system: Optional[FelixSystem] = None
        self.system_running = False

        # Configuration
        self.config_file = "felix_gui_config.json"
        self.app_config = self._load_config()

        # LM connection settings
        self.lm_host = self.app_config.get('lm_host', '127.0.0.1')
        self.lm_port = self.app_config.get('lm_port', 1234)

        # Initialize theme manager
        self.theme_manager = ThemeManager(self)

        # Load saved theme preference
        dark_mode = self.app_config.get('dark_mode', True)
        self.theme_manager.set_mode("dark" if dark_mode else "light")

        # Register for theme changes to update UI colors
        self.theme_manager.register_callback(self._on_theme_change)

        # Initialize responsive layout manager
        self.layout_manager = ResponsiveLayoutManager(self, debounce_ms=100)

        # Thread manager
        self.thread_manager = ThreadManager(self)

        # DB helper
        self.db_helper = DBHelper()

        # Queue for thread-safe communication
        self.result_queue = queue.Queue()

        # Set up UI
        self._setup_ui()

        # Enable mouse wheel scrolling (required for Linux with CTk)
        enable_mouse_scroll(self)

        # Start polling result queue
        self._poll_results()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        logger.info("Felix GUI initialized")

    def _setup_ui(self):
        """Set up the main UI layout."""
        # Apply theme colors to root window
        colors = self.theme_manager.colors
        self.configure(fg_color=colors["bg_primary"])

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create main container with theme color
        self.main_container = ctk.CTkFrame(
            self,
            fg_color=colors["bg_primary"],
            corner_radius=0
        )
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Create tabview with custom styling
        self.tabview = ctk.CTkTabview(
            self.main_container,
            fg_color=colors["bg_secondary"],
            segmented_button_fg_color=colors["bg_tertiary"],
            segmented_button_selected_color=colors["accent"],
            segmented_button_selected_hover_color=colors["accent_hover"],
            segmented_button_unselected_color=colors["bg_tertiary"],
            segmented_button_unselected_hover_color=colors["bg_hover"],
            text_color=colors["fg_primary"],
            corner_radius=12,
            border_width=1,
            border_color=colors["border"]
        )
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 10))

        # Add tabs
        self._create_tabs()

        # Status bar
        self._create_status_bar()

    def _create_tabs(self):
        """Create all application tabs."""
        # Dashboard tab
        self.tabview.add("Dashboard")
        self.dashboard_tab = DashboardTab(
            self.tabview.tab("Dashboard"),
            self.thread_manager,
            main_app=self
        )
        self.dashboard_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.dashboard_tab)

        # Workflows tab
        self.tabview.add("Workflows")
        self.workflows_tab = WorkflowsTab(
            self.tabview.tab("Workflows"),
            self.thread_manager,
            main_app=self
        )
        self.workflows_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.workflows_tab)

        # Chat tab (LM Studio-style chat interface)
        if CHAT_TAB_AVAILABLE:
            self.tabview.add("Chat")
            self.chat_tab = ChatTab(
                self.tabview.tab("Chat"),
                self.thread_manager,
                main_app=self
            )
            self.chat_tab.pack(fill="both", expand=True)
            self._register_tab_with_layout_manager(self.chat_tab)
        else:
            self.chat_tab = None

        # Memory tab
        self.tabview.add("Memory")
        self.memory_tab = MemoryTab(
            self.tabview.tab("Memory"),
            self.thread_manager,
            self.db_helper,
            main_app=self
        )
        self.memory_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.memory_tab)

        # Agents tab
        self.tabview.add("Agents")
        self.agents_tab = AgentsTab(
            self.tabview.tab("Agents"),
            self.thread_manager,
            main_app=self
        )
        self.agents_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.agents_tab)

        # Approvals tab
        self.tabview.add("Approvals")
        self.approvals_tab = ApprovalsTab(
            self.tabview.tab("Approvals"),
            self.thread_manager,
            main_app=self
        )
        self.approvals_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.approvals_tab)

        # Terminal tab
        self.tabview.add("Terminal")
        self.terminal_tab = TerminalTab(
            self.tabview.tab("Terminal"),
            self.thread_manager,
            main_app=self
        )
        self.terminal_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.terminal_tab)

        # Prompts tab
        self.tabview.add("Prompts")
        self.prompts_tab = PromptsTab(
            self.tabview.tab("Prompts"),
            self.thread_manager,
            main_app=self
        )
        self.prompts_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.prompts_tab)

        # Learning tab
        self.tabview.add("Learning")
        self.learning_tab = LearningTab(
            self.tabview.tab("Learning"),
            self.thread_manager,
            main_app=self
        )
        self.learning_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.learning_tab)

        # Knowledge Brain tab
        self.tabview.add("Knowledge Brain")
        self.knowledge_brain_tab = KnowledgeBrainTab(
            self.tabview.tab("Knowledge Brain"),
            self.thread_manager,
            main_app=self
        )
        self.knowledge_brain_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.knowledge_brain_tab)

        # Settings tab
        self.tabview.add("Settings")
        self.settings_tab = SettingsTab(
            self.tabview.tab("Settings"),
            self.thread_manager,
            main_app=self
        )
        self.settings_tab.pack(fill="both", expand=True)
        self._register_tab_with_layout_manager(self.settings_tab)

    def _register_tab_with_layout_manager(self, tab):
        """
        Register a tab with the layout manager if it supports responsive layout.

        Args:
            tab: Tab instance to register
        """
        # Check if tab has set_layout_manager method (ResponsiveTab subclasses)
        if hasattr(tab, 'set_layout_manager'):
            tab.set_layout_manager(self.layout_manager)

    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        colors = self.theme_manager.colors

        self.status_bar = ctk.CTkFrame(
            self.main_container,
            height=36,
            fg_color=colors["bg_secondary"],
            corner_radius=8,
            border_width=1,
            border_color=colors["border"]
        )
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.status_bar.grid_columnconfigure(0, weight=1)

        # Status indicator dot
        self.status_indicator = ctk.CTkFrame(
            self.status_bar,
            width=8,
            height=8,
            corner_radius=4,
            fg_color=colors["success"]
        )
        self.status_indicator.grid(row=0, column=0, sticky="w", padx=(12, 6), pady=10)

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=colors["fg_primary"],
            anchor="w"
        )
        self.status_label.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=8)

        # Breakpoint indicator (shows current responsive mode)
        self.breakpoint_label = ctk.CTkLabel(
            self.status_bar,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=colors["fg_muted"],
            anchor="e"
        )
        self.breakpoint_label.grid(row=0, column=2, sticky="e", padx=10, pady=8)

        self.version_label = ctk.CTkLabel(
            self.status_bar,
            text="Felix v0.9.0 • Modern Edition",
            font=ctk.CTkFont(size=11),
            text_color=colors["fg_muted"],
            anchor="e"
        )
        self.version_label.grid(row=0, column=3, sticky="e", padx=(0, 12), pady=8)

        # Register for breakpoint changes to update the indicator
        self.layout_manager.register_callback(self._on_app_breakpoint_change)

    def _on_app_breakpoint_change(self, breakpoint, config):
        """Update breakpoint indicator in status bar."""
        breakpoint_names = {
            "compact": "📱 Compact",
            "standard": "💻 Standard",
            "wide": "🖥️ Wide",
            "ultrawide": "🖥️ Ultrawide"
        }
        self.breakpoint_label.configure(text=breakpoint_names.get(breakpoint.value, ""))

    def _on_theme_change(self, mode):
        """Update UI colors when theme changes."""
        colors = self.theme_manager.colors

        # Update root window
        self.configure(fg_color=colors["bg_primary"])

        # Update main container
        self.main_container.configure(fg_color=colors["bg_primary"])

        # Update tabview colors
        self.tabview.configure(
            fg_color=colors["bg_secondary"],
            segmented_button_fg_color=colors["bg_tertiary"],
            segmented_button_selected_color=colors["accent"],
            segmented_button_selected_hover_color=colors["accent_hover"],
            segmented_button_unselected_color=colors["bg_tertiary"],
            segmented_button_unselected_hover_color=colors["bg_hover"],
            text_color=colors["fg_primary"],
            border_color=colors["border"]
        )

        # Update status bar
        self.status_bar.configure(
            fg_color=colors["bg_secondary"],
            border_color=colors["border"]
        )
        self.status_label.configure(text_color=colors["fg_primary"])
        self.breakpoint_label.configure(text_color=colors["fg_muted"])
        self.version_label.configure(text_color=colors["fg_muted"])

    def _poll_results(self):
        """Poll the result queue for thread-safe GUI updates."""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                action = result.get('action')

                if action == 'set_status':
                    self.status_label.configure(text=result['status'])
                elif action == 'enable_features':
                    self._enable_all_features()
                elif action == 'show_error':
                    self._show_error(
                        result.get('title', 'Error'),
                        result['message']
                    )
                elif action == 'show_info':
                    self._show_info(
                        result.get('title', 'Info'),
                        result['message']
                    )

        except Exception as e:
            logger.error(f"Error in poll_results: {e}")

        # Schedule next poll
        self.after(100, self._poll_results)

    def _load_config(self) -> dict:
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

    def _save_config(self):
        """Save current configuration to file."""
        try:
            # Update config with current state
            self.app_config['dark_mode'] = self.theme_manager.is_dark_mode()
            self.app_config['lm_host'] = self.lm_host
            self.app_config['lm_port'] = self.lm_port

            with open(self.config_file, 'w') as f:
                json.dump(self.app_config, f, indent=2)

            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def _parse_blocked_domains(self, domains_str: str) -> List[str]:
        """Parse blocked domains from newline-separated string."""
        if not domains_str:
            return ['wikipedia.org', 'reddit.com']

        domains = [d.strip() for d in domains_str.split('\n') if d.strip()]
        return domains if domains else ['wikipedia.org', 'reddit.com']

    def _parse_watch_directories(self, dirs_input) -> List[str]:
        """Parse watch directories from config."""
        if isinstance(dirs_input, list):
            return dirs_input
        if isinstance(dirs_input, str):
            dirs = [d.strip() for d in dirs_input.split('\n') if d.strip()]
            return dirs if dirs else ['./knowledge_sources']
        return ['./knowledge_sources']

    def start_system(self):
        """Start the Felix system."""
        if not self.system_running:
            self.thread_manager.start_thread(self._start_system_thread)

    def _start_system_thread(self):
        """Thread function to start system components."""
        if not FELIX_AVAILABLE:
            self.result_queue.put({
                'action': 'show_error',
                'title': 'System Error',
                'message': 'Felix system modules not available. Check imports.'
            })
            return

        try:
            # Reload config
            self.app_config = self._load_config()
            self.lm_host = self.app_config.get('lm_host', '127.0.0.1')
            self.lm_port = self.app_config.get('lm_port', 1234)

            # Create Felix configuration
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
                # Context compression settings
                compression_target_length=int(self.app_config.get('compression_target_length', 100)),
                compression_ratio=float(self.app_config.get('compression_ratio', 0.3)),
                compression_strategy=self.app_config.get('compression_strategy', 'abstractive'),
                # Feature flags
                enable_metrics=self.app_config.get('enable_metrics', True),
                enable_memory=self.app_config.get('enable_memory', True),
                enable_dynamic_spawning=self.app_config.get('enable_dynamic_spawning', True),
                enable_compression=self.app_config.get('enable_compression', True),
                enable_spoke_topology=self.app_config.get('enable_spoke_topology', True),
                verbose_llm_logging=self.app_config.get('verbose_llm_logging', True),
                enable_streaming=self.app_config.get('enable_streaming', True),
                streaming_batch_interval=float(self.app_config.get('streaming_batch_interval', 0.1)),
                # Web search settings
                web_search_enabled=self.app_config.get('web_search_enabled', False),
                web_search_provider=self.app_config.get('web_search_provider', 'duckduckgo'),
                web_search_max_results=int(self.app_config.get('web_search_max_results', 5)),
                web_search_max_queries=int(self.app_config.get('web_search_max_queries', 3)),
                web_search_min_samples=int(self.app_config.get('web_search_min_samples', 1)),
                web_search_cooldown=float(self.app_config.get('web_search_cooldown', 10.0)),
                web_search_blocked_domains=self._parse_blocked_domains(
                    self.app_config.get('web_search_blocked_domains', 'wikipedia.org\nreddit.com')
                ),
                # Workflow settings
                workflow_max_steps_simple=int(self.app_config.get('workflow_max_steps_simple', 5)),
                workflow_max_steps_medium=int(self.app_config.get('workflow_max_steps_medium', 10)),
                workflow_max_steps_complex=int(self.app_config.get('workflow_max_steps_complex', 20)),
                workflow_simple_threshold=float(self.app_config.get('workflow_simple_threshold', 0.75)),
                workflow_medium_threshold=float(self.app_config.get('workflow_medium_threshold', 0.50)),
                # Knowledge Brain settings
                enable_knowledge_brain=self.app_config.get('enable_knowledge_brain', False),
                knowledge_watch_dirs=self._parse_watch_directories(
                    self.app_config.get('knowledge_watch_dirs')
                ),
                knowledge_embedding_mode=self.app_config.get('knowledge_embedding_mode', 'auto'),
                knowledge_auto_augment=self.app_config.get('knowledge_auto_augment', True),
                knowledge_daemon_enabled=self.app_config.get('knowledge_daemon_enabled', True),
                knowledge_refinement_interval=int(self.app_config.get('knowledge_refinement_interval', 3600)),
                knowledge_processing_threads=int(self.app_config.get('knowledge_processing_threads', 2)),
                knowledge_max_memory_mb=int(self.app_config.get('knowledge_max_memory_mb', 512)),
                knowledge_chunk_size=int(self.app_config.get('knowledge_chunk_size', 1000)),
                knowledge_chunk_overlap=int(self.app_config.get('knowledge_chunk_overlap', 200)),
            )

            # Initialize Felix system
            self.felix_system = FelixSystem(config)

            # Start the system
            if self.felix_system.start():
                self.system_running = True
                self.result_queue.put({'action': 'set_status', 'status': 'System Running'})
                logger.info("Felix system started successfully")
                self.result_queue.put({'action': 'enable_features'})
            else:
                self.system_running = False
                self.felix_system = None
                self.result_queue.put({'action': 'set_status', 'status': 'Start Failed'})
                self.result_queue.put({
                    'action': 'show_error',
                    'title': 'System Start Failed',
                    'message': "Failed to start Felix system. Check logs for details.\n"
                               "Ensure LM Studio is running with a model loaded."
                })

        except Exception as e:
            logger.error(f"Error starting Felix system: {e}", exc_info=True)
            self.system_running = False
            self.felix_system = None
            self.result_queue.put({'action': 'set_status', 'status': 'Start Failed'})
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to start system: {e}"
            })

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
            self.result_queue.put({'action': 'set_status', 'status': 'System Stopped'})
            logger.info("Felix system stopped")

        except Exception as e:
            logger.error(f"Error stopping system: {e}", exc_info=True)
            self.result_queue.put({
                'action': 'show_error',
                'message': f"Failed to stop system: {e}"
            })

    def _enable_all_features(self):
        """Enable features in all tabs when system is running."""
        # Validate system health
        if not self._validate_system_health():
            logger.warning("System health check failed")
            return

        # Enable all tabs
        tabs_to_enable = [
            'dashboard_tab', 'workflows_tab', 'agents_tab', 'memory_tab',
            'approvals_tab', 'terminal_tab', 'prompts_tab', 'settings_tab',
            'learning_tab', 'knowledge_brain_tab'
        ]

        for tab_name in tabs_to_enable:
            if hasattr(self, tab_name):
                tab = getattr(self, tab_name)
                if hasattr(tab, '_enable_features'):
                    tab._enable_features()

        logger.info("All features enabled")

    def _disable_all_features(self):
        """Disable features in all tabs when system is not running."""
        # Disable all tabs
        tabs_to_disable = [
            'dashboard_tab', 'workflows_tab', 'agents_tab', 'memory_tab',
            'approvals_tab', 'terminal_tab', 'prompts_tab', 'settings_tab',
            'learning_tab', 'knowledge_brain_tab'
        ]

        for tab_name in tabs_to_disable:
            if hasattr(self, tab_name):
                tab = getattr(self, tab_name)
                if hasattr(tab, '_disable_features'):
                    tab._disable_features()

        logger.info("All features disabled")

    def _validate_system_health(self) -> bool:
        """Validate Felix system is healthy."""
        if not self.felix_system:
            return False
        if not self.felix_system.running:
            return False
        if not self.felix_system.lm_client:
            return False

        try:
            if not self.felix_system.lm_client.test_connection():
                logger.error("LM client connection test failed")
                return False
        except Exception as e:
            logger.error(f"LM client health check exception: {e}")
            return False

        return True

    def save_watch_directories(self) -> bool:
        """
        Save current watch directories to persistent config.

        Called by Knowledge Brain control panel when directories are added/removed.
        """
        if not self.felix_system or not hasattr(self.felix_system, 'knowledge_daemon'):
            logger.error("Cannot save watch directories: Knowledge daemon not initialized")
            return False

        if not self.felix_system.knowledge_daemon:
            logger.error("Cannot save watch directories: Knowledge daemon is None")
            return False

        try:
            # Load current config
            current_config = self._load_config()

            # Get current watch directories from daemon
            watch_dirs = self.felix_system.knowledge_daemon.config.watch_directories

            # Update config with new watch directories
            current_config['knowledge_watch_dirs'] = watch_dirs

            # Also update our in-memory config
            self.app_config['knowledge_watch_dirs'] = watch_dirs

            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(current_config, f, indent=2)

            logger.info(f"Watch directories saved to {self.config_file}: {watch_dirs}")
            return True

        except Exception as e:
            logger.error(f"Failed to save watch directories: {e}")
            return False

    def start_knowledge_daemon(self) -> bool:
        """Start the knowledge daemon if Felix system is running."""
        if not self.felix_system or not self.system_running:
            logger.error("Cannot start knowledge daemon: Felix system not running")
            return False

        if not self.felix_system.knowledge_daemon:
            logger.error("Knowledge daemon not initialized")
            return False

        try:
            self.felix_system.knowledge_daemon.start()
            logger.info("Knowledge daemon started via GUI")
            return True
        except Exception as e:
            logger.error(f"Failed to start knowledge daemon: {e}")
            return False

    def stop_knowledge_daemon(self) -> bool:
        """Stop the knowledge daemon if running."""
        if not self.felix_system:
            return False

        if not self.felix_system.knowledge_daemon:
            return False

        try:
            self.felix_system.knowledge_daemon.stop()
            logger.info("Knowledge daemon stopped via GUI")
            return True
        except Exception as e:
            logger.error(f"Failed to stop knowledge daemon: {e}")
            return False

    def _show_error(self, title: str, message: str):
        """Show an error dialog."""
        logger.error(f"{title}: {message}")
        MessageDialog(self, title, message, dialog_type="error")

    def _show_info(self, title: str, message: str):
        """Show an info dialog."""
        logger.info(f"{title}: {message}")
        MessageDialog(self, title, message, dialog_type="info")

    def _show_warning(self, title: str, message: str):
        """Show a warning dialog."""
        logger.warning(f"{title}: {message}")
        MessageDialog(self, title, message, dialog_type="warning")

    def _on_close(self):
        """Handle window close event - fast, non-blocking shutdown."""
        # Save config first (quick operation)
        self._save_config()

        # Signal all polling loops to stop immediately
        self.thread_manager.shutdown()

        # Stop system in background - don't wait
        if self.system_running and self.felix_system:
            logger.info("Initiating background system shutdown...")

            def background_cleanup():
                try:
                    self.felix_system.stop()
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")

            # Run as daemon thread - don't join, let it run in background
            import threading
            threading.Thread(target=background_cleanup, daemon=True).start()

        # Destroy window immediately - don't wait for system
        logger.info("Closing GUI window")
        self.destroy()


def main():
    """Entry point for Felix GUI."""
    app = FelixApp()
    app.mainloop()


if __name__ == "__main__":
    main()
