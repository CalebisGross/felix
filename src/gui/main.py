import tkinter as tk
from tkinter import ttk, messagebox
from .utils import ThreadManager, DBHelper, logger
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame
from .felix_system import FelixSystem, FelixConfig

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Felix GUI")
        self.geometry("800x600")

        # Felix system manager (unified integration)
        self.felix_system = None
        self.system_running = False

        # Legacy compatibility properties
        self.lm_host = '127.0.0.1'
        self.lm_port = 1234

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
        dashboard_frame = DashboardFrame(self.notebook, self.thread_manager, main_app=self)
        self.notebook.add(dashboard_frame, text="Dashboard")

        # Workflows tab
        workflows_frame = WorkflowsFrame(self.notebook, self.thread_manager, main_app=self)
        self.notebook.add(workflows_frame, text="Workflows")

        # Memory tab
        memory_frame = MemoryFrame(self.notebook, self.thread_manager, self.db_helper)
        self.notebook.add(memory_frame, text="Memory")

        # Agents tab
        agents_frame = AgentsFrame(self.notebook, self.thread_manager, main_app=self)
        self.notebook.add(agents_frame, text="Agents")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_about(self):
        messagebox.showinfo("About", "Felix GUI - Version 1.0")

    def start_system(self):
        """Start the Felix system with full integration."""
        if not self.system_running:
            self.thread_manager.start_thread(self._start_system_thread)

    def _start_system_thread(self):
        """Thread function to start system components."""
        try:
            # Create Felix configuration
            config = FelixConfig(
                lm_host=self.lm_host,
                lm_port=self.lm_port,
                max_agents=15,
                enable_metrics=True,
                enable_memory=True,
                enable_dynamic_spawning=True
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
        # Enable agents frame features
        for child in self.notebook.winfo_children():
            if hasattr(child, '_enable_features') and callable(getattr(child, '_enable_features')):
                child._enable_features()

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