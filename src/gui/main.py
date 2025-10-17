import tkinter as tk
from tkinter import ttk, messagebox
from .utils import ThreadManager, DBHelper, logger
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Felix GUI")
        self.geometry("800x600")

        # System state
        self.system_running = False
        self.lm_client = None
        self.lm_host = '127.0.0.1'
        self.lm_port = 1234
        self.cp = None

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
        dashboard_frame = DashboardFrame(self.notebook, self.thread_manager)
        self.notebook.add(dashboard_frame, text="Dashboard")

        # Workflows tab
        workflows_frame = WorkflowsFrame(self.notebook, self.thread_manager)
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
        """Start the Felix system with LM Studio verification."""
        if not self.system_running:
            self.thread_manager.start_thread(self._start_system_thread)

    def _start_system_thread(self):
        """Thread function to start system components."""
        try:
            # Initialize central post
            try:
                from src.communication import central_post
                self.cp = central_post.CentralPost()
                logger.info("Central post initialized")
            except ImportError as e:
                logger.warning(f"Central post not available: {e}")

            # Initialize LM Studio client and verify connection
            try:
                from src.llm import lm_studio_client
                base_url = f"http://{self.lm_host}:{self.lm_port}/v1"
                self.lm_client = lm_studio_client.LMStudioClient(base_url=base_url)

                # Test connection with a simple request
                if self.lm_client.test_connection():
                    logger.info(f"Connected to LM Studio at {self.lm_host}:{self.lm_port}")
                    self.system_running = True
                    self.status_var.set("System Running")
                    self._enable_all_features()
                else:
                    logger.error(f"Failed to connect to LM Studio at {self.lm_host}:{self.lm_port}")
                    self.system_running = False
                    self.status_var.set("LM Studio Connection Failed")
                    messagebox.showwarning("Connection Failed",
                                         f"Cannot connect to LM Studio at {self.lm_host}:{self.lm_port}.\n"
                                         "Ensure LM Studio is running with a model loaded.")

            except ImportError as e:
                logger.warning(f"LM Studio client not available: {e}")
                self.system_running = False
                self.status_var.set("LM Studio Not Available")
            except Exception as e:
                logger.error(f"Error initializing LM Studio client: {e}")
                self.system_running = False
                self.status_var.set("LM Studio Error")
                messagebox.showerror("Error", f"Failed to initialize LM Studio client: {e}")

            if self.system_running:
                logger.info("Felix system started")
            else:
                logger.warning("Felix system started with limited functionality")

        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.system_running = False
            self.status_var.set("System Start Failed")
            messagebox.showerror("Error", f"Failed to start system: {e}")

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
            if self.lm_client:
                # Close LM client if it has a close method
                if hasattr(self.lm_client, 'close_async'):
                    import asyncio
                    asyncio.run(self.lm_client.close_async())
                self.lm_client = None

            if self.cp and hasattr(self.cp, 'shutdown'):
                self.cp.shutdown()

            self.system_running = False
            self.status_var.set("System Stopped")
            logger.info("Felix system stopped")

        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            messagebox.showerror("Error", f"Failed to stop system: {e}")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()