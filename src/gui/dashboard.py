import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
from .utils import log_queue, logger

try:
    from src.communication import central_post
    from src.llm import lm_studio_client
except ImportError as e:
    logger.error(f"Failed to import Felix modules: {e}")
    central_post = None
    lm_studio_client = None

class DashboardFrame(ttk.Frame):
    def __init__(self, parent, thread_manager):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.system_running = False
        self.cp = None
        self.client = None
        self.lm_host = "127.0.0.1"
        self.lm_port = "1234"

        # LM Studio config
        config_frame = ttk.Frame(self)
        config_frame.pack(pady=5)
        ttk.Label(config_frame, text="LM Studio Host:").grid(row=0, column=0, padx=5)
        self.host_entry = ttk.Entry(config_frame, width=15)
        self.host_entry.insert(0, self.lm_host)
        self.host_entry.grid(row=0, column=1, padx=5)
        ttk.Label(config_frame, text="Port:").grid(row=0, column=2, padx=5)
        self.port_entry = ttk.Entry(config_frame, width=6)
        self.port_entry.insert(0, self.lm_port)
        self.port_entry.grid(row=0, column=3, padx=5)

        # Start/Stop buttons
        self.start_button = ttk.Button(self, text="Start Felix", command=self.start_system)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(self, text="Stop Felix", command=self.stop_system, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # Reference to main app for system state
        self.main_app = None
        if hasattr(parent.master, 'system_running'):
            self.main_app = parent.master

        # Log display
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Poll log queue
        self.poll_log_queue()

    def poll_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + '\n')
                self.log_text.see(tk.END)
        except:
            pass
        self.after(100, self.poll_log_queue)

    def start_system(self):
        if not self.system_running:
            self.start_button.config(state=tk.DISABLED)
            # Use main app's start_system method
            if self.main_app:
                self.main_app.start_system()
                # Update local state after a short delay to allow main app to process
                self.after(100, self._update_local_state)
            else:
                # Fallback to original implementation
                self.thread_manager.start_thread(self._start_system_thread)

    def _update_local_state(self):
        """Update local dashboard state based on main app state."""
        if self.main_app:
            self.system_running = self.main_app.system_running
            if self.system_running:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.NORMAL)
                self._enable_other_tabs()
            else:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self._disable_other_tabs()

    def _enable_other_tabs(self):
        """Enable features in other tabs when system is running."""
        if self.main_app and self.main_app.notebook:
            for i in range(1, self.main_app.notebook.index("end")):  # Skip Dashboard tab
                tab_frame = self.main_app.notebook.winfo_children()[i]
                if hasattr(tab_frame, '_enable_features'):
                    tab_frame._enable_features()

    def _disable_other_tabs(self):
        """Disable features in other tabs when system is not running."""
        if self.main_app and self.main_app.notebook:
            for i in range(1, self.main_app.notebook.index("end")):  # Skip Dashboard tab
                tab_frame = self.main_app.notebook.winfo_children()[i]
                if hasattr(tab_frame, '_disable_features'):
                    tab_frame._disable_features()

    def _start_system_thread(self):
        try:
            if central_post:
                self.cp = central_post.CentralPost()
            if lm_studio_client:
                host = self.host_entry.get().strip() or "127.0.0.1"
                port = self.port_entry.get().strip() or "1234"
                base_url = f"http://{host}:{port}/v1"
                self.client = lm_studio_client.LMStudioClient(base_url=base_url)
                # Test connection
                if self.client.test_connection():
                    logger.info("Connected to LM Studio")
                else:
                    logger.warning("Failed to connect to LM Studio - ensure server is running")
            logger.info("Felix system started")
            self.system_running = True
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.after(0, lambda: self.stop_button.config(state=tk.NORMAL))
            self.after(0, self._enable_other_tabs)
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to start system: {e}"))
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def stop_system(self):
        if self.system_running:
            self.stop_button.config(state=tk.DISABLED)
            # Use main app's stop_system method
            if self.main_app:
                self.main_app.stop_system()
                # Update local state after a short delay
                self.after(100, self._update_local_state)
            else:
                # Fallback to original implementation
                self.thread_manager.start_thread(self._stop_system_thread)

    def _stop_system_thread(self):
        try:
            if self.cp:
                self.cp.shutdown()
            logger.info("Felix system stopped")
            self.system_running = False
            self.after(0, lambda: self.stop_button.config(state=tk.NORMAL))
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.after(0, self._disable_other_tabs)
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to stop system: {e}"))
            self.after(0, lambda: self.stop_button.config(state=tk.NORMAL))