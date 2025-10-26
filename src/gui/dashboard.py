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
    def __init__(self, parent, thread_manager, main_app=None, theme_manager=None):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager
        self.system_running = False

        # Info label
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(pady=10, fill=tk.X)
        self.title_label = ttk.Label(self.info_frame, text="Felix Framework Dashboard",
                 font=("TkDefaultFont", 12, "bold"))
        self.title_label.pack()
        self.info_label = ttk.Label(self.info_frame, text="Configure LM Studio connection in the Settings tab",
                 foreground="gray")
        self.info_label.pack()

        # Start/Stop buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)
        self.start_button = ttk.Button(button_frame, text="Start Felix", command=self.start_system)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop Felix", command=self.stop_system, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Log display
        self.log_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Apply initial theme
        self.apply_theme()

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
                # Poll for system readiness instead of fixed delay
                self._poll_system_ready(max_attempts=20, poll_interval=250)
            else:
                logger.error("Cannot start system: main_app not available")
                self.after(0, lambda: tk.messagebox.showerror("Error", "Main application reference not available"))
                self.start_button.config(state=tk.NORMAL)

    def _poll_system_ready(self, max_attempts=20, poll_interval=250, attempt=0):
        """Poll for system ready state instead of using fixed delay."""
        if attempt >= max_attempts:
            logger.warning("System startup polling timed out")
            self.start_button.config(state=tk.NORMAL)
            return

        if self.main_app and self.main_app.system_running and self.main_app.felix_system:
            # System is ready
            self._update_local_state()
        else:
            # Keep polling
            self.after(poll_interval, lambda: self._poll_system_ready(max_attempts, poll_interval, attempt + 1))

    def _update_local_state(self):
        """Update local dashboard state based on main app state."""
        if self.main_app:
            self.system_running = self.main_app.system_running
            if self.system_running:
                self.stop_button.config(state=tk.NORMAL)
                self.start_button.config(state=tk.DISABLED)
                self._enable_other_tabs()
                # Show system status
                if self.main_app.felix_system:
                    status = self.main_app.felix_system.get_system_status()
                    logger.info(f"Felix system running: {status['agents']} agents active")
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

    def stop_system(self):
        if self.system_running:
            self.stop_button.config(state=tk.DISABLED)
            # Use main app's stop_system method
            if self.main_app:
                self.main_app.stop_system()
                # Poll for system stopped state
                self._poll_system_stopped(max_attempts=20, poll_interval=250)
            else:
                logger.error("Cannot stop system: main_app not available")
                self.after(0, lambda: tk.messagebox.showerror("Error", "Main application reference not available"))
                self.stop_button.config(state=tk.NORMAL)

    def _poll_system_stopped(self, max_attempts=20, poll_interval=250, attempt=0):
        """Poll for system stopped state."""
        if attempt >= max_attempts:
            logger.warning("System shutdown polling timed out")
            self.stop_button.config(state=tk.NORMAL)
            return

        if self.main_app and not self.main_app.system_running:
            # System is stopped
            self._update_local_state()
        else:
            # Keep polling
            self.after(poll_interval, lambda: self._poll_system_stopped(max_attempts, poll_interval, attempt + 1))

    def apply_theme(self):
        """Apply current theme to the dashboard widgets."""
        if not self.theme_manager:
            return

        theme = self.theme_manager.get_current_theme()

        # Apply to log text widget (ScrolledText)
        try:
            # ScrolledText contains internal Text widget
            for child in self.log_text.winfo_children():
                if isinstance(child, tk.Text):
                    self.theme_manager.apply_to_text_widget(child)
        except Exception as e:
            logger.warning(f"Could not theme log_text: {e}")

        # Apply theme to info label (gray color)
        try:
            self.info_label.configure(foreground=theme["fg_tertiary"])
        except Exception as e:
            logger.warning(f"Could not theme info_label: {e}")

        # Recursively apply theme to all children
        try:
            self.theme_manager.apply_to_all_children(self)
        except Exception as e:
            logger.warning(f"Could not recursively apply theme: {e}")