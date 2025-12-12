"""
Demo/Example: ThinkingView Component Usage

This file demonstrates how to use the ThinkingView and CompactThinkingView
components in a Felix CustomTkinter GUI.

Run this file directly to see a live demo:
    python -m src.gui_ctk.tabs.chat.thinking_view_demo
"""

import customtkinter as ctk
from thinking_view import ThinkingView, CompactThinkingView
import time
import threading


class ThinkingViewDemo(ctk.CTk):
    """Demo application showing ThinkingView usage."""

    def __init__(self):
        super().__init__()

        self.title("Felix ThinkingView Demo")
        self.geometry("700x600")

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._setup_ui()

    def _setup_ui(self):
        """Setup the demo UI."""
        # Main container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title = ctk.CTkLabel(
            container,
            text="ThinkingView Component Demo",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(0, 20))

        # Description
        desc = ctk.CTkLabel(
            container,
            text="This component shows real-time agent activity during multi-agent workflows.\n"
                 "Click the buttons below to simulate agent activity.",
            font=ctk.CTkFont(size=12)
        )
        desc.pack(pady=(0, 20))

        # Main ThinkingView
        self.thinking_view = ThinkingView(
            container,
            on_toggle=self._on_toggle
        )
        self.thinking_view.pack(fill="x", pady=(0, 20))

        # Compact ThinkingView
        compact_label = ctk.CTkLabel(
            container,
            text="Compact Variant:",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        )
        compact_label.pack(fill="x", pady=(10, 5))

        self.compact_view = CompactThinkingView(container)
        self.compact_view.pack(fill="x", pady=(0, 20))

        # Control buttons
        button_frame = ctk.CTkFrame(container, fg_color="transparent")
        button_frame.pack(fill="x")

        ctk.CTkButton(
            button_frame,
            text="Add Research Step",
            command=lambda: self._add_step("research")
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            button_frame,
            text="Add Analysis Step",
            command=lambda: self._add_step("analysis")
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            button_frame,
            text="Add Synthesis Step",
            command=lambda: self._add_step("synthesis")
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            button_frame,
            text="Add Critic Step",
            command=lambda: self._add_step("critic")
        ).pack(side="left", padx=(0, 10))

        # More controls
        control_frame = ctk.CTkFrame(container, fg_color="transparent")
        control_frame.pack(fill="x", pady=(10, 0))

        ctk.CTkButton(
            control_frame,
            text="Simulate Workflow",
            command=self._simulate_workflow,
            fg_color="green"
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            control_frame,
            text="Clear All",
            command=self._clear_all,
            fg_color="orange"
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            control_frame,
            text="Toggle Expand",
            command=lambda: self.thinking_view.set_expanded(not self.thinking_view.get_expanded())
        ).pack(side="left")

        # Status label
        self.status_label = ctk.CTkLabel(
            container,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.pack(pady=(20, 0))

    def _on_toggle(self, expanded: bool):
        """Handle toggle event."""
        state = "expanded" if expanded else "collapsed"
        self.status_label.configure(text=f"ThinkingView {state}")

    def _add_step(self, agent_type: str):
        """Add a single agent step."""
        messages = {
            "research": "Gathering information from knowledge base...",
            "analysis": "Analyzing patterns and relationships...",
            "synthesis": "Combining insights into solution...",
            "critic": "Evaluating quality and completeness..."
        }

        content = messages.get(agent_type, "Working...")

        # Add to main view with progress
        self.thinking_view.add_agent_step(agent_type, content, progress=0.5)

        # Update compact view
        self.compact_view.set_agent(agent_type, content)

        self.status_label.configure(text=f"Added {agent_type} step")

    def _simulate_workflow(self):
        """Simulate a complete multi-agent workflow."""
        def workflow():
            # Clear first
            self.after(0, self._clear_all)
            time.sleep(0.5)

            # Research phase
            self.after(0, lambda: self.thinking_view.add_agent_step(
                "research", "Searching knowledge base...", progress=0.0
            ))
            self.after(0, lambda: self.compact_view.set_agent(
                "research", "Searching knowledge base..."
            ))
            time.sleep(1)

            self.after(0, lambda: self.thinking_view.update_agent_step(
                "research", "Found 15 relevant documents", progress=1.0
            ))
            time.sleep(0.5)

            # Analysis phase
            self.after(0, lambda: self.thinking_view.add_agent_step(
                "analysis", "Processing context...", progress=0.0
            ))
            self.after(0, lambda: self.compact_view.set_agent(
                "analysis", "Processing context..."
            ))
            time.sleep(1)

            self.after(0, lambda: self.thinking_view.update_agent_step(
                "analysis", "Identified key patterns", progress=1.0
            ))
            time.sleep(0.5)

            # Synthesis phase
            self.after(0, lambda: self.thinking_view.add_agent_step(
                "synthesis", "Creating solution...", progress=0.0
            ))
            self.after(0, lambda: self.compact_view.set_agent(
                "synthesis", "Creating solution..."
            ))
            time.sleep(1)

            self.after(0, lambda: self.thinking_view.update_agent_step(
                "synthesis", "Solution generated", progress=1.0
            ))
            time.sleep(0.5)

            # Critic phase
            self.after(0, lambda: self.thinking_view.add_agent_step(
                "critic", "Reviewing quality...", progress=0.0
            ))
            self.after(0, lambda: self.compact_view.set_agent(
                "critic", "Reviewing quality..."
            ))
            time.sleep(1)

            self.after(0, lambda: self.thinking_view.update_agent_step(
                "critic", "Quality check passed", progress=1.0
            ))

            self.after(0, lambda: self.status_label.configure(
                text="Workflow complete!"
            ))

        # Run in background thread
        thread = threading.Thread(target=workflow, daemon=True)
        thread.start()
        self.status_label.configure(text="Simulating workflow...")

    def _clear_all(self):
        """Clear all agent steps."""
        self.thinking_view.clear()
        self.compact_view.clear()
        self.status_label.configure(text="Cleared all steps")


def main():
    """Run the demo application."""
    app = ThinkingViewDemo()
    app.mainloop()


if __name__ == "__main__":
    main()
