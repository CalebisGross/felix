"""
Example usage of InputArea component for Felix Chat Interface

This demonstrates how to use the InputArea component with callbacks
and various features like streaming mode, enable/disable, etc.

Run with: python -m src.gui_ctk.tabs.chat.input_area_example
"""

import customtkinter as ctk
from input_area import InputArea
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputAreaDemo(ctk.CTk):
    """Demo application for InputArea component."""

    def __init__(self):
        super().__init__()

        self.title("Felix InputArea Demo")
        self.geometry("700x400")

        # Set appearance
        ctk.set_appearance_mode("dark")

        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="InputArea Component Demo",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, pady=(0, 10))

        # Output area (to show sent messages)
        self.output = ctk.CTkTextbox(
            main_frame,
            height=200,
            font=ctk.CTkFont(size=12)
        )
        self.output.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        # Control buttons
        controls = ctk.CTkFrame(main_frame, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkButton(
            controls,
            text="Toggle Streaming",
            command=self.toggle_streaming,
            width=150
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            controls,
            text="Toggle Enabled",
            command=self.toggle_enabled,
            width=150
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            controls,
            text="Insert Text",
            command=self.insert_sample_text,
            width=150
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            controls,
            text="Clear Output",
            command=self.clear_output,
            width=150
        ).pack(side="left", padx=5)

        # Input area
        self.input_area = InputArea(
            main_frame,
            on_send=self.on_message_sent,
            on_stop=self.on_stop_generation,
            show_char_count=True
        )
        self.input_area.grid(row=3, column=0, sticky="ew")

        # State
        self.is_streaming = False
        self.is_enabled = True

        # Initial message
        self.output.insert("1.0", "Welcome to InputArea Demo!\n")
        self.output.insert("end", "Try typing a message and pressing Send or Ctrl+Enter.\n\n")

    def on_message_sent(self, message: str):
        """Handle message sent."""
        logger.info(f"Message sent: {message}")
        self.output.insert("end", f"USER: {message}\n\n")
        self.output.see("end")

        # Simulate streaming response
        self.simulate_streaming_response()

    def on_stop_generation(self):
        """Handle stop generation."""
        logger.info("Generation stopped")
        self.output.insert("end", "[Generation stopped]\n\n")
        self.output.see("end")

        # Stop streaming
        self.is_streaming = False
        self.input_area.set_streaming(False)

    def simulate_streaming_response(self):
        """Simulate a streaming response."""
        self.is_streaming = True
        self.input_area.set_streaming(True)

        # Simulate response
        self.output.insert("end", "FELIX: [Thinking...]\n\n")
        self.output.see("end")

        # After 2 seconds, stop streaming
        self.after(2000, self.finish_streaming)

    def finish_streaming(self):
        """Finish streaming simulation."""
        if self.is_streaming:
            self.output.insert("end", "FELIX: This is a simulated response.\n\n")
            self.output.see("end")

            self.is_streaming = False
            self.input_area.set_streaming(False)

    def toggle_streaming(self):
        """Toggle streaming mode."""
        self.is_streaming = not self.is_streaming
        self.input_area.set_streaming(self.is_streaming)

        status = "ON" if self.is_streaming else "OFF"
        self.output.insert("end", f"[Streaming mode: {status}]\n\n")
        self.output.see("end")

    def toggle_enabled(self):
        """Toggle enabled state."""
        self.is_enabled = not self.is_enabled
        self.input_area.set_enabled(self.is_enabled)

        status = "ENABLED" if self.is_enabled else "DISABLED"
        self.output.insert("end", f"[Input area: {status}]\n\n")
        self.output.see("end")

    def insert_sample_text(self):
        """Insert sample text."""
        sample = "This is sample text inserted programmatically."
        self.input_area.insert_text(sample)

        self.output.insert("end", f"[Inserted: '{sample}']\n\n")
        self.output.see("end")

    def clear_output(self):
        """Clear the output area."""
        self.output.delete("1.0", "end")


if __name__ == "__main__":
    app = InputAreaDemo()
    app.mainloop()
