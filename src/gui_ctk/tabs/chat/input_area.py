"""
Message Input Area Component for Felix Chat Interface

The bottom input section of the chat with:
- Multi-line text input with auto-expansion (up to 150px)
- Send button with Ctrl/Cmd+Enter keyboard shortcut
- Stop generation button during streaming
- Character counter with optional token estimate
- Placeholder text when empty
- Enable/disable states

Usage Example:
    ```python
    from src.gui_ctk.tabs.chat.input_area import InputArea

    def handle_send(message: str):
        print(f"User sent: {message}")
        # Start generating response
        input_area.set_streaming(True)

    def handle_stop():
        print("User stopped generation")
        # Cancel generation
        input_area.set_streaming(False)

    # Create input area
    input_area = InputArea(
        parent_frame,
        on_send=handle_send,
        on_stop=handle_stop,
        show_char_count=True
    )
    input_area.pack(fill="x", padx=10, pady=10)

    # Control the input area
    input_area.set_enabled(True)          # Enable/disable input
    input_area.set_streaming(True)        # Show stop button
    input_area.focus()                    # Focus input field
    content = input_area.get_content()    # Get current text
    input_area.clear()                    # Clear input
    ```

Keyboard Shortcuts:
    - Ctrl+Enter (Windows/Linux) or Cmd+Enter (macOS): Send message
    - Escape: Stop generation (when streaming)

API Methods:
    - get_content() -> str: Get current input text
    - clear(): Clear the input field
    - set_content(text): Set input text programmatically
    - set_enabled(bool): Enable or disable input
    - set_streaming(bool): Toggle between send and stop button
    - focus(): Focus the input field
    - insert_text(text): Insert text at cursor position
    - get_last_sent() -> str: Get last sent message
"""

import customtkinter as ctk
from typing import Optional, Callable
import platform
import logging

from ...theme_manager import get_theme_manager
from ...styles import (
    BUTTON_SM, BUTTON_MD,
    FONT_BODY, FONT_CAPTION,
    SPACE_XS, SPACE_SM, SPACE_MD,
    RADIUS_MD
)

logger = logging.getLogger(__name__)


class InputArea(ctk.CTkFrame):
    """
    Multi-line input area with send/stop controls and character counter.

    Features:
    - Auto-expanding textbox (up to max height)
    - Keyboard shortcuts (Ctrl/Cmd+Enter to send, Escape to stop)
    - Character counter
    - Toggle between send and stop button
    - Placeholder text support
    - Enable/disable states
    """

    # Constants
    MIN_HEIGHT = 40
    MAX_HEIGHT = 150
    PLACEHOLDER_TEXT = "Type your message... (Ctrl+Enter to send)"

    def __init__(
        self,
        master,
        on_send: Optional[Callable[[str], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        show_char_count: bool = True,
        **kwargs
    ):
        """
        Initialize input area.

        Args:
            master: Parent widget
            on_send: Callback when message is sent (receives message text)
            on_stop: Callback when generation is stopped
            show_char_count: Whether to show character counter
            **kwargs: Additional arguments for CTkFrame
        """
        self.theme_manager = get_theme_manager()
        self.on_send = on_send
        self.on_stop = on_stop
        self.show_char_count = show_char_count

        self._is_streaming = False
        self._is_enabled = True
        self._last_content = ""
        self._showing_placeholder = True  # Track placeholder state with flag

        # Detect OS for keyboard shortcuts (must be before _setup_ui)
        self._is_macos = platform.system() == "Darwin"
        self._send_shortcut = "Cmd+Enter" if self._is_macos else "Ctrl+Enter"

        super().__init__(
            master,
            fg_color=self.theme_manager.get_color("bg_secondary"),
            corner_radius=RADIUS_MD,
            **kwargs
        )

        self._setup_ui()
        self._show_placeholder()  # Initialize with placeholder

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the UI components."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main input textbox
        self._textbox = ctk.CTkTextbox(
            self,
            font=ctk.CTkFont(size=FONT_BODY),
            fg_color=self.theme_manager.get_color("bg_primary"),
            text_color=self.theme_manager.get_color("fg_primary"),
            border_width=1,
            border_color=self.theme_manager.get_color("border"),
            corner_radius=RADIUS_MD,
            height=self.MIN_HEIGHT,
            wrap="word",
            activate_scrollbars=True
        )
        self._textbox.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=SPACE_SM,
            pady=(SPACE_SM, SPACE_XS)
        )

        # Bind keyboard events
        self._textbox.bind("<KeyPress>", self._on_key_press)
        self._textbox.bind("<KeyRelease>", self._on_key_release)
        self._textbox.bind(f"<{self._get_control_key()}-Return>", self._on_send_shortcut)
        self._textbox.bind("<Escape>", self._on_escape)
        self._textbox.bind("<FocusIn>", self._on_focus_in)
        self._textbox.bind("<FocusOut>", self._on_focus_out)

        # Bottom controls frame
        controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        controls_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=SPACE_SM,
            pady=(0, SPACE_SM)
        )
        controls_frame.grid_columnconfigure(0, weight=1)

        # Character counter (left side)
        if self.show_char_count:
            self._char_label = ctk.CTkLabel(
                controls_frame,
                text="0 characters",
                font=ctk.CTkFont(size=FONT_CAPTION),
                text_color=self.theme_manager.get_color("fg_muted"),
                anchor="w"
            )
            self._char_label.grid(row=0, column=0, sticky="w")

        # Send button (right side)
        self._send_btn = ctk.CTkButton(
            controls_frame,
            text="Send",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_BODY),
            fg_color=self.theme_manager.get_color("accent"),
            hover_color=self.theme_manager.get_color("accent_hover"),
            text_color="#FFFFFF",
            corner_radius=RADIUS_MD,
            command=self._on_send_clicked
        )
        self._send_btn.grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

        # Stop button (hidden by default, shown during streaming)
        self._stop_btn = ctk.CTkButton(
            controls_frame,
            text="Stop",
            width=BUTTON_SM[0],
            height=BUTTON_SM[1],
            font=ctk.CTkFont(size=FONT_BODY),
            fg_color=self.theme_manager.get_color("error"),
            hover_color=self._darken_color(self.theme_manager.get_color("error")),
            text_color="#FFFFFF",
            corner_radius=RADIUS_MD,
            command=self._on_stop_clicked
        )
        self._stop_btn.grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))
        self._stop_btn.grid_remove()  # Hidden by default

    def _get_control_key(self) -> str:
        """Get the control key for current platform."""
        return "Command" if self._is_macos else "Control"

    def _on_key_press(self, event):
        """Handle key press - clear placeholder BEFORE character is typed."""
        # Skip modifier keys and special keys
        if event.keysym in ('Shift_L', 'Shift_R', 'Control_L', 'Control_R',
                           'Alt_L', 'Alt_R', 'Super_L', 'Super_R', 'Caps_Lock',
                           'Tab', 'Escape', 'Return', 'BackSpace', 'Delete'):
            return

        # Clear placeholder before typing if showing
        if self._showing_placeholder:
            self._textbox.delete("1.0", "end")
            self._textbox.configure(text_color=self.theme_manager.get_color("fg_primary"))
            self._showing_placeholder = False

    def _on_key_release(self, event):
        """Handle key release events."""
        # Update character counter
        self._update_char_count()

        # Auto-expand textbox height based on content
        self._auto_expand_textbox()

    def _on_send_shortcut(self, event):
        """Handle Ctrl/Cmd+Enter keyboard shortcut."""
        self._on_send_clicked()
        return "break"  # Prevent default behavior

    def _on_escape(self, event):
        """Handle Escape key."""
        if self._is_streaming:
            self._on_stop_clicked()
        return "break"

    def _on_focus_in(self, event):
        """Handle focus in event - clear placeholder if showing."""
        if self._showing_placeholder:
            self._textbox.delete("1.0", "end")
            self._textbox.configure(text_color=self.theme_manager.get_color("fg_primary"))
            self._showing_placeholder = False

    def _on_focus_out(self, event):
        """Handle focus out event - show placeholder if empty."""
        raw = self._textbox.get("1.0", "end-1c").strip()
        if not raw:
            self._show_placeholder()

    def _on_send_clicked(self):
        """Handle send button click."""
        if not self._is_enabled or self._is_streaming:
            return

        content = self.get_content().strip()
        if not content:
            return

        # Store content and clear input
        self._last_content = content
        self.clear()

        # Call callback
        if self.on_send:
            try:
                self.on_send(content)
            except Exception as e:
                logger.error(f"Error in send callback: {e}")

    def _on_stop_clicked(self):
        """Handle stop button click."""
        if not self._is_streaming:
            return

        # Call callback
        if self.on_stop:
            try:
                self.on_stop()
            except Exception as e:
                logger.error(f"Error in stop callback: {e}")

    def _update_char_count(self):
        """Update the character counter display."""
        if not self.show_char_count:
            return

        content = self.get_content()
        char_count = len(content)

        # Estimate tokens (rough approximation: ~4 chars per token)
        token_estimate = char_count // 4

        if char_count > 0:
            text = f"{char_count} characters"
            if token_estimate > 0:
                text += f" (~{token_estimate} tokens)"
        else:
            text = "0 characters"

        self._char_label.configure(text=text)

    def _auto_expand_textbox(self):
        """Auto-expand textbox height based on content."""
        try:
            # Get number of lines in textbox
            content = self._textbox.get("1.0", "end-1c")
            line_count = content.count('\n') + 1

            # Calculate desired height (roughly 20px per line)
            line_height = 20
            desired_height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, line_count * line_height))

            # Update height if changed
            current_height = self._textbox.cget("height")
            if current_height != desired_height:
                self._textbox.configure(height=desired_height)

        except Exception as e:
            logger.debug(f"Error auto-expanding textbox: {e}")

    def _show_placeholder(self):
        """Show placeholder text."""
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", self.PLACEHOLDER_TEXT)
        self._textbox.configure(text_color=self.theme_manager.get_color("fg_muted"))
        self._showing_placeholder = True

    def _darken_color(self, hex_color: str, factor: float = 0.2) -> str:
        """Darken a hex color by a factor."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))

        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_theme_change(self, mode: str):
        """Handle theme change."""
        colors = self.theme_manager.colors

        self.configure(fg_color=colors["bg_secondary"])
        self._textbox.configure(
            fg_color=colors["bg_primary"],
            text_color=colors["fg_primary"],
            border_color=colors["border"]
        )

        if self.show_char_count:
            self._char_label.configure(text_color=colors["fg_muted"])

        self._send_btn.configure(
            fg_color=colors["accent"],
            hover_color=colors["accent_hover"]
        )

        self._stop_btn.configure(
            fg_color=colors["error"],
            hover_color=self._darken_color(colors["error"])
        )

    # Public API methods

    def get_content(self) -> str:
        """
        Get current input text.

        Returns:
            Current text content (empty string if placeholder is shown)
        """
        # Use flag to reliably detect placeholder state
        if self._showing_placeholder:
            return ""
        return self._textbox.get("1.0", "end-1c")

    def clear(self):
        """Clear the input field and show placeholder."""
        self._show_placeholder()
        self._update_char_count()
        self._auto_expand_textbox()

    def set_content(self, content: str):
        """
        Set input text content.

        Args:
            content: Text to set
        """
        self._textbox.delete("1.0", "end")
        if content:
            self._textbox.insert("1.0", content)
            self._textbox.configure(text_color=self.theme_manager.get_color("fg_primary"))
            self._showing_placeholder = False
        else:
            self._show_placeholder()
        self._update_char_count()
        self._auto_expand_textbox()

    def set_enabled(self, enabled: bool):
        """
        Enable or disable the input area.

        Args:
            enabled: Whether input should be enabled
        """
        self._is_enabled = enabled

        if enabled and not self._is_streaming:
            self._textbox.configure(state="normal")
            self._send_btn.configure(state="normal")
        else:
            self._textbox.configure(state="disabled")
            self._send_btn.configure(state="disabled")

    def set_streaming(self, is_streaming: bool):
        """
        Toggle streaming mode (show stop button instead of send).

        Args:
            is_streaming: Whether generation is currently streaming
        """
        self._is_streaming = is_streaming

        if is_streaming:
            # Show stop button, hide send button
            self._send_btn.grid_remove()
            self._stop_btn.grid()

            # Disable input
            self._textbox.configure(state="disabled")
        else:
            # Show send button, hide stop button
            self._stop_btn.grid_remove()
            self._send_btn.grid()

            # Re-enable input if enabled
            if self._is_enabled:
                self._textbox.configure(state="normal")

    def focus(self):
        """Set focus to the input field."""
        self._textbox.focus()

    def insert_text(self, text: str):
        """
        Insert text at current cursor position.

        Args:
            text: Text to insert
        """
        self._textbox.insert("insert", text)
        self._update_char_count()
        self._auto_expand_textbox()

    def get_last_sent(self) -> str:
        """
        Get the last sent message.

        Returns:
            Last message that was sent
        """
        return self._last_content

    def destroy(self):
        """Cleanup when destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass
        super().destroy()
