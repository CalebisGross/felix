"""
Message Bubble Component for Felix Chat Interface

Displays individual chat messages with:
- Role-based styling (user vs assistant)
- Markdown rendering for code blocks
- Copy button on hover
- Timestamp display
- Collapsible thinking section for assistant messages
"""

import customtkinter as ctk
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
import re
import logging

from ...theme_manager import get_theme_manager
from ...styles import (
    FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD,
    RADIUS_MD, RADIUS_LG
)

logger = logging.getLogger(__name__)


class MessageBubble(ctk.CTkFrame):
    """
    A single message bubble in the chat interface.

    Displays user or assistant messages with appropriate styling,
    optional thinking process view, and copy functionality.
    """

    def __init__(
        self,
        master,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        thinking: Optional[List[Dict[str, str]]] = None,
        knowledge_sources: Optional[List[Dict[str, str]]] = None,
        on_copy: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        """
        Initialize message bubble.

        Args:
            master: Parent widget
            role: Message role ('user' or 'assistant')
            content: Message text content
            timestamp: When the message was sent
            thinking: List of agent thinking steps (for assistant messages)
            knowledge_sources: List of knowledge sources used
            on_copy: Callback when copy button is clicked
            **kwargs: Additional arguments for CTkFrame
        """
        self.theme_manager = get_theme_manager()
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.thinking = thinking or []
        self.knowledge_sources = knowledge_sources or []
        self.on_copy = on_copy
        self._thinking_expanded = False

        # Get colors based on role - use hardcoded visible text colors
        colors = self.theme_manager.colors
        if role == "user":
            bg_color = colors["accent"]
            fg_color = "#FFFFFF"  # White on blue
        else:
            bg_color = colors["bg_secondary"]
            fg_color = "#FFFFFF"  # White on dark gray - guaranteed visible

        super().__init__(
            master,
            fg_color=bg_color,
            corner_radius=RADIUS_LG,
            **kwargs
        )

        self._bg_color = bg_color
        self._fg_color = fg_color

        self._setup_ui()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def _setup_ui(self):
        """Setup the UI components."""
        # Main container with padding
        self.grid_columnconfigure(0, weight=1)

        # Row counter for grid placement
        current_row = 0

        # Header with role label and timestamp (use explicit bg color, not "transparent")
        header_frame = ctk.CTkFrame(self, fg_color=self._bg_color, corner_radius=0)
        header_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        header_frame.grid_columnconfigure(0, weight=1)

        role_label = ctk.CTkLabel(
            header_frame,
            text=self.role.upper() if self.role == "user" else "FELIX",
            font=ctk.CTkFont(size=FONT_CAPTION, weight="bold"),
            text_color=self._fg_color,
            fg_color=self._bg_color,
            anchor="w"
        )
        role_label.grid(row=0, column=0, sticky="w")

        time_label = ctk.CTkLabel(
            header_frame,
            text=self.timestamp.strftime("%H:%M"),
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self._fg_color,
            fg_color=self._bg_color,
            anchor="e"
        )
        time_label.grid(row=0, column=1, sticky="e", padx=(SPACE_SM, 0))

        # Copy button (hidden by default, shown on hover)
        self._copy_btn = ctk.CTkButton(
            header_frame,
            text="Copy",
            width=50,
            height=20,
            font=ctk.CTkFont(size=FONT_SMALL),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self._fg_color,
            command=self._copy_content
        )
        self._copy_btn.grid(row=0, column=2, sticky="e", padx=(SPACE_XS, 0))
        self._copy_btn.grid_remove()  # Hide initially

        current_row += 1

        # Thinking section (for assistant messages with agent activity)
        if self.role == "assistant" and self.thinking:
            self._thinking_frame = self._create_thinking_section()
            self._thinking_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
            current_row += 1

        # Main content
        initial_text = self._format_content(self.content) if self.content else ""
        logger.info(f"MessageBubble: role={self.role}, content='{initial_text[:100]}...'")

        self._content_label = ctk.CTkLabel(
            self,
            text=initial_text,
            font=ctk.CTkFont(size=FONT_BODY),
            text_color=self._fg_color,
            fg_color=self._bg_color,
            anchor="w",
            justify="left",
            wraplength=400  # Default wraplength, updated dynamically
        )
        self._content_label.grid(row=current_row, column=0, sticky="w", padx=SPACE_SM, pady=(0, SPACE_SM))
        current_row += 1

        # Knowledge sources (if any)
        if self.knowledge_sources:
            self._sources_frame = self._create_sources_section()
            self._sources_frame.grid(row=current_row, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))

        # Bind hover events for copy button
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _create_thinking_section(self) -> ctk.CTkFrame:
        """Create the collapsible thinking process section."""
        frame = ctk.CTkFrame(self, fg_color=self.theme_manager.get_color("bg_tertiary"), corner_radius=RADIUS_MD)
        frame.grid_columnconfigure(0, weight=1)

        # Toggle header
        toggle_frame = ctk.CTkFrame(frame, fg_color="transparent")
        toggle_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_XS, pady=SPACE_XS)
        toggle_frame.grid_columnconfigure(0, weight=1)

        self._thinking_toggle = ctk.CTkButton(
            toggle_frame,
            text="▶ Thinking Process",
            font=ctk.CTkFont(size=FONT_CAPTION),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            anchor="w",
            command=self._toggle_thinking
        )
        self._thinking_toggle.grid(row=0, column=0, sticky="w")

        # Content (hidden by default)
        self._thinking_content = ctk.CTkFrame(frame, fg_color="transparent")
        self._thinking_content.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        self._thinking_content.grid_remove()  # Hidden by default

        # Populate thinking steps
        for i, step in enumerate(self.thinking):
            agent_type = step.get("agent", "unknown")
            step_content = step.get("content", "")

            # Agent color coding
            agent_colors = {
                "research": "#3B82F6",    # Blue
                "analysis": "#8B5CF6",    # Purple
                "synthesis": "#10B981",   # Green
                "critic": "#F59E0B",      # Amber
            }
            agent_color = agent_colors.get(agent_type, self.theme_manager.get_color("fg_secondary"))

            step_label = ctk.CTkLabel(
                self._thinking_content,
                text=f"▸ {agent_type.title()}: {step_content}",
                font=ctk.CTkFont(size=FONT_SMALL),
                text_color=agent_color,
                anchor="w",
                justify="left",
                wraplength=450
            )
            step_label.grid(row=i, column=0, sticky="w", pady=(0, SPACE_XS))

        return frame

    def _create_sources_section(self) -> ctk.CTkFrame:
        """Create the knowledge sources section."""
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)

        # Sources header
        header = ctk.CTkLabel(
            frame,
            text="Sources:",
            font=ctk.CTkFont(size=FONT_SMALL, weight="bold"),
            text_color=self.theme_manager.get_color("fg_secondary"),
            anchor="w"
        )
        header.grid(row=0, column=0, sticky="w")

        # Source items
        for i, source in enumerate(self.knowledge_sources):
            source_title = source.get("title", "Unknown source")
            source_label = ctk.CTkLabel(
                frame,
                text=f"  • {source_title}",
                font=ctk.CTkFont(size=FONT_SMALL),
                text_color=self.theme_manager.get_color("fg_muted"),
                anchor="w"
            )
            source_label.grid(row=i + 1, column=0, sticky="w")

        return frame

    def _format_content(self, content: str) -> str:
        """
        Format content for display.

        Basic markdown handling - code blocks are indicated with backticks.
        Full markdown rendering would require a more complex solution.
        """
        # For now, just return the content as-is
        # Future: Could use a markdown-to-text converter or custom rendering
        return content

    def _toggle_thinking(self):
        """Toggle the thinking section visibility."""
        self._thinking_expanded = not self._thinking_expanded

        if self._thinking_expanded:
            self._thinking_content.grid()
            self._thinking_toggle.configure(text="▼ Thinking Process")
        else:
            self._thinking_content.grid_remove()
            self._thinking_toggle.configure(text="▶ Thinking Process")

    def _copy_content(self):
        """Copy message content to clipboard."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.content)

            if self.on_copy:
                self.on_copy(self.content)

            # Visual feedback
            self._copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self._copy_btn.configure(text="Copy"))

        except Exception as e:
            logger.error(f"Failed to copy content: {e}")

    def _on_enter(self, event):
        """Show copy button on hover."""
        self._copy_btn.grid()

    def _on_leave(self, event):
        """Hide copy button when not hovering."""
        self._copy_btn.grid_remove()

    def _on_theme_change(self, mode: str):
        """Update colors when theme changes."""
        colors = self.theme_manager.colors

        if self.role == "user":
            bg_color = colors["accent"]
            fg_color = "#FFFFFF"
        else:
            bg_color = colors["bg_secondary"]
            fg_color = colors["fg_primary"]

        self._bg_color = bg_color
        self._fg_color = fg_color

        self.configure(fg_color=bg_color)
        self._content_label.configure(text_color=fg_color)

    def update_content(self, content: str):
        """Update the message content (for streaming)."""
        self.content = content
        self._content_label.configure(text=self._format_content(content))

    def append_content(self, chunk: str):
        """Append content to the message (for streaming)."""
        self.content += chunk
        logger.info(f"MessageBubble append: total={len(self.content)} chars")
        self._content_label.configure(text=self.content)

    def set_wraplength(self, width: int):
        """Update the wrap length for responsive layout."""
        # Account for padding
        wrap_width = max(200, width - SPACE_SM * 4)
        self._content_label.configure(wraplength=wrap_width)

    def add_thinking_step(self, agent: str, content: str):
        """Add a thinking step (for streaming agent activity)."""
        self.thinking.append({"agent": agent, "content": content})

        # Rebuild thinking section if it exists
        if hasattr(self, '_thinking_frame'):
            # Add new step to the content frame
            i = len(self.thinking) - 1
            agent_colors = {
                "research": "#3B82F6",
                "analysis": "#8B5CF6",
                "synthesis": "#10B981",
                "critic": "#F59E0B",
            }
            agent_color = agent_colors.get(agent, self.theme_manager.get_color("fg_secondary"))

            step_label = ctk.CTkLabel(
                self._thinking_content,
                text=f"▸ {agent.title()}: {content}",
                font=ctk.CTkFont(size=FONT_SMALL),
                text_color=agent_color,
                anchor="w",
                justify="left",
                wraplength=450
            )
            step_label.grid(row=i, column=0, sticky="w", pady=(0, SPACE_XS))

    def destroy(self):
        """Cleanup when destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass
        super().destroy()


class StreamingMessageBubble(MessageBubble):
    """
    A message bubble optimized for streaming content.

    Shows a typing indicator while content is being received,
    and efficiently updates as new chunks arrive.
    """

    def __init__(self, master, **kwargs):
        """Initialize with empty content."""
        kwargs.setdefault("content", "")
        kwargs.setdefault("role", "assistant")
        super().__init__(master, **kwargs)

        self._is_streaming = True
        self._typing_indicator = None
        self._show_typing_indicator()

    def _show_typing_indicator(self):
        """Show typing indicator while waiting for content."""
        if not self._typing_indicator:
            self._typing_indicator = ctk.CTkLabel(
                self,
                text="●●●",
                font=ctk.CTkFont(size=FONT_BODY),
                text_color=self.theme_manager.get_color("fg_muted"),
                fg_color=self._bg_color
            )
            # Place after content label (row 2 for streaming bubbles)
            self._typing_indicator.grid(row=2, column=0, sticky="w", padx=SPACE_SM, pady=(0, SPACE_SM))

    def _hide_typing_indicator(self):
        """Hide typing indicator."""
        if self._typing_indicator:
            self._typing_indicator.destroy()
            self._typing_indicator = None

    def append_content(self, chunk: str):
        """Append streaming content and hide typing indicator."""
        if self._is_streaming and not self.content:
            self._hide_typing_indicator()

        super().append_content(chunk)

    def finish_streaming(self):
        """Mark streaming as complete."""
        self._is_streaming = False
        self._hide_typing_indicator()
