"""
Message Area Component for Felix Chat Interface

Scrollable container for chat message bubbles with:
- Auto-scroll to bottom on new messages
- Lazy loading of history on scroll to top
- Efficient rendering for long conversations
- Support for both static and streaming messages
"""

import customtkinter as ctk
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
import logging

from ...theme_manager import get_theme_manager
from ...styles import SPACE_SM, SPACE_MD, SPACE_LG
from .message_bubble import MessageBubble, StreamingMessageBubble
from .action_bubble import ActionBubble

logger = logging.getLogger(__name__)


class MessageArea(ctk.CTkScrollableFrame):
    """
    Scrollable area containing chat message bubbles.

    Features:
    - Efficient rendering with CTkScrollableFrame
    - Auto-scroll to bottom on new messages
    - Lazy loading of message history on scroll to top
    - Responsive layout for message bubbles
    - Support for user and assistant messages with metadata

    Usage:
        message_area = MessageArea(parent, on_load_more=load_more_callback)
        message_area.add_message("user", "Hello Felix")
        message_area.add_message("assistant", "Hello! How can I help?")

        # For streaming responses
        bubble = message_area.add_streaming_message()
        bubble.append_content("Response chunk")
        bubble.finish_streaming()
    """

    def __init__(
        self,
        master,
        on_load_more: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """
        Initialize message area.

        Args:
            master: Parent widget
            on_load_more: Optional callback triggered when scrolling to top
                         for lazy loading of older messages
            **kwargs: Additional arguments for CTkScrollableFrame
        """
        self.theme_manager = get_theme_manager()

        # Configure scrollable frame
        kwargs.setdefault("fg_color", self.theme_manager.get_color("bg_primary"))
        kwargs.setdefault("corner_radius", 0)

        super().__init__(master, **kwargs)

        self.on_load_more = on_load_more
        self._message_bubbles: List[MessageBubble] = []
        self._action_bubbles: List[ActionBubble] = []
        self._streaming_bubble: Optional[StreamingMessageBubble] = None
        self._auto_scroll = True
        self._is_loading_more = False
        self._scroll_position = 0.0
        self._last_wrap_width = 0  # Cache for debouncing width updates
        self._width_update_pending = None  # Debounce timer ID

        # Configure grid for message layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Bind scroll events for lazy loading detection
        self._scrollbar = self._parent_canvas.yview
        self.bind("<Configure>", self._on_configure)

        # Bind mouse wheel events for scrolling (Linux uses Button-4/5)
        self.bind("<MouseWheel>", self._on_mousewheel)  # Windows/MacOS
        self.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down
        self._parent_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._parent_canvas.bind("<Button-4>", self._on_mousewheel)
        self._parent_canvas.bind("<Button-5>", self._on_mousewheel)

        # Sync canvas background color with theme (100ms delay to ensure canvas exists)
        self.after(100, self._sync_canvas_bg)

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        thinking: Optional[List[Dict[str, str]]] = None,
        knowledge_sources: Optional[List[Dict[str, str]]] = None
    ) -> MessageBubble:
        """
        Add a message bubble to the chat area.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message text content
            timestamp: When the message was sent (defaults to now)
            thinking: List of agent thinking steps (for assistant messages)
            knowledge_sources: List of knowledge sources used

        Returns:
            The created MessageBubble instance
        """
        try:
            # Create message bubble
            bubble = MessageBubble(
                self,
                role=role,
                content=content,
                timestamp=timestamp or datetime.now(),
                thinking=thinking,
                knowledge_sources=knowledge_sources,
                on_copy=self._on_message_copied
            )

            # Position in grid
            row = len(self._message_bubbles)
            bubble.grid(
                row=row,
                column=0,
                sticky="ew",
                padx=SPACE_MD,
                pady=(SPACE_SM if row > 0 else SPACE_MD, SPACE_SM)
            )

            # Bind mousewheel to bubble for scroll passthrough
            bubble.bind("<MouseWheel>", self._on_mousewheel)
            bubble.bind("<Button-4>", self._on_mousewheel)
            bubble.bind("<Button-5>", self._on_mousewheel)

            # Store reference
            self._message_bubbles.append(bubble)

            # Auto-scroll to bottom if enabled
            if self._auto_scroll:
                self.after(50, self.scroll_to_bottom)

            # Update wrap lengths for responsive layout (use after_idle for proper timing)
            self.after_idle(self._update_bubble_widths)

            return bubble

        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise

    def add_streaming_message(self) -> StreamingMessageBubble:
        """
        Add a streaming message bubble for real-time content updates.

        Returns:
            StreamingMessageBubble instance that can be updated with chunks

        Example:
            bubble = message_area.add_streaming_message()
            bubble.append_content("First chunk ")
            bubble.append_content("second chunk")
            bubble.finish_streaming()
        """
        try:
            # Create streaming bubble
            bubble = StreamingMessageBubble(
                self,
                on_copy=self._on_message_copied
            )

            # Position in grid
            row = len(self._message_bubbles)
            bubble.grid(
                row=row,
                column=0,
                sticky="ew",
                padx=SPACE_MD,
                pady=(SPACE_SM if row > 0 else SPACE_MD, SPACE_SM)
            )

            # Bind mousewheel to bubble for scroll passthrough
            bubble.bind("<MouseWheel>", self._on_mousewheel)
            bubble.bind("<Button-4>", self._on_mousewheel)
            bubble.bind("<Button-5>", self._on_mousewheel)

            # Store references
            self._message_bubbles.append(bubble)
            self._streaming_bubble = bubble

            # Auto-scroll to bottom
            if self._auto_scroll:
                self.after(50, self.scroll_to_bottom)

            # Update wrap lengths (use after_idle for proper timing)
            self.after_idle(self._update_bubble_widths)

            return bubble

        except Exception as e:
            logger.error(f"Failed to add streaming message: {e}")
            raise

    def add_action_bubble(
        self,
        command: str,
        trust_level: str,
        on_approve: Optional[Callable[[ActionBubble, str], None]] = None,
        on_deny: Optional[Callable[[ActionBubble, str], None]] = None
    ) -> ActionBubble:
        """
        Add an action approval bubble to the chat area.

        Action bubbles are displayed inline with messages and allow
        users to approve or deny system commands that Felix wants to run.

        Args:
            command: The command Felix wants to execute
            trust_level: Trust classification ('safe', 'review', 'blocked')
            on_approve: Callback when Approve is clicked (bubble, command)
            on_deny: Callback when Deny is clicked (bubble, command)

        Returns:
            ActionBubble instance for status updates

        Example:
            bubble = message_area.add_action_bubble(
                command="cat /path/to/file",
                trust_level="review",
                on_approve=handle_approve,
                on_deny=handle_deny
            )
            # Later, update status:
            bubble.set_status(ActionStatus.COMPLETE, output="file contents...")
        """
        try:
            # Create action bubble
            bubble = ActionBubble(
                self,
                command=command,
                trust_level=trust_level,
                on_approve=on_approve,
                on_deny=on_deny
            )

            # Position in grid (same as messages)
            row = len(self._message_bubbles) + len(self._action_bubbles)
            bubble.grid(
                row=row,
                column=0,
                sticky="ew",
                padx=SPACE_MD,
                pady=(SPACE_SM if row > 0 else SPACE_MD, SPACE_SM)
            )

            # Bind mousewheel to bubble for scroll passthrough
            bubble.bind("<MouseWheel>", self._on_mousewheel)
            bubble.bind("<Button-4>", self._on_mousewheel)
            bubble.bind("<Button-5>", self._on_mousewheel)

            # Store reference
            self._action_bubbles.append(bubble)

            # Auto-scroll to bottom
            if self._auto_scroll:
                self.after(50, self.scroll_to_bottom)

            # Update wrap lengths
            self.after_idle(self._update_bubble_widths)

            return bubble

        except Exception as e:
            logger.error(f"Failed to add action bubble: {e}")
            raise

    def add_system_message(self, content: str) -> ctk.CTkFrame:
        """
        Add a system message (command output, errors, etc.) inline in chat.

        System messages are displayed in a compact, monospace format
        to distinguish them from regular chat messages.

        Args:
            content: Message content (can include command output)

        Returns:
            The created frame

        Example:
            message_area.add_system_message("$ pwd\\n/home/user/project")
        """
        try:
            # Create system message frame
            frame = ctk.CTkFrame(
                self,
                fg_color=self.theme_manager.get_color("bg_tertiary"),
                corner_radius=8
            )

            # Position in grid
            row = len(self._message_bubbles) + len(self._action_bubbles)
            frame.grid(
                row=row,
                column=0,
                sticky="ew",
                padx=SPACE_MD,
                pady=(SPACE_SM if row > 0 else SPACE_MD, SPACE_SM)
            )
            frame.grid_columnconfigure(0, weight=1)

            # Add content label (monospace font)
            # Ensure minimum wraplength of 200 to prevent X11 crash
            width = self.winfo_width()
            wrap_width = max(200, width - SPACE_MD * 4) if width > 100 else 400
            label = ctk.CTkLabel(
                frame,
                text=content,
                font=ctk.CTkFont(family="monospace", size=12),
                text_color=self.theme_manager.get_color("fg_secondary"),
                anchor="w",
                justify="left",
                wraplength=wrap_width
            )
            label.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)

            # Bind mousewheel for scroll passthrough
            frame.bind("<MouseWheel>", self._on_mousewheel)
            frame.bind("<Button-4>", self._on_mousewheel)
            frame.bind("<Button-5>", self._on_mousewheel)
            label.bind("<MouseWheel>", self._on_mousewheel)
            label.bind("<Button-4>", self._on_mousewheel)
            label.bind("<Button-5>", self._on_mousewheel)

            # Auto-scroll to bottom
            if self._auto_scroll:
                self.after(50, self.scroll_to_bottom)

            return frame

        except Exception as e:
            logger.error(f"Failed to add system message: {e}")
            raise

    def load_messages(self, messages: List[Dict[str, Any]]):
        """
        Bulk load messages into the chat area.

        Args:
            messages: List of message dictionaries with keys:
                     - role: 'user' or 'assistant'
                     - content: Message text
                     - timestamp: Optional datetime or ISO string
                     - thinking: Optional list of agent thinking steps
                     - knowledge_sources: Optional list of knowledge sources

        Example:
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            message_area.load_messages(messages)
        """
        try:
            # Disable auto-scroll during bulk load
            original_auto_scroll = self._auto_scroll
            self._auto_scroll = False

            for msg_data in messages:
                # Parse timestamp if it's a string
                timestamp = msg_data.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = None

                self.add_message(
                    role=msg_data.get("role", "user"),
                    content=msg_data.get("content", ""),
                    timestamp=timestamp,
                    thinking=msg_data.get("thinking"),
                    knowledge_sources=msg_data.get("knowledge_sources")
                )

            # Restore auto-scroll and scroll to bottom
            self._auto_scroll = original_auto_scroll
            if self._auto_scroll and messages:
                self.after(100, self.scroll_to_bottom)

        except Exception as e:
            logger.error(f"Failed to load messages: {e}")
            # Restore auto-scroll on error
            self._auto_scroll = original_auto_scroll

    def clear(self):
        """Clear all messages and action bubbles from the chat area."""
        try:
            # Destroy all message bubbles
            for bubble in self._message_bubbles:
                bubble.destroy()

            # Destroy all action bubbles
            for bubble in self._action_bubbles:
                bubble.destroy()

            self._message_bubbles.clear()
            self._action_bubbles.clear()
            self._streaming_bubble = None

            # Reset grid row counter
            self.grid_rowconfigure(0, weight=0)

        except Exception as e:
            logger.error(f"Failed to clear messages: {e}")

    def scroll_to_bottom(self):
        """Scroll to the bottom of the message area (latest message)."""
        try:
            # Force update to ensure layout is complete
            self.update_idletasks()

            # Scroll to bottom
            self._parent_canvas.yview_moveto(1.0)

        except Exception as e:
            logger.error(f"Failed to scroll to bottom: {e}")

    def scroll_to_top(self):
        """Scroll to the top of the message area (oldest message)."""
        try:
            self._parent_canvas.yview_moveto(0.0)
        except Exception as e:
            logger.error(f"Failed to scroll to top: {e}")

    def enable_auto_scroll(self, enabled: bool = True):
        """
        Enable or disable auto-scroll to bottom on new messages.

        Args:
            enabled: True to enable auto-scroll, False to disable
        """
        self._auto_scroll = enabled

    def is_at_bottom(self, threshold: float = 0.95) -> bool:
        """
        Check if the scroll position is near the bottom.

        Args:
            threshold: Fraction of scroll position to consider "at bottom" (0.0-1.0)

        Returns:
            True if scrolled near the bottom
        """
        try:
            # Get current scroll position (returns tuple: (top, bottom))
            position = self._parent_canvas.yview()
            return position[1] >= threshold
        except Exception:
            return True  # Default to True if we can't determine

    def get_message_count(self) -> int:
        """
        Get the number of messages currently displayed.

        Returns:
            Number of message bubbles
        """
        return len(self._message_bubbles)

    def get_last_message(self) -> Optional[MessageBubble]:
        """
        Get the most recent message bubble.

        Returns:
            Last MessageBubble or None if no messages
        """
        return self._message_bubbles[-1] if self._message_bubbles else None

    def remove_last_message(self) -> bool:
        """
        Remove the most recent message bubble.

        Returns:
            True if a message was removed, False otherwise
        """
        try:
            if self._message_bubbles:
                last_bubble = self._message_bubbles.pop()
                last_bubble.destroy()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove last message: {e}")
            return False

    def _on_mousewheel(self, event):
        """Handle mouse wheel scroll events."""
        try:
            if event.num == 4:  # Linux scroll up
                self._parent_canvas.yview_scroll(-3, "units")
            elif event.num == 5:  # Linux scroll down
                self._parent_canvas.yview_scroll(3, "units")
            else:  # Windows/MacOS
                delta = -1 * (event.delta // 120) if event.delta else 0
                self._parent_canvas.yview_scroll(delta, "units")
        except Exception as e:
            logger.debug(f"Scroll error: {e}")
        return "break"  # Stop event propagation so parent widgets don't consume it

    def _sync_canvas_bg(self):
        """Sync canvas background color with theme to prevent white bar."""
        try:
            bg_color = self.theme_manager.get_color("bg_primary")
            self._parent_canvas.configure(bg=bg_color)
        except Exception as e:
            logger.debug(f"Could not sync canvas bg: {e}")

    def _on_configure(self, event):
        """
        Handle widget resize events.

        Updates message bubble wrap lengths for responsive layout.
        """
        self._update_bubble_widths()
        self._check_scroll_for_load_more()
        # Fallback canvas bg sync on every configure (handles missed initial sync)
        self._sync_canvas_bg()

    def _update_bubble_widths(self):
        """Update wrap lengths for all message bubbles based on current width.

        Uses debouncing to prevent excessive X11 operations during streaming.
        """
        # Cancel any pending update
        if self._width_update_pending:
            self.after_cancel(self._width_update_pending)
            self._width_update_pending = None

        # Schedule debounced update (100ms delay)
        self._width_update_pending = self.after(100, self._do_update_bubble_widths)

    def _do_update_bubble_widths(self):
        """Actually update wrap lengths (debounced)."""
        self._width_update_pending = None
        try:
            # Get current width
            width = self.winfo_width()

            # Only update if we have a valid width (minimum 300 to prevent X11 crash)
            if width > 300:
                wrap_width = max(200, width - SPACE_MD * 4)

                # Skip if wrap width hasn't changed significantly (within 10px)
                if abs(wrap_width - self._last_wrap_width) < 10:
                    return

                self._last_wrap_width = wrap_width
                for bubble in self._message_bubbles:
                    if hasattr(bubble, "set_wraplength"):
                        bubble.set_wraplength(wrap_width)

        except Exception as e:
            logger.debug(f"Failed to update bubble widths: {e}")

    def _check_scroll_for_load_more(self):
        """
        Check if scrolled to top and trigger load more callback.

        This enables lazy loading of older messages when scrolling up.
        """
        if not self.on_load_more or self._is_loading_more:
            return

        try:
            # Get scroll position
            position = self._parent_canvas.yview()

            # If scrolled to top (within 5%), trigger load more
            if position[0] <= 0.05 and self._message_bubbles:
                self._is_loading_more = True

                # Store current scroll position to restore after loading
                self._scroll_position = position[0]

                # Trigger callback
                self.on_load_more()

                # Reset loading flag after a delay
                self.after(1000, self._reset_loading_flag)

        except Exception as e:
            logger.error(f"Failed to check scroll for load more: {e}")

    def _reset_loading_flag(self):
        """Reset the loading flag to allow subsequent load more triggers."""
        self._is_loading_more = False

    def _on_message_copied(self, content: str):
        """
        Handle message copy events.

        Args:
            content: The copied message content
        """
        logger.debug(f"Message copied ({len(content)} chars)")
        # Could trigger notifications or analytics here

    def _on_theme_change(self, mode: str):
        """
        Handle theme changes.

        Args:
            mode: New theme mode ('dark' or 'light')
        """
        try:
            # Update frame background
            self.configure(fg_color=self.theme_manager.get_color("bg_primary"))

            # Sync canvas background to prevent white bar
            self._sync_canvas_bg()

            # Message bubbles handle their own theme updates via their callbacks

        except Exception as e:
            logger.error(f"Failed to handle theme change: {e}")

    def destroy(self):
        """Cleanup when destroyed."""
        try:
            # Unregister theme callback
            self.theme_manager.unregister_callback(self._on_theme_change)

            # Clear all messages (properly destroys bubbles)
            self.clear()

        except Exception:
            pass  # Ignore errors during cleanup

        super().destroy()
