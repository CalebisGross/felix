"""
Chat Tab for Felix GUI (CustomTkinter Edition)

LM Studio-style chat interface with:
- Sidebar with conversation list and folder organization
- Chat message area with user/assistant bubbles
- Streaming responses with collapsible thinking process view
- Both simple (direct LLM) and workflow (multi-agent) modes
- Knowledge brain integration with user control
- System command execution with trust system approval
"""

from .chat_tab import ChatTab
from .message_bubble import MessageBubble, StreamingMessageBubble
from .thinking_view import ThinkingView, CompactThinkingView
from .chat_session import ChatSession, ChatSessionManager, ChatMessage

__all__ = [
    "ChatTab",
    "ChatSession",
    "ChatSessionManager",
    "ChatMessage",
    "MessageBubble",
    "StreamingMessageBubble",
    "ThinkingView",
    "CompactThinkingView",
]
