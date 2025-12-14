"""
Chat Tab for Felix GUI (CustomTkinter Edition)

Main chat interface with LM Studio-style layout:
- Sidebar with conversation list and folder organization
- Chat message area with user/assistant bubbles
- Streaming responses with collapsible thinking process view
- Both simple (direct LLM) and workflow (multi-agent) modes
- Knowledge brain integration with user control
- System command execution with trust system approval
"""

import customtkinter as ctk
from typing import Optional, Callable, Dict, Any, List, Tuple, Set
import logging
import threading
import re
import time
import uuid
from queue import Queue

from ..base_tab import ResponsiveTab
from ...theme_manager import get_theme_manager
from ...responsive import Breakpoint, BreakpointConfig
from ...components.resizable_separator import ResizableSeparator
from ...styles import (
    BUTTON_SM, BUTTON_MD, BUTTON_ICON,
    FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_CAPTION, FONT_SMALL,
    SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG,
    RADIUS_MD, SIDEBAR_WIDTH
)

# Import chat components
from .chat_session import ChatSession, ChatSessionManager, ChatMessage
from .action_bubble import ActionBubble, ActionStatus
from .prompt_loader import load_system_prompt, get_prompt_info

logger = logging.getLogger(__name__)

# Pattern for detecting system action requests in Felix responses
# Uses ^ anchor with MULTILINE to only match at start of line (not mid-code)
SYSTEM_ACTION_PATTERN = re.compile(
    r'^SYSTEM_ACTION_NEEDED:\s*([^\n]+)',
    re.IGNORECASE | re.MULTILINE
)

# Sidebar width constants
SIDEBAR_COMPACT = 0  # Hidden in compact mode
SIDEBAR_STANDARD = 250
SIDEBAR_WIDE = 300


class ChatTab(ResponsiveTab):
    """
    Chat tab with LM Studio-style interface.

    Responsive layouts:
    - COMPACT: Sidebar hidden, toggle button to show
    - STANDARD: Sidebar 250px, chat area fills rest
    - WIDE/ULTRAWIDE: Sidebar 300px with more features

    Features:
    - Mode selector (Simple/Workflow dropdown)
    - Knowledge brain toggle
    - System command support
    - Connection status to LM Studio
    """

    def __init__(self, master, thread_manager, main_app=None, **kwargs):
        """
        Initialize chat tab.

        Args:
            master: Parent widget
            thread_manager: Thread manager for background work
            main_app: Reference to main application for Felix system access
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, thread_manager, main_app, **kwargs)

        self.theme_manager = get_theme_manager()

        # Session management
        self.session_manager = ChatSessionManager()
        self.current_session: Optional[ChatSession] = None

        # UI state
        self._sidebar_visible = True
        self._current_layout = None
        self._sidebar_width = SIDEBAR_STANDARD

        # Result queue for thread-safe GUI updates
        self._result_queue: Queue = Queue()

        # LLM client reference (from main_app)
        self._llm_client = None
        self._felix_system = None

        # Workflow and approval tracking
        self._current_workflow_id: Optional[str] = None
        self._displayed_approvals: Set[str] = set()
        self._approval_bubbles: Dict[str, ActionBubble] = {}
        self._content_buffer: str = ""  # Accumulated streaming content
        self._is_continuation = False   # Skip pattern detection on continuations
        self._cancel_event: Optional[threading.Event] = None  # For stopping generation
        self._processed_commands: Set[str] = set()  # Track commands already processed this response

        # Cached gap tracking instances (Issue #25 fix - avoid repeated initialization)
        self._gap_tracker = None
        self._gap_trigger = None

        # Components (will be set in _setup_ui)
        self._sidebar = None
        self._message_area = None
        self._input_area = None
        self._separator = None

        # Setup UI
        self._setup_ui()

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

        # Start a new session by default
        self._start_new_session()

        # Register for theme changes
        self.theme_manager.register_callback(self._on_theme_change)

        # Start result polling
        self._poll_results()

        logger.info("Chat tab initialized")

    def _setup_ui(self):
        """Setup the UI components."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=0)  # Sidebar
        self.grid_columnconfigure(1, weight=0)  # Separator
        self.grid_columnconfigure(2, weight=1)  # Chat area
        self.grid_rowconfigure(0, weight=1)

        # Create main container
        self._main_container = ctk.CTkFrame(self, fg_color="transparent")
        self._main_container.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self._main_container.grid_columnconfigure(0, weight=0)  # Sidebar
        self._main_container.grid_columnconfigure(1, weight=0)  # Separator
        self._main_container.grid_columnconfigure(2, weight=1)  # Chat area
        self._main_container.grid_rowconfigure(0, weight=1)

        # Create sidebar
        self._create_sidebar()

        # Create separator
        self._separator = ResizableSeparator(
            self._main_container,
            orientation="vertical",
            on_drag_complete=self._on_separator_drag
        )
        self._separator.grid(row=0, column=1, sticky="ns", padx=1)

        # Create chat area
        self._create_chat_area()

    def _create_sidebar(self):
        """Create the sidebar with conversation list."""
        self._sidebar_frame = ctk.CTkFrame(
            self._main_container,
            width=self._sidebar_width,
            fg_color=self.theme_manager.get_color("bg_secondary")
        )
        self._sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self._sidebar_frame.grid_propagate(False)
        self._sidebar_frame.grid_columnconfigure(0, weight=1)
        self._sidebar_frame.grid_rowconfigure(1, weight=1)

        # Header with new chat button
        header_frame = ctk.CTkFrame(self._sidebar_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=SPACE_SM, pady=SPACE_SM)
        header_frame.grid_columnconfigure(0, weight=1)

        new_chat_btn = ctk.CTkButton(
            header_frame,
            text="+ New Chat",
            font=ctk.CTkFont(size=FONT_BODY),
            height=32,
            command=self._start_new_session
        )
        new_chat_btn.grid(row=0, column=0, sticky="ew")

        new_folder_btn = ctk.CTkButton(
            header_frame,
            text="📁",
            width=32,
            height=32,
            font=ctk.CTkFont(size=FONT_BODY),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            command=self._create_folder
        )
        new_folder_btn.grid(row=0, column=1, sticky="e", padx=(SPACE_XS, 0))

        # Placeholder for sidebar component (will be replaced)
        self._sidebar_placeholder = ctk.CTkLabel(
            self._sidebar_frame,
            text="Conversations will appear here",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self._sidebar_placeholder.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_LG)

        # Try to import and use the Sidebar component
        self._try_load_sidebar()

    def _try_load_sidebar(self):
        """Try to load the Sidebar component if available."""
        try:
            from .sidebar import Sidebar, ConversationItem, FolderItem

            # Remove placeholder
            self._sidebar_placeholder.destroy()

            # Create actual sidebar
            self._sidebar = Sidebar(
                self._sidebar_frame,
                on_select=self._on_conversation_selected,
                on_new_chat=self._start_new_session,
                on_new_folder=self._create_folder
            )
            self._sidebar.grid(row=1, column=0, sticky="nsew")

            # Load existing conversations
            self._refresh_sidebar()

            logger.info("Sidebar component loaded")
        except ImportError as e:
            logger.debug(f"Sidebar component not yet available: {e}")

    def _create_chat_area(self):
        """Create the main chat area."""
        self._chat_frame = ctk.CTkFrame(
            self._main_container,
            fg_color=self.theme_manager.get_color("bg_primary")
        )
        self._chat_frame.grid(row=0, column=2, sticky="nsew")
        self._chat_frame.grid_columnconfigure(0, weight=1)
        self._chat_frame.grid_rowconfigure(1, weight=1)

        # Header with mode selector and knowledge toggle
        self._create_chat_header()

        # Message area (placeholder)
        self._message_frame = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        self._message_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=SPACE_SM)
        self._message_frame.grid_columnconfigure(0, weight=1)
        self._message_frame.grid_rowconfigure(0, weight=1)

        # Try to load MessageArea component
        self._try_load_message_area()

        # Input area (placeholder)
        self._input_frame = ctk.CTkFrame(self._chat_frame, fg_color="transparent")
        self._input_frame.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._input_frame.grid_columnconfigure(0, weight=1)

        # Try to load InputArea component
        self._try_load_input_area()

    def _create_chat_header(self):
        """Create the chat header with controls."""
        header_frame = ctk.CTkFrame(
            self._chat_frame,
            fg_color=self.theme_manager.get_color("bg_secondary"),
            height=50
        )
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_propagate(False)
        header_frame.grid_columnconfigure(1, weight=1)

        # Sidebar toggle (for compact mode)
        self._sidebar_toggle = ctk.CTkButton(
            header_frame,
            text="☰",
            width=32,
            height=32,
            font=ctk.CTkFont(size=FONT_SECTION),
            fg_color="transparent",
            hover_color=self.theme_manager.get_color("bg_hover"),
            command=self._toggle_sidebar
        )
        self._sidebar_toggle.grid(row=0, column=0, padx=SPACE_SM, pady=SPACE_SM)
        self._sidebar_toggle.grid_remove()  # Hidden by default (shown in compact mode)

        # Mode selector
        mode_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        mode_frame.grid(row=0, column=1, sticky="w", padx=SPACE_SM, pady=SPACE_SM)

        mode_label = ctk.CTkLabel(
            mode_frame,
            text="Mode:",
            font=ctk.CTkFont(size=FONT_CAPTION),
            text_color=self.theme_manager.get_color("fg_secondary")
        )
        mode_label.grid(row=0, column=0, padx=(0, SPACE_XS))

        self._mode_selector = ctk.CTkOptionMenu(
            mode_frame,
            values=["Simple", "Workflow"],
            width=100,
            height=28,
            font=ctk.CTkFont(size=FONT_CAPTION),
            command=self._on_mode_changed
        )
        self._mode_selector.grid(row=0, column=1)
        self._mode_selector.set("Simple")

        # Knowledge brain toggle
        self._knowledge_var = ctk.BooleanVar(value=True)
        self._knowledge_toggle = ctk.CTkSwitch(
            header_frame,
            text="Knowledge",
            font=ctk.CTkFont(size=FONT_CAPTION),
            variable=self._knowledge_var,
            command=self._on_knowledge_toggled,
            width=40
        )
        self._knowledge_toggle.grid(row=0, column=2, padx=SPACE_MD, pady=SPACE_SM)

        # Gap indicator (Issue #25)
        self._gap_indicator_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        self._gap_indicator_frame.grid(row=0, column=3, padx=SPACE_SM, pady=SPACE_SM)

        self._gap_indicator_label = ctk.CTkLabel(
            self._gap_indicator_frame,
            text="",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self._gap_indicator_label.grid(row=0, column=0)
        self._gap_indicator_label.grid_remove()  # Hidden by default

        # Connection status (starts disconnected until Felix is running)
        self._status_label = ctk.CTkLabel(
            header_frame,
            text="○ Start Felix system",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self._status_label.grid(row=0, column=4, padx=SPACE_MD, pady=SPACE_SM, sticky="e")

    def _try_load_message_area(self):
        """Try to load the MessageArea component if available."""
        try:
            from .message_area import MessageArea

            self._message_area = MessageArea(
                self._message_frame,
                on_load_more=self._load_more_messages
            )
            self._message_area.grid(row=0, column=0, sticky="nsew")

            logger.info("MessageArea component loaded")
        except ImportError as e:
            logger.debug(f"MessageArea component not yet available: {e}")

            # Fallback placeholder
            placeholder = ctk.CTkLabel(
                self._message_frame,
                text="Messages will appear here\n\nStart typing below to chat with Felix",
                font=ctk.CTkFont(size=FONT_BODY),
                text_color=self.theme_manager.get_color("fg_muted"),
                justify="center"
            )
            placeholder.grid(row=0, column=0, sticky="nsew")

    def _try_load_input_area(self):
        """Try to load the InputArea component if available."""
        try:
            from .input_area import InputArea

            self._input_area = InputArea(
                self._input_frame,
                on_send=self._on_send_message,
                on_stop=self._on_stop_generation
            )
            self._input_area.grid(row=0, column=0, sticky="ew")

            logger.info("InputArea component loaded")
        except ImportError as e:
            logger.debug(f"InputArea component not yet available: {e}")

            # Fallback: Simple input
            fallback_frame = ctk.CTkFrame(self._input_frame, fg_color="transparent")
            fallback_frame.grid(row=0, column=0, sticky="ew")
            fallback_frame.grid_columnconfigure(0, weight=1)

            self._fallback_input = ctk.CTkTextbox(
                fallback_frame,
                height=60,
                font=ctk.CTkFont(size=FONT_BODY)
            )
            self._fallback_input.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_SM))
            self._fallback_input.bind("<Control-Return>", lambda e: self._on_send_message_fallback())
            self._fallback_input.bind("<Command-Return>", lambda e: self._on_send_message_fallback())

            send_btn = ctk.CTkButton(
                fallback_frame,
                text="Send",
                width=60,
                height=60,
                command=self._on_send_message_fallback
            )
            send_btn.grid(row=0, column=1)

    def _on_send_message_fallback(self):
        """Fallback send handler when InputArea not available."""
        if hasattr(self, '_fallback_input'):
            content = self._fallback_input.get("1.0", "end-1c").strip()
            if content:
                self._on_send_message(content)
                self._fallback_input.delete("1.0", "end")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def _start_new_session(self):
        """Start a new chat session."""
        # Reset input area streaming state when starting new session
        if self._input_area:
            self._input_area.set_streaming(False)

        # Clear workflow and approval state
        self._current_workflow_id = None
        self._displayed_approvals.clear()
        self._approval_bubbles.clear()
        self._content_buffer = ""

        self.current_session = self.session_manager.create_session()

        # Update mode and knowledge settings
        mode = self._mode_selector.get().lower()
        self.current_session.mode = mode
        self.current_session.knowledge_enabled = self._knowledge_var.get()

        # Clear message area
        if self._message_area:
            self._message_area.clear()

        # Refresh sidebar
        self._refresh_sidebar()

        logger.info(f"Started new session: {self.current_session.session_id}")

    def _on_conversation_selected(self, session_id: str):
        """Handle conversation selection from sidebar."""
        if self.current_session and self.current_session.session_id == session_id:
            return  # Already selected

        # Reset input area streaming state when switching conversations
        if self._input_area:
            self._input_area.set_streaming(False)

        # Load the selected session
        self.current_session = self.session_manager.load_session(session_id)

        # Update UI to match session settings
        self._mode_selector.set(self.current_session.mode.title())
        self._knowledge_var.set(self.current_session.knowledge_enabled)

        # Load messages into the message area
        if self._message_area:
            self._message_area.clear()
            for msg in self.current_session.messages:
                self._message_area.add_message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    thinking=msg.thinking,
                    knowledge_sources=msg.knowledge_sources
                )

        logger.info(f"Loaded session: {session_id}")

    def _create_folder(self):
        """Create a new folder."""
        # Simple dialog for folder name
        dialog = ctk.CTkInputDialog(
            text="Enter folder name:",
            title="New Folder"
        )
        name = dialog.get_input()

        if name:
            try:
                self.session_manager.create_folder(name)
                self._refresh_sidebar()
                logger.info(f"Created folder: {name}")
            except Exception as e:
                logger.error(f"Failed to create folder: {e}")

    def _refresh_sidebar(self):
        """Refresh the sidebar with current sessions and folders."""
        if not self._sidebar:
            return

        try:
            conversations = self.session_manager.list_sessions()
            folders = self.session_manager.list_folders()

            # Import dataclasses if sidebar is available
            from .sidebar import ConversationItem, FolderItem

            conv_items = [
                ConversationItem(
                    session_id=c['session_id'],
                    title=c['title'],
                    last_active=c['last_active'],
                    message_count=c['message_count'],
                    pinned=c.get('pinned', False),
                    folder_id=c.get('folder_id')
                )
                for c in conversations
            ]

            folder_items = [
                FolderItem(
                    folder_id=f['folder_id'],
                    name=f['name'],
                    parent_id=f.get('parent_id')
                )
                for f in folders
            ]

            self._sidebar.load_conversations(conv_items, folder_items)

        except Exception as e:
            logger.debug(f"Could not refresh sidebar: {e}")

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    def _on_send_message(self, content: str):
        """
        Handle sending a message.

        Args:
            content: Message content to send
        """
        if not content.strip():
            return

        if not self.current_session:
            self._start_new_session()

        # Add user message
        user_msg = self.current_session.add_user_message(content)

        # Display in message area
        if self._message_area:
            self._message_area.add_message(
                role='user',
                content=content,
                timestamp=user_msg.timestamp
            )
            self._message_area.scroll_to_bottom()

        # Set streaming mode (input already cleared by InputArea._on_send_clicked)
        if self._input_area:
            self._input_area.set_streaming(True)

        # Reset continuation flag - new user message = fresh start
        self._is_continuation = False

        # Process based on mode
        mode = self.current_session.mode
        if mode == 'simple':
            self._process_simple_mode(content)
        else:
            self._process_workflow_mode(content)

    def _process_simple_mode(self, content: str):
        """Process message in simple (direct Felix) mode.

        Uses FelixAgent for direct inference - Felix responds AS Felix,
        not as the raw underlying LLM. This maintains Felix's identity
        while being lightweight (no multi-agent orchestration).
        """
        # Reset state for new response (prevents loops from stale state)
        self._content_buffer = ""
        self._is_continuation = False
        self._processed_commands.clear()

        # Update status to sending
        self._update_connection_status("sending")

        # Start streaming message
        streaming_msg = self.current_session.start_assistant_message()

        # Add streaming bubble to UI
        streaming_bubble = None
        if self._message_area:
            streaming_bubble = self._message_area.add_streaming_message()

        # Get Felix system from main app
        felix_system = self._get_felix_system()

        if not felix_system:
            error_content = "Felix system not available. Please start the system first."
            self.current_session.append_to_streaming(error_content)
            self.current_session.finish_streaming()

            if streaming_bubble:
                streaming_bubble.update_content(error_content)
                streaming_bubble.finish_streaming()

            if self._input_area:
                self._input_area.set_streaming(False)
            return

        # Get conversation history for context
        conversation_history = self.current_session.get_context_messages(limit=10)

        # Create cancel event for this generation
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event  # Local reference for closure

        # Run in background thread
        def process():
            try:
                from src.workflows.felix_inference import run_felix

                # Streaming callback for direct mode
                def on_chunk(chunk_text):
                    # Check cancellation before queuing
                    if cancel_event.is_set():
                        return
                    self._result_queue.put(('chunk', chunk_text))

                # Run Felix in direct mode
                result = run_felix(
                    felix_system=felix_system,
                    user_input=content,
                    mode="direct",
                    streaming_callback=on_chunk,
                    knowledge_enabled=self.current_session.knowledge_enabled,
                    conversation_history=conversation_history,
                    cancel_event=cancel_event
                )

                # Signal completion
                self._result_queue.put(('done', None))

            except Exception as e:
                logger.error(f"Felix error: {e}")
                self._result_queue.put(('error', str(e)))
            finally:
                # Thread-safe GUI update: force reset streaming state
                self.after_idle(self._force_reset_streaming)

        self.thread_manager.start_thread(process)

    def _process_workflow_mode(self, content: str):
        """Process message in workflow (multi-agent) mode.

        Uses FelixAgent for full orchestration - Felix spawns specialist
        agents (Research, Analysis, Critic) and synthesizes their outputs.
        The response still comes FROM Felix as the unified identity.
        """
        # Reset state for new response (prevents loops from stale state)
        self._content_buffer = ""
        self._is_continuation = False
        self._processed_commands.clear()

        # Update status to sending
        self._update_connection_status("sending")

        # Start streaming message
        streaming_msg = self.current_session.start_assistant_message()

        # Add streaming bubble to UI
        streaming_bubble = None
        if self._message_area:
            streaming_bubble = self._message_area.add_streaming_message()

        # Get Felix system from main app
        felix_system = self._get_felix_system()

        if not felix_system:
            error_content = "Felix system not available. Please start the system first."
            self.current_session.append_to_streaming(error_content)
            self.current_session.finish_streaming()

            if streaming_bubble:
                streaming_bubble.update_content(error_content)
                streaming_bubble.finish_streaming()

            if self._input_area:
                self._input_area.set_streaming(False)
            return

        # Get conversation history for context
        conversation_history = self.current_session.get_context_messages(limit=10)

        # Create cancel event for this generation
        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event  # Local reference for closure

        # Run workflow in background
        def process():
            try:
                from src.workflows.felix_inference import run_felix

                # Streaming callback for workflow mode (receives agent name and chunk)
                def on_thinking(agent_name, chunk_text):
                    # Check cancellation before queuing
                    if cancel_event.is_set():
                        return
                    self._result_queue.put(('thinking', (agent_name, chunk_text)))

                # Run Felix in full workflow mode
                result = run_felix(
                    felix_system=felix_system,
                    user_input=content,
                    mode="full",
                    streaming_callback=on_thinking,
                    knowledge_enabled=self.current_session.knowledge_enabled,
                    conversation_history=conversation_history,
                    cancel_event=cancel_event
                )

                # Extract final content and confidence
                final_content = result.get('content', 'No response generated.')
                confidence = result.get('confidence', 0.0)

                self._result_queue.put(('workflow_done', (final_content, confidence)))

            except Exception as e:
                logger.error(f"Workflow error: {e}")
                self._result_queue.put(('error', str(e)))
            finally:
                # Thread-safe GUI update: force reset streaming state
                self.after_idle(self._force_reset_streaming)

        self.thread_manager.start_thread(process)

    def _force_reset_streaming(self):
        """Force reset streaming state - called from background thread via after_idle."""
        if self._input_area:
            self._input_area.set_streaming(False)
        self._update_connection_status("connected")

    def _on_stop_generation(self):
        """Stop the current generation."""
        # Signal cancellation to background thread
        if self._cancel_event:
            self._cancel_event.set()
            logger.info("Cancellation signaled to background thread")

        if self.current_session and self.current_session.state.is_streaming:
            self.current_session.cancel_streaming()

            if self._input_area:
                self._input_area.set_streaming(False)

            # Drain the result queue to prevent stale data
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except:
                    break

            # Finalize any streaming bubble with stopped message
            if self._message_area and self._message_area._streaming_bubble:
                self._message_area._streaming_bubble.append_content("\n\n*[Generation stopped by user]*")
                self._message_area._streaming_bubble.finish_streaming()

            # Clear content buffer
            self._content_buffer = ""

            logger.info("Generation stopped by user")

    def _poll_results(self):
        """Poll the result queue for updates and check for pending approvals."""
        try:
            # Skip processing if cancelled - just drain the queue
            if self._cancel_event and self._cancel_event.is_set():
                while not self._result_queue.empty():
                    try:
                        self._result_queue.get_nowait()
                    except:
                        break
                # Schedule next poll but don't process
                self.after(50, self._poll_results)
                return

            while not self._result_queue.empty():
                msg_type, data = self._result_queue.get_nowait()

                if msg_type == 'chunk':
                    # Update status to streaming on first chunk
                    self._update_connection_status("streaming")

                    # Just accumulate and display - NO pattern detection during streaming
                    # Pattern detection happens AFTER streaming completes (in 'done' handler)
                    # This prevents false positives from partial/incomplete content
                    self._content_buffer += data

                    # Append chunk to UI (incremental, no re-render - fixes X11 BadAlloc)
                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.append_content(data)

                    # Update session with full buffer for persistence
                    if self.current_session:
                        if self.current_session.state.current_streaming_message:
                            self.current_session.state.current_streaming_message.content = self._content_buffer

                elif msg_type == 'thinking':
                    agent, content = data
                    if self.current_session:
                        self.current_session.add_thinking_step(agent, content)

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.add_thinking_step(agent, content)

                elif msg_type == 'done':
                    # Finish streaming (simple mode)
                    self._update_connection_status("connected")

                    # NOW run pattern detection on COMPLETE content
                    # This prevents false positives from partial/incomplete content during streaming
                    # Skip pattern detection on continuation responses (prevents loops)
                    if self._content_buffer and not self._is_continuation:
                        cleaned, actions = self._extract_system_actions(self._content_buffer)

                        if actions:
                            # Update bubble with cleaned content (patterns removed)
                            if self._message_area and self._message_area._streaming_bubble:
                                self._message_area._streaming_bubble.set_content(cleaned)

                            # Process detected actions (with deduplication to prevent loops)
                            for action in actions:
                                cmd = action['command']
                                if cmd not in self._processed_commands:
                                    self._processed_commands.add(cmd)
                                    self._handle_detected_action(cmd)

                            # Update session with cleaned content
                            if self.current_session:
                                if self.current_session.state.current_streaming_message:
                                    self.current_session.state.current_streaming_message.content = cleaned

                    # NOTE: Do NOT reset _content_buffer or _is_continuation here!
                    # They are reset at the start of _process_simple_mode/_process_workflow_mode
                    # Resetting here causes race conditions with continuations (Issue #loop-fix)

                    if self.current_session:
                        self.current_session.finish_streaming()

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.finish_streaming()

                    if self._input_area:
                        self._input_area.set_streaming(False)

                elif msg_type == 'workflow_done':
                    final_content, confidence = data
                    self._update_connection_status("connected")
                    # NOTE: Buffer reset moved to start of _process_workflow_mode

                    # Check final content for patterns too (with deduplication)
                    cleaned, actions = self._extract_system_actions(final_content)
                    if actions:
                        for action in actions:
                            cmd = action['command']
                            if cmd not in self._processed_commands:
                                self._processed_commands.add(cmd)
                                self._handle_detected_action(cmd)
                        final_content = cleaned

                    if self.current_session:
                        self.current_session.append_to_streaming(final_content)
                        self.current_session.finish_streaming(confidence=confidence)

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.update_content(final_content)
                        self._message_area._streaming_bubble.finish_streaming()

                    if self._input_area:
                        self._input_area.set_streaming(False)

                elif msg_type == 'error':
                    error_msg = f"Error: {data}"
                    self._update_connection_status("error", f"● Error: {data[:30]}")
                    # NOTE: Buffer reset moved to start of processing methods

                    if self.current_session:
                        self.current_session.append_to_streaming(error_msg)
                        self.current_session.finish_streaming()

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.update_content(error_msg)
                        self._message_area._streaming_bubble.finish_streaming()

                    if self._input_area:
                        self._input_area.set_streaming(False)

            # Poll for pending approval requests from Felix
            self._poll_approval_messages()

            # Poll for gap status updates (Issue #25) - less frequently
            if not hasattr(self, '_last_gap_poll') or time.time() - self._last_gap_poll > 5.0:
                self._poll_gap_status()
                self._last_gap_poll = time.time()

        except Exception as e:
            logger.error(f"Error polling results: {e}")
            # Reset streaming state on error to prevent stuck input
            if self._input_area:
                self._input_area.set_streaming(False)

        # Schedule next poll (50ms for faster response)
        self.after(50, self._poll_results)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _poll_gap_status(self):
        """
        Poll for knowledge gap status and update indicator (Issue #25).

        Checks GapAcquisitionTrigger for high-priority gaps and updates
        the gap indicator in the chat header.
        """
        try:
            # Lazy initialization - only create instances once
            if self._gap_trigger is None:
                from src.knowledge.gap_directed_learning import GapAcquisitionTrigger
                from src.knowledge.gap_tracker import GapTracker

                self._gap_tracker = GapTracker()
                self._gap_trigger = GapAcquisitionTrigger(self._gap_tracker)

            # Get gap summary using cached instances
            summary = self._gap_trigger.get_gap_summary()

            # Update indicator
            self._update_gap_indicator(summary)

        except ImportError:
            # Gap system not available - hide indicator
            self._gap_indicator_label.grid_remove()
        except Exception as e:
            logger.debug(f"Gap status polling failed: {e}")
            self._gap_indicator_label.grid_remove()

    def _update_gap_indicator(self, summary: Dict[str, Any]):
        """
        Update the gap indicator in the chat header (Issue #25).

        Args:
            summary: Gap summary dict from GapAcquisitionTrigger.get_gap_summary()
        """
        high_priority = summary.get('high_priority_gaps', 0)
        total_active = summary.get('total_active_gaps', 0)

        if high_priority > 0:
            # Show warning indicator for high-priority gaps
            self._gap_indicator_label.configure(
                text=f"⚠ {high_priority} gap{'s' if high_priority > 1 else ''}",
                text_color=self.theme_manager.get_color("warning")
            )
            self._gap_indicator_label.grid()

            # Update tooltip-like behavior via hover
            self._gap_indicator_label.bind("<Enter>", lambda e: self._show_gap_tooltip(summary))
            self._gap_indicator_label.bind("<Leave>", lambda e: self._hide_gap_tooltip())

        elif total_active > 0:
            # Show informational indicator for any active gaps
            self._gap_indicator_label.configure(
                text=f"○ {total_active} gap{'s' if total_active > 1 else ''}",
                text_color=self.theme_manager.get_color("fg_muted")
            )
            self._gap_indicator_label.grid()
        else:
            # No gaps - hide indicator
            self._gap_indicator_label.grid_remove()

    def _show_gap_tooltip(self, summary: Dict[str, Any]):
        """Show tooltip with gap details on hover."""
        top_gaps = summary.get('top_gaps', [])
        if not top_gaps:
            return

        # Build tooltip text
        tooltip_lines = ["Knowledge Gaps:"]
        for gap in top_gaps[:3]:
            domain = gap.get('domain', 'unknown')
            concept = gap.get('concept', '(general)')
            severity = gap.get('severity', 0)
            tooltip_lines.append(f"  • {domain}/{concept} ({severity:.0%})")

        # Update label with full text temporarily
        self._gap_indicator_label.configure(
            text="\n".join(tooltip_lines)
        )

    def _hide_gap_tooltip(self):
        """Hide tooltip and restore normal gap indicator."""
        # Restore normal indicator by re-polling
        self._poll_gap_status()

    def _update_connection_status(self, state: str, message: str = None):
        """
        Update the connection status indicator.

        Args:
            state: One of 'connected', 'disconnected', 'error', 'streaming', 'sending'
            message: Optional custom message to display
        """
        colors = {
            "connected": ("success", "● Connected"),
            "disconnected": ("fg_muted", "○ Disconnected"),
            "error": ("error", "● Error"),
            "streaming": ("warning", "● Streaming..."),
            "sending": ("warning", "● Sending..."),
        }
        color_key, default_text = colors.get(state, ("fg_muted", "○ Unknown"))
        self._status_label.configure(
            text=message or default_text,
            text_color=self.theme_manager.get_color(color_key)
        )

    # =========================================================================
    # SYSTEM ACTION HANDLING
    # =========================================================================

    def _extract_system_actions(self, content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract SYSTEM_ACTION_NEEDED patterns from content.

        Args:
            content: Text content that may contain action patterns

        Returns:
            Tuple of (cleaned_content, list_of_actions)
            Each action is a dict with 'command' and 'span' keys
        """
        actions = []

        for match in SYSTEM_ACTION_PATTERN.finditer(content):
            command = match.group(1).strip()

            # Strip surrounding quotes if present
            if len(command) >= 2 and command[0] in '"\'':
                if command[-1] == command[0]:
                    command = command[1:-1]

            actions.append({
                "command": command,
                "span": match.span()
            })

        # Remove patterns from content
        cleaned = SYSTEM_ACTION_PATTERN.sub('', content)

        # Clean up extra whitespace/newlines left behind
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned).strip()

        return cleaned, actions

    def _poll_approval_messages(self):
        """Poll for pending approval requests from Felix's SystemCommandManager."""
        felix_system = self._get_felix_system()
        if not felix_system:
            return

        # Get SystemCommandManager
        scm = getattr(felix_system, 'system_command_manager', None)
        if not scm:
            return

        try:
            # Get pending actions for current workflow
            pending = scm.get_pending_actions(workflow_id=self._current_workflow_id)

            for action in pending:
                approval_id = action.get('approval_id')
                if not approval_id:
                    continue

                # Skip if already displayed
                if approval_id in self._displayed_approvals:
                    continue

                # Mark as displayed
                self._displayed_approvals.add(approval_id)

                # Show approval bubble
                self._show_approval_bubble(action)

        except Exception as e:
            logger.debug(f"Error polling approvals: {e}")

    def _show_approval_bubble(self, action: dict):
        """Display an ActionBubble for a pending approval from Felix."""
        if not self._message_area:
            return

        approval_id = action.get('approval_id')
        command = action.get('command', '')
        trust_level = action.get('trust_level', 'review')

        # Convert TrustLevel enum to string if needed
        if hasattr(trust_level, 'value'):
            trust_level = trust_level.value

        logger.info(f"Showing approval bubble: {command[:50]}... (approval_id: {approval_id})")

        # Create the action bubble with approval_id bound to callbacks
        bubble = self._message_area.add_action_bubble(
            command=command,
            trust_level=trust_level,
            on_approve=lambda b, cmd: self._approve_felix_action(approval_id, b),
            on_deny=lambda b, cmd: self._deny_felix_action(approval_id, b)
        )

        # Track the bubble
        self._approval_bubbles[approval_id] = bubble

    def _handle_detected_action(self, command: str):
        """
        Route a detected SYSTEM_ACTION_NEEDED pattern through Felix.

        Instead of executing directly via subprocess, we route through
        Felix's SystemCommandManager which handles trust classification,
        approval workflow, and proper execution.

        Flow by trust level:
        - SAFE: Auto-executes immediately, result shown inline
        - REVIEW: Creates pending approval, picked up by _poll_approval_messages()
        - BLOCKED: Denied immediately, result shown inline

        Args:
            command: The command detected in the LLM response
        """
        felix_system = self._get_felix_system()
        if not felix_system:
            logger.warning("Cannot route action: Felix system not available")
            self._show_system_message(f"Cannot execute '{command}': Felix system not running")
            return

        # Get SystemCommandManager
        scm = getattr(felix_system, 'system_command_manager', None)
        if not scm:
            logger.warning("Cannot route action: SystemCommandManager not available")
            self._show_system_message(f"Cannot execute '{command}': SystemCommandManager not available")
            return

        # Create workflow ID if not set (for Simple mode)
        if not self._current_workflow_id:
            self._current_workflow_id = f"chat_{uuid.uuid4().hex[:8]}"
            # Set workflow on CentralPost if available
            central_post = getattr(felix_system, 'central_post', None)
            if central_post and hasattr(central_post, 'set_current_workflow'):
                central_post.set_current_workflow(self._current_workflow_id)

        # Run in background thread to not block UI
        def do_request():
            try:
                # Request action through Felix's proper channel
                # This will classify trust, create approval if needed, or auto-execute if SAFE
                action_id = scm.request_system_action(
                    agent_id="chat_user",
                    command=command,
                    context="Chat session request",
                    workflow_id=self._current_workflow_id,
                    cwd=self._get_project_root()
                )

                logger.info(f"Routed action through Felix: {action_id}")

                # Check what happened:
                # 1. SAFE → executed immediately, result available
                # 2. REVIEW → pending approval created (picked up by _poll_approval_messages)
                # 3. BLOCKED → denial result available

                result = scm.get_action_result(action_id)
                if result:
                    # SAFE or BLOCKED - show result inline (no bubble needed)
                    self.after(0, lambda r=result: self._show_command_result(command, r))
                # else: REVIEW - _poll_approval_messages() will pick it up and show bubble

            except Exception as e:
                logger.error(f"Error routing action through Felix: {e}")
                # Capture exception and command as default args to avoid closure bug
                self.after(0, lambda err=str(e), cmd=command: self._show_system_message(f"Error executing '{cmd}': {err}"))

        thread = threading.Thread(target=do_request, daemon=True)
        thread.start()

    def _show_command_result(self, command: str, result):
        """
        Show command execution result inline in the CURRENT streaming bubble.

        Appends the command output directly to the existing Felix response,
        creating a seamless Claude Code-like experience where commands and
        their results appear inline within the assistant's message.

        Args:
            command: The command that was executed
            result: CommandResult from Felix's execution
        """
        # Extract result data
        success = getattr(result, 'success', False)
        stdout = getattr(result, 'stdout', '') or ''
        stderr = getattr(result, 'stderr', '') or ''
        exit_code = getattr(result, 'exit_code', -1)

        # Format result as inline code block
        if success:
            output = stdout.strip() if stdout.strip() else "(no output)"
            inline_result = f"\n\n```\n$ {command}\n{output}\n```\n"
        else:
            error_msg = stderr.strip() if stderr.strip() else "(command failed)"
            inline_result = f"\n\n```\n$ {command}\nFailed (exit {exit_code}): {error_msg}\n```\n"

        # Append to EXISTING streaming bubble (not a separate system message)
        if self._message_area and self._message_area._streaming_bubble:
            self._content_buffer += inline_result
            self._message_area._streaming_bubble.update_content(self._content_buffer)

            # Also update session content
            if self.current_session and self.current_session.state.current_streaming_message:
                self.current_session.state.current_streaming_message.content = self._content_buffer

        logger.info(f"Appended command result inline: {command[:30]}... success={success}")

        # Feed result back to Felix for continued response
        result_dict = {
            'success': success,
            'stdout': stdout,
            'stderr': stderr,
            'exit_code': exit_code
        }
        self._feed_result_to_felix(command, result_dict)

    def _show_system_message(self, message: str):
        """
        Show a system/error message in chat.

        Args:
            message: Message to display
        """
        if self._message_area:
            self._message_area.add_system_message(message)
        logger.info(f"System message: {message[:50]}...")

    def _approve_felix_action(self, approval_id: str, bubble: ActionBubble):
        """
        Approve an action through Felix's approval system.

        Args:
            approval_id: The approval ID from Felix's ApprovalManager
            bubble: The ActionBubble to update
        """
        bubble.set_status(ActionStatus.EXECUTING)

        felix_system = self._get_felix_system()
        if not felix_system:
            bubble.set_status(ActionStatus.DENIED, output="Felix system not available")
            return

        scm = getattr(felix_system, 'system_command_manager', None)
        if not scm:
            bubble.set_status(ActionStatus.DENIED, output="SystemCommandManager not available")
            return

        # Capture command for result feedback
        command = bubble.command

        # Run approval in background thread (this unblocks waiting workflow)
        def do_approve():
            try:
                # Import ApprovalDecision
                try:
                    from src.execution.approval_manager import ApprovalDecision
                    decision = ApprovalDecision.APPROVE_ONCE
                except ImportError:
                    decision = "approve_once"

                # Approve through Felix's system (returns bool, not CommandResult)
                success = scm.approve_system_action(
                    approval_id=approval_id,
                    decision=decision,
                    decided_by="chat_user"
                )

                if success:
                    # Get actual result after execution
                    result = scm.get_action_result(approval_id)
                    if result:
                        stdout = getattr(result, 'stdout', '') or ''
                        stderr = getattr(result, 'stderr', '') or ''
                        exit_code = getattr(result, 'exit_code', 0)
                        output = stdout.strip() if stdout.strip() else (stderr.strip() if stderr.strip() else "(no output)")
                    else:
                        stdout, stderr, exit_code = '', '', 0
                        output = "(command completed)"

                    self.after(0, lambda: bubble.set_status(
                        ActionStatus.COMPLETE,
                        output=output,
                        exit_code=exit_code
                    ))

                    # Feed result back to Felix for continuation
                    result_dict = {
                        'success': exit_code == 0,
                        'stdout': stdout,
                        'stderr': stderr,
                        'exit_code': exit_code
                    }
                    self.after(0, lambda: self._feed_result_to_felix(command, result_dict))
                else:
                    self.after(0, lambda: bubble.set_status(
                        ActionStatus.DENIED,
                        output="Approval failed - see logs for details"
                    ))

            except Exception as e:
                logger.error(f"Error approving action: {e}")
                self.after(0, lambda: bubble.set_status(
                    ActionStatus.DENIED,
                    output=f"Error: {e}"
                ))

        thread = threading.Thread(target=do_approve, daemon=True)
        thread.start()

    def _deny_felix_action(self, approval_id: str, bubble: ActionBubble):
        """
        Deny an action through Felix's approval system.

        Args:
            approval_id: The approval ID from Felix's ApprovalManager
            bubble: The ActionBubble to update
        """
        felix_system = self._get_felix_system()
        if felix_system:
            scm = getattr(felix_system, 'system_command_manager', None)
            if scm and hasattr(scm, 'deny_system_action'):
                try:
                    scm.deny_system_action(approval_id, reason="Denied by user in chat")
                except Exception as e:
                    logger.error(f"Error denying action: {e}")

        bubble.set_status(ActionStatus.DENIED)
        logger.info(f"Denied action: {approval_id}")

    def _handle_felix_approval_result(self, bubble: ActionBubble, result):
        """
        Handle the result of a Felix approval/execution.

        Args:
            bubble: The ActionBubble to update
            result: CommandResult from Felix's execution
        """
        if result is None:
            bubble.set_status(ActionStatus.DENIED, output="No result returned")
            return

        # Extract output
        stdout = getattr(result, 'stdout', '') or ''
        stderr = getattr(result, 'stderr', '') or ''
        exit_code = getattr(result, 'exit_code', 0)
        success = getattr(result, 'success', exit_code == 0)

        # Combine output
        output = stdout
        if stderr:
            if output:
                output += '\n--- stderr ---\n'
            output += stderr

        # Update bubble
        bubble.set_status(
            ActionStatus.COMPLETE,
            output=output or "(no output)",
            exit_code=exit_code
        )

        logger.info(f"Action completed: exit_code={exit_code}, success={success}")

    def _approve_local_action(self, local_id: str, bubble: ActionBubble, command: str):
        """
        Handle approval of a locally-detected action (when Felix wasn't available initially).

        Tries to route through Felix. If Felix is now available, uses proper approval flow.
        If still unavailable, shows error - no bypass execution allowed.

        Args:
            local_id: Local tracking ID for the action
            bubble: The ActionBubble that was approved
            command: The command to execute
        """
        logger.info(f"Local action approved: {command[:50]}...")

        # Try to route through Felix's SystemCommandManager
        felix_system = self._get_felix_system()
        if not felix_system:
            bubble.set_status(ActionStatus.DENIED, output="Felix system not available. Cannot execute commands.")
            return

        scm = getattr(felix_system, 'system_command_manager', None)
        if not scm:
            bubble.set_status(ActionStatus.DENIED, output="SystemCommandManager not available. Cannot execute commands.")
            return

        bubble.set_status(ActionStatus.EXECUTING)

        # Create workflow ID if needed
        if not self._current_workflow_id:
            self._current_workflow_id = f"chat_{uuid.uuid4().hex[:8]}"

        # Run through Felix in background thread
        def do_execute():
            try:
                # Request through Felix (this handles trust classification)
                action_id = scm.request_system_action(
                    agent_id="chat_user",
                    command=command,
                    context="Chat session - user approved local action",
                    workflow_id=self._current_workflow_id,
                    cwd=self._get_project_root()
                )

                # For SAFE commands, request_system_action executes immediately
                # Check if result is already available
                result = scm.get_action_result(action_id)
                if result:
                    # SAFE command executed
                    self.after(0, lambda: self._handle_felix_approval_result(bubble, result))
                    return

                # For REVIEW commands, we need to approve since user already clicked approve
                pending = scm.get_pending_actions(workflow_id=self._current_workflow_id)
                for action in pending:
                    if action.get('command') == command:
                        # Import ApprovalDecision
                        try:
                            from src.execution.approval_manager import ApprovalDecision
                            decision = ApprovalDecision.APPROVE_ONCE
                        except ImportError:
                            decision = "approve_once"

                        success = scm.approve_system_action(
                            approval_id=action['approval_id'],
                            decision=decision,
                            decided_by="chat_user"
                        )

                        if success:
                            # Get the result after approval
                            result = scm.get_action_result(action_id)
                            if result:
                                self.after(0, lambda r=result: self._handle_felix_approval_result(bubble, r))
                            else:
                                self.after(0, lambda: bubble.set_status(
                                    ActionStatus.COMPLETE,
                                    output="Command executed through Felix (see Terminal tab for output)",
                                    exit_code=0
                                ))
                        else:
                            self.after(0, lambda: bubble.set_status(
                                ActionStatus.DENIED,
                                output="Approval failed"
                            ))
                        return

                # No pending action found - command might have been SAFE and auto-executed
                # Or BLOCKED and denied
                self.after(0, lambda: bubble.set_status(
                    ActionStatus.COMPLETE,
                    output="Command processed through Felix (see Terminal tab for details)",
                    exit_code=0
                ))

            except Exception as e:
                logger.error(f"Error executing local action through Felix: {e}")
                self.after(0, lambda: bubble.set_status(ActionStatus.DENIED, output=f"Error: {e}"))

        thread = threading.Thread(target=do_execute, daemon=True)
        thread.start()

    def _deny_local_action(self, local_id: str, bubble: ActionBubble, command: str):
        """
        Handle denial of a locally-detected action.

        Args:
            local_id: Local tracking ID for the action
            bubble: The ActionBubble that was denied
            command: The command that was denied
        """
        logger.info(f"Local action denied: {command[:50]}...")
        bubble.set_status(ActionStatus.DENIED)

        # Feed denial back to Felix for continued conversation
        self._feed_denial_to_felix(command)

    def _create_action_bubble(self, command: str):
        """
        Create an action approval bubble for a command (fallback when Felix unavailable).

        Routes through Felix if possible, otherwise creates a disabled bubble
        showing that Felix system is required.

        Args:
            command: The command Felix wants to execute
        """
        if not self._message_area:
            logger.warning("Cannot create action bubble: no message area")
            return

        # Get trust level from TrustManager if available
        trust_level = self._classify_command_trust(command)

        logger.info(f"Creating action bubble: {command[:50]}... (trust: {trust_level})")

        # Generate a local approval ID for tracking
        local_approval_id = f"local_{uuid.uuid4().hex[:8]}"

        # Create the action bubble - route through Felix methods
        bubble = self._message_area.add_action_bubble(
            command=command,
            trust_level=trust_level,
            on_approve=lambda b, cmd: self._approve_local_action(local_approval_id, b, cmd),
            on_deny=lambda b, cmd: self._deny_local_action(local_approval_id, b, cmd)
        )

        # Track the bubble
        self._approval_bubbles[local_approval_id] = bubble

    def _classify_command_trust(self, command: str) -> str:
        """
        Classify a command's trust level.

        Args:
            command: The command to classify

        Returns:
            Trust level string: 'safe', 'review', or 'blocked'
        """
        try:
            # Try to get TrustManager from Felix system
            felix_system = self._get_felix_system()
            if felix_system and hasattr(felix_system, 'trust_manager'):
                trust_manager = felix_system.trust_manager
                level = trust_manager.classify_command(command)
                return level.value  # TrustLevel enum to string
        except Exception as e:
            logger.debug(f"Could not use TrustManager: {e}")

        # Fallback: simple pattern-based classification
        blocked_patterns = [
            r'rm\s+-rf', r'sudo\s+rm', r'mkfs', r'dd\s+if=',
            r'>\s*/dev/', r'chmod\s+777', r'curl.*\|\s*sh',
            r'wget.*\|\s*sh', r'\.ssh/', r'\.aws/', r'password'
        ]

        safe_patterns = [
            r'^ls\s', r'^pwd$', r'^cat\s', r'^head\s', r'^tail\s',
            r'^echo\s', r'^date$', r'^whoami$', r'^hostname$',
            r'^git\s+status', r'^git\s+log', r'^git\s+diff',
            r'^pip\s+list', r'^pip\s+show', r'^python\s+--version'
        ]

        # Check blocked first
        for pattern in blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return "blocked"

        # Check safe
        for pattern in safe_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return "safe"

        # Default to review
        return "review"

    def _get_project_root(self) -> str:
        """Get the project root directory."""
        import os
        # Try to get from Felix system
        felix_system = self._get_felix_system()
        if felix_system and hasattr(felix_system, 'project_root'):
            return str(felix_system.project_root)
        # Fallback to current directory
        return os.getcwd()

    def _feed_result_to_felix(self, command: str, result: Dict[str, Any]):
        """
        Feed command result back to Felix for continued response.

        Args:
            command: The executed command
            result: Execution result dict
        """
        # Build context message with result
        context = f"\n[Command executed: {command}]\n"
        if result.get('success'):
            stdout = result.get('stdout', '').strip()
            if stdout:
                # Truncate very long output
                if len(stdout) > 2000:
                    stdout = stdout[:2000] + "\n... (output truncated)"
                context += f"Output:\n```\n{stdout}\n```\n"
            else:
                context += "(command completed with no output)\n"
        else:
            stderr = result.get('stderr', '').strip()
            context += f"Error (exit {result.get('exit_code', -1)}):\n```\n{stderr}\n```\n"

        # Continue the conversation with this context
        self._continue_with_context(context)

    def _feed_denial_to_felix(self, command: str):
        """
        Notify Felix that a command was denied.

        Args:
            command: The denied command
        """
        context = f"\n[Command denied by user: {command}]\n"
        context += "The user chose not to execute this command. Please continue without this information or suggest an alternative approach.\n"

        self._continue_with_context(context)

    def _continue_with_context(self, context: str):
        """
        Continue Felix's response in the EXISTING streaming bubble.

        Feeds the command result back to the LLM so Felix can use it
        to complete the answer. The continuation streams into the same
        bubble, creating a seamless Claude Code-like experience.

        Args:
            context: Command result context to include
        """
        if not self.current_session or not self._message_area:
            return

        logger.info(f"Continuing with context: {context[:100]}...")

        # Mark this as a continuation - skip pattern detection on response
        self._is_continuation = True

        # Get the LLM client
        llm_client = self._get_llm_client()
        if not llm_client:
            logger.warning("Cannot continue: no LLM client")
            return

        # Get original user question
        original_question = ""
        for msg in reversed(self.current_session.messages):
            if msg.role == "user":
                original_question = msg.content
                break

        # Build system prompt with command result context
        system_prompt = self._build_system_prompt() + f"""

COMMAND EXECUTION RESULT:
{context}

Use this result to answer the user's question.

IF THE COMMAND FAILED OR RETURNED NO RESULTS:
- NEVER say you cannot execute commands - you CAN and SHOULD retry
- Do NOT repeat the exact same command - try a DIFFERENT approach
- For shell quoting errors: use heredoc (cat << 'EOF') instead of echo with quotes
- For file search: use `find . -name "filename"` instead of `ls | grep`
- For reading files: verify the path exists first with `find` before using `cat`
- If exit code is non-zero, the command failed - try an alternative approach
- If multiple approaches fail: explain what you tried and ask user for guidance
- Python one-liners are a reliable fallback: python3 -c "open('f.txt','w').write('...')" """

        # Build user prompt WITH conversation history for full context
        conversation_history = self._format_conversation_history(limit=10)
        if conversation_history:
            user_prompt = f"{conversation_history}\n\nBased on the command result above, continue answering: {original_question}"
        else:
            user_prompt = f"Based on the command result above, please answer: {original_question}"

        # DON'T create new bubble - continue in existing one
        # The streaming bubble already exists from the initial response

        # Create/reuse cancel event for continuation
        if not self._cancel_event or self._cancel_event.is_set():
            self._cancel_event = threading.Event()
        cancel_event = self._cancel_event  # Local reference for closure

        # Run continuation in background
        def process_continuation():
            try:
                # Check for cancellation before starting
                if cancel_event.is_set():
                    return

                def on_chunk(chunk):
                    # Check cancellation before queuing
                    if cancel_event.is_set():
                        return
                    # Handle both string chunks and StreamingChunk objects
                    if hasattr(chunk, 'content'):
                        self._result_queue.put(('chunk', chunk.content))
                    else:
                        self._result_queue.put(('chunk', str(chunk)))

                # Use streaming completion with correct API
                llm_client.complete_streaming(
                    agent_id="chat_continuation",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7,
                    callback=on_chunk,
                    batch_interval=0.1,
                    cancel_event=cancel_event
                )

                self._result_queue.put(('done', None))

            except Exception as e:
                logger.error(f"Continuation error: {e}")
                self._result_queue.put(('error', str(e)))

        thread = threading.Thread(target=process_continuation, daemon=True)
        thread.start()

    def _get_llm_client(self):
        """Get LLM client - requires Felix system to be running."""
        # Check if Felix system exists and is running
        if not self.main_app:
            self._update_connection_status("disconnected", "○ No app context")
            return None

        if not hasattr(self.main_app, 'felix_system') or not self.main_app.felix_system:
            self._update_connection_status("disconnected", "○ Start Felix system")
            return None

        system = self.main_app.felix_system
        # Check for lm_client (correct attribute name in FelixSystem)
        if not hasattr(system, 'lm_client') or not system.lm_client:
            self._update_connection_status("disconnected", "○ LLM not initialized")
            return None

        # Got a client - test connection
        self._llm_client = system.lm_client
        if hasattr(self._llm_client, 'test_connection'):
            if not self._llm_client.test_connection():
                self._update_connection_status("error", "● LM Studio offline")
                return None

        self._update_connection_status("connected")
        return self._llm_client

    def _get_felix_system(self):
        """Get Felix system from main app."""
        if self._felix_system:
            return self._felix_system

        if self.main_app and hasattr(self.main_app, 'felix_system'):
            self._felix_system = self.main_app.felix_system
            return self._felix_system

        return None

    def _get_knowledge_context(self, query: str) -> str:
        """Get relevant knowledge context for the query."""
        try:
            if not self.main_app:
                return ""

            felix_system = self._get_felix_system()
            if not felix_system or not hasattr(felix_system, 'knowledge_retriever'):
                return ""

            retriever = felix_system.knowledge_retriever
            if not retriever:
                return ""

            # Retrieve relevant knowledge
            results = retriever.retrieve(query, top_k=3)

            if not results:
                return ""

            # Format knowledge context
            context_parts = ["Relevant knowledge:"]
            for result in results:
                content = result.get('content', str(result))
                context_parts.append(f"- {content[:500]}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.debug(f"Could not get knowledge context: {e}")
            return ""

    def _format_conversation_history(self, limit: int = 10) -> str:
        """
        Format recent conversation history for inclusion in prompts.

        This enables Felix to maintain context across multiple messages,
        like knowing which file was discussed when the user says "that file".

        Args:
            limit: Maximum number of messages to include

        Returns:
            Formatted conversation history string
        """
        if not self.current_session:
            return ""

        messages = self.current_session.get_context_messages(limit=limit)
        if not messages:
            return ""

        history_parts = ["CONVERSATION HISTORY:"]
        for msg in messages:
            # get_context_messages() returns dicts, not objects
            role = "USER" if msg['role'] == "user" else "FELIX"
            # Truncate long messages to avoid context bloat
            content = msg['content']
            if len(content) > 500:
                content = content[:500] + "..."
            history_parts.append(f"{role}: {content}")

        return "\n".join(history_parts)

    def _build_system_prompt(self, knowledge_context: str = "") -> str:
        """Build system prompt for simple mode.

        Loads the system prompt from config/chat_system_prompt.md with
        template variable substitution. Falls back to a minimal prompt
        if the file is not found.
        """
        # Load from external file with variable substitution
        base_prompt = load_system_prompt()

        # Log prompt info on first load for debugging
        info = get_prompt_info()
        if info.get("estimated_tokens"):
            logger.debug(
                f"System prompt: {info['estimated_tokens']} estimated tokens, "
                f"{info['size_chars']} chars"
            )

        # Knowledge context is appended if provided
        if knowledge_context:
            return f"{base_prompt}\n\n{knowledge_context}"

        return base_prompt

    def _load_more_messages(self):
        """Load more message history (for lazy loading)."""
        # TODO: Implement pagination
        pass

    # =========================================================================
    # UI CONTROLS
    # =========================================================================

    def _on_mode_changed(self, mode: str):
        """Handle mode selector change."""
        if self.current_session:
            self.current_session.mode = mode.lower()
            logger.info(f"Mode changed to: {mode}")

    def _on_knowledge_toggled(self):
        """Handle knowledge toggle change."""
        enabled = self._knowledge_var.get()
        if self.current_session:
            self.current_session.knowledge_enabled = enabled
            logger.info(f"Knowledge brain {'enabled' if enabled else 'disabled'}")

    def _toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self._sidebar_visible = not self._sidebar_visible

        if self._sidebar_visible:
            self._sidebar_frame.grid()
            self._separator.grid()
        else:
            self._sidebar_frame.grid_remove()
            self._separator.grid_remove()

    def _on_separator_drag(self, delta: int):
        """Handle separator drag to resize sidebar."""
        new_width = max(150, min(400, self._sidebar_width + delta))
        self._sidebar_width = new_width
        self._sidebar_frame.configure(width=new_width)

    def _on_theme_change(self, mode: str):
        """Update colors when theme changes."""
        colors = self.theme_manager.colors

        self._sidebar_frame.configure(fg_color=colors["bg_secondary"])
        self._chat_frame.configure(fg_color=colors["bg_primary"])

    # =========================================================================
    # RESPONSIVE LAYOUT
    # =========================================================================

    def on_breakpoint_change(self, breakpoint: Breakpoint, config: BreakpointConfig):
        """Handle responsive layout changes based on breakpoint."""
        if self._current_layout == breakpoint:
            return

        self._current_layout = breakpoint

        if breakpoint == Breakpoint.COMPACT:
            # Hide sidebar, show toggle button
            self._sidebar_frame.grid_remove()
            self._separator.grid_remove()
            self._sidebar_toggle.grid()
            self._sidebar_visible = False

        else:
            # Show sidebar
            self._sidebar_frame.grid()
            self._separator.grid()
            self._sidebar_toggle.grid_remove()
            self._sidebar_visible = True

            # Adjust sidebar width based on breakpoint
            if breakpoint in (Breakpoint.WIDE, Breakpoint.ULTRAWIDE):
                self._sidebar_width = SIDEBAR_WIDE
            else:
                self._sidebar_width = SIDEBAR_STANDARD

            self._sidebar_frame.configure(width=self._sidebar_width)

    # =========================================================================
    # FEATURE LIFECYCLE (called by main app when Felix system starts/stops)
    # =========================================================================

    def _enable_features(self):
        """Enable chat features when Felix system starts."""
        self._update_connection_status("connected")
        if self._input_area:
            self._input_area.set_enabled(True)
        logger.info("Chat features enabled")

    def _disable_features(self):
        """Disable chat features when Felix system stops."""
        self._update_connection_status("disconnected", "○ Start Felix system")
        if self._input_area:
            self._input_area.set_enabled(False)
        logger.info("Chat features disabled")

    # =========================================================================
    # KEYBOARD SHORTCUTS
    # =========================================================================

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for chat tab."""
        import platform
        ctrl = "Command" if platform.system() == "Darwin" else "Control"

        # Bind shortcuts to this widget (CustomTkinter doesn't support bind_all)
        # These will work when focus is within the chat tab
        self.bind(f"<{ctrl}-n>", self._on_new_chat_shortcut)
        self.bind(f"<{ctrl}-Shift-N>", self._on_new_folder_shortcut)
        self.bind(f"<{ctrl}-k>", self._on_toggle_knowledge_shortcut)
        self.bind(f"<{ctrl}-m>", self._on_toggle_mode_shortcut)

        # Also bind to the root window for global access
        try:
            root = self.winfo_toplevel()
            root.bind(f"<{ctrl}-n>", self._on_new_chat_shortcut)
            root.bind(f"<{ctrl}-Shift-N>", self._on_new_folder_shortcut)
            root.bind(f"<{ctrl}-k>", self._on_toggle_knowledge_shortcut)
            root.bind(f"<{ctrl}-m>", self._on_toggle_mode_shortcut)
        except Exception:
            pass  # Root binding is optional

    def _on_new_chat_shortcut(self, event=None):
        """Handle Ctrl/Cmd+N shortcut for new chat."""
        # Only respond if chat tab is visible
        if self.winfo_viewable():
            self._start_new_session()
            return "break"

    def _on_new_folder_shortcut(self, event=None):
        """Handle Ctrl/Cmd+Shift+N shortcut for new folder."""
        if self.winfo_viewable():
            self._create_folder()
            return "break"

    def _on_toggle_knowledge_shortcut(self, event=None):
        """Handle Ctrl/Cmd+K shortcut to toggle knowledge brain."""
        if self.winfo_viewable():
            current = self._knowledge_var.get()
            self._knowledge_var.set(not current)
            self._on_knowledge_toggled()
            return "break"

    def _on_toggle_mode_shortcut(self, event=None):
        """Handle Ctrl/Cmd+M shortcut to toggle mode."""
        if self.winfo_viewable():
            current = self._mode_selector.get()
            new_mode = "Workflow" if current == "Simple" else "Simple"
            self._mode_selector.set(new_mode)
            self._on_mode_changed(new_mode)
            return "break"

    def destroy(self):
        """Cleanup when tab is destroyed."""
        try:
            self.theme_manager.unregister_callback(self._on_theme_change)
        except Exception:
            pass
        super().destroy()
