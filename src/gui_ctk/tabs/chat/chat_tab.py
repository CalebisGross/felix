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
from typing import Optional, Callable, Dict, Any
import logging
import threading
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

# Import chat components (will be created by subagents)
from .chat_session import ChatSession, ChatSessionManager, ChatMessage

logger = logging.getLogger(__name__)

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

        # Connection status (starts disconnected until Felix is running)
        self._status_label = ctk.CTkLabel(
            header_frame,
            text="○ Start Felix system",
            font=ctk.CTkFont(size=FONT_SMALL),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self._status_label.grid(row=0, column=3, padx=SPACE_MD, pady=SPACE_SM, sticky="e")

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

        # Process based on mode
        mode = self.current_session.mode
        if mode == 'simple':
            self._process_simple_mode(content)
        else:
            self._process_workflow_mode(content)

    def _process_simple_mode(self, content: str):
        """Process message in simple (direct LLM) mode."""
        # Update status to sending
        self._update_connection_status("sending")

        # Start streaming message
        streaming_msg = self.current_session.start_assistant_message()

        # Add streaming bubble to UI
        streaming_bubble = None
        if self._message_area:
            streaming_bubble = self._message_area.add_streaming_message()

        # Get LLM client from main app
        llm_client = self._get_llm_client()

        if not llm_client:
            # No LLM available - show error
            error_content = "LLM not available. Please ensure LM Studio is running."
            self.current_session.append_to_streaming(error_content)
            self.current_session.finish_streaming()

            if streaming_bubble:
                streaming_bubble.update_content(error_content)
                streaming_bubble.finish_streaming()

            if self._input_area:
                self._input_area.set_streaming(False)
            return

        # Build context
        context_messages = self.current_session.get_context_messages(limit=10)

        # Get knowledge context if enabled
        knowledge_context = ""
        if self.current_session.knowledge_enabled:
            knowledge_context = self._get_knowledge_context(content)

        # Build system prompt
        system_prompt = self._build_system_prompt(knowledge_context)

        # Run in background thread
        def process():
            try:
                # Streaming callback
                def on_chunk(chunk):
                    self._result_queue.put(('chunk', chunk.content if hasattr(chunk, 'content') else str(chunk)))

                # Call LLM with streaming
                if hasattr(llm_client, 'complete_streaming'):
                    response = llm_client.complete_streaming(
                        agent_id="chat_simple",
                        system_prompt=system_prompt,
                        user_prompt=content,
                        temperature=0.7,
                        callback=on_chunk,
                        batch_interval=0.1
                    )
                else:
                    # Fallback to non-streaming
                    response = llm_client.complete(
                        agent_id="chat_simple",
                        system_prompt=system_prompt,
                        user_prompt=content,
                        temperature=0.7
                    )
                    self._result_queue.put(('chunk', response.content))

                self._result_queue.put(('done', None))

            except Exception as e:
                logger.error(f"LLM error: {e}")
                self._result_queue.put(('error', str(e)))
            finally:
                # Thread-safe GUI update: force reset streaming state
                self.after_idle(self._force_reset_streaming)

        self.thread_manager.start_thread(process)

    def _process_workflow_mode(self, content: str):
        """Process message in workflow (multi-agent) mode."""
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

        # Run workflow in background
        def process():
            try:
                # Run Felix workflow
                from src.workflows.workflow_engine import run_felix_workflow

                result = run_felix_workflow(
                    felix_system=felix_system,
                    task_input=content,
                    streaming_callback=lambda agent, chunk: self._result_queue.put(('thinking', (agent, chunk)))
                )

                # Extract synthesis
                synthesis = result.get("centralpost_synthesis", {})
                final_content = synthesis.get("synthesis_content", "No response generated.")
                confidence = synthesis.get("confidence", 0.0)

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
        if self.current_session and self.current_session.state.is_streaming:
            self.current_session.cancel_streaming()

            if self._input_area:
                self._input_area.set_streaming(False)

            logger.info("Generation stopped by user")

    def _poll_results(self):
        """Poll the result queue for updates."""
        try:
            while not self._result_queue.empty():
                msg_type, data = self._result_queue.get_nowait()

                if msg_type == 'chunk':
                    # Update status to streaming on first chunk
                    self._update_connection_status("streaming")

                    # Append streaming content
                    if self.current_session:
                        self.current_session.append_to_streaming(data)

                    # Update UI
                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.append_content(data)

                elif msg_type == 'thinking':
                    agent, content = data
                    if self.current_session:
                        self.current_session.add_thinking_step(agent, content)

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.add_thinking_step(agent, content)

                elif msg_type == 'done':
                    # Finish streaming (simple mode)
                    self._update_connection_status("connected")

                    if self.current_session:
                        self.current_session.finish_streaming()

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.finish_streaming()

                    if self._input_area:
                        self._input_area.set_streaming(False)

                elif msg_type == 'workflow_done':
                    final_content, confidence = data
                    self._update_connection_status("connected")

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

                    if self.current_session:
                        self.current_session.append_to_streaming(error_msg)
                        self.current_session.finish_streaming()

                    if self._message_area and self._message_area._streaming_bubble:
                        self._message_area._streaming_bubble.update_content(error_msg)
                        self._message_area._streaming_bubble.finish_streaming()

                    if self._input_area:
                        self._input_area.set_streaming(False)

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

    def _build_system_prompt(self, knowledge_context: str = "") -> str:
        """Build system prompt for simple mode."""
        base_prompt = """You are Felix, an air-gapped multi-agent AI operating entirely offline with zero external dependencies.

IDENTITY:
- Part of a collaborative team progressing from exploration to synthesis along a helical path
- Your position on the helix determines your focus: broad exploration (top) → precise synthesis (bottom)
- Trust your fellow agents. Build on their work. Don't repeat what's already been discovered.
- You are NOT ChatGPT, GPT, Claude, or any OpenAI/Anthropic product. You are Felix.

CONSTRAINTS:
- OFFLINE-ONLY: No internet, no external APIs, no cloud services. Everything is local.
- NO HALLUCINATION: Never fabricate file paths, function names, dates, or system details.
- MATCH VERBOSITY TO TASK: Simple question = direct answer. Complex task = structured response.

SYSTEM COMMANDS:
When you need system information, use: SYSTEM_ACTION_NEEDED: [command]
- SAFE (auto-execute): ls, pwd, cat, date, pip list
- REVIEW (needs approval): mkdir, pip install, git commit
- BLOCKED (never): rm -rf, sudo, credential access

PHILOSOPHY:
- Precision over verbosity
- Collaboration over duplication
- Facts over speculation
- Action over explanation"""

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
        """Setup global keyboard shortcuts for chat tab."""
        import platform
        ctrl = "Command" if platform.system() == "Darwin" else "Control"

        # Bind shortcuts (only active when this tab has focus)
        self.bind_all(f"<{ctrl}-n>", self._on_new_chat_shortcut)
        self.bind_all(f"<{ctrl}-Shift-N>", self._on_new_folder_shortcut)
        self.bind_all(f"<{ctrl}-k>", self._on_toggle_knowledge_shortcut)
        self.bind_all(f"<{ctrl}-m>", self._on_toggle_mode_shortcut)

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
