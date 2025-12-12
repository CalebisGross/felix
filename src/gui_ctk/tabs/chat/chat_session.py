"""
Chat Session Wrapper for Felix GUI

Wraps the CLI SessionManager with GUI-specific functionality:
- Streaming state management
- Mode tracking (simple vs workflow)
- Knowledge brain toggle state
- Message formatting for display
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import SessionManager from CLI chat module
try:
    from src.cli_chat.session_manager import SessionManager, Session, Message
    HAS_SESSION_MANAGER = True
except ImportError:
    HAS_SESSION_MANAGER = False
    logger.warning("SessionManager not available - using mock implementation")


@dataclass
class ChatMessage:
    """
    A chat message with GUI-specific metadata.

    Extends the CLI Message with additional fields for display.
    """
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[int] = None
    workflow_id: Optional[str] = None

    # GUI-specific fields
    thinking: List[Dict[str, str]] = field(default_factory=list)
    knowledge_sources: List[Dict[str, str]] = field(default_factory=list)
    is_streaming: bool = False
    confidence: Optional[float] = None

    @classmethod
    def from_cli_message(cls, msg: 'Message') -> 'ChatMessage':
        """Convert a CLI Message to ChatMessage."""
        return cls(
            role=msg.role,
            content=msg.content,
            timestamp=msg.timestamp,
            message_id=msg.message_id,
            workflow_id=msg.workflow_id
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'message_id': self.message_id,
            'workflow_id': self.workflow_id,
            'thinking': self.thinking,
            'knowledge_sources': self.knowledge_sources,
            'confidence': self.confidence
        }


@dataclass
class ChatSessionState:
    """
    State for a chat session in the GUI.

    Tracks mode, knowledge settings, and streaming state.
    """
    session_id: str
    title: Optional[str] = None
    mode: str = 'simple'  # 'simple' or 'workflow'
    knowledge_enabled: bool = True
    is_streaming: bool = False
    current_streaming_message: Optional[ChatMessage] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    message_count: int = 0

    # Folder organization
    folder_id: Optional[str] = None
    pinned: bool = False


class ChatSession:
    """
    GUI wrapper for chat session management.

    Provides a high-level interface for the chat tab to interact with
    sessions, handling mode switching, streaming, and knowledge integration.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        db_path: str = "felix_cli_sessions.db",
        on_message_added: Optional[Callable[[ChatMessage], None]] = None,
        on_streaming_update: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize chat session wrapper.

        Args:
            session_id: Existing session ID to load, or None for new session
            db_path: Path to the sessions database
            on_message_added: Callback when a message is added
            on_streaming_update: Callback for streaming content updates
        """
        self.db_path = db_path
        self.on_message_added = on_message_added
        self.on_streaming_update = on_streaming_update

        # Initialize session manager
        if HAS_SESSION_MANAGER:
            self._session_manager = SessionManager(db_path)
        else:
            self._session_manager = None
            logger.warning("Running without SessionManager - messages won't be persisted")

        # Initialize or load session
        if session_id:
            self._load_session(session_id)
        else:
            self._create_new_session()

        # Message cache for quick access
        self._messages: List[ChatMessage] = []
        self._load_messages()

    def _create_new_session(self):
        """Create a new chat session."""
        if self._session_manager:
            session_id = self._session_manager.create_session()
            session = self._session_manager.get_session(session_id)

            self.state = ChatSessionState(
                session_id=session_id,
                title=session.title if session else None,
                created_at=session.created_at if session else datetime.now(),
                last_active=session.last_active if session else datetime.now(),
                message_count=session.message_count if session else 0
            )
        else:
            # Mock session for development
            import uuid
            self.state = ChatSessionState(
                session_id=str(uuid.uuid4())[:8]
            )

        logger.info(f"Created new chat session: {self.state.session_id}")

    def _load_session(self, session_id: str):
        """Load an existing session."""
        if self._session_manager:
            session = self._session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")

            self.state = ChatSessionState(
                session_id=session_id,
                title=session.title,
                created_at=session.created_at,
                last_active=session.last_active,
                message_count=session.message_count,
                # Load extended fields if available
                folder_id=getattr(session, 'folder_id', None),
                pinned=getattr(session, 'pinned', False),
                mode=getattr(session, 'mode', 'simple'),
                knowledge_enabled=getattr(session, 'knowledge_enabled', True)
            )
        else:
            self.state = ChatSessionState(session_id=session_id)

        logger.info(f"Loaded chat session: {session_id}")

    def _load_messages(self):
        """Load messages from storage."""
        self._messages = []

        if self._session_manager:
            cli_messages = self._session_manager.get_messages(self.state.session_id)
            for msg in cli_messages:
                self._messages.append(ChatMessage.from_cli_message(msg))

        logger.debug(f"Loaded {len(self._messages)} messages")

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self.state.session_id

    @property
    def messages(self) -> List[ChatMessage]:
        """Get all messages in the session."""
        return self._messages.copy()

    @property
    def mode(self) -> str:
        """Get current chat mode."""
        return self.state.mode

    @mode.setter
    def mode(self, value: str):
        """Set chat mode and persist."""
        if value not in ('simple', 'workflow'):
            raise ValueError(f"Invalid mode: {value}")

        self.state.mode = value
        if self._session_manager and hasattr(self._session_manager, 'update_session_mode'):
            self._session_manager.update_session_mode(self.state.session_id, value)

    @property
    def knowledge_enabled(self) -> bool:
        """Get knowledge brain state."""
        return self.state.knowledge_enabled

    @knowledge_enabled.setter
    def knowledge_enabled(self, value: bool):
        """Set knowledge brain state and persist."""
        self.state.knowledge_enabled = value
        if self._session_manager and hasattr(self._session_manager, 'update_session_knowledge_enabled'):
            self._session_manager.update_session_knowledge_enabled(self.state.session_id, value)

    def add_user_message(self, content: str) -> ChatMessage:
        """
        Add a user message to the session.

        Args:
            content: Message content

        Returns:
            The created ChatMessage
        """
        message = ChatMessage(
            role='user',
            content=content,
            timestamp=datetime.now()
        )

        # Persist to storage
        if self._session_manager:
            self._session_manager.add_message(
                session_id=self.state.session_id,
                role='user',
                content=content
            )

        self._messages.append(message)
        self.state.message_count += 1
        self.state.last_active = datetime.now()

        if self.on_message_added:
            self.on_message_added(message)

        return message

    def start_assistant_message(self) -> ChatMessage:
        """
        Start a new streaming assistant message.

        Returns:
            The streaming ChatMessage
        """
        message = ChatMessage(
            role='assistant',
            content='',
            timestamp=datetime.now(),
            is_streaming=True
        )

        self._messages.append(message)
        self.state.current_streaming_message = message
        self.state.is_streaming = True

        if self.on_message_added:
            self.on_message_added(message)

        return message

    def append_to_streaming(self, chunk: str):
        """
        Append content to the current streaming message.

        Args:
            chunk: Content to append
        """
        if self.state.current_streaming_message:
            self.state.current_streaming_message.content += chunk

            if self.on_streaming_update:
                self.on_streaming_update(chunk)

    def add_thinking_step(self, agent: str, content: str):
        """
        Add a thinking step to the current streaming message.

        Args:
            agent: Agent type (research, analysis, synthesis, critic)
            content: Thinking content
        """
        if self.state.current_streaming_message:
            self.state.current_streaming_message.thinking.append({
                'agent': agent,
                'content': content
            })

    def add_knowledge_source(self, source_id: str, title: str):
        """
        Add a knowledge source to the current message.

        Args:
            source_id: Knowledge entry ID
            title: Source title/description
        """
        if self.state.current_streaming_message:
            self.state.current_streaming_message.knowledge_sources.append({
                'id': source_id,
                'title': title
            })

    def finish_streaming(
        self,
        confidence: Optional[float] = None,
        workflow_id: Optional[str] = None
    ):
        """
        Finish the current streaming message.

        Args:
            confidence: Final confidence score
            workflow_id: Associated workflow ID
        """
        if self.state.current_streaming_message:
            msg = self.state.current_streaming_message
            msg.is_streaming = False
            msg.confidence = confidence
            msg.workflow_id = workflow_id

            # Persist to storage
            if self._session_manager:
                self._session_manager.add_message(
                    session_id=self.state.session_id,
                    role='assistant',
                    content=msg.content,
                    workflow_id=workflow_id
                )

            self.state.current_streaming_message = None
            self.state.is_streaming = False
            self.state.message_count += 1
            self.state.last_active = datetime.now()

    def cancel_streaming(self):
        """Cancel the current streaming message."""
        if self.state.current_streaming_message:
            # Remove the incomplete message from the list
            if self._messages and self._messages[-1] == self.state.current_streaming_message:
                self._messages.pop()

            self.state.current_streaming_message = None
            self.state.is_streaming = False

    def set_title(self, title: str):
        """Set the session title."""
        self.state.title = title
        if self._session_manager:
            self._session_manager.set_title(self.state.session_id, title)

    def get_context_messages(self, limit: int = 20) -> List[Dict[str, str]]:
        """
        Get recent messages formatted for LLM context.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of message dicts with 'role' and 'content'
        """
        recent = self._messages[-limit:] if len(self._messages) > limit else self._messages

        return [
            {'role': msg.role, 'content': msg.content}
            for msg in recent
            if not msg.is_streaming
        ]

    def clear_messages(self):
        """Clear all messages in the session."""
        self._messages = []
        self.state.message_count = 0

        # Note: This doesn't delete from storage - sessions are preserved

    def export_to_dict(self) -> Dict[str, Any]:
        """Export session to dictionary (for YAML export)."""
        return {
            'id': self.state.session_id,
            'title': self.state.title,
            'created_at': self.state.created_at.isoformat(),
            'last_active': self.state.last_active.isoformat(),
            'folder_id': self.state.folder_id,
            'pinned': self.state.pinned,
            'settings': {
                'mode': self.state.mode,
                'knowledge_enabled': self.state.knowledge_enabled
            },
            'messages': [msg.to_dict() for msg in self._messages]
        }


class ChatSessionManager:
    """
    High-level manager for multiple chat sessions.

    Provides session switching, listing, and folder management for the GUI.
    """

    def __init__(self, db_path: str = "felix_cli_sessions.db"):
        """
        Initialize session manager.

        Args:
            db_path: Path to the sessions database
        """
        self.db_path = db_path

        if HAS_SESSION_MANAGER:
            self._session_manager = SessionManager(db_path)
        else:
            self._session_manager = None

        self._current_session: Optional[ChatSession] = None

    @property
    def current_session(self) -> Optional[ChatSession]:
        """Get the currently active session."""
        return self._current_session

    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.

        Args:
            title: Optional session title

        Returns:
            The new ChatSession
        """
        session = ChatSession(db_path=self.db_path)
        if title:
            session.set_title(title)

        self._current_session = session
        return session

    def load_session(self, session_id: str) -> ChatSession:
        """
        Load an existing session.

        Args:
            session_id: Session ID to load

        Returns:
            The loaded ChatSession
        """
        session = ChatSession(session_id=session_id, db_path=self.db_path)
        self._current_session = session
        return session

    def list_sessions(
        self,
        folder_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List available sessions.

        Args:
            folder_id: Filter by folder (None for root)
            limit: Maximum number of sessions

        Returns:
            List of session metadata dicts
        """
        if not self._session_manager:
            return []

        sessions = self._session_manager.list_sessions(limit=limit)

        result = []
        for session in sessions:
            # Filter by folder if specified
            session_folder = getattr(session, 'folder_id', None)
            if folder_id is not None and session_folder != folder_id:
                continue

            result.append({
                'session_id': session.session_id,
                'title': session.title or f"Chat {session.session_id[:8]}",
                'created_at': session.created_at,
                'last_active': session.last_active,
                'message_count': session.message_count,
                'folder_id': session_folder,
                'pinned': getattr(session, 'pinned', False)
            })

        return result

    def list_folders(self) -> List[Dict[str, Any]]:
        """
        List all folders.

        Returns:
            List of folder metadata dicts
        """
        if not self._session_manager or not hasattr(self._session_manager, 'get_folders'):
            return []

        folders = self._session_manager.get_folders()
        return [
            {
                'folder_id': f.folder_id,
                'name': f.name,
                'parent_id': f.parent_folder_id,
                'position': f.position
            }
            for f in folders
        ]

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """
        Create a new folder.

        Args:
            name: Folder name
            parent_id: Parent folder ID (None for root)

        Returns:
            The new folder ID
        """
        if not self._session_manager or not hasattr(self._session_manager, 'create_folder'):
            raise RuntimeError("Folder management not available")

        return self._session_manager.create_folder(name, parent_id)

    def delete_session(self, session_id: str):
        """Delete a session."""
        if self._session_manager:
            self._session_manager.delete_session(session_id)

        if self._current_session and self._current_session.session_id == session_id:
            self._current_session = None

    def delete_folder(self, folder_id: str):
        """Delete a folder (sessions moved to root)."""
        if self._session_manager and hasattr(self._session_manager, 'delete_folder'):
            self._session_manager.delete_folder(folder_id)
