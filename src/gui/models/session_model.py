"""Session model with SQLite persistence."""

import json
import logging
import sqlite3
import threading
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid

from PySide6.QtCore import QObject, Signal

from .message_model import Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A chat session containing messages."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Chat"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create from dictionary."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "New Chat"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            messages=messages,
        )

    def generate_title(self) -> str:
        """Generate a title from the first user message."""
        for message in self.messages:
            if message.role == MessageRole.USER:
                content = message.content.strip()
                # Truncate to first 50 chars or first line
                if "\n" in content:
                    content = content.split("\n")[0]
                if len(content) > 50:
                    content = content[:47] + "..."
                return content
        return "New Chat"


class SessionStore(QObject):
    """Persistent storage for chat sessions using SQLite.

    Thread-safe with connection-per-thread pattern.

    Signals:
        session_created: Emitted when a new session is created
        session_updated: Emitted when a session is modified
        session_deleted: Emitted when a session is deleted
        sessions_loaded: Emitted when sessions are loaded from DB
    """

    session_created = Signal(Session)
    session_updated = Signal(str)  # session_id
    session_deleted = Signal(str)  # session_id
    sessions_loaded = Signal(list)  # List[Session]

    def __init__(self, db_path: str = "felix_gui_sessions.db", parent=None):
        super().__init__(parent)
        self._db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                mode_used TEXT,
                action_id TEXT,
                action_status TEXT,
                command TEXT,
                position INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, position)
        """)

        conn.commit()
        logger.info(f"Session database initialized at {self._db_path}")

    def create_session(self, title: str = "New Chat") -> Session:
        """Create a new session."""
        session = Session(title=title)

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            session.id,
            session.title,
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
        ))

        conn.commit()
        logger.info(f"Created session: {session.id}")
        self.session_created.emit(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID with all messages."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get session
        cursor.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        session = Session(
            id=row["id"],
            title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

        # Get messages
        cursor.execute("""
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY position ASC
        """, (session_id,))

        for msg_row in cursor.fetchall():
            message = Message(
                id=msg_row["id"],
                role=MessageRole(msg_row["role"]),
                content=msg_row["content"],
                timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                confidence=msg_row["confidence"],
                mode_used=msg_row["mode_used"],
                action_id=msg_row["action_id"],
                action_status=msg_row["action_status"],
                command=msg_row["command"],
            )
            session.messages.append(message)

        return session

    def get_all_sessions(self, limit: int = 50) -> List[Session]:
        """Get all sessions (without messages, for listing)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))

        sessions = []
        for row in cursor.fetchall():
            session = Session(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            sessions.append(session)

        return sessions

    def update_session(self, session: Session):
        """Update session metadata."""
        session.updated_at = datetime.now()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions
            SET title = ?, updated_at = ?
            WHERE id = ?
        """, (
            session.title,
            session.updated_at.isoformat(),
            session.id,
        ))

        conn.commit()
        self.session_updated.emit(session.id)

    def add_message(self, session_id: str, message: Message):
        """Add a message to a session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get next position
        cursor.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE session_id = ?",
            (session_id,)
        )
        position = cursor.fetchone()[0]

        cursor.execute("""
            INSERT INTO messages (
                id, session_id, role, content, timestamp,
                confidence, mode_used, action_id, action_status, command, position
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            session_id,
            message.role.value,
            message.content,
            message.timestamp.isoformat(),
            message.confidence,
            message.mode_used,
            message.action_id,
            message.action_status,
            message.command,
            position,
        ))

        # Update session timestamp
        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), session_id))

        conn.commit()

    def update_message(self, message: Message):
        """Update a message."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE messages
            SET content = ?, confidence = ?, mode_used = ?,
                action_status = ?
            WHERE id = ?
        """, (
            message.content,
            message.confidence,
            message.mode_used,
            message.action_status,
            message.id,
        ))

        conn.commit()

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

        conn.commit()
        logger.info(f"Deleted session: {session_id}")
        self.session_deleted.emit(session_id)

    def search_sessions(self, query: str, limit: int = 20) -> List[Session]:
        """Search sessions by title or message content."""
        conn = self._get_connection()
        cursor = conn.cursor()

        search_term = f"%{query}%"

        cursor.execute("""
            SELECT DISTINCT s.* FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE s.title LIKE ? OR m.content LIKE ?
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (search_term, search_term, limit))

        sessions = []
        for row in cursor.fetchall():
            session = Session(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            sessions.append(session)

        return sessions

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
