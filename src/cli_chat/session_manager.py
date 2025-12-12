"""
Session Manager for Felix conversational CLI.

Handles session persistence, message history, and conversation threading.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class Message:
    """Represents a single message in a conversation."""

    def __init__(
        self,
        role: str,
        content: str,
        workflow_id: Optional[str] = None,
        message_id: Optional[int] = None,
        timestamp: Optional[datetime] = None
    ):
        self.message_id = message_id
        self.role = role  # 'user', 'assistant', 'system'
        self.content = content
        self.workflow_id = workflow_id
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return {
            'id': self.message_id,
            'role': self.role,
            'content': self.content,
            'workflow_id': self.workflow_id,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Folder:
    """Represents a folder for organizing sessions."""
    folder_id: str
    name: str
    parent_folder_id: Optional[str] = None
    position: int = 0
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert folder to dictionary format."""
        return {
            'folder_id': self.folder_id,
            'name': self.name,
            'parent_folder_id': self.parent_folder_id,
            'position': self.position,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Session:
    """Represents a chat session with metadata."""

    def __init__(
        self,
        session_id: str,
        created_at: datetime,
        last_active: datetime,
        message_count: int,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        pinned: bool = False,
        position: int = 0,
        mode: str = 'simple',
        knowledge_enabled: bool = True
    ):
        self.session_id = session_id
        self.created_at = created_at
        self.last_active = last_active
        self.message_count = message_count
        self.title = title
        self.tags = tags or []
        self.folder_id = folder_id
        self.pinned = pinned
        self.position = position
        self.mode = mode
        self.knowledge_enabled = knowledge_enabled

    def to_dict(self) -> Dict:
        """Convert session to dictionary format."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'message_count': self.message_count,
            'title': self.title,
            'tags': self.tags,
            'folder_id': self.folder_id,
            'pinned': self.pinned,
            'position': self.position,
            'mode': self.mode,
            'knowledge_enabled': self.knowledge_enabled
        }


class SessionManager:
    """Manages chat sessions and message history with SQLite persistence."""

    def __init__(self, db_path: str = "felix_cli_sessions.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_active TIMESTAMP NOT NULL,
                message_count INTEGER DEFAULT 0,
                title TEXT,
                tags TEXT DEFAULT '[]'
            )
        ''')

        # Folders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                folder_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_folder_id TEXT,
                position INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_folder_id) REFERENCES folders(folder_id)
            )
        ''')

        # Migrate existing databases: Add columns if they don't exist
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'title' not in columns:
            cursor.execute('ALTER TABLE sessions ADD COLUMN title TEXT')

        if 'tags' not in columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN tags TEXT DEFAULT '[]'")

        if 'folder_id' not in columns:
            cursor.execute('ALTER TABLE sessions ADD COLUMN folder_id TEXT')

        if 'pinned' not in columns:
            cursor.execute('ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0')

        if 'position' not in columns:
            cursor.execute('ALTER TABLE sessions ADD COLUMN position INTEGER DEFAULT 0')

        if 'mode' not in columns:
            cursor.execute("ALTER TABLE sessions ADD COLUMN mode TEXT DEFAULT 'simple'")

        if 'knowledge_enabled' not in columns:
            cursor.execute('ALTER TABLE sessions ADD COLUMN knowledge_enabled INTEGER DEFAULT 1')

        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                workflow_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        ''')

        # Create indices for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, timestamp)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_workflow
            ON messages(workflow_id)
        ''')

        # Create index for title searching
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_title
            ON sessions(title)
        ''')

        # Create index for folder_id
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_folder
            ON sessions(folder_id)
        ''')

        # Create index for pinned sessions
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sessions_pinned
            ON sessions(pinned, position)
        ''')

        conn.commit()
        conn.close()

    def create_session(self, title: Optional[str] = None) -> str:
        """Create a new chat session and return its ID."""
        session_id = str(uuid.uuid4())[:8]  # Short UUID
        now = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO sessions (session_id, created_at, last_active, message_count, title, tags) VALUES (?, ?, ?, ?, ?, ?)',
            (session_id, now, now, 0, title, '[]')
        )

        conn.commit()
        conn.close()

        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT session_id, created_at, last_active, message_count, title, tags,
                      folder_id, pinned, position, mode, knowledge_enabled
               FROM sessions WHERE session_id = ?''',
            (session_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        tags = json.loads(row[5]) if row[5] else []

        return Session(
            session_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            last_active=datetime.fromisoformat(row[2]),
            message_count=row[3],
            title=row[4],
            tags=tags,
            folder_id=row[6],
            pinned=bool(row[7]),
            position=row[8],
            mode=row[9] or 'simple',
            knowledge_enabled=bool(row[10])
        )

    def list_sessions(self, limit: int = 20) -> List[Session]:
        """List recent sessions, most recent first."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT session_id, created_at, last_active, message_count, title, tags,
                      folder_id, pinned, position, mode, knowledge_enabled
               FROM sessions ORDER BY last_active DESC LIMIT ?''',
            (limit,)
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            Session(
                session_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                last_active=datetime.fromisoformat(row[2]),
                message_count=row[3],
                title=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                folder_id=row[6],
                pinned=bool(row[7]),
                position=row[8],
                mode=row[9] or 'simple',
                knowledge_enabled=bool(row[10])
            )
            for row in rows
        ]

    def get_last_session(self) -> Optional[str]:
        """
        Get the most recently active session ID.

        Returns:
            Session ID of the last active session, or None if no sessions exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT session_id FROM sessions ORDER BY last_active DESC LIMIT 1'
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))

        conn.commit()
        conn.close()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        workflow_id: Optional[str] = None
    ) -> Message:
        """Add a message to a session."""
        now = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert message
        cursor.execute(
            'INSERT INTO messages (session_id, role, content, workflow_id, timestamp) VALUES (?, ?, ?, ?, ?)',
            (session_id, role, content, workflow_id, now)
        )

        message_id = cursor.lastrowid

        # Update session metadata
        cursor.execute(
            'UPDATE sessions SET last_active = ?, message_count = message_count + 1 WHERE session_id = ?',
            (now, session_id)
        )

        conn.commit()
        conn.close()

        return Message(role, content, workflow_id, message_id, now)

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Retrieve messages for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if limit:
            cursor.execute(
                'SELECT id, role, content, workflow_id, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?',
                (session_id, limit, offset)
            )
        else:
            cursor.execute(
                'SELECT id, role, content, workflow_id, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp ASC',
                (session_id,)
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            Message(
                role=row[1],
                content=row[2],
                workflow_id=row[3],
                message_id=row[0],
                timestamp=datetime.fromisoformat(row[4])
            )
            for row in rows
        ]

    def get_recent_context(self, session_id: str, message_count: int = 10) -> List[Message]:
        """Get the most recent N messages for context window."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id, role, content, workflow_id, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?',
            (session_id, message_count)
        )

        rows = cursor.fetchall()
        conn.close()

        # Reverse to get chronological order
        messages = [
            Message(
                role=row[1],
                content=row[2],
                workflow_id=row[3],
                message_id=row[0],
                timestamp=datetime.fromisoformat(row[4])
            )
            for row in reversed(rows)
        ]

        return messages

    def search_messages(self, session_id: str, query: str) -> List[Message]:
        """Search messages by content."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT id, role, content, workflow_id, timestamp FROM messages WHERE session_id = ? AND content LIKE ? ORDER BY timestamp ASC',
            (session_id, f'%{query}%')
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            Message(
                role=row[1],
                content=row[2],
                workflow_id=row[3],
                message_id=row[0],
                timestamp=datetime.fromisoformat(row[4])
            )
            for row in rows
        ]

    def get_last_workflow_id(self, session_id: str) -> Optional[str]:
        """Get the most recent workflow ID from the session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT workflow_id FROM messages WHERE session_id = ? AND workflow_id IS NOT NULL ORDER BY timestamp DESC LIMIT 1',
            (session_id,)
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def update_session_activity(self, session_id: str):
        """Update the last_active timestamp for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET last_active = ? WHERE session_id = ?',
            (datetime.now(), session_id)
        )

        conn.commit()
        conn.close()

    def set_title(self, session_id: str, title: str):
        """Set the title for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET title = ? WHERE session_id = ?',
            (title, session_id)
        )

        conn.commit()
        conn.close()

    def generate_auto_title(self, session_id: str, max_length: int = 50) -> Optional[str]:
        """
        Generate an automatic title from the first user message.

        Args:
            session_id: The session ID
            max_length: Maximum title length in characters

        Returns:
            Generated title or None if no messages exist
        """
        messages = self.get_messages(session_id, limit=1)

        if not messages or messages[0].role != 'user':
            return None

        # Take first line of first message, truncate if needed
        content = messages[0].content.strip()
        first_line = content.split('\n')[0]

        if len(first_line) > max_length:
            title = first_line[:max_length-3] + "..."
        else:
            title = first_line

        self.set_title(session_id, title)
        return title

    def add_tags(self, session_id: str, tags: List[str]):
        """Add tags to a session (duplicates are ignored)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current tags
        cursor.execute('SELECT tags FROM sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return

        current_tags = json.loads(row[0]) if row[0] else []

        # Add new tags (avoid duplicates)
        updated_tags = list(set(current_tags + tags))

        cursor.execute(
            'UPDATE sessions SET tags = ? WHERE session_id = ?',
            (json.dumps(updated_tags), session_id)
        )

        conn.commit()
        conn.close()

    def remove_tags(self, session_id: str, tags: List[str]):
        """Remove tags from a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current tags
        cursor.execute('SELECT tags FROM sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return

        current_tags = json.loads(row[0]) if row[0] else []

        # Remove specified tags
        updated_tags = [tag for tag in current_tags if tag not in tags]

        cursor.execute(
            'UPDATE sessions SET tags = ? WHERE session_id = ?',
            (json.dumps(updated_tags), session_id)
        )

        conn.commit()
        conn.close()

    def search_sessions(self, query: str, limit: int = 20) -> List[Session]:
        """
        Search sessions by keyword in title or messages.

        Args:
            query: Search keyword
            limit: Maximum number of results

        Returns:
            List of matching sessions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Search in titles and get session IDs from messages
        cursor.execute('''
            SELECT DISTINCT s.session_id, s.created_at, s.last_active, s.message_count, s.title, s.tags,
                   s.folder_id, s.pinned, s.position, s.mode, s.knowledge_enabled
            FROM sessions s
            LEFT JOIN messages m ON s.session_id = m.session_id
            WHERE s.title LIKE ? OR m.content LIKE ?
            ORDER BY s.last_active DESC
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))

        rows = cursor.fetchall()
        conn.close()

        return [
            Session(
                session_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                last_active=datetime.fromisoformat(row[2]),
                message_count=row[3],
                title=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                folder_id=row[6],
                pinned=bool(row[7]),
                position=row[8],
                mode=row[9] or 'simple',
                knowledge_enabled=bool(row[10])
            )
            for row in rows
        ]

    def filter_sessions_by_tags(self, tags: List[str], match_all: bool = False, limit: int = 20) -> List[Session]:
        """
        Filter sessions by tags.

        Args:
            tags: List of tags to filter by
            match_all: If True, session must have ALL tags. If False, ANY tag matches.
            limit: Maximum number of results

        Returns:
            List of matching sessions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            '''SELECT session_id, created_at, last_active, message_count, title, tags,
                      folder_id, pinned, position, mode, knowledge_enabled
               FROM sessions ORDER BY last_active DESC'''
        )

        rows = cursor.fetchall()
        conn.close()

        # Filter sessions based on tags
        matching_sessions = []

        for row in rows:
            if len(matching_sessions) >= limit:
                break

            session_tags = json.loads(row[5]) if row[5] else []

            if match_all:
                # Session must have ALL specified tags
                if all(tag in session_tags for tag in tags):
                    matching_sessions.append(Session(
                        session_id=row[0],
                        created_at=datetime.fromisoformat(row[1]),
                        last_active=datetime.fromisoformat(row[2]),
                        message_count=row[3],
                        title=row[4],
                        tags=session_tags,
                        folder_id=row[6],
                        pinned=bool(row[7]),
                        position=row[8],
                        mode=row[9] or 'simple',
                        knowledge_enabled=bool(row[10])
                    ))
            else:
                # Session must have ANY of the specified tags
                if any(tag in session_tags for tag in tags):
                    matching_sessions.append(Session(
                        session_id=row[0],
                        created_at=datetime.fromisoformat(row[1]),
                        last_active=datetime.fromisoformat(row[2]),
                        message_count=row[3],
                        title=row[4],
                        tags=session_tags,
                        folder_id=row[6],
                        pinned=bool(row[7]),
                        position=row[8],
                        mode=row[9] or 'simple',
                        knowledge_enabled=bool(row[10])
                    ))

        return matching_sessions

    def get_sessions_today(self) -> List[Session]:
        """Get all sessions active today."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().date()

        cursor.execute(
            '''SELECT session_id, created_at, last_active, message_count, title, tags,
                      folder_id, pinned, position, mode, knowledge_enabled
               FROM sessions WHERE DATE(last_active) = ? ORDER BY last_active DESC''',
            (today.isoformat(),)
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            Session(
                session_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                last_active=datetime.fromisoformat(row[2]),
                message_count=row[3],
                title=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                folder_id=row[6],
                pinned=bool(row[7]),
                position=row[8],
                mode=row[9] or 'simple',
                knowledge_enabled=bool(row[10])
            )
            for row in rows
        ]

    def export_session(self, session_id: str) -> Dict:
        """
        Export a session and all its messages to a dictionary.

        Args:
            session_id: The session ID to export

        Returns:
            Dictionary containing session metadata and messages
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        messages = self.get_messages(session_id)

        return {
            'session': session.to_dict(),
            'messages': [msg.to_dict() for msg in messages],
            'exported_at': datetime.now().isoformat()
        }

    def import_session(self, data: Dict) -> Optional[str]:
        """
        Import a session from exported data.

        Args:
            data: Dictionary containing session and messages

        Returns:
            The new session ID, or None if import failed
        """
        if 'session' not in data or 'messages' not in data:
            return None

        session_data = data['session']

        # Create new session with imported metadata
        session_id = self.create_session(title=session_data.get('title'))

        # Import messages
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for msg in data['messages']:
            cursor.execute(
                'INSERT INTO messages (session_id, role, content, workflow_id, timestamp) VALUES (?, ?, ?, ?, ?)',
                (session_id, msg['role'], msg['content'], msg.get('workflow_id'), msg['timestamp'])
            )

        # Update session metadata
        if 'tags' in session_data and session_data['tags']:
            cursor.execute(
                'UPDATE sessions SET tags = ?, message_count = ? WHERE session_id = ?',
                (json.dumps(session_data['tags']), len(data['messages']), session_id)
            )
        else:
            cursor.execute(
                'UPDATE sessions SET message_count = ? WHERE session_id = ?',
                (len(data['messages']), session_id)
            )

        conn.commit()
        conn.close()

        return session_id

    # ========== Folder Management Methods ==========

    def create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """
        Create a new folder for organizing sessions.

        Args:
            name: Folder name
            parent_id: Optional parent folder ID for nested folders

        Returns:
            The new folder ID
        """
        folder_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'INSERT INTO folders (folder_id, name, parent_folder_id, position, created_at) VALUES (?, ?, ?, ?, ?)',
            (folder_id, name, parent_id, 0, now)
        )

        conn.commit()
        conn.close()

        return folder_id

    def get_folders(self) -> List[Folder]:
        """
        Get all folders ordered by position.

        Returns:
            List of all folders
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT folder_id, name, parent_folder_id, position, created_at FROM folders ORDER BY position ASC'
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            Folder(
                folder_id=row[0],
                name=row[1],
                parent_folder_id=row[2],
                position=row[3],
                created_at=datetime.fromisoformat(row[4]) if row[4] else None
            )
            for row in rows
        ]

    def get_folder(self, folder_id: str) -> Optional[Folder]:
        """
        Get a single folder by ID.

        Args:
            folder_id: The folder ID

        Returns:
            Folder object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT folder_id, name, parent_folder_id, position, created_at FROM folders WHERE folder_id = ?',
            (folder_id,)
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Folder(
            folder_id=row[0],
            name=row[1],
            parent_folder_id=row[2],
            position=row[3],
            created_at=datetime.fromisoformat(row[4]) if row[4] else None
        )

    def update_folder(self, folder_id: str, name: Optional[str] = None, parent_id: Optional[str] = None):
        """
        Update folder properties.

        Args:
            folder_id: The folder ID to update
            name: New folder name (optional)
            parent_id: New parent folder ID (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if name is not None:
            cursor.execute(
                'UPDATE folders SET name = ? WHERE folder_id = ?',
                (name, folder_id)
            )

        if parent_id is not None:
            cursor.execute(
                'UPDATE folders SET parent_folder_id = ? WHERE folder_id = ?',
                (parent_id, folder_id)
            )

        conn.commit()
        conn.close()

    def delete_folder(self, folder_id: str):
        """
        Delete a folder. Sessions in the folder are moved to root (folder_id = NULL).

        Args:
            folder_id: The folder ID to delete
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Move all sessions in this folder to root
        cursor.execute(
            'UPDATE sessions SET folder_id = NULL WHERE folder_id = ?',
            (folder_id,)
        )

        # Delete the folder
        cursor.execute(
            'DELETE FROM folders WHERE folder_id = ?',
            (folder_id,)
        )

        conn.commit()
        conn.close()

    def move_session_to_folder(self, session_id: str, folder_id: Optional[str]):
        """
        Move a session to a folder (or root if folder_id is None).

        Args:
            session_id: The session ID to move
            folder_id: Target folder ID (None for root)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET folder_id = ? WHERE session_id = ?',
            (folder_id, session_id)
        )

        conn.commit()
        conn.close()

    def set_session_pinned(self, session_id: str, pinned: bool):
        """
        Pin or unpin a session.

        Args:
            session_id: The session ID
            pinned: True to pin, False to unpin
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET pinned = ? WHERE session_id = ?',
            (1 if pinned else 0, session_id)
        )

        conn.commit()
        conn.close()

    def get_sessions_in_folder(self, folder_id: Optional[str]) -> List[Session]:
        """
        Get all sessions in a specific folder.

        Args:
            folder_id: The folder ID (None for root/unorganized sessions)

        Returns:
            List of sessions in the folder
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if folder_id is None:
            cursor.execute(
                '''SELECT session_id, created_at, last_active, message_count, title, tags,
                          folder_id, pinned, position, mode, knowledge_enabled
                   FROM sessions WHERE folder_id IS NULL
                   ORDER BY pinned DESC, position ASC, last_active DESC'''
            )
        else:
            cursor.execute(
                '''SELECT session_id, created_at, last_active, message_count, title, tags,
                          folder_id, pinned, position, mode, knowledge_enabled
                   FROM sessions WHERE folder_id = ?
                   ORDER BY pinned DESC, position ASC, last_active DESC''',
                (folder_id,)
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            Session(
                session_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                last_active=datetime.fromisoformat(row[2]),
                message_count=row[3],
                title=row[4],
                tags=json.loads(row[5]) if row[5] else [],
                folder_id=row[6],
                pinned=bool(row[7]),
                position=row[8],
                mode=row[9] or 'simple',
                knowledge_enabled=bool(row[10])
            )
            for row in rows
        ]

    def update_session_mode(self, session_id: str, mode: str):
        """
        Update the chat mode for a session.

        Args:
            session_id: The session ID
            mode: Chat mode (e.g., 'simple', 'advanced', 'workflow')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET mode = ? WHERE session_id = ?',
            (mode, session_id)
        )

        conn.commit()
        conn.close()

    def update_session_knowledge_enabled(self, session_id: str, enabled: bool):
        """
        Enable or disable knowledge system for a session.

        Args:
            session_id: The session ID
            enabled: True to enable, False to disable
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'UPDATE sessions SET knowledge_enabled = ? WHERE session_id = ?',
            (1 if enabled else 0, session_id)
        )

        conn.commit()
        conn.close()
