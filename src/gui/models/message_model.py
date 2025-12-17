"""Message model for chat conversations."""

from typing import Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal


class MessageRole(Enum):
    """Role of a message sender."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ACTION = "action"  # For command approval bubbles


@dataclass
class Message:
    """A single chat message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Optional metadata
    confidence: Optional[float] = None
    mode_used: Optional[str] = None
    thinking_steps: Optional[List[dict]] = None

    # For action messages
    action_id: Optional[str] = None
    action_status: Optional[str] = None
    command: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "mode_used": self.mode_used,
            "action_id": self.action_id,
            "action_status": self.action_status,
            "command": self.command,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            confidence=data.get("confidence"),
            mode_used=data.get("mode_used"),
            action_id=data.get("action_id"),
            action_status=data.get("action_status"),
            command=data.get("command"),
        )


class MessageModel(QAbstractListModel):
    """Qt model for chat messages.

    Provides efficient updates for Qt views and supports
    streaming message updates.

    Custom Roles:
        ContentRole: Message content text
        RoleRole: Message role (user/assistant/system)
        TimestampRole: Message timestamp
        MessageObjectRole: Full Message object
    """

    # Custom roles
    ContentRole = Qt.ItemDataRole.UserRole + 1
    RoleRole = Qt.ItemDataRole.UserRole + 2
    TimestampRole = Qt.ItemDataRole.UserRole + 3
    ConfidenceRole = Qt.ItemDataRole.UserRole + 4
    MessageObjectRole = Qt.ItemDataRole.UserRole + 5

    # Signals
    message_added = Signal(Message)
    message_updated = Signal(str)  # message_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._messages: List[Message] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return number of messages."""
        if parent.isValid():
            return 0
        return len(self._messages)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Return data for the given index and role."""
        if not index.isValid() or index.row() >= len(self._messages):
            return None

        message = self._messages[index.row()]

        if role == Qt.ItemDataRole.DisplayRole or role == self.ContentRole:
            return message.content
        elif role == self.RoleRole:
            return message.role.value
        elif role == self.TimestampRole:
            return message.timestamp
        elif role == self.ConfidenceRole:
            return message.confidence
        elif role == self.MessageObjectRole:
            return message

        return None

    def roleNames(self) -> dict:
        """Return role names for QML compatibility."""
        return {
            self.ContentRole: b"content",
            self.RoleRole: b"role",
            self.TimestampRole: b"timestamp",
            self.ConfidenceRole: b"confidence",
            self.MessageObjectRole: b"message",
        }

    def add_message(self, message: Message) -> int:
        """Add a message to the model.

        Returns:
            Index of the new message
        """
        row = len(self._messages)
        self.beginInsertRows(QModelIndex(), row, row)
        self._messages.append(message)
        self.endInsertRows()
        self.message_added.emit(message)
        return row

    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        message = Message(role=MessageRole.USER, content=content)
        self.add_message(message)
        return message

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """Add an assistant message."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            confidence=kwargs.get("confidence"),
            mode_used=kwargs.get("mode_used"),
            thinking_steps=kwargs.get("thinking_steps"),
        )
        self.add_message(message)
        return message

    def add_system_message(self, content: str) -> Message:
        """Add a system message."""
        message = Message(role=MessageRole.SYSTEM, content=content)
        self.add_message(message)
        return message

    def add_action_message(
        self,
        action_id: str,
        command: str,
        status: str = "pending"
    ) -> Message:
        """Add an action (command approval) message."""
        message = Message(
            role=MessageRole.ACTION,
            content=command,
            action_id=action_id,
            action_status=status,
            command=command,
        )
        self.add_message(message)
        return message

    def update_message(self, message_id: str, **kwargs):
        """Update a message by ID.

        Args:
            message_id: ID of message to update
            **kwargs: Fields to update (content, action_status, etc.)
        """
        for i, message in enumerate(self._messages):
            if message.id == message_id:
                for key, value in kwargs.items():
                    if hasattr(message, key):
                        setattr(message, key, value)

                index = self.index(i)
                self.dataChanged.emit(index, index)
                self.message_updated.emit(message_id)
                break

    def update_last_message(self, content: str):
        """Update the content of the last message.

        Useful for streaming where content is appended incrementally.
        """
        if self._messages:
            self._messages[-1].content = content
            index = self.index(len(self._messages) - 1)
            self.dataChanged.emit(index, index, [self.ContentRole])

    def append_to_last_message(self, chunk: str):
        """Append content to the last message (for streaming)."""
        if self._messages:
            self._messages[-1].content += chunk
            index = self.index(len(self._messages) - 1)
            self.dataChanged.emit(index, index, [self.ContentRole])

    def get_message(self, index: int) -> Optional[Message]:
        """Get message by index."""
        if 0 <= index < len(self._messages):
            return self._messages[index]
        return None

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        for message in self._messages:
            if message.id == message_id:
                return message
        return None

    def get_last_message(self) -> Optional[Message]:
        """Get the last message."""
        return self._messages[-1] if self._messages else None

    def get_all_messages(self) -> List[Message]:
        """Get all messages."""
        return self._messages.copy()

    def get_conversation_history(self) -> List[dict]:
        """Get messages in format suitable for Felix API."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self._messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]

    def clear(self):
        """Clear all messages."""
        self.beginResetModel()
        self._messages.clear()
        self.endResetModel()

    def remove_message(self, index: int):
        """Remove message at index."""
        if 0 <= index < len(self._messages):
            self.beginRemoveRows(QModelIndex(), index, index)
            del self._messages[index]
            self.endRemoveRows()
