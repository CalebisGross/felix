"""Message bubble widgets for chat display."""

from typing import Optional
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QSizePolicy, QPushButton, QTextEdit
)
from PySide6.QtGui import QFont, QTextCursor

from ..core.theme import Colors


class MessageBubble(QFrame):
    """Base message bubble widget.

    Displays a chat message with role label, timestamp, and content.
    Subclasses customize appearance for different roles.
    """

    copy_requested = Signal(str)  # content to copy

    def __init__(
        self,
        content: str = "",
        timestamp: Optional[datetime] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._content = content
        self._timestamp = timestamp or datetime.now()
        self._setup_ui()

    def _setup_ui(self):
        """Set up the bubble UI."""
        self.setObjectName("messageBubble")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row (role + timestamp)
        header = QHBoxLayout()
        header.setSpacing(8)

        self._role_label = QLabel(self._get_role_text())
        self._role_label.setStyleSheet(f"""
            color: {self._get_role_color()};
            font-weight: 600;
            font-size: 13px;
        """)
        header.addWidget(self._role_label)

        self._time_label = QLabel(self._timestamp.strftime("%H:%M"))
        self._time_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
        """)
        header.addWidget(self._time_label)

        header.addStretch()

        # Copy button (appears on hover via stylesheet)
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedSize(50, 24)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-size: 11px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE_LIGHT};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self._copy_btn.clicked.connect(self._on_copy)
        self._copy_btn.hide()  # Show on hover
        header.addWidget(self._copy_btn)

        layout.addLayout(header)

        # Content
        self._content_label = QLabel(self._content)
        self._content_label.setWordWrap(True)
        self._content_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self._content_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 14px;
            line-height: 1.5;
        """)
        layout.addWidget(self._content_label)

        # Apply bubble styling
        self._apply_style()

    def _get_role_text(self) -> str:
        """Get role label text. Override in subclasses."""
        return "Message"

    def _get_role_color(self) -> str:
        """Get role label color. Override in subclasses."""
        return Colors.TEXT_SECONDARY

    def _get_bubble_color(self) -> str:
        """Get bubble background color. Override in subclasses."""
        return Colors.SURFACE

    def _apply_style(self):
        """Apply bubble styling."""
        self.setStyleSheet(f"""
            QFrame#messageBubble {{
                background-color: {self._get_bubble_color()};
                border-radius: 12px;
                border: none;
            }}
        """)

    def _on_copy(self):
        """Handle copy button click."""
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._content)
        self.copy_requested.emit(self._content)

    def enterEvent(self, event):
        """Show copy button on hover."""
        self._copy_btn.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide copy button when not hovering."""
        self._copy_btn.hide()
        super().leaveEvent(event)

    def set_content(self, content: str):
        """Update the message content."""
        self._content = content
        self._content_label.setText(content)

    def append_content(self, chunk: str):
        """Append content (for streaming)."""
        self._content += chunk
        self._content_label.setText(self._content)

    def get_content(self) -> str:
        """Get the message content."""
        return self._content


class UserBubble(MessageBubble):
    """Message bubble for user messages."""

    def _get_role_text(self) -> str:
        return "You"

    def _get_role_color(self) -> str:
        return Colors.ACCENT

    def _get_bubble_color(self) -> str:
        return Colors.USER_BUBBLE


class AssistantBubble(MessageBubble):
    """Message bubble for assistant (Felix) messages.

    Supports streaming mode where content is appended incrementally.
    """

    def __init__(
        self,
        content: str = "",
        timestamp: Optional[datetime] = None,
        streaming: bool = False,
        parent: Optional[QWidget] = None
    ):
        self._streaming = streaming
        super().__init__(content, timestamp, parent)

    def _get_role_text(self) -> str:
        return "Felix"

    def _get_role_color(self) -> str:
        return Colors.SUCCESS

    def _get_bubble_color(self) -> str:
        return Colors.ASSISTANT_BUBBLE

    def set_streaming(self, streaming: bool):
        """Set streaming mode."""
        self._streaming = streaming

    def is_streaming(self) -> bool:
        """Check if in streaming mode."""
        return self._streaming


class SystemBubble(MessageBubble):
    """Message bubble for system messages."""

    def _get_role_text(self) -> str:
        return "System"

    def _get_role_color(self) -> str:
        return Colors.WARNING

    def _get_bubble_color(self) -> str:
        return Colors.SURFACE_LIGHT


class StreamingBubble(AssistantBubble):
    """Assistant bubble optimized for streaming content.

    Uses QTextEdit internally for better performance with
    frequent content updates.
    """

    def __init__(
        self,
        timestamp: Optional[datetime] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__("", timestamp, streaming=True, parent=parent)

    def _setup_ui(self):
        """Set up the streaming bubble UI."""
        self.setObjectName("messageBubble")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)

        self._role_label = QLabel(self._get_role_text())
        self._role_label.setStyleSheet(f"""
            color: {self._get_role_color()};
            font-weight: 600;
            font-size: 13px;
        """)
        header.addWidget(self._role_label)

        self._time_label = QLabel(self._timestamp.strftime("%H:%M"))
        self._time_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
        """)
        header.addWidget(self._time_label)

        header.addStretch()

        # Copy button
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedSize(50, 24)
        self._copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.TEXT_MUTED};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE_LIGHT};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        self._copy_btn.clicked.connect(self._on_copy)
        self._copy_btn.hide()
        header.addWidget(self._copy_btn)

        layout.addLayout(header)

        # Content area - use QTextEdit for streaming performance
        self._content_edit = QTextEdit()
        self._content_edit.setReadOnly(True)
        self._content_edit.setFrameShape(QFrame.Shape.NoFrame)
        self._content_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._content_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {Colors.TEXT_PRIMARY};
                font-size: 14px;
                border: none;
                padding: 0;
            }}
        """)
        # Auto-resize based on content
        self._content_edit.document().contentsChanged.connect(self._adjust_height)
        layout.addWidget(self._content_edit)

        self._apply_style()

    def _adjust_height(self):
        """Adjust height based on content."""
        doc_height = self._content_edit.document().size().height()
        self._content_edit.setFixedHeight(int(doc_height) + 10)

    def append_content(self, chunk: str):
        """Append streaming content efficiently."""
        self._content += chunk
        cursor = self._content_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(chunk)

    def set_content(self, content: str):
        """Set full content."""
        self._content = content
        self._content_edit.setPlainText(content)

    def get_content(self) -> str:
        """Get the content."""
        return self._content_edit.toPlainText()

    def _on_copy(self):
        """Handle copy."""
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.get_content())
