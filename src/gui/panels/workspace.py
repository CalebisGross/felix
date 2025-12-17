"""Main workspace panel with chat area and input."""

from typing import Optional
from datetime import datetime

from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QComboBox, QLabel, QFrame,
    QScrollArea, QSizePolicy, QSpacerItem
)
from PySide6.QtGui import QTextCursor, QFont

from ..core.theme import Colors
from ..widgets.message_bubble import UserBubble, AssistantBubble, StreamingBubble, SystemBubble
from ..widgets.typing_indicator import TypingIndicator
from ..widgets.action_bubble import ActionBubble, ActionStatus
from ..widgets.progress_bubble import ProgressBubble, WorkflowStepStatus
from ..models.message_model import Message, MessageRole, MessageModel


class MessageArea(QScrollArea):
    """Scrollable area containing message bubbles."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("messageArea")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.setStyleSheet(f"""
            QScrollArea#messageArea {{
                background-color: {Colors.BACKGROUND};
                border: none;
            }}
        """)

        # Container widget
        self._container = QWidget()
        self._container.setStyleSheet(f"background-color: {Colors.BACKGROUND};")

        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(12)
        self._layout.addStretch()  # Push messages to top

        self.setWidget(self._container)

        # Track current streaming bubble
        self._streaming_bubble: Optional[StreamingBubble] = None
        self._typing_indicator: Optional[TypingIndicator] = None
        self._progress_bubble: Optional[ProgressBubble] = None

    def add_user_bubble(self, content: str) -> UserBubble:
        """Add a user message bubble."""
        bubble = UserBubble(content)
        self._insert_bubble(bubble)
        return bubble

    def add_assistant_bubble(self, content: str) -> AssistantBubble:
        """Add an assistant message bubble."""
        bubble = AssistantBubble(content)
        self._insert_bubble(bubble)
        return bubble

    def add_system_bubble(self, content: str) -> SystemBubble:
        """Add a system message bubble."""
        bubble = SystemBubble(content)
        self._insert_bubble(bubble)
        return bubble

    def add_action_bubble(
        self,
        action_id: str,
        command: str,
        description: str = "",
        risk_level: str = "low"
    ) -> ActionBubble:
        """Add an action (command approval) bubble."""
        bubble = ActionBubble(action_id, command, description, risk_level)
        self._insert_bubble(bubble)
        return bubble

    def start_streaming(self) -> StreamingBubble:
        """Start streaming mode with a streaming bubble."""
        # Hide typing indicator if shown
        self.hide_typing()

        # Create streaming bubble
        self._streaming_bubble = StreamingBubble()
        self._insert_bubble(self._streaming_bubble)
        return self._streaming_bubble

    def append_streaming_chunk(self, chunk: str):
        """Append chunk to current streaming bubble."""
        if self._streaming_bubble:
            self._streaming_bubble.append_content(chunk)
            self._scroll_to_bottom()

    def end_streaming(self) -> Optional[str]:
        """End streaming mode and return content."""
        content = None
        if self._streaming_bubble:
            self._streaming_bubble.set_streaming(False)
            content = self._streaming_bubble.get_content()
            self._streaming_bubble = None
        return content

    def show_typing(self):
        """Show typing indicator."""
        if not self._typing_indicator:
            self._typing_indicator = TypingIndicator()
            self._insert_bubble(self._typing_indicator)

    def hide_typing(self):
        """Hide typing indicator."""
        if self._typing_indicator:
            self._layout.removeWidget(self._typing_indicator)
            self._typing_indicator.deleteLater()
            self._typing_indicator = None

    def show_progress(self, title: str = "Processing...") -> ProgressBubble:
        """Show workflow progress bubble."""
        # Hide typing indicator if shown
        self.hide_typing()

        # Create progress bubble
        self._progress_bubble = ProgressBubble(title)
        self._insert_bubble(self._progress_bubble)
        return self._progress_bubble

    def update_progress_step(self, step: str, description: str = ""):
        """Update the current step in progress bubble."""
        if self._progress_bubble:
            self._progress_bubble.set_current_step(step, description)
            self._scroll_to_bottom()

    def update_progress_agent(self, agent: str, activity: str = ""):
        """Update agent activity in progress bubble."""
        if self._progress_bubble:
            self._progress_bubble.set_agent_activity(agent, activity)

    def set_progress_value(self, current: int, total: int):
        """Set determinate progress value."""
        if self._progress_bubble:
            self._progress_bubble.set_progress(current, total)

    def complete_progress(self, success: bool = True):
        """Mark progress as complete."""
        if self._progress_bubble:
            self._progress_bubble.set_complete(success)

    def hide_progress(self):
        """Hide and clean up progress bubble."""
        if self._progress_bubble:
            self._progress_bubble.cleanup()
            self._layout.removeWidget(self._progress_bubble)
            self._progress_bubble.deleteLater()
            self._progress_bubble = None

    def get_progress_bubble(self) -> Optional[ProgressBubble]:
        """Get the current progress bubble if any."""
        return self._progress_bubble

    def _insert_bubble(self, bubble: QWidget):
        """Insert a bubble before the stretch."""
        # Insert before the stretch (last item)
        count = self._layout.count()
        self._layout.insertWidget(count - 1, bubble)
        # Schedule scroll to bottom
        QTimer.singleShot(10, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Scroll to the bottom of the message area."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear all messages."""
        # Clean up progress bubble
        if self._progress_bubble:
            self._progress_bubble.cleanup()

        # Remove all widgets except the stretch
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._streaming_bubble = None
        self._typing_indicator = None
        self._progress_bubble = None


class InputArea(QWidget):
    """Input area with text field, mode selector, send and stop buttons.

    Signals:
        message_submitted: Emitted with (message, mode) when send is triggered
        stop_requested: Emitted when stop button is clicked
    """

    message_submitted = Signal(str, str)  # message, mode
    stop_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("inputArea")
        self._is_processing = False
        self._setup_ui()

    def _setup_ui(self):
        """Set up input area UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Container with border
        container = QFrame()
        container.setObjectName("inputContainer")
        container.setStyleSheet(f"""
            QFrame#inputContainer {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
            }}
        """)

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(12, 12, 12, 12)
        container_layout.setSpacing(8)

        # Text input
        self._input = QTextEdit()
        self._input.setObjectName("messageInput")
        self._input.setPlaceholderText("Type your message... (Ctrl+Enter to send)")
        self._input.setAcceptRichText(False)
        self._input.setMinimumHeight(60)
        self._input.setMaximumHeight(150)
        self._input.setStyleSheet(f"""
            QTextEdit#messageInput {{
                background-color: transparent;
                color: {Colors.TEXT_PRIMARY};
                border: none;
                font-size: 14px;
            }}
        """)
        container_layout.addWidget(self._input)

        # Bottom row: mode selector + buttons
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        # Mode selector
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Simple", "Workflow", "Auto"])
        self._mode_combo.setCurrentText("Auto")
        self._mode_combo.setToolTip("Processing mode:\n"
                                    "- Simple: Direct Felix response\n"
                                    "- Workflow: Multi-agent orchestration\n"
                                    "- Auto: Felix decides based on task")
        self._mode_combo.setFixedWidth(100)
        bottom_row.addWidget(self._mode_combo)

        bottom_row.addStretch()

        # Stop button (hidden by default)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setProperty("danger", True)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setFixedWidth(80)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.hide()
        bottom_row.addWidget(self._stop_btn)

        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setProperty("primary", True)
        self._send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._send_btn.setFixedWidth(80)
        self._send_btn.clicked.connect(self._on_send)
        bottom_row.addWidget(self._send_btn)

        container_layout.addLayout(bottom_row)
        layout.addWidget(container)

        # Connect Ctrl+Enter shortcut
        self._input.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle keyboard shortcuts."""
        from PySide6.QtCore import QEvent

        if obj == self._input and event.type() == QEvent.Type.KeyPress:
            key_event = event
            # Ctrl+Enter to send
            if (key_event.key() == Qt.Key.Key_Return and
                key_event.modifiers() == Qt.KeyboardModifier.ControlModifier):
                self._on_send()
                return True
            # Escape to stop
            if key_event.key() == Qt.Key.Key_Escape and self._is_processing:
                self._on_stop()
                return True

        return super().eventFilter(obj, event)

    def _on_send(self):
        """Handle send action."""
        if self._is_processing:
            return

        message = self._input.toPlainText().strip()
        if not message:
            return

        mode_map = {
            "Simple": "direct",
            "Workflow": "full",
            "Auto": "auto"
        }
        mode = mode_map.get(self._mode_combo.currentText(), "auto")

        self.message_submitted.emit(message, mode)

    def _on_stop(self):
        """Handle stop action."""
        self.stop_requested.emit()

    def clear(self):
        """Clear the input field."""
        self._input.clear()

    def set_processing(self, processing: bool):
        """Set processing state."""
        self._is_processing = processing
        self._input.setEnabled(not processing)
        self._send_btn.setVisible(not processing)
        self._stop_btn.setVisible(processing)
        self._mode_combo.setEnabled(not processing)

    def set_enabled(self, enabled: bool):
        """Enable or disable input."""
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._mode_combo.setEnabled(enabled)

    def focus_input(self):
        """Set focus to input field."""
        self._input.setFocus()


class Workspace(QWidget):
    """Main workspace with chat display and input.

    Signals:
        message_submitted: Forwarded from InputArea (message, mode)
        stop_requested: Forwarded from InputArea
        new_chat_requested: Emitted when New Chat is clicked
        action_approved: Emitted when action is approved (action_id)
        action_denied: Emitted when action is denied (action_id)
    """

    message_submitted = Signal(str, str)  # message, mode
    stop_requested = Signal()
    new_chat_requested = Signal()
    action_approved = Signal(str)  # action_id
    action_denied = Signal(str)  # action_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("workspace")
        self._model = MessageModel()
        self._action_bubbles = {}  # action_id -> ActionBubble
        self._setup_ui()

    def _setup_ui(self):
        """Set up workspace UI."""
        self.setStyleSheet(f"""
            QWidget#workspace {{
                background-color: {Colors.BACKGROUND};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-bottom: 1px solid {Colors.BORDER};
        """)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)

        self._title = QLabel("Chat")
        self._title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 16px;
            font-weight: 600;
        """)
        header_layout.addWidget(self._title)
        header_layout.addStretch()

        # New chat button
        new_chat_btn = QPushButton("New Chat")
        new_chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_chat_btn.clicked.connect(self._on_new_chat)
        header_layout.addWidget(new_chat_btn)

        layout.addWidget(header)

        # Message area with bubbles
        self._message_area = MessageArea()
        layout.addWidget(self._message_area, 1)

        # Input area
        input_container = QFrame()
        input_container.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-top: 1px solid {Colors.BORDER};
        """)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(16, 12, 16, 12)

        self._input_area = InputArea()
        self._input_area.message_submitted.connect(self.message_submitted.emit)
        self._input_area.stop_requested.connect(self.stop_requested.emit)
        input_layout.addWidget(self._input_area)

        layout.addWidget(input_container)

    def _on_new_chat(self):
        """Handle new chat button."""
        self._message_area.clear()
        self._model.clear()
        self._action_bubbles.clear()
        self.new_chat_requested.emit()

    def set_title(self, title: str):
        """Set the chat title."""
        self._title.setText(title)

    @Slot(str, str)
    def add_message(self, role: str, content: str):
        """Add a message to the display."""
        if role == "user":
            self._message_area.add_user_bubble(content)
            self._model.add_user_message(content)
        elif role == "assistant":
            self._message_area.add_assistant_bubble(content)
            self._model.add_assistant_message(content)
        else:
            self._message_area.add_system_bubble(content)
            self._model.add_system_message(content)

    def add_action(
        self,
        action_id: str,
        command: str,
        description: str = "",
        risk_level: str = "low"
    ) -> ActionBubble:
        """Add an action bubble for command approval."""
        bubble = self._message_area.add_action_bubble(
            action_id, command, description, risk_level
        )

        # Connect signals
        bubble.approved.connect(self._on_action_approved)
        bubble.denied.connect(self._on_action_denied)

        self._action_bubbles[action_id] = bubble
        return bubble

    def _on_action_approved(self, action_id: str):
        """Handle action approval."""
        self.action_approved.emit(action_id)

    def _on_action_denied(self, action_id: str):
        """Handle action denial."""
        self.action_denied.emit(action_id)

    def update_action_status(self, action_id: str, status: ActionStatus, output: str = ""):
        """Update an action bubble's status."""
        if action_id in self._action_bubbles:
            bubble = self._action_bubbles[action_id]
            bubble.set_status(status)
            if output:
                bubble.set_output(output)

    @Slot()
    def show_typing(self):
        """Show typing indicator."""
        self._message_area.show_typing()

    @Slot()
    def hide_typing(self):
        """Hide typing indicator."""
        self._message_area.hide_typing()

    @Slot()
    def start_streaming(self):
        """Start streaming mode."""
        self._message_area.start_streaming()

    @Slot(str)
    def append_chunk(self, chunk: str):
        """Append streaming chunk."""
        self._message_area.append_streaming_chunk(chunk)

    @Slot()
    def end_streaming(self) -> Optional[str]:
        """End streaming mode and return the content."""
        content = self._message_area.end_streaming()
        if content:
            self._model.add_assistant_message(content)
        return content

    def clear_input(self):
        """Clear the input field."""
        self._input_area.clear()

    def set_input_enabled(self, enabled: bool):
        """Enable or disable input."""
        self._input_area.set_enabled(enabled)

    def set_processing(self, processing: bool):
        """Set processing state (shows stop button)."""
        self._input_area.set_processing(processing)

    def focus_input(self):
        """Focus the input field."""
        self._input_area.focus_input()

    def get_model(self) -> MessageModel:
        """Get the message model."""
        return self._model

    def get_conversation_history(self):
        """Get conversation history for API calls."""
        return self._model.get_conversation_history()

    # Progress methods

    @Slot(str)
    def show_progress(self, title: str = "Processing...") -> ProgressBubble:
        """Show workflow progress indicator."""
        return self._message_area.show_progress(title)

    @Slot(str, str)
    def update_progress_step(self, step: str, description: str = ""):
        """Update the current step in progress display."""
        self._message_area.update_progress_step(step, description)

    @Slot(str, str)
    def update_progress_agent(self, agent: str, activity: str = ""):
        """Update agent activity in progress display."""
        self._message_area.update_progress_agent(agent, activity)

    @Slot(int, int)
    def set_progress_value(self, current: int, total: int):
        """Set determinate progress value."""
        self._message_area.set_progress_value(current, total)

    @Slot(bool)
    def complete_progress(self, success: bool = True):
        """Mark progress as complete."""
        self._message_area.complete_progress(success)

    @Slot()
    def hide_progress(self):
        """Hide workflow progress indicator."""
        self._message_area.hide_progress()

    def get_progress_bubble(self) -> Optional[ProgressBubble]:
        """Get current progress bubble for advanced control."""
        return self._message_area.get_progress_bubble()
