"""Typing indicator widget showing Felix is processing."""

from typing import Optional

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QFrame
from PySide6.QtGui import QPainter, QColor, QBrush

from ..core.theme import Colors


class DotWidget(QWidget):
    """Single animated dot for typing indicator."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._opacity = 0.3

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, value: float):
        self._opacity = value
        self.update()

    opacity = Property(float, get_opacity, set_opacity)

    def paintEvent(self, event):
        """Paint the dot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = QColor(Colors.ACCENT)
        color.setAlphaF(self._opacity)

        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 8, 8)


class TypingIndicator(QFrame):
    """Animated typing indicator showing Felix is thinking.

    Shows three dots that pulse in sequence to indicate processing.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("typingIndicator")
        self._animations = []
        self._setup_ui()

    def _setup_ui(self):
        """Set up the typing indicator UI."""
        self.setStyleSheet(f"""
            QFrame#typingIndicator {{
                background-color: {Colors.ASSISTANT_BUBBLE};
                border-radius: 12px;
                padding: 8px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Label
        label = QLabel("Felix is thinking")
        label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 13px;
        """)
        layout.addWidget(label)

        # Dots container
        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(4)

        self._dots = []
        for i in range(3):
            dot = DotWidget()
            self._dots.append(dot)
            dots_layout.addWidget(dot)

            # Create animation for this dot
            anim = QPropertyAnimation(dot, b"opacity")
            anim.setDuration(600)
            anim.setStartValue(0.3)
            anim.setKeyValueAt(0.5, 1.0)
            anim.setEndValue(0.3)
            anim.setEasingCurve(QEasingCurve.Type.InOutSine)
            anim.setLoopCount(-1)  # Loop forever
            self._animations.append(anim)

        layout.addLayout(dots_layout)
        layout.addStretch()

    def start(self):
        """Start the animation."""
        # Start animations with staggered delays
        for i, anim in enumerate(self._animations):
            QTimer.singleShot(i * 150, anim.start)

    def stop(self):
        """Stop the animation."""
        for anim in self._animations:
            anim.stop()

        # Reset dots
        for dot in self._dots:
            dot.set_opacity(0.3)

    def showEvent(self, event):
        """Start animation when shown."""
        super().showEvent(event)
        self.start()

    def hideEvent(self, event):
        """Stop animation when hidden."""
        self.stop()
        super().hideEvent(event)


class CompactTypingIndicator(QWidget):
    """Compact typing indicator for inline use."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_dots)
        self._dot_count = 0
        self._setup_ui()

    def _setup_ui(self):
        """Set up compact indicator."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel("Felix is typing")
        self._label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 12px;
            font-style: italic;
        """)
        layout.addWidget(self._label)

    def _update_dots(self):
        """Update the dot animation."""
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self._label.setText(f"Felix is typing{dots}")

    def start(self):
        """Start animation."""
        self._dot_count = 0
        self._timer.start(400)

    def stop(self):
        """Stop animation."""
        self._timer.stop()
        self._label.setText("Felix is typing")

    def showEvent(self, event):
        """Start when shown."""
        super().showEvent(event)
        self.start()

    def hideEvent(self, event):
        """Stop when hidden."""
        self.stop()
        super().hideEvent(event)
