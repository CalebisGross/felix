"""Synthesis review bubble widget for low-confidence synthesis approval."""

from typing import Optional, Dict, Any, List
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton, QSizePolicy, QTextEdit,
    QMenu, QInputDialog
)

from ..core.theme import Colors


class SynthesisReviewStatus(Enum):
    """Status of synthesis review."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REGENERATING = "regenerating"


class SynthesisReviewBubble(QFrame):
    """Bubble for reviewing and optionally regenerating low-confidence synthesis.

    Shows synthesis content preview with confidence indicator and provides
    options to accept, reject, or regenerate using various strategies.

    Signals:
        accepted: Emitted when user accepts synthesis (review_id)
        rejected: Emitted when user rejects synthesis (review_id, reason)
        regenerate_requested: Emitted when regeneration requested
            (review_id, strategy, user_input)
    """

    accepted = Signal(str)  # review_id
    rejected = Signal(str, str)  # review_id, reason
    regenerate_requested = Signal(str, str, str)  # review_id, strategy, user_input

    def __init__(
        self,
        review_id: str,
        review_data: Dict[str, Any],
        parent: Optional[QWidget] = None
    ):
        """Initialize the synthesis review bubble.

        Args:
            review_id: Unique identifier for this review
            review_data: Dictionary containing synthesis review information:
                - confidence: Synthesis confidence (0.0-1.0)
                - content_preview: Preview of synthesis content
                - degraded: Whether synthesis is degraded
                - degraded_reason: Reason for degradation
                - validation_score: Validation score (0.0-1.0)
                - validation_flags: List of validation issues
                - agent_count: Number of agents in synthesis
                - task_description: Original task
                - options: List of available options
            parent: Parent widget
        """
        super().__init__(parent)
        self._review_id = review_id
        self._review_data = review_data
        self._status = SynthesisReviewStatus.PENDING
        self._setup_ui()

    def _setup_ui(self):
        """Set up the synthesis review bubble UI."""
        self.setObjectName("synthesisReviewBubble")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        confidence = self._review_data.get('confidence', 0.0)

        # Determine colors based on confidence level
        if confidence < 0.3:
            border_color = Colors.ERROR
            bg_color = "#2d1f1f"
            confidence_text = "Very Low"
        elif confidence < 0.5:
            border_color = Colors.WARNING
            bg_color = "#2d2a1f"
            confidence_text = "Low"
        elif confidence < 0.7:
            border_color = Colors.ACCENT
            bg_color = Colors.SURFACE
            confidence_text = "Moderate"
        else:
            border_color = Colors.SUCCESS
            bg_color = Colors.SURFACE
            confidence_text = "Good"

        self.setStyleSheet(f"""
            QFrame#synthesisReviewBubble {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header with icon and title
        header = QHBoxLayout()
        header.setSpacing(8)

        # Synthesis icon
        icon_label = QLabel("🔬")
        icon_label.setStyleSheet("font-size: 14px;")
        header.addWidget(icon_label)

        title = QLabel("Synthesis Review Required")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-weight: 600;
            font-size: 13px;
        """)
        header.addWidget(title)

        header.addStretch()

        # Status indicator
        self._status_label = QLabel()
        self._update_status_label()
        header.addWidget(self._status_label)

        layout.addLayout(header)

        # Confidence display
        conf_layout = QHBoxLayout()

        conf_label = QLabel(f"Confidence: {confidence:.0%} ({confidence_text})")
        conf_label.setStyleSheet(f"""
            color: {border_color};
            font-weight: 600;
            font-size: 12px;
        """)
        conf_layout.addWidget(conf_label)

        # Agent count
        agent_count = self._review_data.get('agent_count', 0)
        agents_label = QLabel(f"Agents: {agent_count}")
        agents_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
        """)
        conf_layout.addWidget(agents_label)

        conf_layout.addStretch()

        # Meta-confidence if available
        meta_conf = self._review_data.get('meta_confidence')
        if meta_conf is not None and meta_conf != confidence:
            meta_label = QLabel(f"Meta: {meta_conf:.0%}")
            meta_label.setStyleSheet(f"""
                color: {Colors.TEXT_MUTED};
                font-size: 11px;
            """)
            conf_layout.addWidget(meta_label)

        layout.addLayout(conf_layout)

        # Validation flags (if any)
        validation_flags = self._review_data.get('validation_flags', [])
        if validation_flags:
            flags_frame = QFrame()
            flags_frame.setStyleSheet(f"""
                background-color: {Colors.BACKGROUND};
                border-radius: 4px;
            """)
            flags_layout = QVBoxLayout(flags_frame)
            flags_layout.setContentsMargins(8, 6, 8, 6)
            flags_layout.setSpacing(2)

            flags_header = QLabel("⚠️ Validation Issues:")
            flags_header.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 11px;
                font-weight: 500;
            """)
            flags_layout.addWidget(flags_header)

            for flag in validation_flags[:5]:  # Limit to 5 flags
                flag_label = QLabel(f"• {flag}")
                flag_label.setStyleSheet(f"""
                    color: {Colors.TEXT_SECONDARY};
                    font-size: 11px;
                """)
                flags_layout.addWidget(flag_label)

            if len(validation_flags) > 5:
                more_label = QLabel(f"... and {len(validation_flags) - 5} more")
                more_label.setStyleSheet(f"""
                    color: {Colors.TEXT_MUTED};
                    font-size: 10px;
                """)
                flags_layout.addWidget(more_label)

            layout.addWidget(flags_frame)

        # Degradation notice
        if self._review_data.get('degraded'):
            degraded_reason = self._review_data.get('degraded_reason', 'Unknown')
            degraded_label = QLabel(f"⚠️ Degraded: {degraded_reason}")
            degraded_label.setWordWrap(True)
            degraded_label.setStyleSheet(f"""
                color: {Colors.WARNING};
                font-size: 11px;
                padding: 4px 8px;
                background-color: {Colors.BACKGROUND};
                border-radius: 4px;
            """)
            layout.addWidget(degraded_label)

        # Content preview
        preview_frame = QFrame()
        preview_frame.setStyleSheet(f"""
            background-color: {Colors.BACKGROUND};
            border-radius: 4px;
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 6, 8, 6)

        preview_header = QLabel("Content Preview:")
        preview_header.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 11px;
        """)
        preview_layout.addWidget(preview_header)

        content_preview = self._review_data.get('content_preview', '')
        preview_text = QTextEdit()
        preview_text.setPlainText(content_preview)
        preview_text.setReadOnly(True)
        preview_text.setMaximumHeight(150)
        preview_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: transparent;
                color: {Colors.TEXT_PRIMARY};
                border: none;
                font-size: 12px;
            }}
        """)
        preview_layout.addWidget(preview_text)

        layout.addWidget(preview_frame)

        # Action buttons
        self._buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(self._buttons_widget)
        buttons_layout.setContentsMargins(0, 8, 0, 0)
        buttons_layout.setSpacing(8)

        # Reject button
        self._reject_btn = QPushButton("✗ Reject")
        self._reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reject_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.ERROR};
                border: 1px solid {Colors.ERROR};
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.ERROR};
                color: white;
            }}
        """)
        self._reject_btn.clicked.connect(self._on_reject)
        buttons_layout.addWidget(self._reject_btn)

        buttons_layout.addStretch()

        # Regenerate button with dropdown menu
        self._regen_btn = QPushButton("🔄 Regenerate ▼")
        self._regen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._regen_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {Colors.ACCENT};
                border: 1px solid {Colors.ACCENT};
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT};
                color: white;
            }}
        """)

        # Create regeneration menu
        regen_menu = QMenu(self)
        regen_menu.setStyleSheet(f"""
            QMenu {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                color: {Colors.TEXT_PRIMARY};
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.ACCENT};
                color: white;
            }}
        """)

        # Add regeneration options
        options = self._review_data.get('options', [])
        for option in options:
            if option['id'].startswith('regenerate'):
                action = regen_menu.addAction(option['label'])
                strategy = option.get('strategy', option['id'])
                requires_input = option.get('requires_input', False)

                if requires_input:
                    action.triggered.connect(
                        lambda checked, s=strategy: self._on_regenerate_with_input(s)
                    )
                else:
                    action.triggered.connect(
                        lambda checked, s=strategy: self._on_regenerate(s)
                    )

        self._regen_btn.setMenu(regen_menu)
        buttons_layout.addWidget(self._regen_btn)

        # Accept button
        self._accept_btn = QPushButton("✓ Accept")
        self._accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._accept_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: #1a1b26;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #a8d86a;
            }}
        """)
        self._accept_btn.clicked.connect(self._on_accept)
        buttons_layout.addWidget(self._accept_btn)

        layout.addWidget(self._buttons_widget)

        # Update visibility based on status
        self._update_ui_for_status()

    def _update_status_label(self):
        """Update the status indicator."""
        status_config = {
            SynthesisReviewStatus.PENDING: ("Review Required", Colors.WARNING),
            SynthesisReviewStatus.ACCEPTED: ("Accepted", Colors.SUCCESS),
            SynthesisReviewStatus.REJECTED: ("Rejected", Colors.ERROR),
            SynthesisReviewStatus.REGENERATING: ("Regenerating...", Colors.ACCENT),
        }

        text, color = status_config.get(
            self._status,
            ("Unknown", Colors.TEXT_MUTED)
        )

        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"""
            color: {color};
            font-size: 11px;
            font-weight: 500;
        """)

    def _update_ui_for_status(self):
        """Update UI elements based on current status."""
        # Show/hide buttons based on status
        show_buttons = self._status == SynthesisReviewStatus.PENDING
        self._buttons_widget.setVisible(show_buttons)

    def _on_accept(self):
        """Handle accept button click."""
        self._status = SynthesisReviewStatus.ACCEPTED
        self._update_status_label()
        self._update_ui_for_status()
        self.accepted.emit(self._review_id)

    def _on_reject(self):
        """Handle reject button click."""
        self._status = SynthesisReviewStatus.REJECTED
        self._update_status_label()
        self._update_ui_for_status()
        self.rejected.emit(self._review_id, "User rejected synthesis")

    def _on_regenerate(self, strategy: str):
        """Handle regeneration without input."""
        self._status = SynthesisReviewStatus.REGENERATING
        self._update_status_label()
        self._update_ui_for_status()
        self.regenerate_requested.emit(self._review_id, strategy, "")

    def _on_regenerate_with_input(self, strategy: str):
        """Handle regeneration that requires user input (context injection)."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Add Context",
            "Provide additional context to improve the synthesis:",
            ""
        )

        if ok and text:
            self._status = SynthesisReviewStatus.REGENERATING
            self._update_status_label()
            self._update_ui_for_status()
            self.regenerate_requested.emit(self._review_id, strategy, text)

    def set_status(self, status: SynthesisReviewStatus):
        """Update the review status."""
        self._status = status
        self._update_status_label()
        self._update_ui_for_status()

    def get_review_id(self) -> str:
        """Get the review ID."""
        return self._review_id

    def get_status(self) -> SynthesisReviewStatus:
        """Get current status."""
        return self._status

    def get_review_data(self) -> Dict[str, Any]:
        """Get the review data."""
        return self._review_data
