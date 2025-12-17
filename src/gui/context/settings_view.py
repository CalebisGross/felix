"""Quick settings view for common configuration."""

import logging
from typing import Optional, Dict, Any

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QComboBox, QCheckBox,
    QLineEdit, QScrollArea, QGroupBox, QSpinBox,
    QMessageBox
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class SettingsView(QWidget):
    """Quick settings view.

    Features:
    - LLM provider selection
    - Knowledge brain toggle
    - Theme toggle
    - Common feature toggles
    """

    settings_changed = Signal(dict)  # Emitted when settings change

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._felix_system = None
        self._config = {}
        self._setup_ui()

    def _setup_ui(self):
        """Set up settings view UI."""
        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.BACKGROUND};
                border: none;
            }}
        """)

        # Container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)

        # LLM Settings group
        llm_group = self._create_llm_group()
        layout.addWidget(llm_group)

        # Features group
        features_group = self._create_features_group()
        layout.addWidget(features_group)

        # Processing group
        processing_group = self._create_processing_group()
        layout.addWidget(processing_group)

        # Add stretch at bottom
        layout.addStretch()

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._status_label)

        scroll.setWidget(container)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _create_llm_group(self) -> QGroupBox:
        """Create LLM settings group."""
        group = QGroupBox("LLM Settings")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: 600;
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        # Provider selection
        provider_row = QHBoxLayout()
        provider_label = QLabel("Provider:")
        provider_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        provider_label.setFixedWidth(80)
        provider_row.addWidget(provider_label)

        self._provider_combo = QComboBox()
        self._provider_combo.addItems([
            "LM Studio",
            "Anthropic",
            "OpenAI",
            "Google (Gemini)",
            "Router (Multi)"
        ])
        self._provider_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        self._provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_row.addWidget(self._provider_combo, 1)
        layout.addLayout(provider_row)

        # Model selection
        model_row = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        model_label.setFixedWidth(80)
        model_row.addWidget(model_label)

        self._model_input = QLineEdit()
        self._model_input.setPlaceholderText("e.g., llama-3.1-8b")
        self._model_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}
        """)
        model_row.addWidget(self._model_input, 1)
        layout.addLayout(model_row)

        # Connection status
        status_row = QHBoxLayout()
        status_label = QLabel("Status:")
        status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        status_label.setFixedWidth(80)
        status_row.addWidget(status_label)

        self._llm_status = QLabel("Not connected")
        self._llm_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        status_row.addWidget(self._llm_status)
        status_row.addStretch()

        test_btn = QPushButton("Test")
        test_btn.setFixedWidth(50)
        test_btn.clicked.connect(self._test_llm_connection)
        status_row.addWidget(test_btn)
        layout.addLayout(status_row)

        return group

    def _create_features_group(self) -> QGroupBox:
        """Create features toggle group."""
        group = QGroupBox("Features")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: 600;
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        checkbox_style = f"""
            QCheckBox {{
                color: {Colors.TEXT_SECONDARY};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                background-color: {Colors.SURFACE};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.ACCENT};
                border-color: {Colors.ACCENT};
            }}
        """

        # Knowledge brain toggle
        self._kb_checkbox = QCheckBox("Enable Knowledge Brain")
        self._kb_checkbox.setChecked(True)
        self._kb_checkbox.setStyleSheet(checkbox_style)
        self._kb_checkbox.stateChanged.connect(self._on_setting_changed)
        layout.addWidget(self._kb_checkbox)

        # Web search toggle
        self._web_checkbox = QCheckBox("Enable Web Search")
        self._web_checkbox.setChecked(False)
        self._web_checkbox.setStyleSheet(checkbox_style)
        self._web_checkbox.stateChanged.connect(self._on_setting_changed)
        layout.addWidget(self._web_checkbox)

        # Command approval toggle
        self._approval_checkbox = QCheckBox("Require Command Approval")
        self._approval_checkbox.setChecked(True)
        self._approval_checkbox.setStyleSheet(checkbox_style)
        self._approval_checkbox.stateChanged.connect(self._on_setting_changed)
        layout.addWidget(self._approval_checkbox)

        # Streaming toggle
        self._streaming_checkbox = QCheckBox("Enable Streaming")
        self._streaming_checkbox.setChecked(True)
        self._streaming_checkbox.setStyleSheet(checkbox_style)
        self._streaming_checkbox.stateChanged.connect(self._on_setting_changed)
        layout.addWidget(self._streaming_checkbox)

        return group

    def _create_processing_group(self) -> QGroupBox:
        """Create processing settings group."""
        group = QGroupBox("Processing")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: 600;
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
        """)

        layout = QVBoxLayout(group)
        layout.setSpacing(12)

        # Max agents
        agents_row = QHBoxLayout()
        agents_label = QLabel("Max Agents:")
        agents_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        agents_label.setFixedWidth(80)
        agents_row.addWidget(agents_label)

        self._agents_spin = QSpinBox()
        self._agents_spin.setRange(1, 10)
        self._agents_spin.setValue(3)
        self._agents_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        self._agents_spin.valueChanged.connect(self._on_setting_changed)
        agents_row.addWidget(self._agents_spin)
        agents_row.addStretch()
        layout.addLayout(agents_row)

        # Timeout
        timeout_row = QHBoxLayout()
        timeout_label = QLabel("Timeout (s):")
        timeout_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY};")
        timeout_label.setFixedWidth(80)
        timeout_row.addWidget(timeout_label)

        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(30, 600)
        self._timeout_spin.setValue(120)
        self._timeout_spin.setSingleStep(30)
        self._timeout_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        self._timeout_spin.valueChanged.connect(self._on_setting_changed)
        timeout_row.addWidget(self._timeout_spin)
        timeout_row.addStretch()
        layout.addLayout(timeout_row)

        return group

    def set_felix_system(self, felix_system):
        """Set Felix system reference."""
        self._felix_system = felix_system
        self._load_current_settings()

    def _load_current_settings(self):
        """Load current settings from Felix system."""
        if not self._felix_system:
            return

        try:
            # Get current config
            status = self._felix_system.get_system_status()

            # Update provider
            provider = status.get('llm_provider', '')
            provider_map = {
                'lm_studio': 'LM Studio',
                'anthropic': 'Anthropic',
                'openai': 'OpenAI',
                'gemini': 'Google (Gemini)',
                'multi_provider_router': 'Router (Multi)'
            }
            display_provider = provider_map.get(provider, provider)
            index = self._provider_combo.findText(display_provider)
            if index >= 0:
                self._provider_combo.setCurrentIndex(index)

            # Update status
            if status.get('running'):
                self._llm_status.setText("Connected")
                self._llm_status.setStyleSheet(f"color: {Colors.STATUS_RUNNING};")
            else:
                self._llm_status.setText("Not connected")
                self._llm_status.setStyleSheet(f"color: {Colors.TEXT_MUTED};")

            self._status_label.setText("Settings loaded")

        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self._status_label.setText(f"Error: {e}")

    def _on_provider_changed(self, provider: str):
        """Handle provider change."""
        # Update model placeholder based on provider
        placeholders = {
            'LM Studio': 'e.g., llama-3.1-8b',
            'Anthropic': 'e.g., claude-3-sonnet-20240229',
            'OpenAI': 'e.g., gpt-4-turbo',
            'Google (Gemini)': 'e.g., gemini-pro',
            'Router (Multi)': 'Auto-selected'
        }
        self._model_input.setPlaceholderText(placeholders.get(provider, ''))
        self._on_setting_changed()

    def _on_setting_changed(self):
        """Handle any setting change."""
        settings = self._get_current_settings()
        self.settings_changed.emit(settings)
        self._status_label.setText("Settings changed (restart required)")

    def _get_current_settings(self) -> Dict[str, Any]:
        """Get current settings as dict."""
        provider_map = {
            'LM Studio': 'lm_studio',
            'Anthropic': 'anthropic',
            'OpenAI': 'openai',
            'Google (Gemini)': 'gemini',
            'Router (Multi)': 'multi_provider_router'
        }

        return {
            'llm_provider': provider_map.get(
                self._provider_combo.currentText(),
                'lm_studio'
            ),
            'model': self._model_input.text(),
            'knowledge_enabled': self._kb_checkbox.isChecked(),
            'web_search_enabled': self._web_checkbox.isChecked(),
            'command_approval_required': self._approval_checkbox.isChecked(),
            'streaming_enabled': self._streaming_checkbox.isChecked(),
            'max_agents': self._agents_spin.value(),
            'timeout': self._timeout_spin.value(),
        }

    def _test_llm_connection(self):
        """Test LLM connection."""
        if not self._felix_system:
            QMessageBox.warning(self, "Error", "Felix system not running")
            return

        try:
            # Quick test - just check if we can get status
            status = self._felix_system.get_system_status()
            if status.get('running'):
                self._llm_status.setText("Connected")
                self._llm_status.setStyleSheet(f"color: {Colors.STATUS_RUNNING};")
                QMessageBox.information(self, "Success", "LLM connection OK")
            else:
                self._llm_status.setText("Not running")
                self._llm_status.setStyleSheet(f"color: {Colors.STATUS_STOPPED};")
                QMessageBox.warning(self, "Warning", "Felix system not running")

        except Exception as e:
            self._llm_status.setText("Error")
            self._llm_status.setStyleSheet(f"color: {Colors.ERROR};")
            QMessageBox.critical(self, "Error", f"Connection test failed: {e}")

    def cleanup(self):
        """Clean up resources."""
        self._felix_system = None
