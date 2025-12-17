"""Full settings dialog for Felix configuration."""

import logging
from typing import Optional, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QWidget, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QGroupBox, QScrollArea, QFrame,
    QMessageBox
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Full settings dialog with all configuration options.

    Categories:
    - LLM: Provider, model, API settings
    - Processing: Agents, timeouts, modes
    - Knowledge: KB settings, embeddings
    - Features: Toggles for various features
    - Advanced: Helix geometry, trust rules
    """

    settings_saved = Signal(dict)

    def __init__(self, felix_system=None, parent=None):
        super().__init__(parent)
        self._felix_system = felix_system
        self._settings: Dict[str, Any] = {}

        self.setWindowTitle("Felix Settings")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        self.setModal(True)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Set up the settings dialog UI."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BACKGROUND};
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER};
                background-color: {Colors.BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_SECONDARY};
                padding: 8px 16px;
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY};
                font-weight: 600;
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }}
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}
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
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Tab widget
        tabs = QTabWidget()

        # LLM tab
        tabs.addTab(self._create_llm_tab(), "LLM")

        # Processing tab
        tabs.addTab(self._create_processing_tab(), "Processing")

        # Knowledge tab
        tabs.addTab(self._create_knowledge_tab(), "Knowledge")

        # Features tab
        tabs.addTab(self._create_features_tab(), "Features")

        # Advanced tab
        tabs.addTab(self._create_advanced_tab(), "Advanced")

        layout.addWidget(tabs)

        # Buttons
        button_row = QHBoxLayout()
        button_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_settings)
        button_row.addWidget(apply_btn)

        save_btn = QPushButton("Save")
        save_btn.setProperty("primary", True)
        save_btn.clicked.connect(self._save_and_close)
        button_row.addWidget(save_btn)

        layout.addLayout(button_row)

    def _create_llm_tab(self) -> QWidget:
        """Create LLM settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Provider group
        provider_group = QGroupBox("LLM Provider")
        provider_layout = QFormLayout(provider_group)

        self._provider_combo = QComboBox()
        self._provider_combo.addItems([
            "LM Studio (Local)",
            "Anthropic (Claude)",
            "OpenAI (GPT)",
            "Google (Gemini)",
            "Multi-Provider Router"
        ])
        provider_layout.addRow("Provider:", self._provider_combo)

        self._model_input = QLineEdit()
        self._model_input.setPlaceholderText("e.g., llama-3.1-8b")
        provider_layout.addRow("Model:", self._model_input)

        self._api_key_input = QLineEdit()
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText("API key (if required)")
        provider_layout.addRow("API Key:", self._api_key_input)

        self._base_url_input = QLineEdit()
        self._base_url_input.setPlaceholderText("http://localhost:1234/v1")
        provider_layout.addRow("Base URL:", self._base_url_input)

        layout.addWidget(provider_group)

        # Generation group
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QFormLayout(gen_group)

        self._temperature_spin = QDoubleSpinBox()
        self._temperature_spin.setRange(0.0, 2.0)
        self._temperature_spin.setSingleStep(0.1)
        self._temperature_spin.setValue(0.7)
        gen_layout.addRow("Temperature:", self._temperature_spin)

        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(100, 32000)
        self._max_tokens_spin.setSingleStep(100)
        self._max_tokens_spin.setValue(4096)
        gen_layout.addRow("Max Tokens:", self._max_tokens_spin)

        self._streaming_check = QCheckBox("Enable streaming")
        self._streaming_check.setChecked(True)
        gen_layout.addRow("", self._streaming_check)

        layout.addWidget(gen_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_processing_tab(self) -> QWidget:
        """Create processing settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Agents group
        agents_group = QGroupBox("Agent Settings")
        agents_layout = QFormLayout(agents_group)

        self._max_agents_spin = QSpinBox()
        self._max_agents_spin.setRange(1, 10)
        self._max_agents_spin.setValue(3)
        agents_layout.addRow("Max Concurrent Agents:", self._max_agents_spin)

        self._agent_timeout_spin = QSpinBox()
        self._agent_timeout_spin.setRange(30, 600)
        self._agent_timeout_spin.setSingleStep(30)
        self._agent_timeout_spin.setValue(120)
        self._agent_timeout_spin.setSuffix(" seconds")
        agents_layout.addRow("Agent Timeout:", self._agent_timeout_spin)

        layout.addWidget(agents_group)

        # Mode group
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QFormLayout(mode_group)

        self._default_mode_combo = QComboBox()
        self._default_mode_combo.addItems(["Auto", "Simple (Direct)", "Workflow (Full)"])
        mode_layout.addRow("Default Mode:", self._default_mode_combo)

        self._auto_threshold_spin = QDoubleSpinBox()
        self._auto_threshold_spin.setRange(0.0, 1.0)
        self._auto_threshold_spin.setSingleStep(0.1)
        self._auto_threshold_spin.setValue(0.6)
        mode_layout.addRow("Auto Mode Threshold:", self._auto_threshold_spin)

        layout.addWidget(mode_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_knowledge_tab(self) -> QWidget:
        """Create knowledge brain settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # KB group
        kb_group = QGroupBox("Knowledge Brain")
        kb_layout = QFormLayout(kb_group)

        self._kb_enabled_check = QCheckBox("Enable Knowledge Brain")
        self._kb_enabled_check.setChecked(True)
        kb_layout.addRow("", self._kb_enabled_check)

        self._kb_auto_ingest_check = QCheckBox("Auto-ingest new documents")
        self._kb_auto_ingest_check.setChecked(True)
        kb_layout.addRow("", self._kb_auto_ingest_check)

        self._kb_watch_dir_input = QLineEdit()
        self._kb_watch_dir_input.setPlaceholderText("Directory to watch for documents")
        kb_layout.addRow("Watch Directory:", self._kb_watch_dir_input)

        layout.addWidget(kb_group)

        # Embeddings group
        embed_group = QGroupBox("Embeddings")
        embed_layout = QFormLayout(embed_group)

        self._embed_provider_combo = QComboBox()
        self._embed_provider_combo.addItems([
            "LM Studio (Local)",
            "TF-IDF Fallback",
            "SQLite FTS5 Only"
        ])
        embed_layout.addRow("Provider:", self._embed_provider_combo)

        self._embed_chunk_size_spin = QSpinBox()
        self._embed_chunk_size_spin.setRange(100, 2000)
        self._embed_chunk_size_spin.setSingleStep(100)
        self._embed_chunk_size_spin.setValue(500)
        embed_layout.addRow("Chunk Size:", self._embed_chunk_size_spin)

        self._embed_overlap_spin = QSpinBox()
        self._embed_overlap_spin.setRange(0, 200)
        self._embed_overlap_spin.setSingleStep(10)
        self._embed_overlap_spin.setValue(50)
        embed_layout.addRow("Chunk Overlap:", self._embed_overlap_spin)

        layout.addWidget(embed_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_features_tab(self) -> QWidget:
        """Create features toggle tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Features group
        features_group = QGroupBox("Feature Toggles")
        features_layout = QVBoxLayout(features_group)

        self._web_search_check = QCheckBox("Enable Web Search")
        self._web_search_check.setChecked(False)
        features_layout.addWidget(self._web_search_check)

        self._command_approval_check = QCheckBox("Require Command Approval")
        self._command_approval_check.setChecked(True)
        features_layout.addWidget(self._command_approval_check)

        self._learning_check = QCheckBox("Enable Learning from Tasks")
        self._learning_check.setChecked(True)
        features_layout.addWidget(self._learning_check)

        self._memory_check = QCheckBox("Enable Long-term Memory")
        self._memory_check.setChecked(True)
        features_layout.addWidget(self._memory_check)

        self._auto_save_check = QCheckBox("Auto-save Conversations")
        self._auto_save_check.setChecked(True)
        features_layout.addWidget(self._auto_save_check)

        layout.addWidget(features_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)

        # Helix group
        helix_group = QGroupBox("Helix Geometry")
        helix_layout = QFormLayout(helix_group)

        self._helix_layers_spin = QSpinBox()
        self._helix_layers_spin.setRange(3, 10)
        self._helix_layers_spin.setValue(5)
        helix_layout.addRow("Helix Layers:", self._helix_layers_spin)

        self._helix_pitch_spin = QDoubleSpinBox()
        self._helix_pitch_spin.setRange(0.1, 2.0)
        self._helix_pitch_spin.setSingleStep(0.1)
        self._helix_pitch_spin.setValue(0.5)
        helix_layout.addRow("Helix Pitch:", self._helix_pitch_spin)

        layout.addWidget(helix_group)

        # Trust group
        trust_group = QGroupBox("Trust Rules")
        trust_layout = QFormLayout(trust_group)

        self._trust_level_combo = QComboBox()
        self._trust_level_combo.addItems(["Paranoid", "Cautious", "Balanced", "Permissive"])
        self._trust_level_combo.setCurrentText("Cautious")
        trust_layout.addRow("Trust Level:", self._trust_level_combo)

        self._sandbox_check = QCheckBox("Enable Command Sandboxing")
        self._sandbox_check.setChecked(True)
        trust_layout.addRow("", self._sandbox_check)

        layout.addWidget(trust_group)

        # Debug group
        debug_group = QGroupBox("Debug")
        debug_layout = QFormLayout(debug_group)

        self._debug_logging_check = QCheckBox("Enable Debug Logging")
        self._debug_logging_check.setChecked(False)
        debug_layout.addRow("", self._debug_logging_check)

        self._show_thinking_check = QCheckBox("Show Agent Thinking")
        self._show_thinking_check.setChecked(False)
        debug_layout.addRow("", self._show_thinking_check)

        layout.addWidget(debug_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _load_settings(self):
        """Load current settings from Felix system."""
        if not self._felix_system:
            return

        try:
            status = self._felix_system.get_system_status()
            # Load values from status/config
            # This is a simplified version - real implementation would
            # read from config files
            logger.debug("Settings loaded from system")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")

    def _get_settings(self) -> Dict[str, Any]:
        """Get current settings from UI."""
        provider_map = {
            "LM Studio (Local)": "lm_studio",
            "Anthropic (Claude)": "anthropic",
            "OpenAI (GPT)": "openai",
            "Google (Gemini)": "gemini",
            "Multi-Provider Router": "multi_provider_router"
        }

        return {
            "llm": {
                "provider": provider_map.get(self._provider_combo.currentText(), "lm_studio"),
                "model": self._model_input.text(),
                "api_key": self._api_key_input.text(),
                "base_url": self._base_url_input.text(),
                "temperature": self._temperature_spin.value(),
                "max_tokens": self._max_tokens_spin.value(),
                "streaming": self._streaming_check.isChecked(),
            },
            "processing": {
                "max_agents": self._max_agents_spin.value(),
                "agent_timeout": self._agent_timeout_spin.value(),
                "default_mode": self._default_mode_combo.currentText().lower(),
                "auto_threshold": self._auto_threshold_spin.value(),
            },
            "knowledge": {
                "enabled": self._kb_enabled_check.isChecked(),
                "auto_ingest": self._kb_auto_ingest_check.isChecked(),
                "watch_directory": self._kb_watch_dir_input.text(),
                "embed_provider": self._embed_provider_combo.currentText(),
                "chunk_size": self._embed_chunk_size_spin.value(),
                "chunk_overlap": self._embed_overlap_spin.value(),
            },
            "features": {
                "web_search": self._web_search_check.isChecked(),
                "command_approval": self._command_approval_check.isChecked(),
                "learning": self._learning_check.isChecked(),
                "memory": self._memory_check.isChecked(),
                "auto_save": self._auto_save_check.isChecked(),
            },
            "advanced": {
                "helix_layers": self._helix_layers_spin.value(),
                "helix_pitch": self._helix_pitch_spin.value(),
                "trust_level": self._trust_level_combo.currentText().lower(),
                "sandbox": self._sandbox_check.isChecked(),
                "debug_logging": self._debug_logging_check.isChecked(),
                "show_thinking": self._show_thinking_check.isChecked(),
            }
        }

    def _apply_settings(self):
        """Apply settings without closing."""
        settings = self._get_settings()
        self.settings_saved.emit(settings)
        QMessageBox.information(
            self,
            "Settings Applied",
            "Settings have been applied.\n\nSome changes may require a restart."
        )

    def _save_and_close(self):
        """Save settings and close dialog."""
        settings = self._get_settings()
        self.settings_saved.emit(settings)
        self.accept()
