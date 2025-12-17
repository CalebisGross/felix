"""Learning systems view for Felix's self-improvement architecture."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTextEdit, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QCheckBox, QSlider, QSizePolicy, QScrollArea
)

from ..core.theme import Colors

logger = logging.getLogger(__name__)

# Import Felix learning modules (with fallback)
try:
    from src.learning import RecommendationEngine
    from src.workflows.concept_registry import ConceptRegistry
    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Learning modules not available: {e}")
    LEARNING_AVAILABLE = False


class StatusCard(QFrame):
    """Small status display card."""

    def __init__(self, title: str, value: str = "--", subtitle: str = "", parent=None):
        super().__init__(parent)
        self._setup_ui(title, value, subtitle)

    def _setup_ui(self, title: str, value: str, subtitle: str):
        self.setFixedWidth(140)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 6px;
                padding: 4px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(self._title_label)

        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: bold;")
        layout.addWidget(self._value_label)

        self._subtitle_label = QLabel(subtitle)
        self._subtitle_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 9px;")
        layout.addWidget(self._subtitle_label)

    def set_value(self, value: str):
        self._value_label.setText(value)

    def set_subtitle(self, subtitle: str):
        self._subtitle_label.setText(subtitle)

    def set_status_color(self, color: str):
        self._value_label.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")


class LearningView(QWidget):
    """Learning systems view with 8 sub-tabs.

    Features:
    - Overview: System status and unified statistics
    - Feedback: Synthesis feedback integration toggle
    - Agent Performance: Synthesis integration rates
    - Patterns: Historical workflow recommendations
    - Calibration: Agent confidence calibration stats
    - Thresholds: Learned optimal threshold values
    - Concepts: Workflow-scoped concept registry
    - Meta-Learning: Meta-learning boost configuration
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._felix_system = None
        self._auto_refresh_timer = None
        self._auto_refresh_enabled = False
        self._feedback_enabled = True
        self._meta_learning_enabled = True
        self._min_samples = 3
        self._setup_ui()

    def _setup_ui(self):
        """Set up learning view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with controls
        header = self._create_header()
        layout.addWidget(header)

        # Status cards row
        status_row = self._create_status_cards()
        layout.addWidget(status_row)

        # Tab widget for sub-sections
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
            QTabBar::tab {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_SECONDARY};
                padding: 6px 10px;
                border: none;
                border-bottom: 2px solid transparent;
                font-size: 11px;
            }}
            QTabBar::tab:selected {{
                color: {Colors.TEXT_PRIMARY};
                border-bottom: 2px solid {Colors.ACCENT};
            }}
            QTabBar::tab:hover {{
                background-color: {Colors.BACKGROUND_LIGHT};
            }}
        """)

        # Create all 8 sub-tabs
        self._tabs.addTab(self._create_overview_tab(), "Overview")
        self._tabs.addTab(self._create_feedback_tab(), "Feedback")
        self._tabs.addTab(self._create_agent_performance_tab(), "Agents")
        self._tabs.addTab(self._create_patterns_tab(), "Patterns")
        self._tabs.addTab(self._create_calibration_tab(), "Calibration")
        self._tabs.addTab(self._create_thresholds_tab(), "Thresholds")
        self._tabs.addTab(self._create_concepts_tab(), "Concepts")
        self._tabs.addTab(self._create_meta_learning_tab(), "Meta")

        layout.addWidget(self._tabs, 1)

    def _create_header(self) -> QFrame:
        """Create header with title and controls."""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BACKGROUND};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Title
        title = QLabel("Learning Systems")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 13px; font-weight: 600;")
        layout.addWidget(title)

        layout.addStretch()

        # Refresh button
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setFixedWidth(60)
        self._refresh_btn.clicked.connect(self._refresh_statistics)
        self._refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BACKGROUND_LIGHT};
            }}
        """)
        layout.addWidget(self._refresh_btn)

        # Auto-refresh checkbox
        self._auto_refresh_check = QCheckBox("Auto")
        self._auto_refresh_check.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        self._auto_refresh_check.stateChanged.connect(self._toggle_auto_refresh)
        layout.addWidget(self._auto_refresh_check)

        return header

    def _create_status_cards(self) -> QFrame:
        """Create status cards row."""
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {Colors.BACKGROUND};")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Status card
        self._status_card = StatusCard("Status", "Inactive", "Not started")
        layout.addWidget(self._status_card)

        # Systems card
        self._systems_card = StatusCard("Systems", "0/4", "Active modules")
        layout.addWidget(self._systems_card)

        # Data points card
        self._data_card = StatusCard("Data", "0", "Samples")
        layout.addWidget(self._data_card)

        # Success rate card
        self._success_card = StatusCard("Success", "--", "Pattern rate")
        layout.addWidget(self._success_card)

        # Calibration card
        self._calibration_card = StatusCard("Cal Error", "--", "Confidence")
        layout.addWidget(self._calibration_card)

        layout.addStretch()

        return frame

    def _create_overview_tab(self) -> QWidget:
        """Create overview statistics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        self._overview_text = QTextEdit()
        self._overview_text.setReadOnly(True)
        self._overview_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        self._overview_text.setPlainText("Click Refresh to load learning statistics.")
        layout.addWidget(self._overview_text)

        return widget

    def _create_feedback_tab(self) -> QWidget:
        """Create feedback integration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Config section
        config_frame = QFrame()
        config_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(10, 10, 10, 10)

        self._feedback_check = QCheckBox("Enable Feedback Integration Protocol")
        self._feedback_check.setChecked(True)
        self._feedback_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        self._feedback_check.stateChanged.connect(self._toggle_feedback)
        config_layout.addWidget(self._feedback_check)

        info = QLabel("Broadcasts synthesis results to agents for continuous improvement.")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        info.setWordWrap(True)
        config_layout.addWidget(info)

        layout.addWidget(config_frame)

        # Statistics text
        self._feedback_text = QTextEdit()
        self._feedback_text.setReadOnly(True)
        self._feedback_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        self._feedback_text.setPlainText(self._get_feedback_description())
        layout.addWidget(self._feedback_text, 1)

        return widget

    def _create_agent_performance_tab(self) -> QWidget:
        """Create agent performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel("Tracks agent synthesis integration rate and contribution quality")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(info)

        self._agent_tree = QTreeWidget()
        self._agent_tree.setHeaderLabels(["Agent ID", "Integration", "Confidence", "Samples"])
        self._agent_tree.setRootIsDecorated(False)
        self._agent_tree.setStyleSheet(self._get_tree_style())
        self._agent_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._agent_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._agent_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._agent_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._agent_tree.setColumnWidth(1, 80)
        self._agent_tree.setColumnWidth(2, 80)
        self._agent_tree.setColumnWidth(3, 60)
        layout.addWidget(self._agent_tree, 1)

        return widget

    def _create_patterns_tab(self) -> QWidget:
        """Create pattern learning tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel("Historical workflow patterns and recommendations (requires ≥10 samples)")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(info)

        self._patterns_text = QTextEdit()
        self._patterns_text.setReadOnly(True)
        self._patterns_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        layout.addWidget(self._patterns_text, 1)

        return widget

    def _create_calibration_tab(self) -> QWidget:
        """Create confidence calibration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel("Agent confidence calibration: predicted vs actual (requires ≥10 workflows)")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(info)

        self._calibration_tree = QTreeWidget()
        self._calibration_tree.setHeaderLabels(["Agent", "Complexity", "Factor", "Error", "Samples"])
        self._calibration_tree.setRootIsDecorated(False)
        self._calibration_tree.setStyleSheet(self._get_tree_style())
        self._calibration_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._calibration_tree.setColumnWidth(1, 70)
        self._calibration_tree.setColumnWidth(2, 60)
        self._calibration_tree.setColumnWidth(3, 50)
        self._calibration_tree.setColumnWidth(4, 50)
        layout.addWidget(self._calibration_tree, 1)

        return widget

    def _create_thresholds_tab(self) -> QWidget:
        """Create threshold learning tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        info = QLabel("Optimal threshold values learned per task type (requires ≥20 workflows)")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        layout.addWidget(info)

        self._thresholds_tree = QTreeWidget()
        self._thresholds_tree.setHeaderLabels(["Task Type", "Threshold", "Value", "Success", "N"])
        self._thresholds_tree.setRootIsDecorated(False)
        self._thresholds_tree.setStyleSheet(self._get_tree_style())
        self._thresholds_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._thresholds_tree.setColumnWidth(1, 100)
        self._thresholds_tree.setColumnWidth(2, 60)
        self._thresholds_tree.setColumnWidth(3, 60)
        self._thresholds_tree.setColumnWidth(4, 40)
        layout.addWidget(self._thresholds_tree, 1)

        return widget

    def _create_concepts_tab(self) -> QWidget:
        """Create concept registry tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Info row
        info_row = QHBoxLayout()
        info = QLabel("Workflow-scoped concept tracking for terminology consistency")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        info_row.addWidget(info)

        note = QLabel("(from last workflow)")
        note.setStyleSheet(f"color: {Colors.WARNING}; font-size: 9px;")
        info_row.addWidget(note)
        info_row.addStretch()

        layout.addLayout(info_row)

        self._concepts_tree = QTreeWidget()
        self._concepts_tree.setHeaderLabels(["Concept", "Definition", "Source", "Conf", "Uses"])
        self._concepts_tree.setRootIsDecorated(False)
        self._concepts_tree.setStyleSheet(self._get_tree_style())
        self._concepts_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._concepts_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._concepts_tree.setColumnWidth(0, 100)
        self._concepts_tree.setColumnWidth(2, 80)
        self._concepts_tree.setColumnWidth(3, 40)
        self._concepts_tree.setColumnWidth(4, 40)
        layout.addWidget(self._concepts_tree, 1)

        return widget

    def _create_meta_learning_tab(self) -> QWidget:
        """Create meta-learning configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Config section
        config_frame = QFrame()
        config_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
        """)
        config_layout = QVBoxLayout(config_frame)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(8)

        title = QLabel("Meta-Learning Boost Configuration")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 12px; font-weight: bold;")
        config_layout.addWidget(title)

        info = QLabel("Tracks which knowledge entries help specific task types and boosts retrieval relevance.")
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        info.setWordWrap(True)
        config_layout.addWidget(info)

        self._meta_learning_check = QCheckBox("Enable Meta-Learning Boost")
        self._meta_learning_check.setChecked(True)
        self._meta_learning_check.setStyleSheet(f"color: {Colors.TEXT_PRIMARY};")
        self._meta_learning_check.stateChanged.connect(self._toggle_meta_learning)
        config_layout.addWidget(self._meta_learning_check)

        # Min samples slider
        slider_row = QHBoxLayout()
        slider_label = QLabel("Min samples for boost:")
        slider_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px;")
        slider_row.addWidget(slider_label)

        self._min_samples_slider = QSlider(Qt.Orientation.Horizontal)
        self._min_samples_slider.setRange(1, 10)
        self._min_samples_slider.setValue(3)
        self._min_samples_slider.setFixedWidth(100)
        self._min_samples_slider.valueChanged.connect(self._update_min_samples)
        slider_row.addWidget(self._min_samples_slider)

        self._min_samples_label = QLabel("3")
        self._min_samples_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        self._min_samples_label.setFixedWidth(20)
        slider_row.addWidget(self._min_samples_label)

        slider_row.addStretch()
        config_layout.addLayout(slider_row)

        layout.addWidget(config_frame)

        # Statistics text
        self._meta_learning_text = QTextEdit()
        self._meta_learning_text.setReadOnly(True)
        self._meta_learning_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        self._meta_learning_text.setPlainText(self._get_meta_learning_description())
        layout.addWidget(self._meta_learning_text, 1)

        return widget

    def _get_tree_style(self) -> str:
        """Get common tree widget style."""
        return f"""
            QTreeWidget {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QTreeWidget::item {{
                padding: 2px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Colors.ACCENT};
            }}
            QHeaderView::section {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                padding: 4px;
                font-size: 10px;
            }}
        """

    def _get_feedback_description(self) -> str:
        """Get feedback integration description."""
        return """FEEDBACK INTEGRATION PROTOCOL
================================================================================

Status: Enabled

DESCRIPTION
--------------------------------------------------------------------------------
After each synthesis, CentralPost broadcasts feedback to all agents via:
- SYNTHESIS_FEEDBACK messages (overall synthesis quality)
- CONTRIBUTION_EVALUATION messages (individual agent usefulness)

Agents track their 'synthesis integration rate' - how often their
contributions are used in final synthesis. This enables:
- Confidence calibration (predicted vs actual usefulness)
- Adaptive behavior (agents adjust when usefulness < 0.3)
- Self-improvement through explicit feedback loops

BENEFITS
--------------------------------------------------------------------------------
✓ Agents learn which contributions are valuable
✓ Confidence scores become more accurate over time
✓ Low-performing agents can adapt their approach
✓ High-performing agents gain confidence boost
✓ Systemic learning across the agent team
"""

    def _get_meta_learning_description(self) -> str:
        """Get meta-learning description."""
        return f"""META-LEARNING BOOST SYSTEM
================================================================================

Status: {'Enabled' if self._meta_learning_enabled else 'Disabled'}
Minimum samples for boost: {self._min_samples}

HOW IT WORKS
--------------------------------------------------------------------------------
1. Workflow uses knowledge entries from the knowledge store
2. User marks knowledge as 'helpful' or 'unhelpful'
3. Usage patterns stored in knowledge_usage table
4. When ≥{self._min_samples} samples exist, boost factor applied (0.5-1.0x)
5. Future similar tasks retrieve boosted knowledge first

BENEFITS
--------------------------------------------------------------------------------
✓ Knowledge that helped before ranks higher
✓ Task-type specific (research vs analysis vs coding)
✓ Continuous improvement through usage tracking
✓ Learns from user feedback on helpfulness

DATABASE
--------------------------------------------------------------------------------
Location: felix_knowledge.db
Table: knowledge_usage
Columns: entry_id, workflow_id, task_type, usefulness_score, timestamp
"""

    def set_felix_system(self, felix_system):
        """Set Felix system reference."""
        self._felix_system = felix_system
        if felix_system:
            self._refresh_statistics()

    @Slot()
    def _refresh_statistics(self):
        """Refresh all learning statistics."""
        if not LEARNING_AVAILABLE:
            self._overview_text.setPlainText(
                "Learning modules not available.\nCheck that all dependencies are installed."
            )
            self._status_card.set_value("Unavailable")
            self._status_card.set_status_color(Colors.ERROR)
            return

        if not self._felix_system:
            self._overview_text.setPlainText(
                "Felix system not started.\nStart the system to view learning statistics."
            )
            self._status_card.set_value("Inactive")
            self._status_card.set_status_color(Colors.ERROR)
            return

        try:
            # Check if learning is enabled
            config = getattr(self._felix_system, 'config', None)
            if config and hasattr(config, 'enable_learning') and not config.enable_learning:
                self._overview_text.setPlainText(
                    "Learning systems disabled in configuration.\nEnable in Settings."
                )
                self._status_card.set_value("Disabled")
                self._status_card.set_status_color(Colors.WARNING)
                return

            # Get task memory
            task_memory = getattr(self._felix_system, 'task_memory', None)
            if not task_memory:
                self._overview_text.setPlainText("Task memory not available.")
                self._status_card.set_value("Error")
                self._status_card.set_status_color(Colors.ERROR)
                return

            # Create recommendation engine and get statistics
            rec_engine = RecommendationEngine(
                task_memory=task_memory,
                enable_auto_apply=False
            )
            stats = rec_engine.get_unified_statistics(days=30)

            # Update all tabs
            self._update_overview(stats)
            self._update_agent_performance()
            self._update_patterns(rec_engine)
            self._update_calibration(rec_engine)
            self._update_thresholds(rec_engine)
            self._update_concepts()
            self._update_status_cards(stats)

        except Exception as e:
            logger.error(f"Failed to refresh learning statistics: {e}")
            self._overview_text.setPlainText(f"Error loading statistics:\n{str(e)}")
            self._status_card.set_value("Error")
            self._status_card.set_status_color(Colors.ERROR)

    def _update_overview(self, stats: Dict[str, Any]):
        """Update overview tab."""
        text = "FELIX LEARNING SYSTEMS - OVERVIEW\n"
        text += "=" * 60 + "\n\n"

        overall = stats.get('overall', {})
        text += f"Status: {'Active' if overall.get('learning_active') else 'Inactive'}\n"
        text += f"Systems with data: {overall.get('systems_with_data', 0)}/4\n"
        text += f"Total data points: {overall.get('total_data_points', 0):,}\n"
        text += f"Reporting period: Last {stats.get('days', 30)} days\n\n"

        # Pattern learner
        pattern_stats = stats.get('pattern_learner', {})
        text += "1. PATTERN LEARNER\n"
        text += "-" * 60 + "\n"
        total_recs = pattern_stats.get('total_recommendations', 0)
        text += f"   Recommendations: {total_recs}\n"
        text += f"   Applied: {pattern_stats.get('applied_count', 0) or 0}\n"
        text += f"   Successful: {pattern_stats.get('success_count', 0) or 0}\n"
        if pattern_stats.get('success_rate') is not None and total_recs > 0:
            text += f"   Success rate: {pattern_stats['success_rate']:.1%}\n"
        elif total_recs == 0:
            text += "   Status: No data (need ≥10 patterns)\n"
        text += "\n"

        # Calibration
        cal_stats = stats.get('confidence_calibrator', {})
        text += "2. CONFIDENCE CALIBRATOR\n"
        text += "-" * 60 + "\n"
        text += f"   Records: {cal_stats.get('total_records', 0)}\n"
        text += f"   Samples: {cal_stats.get('total_samples', 0)}\n"
        if cal_stats.get('avg_calibration_error') is not None:
            text += f"   Avg error: {cal_stats['avg_calibration_error']:.3f}\n"
        if cal_stats.get('most_overconfident'):
            oc = cal_stats['most_overconfident']
            text += f"   Overconfident: {oc['agent_type']}/{oc['task_complexity']}\n"
        if cal_stats.get('total_records', 0) == 0:
            text += "   Status: No data (need ≥10 workflows)\n"
        text += "\n"

        # Thresholds
        threshold_stats = stats.get('threshold_learner', {})
        text += "3. THRESHOLD LEARNER\n"
        text += "-" * 60 + "\n"
        text += f"   Records: {threshold_stats.get('total_records', 0)}\n"
        text += f"   Samples: {threshold_stats.get('total_samples', 0)}\n"
        if threshold_stats.get('avg_success_rate') is not None:
            text += f"   Avg success: {threshold_stats['avg_success_rate']:.1%}\n"
        if threshold_stats.get('total_records', 0) == 0:
            text += "   Status: No data (need ≥20 workflows)\n"
        text += "\n"

        # Feedback
        feedback_stats = stats.get('feedback_integrator', {})
        text += "4. FEEDBACK INTEGRATION\n"
        text += "-" * 60 + "\n"
        text += f"   Status: {'Enabled' if self._feedback_enabled else 'Disabled'}\n"
        total_ratings = feedback_stats.get('total_ratings', 0)
        if total_ratings > 0:
            text += f"   Total ratings: {total_ratings}\n"
            positive = feedback_stats.get('positive_ratings', 0)
            if total_ratings > 0:
                text += f"   Positive: {positive} ({positive/total_ratings:.0%})\n"
            if feedback_stats.get('avg_accuracy') is not None:
                text += f"   Avg accuracy: {feedback_stats['avg_accuracy']:.2f}\n"
        else:
            text += "   Status: No feedback data yet\n"
        text += "\n"

        text += "=" * 60 + "\n"
        text += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        self._overview_text.setPlainText(text)

    def _update_agent_performance(self):
        """Update agent performance tab."""
        self._agent_tree.clear()

        # Get agent performance from tracker if available
        tracker = getattr(self._felix_system, 'agent_performance_tracker', None)
        if tracker:
            try:
                metrics = tracker.get_all_metrics() if hasattr(tracker, 'get_all_metrics') else {}
                for agent_id, data in metrics.items():
                    item = QTreeWidgetItem([
                        str(agent_id),
                        f"{data.get('integration_rate', 0):.1%}",
                        f"{data.get('avg_confidence', 0):.2f}",
                        str(data.get('samples', 0))
                    ])
                    self._agent_tree.addTopLevelItem(item)
            except Exception as e:
                logger.warning(f"Error getting agent performance: {e}")

        if self._agent_tree.topLevelItemCount() == 0:
            item = QTreeWidgetItem(["No data yet", "--", "--", "--"])
            self._agent_tree.addTopLevelItem(item)

    def _update_patterns(self, rec_engine):
        """Update patterns tab."""
        text = "PATTERN RECOMMENDATIONS\n"
        text += "=" * 60 + "\n\n"

        try:
            stats = rec_engine.pattern_learner.get_recommendation_statistics(days=30)

            total_recs = stats.get('total_recommendations', 0)
            applied = stats.get('applied_count', 0) or 0
            success = stats.get('success_count', 0) or 0

            text += f"Total recommendations: {total_recs}\n"
            text += f"Applied recommendations: {applied}\n"
            text += f"Successful workflows: {success}\n"

            if stats.get('success_rate') is not None and total_recs > 0:
                text += f"\nSuccess rate: {stats['success_rate']:.1%}\n"
            elif total_recs == 0:
                text += "\nNo recommendations generated yet\n"
                text += "(Requires ≥10 historical patterns)\n"

            text += "\n" + "-" * 60 + "\n"
            text += "THRESHOLDS\n"
            text += "  Auto-apply: ≥95% success + ≥20 samples\n"
            text += "  Recommend: ≥80% success + ≥10 samples\n"

        except Exception as e:
            text += f"\nError: {e}\n"

        self._patterns_text.setPlainText(text)

    def _update_calibration(self, rec_engine):
        """Update calibration tab."""
        self._calibration_tree.clear()

        try:
            records = rec_engine.confidence_calibrator.get_all_calibration_records()

            if records:
                for rec in records:
                    if rec.calibration_factor > 1.1:
                        interp = "Under"
                    elif rec.calibration_factor < 0.9:
                        interp = "Over"
                    else:
                        interp = "OK"

                    item = QTreeWidgetItem([
                        f"{rec.agent_type} ({interp})",
                        rec.task_complexity,
                        f"{rec.calibration_factor:.2f}",
                        f"{rec.calibration_error:.2f}",
                        str(rec.sample_size)
                    ])
                    self._calibration_tree.addTopLevelItem(item)
        except Exception as e:
            logger.warning(f"Error loading calibration: {e}")

        if self._calibration_tree.topLevelItemCount() == 0:
            item = QTreeWidgetItem(["No data yet", "--", "--", "--", "--"])
            self._calibration_tree.addTopLevelItem(item)

    def _update_thresholds(self, rec_engine):
        """Update thresholds tab."""
        self._thresholds_tree.clear()

        try:
            records = rec_engine.threshold_learner.get_all_threshold_records()

            if records:
                for rec in records:
                    item = QTreeWidgetItem([
                        rec.task_type,
                        rec.threshold_name,
                        f"{rec.learned_value:.2f}",
                        f"{rec.success_rate:.0%}",
                        str(rec.sample_size)
                    ])
                    self._thresholds_tree.addTopLevelItem(item)
        except Exception as e:
            logger.warning(f"Error loading thresholds: {e}")

        if self._thresholds_tree.topLevelItemCount() == 0:
            item = QTreeWidgetItem(["No data yet", "--", "--", "--", "--"])
            self._thresholds_tree.addTopLevelItem(item)

    def _update_concepts(self):
        """Update concepts tab."""
        self._concepts_tree.clear()

        # Get concept registry from last workflow if available
        workflow = getattr(self._felix_system, 'current_workflow', None)
        if workflow:
            registry = getattr(workflow, 'concept_registry', None)
            if registry:
                try:
                    concepts = registry.get_all_concepts() if hasattr(registry, 'get_all_concepts') else {}
                    for name, data in concepts.items():
                        item = QTreeWidgetItem([
                            name[:20],
                            data.get('definition', '')[:50],
                            data.get('source', '--'),
                            f"{data.get('confidence', 0):.1f}",
                            str(data.get('usage_count', 0))
                        ])
                        self._concepts_tree.addTopLevelItem(item)
                except Exception as e:
                    logger.warning(f"Error loading concepts: {e}")

        if self._concepts_tree.topLevelItemCount() == 0:
            item = QTreeWidgetItem(["No recent workflow", "Concept registry is workflow-scoped", "--", "--", "--"])
            self._concepts_tree.addTopLevelItem(item)

    def _update_status_cards(self, stats: Dict[str, Any]):
        """Update status cards."""
        overall = stats.get('overall', {})

        # Systems active
        systems_active = overall.get('systems_with_data', 0)
        self._systems_card.set_value(f"{systems_active}/4")

        # Total data
        total_data = overall.get('total_data_points', 0)
        self._data_card.set_value(f"{total_data:,}")

        # Success rate
        pattern_stats = stats.get('pattern_learner', {})
        if pattern_stats.get('success_rate') is not None:
            self._success_card.set_value(f"{pattern_stats['success_rate']:.0%}")
        else:
            self._success_card.set_value("--")

        # Calibration error
        cal_stats = stats.get('confidence_calibrator', {})
        if cal_stats.get('avg_calibration_error') is not None:
            self._calibration_card.set_value(f"{cal_stats['avg_calibration_error']:.2f}")
        else:
            self._calibration_card.set_value("--")

        # Status
        if systems_active >= 3:
            self._status_card.set_value("Active")
            self._status_card.set_status_color(Colors.SUCCESS)
        elif systems_active > 0:
            self._status_card.set_value(f"Partial")
            self._status_card.set_status_color(Colors.WARNING)
        else:
            self._status_card.set_value("No Data")
            self._status_card.set_status_color(Colors.TEXT_MUTED)

    @Slot(int)
    def _toggle_auto_refresh(self, state: int):
        """Toggle auto-refresh."""
        self._auto_refresh_enabled = state == Qt.CheckState.Checked.value

        if self._auto_refresh_enabled:
            if not self._auto_refresh_timer:
                self._auto_refresh_timer = QTimer(self)
                self._auto_refresh_timer.timeout.connect(self._refresh_statistics)
            self._auto_refresh_timer.start(30000)  # 30 seconds
            logger.debug("Learning auto-refresh enabled")
        else:
            if self._auto_refresh_timer:
                self._auto_refresh_timer.stop()
            logger.debug("Learning auto-refresh disabled")

    @Slot(int)
    def _toggle_feedback(self, state: int):
        """Toggle feedback integration."""
        self._feedback_enabled = state == Qt.CheckState.Checked.value
        logger.info(f"Feedback integration {'enabled' if self._feedback_enabled else 'disabled'}")

        if self._felix_system:
            config = getattr(self._felix_system, 'config', None)
            if config and hasattr(config, 'enable_feedback'):
                config.enable_feedback = self._feedback_enabled

    @Slot(int)
    def _toggle_meta_learning(self, state: int):
        """Toggle meta-learning boost."""
        self._meta_learning_enabled = state == Qt.CheckState.Checked.value
        logger.info(f"Meta-learning boost {'enabled' if self._meta_learning_enabled else 'disabled'}")
        self._meta_learning_text.setPlainText(self._get_meta_learning_description())

    @Slot(int)
    def _update_min_samples(self, value: int):
        """Update minimum samples slider."""
        self._min_samples = value
        self._min_samples_label.setText(str(value))
        self._meta_learning_text.setPlainText(self._get_meta_learning_description())

    def cleanup(self):
        """Clean up resources."""
        if self._auto_refresh_timer:
            self._auto_refresh_timer.stop()
            self._auto_refresh_timer = None
        self._felix_system = None
