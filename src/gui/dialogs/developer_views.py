"""Developer mode views for agents, memory, and prompts."""

import logging
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QTextEdit, QSplitter, QFrame,
    QLineEdit, QComboBox, QTabWidget, QMessageBox
)

from ..core.theme import Colors

# Import WorkflowHistory for direct database access
try:
    from src.memory.workflow_history import WorkflowHistory
    WORKFLOW_HISTORY_AVAILABLE = True
except ImportError:
    WORKFLOW_HISTORY_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaseDevDialog(QDialog):
    """Base class for developer dialogs."""

    def __init__(self, felix_system, parent=None, title="Developer View"):
        super().__init__(parent)
        self._felix_system = felix_system

        self.setWindowTitle(title)
        self.setMinimumSize(700, 500)
        self.resize(800, 600)

        self._apply_style()

    def _apply_style(self):
        """Apply common styling."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BACKGROUND};
            }}
            QTreeWidget {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QTreeWidget::item {{
                padding: 4px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Colors.ACCENT};
            }}
            QHeaderView::section {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_SECONDARY};
                border: none;
                padding: 6px;
            }}
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
            }}
            QLineEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}
            QComboBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 6px;
            }}
            QLabel {{
                color: {Colors.TEXT_SECONDARY};
            }}
        """)


class AgentsDialog(BaseDevDialog):
    """Developer view for agent monitoring and control."""

    def __init__(self, felix_system, parent=None):
        super().__init__(felix_system, parent, "Agents - Developer View")
        self._polling_timer = None
        self._setup_ui()
        self._start_polling()

    def _setup_ui(self):
        """Set up agents view UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Active Agents")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_agents)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Agent list
        self._agent_tree = QTreeWidget()
        self._agent_tree.setHeaderLabels(["Agent ID", "Type", "Status", "Position"])
        self._agent_tree.setRootIsDecorated(False)
        self._agent_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._agent_tree.itemSelectionChanged.connect(self._on_agent_selected)
        splitter.addWidget(self._agent_tree)

        # Details panel
        details_frame = QFrame()
        details_layout = QVBoxLayout(details_frame)

        details_label = QLabel("Agent Details")
        details_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: 600;")
        details_layout.addWidget(details_label)

        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        details_layout.addWidget(self._details_text)

        splitter.addWidget(details_frame)
        splitter.setSizes([400, 300])

        layout.addWidget(splitter)

        # Stats row
        stats_row = QHBoxLayout()
        self._stats_label = QLabel("Agents: 0 | Running: 0 | Idle: 0")
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        stats_row.addWidget(self._stats_label)
        stats_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        stats_row.addWidget(close_btn)

        layout.addLayout(stats_row)

    def _start_polling(self):
        """Start polling for agent updates."""
        self._polling_timer = QTimer(self)
        self._polling_timer.timeout.connect(self._refresh_agents)
        self._polling_timer.start(2000)
        self._refresh_agents()

    def _refresh_agents(self):
        """Refresh agent list."""
        if not self._felix_system:
            return

        try:
            # Get agent registry from central_post (not directly on felix_system)
            registry = None
            if self._felix_system.central_post:
                registry = getattr(self._felix_system.central_post, 'agent_registry', None)
            if not registry:
                return

            agents = registry.get_active_agents() if hasattr(registry, 'get_active_agents') else []

            self._agent_tree.clear()

            # Count by phase
            exploration = 0
            analysis = 0
            synthesis = 0

            for agent in agents:
                # Use correct field names from AgentRegistry.get_active_agents()
                agent_id = agent.get('agent_id', 'unknown')
                agent_type = agent.get('agent_type', 'unknown')
                phase = agent.get('phase', 'unknown')
                depth_ratio = agent.get('depth_ratio', 0.0)

                if phase == 'exploration':
                    exploration += 1
                elif phase == 'analysis':
                    analysis += 1
                elif phase == 'synthesis':
                    synthesis += 1

                item = QTreeWidgetItem([
                    str(agent_id),
                    agent_type,
                    phase,
                    f"{depth_ratio:.2f}"
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, agent)
                self._agent_tree.addTopLevelItem(item)

            self._stats_label.setText(f"Agents: {len(agents)} | Explore: {exploration} | Analyze: {analysis} | Synth: {synthesis}")

        except Exception as e:
            logger.error(f"Error refreshing agents: {e}")

    def _on_agent_selected(self):
        """Handle agent selection."""
        item = self._agent_tree.currentItem()
        if not item:
            self._details_text.clear()
            return

        agent = item.data(0, Qt.ItemDataRole.UserRole)
        if agent:
            # Use correct field names from AgentRegistry.get_active_agents()
            details = f"Agent ID: {agent.get('agent_id', 'N/A')}\n"
            details += f"Type: {agent.get('agent_type', 'N/A')}\n"
            details += f"Phase: {agent.get('phase', 'N/A')}\n"
            details += f"Depth Ratio: {agent.get('depth_ratio', 'N/A')}\n"
            details += f"Avg Confidence: {agent.get('avg_confidence', 'N/A')}\n"
            self._details_text.setPlainText(details)

    def closeEvent(self, event):
        """Stop polling on close."""
        if self._polling_timer:
            self._polling_timer.stop()
        event.accept()


class MemoryDialog(BaseDevDialog):
    """Developer view for memory/task pattern browser."""

    def __init__(self, felix_system, parent=None):
        super().__init__(felix_system, parent, "Memory - Developer View")
        self._setup_ui()
        self._load_memory()

    def _setup_ui(self):
        """Set up memory view UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Task Memory & Patterns")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()

        # Search
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search patterns...")
        self._search_input.setFixedWidth(200)
        self._search_input.returnPressed.connect(self._search_memory)
        header.addWidget(self._search_input)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_memory)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Tabs for different memory types
        tabs = QTabWidget()

        # Task patterns tab
        patterns_widget = QWidget()
        patterns_layout = QVBoxLayout(patterns_widget)

        self._patterns_tree = QTreeWidget()
        self._patterns_tree.setHeaderLabels(["Pattern", "Success Rate", "Uses", "Last Used"])
        self._patterns_tree.setRootIsDecorated(False)
        self._patterns_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        patterns_layout.addWidget(self._patterns_tree)

        tabs.addTab(patterns_widget, "Task Patterns")

        # Workflow history tab
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)

        self._history_tree = QTreeWidget()
        self._history_tree.setHeaderLabels(["Workflow", "Status", "Duration", "Date"])
        self._history_tree.setRootIsDecorated(False)
        self._history_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self._history_tree)

        tabs.addTab(history_widget, "Workflow History")

        layout.addWidget(tabs)

        # Bottom row
        bottom_row = QHBoxLayout()
        self._stats_label = QLabel("Patterns: 0 | Workflows: 0")
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        bottom_row.addWidget(self._stats_label)
        bottom_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_row.addWidget(close_btn)

        layout.addLayout(bottom_row)

    def _load_memory(self):
        """Load memory data."""
        if not self._felix_system:
            return

        try:
            # Load task patterns using get_patterns() with empty query
            task_memory = getattr(self._felix_system, 'task_memory', None)
            self._patterns_tree.clear()
            if task_memory and hasattr(task_memory, 'get_patterns'):
                try:
                    from src.memory.task_memory import TaskMemoryQuery
                    from datetime import datetime
                    # Empty query with high limit gets all patterns
                    query = TaskMemoryQuery(limit=100)
                    patterns = task_memory.get_patterns(query)

                    for pattern in patterns:
                        # pattern is a TaskPattern dataclass
                        pattern_name = f"{pattern.task_type} ({pattern.complexity.value})"
                        success_rate = pattern.success_rate
                        usage = pattern.usage_count
                        last_used = datetime.fromtimestamp(pattern.updated_at).strftime('%Y-%m-%d') if pattern.updated_at else 'N/A'

                        item = QTreeWidgetItem([
                            pattern_name,
                            f"{success_rate:.0%}",
                            str(usage),
                            last_used
                        ])
                        self._patterns_tree.addTopLevelItem(item)
                except Exception as e:
                    logger.warning(f"Could not load task patterns: {e}")

            # Load workflow history directly from database
            self._history_tree.clear()
            if WORKFLOW_HISTORY_AVAILABLE:
                try:
                    workflow_history = WorkflowHistory()
                    workflows = workflow_history.get_workflow_outputs(limit=50)

                    for wf in workflows:
                        # wf is a WorkflowOutput dataclass
                        task = (wf.task_input or 'N/A')[:50]
                        status = wf.status or 'N/A'
                        duration = wf.processing_time or 0
                        date = (wf.created_at or 'N/A')[:10] if wf.created_at else 'N/A'

                        item = QTreeWidgetItem([
                            task,
                            status,
                            f"{duration:.1f}s",
                            date
                        ])
                        self._history_tree.addTopLevelItem(item)
                except Exception as e:
                    logger.warning(f"Could not load workflow history: {e}")

            self._stats_label.setText(
                f"Patterns: {self._patterns_tree.topLevelItemCount()} | "
                f"Workflows: {self._history_tree.topLevelItemCount()}"
            )

        except Exception as e:
            logger.error(f"Error loading memory: {e}")

    def _search_memory(self):
        """Search memory patterns."""
        query = self._search_input.text().strip()
        if not query:
            self._load_memory()
            return

        # Filter existing items
        for i in range(self._patterns_tree.topLevelItemCount()):
            item = self._patterns_tree.topLevelItem(i)
            text = item.text(0).lower()
            item.setHidden(query.lower() not in text)


class PromptsDialog(BaseDevDialog):
    """Developer view for prompt template management."""

    def __init__(self, felix_system, parent=None):
        super().__init__(felix_system, parent, "Prompts - Developer View")
        self._setup_ui()
        self._load_prompts()

    def _setup_ui(self):
        """Set up prompts view UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Prompt Templates")
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 16px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_prompts)
        header.addWidget(reload_btn)

        layout.addLayout(header)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Prompt list
        list_frame = QFrame()
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)

        # Category filter
        self._category_combo = QComboBox()
        self._category_combo.addItems(["All", "System", "Agent", "Workflow", "Custom"])
        self._category_combo.currentTextChanged.connect(self._filter_prompts)
        list_layout.addWidget(self._category_combo)

        self._prompts_tree = QTreeWidget()
        self._prompts_tree.setHeaderLabels(["Name", "Category"])
        self._prompts_tree.setRootIsDecorated(False)
        self._prompts_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._prompts_tree.itemSelectionChanged.connect(self._on_prompt_selected)
        list_layout.addWidget(self._prompts_tree)

        splitter.addWidget(list_frame)

        # Editor panel
        editor_frame = QFrame()
        editor_layout = QVBoxLayout(editor_frame)

        editor_label = QLabel("Prompt Content")
        editor_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-weight: 600;")
        editor_layout.addWidget(editor_label)

        self._prompt_editor = QTextEdit()
        self._prompt_editor.setReadOnly(True)
        editor_layout.addWidget(self._prompt_editor)

        splitter.addWidget(editor_frame)
        splitter.setSizes([250, 450])

        layout.addWidget(splitter)

        # Bottom row
        bottom_row = QHBoxLayout()
        self._stats_label = QLabel("Prompts: 0")
        self._stats_label.setStyleSheet(f"color: {Colors.TEXT_MUTED};")
        bottom_row.addWidget(self._stats_label)
        bottom_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom_row.addWidget(close_btn)

        layout.addLayout(bottom_row)

    def _load_prompts(self):
        """Load prompt templates."""
        self._prompts_tree.clear()
        self._all_prompts = []

        # Try to load from config/prompts.yaml
        try:
            import yaml
            from pathlib import Path

            prompts_file = Path("config/prompts.yaml")
            if prompts_file.exists():
                with open(prompts_file) as f:
                    prompts_data = yaml.safe_load(f) or {}

                for category, prompts in prompts_data.items():
                    if isinstance(prompts, dict):
                        for name, content in prompts.items():
                            self._all_prompts.append({
                                'name': name,
                                'category': category,
                                'content': content if isinstance(content, str) else str(content)
                            })

            # Add items to tree
            for prompt in self._all_prompts:
                item = QTreeWidgetItem([
                    prompt['name'],
                    prompt['category']
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, prompt)
                self._prompts_tree.addTopLevelItem(item)

            self._stats_label.setText(f"Prompts: {len(self._all_prompts)}")

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            self._stats_label.setText(f"Error loading prompts")

    def _filter_prompts(self, category: str):
        """Filter prompts by category."""
        for i in range(self._prompts_tree.topLevelItemCount()):
            item = self._prompts_tree.topLevelItem(i)
            if category == "All":
                item.setHidden(False)
            else:
                item.setHidden(item.text(1).lower() != category.lower())

    def _on_prompt_selected(self):
        """Handle prompt selection."""
        item = self._prompts_tree.currentItem()
        if not item:
            self._prompt_editor.clear()
            return

        prompt = item.data(0, Qt.ItemDataRole.UserRole)
        if prompt:
            self._prompt_editor.setPlainText(prompt.get('content', ''))
