"""Terminal view showing active commands and history."""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from PySide6.QtCore import Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QTextEdit, QSplitter,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit, QComboBox, QMessageBox
)
from PySide6.QtGui import QFont

from ..core.theme import Colors

logger = logging.getLogger(__name__)


class TerminalView(QWidget):
    """Terminal view with active commands and history.

    Features:
    - Active commands with live output streaming
    - Command history with filtering
    - Kill command functionality
    """

    command_kill_requested = Signal(int)  # execution_id

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._active_commands: Dict[int, Dict[str, Any]] = {}
        self._active_outputs: Dict[int, List[str]] = {}
        self._felix_system = None
        self._polling_timer: Optional[QTimer] = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up terminal view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter for active/history sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
        """)

        # Active commands section
        active_section = self._create_active_section()
        splitter.addWidget(active_section)

        # History section
        history_section = self._create_history_section()
        splitter.addWidget(history_section)

        # Set initial sizes (60% active, 40% history)
        splitter.setSizes([300, 200])

        layout.addWidget(splitter)

    def _create_active_section(self) -> QFrame:
        """Create the active commands section."""
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {Colors.BACKGROUND};")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()

        title = QLabel("Active Commands")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 13px;
            font-weight: 600;
        """)
        header.addWidget(title)
        header.addStretch()

        # Kill button
        self._kill_btn = QPushButton("Kill")
        self._kill_btn.setProperty("danger", True)
        self._kill_btn.setEnabled(False)
        self._kill_btn.clicked.connect(self._on_kill_clicked)
        header.addWidget(self._kill_btn)

        layout.addLayout(header)

        # Active commands tree
        self._active_tree = QTreeWidget()
        self._active_tree.setHeaderLabels(["Command", "Agent", "Duration"])
        self._active_tree.setRootIsDecorated(False)
        self._active_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._active_tree.setMaximumHeight(120)
        self._active_tree.setStyleSheet(f"""
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
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        self._active_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._active_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._active_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._active_tree.setColumnWidth(1, 80)
        self._active_tree.setColumnWidth(2, 60)
        self._active_tree.itemSelectionChanged.connect(self._on_active_selection_changed)
        layout.addWidget(self._active_tree)

        # Output display
        output_label = QLabel("Live Output:")
        output_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(output_label)

        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setFont(QFont("monospace", 10))
        self._output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        layout.addWidget(self._output_text, 1)

        return frame

    def _create_history_section(self) -> QFrame:
        """Create the command history section."""
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {Colors.BACKGROUND};")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        title = QLabel("Command History")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 13px;
            font-weight: 600;
        """)
        layout.addWidget(title)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search...")
        self._search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px 8px;
            }}
        """)
        self._search_input.returnPressed.connect(self._refresh_history)
        filter_row.addWidget(self._search_input, 1)

        self._status_filter = QComboBox()
        self._status_filter.addItems(["All", "Completed", "Failed", "Running"])
        self._status_filter.setFixedWidth(90)
        self._status_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.SURFACE};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 4px;
                padding: 4px;
            }}
        """)
        self._status_filter.currentTextChanged.connect(self._refresh_history)
        filter_row.addWidget(self._status_filter)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_history)
        filter_row.addWidget(refresh_btn)

        layout.addLayout(filter_row)

        # History tree
        self._history_tree = QTreeWidget()
        self._history_tree.setHeaderLabels(["ID", "Command", "Status", "Duration"])
        self._history_tree.setRootIsDecorated(False)
        self._history_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._history_tree.setStyleSheet(f"""
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
                padding: 4px 8px;
                font-size: 11px;
            }}
        """)
        self._history_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._history_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._history_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._history_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._history_tree.setColumnWidth(0, 40)
        self._history_tree.setColumnWidth(2, 70)
        self._history_tree.setColumnWidth(3, 60)
        self._history_tree.itemDoubleClicked.connect(self._on_history_double_click)
        layout.addWidget(self._history_tree, 1)

        return frame

    def set_felix_system(self, felix_system):
        """Set the Felix system reference for data access."""
        self._felix_system = felix_system
        if felix_system:
            self._start_polling()
            self._refresh_history()
        else:
            self._stop_polling()

    def _start_polling(self):
        """Start polling for active commands."""
        if not self._polling_timer:
            self._polling_timer = QTimer(self)
            self._polling_timer.timeout.connect(self._poll_active_commands)
            self._polling_timer.start(1000)  # Poll every second
            logger.debug("Terminal polling started")

    def _stop_polling(self):
        """Stop polling."""
        if self._polling_timer:
            self._polling_timer.stop()
            self._polling_timer = None
            logger.debug("Terminal polling stopped")

    def _poll_active_commands(self):
        """Poll for active commands."""
        if not self._felix_system:
            return

        try:
            central_post = self._felix_system.central_post
            if not central_post or not central_post.command_history:
                return

            # Get active commands
            active = central_post.command_history.get_active_commands()

            # Also get recent completions
            recent_cutoff = time.time() - 30.0
            recent_all = central_post.command_history.get_filtered_history(
                date_from=recent_cutoff,
                limit=10
            )

            # Combine and deduplicate
            combined = []
            seen_ids = set()

            for cmd in active:
                if cmd['execution_id'] not in seen_ids:
                    combined.append(cmd)
                    seen_ids.add(cmd['execution_id'])

            for cmd in recent_all:
                if cmd['execution_id'] not in seen_ids:
                    combined.append(cmd)
                    seen_ids.add(cmd['execution_id'])

            self._update_active_display(combined)

        except Exception as e:
            logger.error(f"Error polling active commands: {e}")

    def _update_active_display(self, commands: List[Dict[str, Any]]):
        """Update active commands display."""
        # Track current selection
        current_item = self._active_tree.currentItem()
        selected_id = int(current_item.data(0, Qt.ItemDataRole.UserRole)) if current_item else None

        # Clear tree
        self._active_tree.clear()

        # Update commands
        new_active = {}
        for cmd in commands:
            exec_id = cmd['execution_id']
            new_active[exec_id] = cmd

            # Calculate duration
            start_time = cmd.get('execution_timestamp', time.time())
            duration = time.time() - start_time

            # Status indicator
            status = cmd.get('status', 'unknown')
            if status == 'running':
                status_icon = "..."
            elif status == 'completed':
                status_icon = "ok"
            elif status == 'failed':
                status_icon = "err"
            else:
                status_icon = "?"

            # Truncate command
            command = cmd['command']
            if len(command) > 40:
                command = command[:37] + "..."

            # Create item
            item = QTreeWidgetItem([
                f"{status_icon} {command}",
                cmd.get('agent_id', 'N/A')[:10],
                f"{duration:.1f}s"
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, exec_id)
            self._active_tree.addTopLevelItem(item)

            # Retrieve live output
            self._update_command_output(exec_id)

        self._active_commands = new_active

        # Restore selection
        if selected_id:
            for i in range(self._active_tree.topLevelItemCount()):
                item = self._active_tree.topLevelItem(i)
                if item.data(0, Qt.ItemDataRole.UserRole) == selected_id:
                    self._active_tree.setCurrentItem(item)
                    break

    def _update_command_output(self, execution_id: int):
        """Update live output for a command."""
        if not self._felix_system:
            return

        try:
            central_post = self._felix_system.central_post
            if central_post:
                live_output = central_post.get_live_command_output(execution_id)
                if live_output:
                    if execution_id not in self._active_outputs:
                        self._active_outputs[execution_id] = []

                    current_size = len(self._active_outputs[execution_id])
                    for i, (line, stream_type) in enumerate(live_output):
                        if i >= current_size:
                            prefix = "[ERR] " if stream_type == "stderr" else ""
                            self._active_outputs[execution_id].append(f"{prefix}{line}")

        except Exception as e:
            logger.error(f"Error getting live output: {e}")

    def _on_active_selection_changed(self):
        """Handle active command selection change."""
        item = self._active_tree.currentItem()
        if item:
            exec_id = item.data(0, Qt.ItemDataRole.UserRole)

            # Update output display
            if exec_id in self._active_outputs:
                output = "\n".join(self._active_outputs[exec_id])
                self._output_text.setPlainText(output)
                # Scroll to bottom
                scrollbar = self._output_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())

            # Enable kill button for running commands
            if exec_id in self._active_commands:
                status = self._active_commands[exec_id].get('status', '')
                self._kill_btn.setEnabled(status == 'running')
        else:
            self._kill_btn.setEnabled(False)

    def _on_kill_clicked(self):
        """Handle kill button click."""
        item = self._active_tree.currentItem()
        if not item:
            return

        exec_id = item.data(0, Qt.ItemDataRole.UserRole)
        if exec_id not in self._active_commands:
            return

        cmd = self._active_commands[exec_id]

        reply = QMessageBox.question(
            self,
            "Confirm Kill",
            f"Kill command #{exec_id}?\n\n{cmd.get('command', 'N/A')[:100]}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._kill_command(exec_id)

    def _kill_command(self, execution_id: int):
        """Kill a command."""
        if not self._felix_system:
            return

        try:
            central_post = self._felix_system.central_post
            if central_post and central_post.command_history:
                success = central_post.command_history.cancel_command(execution_id)
                if success:
                    logger.info(f"Cancelled command #{execution_id}")
                    self._refresh_history()
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Failed to cancel command #{execution_id}"
                    )
        except Exception as e:
            logger.error(f"Error killing command: {e}")
            QMessageBox.critical(self, "Error", str(e))

    @Slot()
    def _refresh_history(self):
        """Refresh command history."""
        if not self._felix_system:
            return

        try:
            central_post = self._felix_system.central_post
            if not central_post or not central_post.command_history:
                return

            # Build filters
            search = self._search_input.text().strip() or None
            status_text = self._status_filter.currentText().lower()
            status = None if status_text == "all" else status_text

            # Query history
            history = central_post.command_history.get_filtered_history(
                search_query=search,
                status=status,
                limit=50
            )

            # Update tree
            self._history_tree.clear()

            for cmd in history:
                # Truncate command
                command = cmd['command']
                if len(command) > 50:
                    command = command[:47] + "..."

                # Format duration
                duration = cmd.get('duration')
                duration_str = f"{duration:.1f}s" if duration else "N/A"

                item = QTreeWidgetItem([
                    str(cmd['execution_id']),
                    command,
                    cmd.get('status', 'unknown'),
                    duration_str
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, cmd['execution_id'])
                self._history_tree.addTopLevelItem(item)

        except Exception as e:
            logger.error(f"Error refreshing history: {e}")

    def _on_history_double_click(self, item: QTreeWidgetItem, column: int):
        """Handle double-click on history item."""
        exec_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not self._felix_system:
            return

        try:
            central_post = self._felix_system.central_post
            if central_post and central_post.command_history:
                details = central_post.command_history.get_command_details(exec_id)
                if details:
                    self._show_command_details(details)
        except Exception as e:
            logger.error(f"Error getting command details: {e}")

    def _show_command_details(self, details: Dict[str, Any]):
        """Show command details in a message box."""
        msg = f"""Command #{details.get('execution_id', 'N/A')}

Command: {details.get('command', 'N/A')}
Status: {details.get('status', 'unknown')}
Exit Code: {details.get('exit_code', 'N/A')}
Duration: {details.get('duration', 'N/A')}s
Agent: {details.get('agent_id', 'N/A')}

Output:
{details.get('stdout_preview', '(no output)')[:500]}

Errors:
{details.get('stderr_preview', '(none)')[:200]}
"""
        QMessageBox.information(self, "Command Details", msg)

    def cleanup(self):
        """Clean up resources."""
        self._stop_polling()
        self._active_commands.clear()
        self._active_outputs.clear()
