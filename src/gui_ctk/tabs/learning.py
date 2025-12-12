"""
Learning Tab for Felix GUI (CustomTkinter Edition)

This tab provides visualization and management of Felix's self-improvement systems:
- Feedback integration and synthesis tracking
- Agent confidence calibration
- Pattern learning and recommendations
- Threshold optimization
- Meta-learning boost configuration
- Concept registry visualization
"""

import customtkinter as ctk
import tkinter as tk
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime

from ..utils import logger, ThreadManager
from ..theme_manager import get_theme_manager
from ..components.status_card import StatusCard
from ..components.themed_treeview import ThemedTreeview

# Import Felix learning modules (with fallback)
try:
    from src.learning import RecommendationEngine
    from src.workflows.concept_registry import ConceptRegistry
    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import learning modules: {e}")
    LEARNING_AVAILABLE = False


class LearningTab(ctk.CTkFrame):
    """
    Learning tab with self-improvement system controls and visualization.

    Displays:
    - Feedback integration status
    - Agent performance metrics (synthesis integration rate)
    - Confidence calibration statistics
    - Pattern learning recommendations
    - Threshold optimization
    - Concept registry (terminology consistency tracking)
    - Meta-learning boost configuration
    """

    def __init__(self, master, thread_manager: ThreadManager, main_app=None, **kwargs):
        """
        Initialize Learning tab.

        Args:
            master: Parent widget (typically CTkTabview)
            thread_manager: ThreadManager instance
            main_app: Reference to main FelixApp
            **kwargs: Additional arguments passed to CTkFrame
        """
        super().__init__(master, **kwargs)

        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = get_theme_manager()
        self._layout_manager = None

        # Refresh timer
        self.refresh_timer = None
        self.auto_refresh_enabled = False

        # Features enabled flag
        self._features_enabled = False

        self._setup_ui()

    def set_layout_manager(self, layout_manager):
        """Set the layout manager (interface compliance)."""
        self._layout_manager = layout_manager

    def _setup_ui(self):
        """Set up the learning tab UI."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Content area expands

        # Header section
        self._create_header()

        # Status cards section
        self._create_status_section()

        # Tabbed content section
        self._create_content_section()

    def _create_header(self):
        """Create the header section with title and controls."""
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        self.title_label = ctk.CTkLabel(
            title_frame,
            text="Felix Learning Systems",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Self-Improvement Architecture with Feedback Loops",
            font=ctk.CTkFont(size=12),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.subtitle_label.pack(anchor="w")

        # Control buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.grid(row=0, column=2, sticky="e")

        self.refresh_button = ctk.CTkButton(
            button_frame,
            text="Refresh Statistics",
            command=self._refresh_statistics,
            width=140,
            height=32
        )
        self.refresh_button.pack(side="left", padx=5)

        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.auto_refresh_check = ctk.CTkCheckBox(
            button_frame,
            text="Auto-refresh (30s)",
            variable=self.auto_refresh_var,
            command=self._toggle_auto_refresh,
            width=120
        )
        self.auto_refresh_check.pack(side="left", padx=5)

        self.export_button = ctk.CTkButton(
            button_frame,
            text="Export Data",
            command=self._export_learning_data,
            width=110,
            height=32,
            fg_color="transparent",
            border_width=1
        )
        self.export_button.pack(side="left", padx=5)

        self.reset_button = ctk.CTkButton(
            button_frame,
            text="Reset All",
            command=self._reset_learning_confirm,
            width=100,
            height=32,
            fg_color=self.theme_manager.get_color("error"),
            hover_color="#a93226"
        )
        self.reset_button.pack(side="left", padx=(20, 0))

    def _create_status_section(self):
        """Create the status cards section."""
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)

        # System status card
        self.system_status_card = StatusCard(
            status_frame,
            title="Learning Status",
            value="Inactive",
            subtitle="System not started",
            status_color=self.theme_manager.get_color("error"),
            width=200
        )
        self.system_status_card.pack(side="left", padx=(0, 10))

        # Active systems card
        self.systems_card = StatusCard(
            status_frame,
            title="Active Systems",
            value="0/4",
            subtitle="Learning modules",
            width=160
        )
        self.systems_card.pack(side="left", padx=(0, 10))

        # Data points card
        self.data_points_card = StatusCard(
            status_frame,
            title="Total Data",
            value="0",
            subtitle="Learning samples",
            width=160
        )
        self.data_points_card.pack(side="left", padx=(0, 10))

        # Success rate card
        self.success_rate_card = StatusCard(
            status_frame,
            title="Success Rate",
            value="--",
            subtitle="Pattern accuracy",
            width=160
        )
        self.success_rate_card.pack(side="left", padx=(0, 10))

        # Calibration card
        self.calibration_card = StatusCard(
            status_frame,
            title="Calibration Error",
            value="--",
            subtitle="Agent confidence",
            width=180
        )
        self.calibration_card.pack(side="left")

    def _create_content_section(self):
        """Create the tabbed content section."""
        # Create tabview for sub-sections
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))

        # Add tabs
        self.tabview.add("Overview")
        self.tabview.add("Feedback")
        self.tabview.add("Agent Performance")
        self.tabview.add("Patterns")
        self.tabview.add("Calibration")
        self.tabview.add("Thresholds")
        self.tabview.add("Concepts")
        self.tabview.add("Meta-Learning")

        # Populate each tab
        self._create_overview_tab()
        self._create_feedback_tab()
        self._create_agent_performance_tab()
        self._create_patterns_tab()
        self._create_calibration_tab()
        self._create_thresholds_tab()
        self._create_concepts_tab()
        self._create_meta_learning_tab()

    def _create_overview_tab(self):
        """Create overview statistics tab."""
        tab = self.tabview.tab("Overview")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Scrollable text area
        self.overview_text = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.overview_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _create_feedback_tab(self):
        """Create feedback integration tab."""
        tab = self.tabview.tab("Feedback")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Configuration section
        config_frame = ctk.CTkFrame(tab)
        config_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        config_frame.grid_columnconfigure(1, weight=1)

        # Feedback enabled toggle
        self.feedback_enabled_var = tk.BooleanVar(value=True)
        feedback_check = ctk.CTkCheckBox(
            config_frame,
            text="Enable Feedback Integration Protocol",
            variable=self.feedback_enabled_var,
            command=self._toggle_feedback
        )
        feedback_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        # Info label
        info_label = ctk.CTkLabel(
            config_frame,
            text="Feedback integration broadcasts synthesis results to agents for continuous improvement.",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        # Statistics display
        self.feedback_text = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.feedback_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_agent_performance_tab(self):
        """Create agent performance metrics tab."""
        tab = self.tabview.tab("Agent Performance")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Info section
        info_frame = ctk.CTkFrame(tab, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Tracks agent synthesis integration rate and contribution quality",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(anchor="w")

        # Performance treeview
        self.agent_performance_tree = ThemedTreeview(
            tab,
            columns=["agent_id", "integration_rate", "avg_confidence", "samples"],
            headings=["Agent ID", "Integration Rate", "Avg Confidence", "Samples"],
            widths=[200, 150, 150, 100],
            height=15
        )
        self.agent_performance_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_patterns_tab(self):
        """Create pattern learning tab."""
        tab = self.tabview.tab("Patterns")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Info section
        info_frame = ctk.CTkFrame(tab, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Historical workflow patterns and recommendations (requires ≥10 samples)",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(anchor="w")

        # Pattern statistics text
        self.patterns_text = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.patterns_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_calibration_tab(self):
        """Create confidence calibration tab."""
        tab = self.tabview.tab("Calibration")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Info section
        info_frame = ctk.CTkFrame(tab, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Agent confidence calibration: predicted vs actual performance (requires ≥10 workflows)",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(anchor="w")

        # Calibration treeview
        self.calibration_tree = ThemedTreeview(
            tab,
            columns=["agent_type", "complexity", "factor", "error", "samples"],
            headings=["Agent Type", "Complexity", "Calibration Factor", "Error", "Samples"],
            widths=[150, 120, 150, 100, 80],
            height=15
        )
        self.calibration_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_thresholds_tab(self):
        """Create threshold learning tab."""
        tab = self.tabview.tab("Thresholds")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Info section
        info_frame = ctk.CTkFrame(tab, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Optimal threshold values learned per task type (requires ≥20 workflows)",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(anchor="w")

        # Threshold treeview
        self.thresholds_tree = ThemedTreeview(
            tab,
            columns=["task_type", "threshold_name", "learned_value", "success_rate", "samples"],
            headings=["Task Type", "Threshold", "Learned Value", "Success Rate", "Samples"],
            widths=[120, 180, 130, 120, 80],
            height=15
        )
        self.thresholds_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_concepts_tab(self):
        """Create concept registry tab."""
        tab = self.tabview.tab("Concepts")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Info section
        info_frame = ctk.CTkFrame(tab, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Workflow-scoped concept tracking ensures terminology consistency across agents",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        info_label.pack(anchor="w", side="left")

        # Note about registry scope
        note_label = ctk.CTkLabel(
            info_frame,
            text="(Shows concepts from last completed workflow)",
            font=ctk.CTkFont(size=10),
            text_color=self.theme_manager.get_color("warning")
        )
        note_label.pack(anchor="w", side="left", padx=(10, 0))

        # Concepts treeview
        self.concepts_tree = ThemedTreeview(
            tab,
            columns=["concept", "definition", "source", "confidence", "usage"],
            headings=["Concept", "Definition", "Source Agent", "Confidence", "Usage Count"],
            widths=[150, 300, 120, 100, 100],
            height=15
        )
        self.concepts_tree.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_meta_learning_tab(self):
        """Create meta-learning configuration tab."""
        tab = self.tabview.tab("Meta-Learning")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Configuration section
        config_frame = ctk.CTkFrame(tab)
        config_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        config_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_label = ctk.CTkLabel(
            config_frame,
            text="Meta-Learning Boost Configuration",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))

        # Info
        info_label = ctk.CTkLabel(
            config_frame,
            text="Meta-learning tracks which knowledge entries help specific task types and boosts their retrieval relevance.",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted"),
            wraplength=600
        )
        info_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 10))

        # Enable meta-learning
        self.meta_learning_enabled_var = tk.BooleanVar(value=True)
        meta_check = ctk.CTkCheckBox(
            config_frame,
            text="Enable Meta-Learning Boost",
            variable=self.meta_learning_enabled_var,
            command=self._toggle_meta_learning
        )
        meta_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Minimum samples slider
        samples_label = ctk.CTkLabel(
            config_frame,
            text="Minimum samples for boost:",
            font=ctk.CTkFont(size=11)
        )
        samples_label.grid(row=3, column=0, sticky="w", padx=10, pady=(10, 5))

        self.min_samples_var = tk.IntVar(value=3)
        self.min_samples_slider = ctk.CTkSlider(
            config_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            variable=self.min_samples_var,
            command=self._update_min_samples_label
        )
        self.min_samples_slider.grid(row=4, column=0, sticky="ew", padx=10, pady=5)

        self.min_samples_value_label = ctk.CTkLabel(
            config_frame,
            text="3 samples",
            font=ctk.CTkFont(size=11),
            text_color=self.theme_manager.get_color("fg_muted")
        )
        self.min_samples_value_label.grid(row=4, column=1, sticky="w", padx=10)

        # Statistics display
        stats_label = ctk.CTkLabel(
            tab,
            text="Meta-Learning Statistics",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        stats_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10, 5))

        self.meta_learning_text = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.meta_learning_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        tab.grid_rowconfigure(2, weight=1)

    # Event handlers

    def _refresh_statistics(self):
        """Refresh all learning statistics."""
        if not LEARNING_AVAILABLE:
            self._update_overview_text("Learning modules not available.\nCheck that all dependencies are installed.")
            self._update_system_status("Unavailable", "error")
            return

        if not self.main_app or not hasattr(self.main_app, 'felix_system'):
            self._update_overview_text("Felix system not started.\nStart the system from Dashboard tab.")
            self._update_system_status("Inactive", "error")
            return

        # Run refresh in background thread
        self.thread_manager.start_thread(target=self._refresh_statistics_bg)

    def _refresh_statistics_bg(self):
        """Background thread for refreshing statistics."""
        try:
            felix_system = self.main_app.felix_system

            if not felix_system:
                self._schedule_update(self._update_overview_text,
                    "Felix system not started.\nStart the system from Dashboard tab.")
                self._schedule_update(self._update_system_status, "Inactive", "error")
                return

            # Check if learning is enabled
            if not hasattr(felix_system, 'config') or not felix_system.config.enable_learning:
                self._schedule_update(self._update_overview_text,
                    "Learning systems disabled in configuration.\nEnable in Settings tab.")
                self._schedule_update(self._update_system_status, "Disabled", "warning")
                return

            # Get task memory
            task_memory = getattr(felix_system, 'task_memory', None)
            if not task_memory:
                self._schedule_update(self._update_overview_text, "Task memory not available.")
                self._schedule_update(self._update_system_status, "Error", "error")
                return

            # Create recommendation engine
            rec_engine = RecommendationEngine(
                task_memory=task_memory,
                enable_auto_apply=False  # Read-only for stats
            )

            # Get unified statistics
            stats = rec_engine.get_unified_statistics(days=30)

            # Update all tabs
            self._schedule_update(self._update_overview_stats, stats)
            self._schedule_update(self._update_feedback_stats, rec_engine)
            self._schedule_update(self._update_agent_performance_stats, felix_system)
            self._schedule_update(self._update_patterns_stats, rec_engine)
            self._schedule_update(self._update_calibration_stats, rec_engine)
            self._schedule_update(self._update_thresholds_stats, rec_engine)
            self._schedule_update(self._update_concepts_stats, felix_system)
            self._schedule_update(self._update_meta_learning_stats, felix_system)

            # Update status cards
            overall = stats.get('overall', {})
            systems_active = overall.get('systems_with_data', 0)
            total_data = overall.get('total_data_points', 0)

            if systems_active >= 3:
                self._schedule_update(self._update_system_status, "Active", "success")
            elif systems_active > 0:
                self._schedule_update(self._update_system_status, f"Partial ({systems_active}/4)", "warning")
            else:
                self._schedule_update(self._update_system_status, "No Data", "error")

            self._schedule_update(self._update_status_cards, stats)

        except Exception as e:
            logger.error(f"Failed to refresh learning statistics: {e}")
            self._schedule_update(self._update_overview_text, f"Error loading statistics:\n{str(e)}")
            self._schedule_update(self._update_system_status, "Error", "error")

    def _schedule_update(self, callback, *args):
        """Schedule a GUI update on the main thread."""
        if self.thread_manager.is_active:
            self.after(0, callback, *args)

    def _update_system_status(self, value: str, status: str):
        """Update system status card."""
        status_colors = {
            "success": self.theme_manager.get_color("success"),
            "warning": self.theme_manager.get_color("warning"),
            "error": self.theme_manager.get_color("error")
        }

        subtitles = {
            "Active": "All systems operational",
            "Inactive": "System not started",
            "Disabled": "Enable in settings",
            "Error": "Check system status",
            "No Data": "Waiting for workflows",
            "Unavailable": "Install dependencies"
        }

        self.system_status_card.set_value(value)
        self.system_status_card.set_subtitle(subtitles.get(value, "Unknown"))
        if status in status_colors:
            self.system_status_card.set_status_color(status_colors[status])

    def _update_status_cards(self, stats: Dict[str, Any]):
        """Update all status cards."""
        overall = stats.get('overall', {})

        # Active systems
        systems_active = overall.get('systems_with_data', 0)
        self.systems_card.set_value(f"{systems_active}/4")

        # Total data points
        total_data = overall.get('total_data_points', 0)
        self.data_points_card.set_value(f"{total_data:,}")

        # Pattern success rate
        pattern_stats = stats.get('pattern_learner', {})
        success_rate = pattern_stats.get('success_rate')
        if success_rate is not None:
            self.success_rate_card.set_value(f"{success_rate:.1%}")
        else:
            self.success_rate_card.set_value("--")

        # Calibration error
        cal_stats = stats.get('confidence_calibrator', {})
        cal_error = cal_stats.get('avg_calibration_error')
        if cal_error is not None:
            self.calibration_card.set_value(f"{cal_error:.3f}")
        else:
            self.calibration_card.set_value("--")

    def _update_overview_text(self, text: str):
        """Update overview text area."""
        self.overview_text.delete("1.0", "end")
        self.overview_text.insert("1.0", text)

    def _update_overview_stats(self, stats: Dict[str, Any]):
        """Update overview tab with unified statistics."""
        text = "FELIX LEARNING SYSTEMS - OVERVIEW\n"
        text += "=" * 80 + "\n\n"

        overall = stats.get('overall', {})
        text += f"Status: {'Active' if overall.get('learning_active') else 'Inactive'}\n"
        text += f"Systems with data: {overall.get('systems_with_data', 0)}/4\n"
        text += f"Total data points: {overall.get('total_data_points', 0):,}\n"
        text += f"Reporting period: Last {stats.get('days', 30)} days\n\n"

        # Pattern learner summary
        pattern_stats = stats.get('pattern_learner', {})
        text += "1. PATTERN LEARNER\n"
        text += "-" * 80 + "\n"
        total_recs = pattern_stats.get('total_recommendations', 0)
        text += f"   Total recommendations: {total_recs}\n"
        text += f"   Applied: {pattern_stats.get('applied_count', 0) or 0}\n"
        text += f"   Success count: {pattern_stats.get('success_count', 0) or 0}\n"
        if pattern_stats.get('success_rate') is not None and total_recs > 0:
            text += f"   Success rate: {pattern_stats['success_rate']:.1%}\n"
        elif total_recs == 0:
            text += f"   Status: No data yet (need ≥10 patterns)\n"
        text += "\n"

        # Confidence calibrator summary
        calibration_stats = stats.get('confidence_calibrator', {})
        text += "2. CONFIDENCE CALIBRATOR\n"
        text += "-" * 80 + "\n"
        total_cal_records = calibration_stats.get('total_records', 0)
        text += f"   Total records: {total_cal_records}\n"
        text += f"   Total samples: {calibration_stats.get('total_samples', 0)}\n"
        if calibration_stats.get('avg_calibration_error') is not None:
            text += f"   Avg calibration error: {calibration_stats['avg_calibration_error']:.3f}\n"

        if calibration_stats.get('most_overconfident'):
            oc = calibration_stats['most_overconfident']
            text += f"   Most overconfident: {oc['agent_type']}/{oc['task_complexity']} "
            text += f"(factor={oc['calibration_factor']:.3f})\n"

        if total_cal_records == 0:
            text += f"   Status: No data yet (need ≥10 workflows)\n"
        text += "\n"

        # Threshold learner summary
        threshold_stats = stats.get('threshold_learner', {})
        text += "3. THRESHOLD LEARNER\n"
        text += "-" * 80 + "\n"
        total_threshold_records = threshold_stats.get('total_records', 0)
        text += f"   Total records: {total_threshold_records}\n"
        text += f"   Total samples: {threshold_stats.get('total_samples', 0)}\n"
        if threshold_stats.get('avg_success_rate') is not None:
            text += f"   Avg success rate: {threshold_stats['avg_success_rate']:.1%}\n"
        if total_threshold_records == 0:
            text += f"   Status: No data yet (need ≥20 workflows)\n"
        text += "\n"

        # Feedback integration summary
        text += "4. FEEDBACK INTEGRATION\n"
        text += "-" * 80 + "\n"
        text += f"   Status: {'Enabled' if self.feedback_enabled_var.get() else 'Disabled'}\n"
        text += f"   Broadcasts synthesis feedback to agents for self-improvement\n"
        text += "\n"

        text += "=" * 80 + "\n"
        text += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        self._update_overview_text(text)

    def _update_feedback_stats(self, rec_engine):
        """Update feedback integration tab."""
        text = "FEEDBACK INTEGRATION PROTOCOL\n"
        text += "=" * 80 + "\n\n"

        text += f"Status: {'Enabled' if self.feedback_enabled_var.get() else 'Disabled'}\n\n"

        text += "DESCRIPTION\n"
        text += "-" * 80 + "\n"
        text += "After each synthesis, CentralPost broadcasts feedback to all agents via:\n"
        text += "- SYNTHESIS_FEEDBACK messages (overall synthesis quality)\n"
        text += "- CONTRIBUTION_EVALUATION messages (individual agent usefulness)\n\n"

        text += "Agents track their 'synthesis integration rate' - how often their\n"
        text += "contributions are used in final synthesis. This enables:\n"
        text += "- Confidence calibration (predicted vs actual usefulness)\n"
        text += "- Adaptive behavior (agents adjust when usefulness < 0.3)\n"
        text += "- Self-improvement through explicit feedback loops\n\n"

        text += "BENEFITS\n"
        text += "-" * 80 + "\n"
        text += "✓ Agents learn which contributions are valuable\n"
        text += "✓ Confidence scores become more accurate over time\n"
        text += "✓ Low-performing agents can adapt their approach\n"
        text += "✓ High-performing agents gain confidence boost\n"
        text += "✓ Systemic learning across the agent team\n\n"

        text += "IMPLEMENTATION\n"
        text += "-" * 80 + "\n"
        text += "Location: src/communication/central_post.py\n"
        text += "Methods: broadcast_synthesis_feedback(), process_synthesis_feedback()\n"
        text += "Messages: SYNTHESIS_FEEDBACK, CONTRIBUTION_EVALUATION\n"
        text += "Tracking: synthesis_integration_rate per agent\n"

        self.feedback_text.delete("1.0", "end")
        self.feedback_text.insert("1.0", text)

    def _update_agent_performance_stats(self, felix_system):
        """Update agent performance metrics."""
        # Clear existing
        self.agent_performance_tree.clear()

        # This would require tracking in felix_system
        # For now, show placeholder
        text = "Agent performance tracking requires workflow completion data.\n"
        text += "This feature tracks synthesis integration rate per agent.\n\n"
        text += "Coming soon: Real-time agent performance metrics."

        # Insert placeholder message
        self.agent_performance_tree.insert(
            values=("No data yet", "--", "--", "--")
        )

    def _update_patterns_stats(self, rec_engine):
        """Update pattern recommendations tab."""
        text = "PATTERN RECOMMENDATIONS\n"
        text += "=" * 80 + "\n\n"

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
                text += "(Requires ≥10 historical patterns in TaskMemory)\n"

            text += "\n" + "-" * 80 + "\n"
            text += "Pattern recommendations track workflow optimization opportunities\n"
            text += "based on historical task execution patterns.\n\n"
            text += "Thresholds:\n"
            text += "  Auto-apply: ≥95% success + ≥20 samples\n"
            text += "  Recommend: ≥80% success + ≥10 samples\n"

        except Exception as e:
            text += f"\nError loading pattern statistics: {e}\n"

        self.patterns_text.delete("1.0", "end")
        self.patterns_text.insert("1.0", text)

    def _update_calibration_stats(self, rec_engine):
        """Update confidence calibration tab."""
        # Clear existing
        self.calibration_tree.clear()

        try:
            records = rec_engine.confidence_calibrator.get_all_calibration_records()

            if records:
                for rec in records:
                    # Determine interpretation
                    if rec.calibration_factor > 1.1:
                        interpretation = "Underconfident"
                    elif rec.calibration_factor < 0.9:
                        interpretation = "Overconfident"
                    else:
                        interpretation = "Well calibrated"

                    self.calibration_tree.insert(
                        values=(
                            f"{rec.agent_type} ({interpretation})",
                            rec.task_complexity,
                            f"{rec.calibration_factor:.3f}",
                            f"{rec.calibration_error:.3f}",
                            str(rec.sample_size)
                        )
                    )
            else:
                self.calibration_tree.insert(
                    values=("No data yet", "--", "--", "--", "--")
                )

        except Exception as e:
            logger.error(f"Failed to load calibration statistics: {e}")
            self.calibration_tree.insert(
                values=("Error loading data", "--", "--", "--", "--")
            )

    def _update_thresholds_stats(self, rec_engine):
        """Update threshold learning tab."""
        # Clear existing
        self.thresholds_tree.clear()

        try:
            records = rec_engine.threshold_learner.get_all_threshold_records()

            if records:
                for rec in records:
                    self.thresholds_tree.insert(
                        values=(
                            rec.task_type,
                            rec.threshold_name,
                            f"{rec.learned_value:.3f}",
                            f"{rec.success_rate:.1%}",
                            str(rec.sample_size)
                        )
                    )
            else:
                self.thresholds_tree.insert(
                    values=("No data yet", "--", "--", "--", "--")
                )

        except Exception as e:
            logger.error(f"Failed to load threshold statistics: {e}")
            self.thresholds_tree.insert(
                values=("Error loading data", "--", "--", "--", "--")
            )

    def _update_concepts_stats(self, felix_system):
        """Update concept registry tab."""
        # Clear existing
        self.concepts_tree.clear()

        # This would require accessing the last workflow's concept registry
        # For now, show placeholder
        self.concepts_tree.insert(
            values=(
                "No recent workflow",
                "Concept registry is workflow-scoped",
                "--",
                "--",
                "--"
            )
        )

    def _update_meta_learning_stats(self, felix_system):
        """Update meta-learning statistics."""
        text = "META-LEARNING BOOST SYSTEM\n"
        text += "=" * 80 + "\n\n"

        text += f"Status: {'Enabled' if self.meta_learning_enabled_var.get() else 'Disabled'}\n"
        text += f"Minimum samples for boost: {self.min_samples_var.get()}\n\n"

        text += "DESCRIPTION\n"
        text += "-" * 80 + "\n"
        text += "Meta-learning tracks which knowledge entries help specific task types.\n"
        text += "Historical usefulness data boosts retrieval relevance for similar tasks.\n\n"

        text += "HOW IT WORKS\n"
        text += "-" * 80 + "\n"
        text += "1. Workflow uses knowledge entries from the knowledge store\n"
        text += "2. User marks knowledge as 'helpful' or 'unhelpful'\n"
        text += "3. Usage patterns stored in knowledge_usage table\n"
        text += "4. When ≥" + str(self.min_samples_var.get()) + " samples exist, boost factor applied (0.5-1.0x)\n"
        text += "5. Future similar tasks retrieve boosted knowledge first\n\n"

        text += "BENEFITS\n"
        text += "-" * 80 + "\n"
        text += "✓ Knowledge that helped before ranks higher\n"
        text += "✓ Task-type specific (research vs analysis vs coding)\n"
        text += "✓ Continuous improvement through usage tracking\n"
        text += "✓ Learns from user feedback on helpfulness\n\n"

        text += "DATABASE\n"
        text += "-" * 80 + "\n"
        text += "Location: felix_knowledge.db\n"
        text += "Table: knowledge_usage\n"
        text += "Columns: entry_id, workflow_id, task_type, usefulness_score, timestamp\n"

        self.meta_learning_text.delete("1.0", "end")
        self.meta_learning_text.insert("1.0", text)

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh timer."""
        self.auto_refresh_enabled = self.auto_refresh_var.get()

        if self.auto_refresh_enabled:
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _start_auto_refresh(self):
        """Start auto-refresh timer."""
        if self.auto_refresh_enabled and self.thread_manager.is_active:
            self._refresh_statistics()
            self.refresh_timer = self.after(30000, self._start_auto_refresh)

    def _stop_auto_refresh(self):
        """Stop auto-refresh timer."""
        if self.refresh_timer:
            self.after_cancel(self.refresh_timer)
            self.refresh_timer = None

    def _toggle_feedback(self):
        """Toggle feedback integration."""
        enabled = self.feedback_enabled_var.get()
        logger.info(f"Feedback integration {'enabled' if enabled else 'disabled'}")

        # This would update felix_system configuration
        # For now, just log
        if self.main_app and hasattr(self.main_app, 'felix_system'):
            felix_system = self.main_app.felix_system
            if felix_system and hasattr(felix_system, 'config'):
                felix_system.config.enable_feedback = enabled

    def _toggle_meta_learning(self):
        """Toggle meta-learning boost."""
        enabled = self.meta_learning_enabled_var.get()
        logger.info(f"Meta-learning boost {'enabled' if enabled else 'disabled'}")

        # This would update knowledge store configuration
        # For now, just log

    def _update_min_samples_label(self, value):
        """Update minimum samples label."""
        self.min_samples_value_label.configure(text=f"{int(value)} samples")

    def _export_learning_data(self):
        """Export learning data to JSON file."""
        try:
            if not self.main_app or not hasattr(self.main_app, 'felix_system'):
                logger.warning("Cannot export: Felix system not started")
                return

            felix_system = self.main_app.felix_system
            task_memory = getattr(felix_system, 'task_memory', None)

            if not task_memory:
                logger.warning("Cannot export: Task memory not available")
                return

            rec_engine = RecommendationEngine(
                task_memory=task_memory,
                enable_auto_apply=False
            )

            # Get all statistics
            stats = rec_engine.get_unified_statistics(days=365)  # All data

            # Export to file
            filename = f"felix_learning_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            logger.info(f"Learning data exported to {filename}")

        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")

    def _reset_learning_confirm(self):
        """Confirm and reset all learning systems."""
        # This should show a confirmation dialog
        # For now, just log
        logger.warning("Reset learning systems requested (confirmation required)")

    # Feature management

    def _enable_features(self):
        """Enable tab features when system is ready."""
        self._features_enabled = True
        self.refresh_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.reset_button.configure(state="normal")
        logger.info("Learning tab features enabled")

    def _disable_features(self):
        """Disable tab features when system stops."""
        self._features_enabled = False
        self._stop_auto_refresh()
        self.auto_refresh_var.set(False)
        self.refresh_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        logger.info("Learning tab features disabled")

    def cleanup(self):
        """Clean up resources."""
        self._stop_auto_refresh()
        logger.info("Learning tab cleaned up")
