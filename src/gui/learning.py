"""
Learning Tab - Display and control Felix learning systems

This tab provides visualization and management of Felix's learning infrastructure:
- Pattern recommendations statistics
- Agent confidence calibration
- Threshold optimization
- Learning system controls
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
from typing import Optional
from datetime import datetime
from .utils import logger

class LearningFrame(ttk.Frame):
    """Frame for learning system visualization and control."""

    def __init__(self, parent, thread_manager, main_app, theme_manager):
        super().__init__(parent)
        self.thread_manager = thread_manager
        self.main_app = main_app
        self.theme_manager = theme_manager

        # Track refresh timer
        self.refresh_timer = None

        self._create_widgets()

    def _create_widgets(self):
        """Create learning tab widgets."""

        # Title and status
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = ttk.Label(title_frame, text="Felix Learning Systems",
                                font=("TkDefaultFont", 16, "bold"))
        title_label.pack(side=tk.LEFT)

        self.status_label = ttk.Label(title_frame, text="●",
                                      font=("TkDefaultFont", 16))
        self.status_label.pack(side=tk.RIGHT)

        # Control buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        self.refresh_button = ttk.Button(button_frame, text="🔄 Refresh Statistics",
                                         command=self._refresh_statistics)
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.auto_refresh_var = tk.BooleanVar(value=False)
        self.auto_refresh_check = ttk.Checkbutton(button_frame, text="Auto-refresh (30s)",
                                                  variable=self.auto_refresh_var,
                                                  command=self._toggle_auto_refresh)
        self.auto_refresh_check.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(button_frame, text="⚠ Reset All Learning",
                                       command=self._reset_learning_confirm)
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        # Create notebook for subsections
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Overview tab
        self.overview_frame = self._create_overview_tab()
        self.sub_notebook.add(self.overview_frame, text="Overview")

        # Pattern Learner tab
        self.patterns_frame = self._create_patterns_tab()
        self.sub_notebook.add(self.patterns_frame, text="Patterns")

        # Confidence Calibration tab
        self.calibration_frame = self._create_calibration_tab()
        self.sub_notebook.add(self.calibration_frame, text="Calibration")

        # Threshold Learning tab
        self.thresholds_frame = self._create_thresholds_tab()
        self.sub_notebook.add(self.thresholds_frame, text="Thresholds")

        # Initial refresh
        self.after(500, self._refresh_statistics)

    def _create_overview_tab(self):
        """Create overview statistics tab."""
        frame = ttk.Frame(self.sub_notebook)

        # Statistics text area
        self.overview_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20,
                                                       font=("Courier", 10))
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return frame

    def _create_patterns_tab(self):
        """Create pattern recommendations tab."""
        frame = ttk.Frame(self.sub_notebook)

        # Pattern statistics
        self.patterns_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20,
                                                       font=("Courier", 10))
        self.patterns_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return frame

    def _create_calibration_tab(self):
        """Create confidence calibration tab."""
        frame = ttk.Frame(self.sub_notebook)

        # Calibration statistics
        self.calibration_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20,
                                                          font=("Courier", 10))
        self.calibration_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return frame

    def _create_thresholds_tab(self):
        """Create threshold learning tab."""
        frame = ttk.Frame(self.sub_notebook)

        # Threshold statistics
        self.thresholds_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=20,
                                                         font=("Courier", 10))
        self.thresholds_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return frame

    def _refresh_statistics(self):
        """Refresh all learning statistics."""
        try:
            # Check if learning is enabled
            if not self.main_app.felix_system:
                self._update_overview("Felix system not started.\nStart the system from Dashboard tab.")
                self._update_status("⚫ Inactive", "gray")
                return

            if not self.main_app.felix_system.config.enable_learning:
                self._update_overview("Learning systems disabled in configuration.\nEnable in Settings tab.")
                self._update_status("⚫ Disabled", "orange")
                return

            # Get learning statistics
            self._update_status("🔄 Updating...", "blue")

            # Import here to avoid circular dependency
            from src.learning import RecommendationEngine

            # Create temporary recommendation engine to query stats
            if self.main_app.felix_system.task_memory:
                try:
                    rec_engine = RecommendationEngine(
                        task_memory=self.main_app.felix_system.task_memory,
                        enable_auto_apply=False  # Read-only for stats
                    )

                    # Get unified statistics
                    stats = rec_engine.get_unified_statistics(days=30)

                    # Update each tab
                    self._update_overview_stats(stats)
                    self._update_patterns_stats(rec_engine)
                    self._update_calibration_stats(rec_engine)
                    self._update_thresholds_stats(rec_engine)

                    # Update status
                    systems_active = stats.get('overall', {}).get('systems_with_data', 0)
                    if systems_active == 3:
                        self._update_status("● All Systems Active", "green")
                    elif systems_active > 0:
                        self._update_status(f"● {systems_active}/3 Systems Active", "yellow")
                    else:
                        self._update_status("⚫ No Data Yet", "gray")

                except Exception as e:
                    logger.error(f"Failed to get learning statistics: {e}")
                    self._update_overview(f"Error loading statistics:\n{str(e)}")
                    self._update_status("⚠ Error", "red")
            else:
                self._update_overview("Task memory not available.")
                self._update_status("⚫ Inactive", "gray")

        except Exception as e:
            logger.error(f"Failed to refresh learning statistics: {e}")
            self._update_overview(f"Error:\n{str(e)}")
            self._update_status("⚠ Error", "red")

    def _update_overview_stats(self, stats):
        """Update overview tab with unified statistics."""
        text = "FELIX LEARNING SYSTEMS - OVERVIEW\n"
        text += "=" * 60 + "\n\n"

        overall = stats.get('overall', {})
        text += f"Status: {'Active' if overall.get('learning_active') else 'Inactive'}\n"
        text += f"Systems with data: {overall.get('systems_with_data', 0)}/3\n"
        text += f"Total data points: {overall.get('total_data_points', 0):,}\n"
        text += f"Reporting period: Last {stats.get('days', 30)} days\n\n"

        # Pattern learner summary
        pattern_stats = stats.get('pattern_learner', {})
        text += "PATTERN LEARNER\n"
        text += "-" * 60 + "\n"
        total_recs = pattern_stats.get('total_recommendations', 0)
        text += f"  Total recommendations: {total_recs}\n"
        text += f"  Applied: {pattern_stats.get('applied_count', 0) or 0}\n"
        text += f"  Success count: {pattern_stats.get('success_count', 0) or 0}\n"
        if pattern_stats.get('success_rate') is not None and total_recs > 0:
            text += f"  Success rate: {pattern_stats['success_rate']:.1%}\n"
        elif total_recs == 0:
            text += f"  Status: No data yet (need ≥10 patterns)\n"
        text += "\n"

        # Confidence calibrator summary
        calibration_stats = stats.get('confidence_calibrator', {})
        text += "CONFIDENCE CALIBRATOR\n"
        text += "-" * 60 + "\n"
        total_cal_records = calibration_stats.get('total_records', 0)
        text += f"  Total records: {total_cal_records}\n"
        text += f"  Total samples: {calibration_stats.get('total_samples', 0)}\n"
        if calibration_stats.get('avg_calibration_error') is not None:
            text += f"  Avg calibration error: {calibration_stats['avg_calibration_error']:.3f}\n"

        if calibration_stats.get('most_overconfident'):
            oc = calibration_stats['most_overconfident']
            text += f"  Most overconfident: {oc['agent_type']}/{oc['task_complexity']} "
            text += f"(factor={oc['calibration_factor']:.3f})\n"

        if calibration_stats.get('most_underconfident'):
            uc = calibration_stats['most_underconfident']
            text += f"  Most underconfident: {uc['agent_type']}/{uc['task_complexity']} "
            text += f"(factor={uc['calibration_factor']:.3f})\n"

        if total_cal_records == 0:
            text += f"  Status: No data yet (need ≥10 workflows)\n"
        text += "\n"

        # Threshold learner summary
        threshold_stats = stats.get('threshold_learner', {})
        text += "THRESHOLD LEARNER\n"
        text += "-" * 60 + "\n"
        total_threshold_records = threshold_stats.get('total_records', 0)
        text += f"  Total records: {total_threshold_records}\n"
        text += f"  Total samples: {threshold_stats.get('total_samples', 0)}\n"
        if threshold_stats.get('avg_success_rate') is not None:
            text += f"  Avg success rate: {threshold_stats['avg_success_rate']:.1%}\n"
        if threshold_stats.get('avg_confidence') is not None:
            text += f"  Avg confidence: {threshold_stats['avg_confidence']:.1%}\n"
        if total_threshold_records == 0:
            text += f"  Status: No data yet (need ≥20 workflows)\n"

        text += "\n" + "=" * 60 + "\n"
        text += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        self._update_overview(text)

    def _update_patterns_stats(self, rec_engine):
        """Update pattern recommendations tab."""
        text = "PATTERN RECOMMENDATIONS\n"
        text += "=" * 60 + "\n\n"

        try:
            stats = rec_engine.pattern_learner.get_recommendation_statistics(days=30)

            total_recs = stats.get('total_recommendations', 0)
            applied = stats.get('applied_count', 0) or 0  # Handle None
            success = stats.get('success_count', 0) or 0  # Handle None

            text += f"Total recommendations: {total_recs}\n"
            text += f"Applied recommendations: {applied}\n"
            text += f"Successful workflows: {success}\n"

            # Only show success rate if we have data
            if stats.get('success_rate') is not None and total_recs > 0:
                text += f"\nSuccess rate: {stats['success_rate']:.1%}\n"
            elif total_recs == 0:
                text += "\n✓ No recommendations generated yet\n"
                text += "  (Requires ≥10 historical patterns in TaskMemory)\n"

            text += "\n" + "-" * 60 + "\n"
            text += "Pattern recommendations track workflow optimization opportunities\n"
            text += "based on historical task execution patterns.\n\n"
            text += "Auto-apply threshold: ≥95% success + ≥20 samples\n"
            text += "Recommend threshold: ≥80% success + ≥10 samples\n"

        except Exception as e:
            text += f"\nError loading pattern statistics: {e}\n"

        self._update_patterns(text)

    def _update_calibration_stats(self, rec_engine):
        """Update confidence calibration tab."""
        text = "AGENT CONFIDENCE CALIBRATION\n"
        text += "=" * 60 + "\n\n"

        try:
            # Get all calibration records
            records = rec_engine.confidence_calibrator.get_all_calibration_records()

            if records:
                text += f"Total calibration records: {len(records)}\n\n"

                # Group by agent type
                by_agent = {}
                for rec in records:
                    if rec.agent_type not in by_agent:
                        by_agent[rec.agent_type] = []
                    by_agent[rec.agent_type].append(rec)

                for agent_type, agent_records in by_agent.items():
                    text += f"\n{agent_type.upper()}\n"
                    text += "-" * 40 + "\n"

                    for rec in agent_records:
                        text += f"  Complexity: {rec.task_complexity}\n"
                        text += f"  Calibration factor: {rec.calibration_factor:.3f}\n"
                        text += f"  Avg predicted: {rec.avg_predicted_confidence:.3f}\n"
                        text += f"  Avg actual: {rec.avg_actual_success:.3f}\n"
                        text += f"  Calibration error: {rec.calibration_error:.3f}\n"
                        text += f"  Samples: {rec.sample_size}\n"

                        # Interpretation
                        if rec.calibration_factor > 1.1:
                            text += f"  → Underconfident (predictions too low)\n"
                        elif rec.calibration_factor < 0.9:
                            text += f"  → Overconfident (predictions too high)\n"
                        else:
                            text += f"  → Well calibrated\n"
                        text += "\n"
            else:
                text += "✓ No calibration data yet\n"
                text += "  (Requires ≥10 workflows with agent confidence tracking)\n\n"
                text += "Calibration data accumulates as workflows complete.\n"
                text += "Each workflow records predicted vs actual agent performance.\n"

            text += "\n" + "-" * 60 + "\n"
            text += "Calibration factors adjust agent confidence predictions:\n"
            text += "  Factor > 1.0: Agent underconfident, boost predictions\n"
            text += "  Factor < 1.0: Agent overconfident, reduce predictions\n"
            text += "  Factor ≈ 1.0: Agent well-calibrated\n"

        except Exception as e:
            text += f"\nError loading calibration statistics: {e}\n"

        self._update_calibration(text)

    def _update_thresholds_stats(self, rec_engine):
        """Update threshold learning tab."""
        text = "THRESHOLD LEARNING\n"
        text += "=" * 60 + "\n\n"

        try:
            # Get all threshold records
            records = rec_engine.threshold_learner.get_all_threshold_records()

            if records:
                text += f"Total threshold records: {len(records)}\n\n"

                # Group by task type
                by_task = {}
                for rec in records:
                    if rec.task_type not in by_task:
                        by_task[rec.task_type] = []
                    by_task[rec.task_type].append(rec)

                for task_type, task_records in by_task.items():
                    text += f"\n{task_type.upper()}\n"
                    text += "-" * 40 + "\n"

                    for rec in task_records:
                        text += f"  {rec.threshold_name}:\n"
                        text += f"    Learned value: {rec.learned_value:.3f}\n"
                        text += f"    Success rate: {rec.success_rate:.1%}\n"
                        text += f"    Confidence: {rec.confidence:.1%}\n"
                        text += f"    Samples: {rec.sample_size}\n"

                        # Show comparison with standard
                        from src.learning.threshold_learner import STANDARD_THRESHOLDS
                        standard = STANDARD_THRESHOLDS.get(rec.threshold_name)
                        if standard:
                            diff = rec.learned_value - standard
                            text += f"    Standard: {standard:.3f}"
                            if abs(diff) > 0.05:
                                text += f" (learned {'+' if diff > 0 else ''}{diff:.3f})\n"
                            else:
                                text += f" (similar)\n"
                        text += "\n"
            else:
                text += "✓ No threshold learning data yet\n"
                text += "  (Requires ≥20 workflows per task type)\n\n"
                text += "Threshold data accumulates as workflows complete.\n"
                text += "Each workflow records threshold performance for optimization.\n"

            text += "\n" + "-" * 60 + "\n"
            text += "Learned thresholds optimize per task type:\n"
            text += "  - confidence_threshold (synthesis quality gate)\n"
            text += "  - team_expansion_threshold (spawn more agents)\n"
            text += "  - volatility_threshold (spawn critic agent)\n"
            text += "  - web_search_threshold (trigger external research)\n"

        except Exception as e:
            text += f"\nError loading threshold statistics: {e}\n"

        self._update_thresholds(text)

    def _update_overview(self, text):
        """Update overview text area."""
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(1.0, text)
        self.overview_text.config(state=tk.DISABLED)

    def _update_patterns(self, text):
        """Update patterns text area."""
        self.patterns_text.config(state=tk.NORMAL)
        self.patterns_text.delete(1.0, tk.END)
        self.patterns_text.insert(1.0, text)
        self.patterns_text.config(state=tk.DISABLED)

    def _update_calibration(self, text):
        """Update calibration text area."""
        self.calibration_text.config(state=tk.NORMAL)
        self.calibration_text.delete(1.0, tk.END)
        self.calibration_text.insert(1.0, text)
        self.calibration_text.config(state=tk.DISABLED)

    def _update_thresholds(self, text):
        """Update thresholds text area."""
        self.thresholds_text.config(state=tk.NORMAL)
        self.thresholds_text.delete(1.0, tk.END)
        self.thresholds_text.insert(1.0, text)
        self.thresholds_text.config(state=tk.DISABLED)

    def _update_status(self, text, color):
        """Update status label."""
        self.status_label.config(text=text, foreground=color)

    def _toggle_auto_refresh(self):
        """Toggle auto-refresh timer."""
        if self.auto_refresh_var.get():
            self._start_auto_refresh()
        else:
            self._stop_auto_refresh()

    def _start_auto_refresh(self):
        """Start auto-refresh timer."""
        self._refresh_statistics()
        self.refresh_timer = self.after(30000, self._start_auto_refresh)  # 30 seconds

    def _stop_auto_refresh(self):
        """Stop auto-refresh timer."""
        if self.refresh_timer:
            self.after_cancel(self.refresh_timer)
            self.refresh_timer = None

    def _reset_learning_confirm(self):
        """Confirm and reset all learning systems."""
        response = messagebox.askyesno(
            "Confirm Reset",
            "This will permanently delete all learning data:\n\n"
            "• Pattern recommendations\n"
            "• Confidence calibration\n"
            "• Threshold learning\n\n"
            "Are you sure you want to continue?"
        )

        if response:
            self._reset_learning()

    def _reset_learning(self):
        """Reset all learning systems."""
        try:
            if not self.main_app.felix_system or not self.main_app.felix_system.task_memory:
                messagebox.showerror("Error", "Felix system not started")
                return

            from src.learning import RecommendationEngine

            rec_engine = RecommendationEngine(
                task_memory=self.main_app.felix_system.task_memory,
                enable_auto_apply=False
            )

            # Reset all learning data
            results = rec_engine.reset_all_learning()

            messagebox.showinfo(
                "Learning Reset Complete",
                f"Learning systems have been reset:\n\n"
                f"• Calibration records deleted: {results['calibration_records']}\n"
                f"• Threshold records deleted: {results['threshold_records']}\n\n"
                f"Learning will restart from scratch with new workflow data."
            )

            # Refresh display
            self._refresh_statistics()

        except Exception as e:
            logger.error(f"Failed to reset learning systems: {e}")
            messagebox.showerror("Error", f"Failed to reset learning systems:\n{str(e)}")

    def apply_theme(self):
        """Apply current theme to learning frame."""
        # Text areas will automatically update with theme
        pass
