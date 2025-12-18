"""
RecommendationEngine - Unified learning orchestrator combining all learning systems.

This module provides the main interface for Felix's learning capabilities:
- PatternLearner: Historical workflow patterns
- ConfidenceCalibrator: Agent confidence adjustment
- ThresholdLearner: Optimal threshold selection

Implements Before+After (B+A) logic:
- BEFORE workflow: Generate recommendations, optionally auto-apply
- AFTER workflow: Record outcomes, update learning systems
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .pattern_learner import PatternLearner, WorkflowRecommendation
from .confidence_calibrator import ConfidenceCalibrator
from .threshold_learner import ThresholdLearner
from src.memory.task_memory import TaskComplexity

# Optional FeedbackManager for unified statistics
try:
    from src.feedback.feedback_manager import FeedbackManager
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRecommendation:
    """Combined recommendation from all learning systems."""
    workflow_recommendation: Optional[WorkflowRecommendation]
    suggested_thresholds: Dict[str, float]
    calibration_available: Dict[str, bool]
    should_auto_apply: bool
    recommendation_summary: str
    timestamp: float


class RecommendationEngine:
    """
    Unified learning orchestrator for Felix.

    Combines PatternLearner, ConfidenceCalibrator, and ThresholdLearner
    to provide comprehensive workflow optimization recommendations.
    """

    def __init__(self,
                 task_memory,
                 db_path=None,
                 enable_auto_apply: bool = True,
                 min_samples_patterns: int = 10,
                 min_samples_calibration: int = 10,
                 min_samples_thresholds: int = 20,
                 performance_tracker=None):
        """
        Initialize RecommendationEngine.

        Args:
            task_memory: TaskMemory instance for pattern queries
            db_path: Path to felix_task_memory.db (default: auto-detect)
            enable_auto_apply: Allow high-confidence auto-apply (default: True)
            min_samples_patterns: Minimum samples for pattern recommendations
            min_samples_calibration: Minimum samples for calibration
            min_samples_thresholds: Minimum samples for threshold learning
            performance_tracker: Optional AgentPerformanceTracker for phase analysis (Issue #56.10)
        """
        self.task_memory = task_memory
        self.enable_auto_apply = enable_auto_apply
        self.performance_tracker = performance_tracker  # Issue #56.10: Adaptive metrics

        # Initialize learning components
        self.pattern_learner = PatternLearner(
            task_memory=task_memory,
            db_path=db_path,
            min_samples=min_samples_patterns
        )

        self.confidence_calibrator = ConfidenceCalibrator(
            db_path=db_path,
            min_samples=min_samples_calibration
        )

        self.threshold_learner = ThresholdLearner(
            db_path=db_path,
            min_samples=min_samples_thresholds
        )

        logger.info(f"RecommendationEngine initialized (auto_apply={enable_auto_apply}, performance_tracker={'enabled' if performance_tracker else 'disabled'})")

    def _get_phase_transition_insights(self) -> Dict[str, Any]:
        """
        Get phase transition insights from performance tracker (Issue #56.10).

        Uses historical phase transition data to provide recommendations
        about optimal phase timing and agent allocation.

        Returns:
            Dictionary with phase transition analysis or empty dict if unavailable
        """
        if not self.performance_tracker:
            return {}

        try:
            analysis = self.performance_tracker.get_phase_transition_analysis()
            if analysis:
                logger.debug(f"Phase transition insights available: {len(analysis)} metrics")
                return analysis
        except Exception as e:
            logger.debug(f"Could not get phase transition insights: {e}")

        return {}

    def get_pre_workflow_recommendations(self,
                                          task_description: str,
                                          task_type: str = "general",
                                          task_complexity: TaskComplexity = TaskComplexity.MODERATE) -> UnifiedRecommendation:
        """
        Get comprehensive recommendations BEFORE starting workflow.

        Queries all learning systems to provide unified recommendations
        for workflow configuration.

        Args:
            task_description: Description of the task
            task_type: Type of task (research, coding, analysis, etc.)
            task_complexity: Complexity level (TaskComplexity enum)

        Returns:
            UnifiedRecommendation with all learning system outputs
        """
        logger.info(f"Generating pre-workflow recommendations for {task_type} task...")

        # 1. Get pattern-based recommendations
        workflow_rec = self.pattern_learner.get_workflow_recommendations(
            task_description=task_description,
            task_type=task_type,
            task_complexity=task_complexity
        )

        # 2. Get learned thresholds for this task type
        suggested_thresholds = {}
        for threshold_name in ['confidence_threshold', 'team_expansion_threshold',
                                'volatility_threshold', 'web_search_threshold']:
            suggested_thresholds[threshold_name] = self.threshold_learner.get_threshold_with_fallback(
                task_type=task_type,
                threshold_name=threshold_name
            )

        # 3. Check calibration availability for agent types
        calibration_available = {}
        for agent_type in ['research', 'analysis', 'critic', 'synthesis']:
            factor = self.confidence_calibrator.get_calibration_factor(
                agent_type=agent_type,
                task_complexity=task_complexity
            )
            calibration_available[agent_type] = (abs(factor - 1.0) > 0.01)

        # 4. Determine if auto-apply should happen
        should_auto_apply = True
        if self.enable_auto_apply and workflow_rec:
            should_auto_apply = workflow_rec.should_auto_apply

        # 4.5 Get phase transition insights from performance tracker (Issue #56.10)
        phase_insights = self._get_phase_transition_insights()

        # 5. Generate summary
        summary_parts = []

        if workflow_rec:
            summary_parts.append(
                f"Pattern recommendation: {workflow_rec.confidence_level} confidence "
                f"({workflow_rec.success_probability:.1%} success over {workflow_rec.patterns_used} patterns)"
            )
            if workflow_rec.should_auto_apply and self.enable_auto_apply:
                summary_parts.append("✓ Auto-applying high-confidence pattern")
            elif workflow_rec.confidence_level != 'low':
                summary_parts.append("→ Pattern suggestion available")

        threshold_learned_count = sum(1 for name, value in suggested_thresholds.items()
                                       if self.threshold_learner.get_learned_threshold(task_type, name) is not None)
        if threshold_learned_count > 0:
            summary_parts.append(f"✓ {threshold_learned_count} learned thresholds for {task_type}")

        calibrated_agents = sum(1 for available in calibration_available.values() if available)
        if calibrated_agents > 0:
            summary_parts.append(f"✓ {calibrated_agents} agents calibrated for {task_complexity} tasks")

        # Add phase transition insights (Issue #56.10)
        if phase_insights:
            summary_parts.append(f"✓ Phase transition data available")

        if not summary_parts:
            summary_parts.append("No learned patterns available - using standard configuration")

        recommendation_summary = " | ".join(summary_parts)

        # Create unified recommendation
        unified = UnifiedRecommendation(
            workflow_recommendation=workflow_rec,
            suggested_thresholds=suggested_thresholds,
            calibration_available=calibration_available,
            should_auto_apply=should_auto_apply,
            recommendation_summary=recommendation_summary,
            timestamp=time.time()
        )

        logger.info(f"Pre-workflow recommendations: {recommendation_summary}")

        return unified

    def apply_high_confidence_recommendations(self,
                                              unified_rec: UnifiedRecommendation,
                                              agent_factory,
                                              config) -> Dict[str, Any]:
        """
        Apply high-confidence recommendations to workflow configuration.

        Only applies if unified_rec.should_auto_apply is True.

        Args:
            unified_rec: UnifiedRecommendation from get_pre_workflow_recommendations()
            agent_factory: AgentFactory instance to modify
            config: FelixConfig instance to modify

        Returns:
            Dictionary of applied changes
        """
        if not unified_rec.should_auto_apply:
            logger.info("Skipping auto-apply (confidence too low)")
            return {'auto_applied': False, 'reason': 'confidence_too_low'}

        applied_changes = {'auto_applied': True, 'changes': []}

        try:
            # 1. Apply pattern recommendations to spawning
            if unified_rec.workflow_recommendation:
                pattern_changes = self.pattern_learner.apply_to_spawning(
                    recommendation=unified_rec.workflow_recommendation,
                    agent_factory=agent_factory
                )
                if pattern_changes:
                    applied_changes['pattern_changes'] = pattern_changes
                    applied_changes['changes'].append('pattern_preferences')

            # 2. Apply learned thresholds to config
            # NOTE: This requires extending FelixConfig to support dynamic threshold updates
            # For now, we log the recommendations
            threshold_changes = {}
            for threshold_name, learned_value in unified_rec.suggested_thresholds.items():
                # Check if this is actually a learned value (not just standard fallback)
                if self.threshold_learner.get_learned_threshold(
                    task_type=unified_rec.workflow_recommendation.task_type if unified_rec.workflow_recommendation else "general",
                    threshold_name=threshold_name
                ) is not None:
                    threshold_changes[threshold_name] = learned_value
                    logger.info(f"Recommended threshold: {threshold_name}={learned_value:.3f}")

            if threshold_changes:
                applied_changes['threshold_recommendations'] = threshold_changes
                applied_changes['changes'].append('threshold_recommendations')

            # 3. Note calibration availability for agents
            calibrated_agents = [agent_type for agent_type, available
                                 in unified_rec.calibration_available.items() if available]
            if calibrated_agents:
                applied_changes['calibrated_agents'] = calibrated_agents
                applied_changes['changes'].append('agent_calibration')

            logger.info(f"Applied {len(applied_changes['changes'])} recommendation types")

            return applied_changes

        except Exception as e:
            logger.error(f"Failed to apply high-confidence recommendations: {e}")
            return {'auto_applied': False, 'error': str(e)}

    def record_workflow_outcome(self,
                                 workflow_id: str,
                                 task_type: str,
                                 task_complexity: TaskComplexity,
                                 agents_used: List[Dict[str, Any]],
                                 workflow_success: bool,
                                 workflow_duration: float,
                                 final_confidence: float,
                                 thresholds_used: Dict[str, float],
                                 recommendation_id: Optional[str] = None) -> bool:
        """
        Record workflow outcome AFTER completion to update all learning systems.

        This is the critical feedback loop that enables learning.

        Args:
            workflow_id: ID of completed workflow
            task_type: Type of task
            task_complexity: Complexity level (TaskComplexity enum)
            agents_used: List of agent info dicts with {type, predicted_confidence}
            workflow_success: Whether workflow succeeded
            workflow_duration: Duration in seconds
            final_confidence: Final synthesis confidence
            thresholds_used: Dict of threshold names and values used
            recommendation_id: Optional recommendation ID if recommendation was used

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            logger.info(f"Recording workflow outcome for {workflow_id[:8]}... (success={workflow_success})")

            # 1. Record recommendation outcome (if recommendation was used)
            if recommendation_id:
                self.pattern_learner.record_recommendation_outcome(
                    recommendation_id=recommendation_id,
                    workflow_id=workflow_id,
                    was_applied=True,
                    workflow_success=workflow_success,
                    actual_duration=workflow_duration
                )

            # 2. Record agent confidence predictions for calibration
            for agent_info in agents_used:
                agent_type = agent_info.get('type', 'unknown')
                predicted_confidence = agent_info.get('predicted_confidence')

                if predicted_confidence is not None:
                    self.confidence_calibrator.record_agent_prediction(
                        agent_type=agent_type,
                        task_complexity=task_complexity,
                        predicted_confidence=predicted_confidence,
                        actual_success=workflow_success
                    )

            # 3. Record threshold performance
            for threshold_name, threshold_value in thresholds_used.items():
                self.threshold_learner.record_threshold_performance(
                    task_type=task_type,
                    threshold_name=threshold_name,
                    threshold_value=threshold_value,
                    workflow_success=workflow_success,
                    workflow_id=workflow_id
                )

            logger.info(f"✓ Recorded outcome: {len(agents_used)} agent predictions, "
                       f"{len(thresholds_used)} thresholds")

            return True

        except Exception as e:
            logger.error(f"Failed to record workflow outcome: {e}")
            return False

    def get_unified_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all learning systems.

        Args:
            days: Number of days to include (default: 30)

        Returns:
            Dictionary with statistics from all learning components
        """
        try:
            stats = {
                'timestamp': time.time(),
                'days': days,
                'pattern_learner': {},
                'confidence_calibrator': {},
                'threshold_learner': {},
                'feedback_integrator': {}
            }

            # Pattern learner statistics
            pattern_stats = self.pattern_learner.get_recommendation_statistics(days=days)
            stats['pattern_learner'] = pattern_stats

            # Confidence calibrator statistics
            calibration_stats = self.confidence_calibrator.get_calibration_statistics(days=days)
            stats['confidence_calibrator'] = calibration_stats

            # Threshold learner statistics
            threshold_stats = self.threshold_learner.get_threshold_statistics(days=days)
            stats['threshold_learner'] = threshold_stats

            # Feedback integrator statistics (if available)
            feedback_stats = {'total_ratings': 0, 'available': False}
            if FEEDBACK_AVAILABLE:
                try:
                    feedback_manager = FeedbackManager()
                    global_feedback = feedback_manager.get_global_feedback_stats()
                    feedback_stats = {
                        'available': True,
                        'total_ratings': global_feedback.get('total_ratings', 0),
                        'positive_ratings': global_feedback.get('positive_ratings', 0),
                        'avg_accuracy': global_feedback.get('avg_accuracy'),
                        'avg_relevance': global_feedback.get('avg_relevance'),
                        'knowledge_feedback_counts': global_feedback.get('knowledge_counts', {})
                    }
                except Exception as e:
                    logger.debug(f"Could not get feedback stats: {e}")
            stats['feedback_integrator'] = feedback_stats

            # Calculate overall learning health (now includes all 4 systems)
            total_data_points = (
                pattern_stats.get('total_recommendations', 0) +
                calibration_stats.get('total_samples', 0) +
                threshold_stats.get('total_samples', 0) +
                feedback_stats.get('total_ratings', 0)
            )

            stats['overall'] = {
                'total_data_points': total_data_points,
                'learning_active': total_data_points > 0,
                'systems_with_data': sum([
                    1 if pattern_stats.get('total_recommendations', 0) > 0 else 0,
                    1 if calibration_stats.get('total_samples', 0) > 0 else 0,
                    1 if threshold_stats.get('total_samples', 0) > 0 else 0,
                    1 if feedback_stats.get('total_ratings', 0) > 0 else 0
                ])
            }

            logger.debug(f"Unified statistics: {stats['overall']['total_data_points']} data points, "
                        f"{stats['overall']['systems_with_data']}/4 systems active")

            return stats

        except Exception as e:
            logger.error(f"Failed to get unified statistics: {e}")
            return {}

    def calibrate_agent_confidence(self,
                                    agent_type: str,
                                    task_complexity: str,
                                    raw_confidence: float) -> float:
        """
        Convenience method to calibrate agent confidence.

        Args:
            agent_type: Type of agent
            task_complexity: Task complexity
            raw_confidence: Raw confidence score

        Returns:
            Calibrated confidence score
        """
        return self.confidence_calibrator.calibrate_confidence(
            agent_type=agent_type,
            task_complexity=task_complexity,
            raw_confidence=raw_confidence
        )

    def get_task_type_recommendations(self, task_type: str) -> Dict[str, Any]:
        """
        Get all available recommendations for a specific task type.

        Useful for displaying learned patterns in GUI.

        Args:
            task_type: Type of task

        Returns:
            Dictionary with all recommendations for this task type
        """
        try:
            recommendations = {
                'task_type': task_type,
                'thresholds': {},
                'threshold_comparisons': {},
                'calibration_records': []
            }

            # Get learned thresholds
            for threshold_name in ['confidence_threshold', 'team_expansion_threshold',
                                    'volatility_threshold', 'web_search_threshold']:
                learned = self.threshold_learner.get_learned_threshold(task_type, threshold_name)
                standard = self.threshold_learner.get_threshold_with_fallback(task_type, threshold_name)

                recommendations['thresholds'][threshold_name] = {
                    'learned': learned,
                    'standard': standard,
                    'using_learned': learned is not None
                }

            # Get threshold comparisons
            recommendations['threshold_comparisons'] = self.threshold_learner.compare_with_standard(task_type)

            # Get calibration records
            calibration_records = self.confidence_calibrator.get_all_calibration_records()
            recommendations['calibration_records'] = [
                {
                    'agent_type': rec.agent_type,
                    'task_complexity': rec.task_complexity,
                    'calibration_factor': rec.calibration_factor,
                    'sample_size': rec.sample_size
                }
                for rec in calibration_records
            ]

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get task type recommendations: {e}")
            return {}

    def reset_all_learning(self) -> Dict[str, int]:
        """
        Reset all learning systems (useful for debugging or retraining).

        Returns:
            Dictionary with counts of deleted records per system
        """
        logger.warning("Resetting ALL learning systems...")

        results = {
            'calibration_records': self.confidence_calibrator.reset_calibration(),
            'threshold_records': self.threshold_learner.reset_thresholds()
        }

        logger.warning(f"Reset complete: {results}")
        return results
