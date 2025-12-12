"""
Feedback Integrator - Bridges feedback collection with knowledge and learning systems.

This module handles the propagation of user feedback to:
- Knowledge store success rates
- Agent performance metrics
- Learning system updates
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, List

from .feedback_manager import FeedbackManager, FeedbackType, ReasonCategory

logger = logging.getLogger(__name__)


class FeedbackIntegrator:
    """
    Integrates user feedback with Felix's knowledge and learning systems.

    Responsibilities:
    - Update knowledge success rates based on feedback
    - Propagate workflow ratings to knowledge entries
    - Track which knowledge entries contributed to workflows
    - Update agent performance based on feedback
    """

    def __init__(self,
                 feedback_manager: FeedbackManager,
                 knowledge_db_path: Path = None,
                 workflow_history_db_path: Path = None):
        """
        Initialize feedback integrator.

        Args:
            feedback_manager: FeedbackManager instance
            knowledge_db_path: Path to felix_knowledge.db
            workflow_history_db_path: Path to felix_workflow_history.db
        """
        self.feedback_manager = feedback_manager

        if knowledge_db_path is None:
            knowledge_db_path = Path("felix_knowledge.db")
        if workflow_history_db_path is None:
            workflow_history_db_path = Path("felix_workflow_history.db")

        self.knowledge_db_path = knowledge_db_path
        self.workflow_history_db_path = workflow_history_db_path

    def submit_workflow_rating_with_propagation(self,
                                                 workflow_id: str,
                                                 positive: bool,
                                                 knowledge_ids_used: List[str] = None) -> None:
        """
        Submit workflow rating and propagate to knowledge entries.

        Args:
            workflow_id: Workflow ID
            positive: True for positive rating
            knowledge_ids_used: List of knowledge entry IDs used in this workflow
        """
        # Submit rating
        rating = self.feedback_manager.submit_workflow_rating(workflow_id, positive)
        logger.info(f"Workflow rating submitted: {workflow_id} = {'positive' if positive else 'negative'}")

        # Propagate to knowledge entries
        if knowledge_ids_used:
            self._update_knowledge_success_rates(knowledge_ids_used, positive)

    def submit_knowledge_feedback_with_update(self,
                                              knowledge_id: str,
                                              feedback_type: FeedbackType,
                                              reason_category: Optional[ReasonCategory] = None,
                                              reason_detail: str = None,
                                              correction_suggestion: str = None) -> None:
        """
        Submit knowledge feedback and update success rate.

        Args:
            knowledge_id: Knowledge entry ID
            feedback_type: Correct, incorrect, or unsure
            reason_category: Category if incorrect
            reason_detail: Detailed reason
            correction_suggestion: Suggested correction
        """
        # Submit feedback
        feedback = self.feedback_manager.submit_knowledge_feedback(
            knowledge_id, feedback_type, reason_category,
            reason_detail, correction_suggestion
        )
        logger.info(f"Knowledge feedback submitted: {knowledge_id} = {feedback_type.value}")

        # Update success rate immediately
        is_positive = (feedback_type == FeedbackType.CORRECT)
        self._update_single_knowledge_success_rate(knowledge_id, is_positive)

    def _update_knowledge_success_rates(self, knowledge_ids: List[str], positive: bool):
        """
        Update success rates for multiple knowledge entries.

        Args:
            knowledge_ids: List of knowledge IDs
            positive: Whether feedback was positive
        """
        for knowledge_id in knowledge_ids:
            self._update_single_knowledge_success_rate(knowledge_id, positive)

    def _update_single_knowledge_success_rate(self, knowledge_id: str, positive: bool):
        """
        Update success rate for a single knowledge entry.

        Uses incremental update formula to avoid needing full history:
        new_rate = (old_rate * old_count + new_result) / (old_count + 1)

        Args:
            knowledge_id: Knowledge entry ID
            positive: Whether this feedback was positive (1) or negative (0)
        """
        try:
            conn = sqlite3.connect(self.knowledge_db_path)
            conn.row_factory = sqlite3.Row

            # Get current success rate and access count (as proxy for feedback count)
            cursor = conn.execute("""
                SELECT success_rate, access_count
                FROM knowledge_entries
                WHERE knowledge_id = ?
            """, (knowledge_id,))

            row = cursor.fetchone()
            if not row:
                logger.warning(f"Knowledge entry {knowledge_id} not found")
                conn.close()
                return

            old_rate = row['success_rate'] if row['success_rate'] is not None else 1.0
            old_count = row['access_count'] if row['access_count'] is not None else 0

            # Incremental update
            new_result = 1.0 if positive else 0.0
            new_count = old_count + 1
            new_rate = (old_rate * old_count + new_result) / new_count

            # Update database
            conn.execute("""
                UPDATE knowledge_entries
                SET success_rate = ?,
                    access_count = ?
                WHERE knowledge_id = ?
            """, (new_rate, new_count, knowledge_id))

            conn.commit()
            conn.close()

            logger.info(f"Updated knowledge {knowledge_id[:8]}... success rate: "
                       f"{old_rate:.2f} -> {new_rate:.2f} (n={new_count})")

        except sqlite3.Error as e:
            logger.error(f"Failed to update knowledge success rate: {e}")

    def get_workflow_knowledge_entries(self, workflow_id: str) -> List[str]:
        """
        Get list of knowledge entry IDs used in a workflow.

        This queries the workflow_history to find which knowledge entries
        were retrieved during the workflow execution.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of knowledge entry IDs
        """
        try:
            conn = sqlite3.connect(self.workflow_history_db_path)
            conn.row_factory = sqlite3.Row

            # Try to get knowledge entries from workflow metadata
            cursor = conn.execute("""
                SELECT knowledge_entries
                FROM workflow_outputs
                WHERE workflow_id = ?
            """, (workflow_id,))

            row = cursor.fetchone()
            conn.close()

            if row and row['knowledge_entries']:
                # Parse JSON array of knowledge IDs
                import json
                return json.loads(row['knowledge_entries'])

            return []

        except sqlite3.Error as e:
            logger.error(f"Failed to get workflow knowledge entries: {e}")
            return []

    def get_knowledge_feedback_summary(self, knowledge_id: str) -> dict:
        """
        Get feedback summary for a knowledge entry.

        Args:
            knowledge_id: Knowledge entry ID

        Returns:
            Dictionary with feedback counts and success rate
        """
        feedbacks = self.feedback_manager.get_knowledge_feedback(knowledge_id)

        correct_count = sum(1 for f in feedbacks if f.feedback_type == 'correct')
        incorrect_count = sum(1 for f in feedbacks if f.feedback_type == 'incorrect')
        unsure_count = sum(1 for f in feedbacks if f.feedback_type == 'unsure')
        total = len(feedbacks)

        if total > 0:
            success_rate = correct_count / total
        else:
            success_rate = None

        return {
            'knowledge_id': knowledge_id,
            'total_feedback': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'unsure': unsure_count,
            'success_rate': success_rate
        }

    def apply_feedback_to_knowledge_batch(self, workflow_id: str, positive: bool) -> int:
        """
        Apply workflow feedback to all knowledge entries used in the workflow.

        Args:
            workflow_id: Workflow ID
            positive: Whether workflow was rated positively

        Returns:
            Number of knowledge entries updated
        """
        knowledge_ids = self.get_workflow_knowledge_entries(workflow_id)

        if not knowledge_ids:
            logger.info(f"No knowledge entries found for workflow {workflow_id}")
            return 0

        self._update_knowledge_success_rates(knowledge_ids, positive)
        logger.info(f"Applied feedback to {len(knowledge_ids)} knowledge entries")

        return len(knowledge_ids)
