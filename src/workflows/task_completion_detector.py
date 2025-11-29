"""
Task Completion Detection System (Phase 2.3)

Distinguishes between "task solved" and "ran out of time" by analyzing
synthesis output against the original task description.
"""

import logging
import re
from enum import Enum
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CompletionStatus(Enum):
    """Task completion status."""
    COMPLETE = "complete"           # Task fully answered
    INCOMPLETE = "incomplete"       # Task partially answered or not addressed
    UNCLEAR = "unclear"             # Cannot determine if task is complete
    ERROR = "error"                # Analysis failed


class TaskCompletionDetector:
    """
    Analyzes synthesis output to determine if the original task was completed.

    Uses multiple heuristics:
    1. Question answering: Did synthesis address all questions?
    2. Request fulfillment: Were requested actions/outputs provided?
    3. Confidence alignment: Does synthesis confidence match output completeness?
    4. Content sufficiency: Is output detailed enough for the task?
    """

    def __init__(self):
        """Initialize task completion detector."""
        # Patterns indicating incomplete answers
        self.incomplete_patterns = [
            r'\b(I don\'t have|I cannot|I\'m unable|cannot determine)\b',
            r'\b(insufficient|incomplete|lacking|missing)\s+(information|data|details)\b',
            r'\bneed\s+(more|additional)\s+(information|data|context)\b',
            r'\b(unclear|ambiguous|vague)\s+(question|request|task)\b',
            r'\bwould need\s+to\b',
            r'\bshould\s+(first|probably|likely)\b',
            r'\brequires?\s+(further|more|additional)\s+(analysis|investigation|research)\b'
        ]

        # Patterns indicating complete answers
        self.complete_patterns = [
            r'(here is|here are|here\'s)\s+(the|a|an)',
            r'(I have|I\'ve)\s+(found|identified|determined|analyzed|completed)',
            r'\b(conclusion|summary|result|answer|solution):\s*',
            r'\b(in summary|to summarize|overall|in conclusion)\b',
            r'\b(successfully|completed|done|finished)\b'
        ]

        # Question words to detect if task is a question
        self.question_words = [
            'what', 'when', 'where', 'who', 'whom', 'whose', 'which',
            'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are',
            'do', 'does', 'did', 'will', 'shall'
        ]

    def detect_completion(
        self,
        task_description: str,
        synthesis_output: str,
        synthesis_confidence: float,
        agent_count: int
    ) -> Tuple[CompletionStatus, float, str]:
        """
        Detect if task was completed based on synthesis output.

        Args:
            task_description: Original task/question
            synthesis_output: Final synthesis result
            synthesis_confidence: Confidence score from synthesis
            agent_count: Number of agents that contributed

        Returns:
            Tuple of (status, completion_score, reason)
            - status: CompletionStatus enum
            - completion_score: 0.0-1.0 indicating how complete
            - reason: Human-readable explanation
        """
        try:
            # Clean inputs
            task = task_description.strip().lower()
            output = synthesis_output.strip().lower()

            if not task or not output:
                return CompletionStatus.UNCLEAR, 0.0, "Empty task or output"

            # Calculate multiple completion indicators
            scores = {}

            # 1. Question answering check
            scores['question_answered'] = self._check_question_answered(task, output)

            # 2. Request fulfillment check
            scores['request_fulfilled'] = self._check_request_fulfilled(task, output)

            # 3. Pattern-based analysis
            scores['pattern_match'] = self._check_patterns(output)

            # 4. Content sufficiency
            scores['content_sufficient'] = self._check_content_sufficiency(task, output)

            # 5. Confidence alignment
            scores['confidence_aligned'] = self._check_confidence_alignment(
                synthesis_confidence, output
            )

            # 6. Agent consensus
            scores['agent_consensus'] = min(agent_count / 3.0, 1.0)  # 3+ agents = full score

            # Calculate weighted completion score
            weights = {
                'question_answered': 0.30,
                'request_fulfilled': 0.25,
                'pattern_match': 0.15,
                'content_sufficient': 0.15,
                'confidence_aligned': 0.10,
                'agent_consensus': 0.05
            }

            completion_score = sum(
                scores[key] * weights[key] for key in weights
            )

            # Determine status based on score
            if completion_score >= 0.75:
                status = CompletionStatus.COMPLETE
                reason = f"Task appears complete (score: {completion_score:.2f})"
            elif completion_score >= 0.40:
                status = CompletionStatus.INCOMPLETE
                reason = f"Task partially complete (score: {completion_score:.2f})"
            else:
                status = CompletionStatus.UNCLEAR
                reason = f"Task completion unclear (score: {completion_score:.2f})"

            # Log detailed breakdown
            logger.info(f"  📊 Task Completion Analysis:")
            logger.info(f"    Overall: {status.value} ({completion_score:.2f})")
            logger.info(f"    Breakdown:")
            for key, score in scores.items():
                logger.info(f"      {key}: {score:.2f}")

            return status, completion_score, reason

        except Exception as e:
            logger.error(f"Task completion detection failed: {e}")
            return CompletionStatus.ERROR, 0.0, f"Analysis error: {str(e)}"

    def _check_question_answered(self, task: str, output: str) -> float:
        """
        Check if questions in task were answered in output.

        Returns: 0.0-1.0 score
        """
        # Extract questions from task
        questions = self._extract_questions(task)

        if not questions:
            # Not a question-based task, give neutral score
            return 0.7

        # Check if each question appears to be addressed in output
        answered_count = 0
        for question in questions:
            # Extract key terms from question (ignore question words)
            key_terms = [
                word for word in question.split()
                if len(word) > 3 and word not in self.question_words
            ]

            # Check if most key terms appear in output
            if key_terms:
                terms_found = sum(1 for term in key_terms if term in output)
                if terms_found / len(key_terms) >= 0.5:
                    answered_count += 1

        return answered_count / len(questions) if questions else 0.7

    def _check_request_fulfilled(self, task: str, output: str) -> float:
        """
        Check if explicit requests (create, list, explain, etc.) were fulfilled.

        Returns: 0.0-1.0 score
        """
        # Common request verbs
        request_verbs = [
            'create', 'make', 'generate', 'write', 'produce', 'build',
            'list', 'show', 'display', 'provide', 'give',
            'explain', 'describe', 'analyze', 'evaluate', 'assess',
            'find', 'search', 'locate', 'identify', 'discover',
            'read', 'view', 'examine', 'inspect', 'check'
        ]

        # Find requests in task
        requests_found = [verb for verb in request_verbs if f" {verb} " in f" {task} "]

        if not requests_found:
            return 0.7  # No explicit requests, give neutral score

        # Check if output contains results or completion indicators
        completion_indicators = [
            'here is', 'here are', "here's", 'i have', "i've",
            'result', 'output', 'created', 'generated', 'found',
            'located', 'identified', 'analyzed', 'listed'
        ]

        fulfilled = any(indicator in output for indicator in completion_indicators)

        return 0.9 if fulfilled else 0.3

    def _check_patterns(self, output: str) -> float:
        """
        Check output for complete/incomplete patterns.

        Returns: 0.0-1.0 score
        """
        incomplete_count = sum(
            1 for pattern in self.incomplete_patterns
            if re.search(pattern, output, re.IGNORECASE)
        )

        complete_count = sum(
            1 for pattern in self.complete_patterns
            if re.search(pattern, output, re.IGNORECASE)
        )

        if incomplete_count > complete_count:
            return 0.2
        elif complete_count > incomplete_count:
            return 0.9
        else:
            return 0.5

    def _check_content_sufficiency(self, task: str, output: str) -> float:
        """
        Check if output has sufficient content for the task complexity.

        Returns: 0.0-1.0 score
        """
        task_length = len(task.split())
        output_length = len(output.split())

        # Expected output/task ratio based on task type
        if any(word in task for word in ['summary', 'brief', 'short']):
            # Brief tasks: expect 2-5x output
            expected_ratio = 3.0
        elif any(word in task for word in ['detailed', 'comprehensive', 'complete', 'entire']):
            # Detailed tasks: expect 5-15x output
            expected_ratio = 10.0
        else:
            # Standard tasks: expect 3-8x output
            expected_ratio = 5.0

        actual_ratio = output_length / task_length if task_length > 0 else 0

        # Score based on how close to expected ratio
        if actual_ratio >= expected_ratio * 0.5:
            return min(actual_ratio / expected_ratio, 1.0)
        else:
            return actual_ratio / (expected_ratio * 0.5)

    def _check_confidence_alignment(self, confidence: float, output: str) -> float:
        """
        Check if synthesis confidence aligns with output completeness.

        Returns: 0.0-1.0 score
        """
        # High confidence should correlate with complete-sounding output
        has_incomplete_signals = any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in self.incomplete_patterns
        )

        if confidence >= 0.8 and not has_incomplete_signals:
            return 1.0  # High confidence, no incomplete signals = good
        elif confidence < 0.5 and has_incomplete_signals:
            return 1.0  # Low confidence, has incomplete signals = consistent
        elif confidence >= 0.8 and has_incomplete_signals:
            return 0.3  # High confidence but incomplete = misaligned
        else:
            return 0.6  # Medium confidence = neutral

    def _extract_questions(self, task: str) -> list[str]:
        """Extract questions from task description."""
        # Split on question marks
        potential_questions = [
            q.strip() + '?' for q in task.split('?') if q.strip()
        ]

        # Filter to actual questions (start with question word or verb)
        questions = []
        for q in potential_questions:
            first_word = q.split()[0].lower() if q.split() else ""
            if first_word in self.question_words:
                questions.append(q)

        # If no explicit questions, check if entire task is a question
        if not questions:
            first_word = task.split()[0].lower() if task.split() else ""
            if first_word in self.question_words:
                questions.append(task)

        return questions
