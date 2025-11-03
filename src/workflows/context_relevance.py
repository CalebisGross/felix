"""
Context Relevance Evaluator for Felix workflows.

Distinguishes between factual accuracy and contextual relevance, ensuring agents
focus on information that actually helps solve the current task rather than
providing accurate but irrelevant facts.

This addresses the workflow recommendation: "Contextual Temporal Flexibility Layer -
Allow agents to interpret data relative to task context."
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RelevanceScore:
    """Score and explanation for context relevance."""
    score: float  # 0.0 (irrelevant) to 1.0 (highly relevant)
    reason: str
    keywords_matched: List[str]
    category: str  # 'highly_relevant', 'somewhat_relevant', 'irrelevant'


class ContextRelevanceEvaluator:
    """
    Evaluates relevance of facts and knowledge to the current task context.

    Prevents agents from providing factually accurate but contextually irrelevant
    information (e.g., reporting Eden, NC time when asked about system improvements).
    """

    def __init__(self):
        """Initialize context relevance evaluator."""
        # Common irrelevant patterns
        self.irrelevant_patterns = {
            'time_queries': ['time', 'clock', 'hour', 'minute', 'timezone', 'dst'],
            'location_queries': ['location', 'place', 'city', 'country', 'address'],
            'weather': ['weather', 'temperature', 'forecast', 'rain', 'sunny']
        }

    def evaluate_relevance(self, fact: str, task_context: str) -> RelevanceScore:
        """
        Evaluate how relevant a fact is to the given task context.

        Args:
            fact: The factual information to evaluate
            task_context: The current task description or context

        Returns:
            RelevanceScore with score (0.0-1.0) and explanation
        """
        # Extract keywords from task and fact
        task_keywords = self._extract_keywords(task_context.lower())
        fact_keywords = self._extract_keywords(fact.lower())

        # Check for keyword overlap
        matched_keywords = task_keywords & fact_keywords
        overlap_ratio = len(matched_keywords) / max(len(task_keywords), 1)

        # Check if fact matches irrelevant patterns
        irrelevance_penalty = self._check_irrelevant_patterns(fact.lower(), task_context.lower())

        # Calculate base relevance score
        base_score = overlap_ratio * 0.7 + (1.0 - irrelevance_penalty) * 0.3

        # Categorize
        if base_score >= 0.7:
            category = 'highly_relevant'
            reason = f"Strong keyword overlap ({len(matched_keywords)} matches)"
        elif base_score >= 0.4:
            category = 'somewhat_relevant'
            reason = f"Moderate relevance ({len(matched_keywords)} keyword matches)"
        else:
            category = 'irrelevant'
            if irrelevance_penalty > 0.5:
                reason = "Appears to be off-topic factual information"
            else:
                reason = "Minimal connection to task objectives"

        return RelevanceScore(
            score=base_score,
            reason=reason,
            keywords_matched=list(matched_keywords),
            category=category
        )

    def detect_context_shift(self, previous_messages: List[Dict[str, Any]],
                            current_query: str) -> bool:
        """
        Detect if the conversation context has shifted to a new topic.

        Args:
            previous_messages: List of previous messages in conversation
            current_query: Current query/task

        Returns:
            True if context has shifted significantly
        """
        if not previous_messages:
            return False

        # Extract keywords from recent messages
        recent_keywords = set()
        for msg in previous_messages[-3:]:  # Last 3 messages
            content = msg.get('content', '')
            if isinstance(content, dict):
                content = str(content.get('result', content))
            recent_keywords.update(self._extract_keywords(str(content).lower()))

        # Extract keywords from current query
        current_keywords = self._extract_keywords(current_query.lower())

        # Calculate overlap
        if not current_keywords or not recent_keywords:
            return False

        overlap = len(current_keywords & recent_keywords) / len(current_keywords)

        # Shift detected if overlap is low (<30%)
        return overlap < 0.3

    def filter_by_relevance(self, items: List[Any], task_context: str,
                           threshold: float = 0.5) -> List[Any]:
        """
        Filter a list of items by relevance to task context.

        Args:
            items: List of items (knowledge entries, facts, etc.)
            task_context: Current task description
            threshold: Minimum relevance score (0.0-1.0)

        Returns:
            Filtered list containing only relevant items
        """
        filtered = []

        for item in items:
            # Extract content from item
            if hasattr(item, 'content'):
                content = str(item.content)
            elif isinstance(item, dict):
                content = str(item.get('content', item.get('result', str(item))))
            else:
                content = str(item)

            # Evaluate relevance
            relevance = self.evaluate_relevance(content, task_context)

            if relevance.score >= threshold:
                filtered.append(item)
                logger.debug(f"  ✓ Relevant ({relevance.score:.2f}): {content[:80]}...")
            else:
                logger.debug(f"  ✗ Filtered out ({relevance.score:.2f}): {content[:80]}...")

        logger.info(f"Relevance filtering: {len(filtered)}/{len(items)} items kept (threshold={threshold})")
        return filtered

    def _extract_keywords(self, text: str, min_length: int = 4) -> set:
        """
        Extract significant keywords from text.

        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length

        Returns:
            Set of keywords
        """
        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }

        # Split and filter
        words = text.split()
        keywords = {
            word.strip('.,!?;:()[]{}"\'\n')
            for word in words
            if len(word) >= min_length and word.lower() not in stop_words
        }

        return keywords

    def _check_irrelevant_patterns(self, fact: str, task: str) -> float:
        """
        Check if fact matches known irrelevant patterns for the task.

        Args:
            fact: Fact text (lowercase)
            task: Task text (lowercase)

        Returns:
            Irrelevance penalty from 0.0 (relevant) to 1.0 (completely irrelevant)
        """
        penalty = 0.0

        for pattern_type, patterns in self.irrelevant_patterns.items():
            # Check if fact contains pattern keywords
            fact_has_pattern = any(p in fact for p in patterns)
            task_has_pattern = any(p in task for p in patterns)

            # If fact discusses pattern but task doesn't, penalize
            if fact_has_pattern and not task_has_pattern:
                penalty = max(penalty, 0.7)

        return penalty

    def create_relevance_report(self, items: List[Any], task_context: str) -> str:
        """
        Create a report showing relevance scores for all items.

        Args:
            items: List of items to evaluate
            task_context: Task description

        Returns:
            Formatted relevance report
        """
        lines = ["# Context Relevance Report\n"]
        lines.append(f"**Task**: {task_context[:100]}...\n\n")
        lines.append("## Item Relevance Scores\n\n")

        for i, item in enumerate(items, 1):
            # Extract content
            if hasattr(item, 'content'):
                content = str(item.content)[:100]
            elif isinstance(item, dict):
                content = str(item.get('content', str(item)))[:100]
            else:
                content = str(item)[:100]

            # Evaluate
            relevance = self.evaluate_relevance(content, task_context)

            # Format
            emoji = "🟢" if relevance.score >= 0.7 else "🟡" if relevance.score >= 0.4 else "🔴"
            lines.append(f"{i}. {emoji} **Score: {relevance.score:.2f}** ({relevance.category})\n")
            lines.append(f"   - {content}...\n")
            lines.append(f"   - Reason: {relevance.reason}\n")
            lines.append(f"   - Keywords: {', '.join(relevance.keywords_matched[:5])}\n\n")

        return "".join(lines)


class AdaptiveContextFilter:
    """
    Adaptive filter that learns which types of content are relevant for different task types.

    Extends basic relevance evaluation with pattern learning.
    """

    def __init__(self):
        """Initialize adaptive context filter."""
        self.evaluator = ContextRelevanceEvaluator()
        self.relevance_history: Dict[str, List[float]] = {}  # task_type -> relevance scores

    def filter_with_adaptation(self, items: List[Any], task_context: str,
                               task_type: str = "general") -> List[Any]:
        """
        Filter items with adaptive threshold based on task type.

        Args:
            items: Items to filter
            task_context: Task description
            task_type: Type of task (research, analysis, etc.)

        Returns:
            Filtered items
        """
        # Get adaptive threshold
        threshold = self._get_adaptive_threshold(task_type)

        # Filter
        filtered = self.evaluator.filter_by_relevance(items, task_context, threshold)

        # Update history
        avg_relevance = sum(
            self.evaluator.evaluate_relevance(str(item), task_context).score
            for item in filtered
        ) / max(len(filtered), 1)

        if task_type not in self.relevance_history:
            self.relevance_history[task_type] = []
        self.relevance_history[task_type].append(avg_relevance)

        return filtered

    def _get_adaptive_threshold(self, task_type: str) -> float:
        """
        Get adaptive relevance threshold based on historical performance.

        Args:
            task_type: Type of task

        Returns:
            Relevance threshold (0.0-1.0)
        """
        if task_type not in self.relevance_history or not self.relevance_history[task_type]:
            return 0.5  # Default threshold

        # Use average historical relevance as threshold
        history = self.relevance_history[task_type][-10:]  # Last 10 tasks
        avg_relevance = sum(history) / len(history)

        # Adjust threshold: if historical relevance is high, can be more selective
        return max(0.3, min(0.7, avg_relevance - 0.1))
