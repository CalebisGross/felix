"""
Gap-Directed Learning for the Felix Framework.

Enables proactive knowledge acquisition by identifying high-priority gaps
and generating strategies to fill them. This is an OPT-IN feature that
remains disabled by default.

Key Features:
- Identify priority gaps for acquisition
- Generate search queries to fill gaps
- Track acquisition attempts and success rates
- Integrate with Knowledge Daemon for autonomous filling
- GapAcquisitionTrigger for monitoring and triggering acquisition (Issue #25)
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GapNotificationType(Enum):
    """Types of gap notifications for the UI."""
    NEW_GAP = "new_gap"           # New high-priority gap detected
    GAP_RESOLVED = "gap_resolved"  # Gap was successfully resolved
    ACQUISITION_STARTED = "acquisition_started"  # Acquisition attempt started
    ACQUISITION_FAILED = "acquisition_failed"    # Acquisition attempt failed


@dataclass
class AcquisitionTarget:
    """Represents a gap targeted for knowledge acquisition."""
    gap_id: str
    domain: str
    concept: Optional[str]
    priority_score: float  # 0.0-1.0 (severity * occurrences normalized)
    suggested_queries: List[str] = field(default_factory=list)
    acquisition_method: str = "web_search"  # web_search, manual, document_ingestion
    attempts: int = 0
    last_attempt: Optional[float] = None
    status: str = "pending"  # pending, in_progress, resolved, failed


@dataclass
class GapNotification:
    """Notification about a gap for the UI (Issue #25)."""
    notification_type: GapNotificationType
    gap_id: str
    domain: str
    concept: Optional[str]
    severity: float
    occurrence_count: int
    message: str
    timestamp: float = field(default_factory=time.time)


class GapAcquisitionTrigger:
    """
    Monitors high-priority gaps and triggers acquisition attempts (Issue #25).

    Implements the trigger conditions:
    - severity > min_severity (default 0.6)
    - occurrence_count >= min_occurrences (default 3)

    Actions when triggered:
    1. Queue targeted document search in watched directories
    2. Flag for user attention in chat UI
    3. If web search enabled (opt-in), queue web search
    """

    def __init__(self,
                 gap_tracker: Any,
                 min_severity: float = 0.6,
                 min_occurrences: int = 3,
                 notification_callback: Optional[Callable[[GapNotification], None]] = None):
        """
        Initialize Gap Acquisition Trigger.

        Args:
            gap_tracker: GapTracker instance for monitoring gaps
            min_severity: Minimum severity threshold for triggering (0.0-1.0)
            min_occurrences: Minimum occurrence count for triggering
            notification_callback: Optional callback for UI notifications
        """
        self.gap_tracker = gap_tracker
        self.min_severity = min_severity
        self.min_occurrences = min_occurrences
        self.notification_callback = notification_callback

        # Track notified gaps to avoid duplicate notifications
        self._notified_gaps: Dict[str, float] = {}  # gap_id -> last_notification_time
        self._notification_cooldown = 3600  # 1 hour cooldown between notifications for same gap

        # Track pending notifications for UI polling
        self._pending_notifications: List[GapNotification] = []

        logger.info(f"GapAcquisitionTrigger initialized (severity >= {min_severity}, "
                   f"occurrences >= {min_occurrences})")

    def check_triggers(self) -> List[AcquisitionTarget]:
        """
        Check for gaps that meet trigger conditions.

        Returns:
            List of AcquisitionTarget objects for gaps that should be acquired
        """
        triggered_targets = []

        try:
            # Get gaps that meet threshold criteria
            priority_gaps = self.gap_tracker.get_priority_gaps(
                min_severity=self.min_severity,
                min_occurrences=self.min_occurrences,
                limit=10
            )

            current_time = time.time()

            for gap in priority_gaps:
                # Check notification cooldown
                last_notified = self._notified_gaps.get(gap.gap_id, 0)
                if current_time - last_notified < self._notification_cooldown:
                    continue  # Skip - recently notified

                # Calculate priority score
                priority_score = gap.impact_severity_avg * min(1.0, gap.occurrence_count / 10)

                target = AcquisitionTarget(
                    gap_id=gap.gap_id,
                    domain=gap.domain,
                    concept=gap.concept,
                    priority_score=priority_score,
                    acquisition_method="web_search",
                    status="pending"
                )

                triggered_targets.append(target)

                # Create notification for UI
                self._create_notification(
                    GapNotificationType.NEW_GAP,
                    gap,
                    f"Knowledge gap detected: {gap.domain}/{gap.concept or 'general'} "
                    f"(severity: {gap.impact_severity_avg:.1%}, occurrences: {gap.occurrence_count})"
                )

                # Update notification timestamp
                self._notified_gaps[gap.gap_id] = current_time

            if triggered_targets:
                logger.info(f"GapAcquisitionTrigger: {len(triggered_targets)} gaps meet trigger conditions")

        except Exception as e:
            logger.error(f"Error checking gap triggers: {e}")

        return triggered_targets

    def _create_notification(self, notification_type: GapNotificationType,
                            gap: Any, message: str):
        """
        Create a notification for the UI.

        Args:
            notification_type: Type of notification
            gap: Gap object from tracker
            message: Human-readable message
        """
        notification = GapNotification(
            notification_type=notification_type,
            gap_id=gap.gap_id,
            domain=gap.domain,
            concept=gap.concept,
            severity=gap.impact_severity_avg,
            occurrence_count=gap.occurrence_count,
            message=message
        )

        # Add to pending notifications
        self._pending_notifications.append(notification)

        # Invoke callback if registered
        if self.notification_callback:
            try:
                self.notification_callback(notification)
            except Exception as e:
                logger.warning(f"Notification callback failed: {e}")

    def get_pending_notifications(self, clear: bool = True) -> List[GapNotification]:
        """
        Get pending notifications for the UI.

        Args:
            clear: If True, clear notifications after returning

        Returns:
            List of pending GapNotification objects
        """
        notifications = list(self._pending_notifications)
        if clear:
            self._pending_notifications.clear()
        return notifications

    def notify_resolution(self, gap_id: str, method: str):
        """
        Notify that a gap was resolved.

        Args:
            gap_id: ID of the resolved gap
            method: Resolution method (web_search, manual, document_ingestion)
        """
        # Get gap info for notification (may fail if already removed)
        try:
            # Create a simple notification without full gap info
            notification = GapNotification(
                notification_type=GapNotificationType.GAP_RESOLVED,
                gap_id=gap_id,
                domain="",  # Unknown at this point
                concept=None,
                severity=0.0,
                occurrence_count=0,
                message=f"Gap {gap_id} resolved via {method}"
            )
            self._pending_notifications.append(notification)

            if self.notification_callback:
                self.notification_callback(notification)

            # Remove from notified gaps tracking
            self._notified_gaps.pop(gap_id, None)

        except Exception as e:
            logger.debug(f"Could not create resolution notification: {e}")

    def get_gap_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current gap status for UI display.

        Returns:
            Dict with gap counts and top gaps
        """
        try:
            # Get all high-priority gaps
            priority_gaps = self.gap_tracker.get_priority_gaps(
                min_severity=self.min_severity,
                min_occurrences=self.min_occurrences,
                limit=5
            )

            # Get total unresolved gaps
            all_gaps = self.gap_tracker.get_gaps_for_display(limit=100, include_resolved=False)
            total_active = len(all_gaps)
            high_priority_count = len(priority_gaps)

            return {
                "total_active_gaps": total_active,
                "high_priority_gaps": high_priority_count,
                "top_gaps": [
                    {
                        "gap_id": g.gap_id,
                        "domain": g.domain,
                        "concept": g.concept or "(general)",
                        "severity": g.impact_severity_avg,
                        "occurrences": g.occurrence_count
                    }
                    for g in priority_gaps[:3]
                ],
                "needs_attention": high_priority_count > 0
            }

        except Exception as e:
            logger.error(f"Error getting gap summary: {e}")
            return {
                "total_active_gaps": 0,
                "high_priority_gaps": 0,
                "top_gaps": [],
                "needs_attention": False
            }


class GapDirectedLearner:
    """
    Manages proactive knowledge acquisition to fill identified gaps.

    This is an OPT-IN feature that remains disabled by default.
    Enable via config: epistemic_cartography.gap_directed_learning_enabled = true
    """

    def __init__(self, gap_tracker: Any, knowledge_store: Any,
                 web_search_client: Optional[Any] = None,
                 max_queries_per_gap: int = 3,
                 max_acquisition_attempts: int = 3):
        """
        Initialize Gap-Directed Learner.

        Args:
            gap_tracker: GapTracker instance for querying gaps
            knowledge_store: KnowledgeStore for storing acquired knowledge
            web_search_client: Optional web search client for acquisition
            max_queries_per_gap: Maximum search queries to generate per gap
            max_acquisition_attempts: Maximum attempts before marking gap as failed
        """
        self.gap_tracker = gap_tracker
        self.knowledge_store = knowledge_store
        self.web_search_client = web_search_client
        self.max_queries_per_gap = max_queries_per_gap
        self.max_acquisition_attempts = max_acquisition_attempts

        # Track acquisition state
        self._acquisition_queue: List[AcquisitionTarget] = []
        self._acquisition_history: Dict[str, AcquisitionTarget] = {}

        logger.info("GapDirectedLearner initialized (opt-in feature)")

    def get_acquisition_targets(self, min_severity: float = 0.6,
                                min_occurrences: int = 3,
                                limit: int = 5) -> List[AcquisitionTarget]:
        """
        Get high-priority gaps as acquisition targets.

        Only returns gaps that meet severity and occurrence thresholds,
        ensuring we focus on gaps that consistently hurt workflow quality.

        Args:
            min_severity: Minimum average severity to include (0.0-1.0)
            min_occurrences: Minimum times gap must have occurred
            limit: Maximum number of targets to return

        Returns:
            List of AcquisitionTarget objects sorted by priority
        """
        targets = []

        try:
            # Get priority gaps from tracker
            priority_gaps = self.gap_tracker.get_priority_gaps(
                min_severity=min_severity,
                min_occurrences=min_occurrences,
                limit=limit
            )

            for gap in priority_gaps:
                # Skip if already in acquisition history with too many attempts
                if gap.gap_id in self._acquisition_history:
                    existing = self._acquisition_history[gap.gap_id]
                    if existing.attempts >= self.max_acquisition_attempts:
                        continue  # Skip failed acquisitions
                    if existing.status == "resolved":
                        continue  # Already resolved

                # Calculate priority score (normalized)
                priority_score = gap.impact_severity_avg * min(1.0, gap.occurrence_count / 10)

                # Generate search queries for this gap
                queries = self.suggest_search_queries(gap.domain, gap.concept)

                target = AcquisitionTarget(
                    gap_id=gap.gap_id,
                    domain=gap.domain,
                    concept=gap.concept,
                    priority_score=priority_score,
                    suggested_queries=queries,
                    acquisition_method="web_search" if self.web_search_client else "manual"
                )

                # Restore attempt count from history
                if gap.gap_id in self._acquisition_history:
                    target.attempts = self._acquisition_history[gap.gap_id].attempts
                    target.last_attempt = self._acquisition_history[gap.gap_id].last_attempt

                targets.append(target)

            # Sort by priority (highest first)
            targets.sort(key=lambda t: t.priority_score, reverse=True)

            logger.info(f"Generated {len(targets)} acquisition targets "
                       f"(severity >= {min_severity}, occurrences >= {min_occurrences})")

        except Exception as e:
            logger.error(f"Failed to get acquisition targets: {e}")

        return targets

    def suggest_search_queries(self, domain: str,
                               concept: Optional[str] = None) -> List[str]:
        """
        Generate search queries to fill a knowledge gap.

        Creates targeted queries based on domain and concept, using
        patterns optimized for web search.

        Args:
            domain: Knowledge domain (e.g., "python", "machine_learning")
            concept: Optional specific concept within domain

        Returns:
            List of search query strings
        """
        queries = []

        # Normalize domain name for search
        domain_clean = domain.replace("_", " ").replace("-", " ")

        if concept:
            # Specific concept queries
            concept_clean = concept.replace("_", " ").replace("-", " ")
            queries.extend([
                f"what is {concept_clean} in {domain_clean}",
                f"{concept_clean} {domain_clean} tutorial",
                f"{concept_clean} best practices {domain_clean}",
            ])
        else:
            # General domain queries
            queries.extend([
                f"{domain_clean} fundamentals overview",
                f"{domain_clean} key concepts explained",
                f"{domain_clean} getting started guide",
            ])

        # Add timestamp-based query for current information
        queries.append(f"{domain_clean} latest updates 2024")

        # Limit to max queries
        return queries[:self.max_queries_per_gap]

    def attempt_acquisition(self, target: AcquisitionTarget) -> Dict[str, Any]:
        """
        Attempt to acquire knowledge for a gap.

        Uses web search (if available) to find information and stores
        results in the knowledge store.

        Args:
            target: AcquisitionTarget to fill

        Returns:
            Dict with acquisition result:
                - success: bool
                - entries_created: int
                - message: str
        """
        target.attempts += 1
        target.last_attempt = time.time()
        target.status = "in_progress"

        # Store in history
        self._acquisition_history[target.gap_id] = target

        if not self.web_search_client:
            target.status = "failed"
            return {
                "success": False,
                "entries_created": 0,
                "message": "No web search client available for acquisition"
            }

        entries_created = 0

        try:
            from src.memory.knowledge_store import KnowledgeType, ConfidenceLevel

            for query in target.suggested_queries:
                logger.info(f"Searching for gap '{target.gap_id}': {query}")

                try:
                    # Perform web search
                    results = self.web_search_client.search(query, max_results=3)

                    if not results:
                        continue

                    # Store each result as knowledge
                    for result in results:
                        # Create knowledge entry from search result
                        content = {
                            "title": result.get("title", ""),
                            "snippet": result.get("snippet", result.get("body", "")),
                            "url": result.get("href", result.get("url", "")),
                            "source": "gap_directed_learning",
                            "gap_id": target.gap_id,
                            "query": query
                        }

                        entry_id = self.knowledge_store.store_knowledge(
                            knowledge_type=KnowledgeType.FACT,
                            content=content,
                            confidence_level=ConfidenceLevel.MEDIUM,  # Web search = MEDIUM confidence
                            source_agent="gap_directed_learner",
                            domain=target.domain,
                            tags=[target.domain, target.concept or "general", "gap_acquisition"]
                        )

                        if entry_id:
                            entries_created += 1
                            logger.debug(f"Created knowledge entry {entry_id} for gap {target.gap_id}")

                except Exception as search_error:
                    logger.warning(f"Search failed for query '{query}': {search_error}")
                    continue

            # Determine success
            if entries_created > 0:
                target.status = "resolved"
                self.mark_gap_resolved(target.gap_id, "web_search")
                return {
                    "success": True,
                    "entries_created": entries_created,
                    "message": f"Created {entries_created} knowledge entries for gap '{target.gap_id}'"
                }
            else:
                if target.attempts >= self.max_acquisition_attempts:
                    target.status = "failed"
                else:
                    target.status = "pending"  # Can retry
                return {
                    "success": False,
                    "entries_created": 0,
                    "message": f"No results found for gap '{target.gap_id}' (attempt {target.attempts}/{self.max_acquisition_attempts})"
                }

        except Exception as e:
            target.status = "failed" if target.attempts >= self.max_acquisition_attempts else "pending"
            logger.error(f"Acquisition failed for gap '{target.gap_id}': {e}")
            return {
                "success": False,
                "entries_created": 0,
                "message": f"Acquisition error: {e}"
            }

    def mark_gap_resolved(self, gap_id: str, method: str) -> bool:
        """
        Mark a gap as resolved in the tracker.

        Args:
            gap_id: Gap to mark resolved
            method: Resolution method (web_search, manual, document_ingestion)

        Returns:
            True if successful
        """
        try:
            result = self.gap_tracker.mark_gap_resolved(gap_id, method)

            # Update local history
            if gap_id in self._acquisition_history:
                self._acquisition_history[gap_id].status = "resolved"

            logger.info(f"Gap '{gap_id}' marked as resolved via {method}")
            return result

        except Exception as e:
            logger.error(f"Failed to mark gap '{gap_id}' as resolved: {e}")
            return False

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """
        Get statistics about acquisition attempts.

        Returns:
            Dict with acquisition statistics
        """
        total = len(self._acquisition_history)
        resolved = sum(1 for t in self._acquisition_history.values() if t.status == "resolved")
        failed = sum(1 for t in self._acquisition_history.values() if t.status == "failed")
        pending = sum(1 for t in self._acquisition_history.values() if t.status == "pending")
        in_progress = sum(1 for t in self._acquisition_history.values() if t.status == "in_progress")

        return {
            "total_targets": total,
            "resolved": resolved,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "success_rate": resolved / total if total > 0 else 0.0
        }

    def run_acquisition_cycle(self, max_targets: int = 3) -> Dict[str, Any]:
        """
        Run a single acquisition cycle.

        Gets top priority targets and attempts to fill them.
        Used by Knowledge Daemon Mode E.

        Args:
            max_targets: Maximum targets to process in this cycle

        Returns:
            Dict with cycle results
        """
        logger.info("=" * 60)
        logger.info("GAP-DIRECTED LEARNING CYCLE STARTING")
        logger.info("=" * 60)

        targets = self.get_acquisition_targets(limit=max_targets)

        if not targets:
            logger.info("No acquisition targets found - all gaps resolved or below threshold")
            return {
                "targets_found": 0,
                "targets_resolved": 0,
                "entries_created": 0
            }

        targets_resolved = 0
        total_entries = 0

        for target in targets:
            logger.info(f"Processing gap: {target.domain}/{target.concept or 'general'} "
                       f"(priority: {target.priority_score:.2f})")

            result = self.attempt_acquisition(target)

            if result["success"]:
                targets_resolved += 1
                total_entries += result["entries_created"]
                logger.info(f"  ✓ {result['message']}")
            else:
                logger.info(f"  ✗ {result['message']}")

        logger.info("=" * 60)
        logger.info(f"CYCLE COMPLETE: {targets_resolved}/{len(targets)} gaps resolved, "
                   f"{total_entries} entries created")
        logger.info("=" * 60)

        return {
            "targets_found": len(targets),
            "targets_resolved": targets_resolved,
            "entries_created": total_entries
        }
