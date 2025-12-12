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
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
