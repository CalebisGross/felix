"""
Knowledge Coverage Analyzer for the Felix Framework.

Analyzes task requirements against knowledge coverage to identify gaps
before workflow execution. Enables epistemic self-awareness by letting
Felix know what it knows and doesn't know.

Key Features:
- Domain coverage metrics (entry count, confidence distribution, freshness)
- Task-based gap detection (infer domains from task description)
- Pre-workflow coverage reports
- Gap severity scoring
"""

import re
import time
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Report on knowledge coverage for a task."""
    covered_domains: List[str] = field(default_factory=list)
    weak_domains: List[str] = field(default_factory=list)
    missing_domains: List[str] = field(default_factory=list)
    overall_coverage_score: float = 0.0
    gap_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    domain_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class KnowledgeCoverageAnalyzer:
    """
    Analyzes knowledge coverage against task requirements.

    Provides pre-workflow analysis to identify knowledge gaps before
    agents begin processing, enabling proactive gap-filling strategies.
    """

    # Domain inference patterns - map keywords to knowledge domains
    DOMAIN_PATTERNS = {
        'technology': [
            r'\b(software|hardware|computer|system|server|network|api|database|cloud)\b',
            r'\b(programming|code|coding|developer|development)\b',
        ],
        'python': [
            r'\bpython\b',
            r'\b(pip|virtualenv|pytest|django|flask|pandas|numpy)\b',
        ],
        'javascript': [
            r'\b(javascript|js|typescript|ts|node|npm|react|vue|angular)\b',
        ],
        'machine_learning': [
            r'\b(machine\s+learning|ml|deep\s+learning|neural|ai|artificial\s+intelligence)\b',
            r'\b(model|training|dataset|tensorflow|pytorch|sklearn)\b',
        ],
        'data_science': [
            r'\b(data\s+science|analytics|statistics|visualization)\b',
            r'\b(pandas|numpy|matplotlib|jupyter)\b',
        ],
        'security': [
            r'\b(security|encryption|authentication|authorization|vulnerability)\b',
            r'\b(ssl|tls|oauth|jwt|password|hash)\b',
        ],
        'devops': [
            r'\b(devops|ci|cd|docker|kubernetes|k8s|jenkins|github\s+actions)\b',
            r'\b(deployment|infrastructure|monitoring|logging)\b',
        ],
        'web_development': [
            r'\b(web|html|css|http|rest|graphql|frontend|backend)\b',
            r'\b(browser|dom|ajax|fetch|cors)\b',
        ],
        'database': [
            r'\b(database|sql|nosql|postgres|mysql|mongodb|redis|sqlite)\b',
            r'\b(query|table|index|schema|migration)\b',
        ],
    }

    # Coverage thresholds
    COVERAGE_THRESHOLD_LOW = 0.3       # Below = critical gap (missing)
    COVERAGE_THRESHOLD_ADEQUATE = 0.6  # Above = covered

    def __init__(self, knowledge_store: Any):
        """
        Initialize Coverage Analyzer.

        Args:
            knowledge_store: KnowledgeStore instance for querying coverage metrics
        """
        self.knowledge_store = knowledge_store
        logger.info("KnowledgeCoverageAnalyzer initialized")

    def analyze_coverage(self, task_description: str,
                        task_domains: Optional[List[str]] = None) -> CoverageReport:
        """
        Analyze knowledge coverage for a task.

        Args:
            task_description: The task description to analyze
            task_domains: Optional explicit list of domains (if None, inferred from task)

        Returns:
            CoverageReport with coverage analysis
        """
        # Infer domains from task if not provided
        if task_domains is None:
            task_domains = self._infer_domains_from_task(task_description)

        if not task_domains:
            logger.info("No specific domains identified for task, using general coverage")
            task_domains = ['general']

        logger.info(f"Analyzing coverage for domains: {task_domains}")

        # Compute coverage for each domain
        covered = []
        weak = []
        missing = []
        domain_metrics = {}

        for domain in task_domains:
            metrics = self._compute_domain_coverage(domain)
            domain_metrics[domain] = metrics

            coverage_score = metrics.get('coverage_score', 0.0)

            if coverage_score >= self.COVERAGE_THRESHOLD_ADEQUATE:
                covered.append(domain)
            elif coverage_score >= self.COVERAGE_THRESHOLD_LOW:
                weak.append(domain)
            else:
                missing.append(domain)

        # Calculate overall coverage score
        if domain_metrics:
            scores = [m.get('coverage_score', 0.0) for m in domain_metrics.values()]
            overall_score = sum(scores) / len(scores)
        else:
            overall_score = 0.0

        # Generate gap summary
        gap_parts = []
        if missing:
            gap_parts.append(f"No knowledge for: {', '.join(missing)}")
        if weak:
            gap_parts.append(f"Weak coverage: {', '.join(weak)}")
        gap_summary = "; ".join(gap_parts) if gap_parts else "Adequate coverage"

        # Generate recommendations
        recommendations = []
        if missing:
            recommendations.append("trigger_web_search")
            for domain in missing:
                recommendations.append(f"acquire_knowledge:{domain}")
        if weak:
            recommendations.append("consider_web_search")

        return CoverageReport(
            covered_domains=covered,
            weak_domains=weak,
            missing_domains=missing,
            overall_coverage_score=overall_score,
            gap_summary=gap_summary,
            recommendations=recommendations,
            domain_metrics=domain_metrics
        )

    def _infer_domains_from_task(self, task_description: str) -> List[str]:
        """
        Infer relevant domains from task description using pattern matching.

        Args:
            task_description: Task description text

        Returns:
            List of inferred domain names
        """
        task_lower = task_description.lower()
        inferred_domains = []

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, task_lower, re.IGNORECASE):
                    if domain not in inferred_domains:
                        inferred_domains.append(domain)
                        logger.debug(f"Inferred domain '{domain}' from pattern '{pattern}'")
                    break  # One match per domain is enough

        return inferred_domains

    def _compute_domain_coverage(self, domain: str) -> Dict[str, Any]:
        """
        Compute coverage metrics for a specific domain.

        Args:
            domain: Domain name to compute coverage for

        Returns:
            Dictionary with coverage metrics:
            - entry_count: Number of knowledge entries
            - avg_confidence: Average confidence level
            - freshness_score: How recent the entries are
            - coverage_score: Overall coverage score (0.0-1.0)
        """
        if not self.knowledge_store:
            return {
                'entry_count': 0,
                'avg_confidence': 0.0,
                'freshness_score': 0.0,
                'coverage_score': 0.0
            }

        try:
            # Query knowledge store for domain metrics
            # Use get_domain_coverage_metrics if available, otherwise compute directly
            if hasattr(self.knowledge_store, 'get_domain_coverage_metrics'):
                return self.knowledge_store.get_domain_coverage_metrics(domain)

            # Fallback: compute metrics directly via SQL
            return self._compute_metrics_directly(domain)

        except Exception as e:
            logger.warning(f"Failed to compute coverage for domain '{domain}': {e}")
            return {
                'entry_count': 0,
                'avg_confidence': 0.0,
                'freshness_score': 0.0,
                'coverage_score': 0.0
            }

    def _compute_metrics_directly(self, domain: str) -> Dict[str, Any]:
        """
        Compute domain metrics directly via SQL query.

        Args:
            domain: Domain to query

        Returns:
            Coverage metrics dictionary
        """
        try:
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                # Count entries and compute average confidence
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as entry_count,
                        AVG(CASE
                            WHEN confidence_level = 'VERIFIED' THEN 1.0
                            WHEN confidence_level = 'HIGH' THEN 0.8
                            WHEN confidence_level = 'MEDIUM' THEN 0.6
                            WHEN confidence_level = 'LOW' THEN 0.4
                            WHEN confidence_level = 'SPECULATIVE' THEN 0.2
                            ELSE 0.5
                        END) as avg_confidence,
                        AVG(updated_at) as avg_updated
                    FROM knowledge_entries
                    WHERE domain = ? OR domain LIKE ?
                """, (domain, f"%{domain}%"))

                row = cursor.fetchone()
                entry_count = row[0] or 0
                avg_confidence = row[1] or 0.0
                avg_updated = row[2] or 0

                # Compute freshness score (entries updated in last 7 days = 1.0, older = decays)
                current_time = time.time()
                if avg_updated > 0:
                    age_days = (current_time - avg_updated) / (24 * 3600)
                    freshness_score = max(0.0, 1.0 - (age_days / 30))  # Decays over 30 days
                else:
                    freshness_score = 0.0

                # Compute coverage score as weighted combination
                # entry_count_factor: more entries = higher coverage (capped)
                entry_factor = min(1.0, entry_count / 10)  # Cap at 10 entries

                coverage_score = (
                    0.5 * entry_factor +
                    0.3 * avg_confidence +
                    0.2 * freshness_score
                )

                return {
                    'entry_count': entry_count,
                    'avg_confidence': avg_confidence,
                    'freshness_score': freshness_score,
                    'coverage_score': coverage_score
                }

        except Exception as e:
            logger.error(f"SQL error computing metrics for '{domain}': {e}")
            return {
                'entry_count': 0,
                'avg_confidence': 0.0,
                'freshness_score': 0.0,
                'coverage_score': 0.0
            }

    def get_all_domain_coverages(self) -> Dict[str, float]:
        """
        Get coverage scores for all known domains.

        Useful for GUI visualization.

        Returns:
            Dictionary mapping domain names to coverage scores
        """
        coverages = {}

        try:
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT domain FROM knowledge_entries
                """)
                domains = [row[0] for row in cursor.fetchall()]

                for domain in domains:
                    metrics = self._compute_domain_coverage(domain)
                    coverages[domain] = metrics.get('coverage_score', 0.0)

        except Exception as e:
            logger.error(f"Failed to get all domain coverages: {e}")

        return coverages

    def get_coverage_history(self, domain: str, days: int = 30) -> List[Tuple[float, float]]:
        """
        Get coverage trend over time for a domain.

        Args:
            domain: Domain to query
            days: Number of days to look back

        Returns:
            List of (timestamp, coverage_score) tuples
        """
        # Note: This is a simplified implementation. A full implementation would
        # require storing historical coverage snapshots.
        current_metrics = self._compute_domain_coverage(domain)
        current_time = time.time()

        # Return single point for now (actual history would require storing snapshots)
        return [(current_time, current_metrics.get('coverage_score', 0.0))]
