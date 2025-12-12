"""
Knowledge Quality Checker for Felix Knowledge Brain

Provides tools for detecting and resolving quality issues in the knowledge base:
- Duplicate detection using embedding similarity and text matching
- Contradiction finding through semantic analysis
- Quality scoring based on multiple factors
- Merge suggestions with confidence ratings

Phase 5 feature for maintaining knowledge base quality.
"""

import logging
import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from src.memory.knowledge_store import KnowledgeStore, KnowledgeEntry, ConfidenceLevel

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCandidate:
    """Pair of potentially duplicate entries."""
    entry1_id: str
    entry2_id: str
    similarity_score: float
    similarity_type: str  # 'embedding', 'text', 'concept'
    entry1_content: Dict[str, Any]
    entry2_content: Dict[str, Any]
    suggested_action: str  # 'merge', 'review', 'keep_both'
    reason: str


@dataclass
class ContradictionCandidate:
    """Pair of potentially contradictory entries."""
    entry1_id: str
    entry2_id: str
    contradiction_type: str  # 'direct', 'semantic', 'temporal'
    entry1_content: Dict[str, Any]
    entry2_content: Dict[str, Any]
    confidence_score: float
    reason: str


@dataclass
class QualityScore:
    """Overall quality score for a knowledge entry."""
    knowledge_id: str
    overall_score: float  # 0.0 to 1.0
    confidence_score: float
    relationship_score: float
    validation_score: float
    access_success_score: float
    issues: List[str]
    recommendations: List[str]


class QualityChecker:
    """
    Knowledge base quality checker with duplicate detection,
    contradiction finding, and quality scoring.
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        """
        Initialize quality checker.

        Args:
            knowledge_store: KnowledgeStore instance
        """
        self.knowledge_store = knowledge_store
        self.storage_path = knowledge_store.storage_path

        # Check for embedding availability
        self.has_embeddings = self._check_embeddings_available()

    def _check_embeddings_available(self) -> bool:
        """Check if embeddings are available in the database."""
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE embedding IS NOT NULL
            """)

            count = cursor.fetchone()[0]
            conn.close()

            logger.info(f"Embeddings available for {count} entries")
            return count > 0

        except Exception as e:
            logger.error(f"Error checking embeddings: {e}")
            return False

    def find_duplicates(
        self,
        similarity_threshold: float = 0.90,
        method: str = "auto"
    ) -> List[DuplicateCandidate]:
        """
        Find potential duplicate knowledge entries.

        Args:
            similarity_threshold: Minimum similarity score (0.0-1.0)
            method: Detection method ('auto', 'embedding', 'text', 'concept')

        Returns:
            List of duplicate candidate pairs
        """
        candidates = []

        if method == "auto":
            # Use best available method
            if self.has_embeddings:
                candidates.extend(self._find_duplicates_by_embedding(similarity_threshold))
            else:
                candidates.extend(self._find_duplicates_by_text(similarity_threshold))

            # Always add concept-based duplicates
            candidates.extend(self._find_duplicates_by_concept())

        elif method == "embedding":
            if self.has_embeddings:
                candidates = self._find_duplicates_by_embedding(similarity_threshold)
            else:
                logger.warning("Embeddings not available, falling back to text method")
                candidates = self._find_duplicates_by_text(similarity_threshold)

        elif method == "text":
            candidates = self._find_duplicates_by_text(similarity_threshold)

        elif method == "concept":
            candidates = self._find_duplicates_by_concept()

        # Deduplicate and sort by similarity
        seen_pairs = set()
        unique_candidates = []

        for candidate in candidates:
            # Create canonical pair (sorted order)
            pair = tuple(sorted([candidate.entry1_id, candidate.entry2_id]))

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_candidates.append(candidate)

        unique_candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"Found {len(unique_candidates)} potential duplicate pairs")
        return unique_candidates

    def _find_duplicates_by_embedding(
        self,
        threshold: float
    ) -> List[DuplicateCandidate]:
        """Find duplicates using embedding cosine similarity."""
        try:
            from src.knowledge.embeddings import deserialize_embedding
        except ImportError:
            logger.warning("Embeddings module not available")
            return []

        candidates = []

        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Get all entries with embeddings
            cursor.execute("""
                SELECT knowledge_id, content_json, embedding, confidence_level
                FROM knowledge_entries
                WHERE embedding IS NOT NULL
            """)

            entries = []
            for row in cursor.fetchall():
                try:
                    embedding = deserialize_embedding(row[2])
                    entries.append({
                        'id': row[0],
                        'content': json.loads(row[1]) if row[1] else {},
                        'embedding': embedding,
                        'confidence': row[3]
                    })
                except Exception as e:
                    logger.debug(f"Error loading entry: {e}")

            conn.close()

            # Compare embeddings pairwise (N^2 operation, limit for performance)
            if len(entries) > 1000:
                logger.warning(f"Large dataset ({len(entries)} entries), duplicate detection may be slow")

            import numpy as np

            for i in range(len(entries)):
                for j in range(i + 1, min(i + 100, len(entries))):  # Limit comparisons per entry
                    e1, e2 = entries[i], entries[j]

                    # Cosine similarity
                    similarity = np.dot(e1['embedding'], e2['embedding']) / (
                        np.linalg.norm(e1['embedding']) * np.linalg.norm(e2['embedding'])
                    )

                    if similarity >= threshold:
                        # Determine action based on confidence
                        if e1['confidence'] == e2['confidence']:
                            action = 'review'
                            reason = "Equal confidence - manual review recommended"
                        else:
                            action = 'merge'
                            reason = f"Keep {e1['confidence'] if e1['confidence'] > e2['confidence'] else e2['confidence']} confidence entry"

                        candidates.append(DuplicateCandidate(
                            entry1_id=e1['id'],
                            entry2_id=e2['id'],
                            similarity_score=float(similarity),
                            similarity_type='embedding',
                            entry1_content=e1['content'],
                            entry2_content=e2['content'],
                            suggested_action=action,
                            reason=reason
                        ))

            return candidates

        except Exception as e:
            logger.error(f"Embedding-based duplicate detection failed: {e}")
            return []

    def _find_duplicates_by_text(
        self,
        threshold: float
    ) -> List[DuplicateCandidate]:
        """Find duplicates using text similarity (edit distance ratio)."""
        candidates = []

        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT knowledge_id, content_json, confidence_level, domain
                FROM knowledge_entries
                ORDER BY domain, created_at DESC
                LIMIT 500
            """)

            entries = []
            for row in cursor.fetchall():
                content = json.loads(row[1]) if row[1] else {}
                # Combine text for comparison
                text = ' '.join(str(v) for v in content.values() if v)

                entries.append({
                    'id': row[0],
                    'content': content,
                    'text': text.lower(),
                    'confidence': row[2],
                    'domain': row[3]
                })

            conn.close()

            # Compare text similarity within same domain
            from difflib import SequenceMatcher

            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    e1, e2 = entries[i], entries[j]

                    # Only compare within same domain
                    if e1['domain'] != e2['domain']:
                        continue

                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, e1['text'], e2['text']).ratio()

                    if similarity >= threshold:
                        candidates.append(DuplicateCandidate(
                            entry1_id=e1['id'],
                            entry2_id=e2['id'],
                            similarity_score=similarity,
                            similarity_type='text',
                            entry1_content=e1['content'],
                            entry2_content=e2['content'],
                            suggested_action='merge' if similarity > 0.95 else 'review',
                            reason=f"Text similarity {similarity:.1%} within domain {e1['domain']}"
                        ))

            return candidates

        except Exception as e:
            logger.error(f"Text-based duplicate detection failed: {e}")
            return []

    def _find_duplicates_by_concept(self) -> List[DuplicateCandidate]:
        """Find duplicate concept names across entries."""
        candidates = []

        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Find entries with same concept name
            cursor.execute("""
                SELECT
                    knowledge_id,
                    content_json,
                    confidence_level,
                    domain
                FROM knowledge_entries
                WHERE knowledge_type = 'agent_insight'
                  AND content_json LIKE '%"concept":%'
                ORDER BY domain
            """)

            # Group by concept name
            concept_groups = {}

            for row in cursor.fetchall():
                content = json.loads(row[1]) if row[1] else {}
                concept = content.get('concept', '').strip().lower()

                if concept:
                    if concept not in concept_groups:
                        concept_groups[concept] = []

                    concept_groups[concept].append({
                        'id': row[0],
                        'content': content,
                        'confidence': row[2],
                        'domain': row[3]
                    })

            conn.close()

            # Find groups with duplicates
            for concept, entries in concept_groups.items():
                if len(entries) > 1:
                    # Create candidates for each pair
                    for i in range(len(entries)):
                        for j in range(i + 1, len(entries)):
                            e1, e2 = entries[i], entries[j]

                            candidates.append(DuplicateCandidate(
                                entry1_id=e1['id'],
                                entry2_id=e2['id'],
                                similarity_score=1.0,  # Exact concept match
                                similarity_type='concept',
                                entry1_content=e1['content'],
                                entry2_content=e2['content'],
                                suggested_action='merge',
                                reason=f"Same concept '{concept}' in domain {e1['domain']}"
                            ))

            return candidates

        except Exception as e:
            logger.error(f"Concept-based duplicate detection failed: {e}")
            return []

    def find_contradictions(self) -> List[ContradictionCandidate]:
        """
        Find potentially contradictory entries.

        Returns:
            List of contradiction candidates
        """
        candidates = []

        # Find entries with same concept but different definitions
        candidates.extend(self._find_direct_contradictions())

        logger.info(f"Found {len(candidates)} potential contradictions")
        return candidates

    def _find_direct_contradictions(self) -> List[ContradictionCandidate]:
        """Find direct contradictions (same concept, different definitions)."""
        candidates = []

        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            # Find entries with same concept
            cursor.execute("""
                SELECT
                    knowledge_id,
                    content_json,
                    confidence_level,
                    domain
                FROM knowledge_entries
                WHERE content_json LIKE '%"concept":%'
                ORDER BY domain
            """)

            concept_groups = {}

            for row in cursor.fetchall():
                content = json.loads(row[1]) if row[1] else {}
                concept = content.get('concept', '').strip().lower()
                definition = content.get('definition', '').strip().lower()

                if concept and definition:
                    if concept not in concept_groups:
                        concept_groups[concept] = []

                    concept_groups[concept].append({
                        'id': row[0],
                        'content': content,
                        'definition': definition,
                        'confidence': row[2],
                        'domain': row[3]
                    })

            conn.close()

            # Check for different definitions
            from difflib import SequenceMatcher

            for concept, entries in concept_groups.items():
                if len(entries) > 1:
                    for i in range(len(entries)):
                        for j in range(i + 1, len(entries)):
                            e1, e2 = entries[i], entries[j]

                            # Calculate definition similarity
                            similarity = SequenceMatcher(
                                None,
                                e1['definition'],
                                e2['definition']
                            ).ratio()

                            # If definitions are different (low similarity)
                            if similarity < 0.5:
                                confidence = 1.0 - similarity  # Higher confidence for more different

                                candidates.append(ContradictionCandidate(
                                    entry1_id=e1['id'],
                                    entry2_id=e2['id'],
                                    contradiction_type='direct',
                                    entry1_content=e1['content'],
                                    entry2_content=e2['content'],
                                    confidence_score=confidence,
                                    reason=f"Concept '{concept}' has conflicting definitions (similarity: {similarity:.1%})"
                                ))

            return candidates

        except Exception as e:
            logger.error(f"Direct contradiction detection failed: {e}")
            return []

    def calculate_quality_score(self, knowledge_id: str) -> Optional[QualityScore]:
        """
        Calculate comprehensive quality score for an entry.

        Args:
            knowledge_id: Entry ID to score

        Returns:
            QualityScore object or None if entry not found
        """
        try:
            entry = self.knowledge_store.get_entry_by_id(knowledge_id)
            if not entry:
                return None

            # Component scores (0.0 to 1.0)
            confidence_score = self._score_confidence(entry)
            relationship_score = self._score_relationships(entry)
            validation_score = entry.validation_score
            access_success_score = entry.success_rate

            # Weighted overall score
            overall_score = (
                confidence_score * 0.35 +
                relationship_score * 0.25 +
                validation_score * 0.25 +
                access_success_score * 0.15
            )

            # Identify issues
            issues = []
            recommendations = []

            if confidence_score < 0.5:
                issues.append("Low confidence level")
                recommendations.append("Review and validate entry accuracy")

            if relationship_score < 0.3:
                issues.append("Few or no relationships to other concepts")
                recommendations.append("Build connections with related knowledge")

            if validation_score < 0.7:
                issues.append("Low validation score")
                recommendations.append("Re-validate entry with updated criteria")

            if access_success_score < 0.5 and entry.access_count > 3:
                issues.append("Low success rate in usage")
                recommendations.append("Review relevance and accuracy")

            if not issues:
                recommendations.append("Entry quality is good - no action needed")

            return QualityScore(
                knowledge_id=knowledge_id,
                overall_score=overall_score,
                confidence_score=confidence_score,
                relationship_score=relationship_score,
                validation_score=validation_score,
                access_success_score=access_success_score,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Quality scoring failed for {knowledge_id}: {e}")
            return None

    def _score_confidence(self, entry: KnowledgeEntry) -> float:
        """Score confidence level (0.0 to 1.0)."""
        confidence_map = {
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERIFIED: 1.0
        }
        return confidence_map.get(entry.confidence_level, 0.5)

    def _score_relationships(self, entry: KnowledgeEntry) -> float:
        """Score based on relationship count (0.0 to 1.0)."""
        relationship_count = len(entry.related_entries)

        # Scoring curve: 0 rels=0.0, 1=0.3, 3=0.6, 5=0.8, 10+=1.0
        if relationship_count == 0:
            return 0.0
        elif relationship_count == 1:
            return 0.3
        elif relationship_count <= 3:
            return 0.6
        elif relationship_count <= 5:
            return 0.8
        else:
            return min(1.0, 0.8 + (relationship_count - 5) * 0.04)

    def get_merge_suggestions(
        self,
        duplicates: List[DuplicateCandidate]
    ) -> List[Dict[str, Any]]:
        """
        Generate merge suggestions with confidence rankings.

        Args:
            duplicates: List of duplicate candidates

        Returns:
            List of merge suggestions with rankings
        """
        suggestions = []

        for dup in duplicates:
            # Get entries
            entry1 = self.knowledge_store.get_entry_by_id(dup.entry1_id)
            entry2 = self.knowledge_store.get_entry_by_id(dup.entry2_id)

            if not entry1 or not entry2:
                continue

            # Determine which to keep as primary
            confidence_order = {
                ConfidenceLevel.LOW: 1,
                ConfidenceLevel.MEDIUM: 2,
                ConfidenceLevel.HIGH: 3,
                ConfidenceLevel.VERIFIED: 4
            }

            c1 = confidence_order.get(entry1.confidence_level, 0)
            c2 = confidence_order.get(entry2.confidence_level, 0)

            if c1 > c2:
                primary_id = entry1.knowledge_id
                secondary_id = entry2.knowledge_id
                reason = f"Higher confidence ({entry1.confidence_level.value})"
            elif c2 > c1:
                primary_id = entry2.knowledge_id
                secondary_id = entry1.knowledge_id
                reason = f"Higher confidence ({entry2.confidence_level.value})"
            else:
                # Same confidence - prefer newer or more accessed
                if entry1.access_count > entry2.access_count:
                    primary_id = entry1.knowledge_id
                    secondary_id = entry2.knowledge_id
                    reason = "More frequently accessed"
                elif entry2.access_count > entry1.access_count:
                    primary_id = entry2.knowledge_id
                    secondary_id = entry1.knowledge_id
                    reason = "More frequently accessed"
                else:
                    # Prefer newer
                    if entry1.created_at > entry2.created_at:
                        primary_id = entry1.knowledge_id
                        secondary_id = entry2.knowledge_id
                        reason = "More recent"
                    else:
                        primary_id = entry2.knowledge_id
                        secondary_id = entry1.knowledge_id
                        reason = "More recent"

            suggestions.append({
                'primary_id': primary_id,
                'secondary_id': secondary_id,
                'similarity_score': dup.similarity_score,
                'similarity_type': dup.similarity_type,
                'merge_strategy': 'combine_content' if dup.similarity_score < 0.98 else 'keep_primary',
                'reason': reason,
                'suggested_action': dup.suggested_action,
                'confidence_rating': 'high' if dup.similarity_score > 0.95 else 'medium'
            })

        # Sort by similarity score (highest first)
        suggestions.sort(key=lambda x: x['similarity_score'], reverse=True)

        return suggestions
