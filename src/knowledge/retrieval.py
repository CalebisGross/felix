"""
Knowledge Retrieval System for Felix Knowledge Brain

Semantic search and context building for workflow augmentation:
- Multi-strategy search (embeddings, FTS5, keyword)
- Relevance scoring with meta-learning boost
- Context formatting for agent consumption
- Usage tracking for continuous improvement

Agents can query the brain to retrieve relevant domain knowledge.
"""

import logging
import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from src.memory.knowledge_store import KnowledgeStore
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with relevance scoring."""
    knowledge_id: str
    content: Dict[str, Any]
    domain: str
    confidence_level: str
    relevance_score: float  # 0.0 to 1.0
    source_doc_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    reasoning: str = ""  # Why this result is relevant

    def to_context_str(self) -> str:
        """Format result for agent context."""
        concept = self.content.get('concept', '')
        definition = self.content.get('definition', '')
        examples = self.content.get('examples', [])

        text = f"**{concept}**"
        if definition:
            text += f": {definition}"
        if examples:
            text += f"\nExamples: {', '.join(examples[:2])}"
        text += f" (relevance: {self.relevance_score:.2f})"
        return text


@dataclass
class RetrievalContext:
    """Formatted context for agent consumption."""
    query: str
    results: List[SearchResult]
    total_results: int
    retrieval_method: str  # embedding/tfidf/fts5/hybrid
    processing_time: float

    def to_agent_context(self, max_results: int = 10) -> str:
        """Format as context string for agents."""
        if not self.results:
            return f"No relevant knowledge found for: {self.query}"

        text = f"RELEVANT KNOWLEDGE (from {len(self.results)} sources):\n\n"

        for i, result in enumerate(self.results[:max_results], 1):
            text += f"{i}. {result.to_context_str()}\n"

        text += f"\nRetrieval method: {self.retrieval_method}"
        return text

    def get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of domains in results."""
        dist = defaultdict(int)
        for result in self.results:
            dist[result.domain] += 1
        return dict(dist)


class KnowledgeRetriever:
    """
    Semantic search and retrieval system for knowledge brain.

    Combines multiple search strategies with meta-learning boost.
    """

    def __init__(self,
                 knowledge_store: KnowledgeStore,
                 embedding_provider: EmbeddingProvider,
                 enable_meta_learning: bool = True):
        """
        Initialize knowledge retriever.

        Args:
            knowledge_store: KnowledgeStore instance
            embedding_provider: Embedding provider for semantic search
            enable_meta_learning: Enable relevance boosting based on usage patterns
        """
        self.knowledge_store = knowledge_store
        self.embedding_provider = embedding_provider
        self.enable_meta_learning = enable_meta_learning

    def search(self,
               query: str,
               task_type: Optional[str] = None,
               task_complexity: Optional[str] = None,
               top_k: int = 10,
               min_confidence: Optional[str] = None,
               domains: Optional[List[str]] = None) -> RetrievalContext:
        """
        Search knowledge brain for relevant information.

        Args:
            query: Search query
            task_type: Optional task type for meta-learning boost
            task_complexity: Optional task complexity
            top_k: Number of results to return
            min_confidence: Minimum confidence level filter
            domains: Optional domain filter

        Returns:
            RetrievalContext with search results and metadata
        """
        start_time = time.time()

        try:
            # Determine search strategy based on embedding provider tier
            tier_info = self.embedding_provider.get_tier_info()
            active_tier = tier_info['active_tier']

            if active_tier in ['lm_studio', 'tfidf']:
                # Use embedding-based search
                results = self._embedding_search(query, top_k * 2, domains, min_confidence)
                retrieval_method = f"embedding_{active_tier}"
            else:
                # Use FTS5 keyword search
                results = self._fts5_search(query, top_k * 2, domains, min_confidence)
                retrieval_method = "fts5"

            # Apply meta-learning boost if enabled
            if self.enable_meta_learning and task_type:
                results = self._apply_meta_learning_boost(results, task_type, task_complexity)

            # Sort by relevance and limit
            results.sort(key=lambda r: r.relevance_score, reverse=True)
            top_results = results[:top_k]

            processing_time = time.time() - start_time

            return RetrievalContext(
                query=query,
                results=top_results,
                total_results=len(results),
                retrieval_method=retrieval_method,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return RetrievalContext(
                query=query,
                results=[],
                total_results=0,
                retrieval_method="error",
                processing_time=time.time() - start_time
            )

    def _embedding_search(self,
                          query: str,
                          top_k: int,
                          domains: Optional[List[str]] = None,
                          min_confidence: Optional[str] = None) -> List[SearchResult]:
        """Search using embedding similarity."""
        try:
            # Generate query embedding
            query_result = self.embedding_provider.embed(query)

            if query_result.embedding is None:
                logger.warning("Query embedding failed, falling back to FTS5")
                return self._fts5_search(query, top_k, domains, min_confidence)

            # Get all knowledge entries with embeddings
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            # Build query with filters
            sql = """
                SELECT knowledge_id, content_json, domain, confidence_level,
                       tags_json, source_doc_id, embedding
                FROM knowledge_entries
                WHERE embedding IS NOT NULL
            """
            params = []

            if domains:
                placeholders = ','.join('?' * len(domains))
                sql += f" AND domain IN ({placeholders})"
                params.extend(domains)

            if min_confidence:
                sql += " AND confidence_level IN ('high', 'verified')"

            cursor = conn.execute(sql, params)

            # Compute similarities
            results = []
            from .embeddings import deserialize_embedding
            import json

            for row in cursor:
                try:
                    # Deserialize embedding
                    doc_embedding = deserialize_embedding(row['embedding'])

                    # Compute similarity
                    similarity = self.embedding_provider.cosine_similarity(
                        query_result.embedding,
                        doc_embedding
                    )

                    # Parse content
                    content = json.loads(row['content_json'])
                    tags = json.loads(row['tags_json']) if row['tags_json'] else []

                    result = SearchResult(
                        knowledge_id=row['knowledge_id'],
                        content=content,
                        domain=row['domain'],
                        confidence_level=row['confidence_level'],
                        relevance_score=similarity,
                        source_doc_id=row['source_doc_id'],
                        tags=tags,
                        reasoning="embedding_similarity"
                    )
                    results.append(result)

                except Exception as e:
                    logger.debug(f"Failed to process result: {e}")
                    continue

            conn.close()

            # Sort by similarity
            results.sort(key=lambda r: r.relevance_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []

    def _fts5_search(self,
                     query: str,
                     top_k: int,
                     domains: Optional[List[str]] = None,
                     min_confidence: Optional[str] = None) -> List[SearchResult]:
        """Search using SQLite FTS5."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            # FTS5 search with BM25 ranking
            sql = """
                SELECT k.knowledge_id, k.content_json, k.domain, k.confidence_level,
                       k.tags_json, k.source_doc_id, f.rank
                FROM knowledge_fts f
                JOIN knowledge_entries k ON f.knowledge_id = k.knowledge_id
                WHERE knowledge_fts MATCH ?
            """
            params = [query]

            if domains:
                placeholders = ','.join('?' * len(domains))
                sql += f" AND k.domain IN ({placeholders})"
                params.extend(domains)

            if min_confidence:
                sql += " AND k.confidence_level IN ('high', 'verified')"

            sql += " ORDER BY f.rank LIMIT ?"
            params.append(top_k)

            cursor = conn.execute(sql, params)

            results = []
            import json

            for row in cursor:
                try:
                    content = json.loads(row['content_json'])
                    tags = json.loads(row['tags_json']) if row['tags_json'] else []

                    # Convert FTS5 rank to relevance score (rank is negative)
                    relevance_score = min(1.0, -row['rank'] / 10.0)

                    result = SearchResult(
                        knowledge_id=row['knowledge_id'],
                        content=content,
                        domain=row['domain'],
                        confidence_level=row['confidence_level'],
                        relevance_score=relevance_score,
                        source_doc_id=row['source_doc_id'],
                        tags=tags,
                        reasoning="fts5_match"
                    )
                    results.append(result)

                except Exception as e:
                    logger.debug(f"Failed to process FTS5 result: {e}")
                    continue

            conn.close()
            return results

        except Exception as e:
            logger.error(f"FTS5 search failed: {e}")
            return []

    def _apply_meta_learning_boost(self,
                                   results: List[SearchResult],
                                   task_type: str,
                                   task_complexity: Optional[str] = None) -> List[SearchResult]:
        """
        Apply meta-learning boost to results.

        Boosts relevance of knowledge that was historically useful for this task type.
        """
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Get usage statistics for each knowledge entry
            for result in results:
                cursor = conn.execute("""
                    SELECT AVG(useful_score) as avg_usefulness, COUNT(*) as usage_count
                    FROM knowledge_usage
                    WHERE knowledge_id = ?
                    AND task_type = ?
                """, (result.knowledge_id, task_type))

                row = cursor.fetchone()
                if row and row[0] is not None:
                    avg_usefulness = row[0]
                    usage_count = row[1]

                    # Boost relevance based on historical usefulness
                    if usage_count >= 3:  # Minimum samples for reliable boost
                        boost_factor = 0.5 + (avg_usefulness * 0.5)  # 0.5 to 1.0
                        result.relevance_score = result.relevance_score * boost_factor
                        result.reasoning += f"+meta_boost({avg_usefulness:.2f})"

            conn.close()

        except Exception as e:
            logger.error(f"Meta-learning boost failed: {e}")

        return results

    def record_usage(self,
                     workflow_id: str,
                     knowledge_ids: List[str],
                     task_type: str,
                     task_complexity: Optional[str] = None,
                     useful_score: float = 0.5,
                     retrieval_method: Optional[str] = None):
        """
        Record knowledge usage for meta-learning.

        Args:
            workflow_id: Workflow that used the knowledge
            knowledge_ids: IDs of knowledge entries retrieved
            task_type: Type of task
            task_complexity: Optional complexity level
            useful_score: How useful was the knowledge (0.0-1.0)
            retrieval_method: Method used for retrieval
        """
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            for knowledge_id in knowledge_ids:
                conn.execute("""
                    INSERT INTO knowledge_usage
                    (workflow_id, knowledge_id, task_type, task_complexity,
                     useful_score, retrieval_method, recorded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (workflow_id, knowledge_id, task_type, task_complexity,
                      useful_score, retrieval_method, time.time()))

            conn.commit()
            conn.close()

            logger.debug(f"Recorded usage for {len(knowledge_ids)} knowledge entries")

        except Exception as e:
            logger.error(f"Failed to record usage: {e}")

    def get_usage_statistics(self,
                            task_type: Optional[str] = None,
                            days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for meta-learning insights."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            cutoff_time = time.time() - (days * 24 * 60 * 60)

            # Overall statistics
            cursor = conn.execute("""
                SELECT
                    COUNT(DISTINCT workflow_id) as workflow_count,
                    COUNT(DISTINCT knowledge_id) as knowledge_used_count,
                    AVG(useful_score) as avg_usefulness,
                    COUNT(*) as total_usages
                FROM knowledge_usage
                WHERE recorded_at > ?
            """ + (" AND task_type = ?" if task_type else ""),
            (cutoff_time, task_type) if task_type else (cutoff_time,))

            row = cursor.fetchone()
            stats = {
                'workflow_count': row[0] or 0,
                'knowledge_used_count': row[1] or 0,
                'avg_usefulness': row[2] or 0.0,
                'total_usages': row[3] or 0,
                'days': days,
                'task_type': task_type
            }

            # Most useful knowledge
            cursor = conn.execute("""
                SELECT k.knowledge_id, k.content_json, k.domain,
                       AVG(u.useful_score) as avg_score,
                       COUNT(*) as usage_count
                FROM knowledge_usage u
                JOIN knowledge_entries k ON u.knowledge_id = k.knowledge_id
                WHERE u.recorded_at > ?
            """ + (" AND u.task_type = ?" if task_type else "") + """
                GROUP BY k.knowledge_id
                HAVING usage_count >= 3
                ORDER BY avg_score DESC, usage_count DESC
                LIMIT 10
            """, (cutoff_time, task_type) if task_type else (cutoff_time,))

            import json
            most_useful = []
            for row in cursor:
                content = json.loads(row[1])
                most_useful.append({
                    'knowledge_id': row[0],
                    'concept': content.get('concept', 'Unknown'),
                    'domain': row[2],
                    'avg_usefulness': row[3],
                    'usage_count': row[4]
                })

            stats['most_useful'] = most_useful

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {}

    def build_augmented_context(self,
                                task_description: str,
                                task_type: Optional[str] = None,
                                task_complexity: Optional[str] = None,
                                max_concepts: int = 10) -> str:
        """
        Build augmented context for workflow by retrieving relevant knowledge.

        Args:
            task_description: Description of the task
            task_type: Optional task type
            task_complexity: Optional complexity level
            max_concepts: Maximum concepts to include

        Returns:
            Formatted context string for agent consumption
        """
        # Search for relevant knowledge
        retrieval_context = self.search(
            query=task_description,
            task_type=task_type,
            task_complexity=task_complexity,
            top_k=max_concepts
        )

        if not retrieval_context.results:
            return ""

        # Format as agent context
        context = "\n" + "=" * 70 + "\n"
        context += "DOMAIN KNOWLEDGE (from Knowledge Brain)\n"
        context += "=" * 70 + "\n\n"
        context += retrieval_context.to_agent_context(max_results=max_concepts)
        context += "\n" + "=" * 70 + "\n"

        return context

    def get_related_concepts(self, knowledge_id: str, max_depth: int = 2) -> List[SearchResult]:
        """
        Get concepts related to a given knowledge entry.

        Traverses relationship graph up to max_depth hops.
        Uses knowledge_relationships table (preferred) with fallback to related_entries_json.

        Args:
            knowledge_id: Starting knowledge ID
            max_depth: Maximum relationship depth to traverse

        Returns:
            List of related SearchResult objects
        """
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            visited = set()
            queue = [(knowledge_id, 0, 1.0)]  # (id, depth, inherited_strength)
            related = []

            # Check if knowledge_relationships table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)
            use_relationships_table = cursor.fetchone() is not None

            while queue:
                current_id, depth, inherited_strength = queue.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                # Get entry details
                cursor = conn.execute("""
                    SELECT knowledge_id, content_json, domain, confidence_level,
                           tags_json, source_doc_id, related_entries_json
                    FROM knowledge_entries
                    WHERE knowledge_id = ?
                """, (current_id,))

                row = cursor.fetchone()
                if row:
                    import json

                    if depth > 0:  # Don't include the starting node
                        try:
                            content = json.loads(row['content_json']) if row['content_json'] else {}
                        except (json.JSONDecodeError, TypeError):
                            content = {"concept": "Unknown", "definition": ""}

                        try:
                            tags = json.loads(row['tags_json']) if row['tags_json'] else []
                        except (json.JSONDecodeError, TypeError):
                            tags = []

                        # Relevance decays with depth and is weighted by relationship strength
                        relevance = inherited_strength * (1.0 - (depth * 0.2))

                        result = SearchResult(
                            knowledge_id=row['knowledge_id'],
                            content=content,
                            domain=row['domain'] or "unknown",
                            confidence_level=row['confidence_level'] or "medium",
                            relevance_score=relevance,
                            source_doc_id=row['source_doc_id'],
                            tags=tags,
                            reasoning=f"related_depth_{depth}"
                        )
                        related.append(result)

                    # Queue related entries - prefer relationships table
                    if depth < max_depth:
                        related_ids_with_strength = []

                        if use_relationships_table:
                            # Get relationships from knowledge_relationships table
                            rel_cursor = conn.execute("""
                                SELECT target_id, confidence
                                FROM knowledge_relationships
                                WHERE source_id = ?
                                UNION
                                SELECT source_id, confidence
                                FROM knowledge_relationships
                                WHERE target_id = ?
                            """, (current_id, current_id))

                            for rel_row in rel_cursor:
                                related_ids_with_strength.append(
                                    (rel_row['target_id'] or rel_row[0], rel_row['confidence'] or rel_row[1] or 0.5)
                                )

                        # Fallback to related_entries_json if no relationships found
                        if not related_ids_with_strength:
                            legacy_ids = json.loads(row['related_entries_json']) if row['related_entries_json'] else []
                            for related_id in legacy_ids:
                                related_ids_with_strength.append((related_id, 0.7))  # Default strength

                        # Add to queue
                        for related_id, strength in related_ids_with_strength:
                            if related_id not in visited:
                                queue.append((related_id, depth + 1, strength))

            conn.close()

            # Sort by relevance
            related.sort(key=lambda r: r.relevance_score, reverse=True)

            return related

        except Exception as e:
            logger.error(f"Failed to get related concepts: {e}")
            return []
