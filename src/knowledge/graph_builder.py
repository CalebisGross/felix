"""
Knowledge Graph Builder for Felix Knowledge Brain

Builds connections between concepts and entities across documents:
- Entity linking: Identify same entities across different documents
- Relationship discovery: Find related concepts via embedding similarity
- Cross-document synthesis: Merge knowledge from multiple sources
- Graph construction: Build bidirectional relationships

Uses existing KnowledgeStore.related_entries field for graph storage.
"""

import logging
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from src.memory.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """A node in the knowledge graph."""
    knowledge_id: str
    concept_name: str
    domain: str
    confidence_level: str
    related_ids: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class RelationshipEdge:
    """An edge connecting two knowledge nodes."""
    source_id: str
    target_id: str
    relationship_type: str  # related_to, prerequisite_of, part_of, etc.
    strength: float  # 0.0 to 1.0
    basis: str  # how relationship was discovered


class KnowledgeGraphBuilder:
    """
    Builds and maintains knowledge graph from extracted concepts.

    Uses multiple strategies to discover relationships:
    1. Explicit: Concepts list each other as related
    2. Embedding similarity: Semantically similar concepts
    3. Co-occurrence: Concepts from same document/section
    4. Entity linking: Same entity mentioned in different contexts
    """

    def __init__(self,
                 knowledge_store: KnowledgeStore,
                 embedding_provider=None,
                 similarity_threshold: float = 0.75,
                 cooccurrence_window: int = 5):
        """
        Initialize knowledge graph builder.

        Args:
            knowledge_store: KnowledgeStore instance
            embedding_provider: Optional embedding provider for similarity
            similarity_threshold: Minimum similarity for relationship (0.0-1.0)
            cooccurrence_window: Chunk distance for co-occurrence (chunks)
        """
        self.knowledge_store = knowledge_store
        self.embedding_provider = embedding_provider
        self.similarity_threshold = similarity_threshold
        self.cooccurrence_window = cooccurrence_window

    def _normalize_concept_name(self, name: str) -> str:
        """
        Normalize concept name for matching.

        Strips markdown formatting and normalizes whitespace/case.
        This fixes the bug where concept names stored with markdown
        (e.g., **Confidence Scoring**) wouldn't match related_concepts
        stored without markdown (e.g., Confidence Scoring).
        """
        if not name:
            return ""
        # Strip markdown formatting: **bold**, *italic*, __underline__, `code`
        name = re.sub(r'\*\*|\*|__|_|`', '', name)
        # Normalize whitespace and lowercase
        return name.lower().strip()

    def build_graph_for_document(self, document_id: str) -> Dict[str, Any]:
        """
        Build knowledge graph for concepts from a single document.

        Args:
            document_id: Document ID to process

        Returns:
            Dict with statistics: relationships_created, concepts_processed
        """
        logger.info(f"Building knowledge graph for document: {document_id}")

        try:
            # Get all concepts from this document
            concepts = self._get_document_concepts(document_id)

            if not concepts:
                logger.warning(f"No concepts found for document: {document_id}")
                return {'relationships_created': 0, 'concepts_processed': 0}

            relationships_created = 0

            # Strategy 1: Explicit relationships (concepts mention each other)
            explicit_rels = self._discover_explicit_relationships(concepts)
            relationships_created += self._store_relationships(explicit_rels)

            # Strategy 2: Embedding similarity (if available)
            if self.embedding_provider:
                similarity_rels = self._discover_similarity_relationships(concepts)
                relationships_created += self._store_relationships(similarity_rels)

            # Strategy 3: Co-occurrence (same document/section)
            cooccurrence_rels = self._discover_cooccurrence_relationships(concepts)
            relationships_created += self._store_relationships(cooccurrence_rels)

            logger.info(f"Created {relationships_created} relationships for {len(concepts)} concepts")

            return {
                'relationships_created': relationships_created,
                'concepts_processed': len(concepts)
            }

        except Exception as e:
            logger.error(f"Failed to build graph for document {document_id}: {e}")
            return {'relationships_created': 0, 'concepts_processed': 0, 'error': str(e)}

    def build_global_graph(self, max_documents: Optional[int] = None) -> Dict[str, Any]:
        """
        Build knowledge graph across all documents.

        Args:
            max_documents: Optional limit on documents to process

        Returns:
            Dict with statistics
        """
        logger.info("Building global knowledge graph...")

        try:
            # Get all unique document IDs
            doc_ids = self._get_all_document_ids()

            if max_documents:
                doc_ids = doc_ids[:max_documents]

            total_relationships = 0
            total_concepts = 0

            for doc_id in doc_ids:
                result = self.build_graph_for_document(doc_id)
                total_relationships += result.get('relationships_created', 0)
                total_concepts += result.get('concepts_processed', 0)

            # Cross-document entity linking
            entity_links = self.link_entities_across_documents()
            total_relationships += entity_links

            # Find concept duplicates and merge
            merged = self.merge_duplicate_concepts()

            logger.info(f"Global graph complete: {total_relationships} relationships, "
                       f"{total_concepts} concepts, {merged} merged")

            return {
                'total_relationships': total_relationships,
                'total_concepts': total_concepts,
                'documents_processed': len(doc_ids),
                'entities_linked': entity_links,
                'concepts_merged': merged
            }

        except Exception as e:
            logger.error(f"Failed to build global graph: {e}")
            return {'error': str(e)}

    def _get_document_concepts(self, document_id: str) -> List[ConceptNode]:
        """Get all concepts from a document."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT knowledge_id, content_json, domain, confidence_level,
                       related_entries_json, embedding
                FROM knowledge_entries
                WHERE source_doc_id = ?
                AND json_extract(content_json, '$.concept') IS NOT NULL
            """, (document_id,))

            concepts = []
            for row in cursor:
                import json

                content = json.loads(row['content_json']) if row['content_json'] else {}
                related_entries = json.loads(row['related_entries_json']) if row['related_entries_json'] else []

                # Deserialize embedding if available
                embedding = None
                if row['embedding']:
                    from .embeddings import deserialize_embedding
                    try:
                        embedding = deserialize_embedding(row['embedding'])
                    except:
                        pass

                concept = ConceptNode(
                    knowledge_id=row['knowledge_id'],
                    concept_name=content.get('concept', 'Unknown'),
                    domain=row['domain'],
                    confidence_level=row['confidence_level'],
                    related_ids=related_entries,
                    embedding=embedding
                )
                concepts.append(concept)

            conn.close()
            return concepts

        except Exception as e:
            logger.error(f"Failed to get concepts for {document_id}: {e}")
            return []

    def _get_all_document_ids(self) -> List[str]:
        """Get all unique document IDs with concepts."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.execute("""
                SELECT DISTINCT source_doc_id
                FROM knowledge_entries
                WHERE source_doc_id IS NOT NULL
            """)
            doc_ids = [row[0] for row in cursor]
            conn.close()
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to get document IDs: {e}")
            return []

    def _discover_explicit_relationships(self, concepts: List[ConceptNode]) -> List[RelationshipEdge]:
        """
        Discover relationships explicitly mentioned in concept metadata.

        Concepts list related_concepts in their content.

        Uses normalized names for matching to handle markdown formatting
        differences between stored concept names and related_concepts lists.
        """
        relationships = []

        # Build name-to-id mapping using NORMALIZED names
        # This fixes the bug where **Confidence Scoring** wouldn't match Confidence Scoring
        name_to_id = {}
        for c in concepts:
            normalized = self._normalize_concept_name(c.concept_name)
            if normalized:  # Skip empty names
                name_to_id[normalized] = c.knowledge_id

        for concept in concepts:
            # Check if concept has related_concepts in its content
            try:
                conn = sqlite3.connect(self.knowledge_store.storage_path)
                cursor = conn.execute("""
                    SELECT content_json FROM knowledge_entries
                    WHERE knowledge_id = ?
                """, (concept.knowledge_id,))

                row = cursor.fetchone()
                if row:
                    import json
                    content = json.loads(row[0])
                    related_concepts = content.get('related_concepts', [])

                    # Match related concept names to IDs using NORMALIZED names
                    for related_name in related_concepts:
                        normalized_related = self._normalize_concept_name(related_name)
                        related_id = name_to_id.get(normalized_related)
                        if related_id and related_id != concept.knowledge_id:
                            relationships.append(RelationshipEdge(
                                source_id=concept.knowledge_id,
                                target_id=related_id,
                                relationship_type='related_to',
                                strength=0.9,  # High confidence (explicit mention)
                                basis='explicit_mention'
                            ))

                conn.close()

            except Exception as e:
                logger.warning(f"Failed to extract explicit relationships for {concept.knowledge_id}: {e}")
                continue

        return relationships

    def _discover_similarity_relationships(self, concepts: List[ConceptNode]) -> List[RelationshipEdge]:
        """
        Discover relationships via embedding similarity.

        Requires concepts to have embeddings.
        """
        relationships = []

        # Filter concepts with embeddings
        concepts_with_embeddings = [c for c in concepts if c.embedding is not None]

        if len(concepts_with_embeddings) < 2:
            return relationships

        # Compare each pair
        for i, concept_a in enumerate(concepts_with_embeddings):
            for concept_b in concepts_with_embeddings[i + 1:]:
                # Compute cosine similarity
                similarity = self.embedding_provider.cosine_similarity(
                    concept_a.embedding,
                    concept_b.embedding
                )

                if similarity >= self.similarity_threshold:
                    relationships.append(RelationshipEdge(
                        source_id=concept_a.knowledge_id,
                        target_id=concept_b.knowledge_id,
                        relationship_type='similar_to',
                        strength=similarity,
                        basis='embedding_similarity'
                    ))

        return relationships

    def _discover_cooccurrence_relationships(self, concepts: List[ConceptNode]) -> List[RelationshipEdge]:
        """
        Discover relationships via co-occurrence in the SAME chunk only.

        This is a high-precision approach: concepts mentioned in the same chunk
        have the strongest semantic connection. The previous O(n²) algorithm
        connected all concepts within 5 chunks, creating 272k+ noise relationships.

        Now we only connect concepts that appear in the SAME chunk, which provides
        actual semantic signal rather than proximity noise.
        """
        relationships = []

        try:
            # Group concepts by chunk_index
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            concept_chunks = defaultdict(list)
            for concept in concepts:
                cursor = conn.execute("""
                    SELECT chunk_index FROM knowledge_entries
                    WHERE knowledge_id = ?
                """, (concept.knowledge_id,))
                row = cursor.fetchone()
                if row and row[0] is not None:
                    chunk_idx = row[0]
                    concept_chunks[chunk_idx].append(concept.knowledge_id)

            conn.close()

            # Only connect concepts in the SAME chunk (high precision)
            # This eliminates the O(n²) explosion from connecting across chunks
            for chunk_idx, concept_ids in concept_chunks.items():
                # Skip chunks with only one concept (no relationships to create)
                if len(concept_ids) < 2:
                    continue

                # Connect all concepts within this chunk
                for i, id_a in enumerate(concept_ids):
                    for id_b in concept_ids[i + 1:]:
                        relationships.append(RelationshipEdge(
                            source_id=id_a,
                            target_id=id_b,
                            relationship_type='cooccurs_with',
                            strength=0.7,  # Same chunk = strong semantic signal
                            basis='same_chunk'
                        ))

        except Exception as e:
            logger.error(f"Failed to discover co-occurrence relationships: {e}")

        return relationships

    def _store_relationships(self, relationships: List[RelationshipEdge]) -> int:
        """
        Store relationships in knowledge_relationships table.
        """
        if not relationships:
            return 0

        stored_count = 0

        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            cursor = conn.cursor()

            # Batch insert all relationships
            for rel in relationships:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO knowledge_relationships
                        (source_id, target_id, relationship_type, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (rel.source_id, rel.target_id, rel.relationship_type, rel.strength))
                    stored_count += 1
                except sqlite3.Error:
                    pass  # Skip duplicates silently

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store relationships: {e}")

        return stored_count

    def link_entities_across_documents(self) -> int:
        """
        Link same entities mentioned in different documents.

        Returns:
            Number of entity links created
        """
        logger.info("Linking entities across documents...")

        try:
            # Get all entity knowledge entries
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT knowledge_id, content_json, source_doc_id
                FROM knowledge_entries
                WHERE json_extract(content_json, '$.entity') IS NOT NULL
            """)

            # Group entities by name
            entity_groups = defaultdict(list)

            for row in cursor:
                import json
                content = json.loads(row['content_json'])
                entity_name = content.get('entity', '').lower()
                entity_groups[entity_name].append({
                    'knowledge_id': row['knowledge_id'],
                    'source_doc_id': row['source_doc_id'],
                    'entity_type': content.get('type', 'unknown')
                })

            conn.close()

            # Link entities with same name from different documents
            links_created = 0

            for entity_name, instances in entity_groups.items():
                if len(instances) > 1:
                    # Link all instances together
                    for i, instance_a in enumerate(instances):
                        for instance_b in instances[i + 1:]:
                            # Only link if from different documents
                            if instance_a['source_doc_id'] != instance_b['source_doc_id']:
                                try:
                                    self.knowledge_store.add_related_entry(
                                        instance_a['knowledge_id'],
                                        instance_b['knowledge_id']
                                    )
                                    links_created += 1
                                except:
                                    pass

            logger.info(f"Created {links_created} cross-document entity links")
            return links_created

        except Exception as e:
            logger.error(f"Failed to link entities: {e}")
            return 0

    def merge_duplicate_concepts(self, similarity_threshold: float = 0.95) -> int:
        """
        Find and merge duplicate concepts across documents.

        Concepts with very high similarity (>0.95) are likely duplicates.

        Returns:
            Number of concepts merged
        """
        logger.info("Merging duplicate concepts...")

        if not self.embedding_provider:
            logger.info("No embedding provider available, skipping merge")
            return 0

        try:
            # Get all concepts with embeddings
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT knowledge_id, content_json, embedding, access_count
                FROM knowledge_entries
                WHERE embedding IS NOT NULL
                AND json_extract(content_json, '$.concept') IS NOT NULL
            """)

            concepts = []
            for row in cursor:
                from .embeddings import deserialize_embedding
                try:
                    embedding = deserialize_embedding(row['embedding'])
                    import json
                    content = json.loads(row['content_json'])
                    concepts.append({
                        'knowledge_id': row['knowledge_id'],
                        'concept_name': content.get('concept', ''),
                        'embedding': embedding,
                        'access_count': row['access_count'] or 0
                    })
                except:
                    continue

            conn.close()

            # Find duplicates
            merged_count = 0
            processed = set()

            for i, concept_a in enumerate(concepts):
                if concept_a['knowledge_id'] in processed:
                    continue

                for concept_b in concepts[i + 1:]:
                    if concept_b['knowledge_id'] in processed:
                        continue

                    # Check similarity
                    similarity = self.embedding_provider.cosine_similarity(
                        concept_a['embedding'],
                        concept_b['embedding']
                    )

                    if similarity >= similarity_threshold:
                        # Merge: keep the one with higher access count
                        if concept_a['access_count'] >= concept_b['access_count']:
                            keep_id = concept_a['knowledge_id']
                            merge_id = concept_b['knowledge_id']
                        else:
                            keep_id = concept_b['knowledge_id']
                            merge_id = concept_a['knowledge_id']

                        logger.info(f"Merging duplicate: {concept_b['concept_name']} -> {concept_a['concept_name']} "
                                   f"(similarity: {similarity:.3f})")

                        # Link them as related
                        try:
                            self.knowledge_store.add_related_entry(keep_id, merge_id)
                            processed.add(merge_id)
                            merged_count += 1
                        except:
                            pass

            logger.info(f"Merged {merged_count} duplicate concepts")
            return merged_count

        except Exception as e:
            logger.error(f"Failed to merge duplicates: {e}")
            return 0

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        try:
            conn = sqlite3.connect(self.knowledge_store.storage_path)

            # Total nodes (concepts + entities)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE source_doc_id IS NOT NULL
            """)
            total_nodes = cursor.fetchone()[0]

            # Total edges (relationships)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE json_array_length(related_entries_json) > 0
            """)
            nodes_with_edges = cursor.fetchone()[0]

            # Average degree (connections per node)
            cursor = conn.execute("""
                SELECT AVG(json_array_length(related_entries_json))
                FROM knowledge_entries
                WHERE source_doc_id IS NOT NULL
            """)
            avg_degree = cursor.fetchone()[0] or 0

            # Documents covered
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT source_doc_id)
                FROM knowledge_entries
                WHERE source_doc_id IS NOT NULL
            """)
            documents_covered = cursor.fetchone()[0]

            conn.close()

            return {
                'total_nodes': total_nodes,
                'nodes_with_relationships': nodes_with_edges,
                'average_degree': round(avg_degree, 2),
                'documents_covered': documents_covered
            }

        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}
