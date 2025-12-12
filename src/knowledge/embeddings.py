"""
Tiered Embeddings System for Felix Knowledge Brain

Three-tier approach for maximum reliability with zero external dependencies:
1. Tier 1 (Best): LM Studio embeddings - High quality semantic search
2. Tier 2 (Good): TF-IDF embeddings - Pure Python keyword-based semantic matching
3. Tier 3 (Always Works): SQLite FTS5 - Built-in full-text search

System automatically selects best available tier and provides unified interface.
"""

import logging
import numpy as np
import sqlite3
import time
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EmbeddingTier(Enum):
    """Available embedding tiers."""
    LM_STUDIO = "lm_studio"  # Best quality
    TFIDF = "tfidf"  # Good quality
    FTS5 = "fts5"  # Always works


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    text: str
    embedding: Optional[List[float]]
    tier_used: EmbeddingTier
    processing_time: float
    error: Optional[str] = None


class LMStudioEmbedder:
    """
    Tier 1: LM Studio embeddings provider.

    Uses existing LM Studio connection to generate high-quality embeddings.
    """

    def __init__(self, lm_studio_client):
        """
        Initialize LM Studio embedder.

        Args:
            lm_studio_client: Existing LMStudioClient instance
        """
        self.client = lm_studio_client
        self.available = False  # Set default before availability check to avoid circular dependency
        self.embedding_dim = 768  # Most embedding models use 768 dimensions
        self.available = self._check_availability()  # Now check actual availability

    def _check_availability(self) -> bool:
        """Check if LM Studio has an embedding model loaded."""
        if self.client is None:
            return False

        try:
            # Use quick test with 5-second timeout to avoid blocking GUI
            # This bypasses the main OpenAI client which has 120s timeout
            return self.client.test_embedding_availability(timeout=5.0)
        except Exception as e:
            logger.info(f"LM Studio embeddings not available: {e}")
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using LM Studio.

        Args:
            text: Input text

        Returns:
            List of floats (embedding vector) or None if failed
        """
        if not self.available:
            return None

        try:
            # LM Studio typically supports embeddings via /v1/embeddings endpoint
            # This is a placeholder - actual implementation depends on LM Studio API
            response = self.client.generate_embedding(text)
            return response
        except Exception as e:
            logger.warning(f"LM Studio embedding failed: {e}")
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class TFIDFEmbedder:
    """
    Tier 2: TF-IDF embeddings provider.

    Pure Python implementation using numpy for keyword-based semantic matching.
    """

    def __init__(self, max_features: int = 768):
        """
        Initialize TF-IDF embedder.

        Args:
            max_features: Maximum number of features (dimensions)
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}
        self.document_count = 0
        self.fitted = False

    def fit(self, documents: List[str]):
        """
        Fit TF-IDF model on a collection of documents.

        Args:
            documents: List of text documents
        """
        if not documents:
            logger.warning("No documents provided for TF-IDF fitting")
            return

        # Build vocabulary and document frequency
        doc_frequency = {}

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                doc_frequency[word] = doc_frequency.get(word, 0) + 1

        # Select most common words up to max_features
        sorted_words = sorted(doc_frequency.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.max_features])}

        # Compute IDF values
        self.document_count = len(documents)
        for word, df in doc_frequency.items():
            if word in self.vocabulary:
                self.idf_values[word] = np.log((self.document_count + 1) / (df + 1)) + 1

        self.fitted = True
        logger.info(f"TF-IDF fitted: {len(self.vocabulary)} features, {self.document_count} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (split on whitespace and punctuation)."""
        import re
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 2]  # Filter short words

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Generate TF-IDF embedding for text.

        Args:
            text: Input text

        Returns:
            List of floats (TF-IDF vector)
        """
        if not self.fitted:
            logger.warning("TF-IDF not fitted yet, returning None")
            return None

        # Compute term frequencies
        words = self._tokenize(text)
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        # Normalize by document length
        total_words = len(words)
        if total_words > 0:
            for word in tf:
                tf[word] = tf[word] / total_words

        # Build TF-IDF vector
        vector = np.zeros(len(self.vocabulary))
        for word, tf_val in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                idf_val = self.idf_values.get(word, 1.0)
                vector[idx] = tf_val * idf_val

        # Normalize to unit length (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2))  # Already normalized, so just dot product


class FTS5Searcher:
    """
    Tier 3: SQLite FTS5 full-text search.

    Keyword-based search using SQLite's built-in FTS5 with BM25 ranking.
    No embeddings needed, always available.
    """

    def __init__(self, db_path: str):
        """
        Initialize FTS5 searcher.

        Args:
            db_path: Path to SQLite database with FTS5 tables
        """
        self.db_path = db_path

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using FTS5.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching records with BM25 scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # FTS5 search with BM25 ranking
            cursor = conn.execute("""
                SELECT knowledge_id, rank
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, top_k))

            results = []
            for row in cursor:
                results.append({
                    'knowledge_id': row['knowledge_id'],
                    'score': -row['rank']  # FTS5 rank is negative
                })

            conn.close()
            return results

        except sqlite3.Error as e:
            logger.error(f"FTS5 search failed: {e}")
            return []


class EmbeddingProvider:
    """
    Unified embedding provider with automatic tier selection.

    Tries Tier 1 (LM Studio) → Tier 2 (TF-IDF) → Tier 3 (FTS5) in order.
    """

    def __init__(self,
                 lm_studio_client=None,
                 db_path: str = "felix_knowledge.db",
                 preferred_tier: Optional[EmbeddingTier] = None):
        """
        Initialize embedding provider.

        Args:
            lm_studio_client: Optional LMStudioClient instance
            db_path: Path to knowledge database
            preferred_tier: Force specific tier (for testing)
        """
        self.db_path = db_path

        # Initialize all tiers
        self.lm_studio_embedder = LMStudioEmbedder(lm_studio_client) if lm_studio_client else None
        self.tfidf_embedder = TFIDFEmbedder()
        self.fts5_searcher = FTS5Searcher(db_path)

        # Determine active tier
        if preferred_tier:
            self.active_tier = preferred_tier
        else:
            self.active_tier = self._select_best_tier()

        logger.info(f"Embedding provider initialized: Tier = {self.active_tier.value}")

    def _select_best_tier(self) -> EmbeddingTier:
        """Select best available tier."""
        # Try Tier 1: LM Studio
        if self.lm_studio_embedder and self.lm_studio_embedder.available:
            return EmbeddingTier.LM_STUDIO

        # Try Tier 2: TF-IDF (need to fit first)
        # Will be available after fit_tfidf() is called
        if self.tfidf_embedder.fitted:
            return EmbeddingTier.TFIDF

        # Tier 3: FTS5 (always available)
        return EmbeddingTier.FTS5

    def fit_tfidf(self, documents: List[str]):
        """
        Fit TF-IDF model on document collection.

        Should be called during initial ingestion to enable Tier 2.

        Args:
            documents: List of document contents
        """
        if not documents:
            logger.warning("No documents provided for TF-IDF fitting")
            return

        logger.info(f"Fitting TF-IDF on {len(documents)} documents...")
        self.tfidf_embedder.fit(documents)

        # Re-select tier (TF-IDF might now be available)
        if self.active_tier == EmbeddingTier.FTS5 and self.tfidf_embedder.fitted:
            self.active_tier = EmbeddingTier.TFIDF
            logger.info("Upgraded to Tier 2 (TF-IDF) after fitting")

    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for text using best available tier.

        Args:
            text: Input text

        Returns:
            EmbeddingResult with embedding vector and metadata
        """
        start_time = time.time()

        try:
            if self.active_tier == EmbeddingTier.LM_STUDIO:
                embedding = self.lm_studio_embedder.embed(text)
                if embedding is not None:
                    return EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        tier_used=EmbeddingTier.LM_STUDIO,
                        processing_time=time.time() - start_time
                    )
                # Fallback to TF-IDF if LM Studio fails
                logger.warning("LM Studio embedding failed, falling back to TF-IDF")
                self.active_tier = EmbeddingTier.TFIDF

            if self.active_tier == EmbeddingTier.TFIDF:
                embedding = self.tfidf_embedder.embed(text)
                if embedding is not None:
                    return EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        tier_used=EmbeddingTier.TFIDF,
                        processing_time=time.time() - start_time
                    )
                # Fallback to FTS5 if TF-IDF fails
                logger.warning("TF-IDF embedding failed, falling back to FTS5")
                self.active_tier = EmbeddingTier.FTS5

            # Tier 3: FTS5 doesn't need embeddings
            return EmbeddingResult(
                text=text,
                embedding=None,
                tier_used=EmbeddingTier.FTS5,
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return EmbeddingResult(
                text=text,
                embedding=None,
                tier_used=self.active_tier,
                processing_time=time.time() - start_time,
                error=str(e)
            )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of EmbeddingResult objects
        """
        return [self.embed(text) for text in texts]

    def search(self,
               query: str,
               embeddings_db: List[Tuple[str, Optional[List[float]]]],
               top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar texts.

        Args:
            query: Search query
            embeddings_db: List of (id, embedding) tuples from database
            top_k: Number of results to return

        Returns:
            List of (id, similarity_score) tuples, sorted by score descending
        """
        if self.active_tier == EmbeddingTier.FTS5:
            # Use FTS5 search directly
            results = self.fts5_searcher.search(query, top_k)
            return [(r['knowledge_id'], r['score']) for r in results]

        # Tier 1 or 2: Compute similarity with embeddings
        query_result = self.embed(query)
        if query_result.embedding is None:
            logger.warning("Query embedding failed, falling back to FTS5")
            results = self.fts5_searcher.search(query, top_k)
            return [(r['knowledge_id'], r['score']) for r in results]

        query_vec = np.array(query_result.embedding)

        # Compute cosine similarity with all stored embeddings
        similarities = []
        for doc_id, doc_embedding in embeddings_db:
            if doc_embedding is None:
                continue

            doc_vec = np.array(doc_embedding)

            # Cosine similarity (assume vectors are normalized)
            similarity = float(np.dot(query_vec, doc_vec))
            similarities.append((doc_id, similarity))

        # Sort by similarity descending and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        v1_normalized = v1 / norm1
        v2_normalized = v2 / norm2

        return float(np.dot(v1_normalized, v2_normalized))

    def get_tier_info(self) -> Dict[str, Any]:
        """Get information about current tier and availability."""
        return {
            'active_tier': self.active_tier.value,
            'tiers_available': {
                'lm_studio': self.lm_studio_embedder.available if self.lm_studio_embedder else False,
                'tfidf': self.tfidf_embedder.fitted,
                'fts5': True  # Always available
            },
            'embedding_dim': {
                'lm_studio': 768 if self.lm_studio_embedder else None,
                'tfidf': len(self.tfidf_embedder.vocabulary) if self.tfidf_embedder.fitted else None,
                'fts5': None  # No embeddings
            }
        }


# Utility functions

def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for database storage."""
    return np.array(embedding, dtype=np.float32).tobytes()


def deserialize_embedding(embedding_bytes: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
