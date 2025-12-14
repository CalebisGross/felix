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
import threading
import time
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

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


@dataclass
class TierRecoveryConfig:
    """Configuration for embedding tier recovery."""
    mode: str = "auto"  # "auto" or "manual"
    check_interval: float = 60.0  # Seconds between automatic recovery checks
    check_timeout: float = 5.0  # Timeout for availability checks
    max_recovery_attempts: int = 3  # Max consecutive failures before pausing


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
    Automatically fits with default corpus on initialization for immediate availability.
    """

    # Default corpus path relative to this file
    DEFAULT_CORPUS_PATH = Path(__file__).parent.parent.parent / "data" / "default_tfidf_corpus.txt"

    def __init__(self, max_features: int = 768, default_corpus_path: Optional[Union[str, Path]] = None):
        """
        Initialize TF-IDF embedder.

        Args:
            max_features: Maximum number of features (dimensions)
            default_corpus_path: Path to default corpus file. If None, uses built-in default.
                                 Set to False to skip default corpus fitting.
        """
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf_values: Dict[str, float] = {}
        self.document_count = 0
        self.fitted = False
        # Track document frequencies for incremental updates
        self._doc_frequency: Dict[str, int] = {}

        # Auto-fit with default corpus if available
        if default_corpus_path is not False:
            corpus_path = Path(default_corpus_path) if default_corpus_path else self.DEFAULT_CORPUS_PATH
            if corpus_path.exists():
                self._fit_default_corpus(corpus_path)

    def _fit_default_corpus(self, corpus_path: Path):
        """
        Fit TF-IDF with default corpus for immediate availability.

        Args:
            corpus_path: Path to default corpus file (one document per line)
        """
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                # Filter out empty lines and comments
                documents = [
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
            if documents:
                self.fit(documents)
                logger.info(f"TF-IDF initialized with default corpus ({len(documents)} documents, "
                           f"{len(self.vocabulary)} features)")
        except Exception as e:
            logger.warning(f"Failed to load default TF-IDF corpus: {e}")

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
        doc_frequency: Dict[str, int] = {}

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                doc_frequency[word] = doc_frequency.get(word, 0) + 1

        # Store doc_frequency for incremental updates
        self._doc_frequency = doc_frequency

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

    def update_vocabulary(self, documents: List[str]):
        """
        Incrementally update vocabulary with new documents.

        This method merges new document frequencies with existing ones,
        allowing the TF-IDF model to grow as documents are ingested.

        Args:
            documents: List of new text documents to incorporate
        """
        if not documents:
            return

        if not self.fitted:
            # Not fitted yet, do a full fit instead
            self.fit(documents)
            return

        # Merge new document frequencies with existing
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                self._doc_frequency[word] = self._doc_frequency.get(word, 0) + 1

        # Update document count
        self.document_count += len(documents)

        # Re-sort vocabulary by frequency and trim to max_features
        sorted_words = sorted(self._doc_frequency.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.max_features])}

        # Recompute IDF values for all words in vocabulary
        self.idf_values = {}
        for word, df in self._doc_frequency.items():
            if word in self.vocabulary:
                self.idf_values[word] = np.log((self.document_count + 1) / (df + 1)) + 1

        logger.info(f"TF-IDF vocabulary updated: {len(self.vocabulary)} features, {self.document_count} documents")

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

            # Escape FTS5 special characters to prevent syntax errors
            # FTS5 treats ?, *, ", (, ), etc. as operators
            escaped_query = query.replace('"', '""')
            escaped_query = f'"{escaped_query}"'  # Wrap in quotes for literal search

            # FTS5 search with BM25 ranking
            cursor = conn.execute("""
                SELECT knowledge_id, rank
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (escaped_query, top_k))

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


class TierRecoveryManager:
    """
    Manages automatic and manual recovery to higher embedding tiers.

    Follows the pattern from knowledge_daemon.py for background thread management
    and circuit_breaker.py for thread-safe state transitions.
    """

    def __init__(self, config: Optional[TierRecoveryConfig] = None):
        """
        Initialize tier recovery manager.

        Args:
            config: Recovery configuration (uses defaults if None)
        """
        self.config = config or TierRecoveryConfig()

        # State tracking
        self._current_tier: EmbeddingTier = EmbeddingTier.FTS5
        self._optimal_tier: EmbeddingTier = EmbeddingTier.LM_STUDIO
        self._consecutive_failures: int = 0
        self._last_check_time: Optional[float] = None
        self._last_recovery_time: Optional[float] = None
        self._recovery_paused: bool = False

        # Thread safety (from circuit_breaker.py pattern)
        self._lock = threading.Lock()

        # Background thread management (from knowledge_daemon.py pattern)
        self._stop_event = threading.Event()
        self._recovery_thread: Optional[threading.Thread] = None
        self._running = False

        # Tier availability checker (set by EmbeddingProvider)
        self._tier_checker: Optional[Callable[[EmbeddingTier], bool]] = None

        # Recovery callback (notifies EmbeddingProvider to upgrade)
        self._on_tier_available: Optional[Callable[[EmbeddingTier], None]] = None

    def start(self) -> None:
        """Start automatic recovery checking (only in auto mode)."""
        if self.config.mode != "auto":
            logger.info("Tier recovery in manual mode, background checking disabled")
            return

        if self._running:
            logger.warning("Tier recovery manager already running")
            return

        self._running = True
        self._stop_event.clear()
        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            name="EmbeddingTierRecovery",
            daemon=True
        )
        self._recovery_thread.start()
        logger.info(f"Tier recovery manager started (interval={self.config.check_interval}s)")

    def stop(self) -> None:
        """Stop automatic recovery checking."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._recovery_thread:
            self._recovery_thread.join(timeout=10.0)
            if self._recovery_thread.is_alive():
                logger.warning("Tier recovery thread still alive after timeout")
            self._recovery_thread = None

        logger.info("Tier recovery manager stopped")

    def _recovery_loop(self) -> None:
        """Background loop for automatic tier recovery checks."""
        while self._running and not self._stop_event.is_set():
            # Wait for interval (interruptible)
            if self._stop_event.wait(timeout=self.config.check_interval):
                break

            if not self._running:
                break

            # Only check if we're in a degraded state
            with self._lock:
                if self._current_tier == self._optimal_tier:
                    continue  # Already at optimal tier
                if self._recovery_paused:
                    continue  # Paused after too many failures

            # Try to upgrade
            self._try_recovery()

        logger.debug("Tier recovery loop exited")

    def try_upgrade_tier(self) -> Optional[EmbeddingTier]:
        """
        Manually trigger tier upgrade check.

        Returns:
            The new tier if upgraded, None if no upgrade occurred
        """
        # Reset pause state on manual attempt
        with self._lock:
            self._recovery_paused = False
            self._consecutive_failures = 0

        return self._try_recovery()

    def _try_recovery(self) -> Optional[EmbeddingTier]:
        """
        Attempt to recover to a higher tier.

        Returns:
            The new tier if upgraded, None otherwise
        """
        if self._tier_checker is None:
            logger.warning("No tier checker configured")
            return None

        with self._lock:
            self._last_check_time = time.time()
            current = self._current_tier

            # Define tier priority (highest to lowest)
            tier_priority = [EmbeddingTier.LM_STUDIO, EmbeddingTier.TFIDF, EmbeddingTier.FTS5]
            current_idx = tier_priority.index(current)

            # Check each tier from highest to current (exclusive)
            for tier in tier_priority[:current_idx]:
                try:
                    if self._tier_checker(tier):
                        # Tier is available! Upgrade
                        old_tier = self._current_tier
                        self._current_tier = tier
                        self._consecutive_failures = 0
                        self._last_recovery_time = time.time()

                        logger.info(f"Tier recovery successful: {old_tier.value} -> {tier.value}")

                        # Notify EmbeddingProvider (outside lock to avoid deadlock)
                        callback = self._on_tier_available
                        if callback:
                            # Release lock before callback
                            self._lock.release()
                            try:
                                callback(tier)
                            finally:
                                self._lock.acquire()

                        return tier
                except Exception as e:
                    logger.debug(f"Tier {tier.value} check failed: {e}")

            # No tier recovered
            self._consecutive_failures += 1

            # Check if we should pause automatic recovery
            if self._consecutive_failures >= self.config.max_recovery_attempts:
                self._recovery_paused = True
                logger.info(f"Pausing recovery after {self._consecutive_failures} consecutive failures")

            return None

    def set_current_tier(self, tier: EmbeddingTier) -> None:
        """Update current tier (called by EmbeddingProvider on downgrade)."""
        with self._lock:
            old_tier = self._current_tier
            self._current_tier = tier
            if tier != old_tier:
                logger.debug(f"Recovery manager notified of tier change: {old_tier.value} -> {tier.value}")
                # Reset failure count when tier changes (fresh start for recovery)
                if tier.value > old_tier.value:  # Degraded
                    self._consecutive_failures = 0
                    self._recovery_paused = False

    def get_status(self) -> Dict[str, Any]:
        """Get recovery manager status."""
        with self._lock:
            return {
                "mode": self.config.mode,
                "running": self._running,
                "current_tier": self._current_tier.value,
                "optimal_tier": self._optimal_tier.value,
                "is_degraded": self._current_tier != self._optimal_tier,
                "consecutive_failures": self._consecutive_failures,
                "recovery_paused": self._recovery_paused,
                "last_check_time": self._last_check_time,
                "last_recovery_time": self._last_recovery_time,
                "check_interval": self.config.check_interval
            }


class EmbeddingProvider:
    """
    Unified embedding provider with automatic tier selection.

    Tries Tier 1 (LM Studio) → Tier 2 (TF-IDF) → Tier 3 (FTS5) in order.
    """

    def __init__(self,
                 lm_studio_client=None,
                 db_path: str = "felix_knowledge.db",
                 preferred_tier: Optional[EmbeddingTier] = None,
                 recovery_config: Optional[TierRecoveryConfig] = None):
        """
        Initialize embedding provider.

        Args:
            lm_studio_client: Optional LMStudioClient instance
            db_path: Path to knowledge database
            preferred_tier: Force specific tier (for testing)
            recovery_config: Configuration for tier recovery (auto/manual mode)
        """
        self.db_path = db_path

        # Store client reference for recovery checks
        self._lm_studio_client = lm_studio_client

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

        # Initialize tier recovery manager
        self._recovery_manager = TierRecoveryManager(recovery_config)
        self._recovery_manager._tier_checker = self._check_tier_availability
        self._recovery_manager._on_tier_available = self._handle_tier_recovery
        self._recovery_manager.set_current_tier(self.active_tier)
        self._recovery_manager.start()

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

    def _check_tier_availability(self, tier: EmbeddingTier) -> bool:
        """
        Check if a specific tier is currently available.

        Args:
            tier: The tier to check

        Returns:
            True if the tier is available, False otherwise
        """
        if tier == EmbeddingTier.LM_STUDIO:
            if self._lm_studio_client is None:
                return False
            try:
                return self._lm_studio_client.test_embedding_availability(
                    timeout=self._recovery_manager.config.check_timeout
                )
            except Exception as e:
                logger.debug(f"LM Studio availability check failed: {e}")
                return False
        elif tier == EmbeddingTier.TFIDF:
            return self.tfidf_embedder.fitted
        elif tier == EmbeddingTier.FTS5:
            return True  # Always available
        return False

    def _handle_tier_recovery(self, new_tier: EmbeddingTier) -> None:
        """
        Handle successful tier recovery.

        Args:
            new_tier: The tier that became available
        """
        logger.info(f"Upgrading embedding tier: {self.active_tier.value} -> {new_tier.value}")

        # For LM Studio recovery, refresh the embedder availability
        if new_tier == EmbeddingTier.LM_STUDIO and self.lm_studio_embedder:
            self.lm_studio_embedder.available = True

        self.active_tier = new_tier

    def fit_tfidf(self, documents: List[str]):
        """
        Update TF-IDF model with new documents.

        Uses incremental update if already fitted (e.g., with default corpus),
        otherwise performs initial fit.

        Args:
            documents: List of document contents
        """
        if not documents:
            logger.warning("No documents provided for TF-IDF fitting")
            return

        if self.tfidf_embedder.fitted:
            # Incremental update - merge new documents with existing vocabulary
            logger.info(f"Incrementally updating TF-IDF with {len(documents)} documents...")
            self.tfidf_embedder.update_vocabulary(documents)
        else:
            # Initial fit
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
                self._recovery_manager.set_current_tier(self.active_tier)

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
                self._recovery_manager.set_current_tier(self.active_tier)

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
        recovery_status = self._recovery_manager.get_status()

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
            },
            'recovery': {
                'mode': recovery_status['mode'],
                'is_degraded': recovery_status['is_degraded'],
                'recovery_paused': recovery_status['recovery_paused'],
                'last_check_time': recovery_status['last_check_time'],
                'last_recovery_time': recovery_status['last_recovery_time'],
                'check_interval': recovery_status['check_interval']
            }
        }

    def try_upgrade_tier(self) -> Optional[EmbeddingTier]:
        """
        Manually attempt to upgrade to a higher tier.

        Returns:
            The new tier if upgraded, None if no upgrade occurred
        """
        return self._recovery_manager.try_upgrade_tier()

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get tier recovery status."""
        return self._recovery_manager.get_status()

    def stop_recovery(self) -> None:
        """Stop the tier recovery manager (call on shutdown)."""
        self._recovery_manager.stop()

    def set_recovery_mode(self, mode: str) -> None:
        """
        Change embedding tier recovery mode at runtime.

        Args:
            mode: "auto" or "manual"
        """
        if mode not in ("auto", "manual"):
            raise ValueError("Mode must be 'auto' or 'manual'")

        # Stop current recovery if running
        self._recovery_manager.stop()

        # Update mode and restart if auto
        self._recovery_manager.config.mode = mode
        if mode == "auto":
            self._recovery_manager.start()

        logger.info(f"Tier recovery mode set to: {mode}")


# Utility functions

def serialize_embedding(embedding: List[float]) -> bytes:
    """Serialize embedding to bytes for database storage."""
    return np.array(embedding, dtype=np.float32).tobytes()


def deserialize_embedding(embedding_bytes: bytes) -> List[float]:
    """Deserialize embedding from bytes."""
    return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
