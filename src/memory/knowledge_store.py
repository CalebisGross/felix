"""
Shared Knowledge Base for the Felix Framework.

Provides persistent storage and retrieval of knowledge across multiple runs,
enabling cross-run learning and knowledge accumulation.
"""

import json
import sqlite3
import hashlib
import time
import pickle
import logging
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Import validation functions for quality control
try:
    from src.workflows.truth_assessment import validate_knowledge_entry
    VALIDATION_AVAILABLE = True
except ImportError:
    logger.warning("Validation functions not available - running without quality control")
    VALIDATION_AVAILABLE = False

# Import audit logging for CRUD operations tracking
try:
    from src.memory.audit_log import audit_logged
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    logger.debug("Audit logging not available - running without audit trail")
    AUDIT_LOGGING_AVAILABLE = False
    # Define no-op decorator if audit logging not available
    def audit_logged(operation: str, user_agent: str = "KnowledgeStore"):
        def decorator(func):
            return func
        return decorator

# Lazy import for embeddings to avoid circular dependency
# (src.knowledge modules import KnowledgeStore, so we can't import at module level)
_serialize_embedding = None
_deserialize_embedding = None

def _get_embedding_functions():
    """Lazy import of embedding functions to avoid circular dependency."""
    global _serialize_embedding, _deserialize_embedding
    if _serialize_embedding is None:
        try:
            from src.knowledge.embeddings import serialize_embedding, deserialize_embedding
            _serialize_embedding = serialize_embedding
            _deserialize_embedding = deserialize_embedding
            logger.debug("Embedding serialization functions loaded successfully")
        except ImportError as e:
            logger.debug(f"Embeddings module not available: {e}")
            _serialize_embedding = False  # Mark as tried and failed
            _deserialize_embedding = False
    return _serialize_embedding, _deserialize_embedding

class KnowledgeType(Enum):
    """Types of knowledge that can be stored."""
    TASK_RESULT = "task_result"
    AGENT_INSIGHT = "agent_insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    FAILURE_ANALYSIS = "failure_analysis"
    OPTIMIZATION_DATA = "optimization_data"
    DOMAIN_EXPERTISE = "domain_expertise"
    TOOL_INSTRUCTION = "tool_instruction"  # For storing conditional tool instructions
    FILE_LOCATION = "file_location"  # For storing discovered file paths (meta-learning)

class ConfidenceLevel(Enum):
    """Confidence levels for knowledge entries."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

@dataclass
class KnowledgeEntry:
    """Single entry in the knowledge base."""
    knowledge_id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    confidence_level: ConfidenceLevel
    source_agent: str
    domain: str
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    access_count: int = 0
    success_rate: float = 1.0
    related_entries: List[str] = field(default_factory=list)
    # Validation fields (added for quality control)
    validation_score: float = 1.0
    validation_flags: List[str] = field(default_factory=list)
    validation_status: str = "trusted"
    validated_at: Optional[float] = None
    # Knowledge Brain fields (added for document ingestion)
    embedding: Optional[bytes] = None
    source_doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['knowledge_type'] = self.knowledge_type.value
        data['confidence_level'] = self.confidence_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        """Create from dictionary."""
        data['knowledge_type'] = KnowledgeType(data['knowledge_type'])
        data['confidence_level'] = ConfidenceLevel(data['confidence_level'])
        return cls(**data)

@dataclass
class KnowledgeQuery:
    """Query structure for knowledge retrieval."""
    knowledge_types: Optional[List[KnowledgeType]] = None
    domains: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_confidence: Optional[ConfidenceLevel] = None
    min_success_rate: Optional[float] = None
    content_keywords: Optional[List[str]] = None
    time_range: Optional[tuple[float, float]] = None
    limit: int = 10
    # Task context for meta-learning (new fields)
    task_type: Optional[str] = None
    task_complexity: Optional[str] = None
    use_semantic_search: bool = False
    
class KnowledgeStore:
    """
    Persistent knowledge storage system.
    
    Stores and retrieves knowledge entries across multiple framework runs,
    enabling learning and knowledge accumulation over time.
    """
    
    def __init__(self, storage_path: str = "felix_knowledge.db"):
        """
        Initialize knowledge store.

        Args:
            storage_path: Path to SQLite database file
        """
        self.storage_path = Path(storage_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.storage_path) as conn:
            # Main knowledge entries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    knowledge_id TEXT PRIMARY KEY,
                    knowledge_type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    content_compressed BLOB,
                    confidence_level TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    related_entries_json TEXT DEFAULT '[]',
                    validation_score REAL DEFAULT 1.0,
                    validation_flags TEXT DEFAULT '[]',
                    validation_status TEXT DEFAULT 'trusted',
                    validated_at REAL,
                    embedding BLOB,
                    source_doc_id TEXT,
                    chunk_index INTEGER
                )
            """)
            
            # Normalized tags table for efficient tag filtering
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_tags (
                    knowledge_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (knowledge_id, tag),
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge_entries(knowledge_id) ON DELETE CASCADE
                )
            """)
            
            # Indexes on main table
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_type 
                ON knowledge_entries(knowledge_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain 
                ON knowledge_entries(domain)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_confidence 
                ON knowledge_entries(confidence_level)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON knowledge_entries(created_at)
            """)
            
            # Indexes on tags table for efficient JOIN operations
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tag_lookup 
                ON knowledge_tags(tag)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_id_tag
                ON knowledge_tags(knowledge_id)
            """)

            # Create FTS5 virtual table for full-text search (Knowledge Brain)
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                        knowledge_id UNINDEXED,
                        content,
                        domain,
                        tags,
                        tokenize='porter unicode61'
                    )
                """)
            except sqlite3.OperationalError as e:
                # FTS5 might not be available in some SQLite builds
                logger.debug(f"FTS5 table creation skipped: {e}")

            # Knowledge relationships table for knowledge graph
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(source_id, target_id, relationship_type)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kr_source
                ON knowledge_relationships(source_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kr_target
                ON knowledge_relationships(target_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_kr_type
                ON knowledge_relationships(relationship_type)
            """)

            # Migrate existing data if needed
            self._migrate_existing_tags(conn)
    
    def _migrate_existing_tags(self, conn) -> None:
        """Migrate tags from JSON format to normalized table."""
        try:
            # Check if migration is needed by looking for entries with tags but no rows in knowledge_tags
            cursor = conn.execute("""
                SELECT ke.knowledge_id, ke.tags_json 
                FROM knowledge_entries ke 
                LEFT JOIN knowledge_tags kt ON ke.knowledge_id = kt.knowledge_id
                WHERE ke.tags_json != '[]' AND kt.knowledge_id IS NULL
            """)
            
            entries_to_migrate = cursor.fetchall()
            
            if entries_to_migrate:
                print(f"Migrating tags for {len(entries_to_migrate)} existing knowledge entries...")
                
                for knowledge_id, tags_json in entries_to_migrate:
                    try:
                        tags = json.loads(tags_json)
                        for tag in tags:
                            conn.execute("""
                                INSERT OR IGNORE INTO knowledge_tags (knowledge_id, tag) 
                                VALUES (?, ?)
                            """, (knowledge_id, tag))
                    except (json.JSONDecodeError, TypeError):
                        # Skip entries with invalid JSON
                        continue
                
                print(f"Tag migration completed for {len(entries_to_migrate)} entries.")
                
        except sqlite3.Error as e:
            # Migration failed, but don't crash - system will fall back to JSON tags
            print(f"Tag migration failed (non-critical): {e}")
            pass

    @contextmanager
    def transaction(self):
        """
        Context manager for safe database transactions.

        Provides automatic transaction management with proper error handling
        and rollback on failure. Foreign keys are enabled for CASCADE DELETE.

        Usage:
            with knowledge_store.transaction() as conn:
                conn.execute("DELETE FROM document_sources WHERE doc_id = ?", (doc_id,))
                conn.execute("UPDATE knowledge_entries SET ...")
                # Automatically commits on success, rolls back on exception

        Yields:
            sqlite3.Connection: Database connection with active transaction

        Raises:
            Exception: On commit failure (after automatic rollback)

        Example:
            >>> with store.transaction() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("DELETE FROM knowledge_entries WHERE domain = ?", ("test",))
            ...     cursor.execute("UPDATE document_sources SET status = ?", ("processed",))
            ...     # Changes committed automatically if no exceptions
        """
        conn = None
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.execute("PRAGMA foreign_keys = ON")  # Enable CASCADE DELETE
            conn.execute("BEGIN TRANSACTION")

            yield conn

            conn.commit()
            logger.debug("Transaction committed successfully")

        except Exception as e:
            if conn:
                conn.rollback()
                logger.error(f"Transaction failed, rolled back: {e}")
            raise

        finally:
            if conn:
                conn.close()

    def _generate_knowledge_id(self, content: Dict[str, Any],
                              source_agent: str,
                              domain: str) -> str:
        """
        Generate knowledge ID for deduplication.

        For web_search domain: Uses source_url for deduplication (same URL = same entry)
        For other domains: Uses content hash + source_agent

        Args:
            content: Knowledge content dictionary
            source_agent: Agent that generated this knowledge
            domain: Domain this knowledge applies to

        Returns:
            Unique knowledge ID string
        """
        # For web_search domain, deduplicate by source_url
        if domain == "web_search" and isinstance(content, dict) and "source_url" in content:
            source_url = content.get("source_url", "")
            # Hash: domain + source_url (no timestamp)
            hash_input = f"web_search:{source_url}"
            logger.debug(f"   Deduplication key: web_search + {source_url}")
        else:
            # For other domains, use content + source_agent (no timestamp)
            content_str = json.dumps(content, sort_keys=True)
            hash_input = f"{domain}:{content_str}:{source_agent}"
            logger.debug(f"   Deduplication key: {domain} + content + {source_agent}")

        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _decompress_content(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress legacy compressed content.

        Note: For backward compatibility only. New entries are stored as JSON.
        """
        return pickle.loads(compressed_data)

    @audit_logged("INSERT", "KnowledgeStore")
    def store_knowledge(self, knowledge_type: KnowledgeType,
                       content: Dict[str, Any],
                       confidence_level: ConfidenceLevel,
                       source_agent: str,
                       domain: str,
                       tags: Optional[List[str]] = None,
                       embedding: Optional[List[float]] = None,
                       source_doc_id: Optional[str] = None,
                       chunk_index: Optional[int] = None) -> str:
        """
        Store new knowledge entry.

        Args:
            knowledge_type: Type of knowledge
            content: Knowledge content
            confidence_level: Confidence in this knowledge
            source_agent: Agent that generated this knowledge
            domain: Domain this knowledge applies to
            tags: Optional tags for categorization
            embedding: Optional embedding vector for semantic search
            source_doc_id: Optional source document ID
            chunk_index: Optional chunk index within source document

        Returns:
            Knowledge ID of stored entry
        """
        logger.info("💾 KNOWLEDGE_STORE.store_knowledge() CALLED")
        logger.info(f"   Database: {self.storage_path}")
        logger.info(f"   Domain: {domain}")
        logger.info(f"   Confidence: {confidence_level.value}")
        logger.info(f"   Source agent: {source_agent}")
        logger.info(f"   Content preview: {str(content)[:200]}...")

        if tags is None:
            tags = []

        knowledge_id = self._generate_knowledge_id(content, source_agent, domain)
        logger.info(f"   Generated knowledge_id: {knowledge_id}")

        # Perform validation (if enabled)
        validation_result = None
        if VALIDATION_AVAILABLE:
            try:
                validation_result = validate_knowledge_entry(
                    content, source_agent, domain, confidence_level
                )
                logger.info(f"   Validation score: {validation_result['validation_score']:.2f}")
                logger.info(f"   Validation status: {validation_result['validation_status']}")
                if validation_result['validation_flags']:
                    logger.info(f"   Validation flags: {', '.join(validation_result['validation_flags'])}")

                # Handle quarantine case
                if not validation_result['should_store']:
                    logger.warning(f"   ⛔ Entry QUARANTINED - not stored (score too low)")
                    return knowledge_id  # Return ID but don't store
            except Exception as e:
                logger.warning(f"   Validation failed with error: {e} - proceeding without validation")
                validation_result = {
                    'validation_score': 1.0,
                    'validation_flags': [],
                    'validation_status': 'trusted',
                    'should_store': True
                }
        else:
            # No validation available - default to trusted
            validation_result = {
                'validation_score': 1.0,
                'validation_flags': [],
                'validation_status': 'trusted',
                'should_store': True
            }

        # Check if entry already exists (for deduplication logging)
        with sqlite3.connect(self.storage_path) as conn:
            existing = conn.execute(
                "SELECT created_at FROM knowledge_entries WHERE knowledge_id = ?",
                (knowledge_id,)
            ).fetchone()

            if existing:
                logger.info(f"   🔄 REFRESHING: Entry exists (originally created: {time.ctime(existing[0])}) - treating as new for freshness")
                # Update created_at to current time so fresh content appears in time-range queries
                created_at = time.time()
                updated_at = created_at
            else:
                logger.info(f"   ✨ NEW ENTRY: Storing for first time")
                created_at = time.time()
                updated_at = created_at

        entry = KnowledgeEntry(
            knowledge_id=knowledge_id,
            knowledge_type=knowledge_type,
            content=content,
            confidence_level=confidence_level,
            source_agent=source_agent,
            domain=domain,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at
        )

        # Store content as JSON (no compression)
        content_json = json.dumps(content)
        content_compressed = None  # Always None for new entries

        # Serialize embedding if provided (using lazy import to avoid circular dependency)
        embedding_blob = None
        if embedding:
            serialize_fn, _ = _get_embedding_functions()
            if serialize_fn and serialize_fn is not False:
                try:
                    embedding_blob = serialize_fn(embedding)
                    logger.debug(f"Serialized embedding: {len(embedding)} floats → {len(embedding_blob)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to serialize embedding: {e}")

        with sqlite3.connect(self.storage_path) as conn:
            # Store main entry (INSERT OR REPLACE will update if exists)
            logger.info(f"   📝 Executing INSERT OR REPLACE INTO knowledge_entries...")
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, content_compressed,
                 confidence_level, source_agent, domain, tags_json,
                 created_at, updated_at, access_count, success_rate, related_entries_json,
                 validation_score, validation_flags, validation_status, validated_at,
                 embedding, source_doc_id, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                knowledge_id,
                knowledge_type.value,
                content_json,
                content_compressed,
                confidence_level.value,
                source_agent,
                domain,
                json.dumps(tags),
                entry.created_at,
                entry.updated_at,
                0,
                1.0,
                json.dumps([]),
                validation_result['validation_score'],
                json.dumps(validation_result['validation_flags']),
                validation_result['validation_status'],
                time.time(),
                embedding_blob,
                source_doc_id,
                chunk_index
            ))
            logger.info(f"   ✓ INSERT OR REPLACE executed")

            # Store tags in normalized table for efficient filtering
            # First remove existing tags for this entry
            conn.execute("DELETE FROM knowledge_tags WHERE knowledge_id = ?", (knowledge_id,))

            # Insert new tags
            for tag in tags:
                conn.execute("""
                    INSERT INTO knowledge_tags (knowledge_id, tag)
                    VALUES (?, ?)
                """, (knowledge_id, tag))

            # CRITICAL: Commit the transaction!
            logger.info(f"   💾 Committing transaction...")
            conn.commit()
            logger.info(f"   ✅ Knowledge stored and committed successfully!")

            # Verify it was stored
            count = conn.execute("SELECT COUNT(*) FROM knowledge_entries WHERE knowledge_id = ?", (knowledge_id,)).fetchone()[0]
            logger.info(f"   ✓ Verification: {count} entry with knowledge_id={knowledge_id} in database")

        return knowledge_id
    
    def retrieve_knowledge(self, query: KnowledgeQuery) -> List[KnowledgeEntry]:
        """
        Retrieve knowledge entries matching query.

        Args:
            query: Query parameters

        Returns:
            List of matching knowledge entries
        """
        logger.info("🔍 KNOWLEDGE_STORE.retrieve_knowledge() CALLED")
        logger.info(f"   Database: {self.storage_path}")
        logger.info(f"   Query domains: {query.domains}")
        logger.info(f"   Query confidence: {query.min_confidence}")
        logger.info(f"   Query time_range: {query.time_range}")
        logger.info(f"   Query limit: {query.limit}")

        # Determine if we need to JOIN with tags table
        if query.tags:
            sql_parts = [
                "SELECT DISTINCT ke.* FROM knowledge_entries ke",
                "INNER JOIN knowledge_tags kt ON ke.knowledge_id = kt.knowledge_id",
                "WHERE 1=1"
            ]
        else:
            sql_parts = ["SELECT * FROM knowledge_entries ke WHERE 1=1"]

        params = []
        
        # Build WHERE clause
        if query.knowledge_types:
            type_placeholders = ",".join("?" * len(query.knowledge_types))
            sql_parts.append(f"AND ke.knowledge_type IN ({type_placeholders})")
            params.extend([kt.value for kt in query.knowledge_types])
        
        if query.domains:
            domain_placeholders = ",".join("?" * len(query.domains))
            sql_parts.append(f"AND ke.domain IN ({domain_placeholders})")
            params.extend(query.domains)
        
        if query.min_confidence:
            confidence_order = {
                ConfidenceLevel.LOW: 0,
                ConfidenceLevel.MEDIUM: 1,
                ConfidenceLevel.HIGH: 2,
                ConfidenceLevel.VERIFIED: 3
            }
            min_level = confidence_order[query.min_confidence]
            valid_levels = [level.value for level, order in confidence_order.items() 
                          if order >= min_level]
            level_placeholders = ",".join("?" * len(valid_levels))
            sql_parts.append(f"AND ke.confidence_level IN ({level_placeholders})")
            params.extend(valid_levels)
        
        if query.min_success_rate:
            sql_parts.append("AND ke.success_rate >= ?")
            params.append(query.min_success_rate)
        
        if query.time_range:
            sql_parts.append("AND ke.created_at BETWEEN ? AND ?")
            params.extend(query.time_range)
        
        # Tag filtering at SQL level for efficiency
        if query.tags:
            tag_placeholders = ",".join("?" * len(query.tags))
            sql_parts.append(f"AND kt.tag IN ({tag_placeholders})")
            params.extend(query.tags)
        
        # Add ordering and limit
        # Priority: confidence > success_rate > newest first (created_at) > recently updated
        sql_parts.append("""ORDER BY
            CASE ke.confidence_level
                WHEN 'verified' THEN 4
                WHEN 'high' THEN 3
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 1
                ELSE 0
            END DESC,
            ke.success_rate DESC,
            ke.created_at DESC,
            ke.updated_at DESC""")
        sql_parts.append("LIMIT ?")
        params.append(query.limit)
        
        sql = " ".join(sql_parts)

        logger.info("   📝 Final SQL Query:")
        logger.info(f"      {sql}")
        logger.info(f"   📝 Query Parameters:")
        logger.info(f"      {params}")

        entries = []
        with sqlite3.connect(self.storage_path) as conn:
            # First, check how many total entries exist in the database
            total_count = conn.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0]
            logger.info(f"   📊 Total entries in database: {total_count}")

            # Check entries in the queried domains
            if query.domains:
                domain_count = conn.execute(
                    f"SELECT COUNT(*) FROM knowledge_entries WHERE domain IN ({','.join('?' * len(query.domains))})",
                    query.domains
                ).fetchone()[0]
                logger.info(f"   📊 Entries in queried domains ({query.domains}): {domain_count}")

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            logger.info(f"   ✓ SQL returned {len(rows)} rows")

            for i, row in enumerate(rows, 1):
                entry = self._row_to_entry(row, conn)
                logger.info(f"      Row {i}: domain={entry.domain}, confidence={entry.confidence_level.value}, created={entry.created_at}")

                # Apply content filtering if specified
                if query.content_keywords:
                    content_str = json.dumps(entry.content).lower()
                    if not any(keyword.lower() in content_str
                             for keyword in query.content_keywords):
                        logger.info(f"      Row {i}: FILTERED OUT by content keywords")
                        continue

                entries.append(entry)

                # Update access count
                self._increment_access_count(entry.knowledge_id)

        logger.info(f"   ✅ Retrieved {len(entries)} entries before meta-learning boost")

        # Apply meta-learning boost if task context is provided
        if query.task_type and len(entries) > 0:
            logger.info(f"   🎯 Applying meta-learning boost for task_type='{query.task_type}'")
            entries = self._apply_meta_learning_boost(
                entries,
                query.task_type,
                query.task_complexity
            )
            logger.info(f"   ✅ Re-ranked entries using historical usefulness data")

        logger.info(f"   ✅ Returning {len(entries)} knowledge entries")

        return entries

    def _apply_meta_learning_boost(self,
                                    entries: List[KnowledgeEntry],
                                    task_type: str,
                                    task_complexity: Optional[str] = None) -> List[KnowledgeEntry]:
        """
        Apply meta-learning boost to re-rank entries based on historical usefulness.

        Entries that have been useful for similar tasks in the past get boosted
        in the ranking. Requires knowledge_usage table with historical data.

        Args:
            entries: List of knowledge entries to re-rank
            task_type: Type of task (for matching historical usage)
            task_complexity: Optional task complexity (SIMPLE_FACTUAL, MEDIUM, COMPLEX)

        Returns:
            Re-ranked list of entries (most useful first)
        """
        if not entries:
            return entries

        try:
            with sqlite3.connect(self.storage_path) as conn:
                # For each entry, get its historical usefulness score
                entry_scores = []
                for entry in entries:
                    # Query knowledge_usage table for this entry's usefulness
                    cursor = conn.execute("""
                        SELECT AVG(useful_score) as avg_usefulness, COUNT(*) as usage_count
                        FROM knowledge_usage
                        WHERE knowledge_id = ?
                        AND task_type = ?
                    """, (entry.knowledge_id, task_type))

                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        avg_usefulness = row[0]
                        usage_count = row[1]

                        # Calculate boost factor (requires min 2 samples for reliability)
                        if usage_count >= 2:
                            boost_factor = 0.7 + (avg_usefulness * 0.3)  # 0.7 to 1.0
                            logger.info(f"      Entry {entry.knowledge_id[:8]}: "
                                      f"usefulness={avg_usefulness:.2f}, uses={usage_count}, boost={boost_factor:.2f}")
                        else:
                            boost_factor = 1.0  # Neutral boost for insufficient data
                    else:
                        boost_factor = 1.0  # Neutral boost for new/unused entries

                    # Base score from confidence and success rate
                    confidence_scores = {
                        ConfidenceLevel.LOW: 1,
                        ConfidenceLevel.MEDIUM: 2,
                        ConfidenceLevel.HIGH: 3,
                        ConfidenceLevel.VERIFIED: 4
                    }
                    base_score = (confidence_scores.get(entry.confidence_level, 2) * 10 +
                                entry.success_rate * 10)

                    # Apply boost to base score
                    final_score = base_score * boost_factor
                    entry_scores.append((entry, final_score))

                # Sort by final score (highest first)
                entry_scores.sort(key=lambda x: x[1], reverse=True)
                reranked_entries = [entry for entry, score in entry_scores]

                logger.info(f"   📊 Meta-learning boost applied: reranked {len(reranked_entries)} entries")
                return reranked_entries

        except sqlite3.Error as e:
            logger.warning(f"   ⚠ Meta-learning boost failed: {e}, returning original order")
            return entries

    def _row_to_entry(self, row, conn=None) -> KnowledgeEntry:
        """
        Convert database row to KnowledgeEntry.

        Note: Handles legacy compressed data for backward compatibility.
        New entries always use content_json, but old entries may have
        content stored in content_compressed using pickle serialization.
        """
        # Handle old (13), validation (17), and full (20 columns) schemas for backward compatibility
        if len(row) == 20:
            # Full schema with validation + knowledge brain fields
            (knowledge_id, knowledge_type, content_json, content_compressed,
             confidence_level, source_agent, domain, tags_json,
             created_at, updated_at, access_count, success_rate, related_entries_json,
             validation_score, validation_flags, validation_status, validated_at,
             embedding, source_doc_id, chunk_index) = row
        elif len(row) == 17:
            # Schema with validation fields only
            (knowledge_id, knowledge_type, content_json, content_compressed,
             confidence_level, source_agent, domain, tags_json,
             created_at, updated_at, access_count, success_rate, related_entries_json,
             validation_score, validation_flags, validation_status, validated_at) = row
            # Set default knowledge brain values
            embedding = None
            source_doc_id = None
            chunk_index = None
        elif len(row) == 13:
            # Old schema without validation or knowledge brain fields
            (knowledge_id, knowledge_type, content_json, content_compressed,
             confidence_level, source_agent, domain, tags_json,
             created_at, updated_at, access_count, success_rate, related_entries_json) = row
            # Set default validation values
            validation_score = 1.0
            validation_flags = "[]"
            validation_status = "trusted"
            validated_at = None
            # Set default knowledge brain values
            embedding = None
            source_doc_id = None
            chunk_index = None
        else:
            raise ValueError(f"Unexpected row length: {len(row)}. Expected 13, 17, or 20 columns.")

        # Determine content source
        if content_compressed:
            content = self._decompress_content(content_compressed)
        else:
            content = json.loads(content_json)

        # Get tags from normalized table if connection provided, otherwise fallback to JSON
        tags = []
        if conn:
            try:
                cursor = conn.execute("SELECT tag FROM knowledge_tags WHERE knowledge_id = ?", (knowledge_id,))
                tags = [row[0] for row in cursor.fetchall()]
            except sqlite3.Error:
                # Fallback to JSON tags if query fails
                tags = json.loads(tags_json)
        else:
            tags = json.loads(tags_json)

        # Parse validation_flags JSON if it's a string
        if isinstance(validation_flags, str):
            validation_flags = json.loads(validation_flags)

        return KnowledgeEntry(
            knowledge_id=knowledge_id,
            knowledge_type=KnowledgeType(knowledge_type),
            content=content,
            confidence_level=ConfidenceLevel(confidence_level),
            source_agent=source_agent,
            domain=domain,
            tags=tags,
            created_at=created_at,
            updated_at=updated_at,
            access_count=access_count,
            success_rate=success_rate,
            related_entries=json.loads(related_entries_json),
            validation_score=validation_score,
            validation_flags=validation_flags,
            validation_status=validation_status,
            validated_at=validated_at,
            embedding=embedding,
            source_doc_id=source_doc_id,
            chunk_index=chunk_index
        )
    
    def _increment_access_count(self, knowledge_id: str) -> None:
        """Increment access count for knowledge entry."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                UPDATE knowledge_entries
                SET access_count = access_count + 1
                WHERE knowledge_id = ?
            """, (knowledge_id,))

    def record_knowledge_usage(self,
                              workflow_id: str,
                              knowledge_ids: List[str],
                              task_type: str,
                              task_complexity: Optional[str] = None,
                              useful_score: float = 0.5,
                              retrieval_method: str = "sql") -> bool:
        """
        Record knowledge usage for meta-learning.

        Tracks which knowledge entries were helpful for which types of tasks,
        enabling meta-learning boost in future retrievals.

        Args:
            workflow_id: Unique workflow identifier
            knowledge_ids: List of knowledge entry IDs that were used
            task_type: Type of task (for matching future similar tasks)
            task_complexity: Optional task complexity (SIMPLE_FACTUAL, MEDIUM, COMPLEX)
            useful_score: Usefulness score 0.0-1.0 (0.2=unhelpful, 0.5=neutral, 0.9=very helpful)
            retrieval_method: How knowledge was retrieved (sql, semantic, hybrid)

        Returns:
            True if recorded successfully, False otherwise
        """
        if not knowledge_ids:
            return True  # Nothing to record

        try:
            with sqlite3.connect(self.storage_path) as conn:
                for knowledge_id in knowledge_ids:
                    conn.execute("""
                        INSERT INTO knowledge_usage
                        (workflow_id, knowledge_id, task_type, task_complexity,
                         useful_score, retrieval_method, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (workflow_id, knowledge_id, task_type, task_complexity,
                          useful_score, retrieval_method, time.time()))

                conn.commit()
                logger.info(f"   📊 Recorded usage for {len(knowledge_ids)} knowledge entries "
                          f"(task_type={task_type}, usefulness={useful_score:.2f})")
                return True

        except sqlite3.Error as e:
            logger.error(f"Failed to record knowledge usage: {e}")
            return False

    def get_domain_coverage_metrics(self, domain: str) -> Dict[str, Any]:
        """
        Get coverage metrics for a specific domain.

        Computes entry count, average confidence, freshness, and overall
        coverage score for epistemic awareness (Phase 5 - Knowledge Gap Cartography).

        Args:
            domain: Domain name to compute metrics for

        Returns:
            Dictionary with coverage metrics:
            - entry_count: Number of knowledge entries
            - avg_confidence: Average confidence level (0.0-1.0)
            - freshness_score: How recent the entries are (0.0-1.0)
            - coverage_score: Overall coverage score (0.0-1.0)
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
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

                # Compute freshness score (entries updated recently = higher score)
                current_time = time.time()
                if avg_updated > 0:
                    age_days = (current_time - avg_updated) / (24 * 3600)
                    freshness_score = max(0.0, 1.0 - (age_days / 30))  # Decays over 30 days
                else:
                    freshness_score = 0.0

                # Compute coverage score as weighted combination
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

        except sqlite3.Error as e:
            logger.error(f"Failed to get domain coverage metrics: {e}")
            return {
                'entry_count': 0,
                'avg_confidence': 0.0,
                'freshness_score': 0.0,
                'coverage_score': 0.0
            }

    def update_success_rate(self, knowledge_id: str, 
                           success_rate: float) -> bool:
        """
        Update success rate for knowledge entry.
        
        Args:
            knowledge_id: ID of knowledge entry
            success_rate: New success rate (0.0 to 1.0)
            
        Returns:
            True if updated successfully
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                UPDATE knowledge_entries 
                SET success_rate = ?, updated_at = ?
                WHERE knowledge_id = ?
            """, (success_rate, time.time(), knowledge_id))
            return cursor.rowcount > 0
    
    def add_related_entry(self, knowledge_id: str, 
                         related_id: str) -> bool:
        """
        Add relationship between knowledge entries.
        
        Args:
            knowledge_id: Primary knowledge entry ID
            related_id: Related knowledge entry ID
            
        Returns:
            True if relationship added successfully
        """
        with sqlite3.connect(self.storage_path) as conn:
            # Get current related entries
            cursor = conn.execute("""
                SELECT related_entries_json FROM knowledge_entries 
                WHERE knowledge_id = ?
            """, (knowledge_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            related_entries = json.loads(row[0])
            if related_id not in related_entries:
                related_entries.append(related_id)
                
                conn.execute("""
                    UPDATE knowledge_entries 
                    SET related_entries_json = ?, updated_at = ?
                    WHERE knowledge_id = ?
                """, (json.dumps(related_entries), time.time(), knowledge_id))
            
            return True

    def get_entry_by_id(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a single knowledge entry by ID.

        Args:
            knowledge_id: Entry ID

        Returns:
            KnowledgeEntry or None if not found
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM knowledge_entries WHERE knowledge_id = ?",
                    (knowledge_id,)
                )
                row = cursor.fetchone()

                if row:
                    return self._row_to_entry(row, conn)
                return None

        except Exception as e:
            logger.error(f"Error retrieving entry {knowledge_id}: {e}")
            return None

    @audit_logged("UPDATE", "KnowledgeStore")
    def update_knowledge_entry(self, knowledge_id: str,
                               updates: Dict[str, Any]) -> bool:
        """
        Update an existing knowledge entry.

        Args:
            knowledge_id: Entry ID to update
            updates: Dict with fields to update (content, confidence_level,
                    domain, tags, etc.)

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Get existing entry
            entry = self.get_entry_by_id(knowledge_id)
            if not entry:
                logger.warning(f"Entry not found: {knowledge_id}")
                return False

            with sqlite3.connect(self.storage_path) as conn:
                # Build UPDATE statement dynamically
                update_fields = []
                params = []

                if 'content' in updates:
                    update_fields.append("content_json = ?")
                    params.append(json.dumps(updates['content']))
                    # Invalidate embedding if content changes
                    update_fields.append("embedding = NULL")

                if 'confidence_level' in updates:
                    update_fields.append("confidence_level = ?")
                    level = updates['confidence_level']
                    if isinstance(level, ConfidenceLevel):
                        params.append(level.value)
                    else:
                        params.append(level)

                if 'domain' in updates:
                    update_fields.append("domain = ?")
                    params.append(updates['domain'])

                if 'tags' in updates:
                    update_fields.append("tags_json = ?")
                    params.append(json.dumps(updates['tags']))

                    # Update normalized tags table
                    conn.execute("DELETE FROM knowledge_tags WHERE knowledge_id = ?",
                               (knowledge_id,))
                    for tag in updates['tags']:
                        conn.execute(
                            "INSERT INTO knowledge_tags (knowledge_id, tag) VALUES (?, ?)",
                            (knowledge_id, tag)
                        )

                # Always update timestamp
                update_fields.append("updated_at = ?")
                params.append(time.time())

                # Execute update
                params.append(knowledge_id)
                sql = f"UPDATE knowledge_entries SET {', '.join(update_fields)} WHERE knowledge_id = ?"

                cursor = conn.execute(sql, params)
                conn.commit()

                logger.info(f"Updated entry {knowledge_id}")
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating entry {knowledge_id}: {e}")
            return False

    @audit_logged("MERGE", "KnowledgeStore")
    def merge_knowledge_entries(self, primary_id: str,
                                secondary_ids: List[str],
                                merge_strategy: str = "keep_primary") -> bool:
        """
        Merge multiple knowledge entries into one.

        Args:
            primary_id: ID of entry to keep
            secondary_ids: IDs of entries to merge and delete
            merge_strategy: How to merge ("keep_primary", "combine_content",
                           "highest_confidence")

        Returns:
            True if merged successfully, False otherwise
        """
        try:
            # Get all entries
            primary = self.get_entry_by_id(primary_id)
            if not primary:
                logger.error(f"Primary entry not found: {primary_id}")
                return False

            secondaries = [self.get_entry_by_id(sid) for sid in secondary_ids]
            secondaries = [s for s in secondaries if s is not None]

            if not secondaries:
                logger.warning("No valid secondary entries to merge")
                return False

            # Merge logic based on strategy
            merged_content = primary.content.copy()
            merged_tags = set(primary.tags)
            merged_related = set(primary.related_entries)
            max_confidence = primary.confidence_level

            for secondary in secondaries:
                merged_tags.update(secondary.tags)
                merged_related.update(secondary.related_entries)

                # Combine confidence (use highest)
                confidence_order = {
                    ConfidenceLevel.LOW: 0,
                    ConfidenceLevel.MEDIUM: 1,
                    ConfidenceLevel.HIGH: 2,
                    ConfidenceLevel.VERIFIED: 3
                }
                if confidence_order[secondary.confidence_level] > confidence_order[max_confidence]:
                    max_confidence = secondary.confidence_level

                # Combine content based on strategy
                if merge_strategy == "combine_content":
                    for key, value in secondary.content.items():
                        if key not in merged_content:
                            merged_content[key] = value
                        elif isinstance(value, list):
                            if isinstance(merged_content[key], list):
                                merged_content[key].extend(value)

            # Update primary entry
            updates = {
                'content': merged_content,
                'confidence_level': max_confidence,
                'tags': list(merged_tags)
            }

            with sqlite3.connect(self.storage_path) as conn:
                # Update primary
                self.update_knowledge_entry(primary_id, updates)

                # Update related_entries_json
                conn.execute("""
                    UPDATE knowledge_entries
                    SET related_entries_json = ?
                    WHERE knowledge_id = ?
                """, (json.dumps(list(merged_related)), primary_id))

                # Delete secondary entries
                for sec_id in secondary_ids:
                    self.delete_knowledge(sec_id)

                conn.commit()

            logger.info(f"Merged {len(secondary_ids)} entries into {primary_id}")
            return True

        except Exception as e:
            logger.error(f"Error merging entries: {e}")
            return False

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary statistics of knowledge store."""
        with sqlite3.connect(self.storage_path) as conn:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
            total_entries = cursor.fetchone()[0]
            
            # Entries by type
            cursor = conn.execute("""
                SELECT knowledge_type, COUNT(*) 
                FROM knowledge_entries 
                GROUP BY knowledge_type
            """)
            by_type = dict(cursor.fetchall())
            
            # Entries by domain
            cursor = conn.execute("""
                SELECT domain, COUNT(*) 
                FROM knowledge_entries 
                GROUP BY domain
            """)
            by_domain = dict(cursor.fetchall())
            
            # Confidence distribution
            cursor = conn.execute("""
                SELECT confidence_level, COUNT(*) 
                FROM knowledge_entries 
                GROUP BY confidence_level
            """)
            by_confidence = dict(cursor.fetchall())
            
            # Average success rate
            cursor = conn.execute("""
                SELECT AVG(success_rate) FROM knowledge_entries
            """)
            avg_success_rate = cursor.fetchone()[0] or 0.0

            # Count by tag (concepts vs entities)
            try:
                cursor = conn.execute("""
                    SELECT kt.tag, COUNT(DISTINCT kt.knowledge_id)
                    FROM knowledge_tags kt
                    WHERE kt.tag IN ('concept', 'entity')
                    GROUP BY kt.tag
                """)
                tag_counts = dict(cursor.fetchall())
                concept_count = tag_counts.get('concept', 0)
                entity_count = tag_counts.get('entity', 0)
            except sqlite3.Error as e:
                logger.warning(f"Failed to count by tag: {e}")
                concept_count = 0
                entity_count = 0

            # Count high confidence entries
            high_conf_count = by_confidence.get('high', 0) + by_confidence.get('verified', 0)

            return {
                "total_entries": total_entries,
                "concept_count": concept_count,
                "entity_count": entity_count,
                "high_confidence_entries": high_conf_count,
                "by_type": by_type,
                "domain_distribution": by_domain,  # Renamed for consistency with GUI
                "by_confidence": by_confidence,
                "average_success_rate": avg_success_rate,
                "storage_path": str(self.storage_path)
            }

    @audit_logged("CLEANUP", "KnowledgeStore")
    def cleanup_old_entries(self, max_age_days: int = 30,
                           min_success_rate: float = 0.3) -> int:
        """
        Clean up old or low-performing knowledge entries.
        
        Args:
            max_age_days: Maximum age in days
            min_success_rate: Minimum success rate to keep
            
        Returns:
            Number of entries deleted
        """
        max_age_seconds = max_age_days * 24 * 3600
        cutoff_time = time.time() - max_age_seconds
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                DELETE FROM knowledge_entries
                WHERE (created_at < ? AND success_rate < ?)
                   OR (access_count = 0 AND created_at < ?)
            """, (cutoff_time, min_success_rate, cutoff_time))

            return cursor.rowcount

    @audit_logged("DELETE", "KnowledgeStore")
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete a single knowledge entry by ID.

        Args:
            knowledge_id: ID of the knowledge entry to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Delete from main table (FTS5 trigger will handle knowledge_fts)
                cursor = conn.execute(
                    "DELETE FROM knowledge_entries WHERE knowledge_id = ?",
                    (knowledge_id,)
                )

                # Also remove from tags table
                conn.execute(
                    "DELETE FROM knowledge_tags WHERE knowledge_id = ?",
                    (knowledge_id,)
                )

                # Remove this entry from all related_entries_json arrays
                # This prevents broken relationships
                conn.execute("""
                    UPDATE knowledge_entries
                    SET related_entries_json = (
                        SELECT json_remove(
                            related_entries_json,
                            '$[' || idx || ']'
                        )
                        FROM (
                            SELECT key as idx
                            FROM json_each(related_entries_json)
                            WHERE value = ?
                        )
                    )
                    WHERE related_entries_json LIKE ?
                """, (knowledge_id, f'%{knowledge_id}%'))

                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting knowledge entry {knowledge_id}: {e}")
            return False

    def preview_delete_by_pattern(self, path_pattern: str,
                                   include_entries: bool = True) -> Dict[str, Any]:
        """
        Preview what would be deleted by a path pattern without actually deleting.

        Args:
            path_pattern: SQL LIKE pattern or glob pattern for file paths
            include_entries: If True, count associated knowledge entries

        Returns:
            Dict with counts and sample paths
        """
        import fnmatch

        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Convert glob pattern to SQL LIKE if needed
                if '*' in path_pattern or '?' in path_pattern:
                    # For glob patterns, fetch all and filter in Python
                    cursor = conn.execute(
                        "SELECT doc_id, file_path, ingestion_status FROM document_sources"
                    )
                    all_docs = cursor.fetchall()

                    matching_docs = [
                        (doc_id, path, status)
                        for doc_id, path, status in all_docs
                        if fnmatch.fnmatch(path, path_pattern)
                    ]
                    doc_count = len(matching_docs)
                    doc_ids = [doc_id for doc_id, _, _ in matching_docs]
                    sample_paths = [path for _, path, _ in matching_docs[:10]]
                else:
                    # SQL LIKE pattern
                    cursor = conn.execute("""
                        SELECT doc_id, file_path
                        FROM document_sources
                        WHERE file_path LIKE ?
                        LIMIT 10
                    """, (path_pattern,))
                    sample_paths = [row[1] for row in cursor.fetchall()]

                    cursor = conn.execute("""
                        SELECT COUNT(*), GROUP_CONCAT(doc_id)
                        FROM document_sources
                        WHERE file_path LIKE ?
                    """, (path_pattern,))
                    row = cursor.fetchone()
                    doc_count = row[0]
                    doc_ids = row[1].split(',') if row[1] else []

                # Count associated knowledge entries
                entry_count = 0
                if include_entries and doc_ids:
                    placeholders = ','.join('?' * len(doc_ids))
                    cursor = conn.execute(f"""
                        SELECT COUNT(*)
                        FROM knowledge_entries
                        WHERE source_doc_id IN ({placeholders})
                    """, doc_ids)
                    entry_count = cursor.fetchone()[0]

                return {
                    "document_count": doc_count,
                    "entry_count": entry_count,
                    "sample_paths": sample_paths,
                    "pattern": path_pattern
                }
        except Exception as e:
            logger.error(f"Error previewing delete by pattern: {e}")
            return {"error": str(e)}

    @audit_logged("DELETE", "KnowledgeStore")
    def delete_documents_by_pattern(self, path_pattern: str,
                                     cascade_entries: bool = False,
                                     dry_run: bool = False) -> Dict[str, Any]:
        """
        Delete document sources matching a path pattern.

        Args:
            path_pattern: SQL LIKE pattern or glob pattern for file paths
            cascade_entries: If True, also delete associated knowledge entries
            dry_run: If True, only preview without deleting

        Returns:
            Dict with deleted counts and details
        """
        import fnmatch

        if dry_run:
            return self.preview_delete_by_pattern(path_pattern, cascade_entries)

        try:
            with self.transaction() as conn:
                # Find matching documents
                if '*' in path_pattern or '?' in path_pattern:
                    cursor = conn.execute(
                        "SELECT doc_id, file_path FROM document_sources"
                    )
                    all_docs = cursor.fetchall()

                    matching_docs = [
                        (doc_id, path)
                        for doc_id, path in all_docs
                        if fnmatch.fnmatch(path, path_pattern)
                    ]
                    doc_ids = [doc_id for doc_id, _ in matching_docs]
                else:
                    cursor = conn.execute("""
                        SELECT doc_id FROM document_sources
                        WHERE file_path LIKE ?
                    """, (path_pattern,))
                    doc_ids = [row[0] for row in cursor.fetchall()]

                if not doc_ids:
                    return {"documents_deleted": 0, "entries_deleted": 0}

                entries_deleted = 0

                # Delete associated knowledge entries if requested
                if cascade_entries:
                    placeholders = ','.join('?' * len(doc_ids))

                    # Get entry IDs first for cleanup
                    cursor = conn.execute(f"""
                        SELECT knowledge_id FROM knowledge_entries
                        WHERE source_doc_id IN ({placeholders})
                    """, doc_ids)
                    entry_ids = [row[0] for row in cursor.fetchall()]

                    # Delete entries
                    cursor = conn.execute(f"""
                        DELETE FROM knowledge_entries
                        WHERE source_doc_id IN ({placeholders})
                    """, doc_ids)
                    entries_deleted = cursor.rowcount

                    # Delete from tags table
                    if entry_ids:
                        placeholders = ','.join('?' * len(entry_ids))
                        conn.execute(f"""
                            DELETE FROM knowledge_tags
                            WHERE knowledge_id IN ({placeholders})
                        """, entry_ids)

                # Delete document sources
                placeholders = ','.join('?' * len(doc_ids))
                cursor = conn.execute(f"""
                    DELETE FROM document_sources
                    WHERE doc_id IN ({placeholders})
                """, doc_ids)
                docs_deleted = cursor.rowcount

                return {
                    "documents_deleted": docs_deleted,
                    "entries_deleted": entries_deleted,
                    "pattern": path_pattern
                }

        except Exception as e:
            logger.error(f"Error deleting documents by pattern: {e}")
            return {"error": str(e)}

    def delete_entries_by_source_pattern(self, path_pattern: str,
                                         dry_run: bool = False) -> Dict[str, Any]:
        """
        Delete knowledge entries whose source documents match a path pattern.
        Does NOT delete the document sources themselves.

        Args:
            path_pattern: SQL LIKE pattern or glob pattern for file paths
            dry_run: If True, only preview without deleting

        Returns:
            Dict with deleted count
        """
        if dry_run:
            preview = self.preview_delete_by_pattern(path_pattern, include_entries=True)
            return {"entries_would_delete": preview.get("entry_count", 0)}

        import fnmatch

        try:
            with self.transaction() as conn:
                # Find matching document IDs
                if '*' in path_pattern or '?' in path_pattern:
                    cursor = conn.execute(
                        "SELECT doc_id, file_path FROM document_sources"
                    )
                    all_docs = cursor.fetchall()

                    doc_ids = [
                        doc_id
                        for doc_id, path in all_docs
                        if fnmatch.fnmatch(path, path_pattern)
                    ]
                else:
                    cursor = conn.execute("""
                        SELECT doc_id FROM document_sources
                        WHERE file_path LIKE ?
                    """, (path_pattern,))
                    doc_ids = [row[0] for row in cursor.fetchall()]

                if not doc_ids:
                    return {"entries_deleted": 0}

                # Get entry IDs for tag cleanup
                placeholders = ','.join('?' * len(doc_ids))
                cursor = conn.execute(f"""
                    SELECT knowledge_id FROM knowledge_entries
                    WHERE source_doc_id IN ({placeholders})
                """, doc_ids)
                entry_ids = [row[0] for row in cursor.fetchall()]

                # Delete entries
                cursor = conn.execute(f"""
                    DELETE FROM knowledge_entries
                    WHERE source_doc_id IN ({placeholders})
                """, doc_ids)
                entries_deleted = cursor.rowcount

                # Delete from tags table
                if entry_ids:
                    placeholders = ','.join('?' * len(entry_ids))
                    conn.execute(f"""
                        DELETE FROM knowledge_tags
                        WHERE knowledge_id IN ({placeholders})
                    """, entry_ids)

                return {
                    "entries_deleted": entries_deleted,
                    "pattern": path_pattern
                }

        except Exception as e:
            logger.error(f"Error deleting entries by source pattern: {e}")
            return {"error": str(e)}

    @audit_logged("CLEANUP", "KnowledgeStore")
    def delete_orphaned_entries(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Delete knowledge entries that have no corresponding document source.

        Args:
            dry_run: If True, only count without deleting

        Returns:
            Dict with deleted count
        """
        try:
            # Find orphaned entries first (read-only query)
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT knowledge_id
                    FROM knowledge_entries
                    WHERE source_doc_id IS NOT NULL
                    AND source_doc_id NOT IN (SELECT doc_id FROM document_sources)
                """)
                orphaned_ids = [row[0] for row in cursor.fetchall()]

            if dry_run:
                return {"orphaned_count": len(orphaned_ids)}

            if not orphaned_ids:
                return {"entries_deleted": 0}

            # Delete in transaction
            with self.transaction() as conn:
                # Delete orphaned entries
                placeholders = ','.join('?' * len(orphaned_ids))
                cursor = conn.execute(f"""
                    DELETE FROM knowledge_entries
                    WHERE knowledge_id IN ({placeholders})
                """, orphaned_ids)
                deleted_count = cursor.rowcount

                # Delete from tags table
                conn.execute(f"""
                    DELETE FROM knowledge_tags
                    WHERE knowledge_id IN ({placeholders})
                """, orphaned_ids)

                return {"entries_deleted": deleted_count}

        except Exception as e:
            logger.error(f"Error deleting orphaned entries: {e}")
            return {"error": str(e)}

    @audit_logged("CLEANUP", "KnowledgeStore")
    def delete_failed_documents(self, max_age_days: int = 7,
                                 cascade_entries: bool = False,
                                 dry_run: bool = False) -> Dict[str, Any]:
        """
        Delete document sources with 'failed' status older than specified age.

        Args:
            max_age_days: Minimum age in days for failed documents to delete
            cascade_entries: If True, also delete associated knowledge entries
            dry_run: If True, only count without deleting

        Returns:
            Dict with deleted counts
        """
        max_age_seconds = max_age_days * 24 * 3600
        cutoff_time = time.time() - max_age_seconds

        try:
            # Find failed documents first (read-only query)
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT doc_id
                    FROM document_sources
                    WHERE ingestion_status = 'failed'
                    AND added_at < ?
                """, (cutoff_time,))
                failed_ids = [row[0] for row in cursor.fetchall()]

                if dry_run:
                    entry_count = 0
                    if cascade_entries and failed_ids:
                        placeholders = ','.join('?' * len(failed_ids))
                        cursor = conn.execute(f"""
                            SELECT COUNT(*) FROM knowledge_entries
                            WHERE source_doc_id IN ({placeholders})
                        """, failed_ids)
                        entry_count = cursor.fetchone()[0]

                    return {
                        "failed_documents": len(failed_ids),
                        "entries_affected": entry_count
                    }

            if not failed_ids:
                return {"documents_deleted": 0, "entries_deleted": 0}

            # Delete in transaction
            with self.transaction() as conn:
                entries_deleted = 0

                # Delete associated entries if requested
                if cascade_entries:
                    placeholders = ','.join('?' * len(failed_ids))

                    # Get entry IDs for tag cleanup
                    cursor = conn.execute(f"""
                        SELECT knowledge_id FROM knowledge_entries
                        WHERE source_doc_id IN ({placeholders})
                    """, failed_ids)
                    entry_ids = [row[0] for row in cursor.fetchall()]

                    # Delete entries
                    cursor = conn.execute(f"""
                        DELETE FROM knowledge_entries
                        WHERE source_doc_id IN ({placeholders})
                    """, failed_ids)
                    entries_deleted = cursor.rowcount

                    # Delete from tags
                    if entry_ids:
                        placeholders = ','.join('?' * len(entry_ids))
                        conn.execute(f"""
                            DELETE FROM knowledge_tags
                            WHERE knowledge_id IN ({placeholders})
                        """, entry_ids)

                # Delete failed documents
                placeholders = ','.join('?' * len(failed_ids))
                cursor = conn.execute(f"""
                    DELETE FROM document_sources
                    WHERE doc_id IN ({placeholders})
                """, failed_ids)
                docs_deleted = cursor.rowcount

                return {
                    "documents_deleted": docs_deleted,
                    "entries_deleted": entries_deleted
                }

        except Exception as e:
            logger.error(f"Error deleting failed documents: {e}")
            return {"error": str(e)}

    # Watch Directory Management

    def add_watch_directory(self, directory_path: str, notes: str = None) -> bool:
        """
        Add a watched directory to the database.

        Args:
            directory_path: Absolute path to directory
            notes: Optional notes about this directory

        Returns:
            True if added successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO watch_directories
                    (directory_path, added_at, enabled, notes)
                    VALUES (?, ?, 1, ?)
                """, (directory_path, time.time(), notes))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error adding watch directory: {e}")
            return False

    def remove_watch_directory(self, directory_path: str) -> bool:
        """
        Remove a watched directory from the database.

        Args:
            directory_path: Absolute path to directory

        Returns:
            True if removed successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM watch_directories
                    WHERE directory_path = ?
                """, (directory_path,))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error removing watch directory: {e}")
            return False

    def update_watch_directory_stats(self, directory_path: str) -> bool:
        """
        Update statistics for a watched directory.

        Counts documents and entries from this directory and updates the table.

        Args:
            directory_path: Absolute path to directory

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Count documents from this directory
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM document_sources
                    WHERE file_path LIKE ?
                """, (f"{directory_path}%",))
                doc_count = cursor.fetchone()[0]

                # Count entries from this directory
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT ke.knowledge_id)
                    FROM knowledge_entries ke
                    JOIN document_sources ds ON ke.source_doc_id = ds.doc_id
                    WHERE ds.file_path LIKE ?
                """, (f"{directory_path}%",))
                entry_count = cursor.fetchone()[0]

                # Update watch directory
                cursor = conn.execute("""
                    UPDATE watch_directories
                    SET document_count = ?,
                        entry_count = ?,
                        last_scan = ?
                    WHERE directory_path = ?
                """, (doc_count, entry_count, time.time(), directory_path))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error updating watch directory stats: {e}")
            return False

    def get_watch_directories(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get all watched directories with their statistics.

        Args:
            enabled_only: If True, only return enabled directories

        Returns:
            List of directory info dicts
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                query = """
                    SELECT watch_id, directory_path, added_at, enabled,
                           last_scan, document_count, entry_count, notes
                    FROM watch_directories
                """

                if enabled_only:
                    query += " WHERE enabled = 1"

                query += " ORDER BY added_at DESC"

                cursor = conn.execute(query)
                directories = []

                for row in cursor.fetchall():
                    directories.append({
                        'watch_id': row[0],
                        'directory_path': row[1],
                        'added_at': row[2],
                        'enabled': bool(row[3]),
                        'last_scan': row[4],
                        'document_count': row[5],
                        'entry_count': row[6],
                        'notes': row[7]
                    })

                return directories

        except Exception as e:
            logger.error(f"Error getting watch directories: {e}")
            return []

    def toggle_watch_directory(self, directory_path: str) -> bool:
        """
        Toggle enabled/disabled state of a watched directory.

        Args:
            directory_path: Absolute path to directory

        Returns:
            True if toggled successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Get current state
                cursor = conn.execute("""
                    SELECT enabled FROM watch_directories
                    WHERE directory_path = ?
                """, (directory_path,))

                row = cursor.fetchone()
                if not row:
                    return False

                new_state = 0 if row[0] else 1

                # Toggle state
                cursor = conn.execute("""
                    UPDATE watch_directories
                    SET enabled = ?
                    WHERE directory_path = ?
                """, (new_state, directory_path))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Error toggling watch directory: {e}")
            return False

    # ===== Phase 5: Advanced Search & Analytics =====

    def advanced_search(
        self,
        content: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence_level: Optional[ConfidenceLevel] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        min_validation_score: Optional[float] = None,
        logic: str = "AND",
        limit: int = 100
    ) -> List[KnowledgeEntry]:
        """
        Advanced multi-field search with AND/OR logic.

        Args:
            content: Search in content (uses FTS5 if available)
            domain: Filter by domain (exact match or LIKE pattern)
            tags: Filter by tags (all must match for AND, any for OR)
            confidence_level: Minimum confidence level
            start_date: Filter by creation date (start)
            end_date: Filter by creation date (end)
            knowledge_type: Filter by knowledge type
            min_validation_score: Minimum validation score
            logic: "AND" or "OR" for combining conditions
            limit: Maximum results to return

        Returns:
            List of matching knowledge entries
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row

            # Build query based on logic
            if logic not in ("AND", "OR"):
                logic = "AND"

            conditions = []
            params = []

            # Content search (use FTS5 if available and content specified)
            if content:
                # Try FTS5 first for better performance
                fts_cursor = conn.cursor()
                fts_cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='knowledge_fts'
                """)
                if fts_cursor.fetchone():
                    # Use FTS5
                    fts_cursor.execute("""
                        SELECT knowledge_id FROM knowledge_fts
                        WHERE knowledge_fts MATCH ?
                        LIMIT ?
                    """, (content, limit * 2))  # Get more from FTS, filter later
                    fts_ids = [row[0] for row in fts_cursor.fetchall()]
                    if fts_ids:
                        placeholders = ','.join('?' * len(fts_ids))
                        conditions.append(f"knowledge_id IN ({placeholders})")
                        params.extend(fts_ids)
                else:
                    # Fallback to LIKE search
                    conditions.append("content_json LIKE ?")
                    params.append(f"%{content}%")

            # Domain filter
            if domain:
                if "%" in domain or "*" in domain:
                    domain = domain.replace("*", "%")
                    conditions.append("domain LIKE ?")
                else:
                    conditions.append("domain = ?")
                params.append(domain)

            # Tags filter
            if tags:
                if logic == "AND":
                    # All tags must match
                    for tag in tags:
                        conditions.append("tags_json LIKE ?")
                        params.append(f'%"{tag}"%')
                else:
                    # Any tag matches
                    tag_conditions = " OR ".join(["tags_json LIKE ?" for _ in tags])
                    conditions.append(f"({tag_conditions})")
                    params.extend([f'%"{tag}"%' for tag in tags])

            # Confidence level
            if confidence_level:
                conditions.append("confidence_level = ?")
                params.append(confidence_level.value)

            # Date range
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date.timestamp())

            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date.timestamp())

            # Knowledge type
            if knowledge_type:
                conditions.append("knowledge_type = ?")
                params.append(knowledge_type.value)

            # Validation score
            if min_validation_score is not None:
                conditions.append("validation_score >= ?")
                params.append(min_validation_score)

            # Build final query
            if conditions:
                where_clause = f" WHERE {f' {logic} '.join(conditions)}"
            else:
                where_clause = ""

            query = f"""
                SELECT * FROM knowledge_entries
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = conn.cursor()
            cursor.execute(query, params)

            # Convert rows to KnowledgeEntry objects
            entries = []
            for row in cursor.fetchall():
                entry = self._row_to_knowledge_entry(row)
                if entry:
                    entries.append(entry)

            conn.close()

            logger.info(f"Advanced search returned {len(entries)} results")
            return entries

        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def get_analytics_data(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics data for dashboard.

        Returns:
            Dictionary with analytics metrics:
            - entries_per_domain_over_time: Time series data
            - entry_growth_trend: Monthly growth statistics
            - confidence_distribution: Breakdown by confidence level
            - top_domains: Top 10 domains by entry count
            - quality_metrics: Average confidence, validation scores
            - relationship_stats: Relationship counts and types
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()

            analytics = {}

            # 1. Entries per domain (top 10)
            cursor.execute("""
                SELECT domain, COUNT(*) as count
                FROM knowledge_entries
                WHERE domain IS NOT NULL
                GROUP BY domain
                ORDER BY count DESC
                LIMIT 10
            """)
            analytics['top_domains'] = [
                {'domain': row[0], 'count': row[1]}
                for row in cursor.fetchall()
            ]

            # 2. Confidence distribution
            cursor.execute("""
                SELECT confidence_level, COUNT(*) as count
                FROM knowledge_entries
                GROUP BY confidence_level
            """)
            analytics['confidence_distribution'] = {
                row[0]: row[1] for row in cursor.fetchall()
            }

            # 3. Entry growth trend (last 12 months)
            import time
            from datetime import datetime, timedelta

            monthly_data = []
            for i in range(11, -1, -1):  # Last 12 months
                month_start = datetime.now() - timedelta(days=30 * i)
                month_end = month_start + timedelta(days=30)

                cursor.execute("""
                    SELECT COUNT(*) FROM knowledge_entries
                    WHERE created_at >= ? AND created_at < ?
                """, (month_start.timestamp(), month_end.timestamp()))

                count = cursor.fetchone()[0]
                monthly_data.append({
                    'month': month_start.strftime('%Y-%m'),
                    'count': count
                })

            analytics['entry_growth_trend'] = monthly_data

            # 4. Quality metrics
            cursor.execute("""
                SELECT
                    AVG(CASE confidence_level
                        WHEN 'low' THEN 1
                        WHEN 'medium' THEN 2
                        WHEN 'high' THEN 3
                        WHEN 'verified' THEN 4
                        ELSE 0
                    END) as avg_confidence,
                    AVG(validation_score) as avg_validation,
                    AVG(success_rate) as avg_success_rate,
                    COUNT(*) as total_entries
                FROM knowledge_entries
            """)
            row = cursor.fetchone()
            analytics['quality_metrics'] = {
                'avg_confidence': row[0] or 0,
                'avg_validation_score': row[1] or 0,
                'avg_success_rate': row[2] or 0,
                'total_entries': row[3]
            }

            # 5. Relationship statistics
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)
            if cursor.fetchone():
                cursor.execute("""
                    SELECT
                        relationship_type,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM knowledge_relationships
                    GROUP BY relationship_type
                """)
                analytics['relationship_stats'] = [
                    {
                        'type': row[0],
                        'count': row[1],
                        'avg_confidence': row[2]
                    }
                    for row in cursor.fetchall()
                ]
            else:
                analytics['relationship_stats'] = []

            # 6. Knowledge type distribution
            cursor.execute("""
                SELECT knowledge_type, COUNT(*) as count
                FROM knowledge_entries
                GROUP BY knowledge_type
                ORDER BY count DESC
            """)
            analytics['type_distribution'] = {
                row[0]: row[1] for row in cursor.fetchall()
            }

            # 7. Entries per domain over time (last 30 days, top 5 domains)
            top_5_domains = [d['domain'] for d in analytics['top_domains'][:5]]
            domain_time_series = {}

            for domain in top_5_domains:
                daily_data = []
                for i in range(29, -1, -1):  # Last 30 days
                    day_start = datetime.now() - timedelta(days=i)
                    day_end = day_start + timedelta(days=1)

                    cursor.execute("""
                        SELECT COUNT(*) FROM knowledge_entries
                        WHERE domain = ? AND created_at >= ? AND created_at < ?
                    """, (domain, day_start.timestamp(), day_end.timestamp()))

                    count = cursor.fetchone()[0]
                    daily_data.append({
                        'date': day_start.strftime('%Y-%m-%d'),
                        'count': count
                    })

                domain_time_series[domain] = daily_data

            analytics['entries_per_domain_over_time'] = domain_time_series

            conn.close()

            return analytics

        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            if 'conn' in locals():
                conn.close()
            return {}

    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate quality report identifying entries needing attention.

        Returns:
            Dictionary with quality issues:
            - low_confidence_entries: Entries with confidence < 0.5
            - entries_with_flags: Entries with validation flags
            - orphaned_entries: Entries with no relationships
            - domains_low_quality: Domains with lowest success rates
            - unvalidated_entries: Entries never validated
            - recent_failures: Recent entries with low success rates
        """
        try:
            conn = sqlite3.connect(self.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            report = {}

            # 1. Low confidence entries (< medium)
            cursor.execute("""
                SELECT knowledge_id, domain, confidence_level,
                       content_json, created_at
                FROM knowledge_entries
                WHERE confidence_level IN ('low')
                ORDER BY created_at DESC
                LIMIT 50
            """)
            report['low_confidence_entries'] = [
                {
                    'knowledge_id': row['knowledge_id'],
                    'domain': row['domain'],
                    'confidence_level': row['confidence_level'],
                    'content': json.loads(row['content_json']) if row['content_json'] else {},
                    'age_days': (time.time() - row['created_at']) / 86400
                }
                for row in cursor.fetchall()
            ]

            # 2. Entries with validation flags
            cursor.execute("""
                SELECT knowledge_id, domain, validation_flags,
                       validation_score, content_json
                FROM knowledge_entries
                WHERE validation_flags IS NOT NULL
                  AND validation_flags != '[]'
                ORDER BY validation_score ASC
                LIMIT 50
            """)
            report['entries_with_flags'] = [
                {
                    'knowledge_id': row['knowledge_id'],
                    'domain': row['domain'],
                    'flags': json.loads(row['validation_flags']) if row['validation_flags'] else [],
                    'validation_score': row['validation_score'],
                    'content': json.loads(row['content_json']) if row['content_json'] else {}
                }
                for row in cursor.fetchall()
            ]

            # 3. Orphaned entries (no relationships)
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_relationships'
            """)
            if cursor.fetchone():
                cursor.execute("""
                    SELECT ke.knowledge_id, ke.domain, ke.content_json, ke.created_at
                    FROM knowledge_entries ke
                    LEFT JOIN knowledge_relationships kr
                      ON ke.knowledge_id = kr.source_id
                      OR ke.knowledge_id = kr.target_id
                    WHERE kr.source_id IS NULL
                    ORDER BY ke.created_at DESC
                    LIMIT 50
                """)
                report['orphaned_entries'] = [
                    {
                        'knowledge_id': row['knowledge_id'],
                        'domain': row['domain'],
                        'content': json.loads(row['content_json']) if row['content_json'] else {},
                        'age_days': (time.time() - row['created_at']) / 86400
                    }
                    for row in cursor.fetchall()
                ]
            else:
                report['orphaned_entries'] = []

            # 4. Domains with lowest success rates
            cursor.execute("""
                SELECT domain,
                       COUNT(*) as entry_count,
                       AVG(success_rate) as avg_success_rate,
                       AVG(validation_score) as avg_validation
                FROM knowledge_entries
                WHERE domain IS NOT NULL
                GROUP BY domain
                HAVING COUNT(*) >= 5
                ORDER BY avg_success_rate ASC
                LIMIT 10
            """)
            report['domains_low_quality'] = [
                {
                    'domain': row[0],
                    'entry_count': row[1],
                    'avg_success_rate': row[2],
                    'avg_validation': row[3]
                }
                for row in cursor.fetchall()
            ]

            # 5. Unvalidated entries
            cursor.execute("""
                SELECT knowledge_id, domain, content_json, created_at
                FROM knowledge_entries
                WHERE validated_at IS NULL
                  AND created_at < ?
                ORDER BY created_at ASC
                LIMIT 50
            """, (time.time() - 86400 * 7,))  # Older than 7 days
            report['unvalidated_entries'] = [
                {
                    'knowledge_id': row['knowledge_id'],
                    'domain': row['domain'],
                    'content': json.loads(row['content_json']) if row['content_json'] else {},
                    'age_days': (time.time() - row['created_at']) / 86400
                }
                for row in cursor.fetchall()
            ]

            # 6. Recent failures (low success rate, accessed recently)
            cursor.execute("""
                SELECT knowledge_id, domain, success_rate,
                       access_count, content_json
                FROM knowledge_entries
                WHERE success_rate < 0.3
                  AND access_count > 0
                ORDER BY access_count DESC
                LIMIT 30
            """)
            report['recent_failures'] = [
                {
                    'knowledge_id': row['knowledge_id'],
                    'domain': row['domain'],
                    'success_rate': row['success_rate'],
                    'access_count': row['access_count'],
                    'content': json.loads(row['content_json']) if row['content_json'] else {}
                }
                for row in cursor.fetchall()
            ]

            # Summary statistics
            report['summary'] = {
                'total_issues': (
                    len(report['low_confidence_entries']) +
                    len(report['entries_with_flags']) +
                    len(report['orphaned_entries']) +
                    len(report['unvalidated_entries']) +
                    len(report['recent_failures'])
                ),
                'low_confidence_count': len(report['low_confidence_entries']),
                'flagged_count': len(report['entries_with_flags']),
                'orphaned_count': len(report['orphaned_entries']),
                'unvalidated_count': len(report['unvalidated_entries']),
                'failing_count': len(report['recent_failures'])
            }

            conn.close()

            logger.info(f"Quality report generated: {report['summary']['total_issues']} issues found")
            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            if 'conn' in locals():
                conn.close()
            return {'summary': {'total_issues': 0}, 'error': str(e)}
