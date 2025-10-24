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

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge that can be stored."""
    TASK_RESULT = "task_result"
    AGENT_INSIGHT = "agent_insight"
    PATTERN_RECOGNITION = "pattern_recognition"
    FAILURE_ANALYSIS = "failure_analysis"
    OPTIMIZATION_DATA = "optimization_data"
    DOMAIN_EXPERTISE = "domain_expertise"

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
    
class KnowledgeStore:
    """
    Persistent knowledge storage system.
    
    Stores and retrieves knowledge entries across multiple framework runs,
    enabling learning and knowledge accumulation over time.
    """
    
    def __init__(self, storage_path: str = "felix_knowledge.db", 
                 enable_compression: bool = True):
        """
        Initialize knowledge store.
        
        Args:
            storage_path: Path to SQLite database file
            enable_compression: Whether to compress large content
        """
        self.storage_path = Path(storage_path)
        self.enable_compression = enable_compression
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
                    related_entries_json TEXT DEFAULT '[]'
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
    
    def _compress_content(self, content: Dict[str, Any]) -> bytes:
        """Compress large content using pickle."""
        return pickle.dumps(content)
    
    def _decompress_content(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress content from bytes."""
        return pickle.loads(compressed_data)
    
    def store_knowledge(self, knowledge_type: KnowledgeType,
                       content: Dict[str, Any],
                       confidence_level: ConfidenceLevel,
                       source_agent: str,
                       domain: str,
                       tags: Optional[List[str]] = None) -> str:
        """
        Store new knowledge entry.

        Args:
            knowledge_type: Type of knowledge
            content: Knowledge content
            confidence_level: Confidence in this knowledge
            source_agent: Agent that generated this knowledge
            domain: Domain this knowledge applies to
            tags: Optional tags for categorization

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

        # Check if entry already exists (for deduplication logging)
        with sqlite3.connect(self.storage_path) as conn:
            existing = conn.execute(
                "SELECT created_at FROM knowledge_entries WHERE knowledge_id = ?",
                (knowledge_id,)
            ).fetchone()

            if existing:
                logger.info(f"   📌 DEDUPLICATION: Entry exists (created: {time.ctime(existing[0])}) - updating")
                # Preserve original created_at for deduplication
                created_at = existing[0]
                updated_at = time.time()
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

        # Determine storage method based on content size
        content_json = json.dumps(content)
        content_compressed = None

        if self.enable_compression and len(content_json) > 1000:
            content_compressed = self._compress_content(content)
            content_json = ""  # Clear JSON to save space

        with sqlite3.connect(self.storage_path) as conn:
            # Store main entry (INSERT OR REPLACE will update if exists)
            logger.info(f"   📝 Executing INSERT OR REPLACE INTO knowledge_entries...")
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, content_compressed,
                 confidence_level, source_agent, domain, tags_json,
                 created_at, updated_at, access_count, success_rate, related_entries_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps([])
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

        logger.info(f"   ✅ Returning {len(entries)} knowledge entries")

        return entries
    
    def _row_to_entry(self, row, conn=None) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry."""
        (knowledge_id, knowledge_type, content_json, content_compressed,
         confidence_level, source_agent, domain, tags_json,
         created_at, updated_at, access_count, success_rate, related_entries_json) = row
        
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
            related_entries=json.loads(related_entries_json)
        )
    
    def _increment_access_count(self, knowledge_id: str) -> None:
        """Increment access count for knowledge entry."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                UPDATE knowledge_entries 
                SET access_count = access_count + 1 
                WHERE knowledge_id = ?
            """, (knowledge_id,))
    
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
            
            return {
                "total_entries": total_entries,
                "by_type": by_type,
                "by_domain": by_domain,
                "by_confidence": by_confidence,
                "average_success_rate": avg_success_rate,
                "storage_path": str(self.storage_path)
            }
    
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
