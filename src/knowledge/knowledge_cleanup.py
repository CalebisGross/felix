"""
Knowledge Brain Cleanup Utilities

Provides high-level cleanup operations for the knowledge brain system.
Includes pattern-based deletion, orphan cleanup, and batch operations.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3

from src.memory.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeCleanupManager:
    """
    High-level manager for knowledge brain cleanup operations.

    Provides safe, user-friendly methods for cleaning up knowledge entries,
    documents, and maintaining database integrity.
    """

    # Common exclusion patterns for unwanted directories
    DEFAULT_EXCLUSION_PATTERNS = [
        "*/.venv/*",
        "*/.venv/**/*",
        "*/venv/*",
        "*/venv/**/*",
        "*/node_modules/*",
        "*/node_modules/**/*",
        "*/.git/*",
        "*/.git/**/*",
        "*/__pycache__/*",
        "*/__pycache__/**/*",
        "*/dist/*",
        "*/build/*",
        "*/.pytest_cache/*",
        "*/.mypy_cache/*",
        "*/site-packages/*",
        "*/site-packages/**/*",
    ]

    def __init__(self, knowledge_store: Optional[KnowledgeStore] = None):
        """
        Initialize cleanup manager.

        Args:
            knowledge_store: KnowledgeStore instance (creates new if None)
        """
        self.knowledge_store = knowledge_store or KnowledgeStore()

    def preview_cleanup_by_pattern(self, pattern: str) -> Dict[str, Any]:
        """
        Preview what would be deleted by a cleanup operation.

        Args:
            pattern: Path pattern (glob or SQL LIKE)

        Returns:
            Dict with preview information
        """
        return self.knowledge_store.preview_delete_by_pattern(
            pattern,
            include_entries=True
        )

    def cleanup_virtual_environments(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove all documents and entries from virtual environment directories.

        Args:
            dry_run: If True, only preview without deleting

        Returns:
            Dict with cleanup results
        """
        results = {
            "patterns_processed": [],
            "total_docs_deleted": 0,
            "total_entries_deleted": 0,
            "errors": []
        }

        venv_patterns = [
            "*/.venv/*",
            "*/venv/*",
            "*/site-packages/*"
        ]

        for pattern in venv_patterns:
            try:
                result = self.knowledge_store.delete_documents_by_pattern(
                    pattern,
                    cascade_entries=True,
                    dry_run=dry_run
                )

                if "error" in result:
                    results["errors"].append({
                        "pattern": pattern,
                        "error": result["error"]
                    })
                else:
                    results["patterns_processed"].append(pattern)
                    results["total_docs_deleted"] += result.get("documents_deleted", 0)
                    results["total_entries_deleted"] += result.get("entries_deleted", 0)

            except Exception as e:
                logger.error(f"Error cleaning pattern {pattern}: {e}")
                results["errors"].append({
                    "pattern": pattern,
                    "error": str(e)
                })

        return results

    def cleanup_by_patterns(self, patterns: List[str],
                           cascade_entries: bool = True,
                           dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up documents matching multiple patterns.

        Args:
            patterns: List of path patterns (glob or SQL LIKE)
            cascade_entries: If True, also delete knowledge entries
            dry_run: If True, only preview without deleting

        Returns:
            Dict with cleanup results for each pattern
        """
        results = {
            "patterns": [],
            "total_docs_deleted": 0,
            "total_entries_deleted": 0,
            "errors": []
        }

        for pattern in patterns:
            try:
                result = self.knowledge_store.delete_documents_by_pattern(
                    pattern,
                    cascade_entries=cascade_entries,
                    dry_run=dry_run
                )

                if "error" in result:
                    results["errors"].append({
                        "pattern": pattern,
                        "error": result["error"]
                    })
                else:
                    pattern_result = {
                        "pattern": pattern,
                        "docs_deleted": result.get("documents_deleted", 0),
                        "entries_deleted": result.get("entries_deleted", 0)
                    }
                    results["patterns"].append(pattern_result)
                    results["total_docs_deleted"] += pattern_result["docs_deleted"]
                    results["total_entries_deleted"] += pattern_result["entries_deleted"]

            except Exception as e:
                logger.error(f"Error cleaning pattern {pattern}: {e}")
                results["errors"].append({
                    "pattern": pattern,
                    "error": str(e)
                })

        return results

    def cleanup_pending_documents(self, path_pattern: Optional[str] = None,
                                   dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up documents with 'pending' status.

        Args:
            path_pattern: Optional pattern to filter pending docs (None = all pending)
            dry_run: If True, only preview without deleting

        Returns:
            Dict with cleanup results
        """
        try:
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                # Get pending documents
                if path_pattern:
                    import fnmatch
                    cursor = conn.execute("""
                        SELECT doc_id, file_path
                        FROM document_sources
                        WHERE ingestion_status = 'pending'
                    """)
                    all_pending = cursor.fetchall()

                    pending_ids = [
                        doc_id
                        for doc_id, path in all_pending
                        if fnmatch.fnmatch(path, path_pattern)
                    ]
                else:
                    cursor = conn.execute("""
                        SELECT doc_id, file_path
                        FROM document_sources
                        WHERE ingestion_status = 'pending'
                    """)
                    all_pending = cursor.fetchall()
                    pending_ids = [row[0] for row in all_pending]

                if dry_run:
                    # Get sample file paths for preview
                    cursor = conn.execute("""
                        SELECT file_path
                        FROM document_sources
                        WHERE ingestion_status = 'pending'
                        LIMIT 10
                    """)
                    sample_paths = [row[0] for row in cursor.fetchall()]

                    return {
                        "document_count": len(pending_ids),
                        "would_delete": len(pending_ids),
                        "sample_paths": sample_paths
                    }

                if not pending_ids:
                    return {"documents_deleted": 0}

                # Delete pending documents (no entries to worry about)
                placeholders = ','.join('?' * len(pending_ids))
                cursor = conn.execute(f"""
                    DELETE FROM document_sources
                    WHERE doc_id IN ({placeholders})
                """, pending_ids)

                conn.commit()

                return {
                    "documents_deleted": cursor.rowcount
                }

        except Exception as e:
            logger.error(f"Error cleaning pending documents: {e}")
            return {"error": str(e)}

    def cleanup_failed_documents(self, max_age_days: int = 7,
                                  cascade_entries: bool = False,
                                  dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up old failed documents.

        Args:
            max_age_days: Minimum age in days for failed docs to delete
            cascade_entries: If True, also delete knowledge entries
            dry_run: If True, only preview without deleting

        Returns:
            Dict with cleanup results
        """
        return self.knowledge_store.delete_failed_documents(
            max_age_days=max_age_days,
            cascade_entries=cascade_entries,
            dry_run=dry_run
        )

    def cleanup_orphaned_entries(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up knowledge entries with no source document.

        Args:
            dry_run: If True, only preview without deleting

        Returns:
            Dict with cleanup results
        """
        return self.knowledge_store.delete_orphaned_entries(dry_run=dry_run)

    def get_cleanup_recommendations(self) -> Dict[str, Any]:
        """
        Analyze knowledge base and recommend cleanup actions.

        Returns:
            Dict with recommendations and statistics
        """
        recommendations = {
            "urgent": [],
            "suggested": [],
            "stats": {}
        }

        try:
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                # Check for unwanted virtual environment files
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM document_sources
                    WHERE file_path LIKE '%/.venv/%'
                       OR file_path LIKE '%/venv/%'
                       OR file_path LIKE '%/site-packages/%'
                """)
                venv_count = cursor.fetchone()[0]

                if venv_count > 0:
                    recommendations["urgent"].append({
                        "action": "cleanup_virtual_environments",
                        "reason": f"{venv_count} documents from virtual environments detected",
                        "severity": "high",
                        "count": venv_count
                    })

                # Check for pending documents
                cursor = conn.execute("""
                    SELECT COUNT(*),
                           ROUND((julianday('now') - julianday(added_at, 'unixepoch')) * 24) as hours_old
                    FROM document_sources
                    WHERE ingestion_status = 'pending'
                    GROUP BY hours_old > 24
                """)
                pending_stats = cursor.fetchall()

                if pending_stats:
                    for count, is_old in pending_stats:
                        if is_old:
                            recommendations["urgent"].append({
                                "action": "cleanup_pending_documents",
                                "reason": f"{count} pending documents older than 24 hours",
                                "severity": "medium",
                                "count": count
                            })

                # Check for orphaned entries
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM knowledge_entries
                    WHERE source_doc_id IS NOT NULL
                    AND source_doc_id NOT IN (SELECT doc_id FROM document_sources)
                """)
                orphan_count = cursor.fetchone()[0]

                if orphan_count > 0:
                    recommendations["suggested"].append({
                        "action": "cleanup_orphaned_entries",
                        "reason": f"{orphan_count} orphaned knowledge entries detected",
                        "severity": "low",
                        "count": orphan_count
                    })

                # Check for failed documents
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM document_sources
                    WHERE ingestion_status = 'failed'
                    AND added_at < ?
                """, (time.time() - 7 * 24 * 3600,))
                failed_count = cursor.fetchone()[0]

                if failed_count > 0:
                    recommendations["suggested"].append({
                        "action": "cleanup_failed_documents",
                        "reason": f"{failed_count} failed documents older than 7 days",
                        "severity": "low",
                        "count": failed_count
                    })

                # Get overall stats
                cursor = conn.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM document_sources) as total_docs,
                        (SELECT COUNT(*) FROM knowledge_entries) as total_entries,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='pending') as pending,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='failed') as failed,
                        (SELECT COUNT(*) FROM document_sources WHERE ingestion_status='completed') as completed
                """)
                stats = cursor.fetchone()

                recommendations["stats"] = {
                    "total_documents": stats[0],
                    "total_entries": stats[1],
                    "pending_documents": stats[2],
                    "failed_documents": stats[3],
                    "completed_documents": stats[4]
                }

        except Exception as e:
            logger.error(f"Error getting cleanup recommendations: {e}")
            recommendations["error"] = str(e)

        return recommendations

    def execute_recommended_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute all recommended cleanup actions.

        Args:
            dry_run: If True, only preview without deleting

        Returns:
            Dict with results for each action
        """
        results = {
            "actions": [],
            "total_docs_deleted": 0,
            "total_entries_deleted": 0
        }

        recommendations = self.get_cleanup_recommendations()

        for rec in recommendations.get("urgent", []) + recommendations.get("suggested", []):
            action = rec["action"]

            try:
                if action == "cleanup_virtual_environments":
                    result = self.cleanup_virtual_environments(dry_run=dry_run)
                elif action == "cleanup_pending_documents":
                    result = self.cleanup_pending_documents(dry_run=dry_run)
                elif action == "cleanup_orphaned_entries":
                    result = self.cleanup_orphaned_entries(dry_run=dry_run)
                elif action == "cleanup_failed_documents":
                    result = self.cleanup_failed_documents(dry_run=dry_run)
                else:
                    continue

                results["actions"].append({
                    "action": action,
                    "result": result
                })

                results["total_docs_deleted"] += result.get("documents_deleted", 0)
                results["total_docs_deleted"] += result.get("total_docs_deleted", 0)
                results["total_entries_deleted"] += result.get("entries_deleted", 0)
                results["total_entries_deleted"] += result.get("total_entries_deleted", 0)

            except Exception as e:
                logger.error(f"Error executing {action}: {e}")
                results["actions"].append({
                    "action": action,
                    "error": str(e)
                })

        return results

    def get_document_statistics_by_path(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get statistics showing which path prefixes have the most documents.

        Args:
            limit: Maximum number of path prefixes to return

        Returns:
            List of dicts with path prefix statistics
        """
        try:
            with sqlite3.connect(self.knowledge_store.storage_path) as conn:
                # Get path statistics by extracting directory prefix
                cursor = conn.execute("""
                    SELECT
                        SUBSTR(file_path, 1, 60) as path_prefix,
                        ingestion_status,
                        COUNT(*) as doc_count,
                        SUM(concept_count) as total_concepts
                    FROM document_sources
                    GROUP BY path_prefix, ingestion_status
                    ORDER BY doc_count DESC
                    LIMIT ?
                """, (limit,))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "path_prefix": row[0],
                        "status": row[1],
                        "document_count": row[2],
                        "concept_count": row[3] or 0
                    })

                return results

        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return []


# Convenience functions for common operations

def quick_cleanup_venv(dry_run: bool = True) -> Dict[str, Any]:
    """
    Quick function to clean up virtual environment files.

    Args:
        dry_run: If True, only preview without deleting

    Returns:
        Cleanup results
    """
    manager = KnowledgeCleanupManager()
    return manager.cleanup_virtual_environments(dry_run=dry_run)


def quick_cleanup_pending(path_pattern: Optional[str] = None,
                          dry_run: bool = True) -> Dict[str, Any]:
    """
    Quick function to clean up pending documents.

    Args:
        path_pattern: Optional pattern to filter pending docs
        dry_run: If True, only preview without deleting

    Returns:
        Cleanup results
    """
    manager = KnowledgeCleanupManager()
    return manager.cleanup_pending_documents(path_pattern=path_pattern, dry_run=dry_run)


def get_cleanup_report() -> Dict[str, Any]:
    """
    Get a comprehensive cleanup report with recommendations.

    Returns:
        Report with recommendations and statistics
    """
    manager = KnowledgeCleanupManager()
    return manager.get_cleanup_recommendations()


# Import time module for cleanup_failed_documents
import time
