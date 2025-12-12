"""
Extended Backup Manager for Knowledge Base

Provides JSON export/import with compression, selective backups,
and conflict resolution for knowledge entries. Extends the base
BackupManager with knowledge-specific functionality.

Features:
- JSON export with gzip compression
- Selective export by domain, date range, confidence level
- Import with conflict resolution (skip, replace, merge)
- Integrity verification
- Scheduled backup support
"""

import json
import gzip
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from src.migration.backup_manager import BackupManager
from src.memory.knowledge_store import KnowledgeStore, ConfidenceLevel

logger = logging.getLogger(__name__)


class KnowledgeBackupManager:
    """
    Extended backup manager for knowledge base with JSON export/import.

    Complements the existing BackupManager for SQLite backups with
    knowledge-specific JSON format for portability and selective operations.
    """

    def __init__(self, knowledge_store: KnowledgeStore, backup_dir: str = "backups/knowledge"):
        """
        Initialize knowledge backup manager.

        Args:
            knowledge_store: Knowledge store instance
            backup_dir: Directory for knowledge backups
        """
        self.knowledge_store = knowledge_store
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Also initialize base backup manager for SQLite backups
        self.db_backup_manager = BackupManager(backup_dir="backups/database")

        logger.info(f"Knowledge backup manager initialized: {self.backup_dir}")

    def export_to_json(
        self,
        output_path: Optional[str] = None,
        compress: bool = True,
        domain: Optional[str] = None,
        confidence_level: Optional[ConfidenceLevel] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_relationships: bool = True
    ) -> str:
        """
        Export knowledge entries to JSON format with optional compression.

        Args:
            output_path: Output file path (auto-generated if None)
            compress: Use gzip compression
            domain: Filter by domain
            confidence_level: Filter by minimum confidence level
            start_date: Filter by start date
            end_date: Filter by end date
            include_relationships: Include relationship data

        Returns:
            Path to exported file

        Raises:
            IOError: If export fails
        """
        try:
            # Generate filename if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"knowledge_export_{timestamp}.json"
                if compress:
                    filename += ".gz"
                output_path = str(self.backup_dir / filename)

            # Build query filters
            query = "SELECT * FROM knowledge_entries WHERE 1=1"
            params = []

            if domain:
                query += " AND domain = ?"
                params.append(domain)

            if confidence_level:
                # Map confidence level to filter
                confidence_order = {
                    ConfidenceLevel.LOW: 1,
                    ConfidenceLevel.MEDIUM: 2,
                    ConfidenceLevel.HIGH: 3,
                    ConfidenceLevel.VERIFIED: 4
                }
                min_level = confidence_order[confidence_level]
                confidence_values = [k.value for k, v in confidence_order.items() if v >= min_level]
                placeholders = ','.join('?' * len(confidence_values))
                query += f" AND confidence_level IN ({placeholders})"
                params.extend(confidence_values)

            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date.timestamp())

            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date.timestamp())

            # Execute query
            conn = sqlite3.connect(self.knowledge_store.storage_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)

            # Build export data
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'export_version': '1.0',
                'filters': {
                    'domain': domain,
                    'confidence_level': confidence_level.value if confidence_level else None,
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None
                },
                'entries': []
            }

            for row in cursor.fetchall():
                entry = dict(row)

                # Convert binary embedding to null (not portable)
                if 'embedding' in entry:
                    entry['embedding'] = None

                export_data['entries'].append(entry)

            # Fetch relationships if requested
            if include_relationships:
                cursor.execute("SELECT * FROM knowledge_relationships")
                export_data['relationships'] = [dict(row) for row in cursor.fetchall()]

            conn.close()

            # Write to file
            json_data = json.dumps(export_data, indent=2, default=str)

            if compress:
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    f.write(json_data)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)

            logger.info(f"Exported {len(export_data['entries'])} entries to {output_path}")

            if compress:
                import os
                original_size = len(json_data.encode('utf-8'))
                compressed_size = os.path.getsize(output_path)
                ratio = (1 - compressed_size / original_size) * 100
                logger.info(f"Compression: {original_size:,} → {compressed_size:,} bytes ({ratio:.1f}% reduction)")

            return output_path

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise IOError(f"Failed to export knowledge: {e}") from e

    def import_from_json(
        self,
        import_path: str,
        conflict_strategy: str = "skip",
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Import knowledge entries from JSON backup.

        Args:
            import_path: Path to JSON or JSON.gz file
            conflict_strategy: How to handle conflicts:
                - "skip": Skip entries that already exist
                - "replace": Replace existing entries
                - "merge": Merge with existing (keep highest confidence)
            dry_run: If True, analyze without importing

        Returns:
            Dictionary with import statistics

        Raises:
            IOError: If import fails
            ValueError: If invalid conflict strategy
        """
        if conflict_strategy not in ("skip", "replace", "merge"):
            raise ValueError(f"Invalid conflict strategy: {conflict_strategy}")

        try:
            # Read JSON data
            import_path = Path(import_path)

            if import_path.suffix == '.gz':
                with gzip.open(import_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(import_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            logger.info(f"Loaded {len(data.get('entries', []))} entries from {import_path}")

            # Statistics
            stats = {
                'total_entries': len(data.get('entries', [])),
                'imported': 0,
                'skipped': 0,
                'replaced': 0,
                'merged': 0,
                'errors': 0,
                'dry_run': dry_run
            }

            if dry_run:
                logger.info("DRY RUN mode: analyzing without importing")

            # Process entries
            for entry_data in data.get('entries', []):
                knowledge_id = entry_data.get('knowledge_id')

                if not knowledge_id:
                    stats['errors'] += 1
                    continue

                try:
                    # Check if entry exists
                    existing = self.knowledge_store.get_entry_by_id(knowledge_id)

                    if existing and conflict_strategy == "skip":
                        stats['skipped'] += 1
                        continue

                    if existing and conflict_strategy == "replace":
                        if not dry_run:
                            self._replace_entry(knowledge_id, entry_data)
                        stats['replaced'] += 1

                    elif existing and conflict_strategy == "merge":
                        if not dry_run:
                            self._merge_entry(existing, entry_data)
                        stats['merged'] += 1

                    else:
                        # New entry
                        if not dry_run:
                            self._import_entry(entry_data)
                        stats['imported'] += 1

                except Exception as e:
                    logger.error(f"Error importing entry {knowledge_id}: {e}")
                    stats['errors'] += 1

            # Import relationships if present
            if not dry_run and 'relationships' in data:
                self._import_relationships(data['relationships'])

            logger.info(f"Import complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise IOError(f"Failed to import knowledge: {e}") from e

    def _import_entry(self, entry_data: Dict[str, Any]):
        """Import a new knowledge entry."""
        # Reconstruct entry and insert
        conn = sqlite3.connect(self.knowledge_store.storage_path)
        cursor = conn.cursor()

        # Build INSERT statement (excluding embedding which is not portable)
        columns = [k for k in entry_data.keys() if k != 'embedding']
        placeholders = ','.join(['?' for _ in columns])
        col_names = ','.join(columns)

        query = f"INSERT INTO knowledge_entries ({col_names}) VALUES ({placeholders})"
        values = [entry_data[col] for col in columns]

        cursor.execute(query, values)
        conn.commit()
        conn.close()

    def _replace_entry(self, knowledge_id: str, entry_data: Dict[str, Any]):
        """Replace existing entry with imported data."""
        # Delete old entry
        self.knowledge_store.delete_knowledge(knowledge_id)

        # Import new entry
        self._import_entry(entry_data)

    def _merge_entry(self, existing, entry_data: Dict[str, Any]):
        """Merge imported entry with existing (keep highest confidence)."""
        from src.memory.knowledge_store import ConfidenceLevel

        existing_confidence = existing.confidence_level
        imported_confidence = ConfidenceLevel(entry_data.get('confidence_level'))

        confidence_order = {
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.VERIFIED: 4
        }

        # Keep highest confidence version
        if confidence_order[imported_confidence] > confidence_order[existing_confidence]:
            self._replace_entry(entry_data['knowledge_id'], entry_data)
        # Otherwise keep existing (do nothing)

    def _import_relationships(self, relationships: List[Dict[str, Any]]):
        """Import knowledge relationships."""
        conn = sqlite3.connect(self.knowledge_store.storage_path)
        cursor = conn.cursor()

        for rel in relationships:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_relationships
                    (source_id, target_id, relationship_type, confidence)
                    VALUES (?, ?, ?, ?)
                """, (
                    rel['source_id'],
                    rel['target_id'],
                    rel['relationship_type'],
                    rel['confidence']
                ))
            except Exception as e:
                logger.debug(f"Relationship import error (may be duplicate): {e}")

        conn.commit()
        conn.close()

    def create_scheduled_backup(
        self,
        include_database: bool = True,
        include_json: bool = True,
        compress: bool = True
    ) -> Dict[str, str]:
        """
        Create a complete backup (database + JSON).

        Args:
            include_database: Backup SQLite database
            include_json: Export to JSON
            compress: Use compression for JSON

        Returns:
            Dictionary with backup file paths
        """
        backups = {}

        try:
            # SQLite backup
            if include_database:
                db_path = Path(self.knowledge_store.storage_path)
                backup_path = self.db_backup_manager.create_backup(
                    db_path,
                    prefix="scheduled"
                )
                backups['database'] = str(backup_path)
                logger.info(f"Database backup created: {backup_path}")

            # JSON export
            if include_json:
                json_path = self.export_to_json(compress=compress)
                backups['json'] = json_path
                logger.info(f"JSON export created: {json_path}")

            return backups

        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
            return backups

    def verify_backup(self, backup_path: str) -> bool:
        """
        Verify backup file integrity.

        Args:
            backup_path: Path to backup file (.db or .json/.json.gz)

        Returns:
            True if valid, False otherwise
        """
        backup_path = Path(backup_path)

        try:
            if backup_path.suffix == '.db':
                # Verify SQLite database
                return self.db_backup_manager.verify_backup(backup_path)

            elif backup_path.suffix in ('.json', '.gz'):
                # Verify JSON format
                if backup_path.suffix == '.gz':
                    with gzip.open(backup_path, 'rt') as f:
                        data = json.load(f)
                else:
                    with open(backup_path, 'r') as f:
                        data = json.load(f)

                # Check required fields
                if 'entries' not in data:
                    logger.error("JSON backup missing 'entries' field")
                    return False

                logger.info(f"JSON backup verified: {len(data['entries'])} entries")
                return True

            else:
                logger.error(f"Unknown backup format: {backup_path.suffix}")
                return False

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    def list_backups(self, backup_type: str = "all") -> Dict[str, List[Dict]]:
        """
        List available backups.

        Args:
            backup_type: "all", "database", or "json"

        Returns:
            Dictionary with backup lists by type
        """
        backups = {
            'database': [],
            'json': []
        }

        if backup_type in ("all", "database"):
            backups['database'] = self.db_backup_manager.list_backups("felix_knowledge")

        if backup_type in ("all", "json"):
            for json_file in sorted(self.backup_dir.glob("knowledge_export_*.json*"), reverse=True):
                backups['json'].append({
                    'filename': json_file.name,
                    'path': str(json_file),
                    'size': json_file.stat().st_size,
                    'created': datetime.fromtimestamp(json_file.stat().st_ctime),
                    'compressed': json_file.suffix == '.gz'
                })

        return backups

    def cleanup_old_backups(self, max_age_days: int = 30, keep_minimum: int = 5) -> Dict[str, int]:
        """
        Clean up old backups (both database and JSON).

        Args:
            max_age_days: Delete backups older than this many days
            keep_minimum: Keep at least this many recent backups

        Returns:
            Dictionary with deletion counts by type
        """
        deleted = {
            'database': 0,
            'json': 0
        }

        # Clean database backups
        deleted['database'] = self.db_backup_manager.cleanup_old_backups(
            max_age_days,
            keep_minimum
        )

        # Clean JSON backups
        max_age_seconds = max_age_days * 24 * 3600
        import time
        current_time = time.time()

        json_backups = sorted(self.backup_dir.glob("knowledge_export_*.json*"),
                             key=lambda p: p.stat().st_mtime,
                             reverse=True)

        for backup_file in json_backups[keep_minimum:]:
            age = current_time - backup_file.stat().st_mtime

            if age > max_age_seconds:
                try:
                    backup_file.unlink()
                    logger.info(f"Deleted old JSON backup: {backup_file.name}")
                    deleted['json'] += 1
                except Exception as e:
                    logger.error(f"Failed to delete {backup_file}: {e}")

        logger.info(f"Cleanup complete: {deleted}")
        return deleted
