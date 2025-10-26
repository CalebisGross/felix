"""
Automated backup management for database migrations.

Creates timestamped backups before migrations and provides
restore functionality if migrations fail.
"""

import shutil
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BackupManager:
    """
    Manages database backups for safe migrations.

    Features:
    - Timestamped backups before each migration
    - Automatic old backup cleanup
    - Restore functionality
    - Backup verification
    """

    def __init__(self, backup_dir: str = "backups"):
        """
        Initialize backup manager.

        Args:
            backup_dir: Directory to store backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup manager initialized: {self.backup_dir}")

    def create_backup(self, db_path: Path, prefix: Optional[str] = None) -> Path:
        """
        Create a timestamped backup of a database.

        Args:
            db_path: Path to the database file
            prefix: Optional prefix for backup filename

        Returns:
            Path to the backup file

        Raises:
            IOError: If backup creation fails
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = db_path.stem  # filename without extension

        if prefix:
            backup_name = f"{prefix}_{db_name}_{timestamp}.db"
        else:
            backup_name = f"{db_name}_{timestamp}.db"

        backup_path = self.backup_dir / backup_name

        try:
            logger.info(f"Creating backup: {db_path} → {backup_path}")
            shutil.copy2(db_path, backup_path)

            # Verify backup
            if backup_path.exists() and backup_path.stat().st_size > 0:
                logger.info(f"✓ Backup created successfully: {backup_path}")
                logger.info(f"  Size: {backup_path.stat().st_size:,} bytes")
                return backup_path
            else:
                raise IOError(f"Backup verification failed: {backup_path}")

        except Exception as e:
            logger.error(f"✗ Backup creation failed: {e}")
            raise IOError(f"Failed to create backup: {e}") from e

    def backup_all(self, databases: Optional[List[Path]] = None) -> Dict[str, Path]:
        """
        Create backups of all Felix databases.

        Args:
            databases: Optional list of database paths. If None, discovers databases automatically.

        Returns:
            Dictionary mapping database names to backup paths
        """
        if databases is None:
            # Auto-discover Felix databases
            databases = self._discover_databases()

        if not databases:
            logger.warning("No databases found to backup")
            return {}

        logger.info(f"\n{'='*60}")
        logger.info(f"Creating backups for {len(databases)} databases")
        logger.info(f"{'='*60}\n")

        backups = {}

        for db_path in databases:
            try:
                backup_path = self.create_backup(db_path, prefix="pre_migration")
                backups[db_path.name] = backup_path
            except Exception as e:
                logger.error(f"Failed to backup {db_path}: {e}")

        logger.info(f"\n✓ Created {len(backups)} backups in {self.backup_dir}")
        return backups

    def _discover_databases(self) -> List[Path]:
        """
        Discover all Felix database files in the current directory.

        Returns:
            List of database file paths
        """
        # Look for Felix database files
        db_patterns = [
            "felix_knowledge.db",
            "felix_task_memory.db",
            "felix_memory.db",
            "felix_workflow_history.db",
            "felix_agent_performance.db",
            "felix_system_actions.db"
        ]

        databases = []
        current_dir = Path.cwd()

        for pattern in db_patterns:
            db_path = current_dir / pattern
            if db_path.exists():
                databases.append(db_path)
                logger.debug(f"Found database: {db_path}")

        return databases

    def restore_backup(self, backup_path: Path, target_path: Path) -> bool:
        """
        Restore a database from a backup.

        Args:
            backup_path: Path to the backup file
            target_path: Path to restore to

        Returns:
            True if successful, False otherwise
        """
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False

        try:
            logger.info(f"Restoring backup: {backup_path} → {target_path}")

            # Create backup of current file if it exists
            if target_path.exists():
                current_backup = self.create_backup(target_path, prefix="pre_restore")
                logger.info(f"Current file backed up to: {current_backup}")

            # Restore from backup
            shutil.copy2(backup_path, target_path)

            logger.info(f"✓ Restore complete: {target_path}")
            return True

        except Exception as e:
            logger.error(f"✗ Restore failed: {e}")
            return False

    def list_backups(self, db_name: Optional[str] = None) -> List[Dict]:
        """
        List available backups.

        Args:
            db_name: Optional database name to filter by

        Returns:
            List of backup info dictionaries
        """
        backups = []

        for backup_file in sorted(self.backup_dir.glob("*.db"), reverse=True):
            if db_name and db_name not in backup_file.name:
                continue

            backups.append({
                'filename': backup_file.name,
                'path': backup_file,
                'size': backup_file.stat().st_size,
                'created': datetime.fromtimestamp(backup_file.stat().st_ctime),
                'age_hours': (time.time() - backup_file.stat().st_ctime) / 3600
            })

        return backups

    def cleanup_old_backups(self, max_age_days: int = 30, keep_minimum: int = 5) -> int:
        """
        Clean up old backup files.

        Args:
            max_age_days: Delete backups older than this many days
            keep_minimum: Always keep at least this many recent backups per database

        Returns:
            Number of backups deleted
        """
        logger.info(f"Cleaning up backups older than {max_age_days} days (keeping minimum {keep_minimum})")

        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        deleted = 0

        # Group backups by database name
        backup_groups = {}
        for backup_file in self.backup_dir.glob("*.db"):
            # Extract database name from backup filename
            # e.g., "pre_migration_felix_knowledge_20250124_143022.db" → "felix_knowledge"
            parts = backup_file.stem.split('_')
            if len(parts) >= 3:
                db_name = '_'.join(parts[1:-2])  # Skip prefix and timestamp
                if db_name not in backup_groups:
                    backup_groups[db_name] = []
                backup_groups[db_name].append(backup_file)

        # Clean up each group
        for db_name, backups in backup_groups.items():
            # Sort by modification time (newest first)
            backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Keep minimum recent backups
            backups_to_check = backups[keep_minimum:]

            for backup_file in backups_to_check:
                age = current_time - backup_file.stat().st_mtime

                if age > max_age_seconds:
                    try:
                        backup_file.unlink()
                        logger.info(f"Deleted old backup: {backup_file.name}")
                        deleted += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {backup_file}: {e}")

        logger.info(f"✓ Cleanup complete: {deleted} backups deleted")
        return deleted

    def get_backup_summary(self) -> Dict:
        """
        Get summary of backup status.

        Returns:
            Dictionary with backup statistics
        """
        backups = self.list_backups()

        if not backups:
            return {
                'total_backups': 0,
                'total_size_mb': 0,
                'oldest_backup': None,
                'newest_backup': None
            }

        total_size = sum(b['size'] for b in backups)

        return {
            'total_backups': len(backups),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_backup': backups[-1]['filename'] if backups else None,
            'newest_backup': backups[0]['filename'] if backups else None,
            'backup_directory': str(self.backup_dir)
        }

    def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify a backup file is valid SQLite database.

        Args:
            backup_path: Path to backup file

        Returns:
            True if backup is valid, False otherwise
        """
        try:
            import sqlite3

            # Try to open as SQLite database
            conn = sqlite3.connect(backup_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            tables = cursor.fetchall()
            conn.close()

            if tables:
                logger.info(f"✓ Backup verified: {backup_path}")
                return True
            else:
                logger.warning(f"⚠ Backup has no tables: {backup_path}")
                return False

        except Exception as e:
            logger.error(f"✗ Backup verification failed: {e}")
            return False
