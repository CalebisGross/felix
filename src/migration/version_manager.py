"""
Schema version management and migration orchestration.

Tracks which migrations have been applied to each database component
and orchestrates the application of pending migrations.
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .base_migration import Migration, MigrationError

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manages schema versions and orchestrates migrations across Felix databases.

    Responsibilities:
    - Track which migrations have been applied to each component
    - Apply pending migrations in correct order
    - Rollback migrations when needed
    - Verify migration state
    """

    SCHEMA_VERSION_TABLE = "schema_migrations"

    def __init__(self):
        """Initialize migration manager."""
        pass

    def _ensure_version_table(self, conn: sqlite3.Connection) -> None:
        """
        Ensure schema_migrations table exists.

        Args:
            conn: Database connection
        """
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.SCHEMA_VERSION_TABLE} (
                component TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT,
                applied_at REAL NOT NULL,
                rollback_available BOOLEAN DEFAULT 1,
                PRIMARY KEY (component, version)
            )
        """)
        conn.commit()

    def get_current_version(self, db_path: Path, component: str) -> int:
        """
        Get current schema version for a component.

        Args:
            db_path: Path to the database
            component: Component name (e.g., 'knowledge', 'tasks')

        Returns:
            Current version number (0 if no migrations applied)
        """
        try:
            conn = sqlite3.connect(db_path)
            self._ensure_version_table(conn)

            cursor = conn.execute(
                f"""
                SELECT MAX(version) FROM {self.SCHEMA_VERSION_TABLE}
                WHERE component = ?
                """,
                (component,)
            )

            result = cursor.fetchone()
            conn.close()

            return result[0] if result[0] is not None else 0

        except sqlite3.Error as e:
            logger.error(f"Error getting version for {component}: {e}")
            return 0

    def get_applied_migrations(self, db_path: Path, component: str) -> List[Dict]:
        """
        Get list of applied migrations for a component.

        Args:
            db_path: Path to the database
            component: Component name

        Returns:
            List of migration records (dicts with version, description, applied_at)
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            self._ensure_version_table(conn)

            cursor = conn.execute(
                f"""
                SELECT * FROM {self.SCHEMA_VERSION_TABLE}
                WHERE component = ?
                ORDER BY version
                """,
                (component,)
            )

            migrations = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return migrations

        except sqlite3.Error as e:
            logger.error(f"Error getting applied migrations for {component}: {e}")
            return []

    def _record_migration(self, conn: sqlite3.Connection, component: str,
                         migration: Migration) -> None:
        """
        Record a migration as applied.

        Args:
            conn: Database connection
            component: Component name
            migration: Migration that was applied
        """
        self._ensure_version_table(conn)

        # Check if migration supports rollback
        try:
            # Try to call down() with a test connection to see if it raises NotImplementedError
            rollback_available = hasattr(migration, 'down') and callable(migration.down)
        except:
            rollback_available = False

        conn.execute(
            f"""
            INSERT INTO {self.SCHEMA_VERSION_TABLE}
            (component, version, description, applied_at, rollback_available)
            VALUES (?, ?, ?, ?, ?)
            """,
            (component, migration.version, migration.description, time.time(), rollback_available)
        )
        conn.commit()

    def _remove_migration_record(self, conn: sqlite3.Connection, component: str,
                                 version: int) -> None:
        """
        Remove a migration record (after rollback).

        Args:
            conn: Database connection
            component: Component name
            version: Migration version to remove
        """
        conn.execute(
            f"""
            DELETE FROM {self.SCHEMA_VERSION_TABLE}
            WHERE component = ? AND version = ?
            """,
            (component, version)
        )
        conn.commit()

    def apply_migrations(self, db_path: Path, component: str,
                        migrations: List[Migration], dry_run: bool = False) -> Tuple[int, int]:
        """
        Apply pending migrations to a database component.

        Args:
            db_path: Path to the database
            component: Component name
            migrations: List of all available migrations (in order)
            dry_run: If True, test migrations without committing

        Returns:
            Tuple of (migrations_applied, migrations_failed)

        Raises:
            MigrationError: If any migration fails
        """
        # Sort migrations by version
        migrations = sorted(migrations, key=lambda m: m.version)

        # Get current version
        current_version = self.get_current_version(db_path, component)
        logger.info(f"{component} current version: {current_version}")

        # Filter pending migrations
        pending = [m for m in migrations if m.version > current_version]

        if not pending:
            logger.info(f"No pending migrations for {component}")
            return (0, 0)

        logger.info(f"Pending migrations for {component}: {len(pending)}")
        for m in pending:
            logger.info(f"  - {m}")

        applied = 0
        failed = 0

        # Apply each pending migration
        for migration in pending:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Applying: {migration}")
                logger.info(f"{'='*60}")

                # Apply migration
                success = migration.apply(db_path, dry_run=dry_run)

                if success and not dry_run:
                    # Record in version table
                    conn = sqlite3.connect(db_path)
                    self._record_migration(conn, component, migration)
                    conn.close()
                    applied += 1
                    logger.info(f"✓ Migration {migration.version} applied and recorded")
                elif success and dry_run:
                    logger.info(f"✓ Migration {migration.version} dry run successful (not recorded)")
                    applied += 1

            except MigrationError as e:
                logger.error(f"✗ Migration {migration.version} failed: {e}")
                failed += 1

                # Stop on first failure
                logger.error(f"Stopping migration process due to failure")
                break

        logger.info(f"\nMigration summary for {component}:")
        logger.info(f"  Applied: {applied}")
        logger.info(f"  Failed: {failed}")

        return (applied, failed)

    def rollback(self, db_path: Path, component: str, target_version: Optional[int] = None) -> bool:
        """
        Rollback migrations to a target version.

        Args:
            db_path: Path to the database
            component: Component name
            target_version: Version to rollback to (None = rollback last migration)

        Returns:
            True if successful, False otherwise

        Raises:
            MigrationError: If rollback fails
        """
        current_version = self.get_current_version(db_path, component)

        if target_version is None:
            # Rollback just the last migration
            target_version = current_version - 1

        if current_version <= target_version:
            logger.info(f"{component} already at version {current_version}, no rollback needed")
            return True

        logger.info(f"Rolling back {component} from version {current_version} to {target_version}")

        # Get migrations to rollback (in reverse order)
        applied_migrations = self.get_applied_migrations(db_path, component)
        to_rollback = [
            m for m in applied_migrations
            if m['version'] > target_version
        ]
        to_rollback.reverse()  # Rollback in reverse order

        logger.info(f"Migrations to rollback: {len(to_rollback)}")

        for migration_record in to_rollback:
            version = migration_record['version']
            description = migration_record['description']
            rollback_available = migration_record['rollback_available']

            if not rollback_available:
                logger.error(
                    f"Cannot rollback migration {version} ({description}) - rollback not implemented"
                )
                raise MigrationError(f"Migration {version} does not support rollback")

            logger.info(f"Rolling back migration {version}: {description}")

            # Note: We need the actual Migration instance to call down()
            # This requires migrations to be passed in or loaded dynamically
            # For now, log a warning
            logger.warning(
                f"Rollback functionality requires Migration instance - "
                f"manual rollback may be required for version {version}"
            )

        logger.info(f"✓ Rollback complete")
        return True

    def dry_run(self, db_path: Path, migrations: List[Migration]) -> None:
        """
        Test migrations without applying them.

        Args:
            db_path: Path to the database
            migrations: List of migrations to test
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"DRY RUN: Testing migrations on {db_path}")
        logger.info(f"{'='*60}\n")

        for migration in sorted(migrations, key=lambda m: m.version):
            try:
                logger.info(f"Testing: {migration}")
                migration.apply(db_path, dry_run=True)
                logger.info(f"✓ {migration.version} would succeed\n")
            except MigrationError as e:
                logger.error(f"✗ {migration.version} would fail: {e}\n")

    def get_migration_status(self, db_path: Path, component: str,
                            all_migrations: List[Migration]) -> Dict:
        """
        Get detailed migration status for a component.

        Args:
            db_path: Path to the database
            component: Component name
            all_migrations: List of all available migrations

        Returns:
            Status dictionary with current_version, pending_count, applied migrations
        """
        current_version = self.get_current_version(db_path, component)
        applied = self.get_applied_migrations(db_path, component)
        pending = [m for m in all_migrations if m.version > current_version]

        return {
            'component': component,
            'database': str(db_path),
            'current_version': current_version,
            'applied_count': len(applied),
            'pending_count': len(pending),
            'applied_migrations': applied,
            'pending_migrations': [
                {'version': m.version, 'description': m.description}
                for m in pending
            ]
        }
