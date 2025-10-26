"""
Base migration classes and utilities.

Provides abstract base class for database migrations with
versioning, backup, and rollback support.
"""

import sqlite3
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration fails."""
    pass


class Migration(ABC):
    """
    Base class for database migrations.

    Each migration must define:
    - version: Integer version number (sequential)
    - description: Human-readable description of what the migration does
    - up(): Method to apply the migration
    - down(): Method to rollback the migration
    - verify(): Optional method to verify migration succeeded

    Example:
        class AddIndexMigration(Migration):
            version = 1
            description = "Add index on created_at column"

            def up(self, conn: sqlite3.Connection):
                conn.execute("CREATE INDEX idx_created ON table(created_at)")

            def down(self, conn: sqlite3.Connection):
                conn.execute("DROP INDEX idx_created")

            def verify(self, conn: sqlite3.Connection) -> bool:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_created'"
                )
                return cursor.fetchone() is not None
    """

    # Subclasses must define these
    version: int
    description: str

    def __init__(self):
        """Initialize migration."""
        if not hasattr(self, 'version') or not isinstance(self.version, int):
            raise ValueError(f"{self.__class__.__name__} must define version as integer")
        if not hasattr(self, 'description') or not isinstance(self.description, str):
            raise ValueError(f"{self.__class__.__name__} must define description as string")

    @abstractmethod
    def up(self, conn: sqlite3.Connection) -> None:
        """
        Apply the migration.

        Args:
            conn: SQLite connection to the database

        Raises:
            MigrationError: If migration fails
        """
        pass

    def down(self, conn: sqlite3.Connection) -> None:
        """
        Rollback the migration (optional, but recommended).

        Args:
            conn: SQLite connection to the database

        Raises:
            MigrationError: If rollback fails
        """
        raise NotImplementedError(
            f"Migration {self.version} ({self.description}) does not support rollback"
        )

    def verify(self, conn: sqlite3.Connection) -> bool:
        """
        Verify that the migration was applied successfully.

        Optional but recommended for critical migrations.

        Args:
            conn: SQLite connection to the database

        Returns:
            True if migration is verified, False otherwise
        """
        # Default: assume success if no verification implemented
        return True

    def apply(self, db_path: Path, dry_run: bool = False) -> bool:
        """
        Apply this migration to a database.

        Args:
            db_path: Path to the SQLite database file
            dry_run: If True, test migration without committing

        Returns:
            True if successful, False otherwise

        Raises:
            MigrationError: If migration fails
        """
        logger.info(f"Applying migration {self.version}: {self.description}")
        logger.info(f"Target database: {db_path}")

        if dry_run:
            logger.info("DRY RUN MODE - Changes will NOT be committed")

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Apply migration
            self.up(conn)

            # Verify if verification implemented
            if not dry_run:
                if self.verify(conn):
                    logger.info(f"✓ Migration {self.version} verified successfully")
                    conn.commit()
                else:
                    logger.error(f"✗ Migration {self.version} verification failed")
                    conn.rollback()
                    raise MigrationError(f"Migration {self.version} verification failed")
            else:
                logger.info(f"Dry run complete - rolling back changes")
                conn.rollback()

            conn.close()
            logger.info(f"✓ Migration {self.version} completed successfully")
            return True

        except sqlite3.Error as e:
            logger.error(f"✗ Migration {self.version} failed: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise MigrationError(f"Migration {self.version} failed: {e}") from e
        except Exception as e:
            logger.error(f"✗ Migration {self.version} failed with unexpected error: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise MigrationError(f"Migration {self.version} failed: {e}") from e

    def rollback(self, db_path: Path) -> bool:
        """
        Rollback this migration from a database.

        Args:
            db_path: Path to the SQLite database file

        Returns:
            True if successful, False otherwise

        Raises:
            MigrationError: If rollback fails
        """
        logger.info(f"Rolling back migration {self.version}: {self.description}")
        logger.info(f"Target database: {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Rollback migration
            self.down(conn)
            conn.commit()
            conn.close()

            logger.info(f"✓ Migration {self.version} rolled back successfully")
            return True

        except NotImplementedError as e:
            logger.warning(f"Migration {self.version} does not support rollback")
            raise MigrationError(str(e)) from e
        except sqlite3.Error as e:
            logger.error(f"✗ Rollback of migration {self.version} failed: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            raise MigrationError(f"Rollback failed: {e}") from e

    def __str__(self) -> str:
        """String representation of migration."""
        return f"Migration{self.version:03d}: {self.description}"

    def __repr__(self) -> str:
        """Developer representation of migration."""
        return f"<{self.__class__.__name__} version={self.version}>"
