"""
Database Migration System for Felix Framework.

This module provides infrastructure for versioned database migrations,
automated backups, and rollback support across all Felix databases.

Components:
- base_migration: Base classes for migrations
- version_manager: Schema version tracking and migration orchestration
- backup_manager: Automated database backups before migrations
- *_migrations: Migration definitions for each database component
"""

from .base_migration import Migration, MigrationError
from .version_manager import MigrationManager
from .backup_manager import BackupManager

__all__ = [
    'Migration',
    'MigrationError',
    'MigrationManager',
    'BackupManager'
]
