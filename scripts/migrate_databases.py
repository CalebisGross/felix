#!/usr/bin/env python3
"""
Run all database migrations for Felix framework.

This script orchestrates the complete database migration process:
- Creates backups before migration
- Applies migrations for all components
- Verifies migration success
- Provides rollback capability

Usage:
    # Test migrations without applying
    python scripts/migrate_databases.py --dry-run

    # Apply all migrations
    python scripts/migrate_databases.py

    # Apply migrations for specific component only
    python scripts/migrate_databases.py --component knowledge

    # Rollback last migration
    python scripts/migrate_databases.py --rollback

    # Get migration status
    python scripts/migrate_databases.py --status

Components:
    - knowledge: felix_knowledge.db (KnowledgeStore)
    - tasks: felix_task_memory.db (TaskMemory)
    - workflows: felix_workflow_history.db (WorkflowHistory)
    - agent_performance: felix_agent_performance.db (NEW)
    - system_actions: felix_system_actions.db (NEW)
"""

import sys
import argparse
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.migration.version_manager import MigrationManager
from src.migration.backup_manager import BackupManager
from src.migration import (
    knowledge_migrations,
    task_migrations,
    workflow_migrations,
    create_agent_performance,
    create_system_actions
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main migration orchestration."""
    parser = argparse.ArgumentParser(
        description="Migrate Felix databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test migrations without applying"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback last migration"
    )
    parser.add_argument(
        "--component",
        choices=["knowledge", "tasks", "workflows", "agent_performance", "system_actions"],
        help="Migrate specific component only"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status for all components"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup creation (not recommended)"
    )

    args = parser.parse_args()

    # Banner
    print("\n" + "="*60)
    print("FELIX DATABASE MIGRATION")
    print("="*60)

    # Initialize managers
    backup_mgr = BackupManager(backup_dir="backups")
    migration_mgr = MigrationManager()

    # Define components
    components = {
        "knowledge": {
            "db_path": "felix_knowledge.db",
            "migrations": knowledge_migrations.get_migrations(),
            "description": "KnowledgeStore (2 migrations)"
        },
        "tasks": {
            "db_path": "felix_task_memory.db",
            "migrations": task_migrations.get_migrations(),
            "description": "TaskMemory (4 migrations)"
        },
        "workflows": {
            "db_path": "felix_workflow_history.db",
            "migrations": workflow_migrations.get_migrations(),
            "description": "WorkflowHistory (2 migrations)"
        },
        "agent_performance": {
            "db_path": "felix_agent_performance.db",
            "migrations": create_agent_performance.get_migrations(),
            "description": "AgentPerformanceStore (NEW - 1 migration)"
        },
        "system_actions": {
            "db_path": "felix_system_actions.db",
            "migrations": create_system_actions.get_migrations(),
            "description": "SystemActionsStore (NEW - 1 migration)"
        }
    }

    # Filter to specific component if requested
    if args.component:
        components = {args.component: components[args.component]}

    # Show status and exit
    if args.status:
        print("\nMigration Status:")
        print("-"*60)
        for name, config in components.items():
            db_path = Path(config["db_path"])
            if not db_path.exists():
                print(f"\n{name} ({config['db_path']}): NOT CREATED YET")
                continue

            status = migration_mgr.get_migration_status(
                db_path, name, config["migrations"]
            )

            print(f"\n{name} ({config['db_path']}):")
            print(f"  Current version: {status['current_version']}")
            print(f"  Applied: {status['applied_count']}")
            print(f"  Pending: {status['pending_count']}")

            if status['pending_migrations']:
                print(f"  Pending migrations:")
                for m in status['pending_migrations']:
                    print(f"    - v{m['version']}: {m['description']}")

        print("\n" + "="*60)
        return

    # Create backups before migrating (unless disabled or dry run)
    if not args.no_backup and not args.dry_run and not args.rollback:
        print("\n" + "="*60)
        print("CREATING BACKUPS")
        print("="*60)

        # Only backup existing databases
        existing_dbs = [
            Path(config["db_path"])
            for config in components.values()
            if Path(config["db_path"]).exists()
        ]

        if existing_dbs:
            backups = backup_mgr.backup_all(existing_dbs)
            print(f"\n✓ Created {len(backups)} backups in {backup_mgr.backup_dir}")
        else:
            print("\n⚠ No existing databases to backup")

    # Rollback mode
    if args.rollback:
        print("\n" + "="*60)
        print("ROLLING BACK MIGRATIONS")
        print("="*60)

        for name in components:
            db_path = Path(components[name]["db_path"])
            if not db_path.exists():
                print(f"\n{name}: Database does not exist, skipping")
                continue

            print(f"\n{name}:")
            try:
                migration_mgr.rollback(db_path, name)
            except Exception as e:
                print(f"  ✗ Rollback failed: {e}")

        print("\n" + "="*60)
        return

    # Migration mode (normal or dry run)
    print("\n" + "="*60)
    print("APPLYING MIGRATIONS" if not args.dry_run else "DRY RUN MODE")
    print("="*60)

    start_time = time.time()
    total_applied = 0
    total_failed = 0
    results = {}

    for component_name, config in components.items():
        db_path = Path(config["db_path"])
        migrations = config["migrations"]

        print(f"\n{'='*60}")
        print(f"Component: {component_name}")
        print(f"Database: {config['db_path']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")

        # Check if database exists
        if not db_path.exists():
            print(f"  Database does not exist yet")
            if not args.dry_run:
                print(f"  Creating new database...")

        try:
            if args.dry_run:
                # Dry run mode
                migration_mgr.dry_run(db_path, migrations)
                results[component_name] = {"status": "dry_run", "applied": "N/A", "failed": "N/A"}
            else:
                # Apply migrations
                applied, failed = migration_mgr.apply_migrations(
                    db_path, component_name, migrations, dry_run=False
                )

                total_applied += applied
                total_failed += failed
                results[component_name] = {"status": "completed", "applied": applied, "failed": failed}

                if failed > 0:
                    print(f"\n⚠ WARNING: {failed} migration(s) failed for {component_name}")

        except Exception as e:
            logger.error(f"Migration failed for {component_name}: {e}")
            results[component_name] = {"status": "error", "applied": 0, "failed": 1}
            total_failed += 1

    # Summary
    duration = time.time() - start_time
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)

    for component_name, result in results.items():
        status_icon = "✓" if result["status"] == "completed" and result["failed"] == 0 else "⚠"
        print(f"{status_icon} {component_name}: {result['status']}")
        if result["status"] == "completed":
            print(f"  Applied: {result['applied']}, Failed: {result['failed']}")

    if not args.dry_run:
        print(f"\nTotal migrations applied: {total_applied}")
        print(f"Total migrations failed: {total_failed}")

    print(f"Duration: {duration:.2f}s")

    if total_failed > 0:
        print("\n⚠ Some migrations failed. Check logs above for details.")
        print("  You can restore from backups in the 'backups/' directory if needed.")

    print("\n" + "="*60)

    # Exit with error code if any migrations failed
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Migration script failed: {e}", exc_info=True)
        sys.exit(1)
