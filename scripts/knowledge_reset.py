#!/usr/bin/env python3
"""
Knowledge Brain Reset Script

Resets ONLY the knowledge brain data (knowledge entries, relationships, document sources)
while preserving all other Felix databases (workflow history, task memory, etc.).

This is designed for clean re-processing after bug fixes.

Usage:
    python scripts/knowledge_reset.py              # Interactive mode with confirmation
    python scripts/knowledge_reset.py --force      # Skip confirmation
    python scripts/knowledge_reset.py --dry-run    # Show what would be deleted without doing it
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def get_knowledge_stats(db_path: str) -> dict:
    """Get current statistics from knowledge database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    stats = {}

    # Count tables that exist
    tables_to_check = [
        ("knowledge_entries", "SELECT COUNT(*) FROM knowledge_entries"),
        ("knowledge_relationships", "SELECT COUNT(*) FROM knowledge_relationships"),
        ("document_sources", "SELECT COUNT(*) FROM document_sources"),
        ("knowledge_usage", "SELECT COUNT(*) FROM knowledge_usage"),
        ("knowledge_fts", "SELECT COUNT(*) FROM knowledge_fts"),
        ("knowledge_gaps", "SELECT COUNT(*) FROM knowledge_gaps"),
    ]

    for table_name, query in tables_to_check:
        try:
            cursor.execute(query)
            stats[table_name] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            stats[table_name] = "table not found"

    conn.close()
    return stats


def reset_knowledge_brain(db_path: str, dry_run: bool = False) -> dict:
    """
    Reset knowledge brain data while preserving schema.

    Args:
        db_path: Path to felix_knowledge.db
        dry_run: If True, only show what would be deleted

    Returns:
        Dict with deletion results
    """
    if not Path(db_path).exists():
        return {
            "success": False,
            "error": f"Database not found: {db_path}"
        }

    # Get before stats
    before_stats = get_knowledge_stats(db_path)

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "would_delete": before_stats
        }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    deleted = {}
    errors = []

    # Delete in correct order (foreign key constraints)
    # knowledge_relationships references knowledge_entries
    # knowledge_usage references knowledge_entries
    # knowledge_entries references document_sources

    delete_order = [
        "knowledge_relationships",
        "knowledge_usage",
        "knowledge_gaps",
        "knowledge_fts",
        "knowledge_entries",
        "document_sources",
    ]

    for table in delete_order:
        try:
            cursor.execute(f"DELETE FROM {table}")
            deleted[table] = cursor.rowcount
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                deleted[table] = "table not found"
            else:
                errors.append(f"{table}: {str(e)}")

    conn.commit()

    # Vacuum to reclaim space
    try:
        conn.execute("VACUUM")
    except Exception as e:
        errors.append(f"VACUUM: {str(e)}")

    conn.close()

    # Get after stats to verify
    after_stats = get_knowledge_stats(db_path)

    return {
        "success": len(errors) == 0,
        "before": before_stats,
        "deleted": deleted,
        "after": after_stats,
        "errors": errors if errors else None
    }


def format_number(n) -> str:
    """Format number with thousands separator or return as-is if not a number."""
    if isinstance(n, int):
        return f"{n:,}"
    return str(n)


def main():
    parser = argparse.ArgumentParser(
        description="Reset Felix Knowledge Brain data for clean re-processing"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--db-path",
        default="felix_knowledge.db",
        help="Path to knowledge database (default: felix_knowledge.db)"
    )

    args = parser.parse_args()

    # Resolve database path
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        # Try relative to current directory
        if not db_path.exists():
            # Try relative to script location
            script_dir = Path(__file__).parent.parent
            db_path = script_dir / args.db_path

    db_path = str(db_path)

    print("\n" + "=" * 60)
    print("FELIX KNOWLEDGE BRAIN RESET")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No data will be deleted]\n")

    # Show current stats
    if Path(db_path).exists():
        stats = get_knowledge_stats(db_path)
        print(f"\nDatabase: {db_path}")
        print(f"Database size: {Path(db_path).stat().st_size / (1024*1024):.2f} MB")
        print("\nCurrent data:")
        print("-" * 40)
        for table, count in stats.items():
            print(f"  {table}: {format_number(count)}")
    else:
        print(f"\nDatabase not found: {db_path}")
        sys.exit(1)

    # Confirmation
    if not args.dry_run and not args.force:
        print("\n" + "!" * 60)
        print("WARNING: This will permanently delete ALL knowledge brain data!")
        print("This includes:")
        print("  - All knowledge entries (concepts, definitions)")
        print("  - All relationships between concepts")
        print("  - All document source records")
        print("  - All knowledge usage tracking")
        print("  - All FTS5 search index entries")
        print("\nThe schema will be preserved for re-processing.")
        print("!" * 60)

        response = input("\nType 'RESET' to confirm deletion: ")
        if response != "RESET":
            print("\nAborted. No data was deleted.")
            sys.exit(0)

    # Perform reset
    print("\nResetting knowledge brain...")
    result = reset_knowledge_brain(db_path, dry_run=args.dry_run)

    if result["success"]:
        if args.dry_run:
            print("\n[DRY RUN] Would delete:")
            for table, count in result["would_delete"].items():
                print(f"  {table}: {format_number(count)} rows")
        else:
            print("\n" + "=" * 60)
            print("RESET COMPLETE")
            print("=" * 60)
            print("\nDeleted:")
            for table, count in result["deleted"].items():
                print(f"  {table}: {format_number(count)} rows")

            print("\nVerification (should all be 0):")
            for table, count in result["after"].items():
                status = "OK" if count == 0 else "WARNING"
                print(f"  {table}: {format_number(count)} [{status}]")

            print("\n" + "-" * 60)
            print("Knowledge brain reset complete.")
            print("Ready for re-processing with fixed code.")
            print("-" * 60)
    else:
        print(f"\nReset failed: {result.get('error', 'Unknown error')}")
        if result.get("errors"):
            for error in result["errors"]:
                print(f"  - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
