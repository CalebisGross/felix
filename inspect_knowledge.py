#!/usr/bin/env python3
"""
Utility to inspect and optionally clean the Felix knowledge database.
"""
import sqlite3
import time
from datetime import datetime

DB_PATH = "felix_knowledge.db"

def inspect_knowledge():
    """Inspect all entries in the knowledge store."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all entries
    cursor.execute("""
        SELECT knowledge_id, domain, source_agent, confidence_level, created_at, content_json
        FROM knowledge_entries
        ORDER BY created_at DESC
    """)

    entries = cursor.fetchall()

    print("=" * 80)
    print(f"KNOWLEDGE STORE INSPECTION ({DB_PATH})")
    print("=" * 80)
    print(f"Total entries: {len(entries)}\n")

    current_time = time.time()
    one_hour_ago = current_time - 3600

    for idx, entry in enumerate(entries, 1):
        id_, domain, source, confidence, timestamp, content = entry

        # Convert timestamp
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            age_minutes = (current_time - timestamp) / 60
            age_str = f"{age_minutes:.1f} minutes ago"
        else:
            dt = "No timestamp"
            age_str = "Unknown age"

        # Truncate content
        content_preview = str(content)[:100].replace('\n', ' ')

        # Check if within last hour
        recent = "✓ RECENT" if timestamp and timestamp >= one_hour_ago else "✗ OLD"

        print(f"[{idx}] ID: {id_} | {recent}")
        print(f"    Domain: {domain}")
        print(f"    Source: {source}")
        print(f"    Confidence: {confidence}")
        print(f"    Timestamp: {dt} ({age_str})")
        print(f"    Content: {content_preview}...")
        print()

    conn.close()

def clear_old_entries():
    """Clear entries older than 1 hour."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    current_time = time.time()
    one_hour_ago = current_time - 3600

    # Count old entries
    cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE created_at < ?", (one_hour_ago,))
    old_count = cursor.fetchone()[0]

    if old_count == 0:
        print("No old entries to clear.")
        conn.close()
        return

    print(f"Found {old_count} entries older than 1 hour.")
    response = input("Delete these entries? (yes/no): ").strip().lower()

    if response == 'yes':
        cursor.execute("DELETE FROM knowledge_entries WHERE created_at < ?", (one_hour_ago,))
        conn.commit()
        print(f"✓ Deleted {old_count} old entries.")
    else:
        print("Cancelled.")

    conn.close()

def clear_domain(domain_name):
    """Clear all entries from a specific domain."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count entries in domain
    cursor.execute("SELECT COUNT(*) FROM knowledge_entries WHERE domain = ?", (domain_name,))
    domain_count = cursor.fetchone()[0]

    if domain_count == 0:
        print(f"No entries found in domain '{domain_name}'.")
        conn.close()
        return

    print(f"Found {domain_count} entries in domain '{domain_name}'.")
    response = input(f"Delete all entries from domain '{domain_name}'? (yes/no): ").strip().lower()

    if response == 'yes':
        cursor.execute("DELETE FROM knowledge_entries WHERE domain = ?", (domain_name,))
        conn.commit()
        print(f"✓ Deleted {domain_count} entries from domain '{domain_name}'.")
    else:
        print("Cancelled.")

    conn.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "clear-old":
            clear_old_entries()
        elif command == "clear-domain" and len(sys.argv) > 2:
            clear_domain(sys.argv[2])
        else:
            print("Usage:")
            print("  python inspect_knowledge.py              # Inspect all entries")
            print("  python inspect_knowledge.py clear-old    # Clear entries older than 1 hour")
            print("  python inspect_knowledge.py clear-domain workflow_task  # Clear specific domain")
    else:
        inspect_knowledge()
