"""Quick script to check live agents database."""
import sqlite3
import time
from pathlib import Path

db_path = "felix_live_agents.db"

if not Path(db_path).exists():
    print(f"[X] Database not found: {db_path}")
    print("   The database should be created when Felix system starts.")
else:
    print(f"[OK] Database exists: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_agents'")
    if not cursor.fetchone():
        print("[X] Table 'live_agents' does not exist!")
    else:
        print("[OK] Table 'live_agents' exists")

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM live_agents")
        total = cursor.fetchone()[0]
        print(f"\nTotal agents in database: {total}")

        if total > 0:
            # Get recent agents
            cursor.execute("""
                SELECT agent_id, agent_type, progress, confidence, last_update, status
                FROM live_agents
                ORDER BY last_update DESC
                LIMIT 10
            """)

            print("\nMost recent agents:")
            current_time = time.time()
            for row in cursor.fetchall():
                agent_id, agent_type, progress, confidence, last_update, status = row
                age = current_time - last_update
                print(f"  - {agent_id} ({agent_type})")
                print(f"    Progress: {progress:.2%}, Confidence: {confidence:.2%}")
                print(f"    Status: {status}, Updated: {age:.1f}s ago")

            # Check active agents
            cutoff = current_time - 5.0
            cursor.execute("""
                SELECT COUNT(*) FROM live_agents
                WHERE last_update > ? AND status = 'active'
            """, (cutoff,))
            active_count = cursor.fetchone()[0]
            print(f"\n[OK] Active agents (last 5s): {active_count}")
        else:
            print("\n[!] No agents in database yet.")
            print("   This means either:")
            print("   1. Workflow hasn't started yet")
            print("   2. Agent positions aren't being tracked")
            print("   3. LiveAgentTracker isn't initialized in tkinter GUI")

    conn.close()
