"""
Tests for FTS5 Auto-Sync Triggers

Verifies that the knowledge_fts table automatically stays in sync with
knowledge_entries through database triggers.
"""

import sqlite3
import json
import time
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFTS5Triggers:
    """Test suite for FTS5 auto-sync triggers."""

    def __init__(self, db_path="felix_knowledge.db"):
        """
        Initialize test suite.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.test_ids = []  # Track test entries for cleanup

    def setup(self):
        """Setup test environment."""
        print("=" * 70)
        print("FTS5 TRIGGERS TEST SUITE")
        print("=" * 70)
        print(f"Database: {self.db_path}")
        print()

    def teardown(self):
        """Cleanup test data."""
        if not self.test_ids:
            return

        print("\nCleaning up test entries...")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.isolation_level = None  # Autocommit mode
            cursor = conn.cursor()

            for test_id in self.test_ids:
                cursor.execute("DELETE FROM knowledge_entries WHERE knowledge_id = ?", (test_id,))

            conn.close()
            print(f"✓ Cleaned up {len(self.test_ids)} test entries")

        except Exception as e:
            print(f"✗ Cleanup failed: {e}")

    def _get_fts_entry(self, knowledge_id):
        """
        Get FTS5 entry by knowledge_id.

        Args:
            knowledge_id: Entry ID

        Returns:
            FTS5 row as dict or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT knowledge_id, content, domain, tags
            FROM knowledge_fts
            WHERE knowledge_id = ?
        """, (knowledge_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'knowledge_id': row[0],
                'content': row[1],
                'domain': row[2],
                'tags': row[3]
            }
        return None

    def test_insert_trigger(self):
        """Test that INSERT trigger populates FTS5."""
        print("Test 1: INSERT Trigger")
        print("-" * 70)

        test_id = f"test_insert_{int(time.time())}"
        self.test_ids.append(test_id)

        content = {
            'concept': 'FTS5 Trigger Test',
            'definition': 'Testing automatic FTS5 population on INSERT',
            'summary': 'This entry tests the INSERT trigger'
        }

        try:
            conn = sqlite3.connect(self.db_path)
            conn.isolation_level = None  # Autocommit mode to avoid locking
            cursor = conn.cursor()

            # Insert into knowledge_entries
            now = time.time()
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                'concept',
                json.dumps(content),
                'testing',
                json.dumps(['trigger', 'fts5', 'test']),
                'HIGH',
                'test_agent',
                now,
                now
            ))

            conn.close()

            # Check if FTS5 was populated
            fts_entry = self._get_fts_entry(test_id)

            if not fts_entry:
                print("✗ FAILED: FTS5 entry not created")
                return False

            # Verify content was extracted correctly
            expected_content = "FTS5 Trigger Test Testing automatic FTS5 population on INSERT This entry tests the INSERT trigger"

            if fts_entry['content'] != expected_content:
                print(f"✗ FAILED: Content mismatch")
                print(f"  Expected: {expected_content}")
                print(f"  Got: {fts_entry['content']}")
                return False

            # Verify domain
            if fts_entry['domain'] != 'testing':
                print(f"✗ FAILED: Domain mismatch (expected 'testing', got '{fts_entry['domain']}')")
                return False

            # Verify tags
            expected_tags = json.dumps(['trigger', 'fts5', 'test'])
            if fts_entry['tags'] != expected_tags:
                print(f"✗ FAILED: Tags mismatch")
                return False

            print("✓ PASSED: INSERT trigger correctly populated FTS5")
            print(f"  Entry ID: {test_id}")
            print(f"  Content: {fts_entry['content'][:60]}...")
            print(f"  Domain: {fts_entry['domain']}")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_update_trigger(self):
        """Test that UPDATE trigger syncs FTS5."""
        print("\nTest 2: UPDATE Trigger")
        print("-" * 70)

        test_id = f"test_update_{int(time.time())}"
        self.test_ids.append(test_id)

        original_content = {
            'concept': 'Original Concept',
            'definition': 'Original definition',
            'summary': 'Original summary'
        }

        updated_content = {
            'concept': 'Updated Concept',
            'definition': 'Updated definition with new information',
            'summary': 'Updated summary reflects changes'
        }

        try:
            conn = sqlite3.connect(self.db_path)
            conn.isolation_level = None  # Autocommit mode to avoid locking
            cursor = conn.cursor()

            # Insert original entry
            now = time.time()
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                'concept',
                json.dumps(original_content),
                'testing',
                json.dumps(['update', 'test']),
                'HIGH',
                'test_agent',
                now,
                now
            ))

            # Verify FTS5 has original content
            fts_before = self._get_fts_entry(test_id)
            if not fts_before or 'Original Concept' not in fts_before['content']:
                print("✗ FAILED: Original FTS5 entry not found")
                conn.close()
                return False

            # Update the entry
            cursor.execute("""
                UPDATE knowledge_entries
                SET content_json = ?, domain = ?, tags_json = ?
                WHERE knowledge_id = ?
            """, (
                json.dumps(updated_content),
                'updated_domain',
                json.dumps(['updated', 'trigger', 'test']),
                test_id
            ))

            conn.close()

            # Verify FTS5 was updated
            fts_after = self._get_fts_entry(test_id)

            if not fts_after:
                print("✗ FAILED: FTS5 entry disappeared after update")
                return False

            # Check content was updated
            if 'Updated Concept' not in fts_after['content']:
                print(f"✗ FAILED: FTS5 content not updated")
                print(f"  Expected: Updated Concept in content")
                print(f"  Got: {fts_after['content']}")
                return False

            # Check old content is gone
            if 'Original Concept' in fts_after['content']:
                print("✗ FAILED: FTS5 still contains old content")
                return False

            # Check domain was updated
            if fts_after['domain'] != 'updated_domain':
                print(f"✗ FAILED: Domain not updated (got '{fts_after['domain']}')")
                return False

            print("✓ PASSED: UPDATE trigger correctly synced FTS5")
            print(f"  Entry ID: {test_id}")
            print(f"  Old content: Original Concept...")
            print(f"  New content: {fts_after['content'][:60]}...")
            print(f"  Domain updated: testing → {fts_after['domain']}")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_delete_trigger(self):
        """Test that DELETE trigger removes from FTS5."""
        print("\nTest 3: DELETE Trigger")
        print("-" * 70)

        test_id = f"test_delete_{int(time.time())}"

        content = {
            'concept': 'Temporary Entry',
            'definition': 'This entry will be deleted',
            'summary': 'Testing DELETE trigger'
        }

        try:
            conn = sqlite3.connect(self.db_path)
            conn.isolation_level = None  # Autocommit mode to avoid locking
            cursor = conn.cursor()

            # Insert entry
            now = time.time()
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id,
                'concept',
                json.dumps(content),
                'testing',
                json.dumps(['delete', 'test']),
                'HIGH',
                'test_agent',
                now,
                now
            ))

            # Verify FTS5 entry exists
            fts_before = self._get_fts_entry(test_id)
            if not fts_before:
                print("✗ FAILED: FTS5 entry not created before delete test")
                conn.close()
                return False

            # Delete the entry
            cursor.execute("DELETE FROM knowledge_entries WHERE knowledge_id = ?", (test_id,))
            conn.close()

            # Verify FTS5 entry was removed
            fts_after = self._get_fts_entry(test_id)

            if fts_after is not None:
                print("✗ FAILED: FTS5 entry still exists after DELETE")
                print(f"  FTS5 entry: {fts_after}")
                return False

            print("✓ PASSED: DELETE trigger correctly removed FTS5 entry")
            print(f"  Entry ID: {test_id}")
            print(f"  Verified FTS5 entry deleted")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_sync_consistency(self):
        """Test that FTS5 and knowledge_entries remain in sync."""
        print("\nTest 4: Sync Consistency Check")
        print("-" * 70)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get counts
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            entries_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM knowledge_fts")
            fts_count = cursor.fetchone()[0]

            conn.close()

            if entries_count != fts_count:
                print(f"✗ FAILED: Counts don't match")
                print(f"  knowledge_entries: {entries_count}")
                print(f"  knowledge_fts: {fts_count}")
                return False

            print("✓ PASSED: FTS5 and knowledge_entries counts match")
            print(f"  Both tables: {entries_count} entries")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_trigger_definitions(self):
        """Test that trigger definitions are correct."""
        print("\nTest 5: Trigger Definitions")
        print("-" * 70)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check all triggers exist
            cursor.execute("""
                SELECT name, sql FROM sqlite_master
                WHERE type='trigger' AND name IN (
                    'knowledge_entries_ai',
                    'knowledge_entries_au',
                    'knowledge_entries_ad'
                )
                ORDER BY name
            """)

            triggers = cursor.fetchall()
            conn.close()

            if len(triggers) != 3:
                print(f"✗ FAILED: Expected 3 triggers, found {len(triggers)}")
                return False

            # Verify trigger names and basic structure
            trigger_names = [t[0] for t in triggers]
            expected = ['knowledge_entries_ad', 'knowledge_entries_ai', 'knowledge_entries_au']

            if trigger_names != expected:
                print(f"✗ FAILED: Unexpected trigger names")
                print(f"  Expected: {expected}")
                print(f"  Got: {trigger_names}")
                return False

            # Verify INSERT trigger targets knowledge_fts
            insert_trigger_sql = triggers[1][1]  # ai is at index 1 after sorting
            if 'knowledge_fts' not in insert_trigger_sql:
                print("✗ FAILED: INSERT trigger doesn't reference knowledge_fts")
                return False

            # Verify UPDATE trigger uses UPDATE statement
            update_trigger_sql = triggers[2][1]  # au is at index 2
            if 'UPDATE knowledge_fts' not in update_trigger_sql:
                print("✗ FAILED: UPDATE trigger doesn't UPDATE knowledge_fts")
                return False

            # Verify DELETE trigger uses DELETE statement
            delete_trigger_sql = triggers[0][1]  # ad is at index 0
            if 'DELETE FROM knowledge_fts' not in delete_trigger_sql:
                print("✗ FAILED: DELETE trigger doesn't DELETE FROM knowledge_fts")
                return False

            print("✓ PASSED: All trigger definitions are correct")
            print("  ✓ knowledge_entries_ai (INSERT)")
            print("  ✓ knowledge_entries_au (UPDATE)")
            print("  ✓ knowledge_entries_ad (DELETE)")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and report results."""
        self.setup()

        tests = [
            self.test_trigger_definitions,
            self.test_sync_consistency,
            self.test_insert_trigger,
            self.test_update_trigger,
            self.test_delete_trigger,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"\n✗ Test failed with exception: {e}")
                results.append(False)

        self.teardown()

        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(results)
        total = len(results)

        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")

        if all(results):
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "felix_knowledge.db"

    if not os.path.exists(db_path):
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    tester = TestFTS5Triggers(db_path)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)
