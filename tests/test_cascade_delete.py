"""
Tests for CASCADE DELETE Foreign Key

Verifies that when a document is deleted, all associated knowledge entries
are automatically deleted via the CASCADE DELETE constraint.
"""

import sqlite3
import json
import time
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCascadeDelete:
    """Test suite for CASCADE DELETE foreign key constraint."""

    def __init__(self, db_path="felix_knowledge.db"):
        """
        Initialize test suite.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.test_doc_ids = []
        self.test_entry_ids = []

    def setup(self):
        """Setup test environment."""
        print("=" * 70)
        print("CASCADE DELETE TEST SUITE")
        print("=" * 70)
        print(f"Database: {self.db_path}")
        print()

    def teardown(self):
        """Cleanup test data."""
        if not self.test_doc_ids and not self.test_entry_ids:
            return

        print("\nCleaning up test data...")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None  # Autocommit
            cursor = conn.cursor()

            for doc_id in self.test_doc_ids:
                cursor.execute("DELETE FROM document_sources WHERE doc_id = ?", (doc_id,))

            for entry_id in self.test_entry_ids:
                cursor.execute("DELETE FROM knowledge_entries WHERE knowledge_id = ?", (entry_id,))

            conn.close()
            print(f"✓ Cleaned up {len(self.test_doc_ids)} documents and {len(self.test_entry_ids)} entries")

        except Exception as e:
            print(f"✗ Cleanup failed: {e}")

    def test_foreign_key_exists(self):
        """Test that CASCADE DELETE foreign key exists."""
        print("Test 1: CASCADE DELETE Foreign Key Exists")
        print("-" * 70)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check foreign key
            cursor.execute("PRAGMA foreign_key_list(knowledge_entries)")
            fk_list = cursor.fetchall()

            conn.close()

            # Look for CASCADE DELETE on source_doc_id
            cascade_found = False
            for fk in fk_list:
                # fk format: (id, seq, table, from, to, on_update, on_delete, match)
                if fk[2] == 'document_sources' and fk[3] == 'source_doc_id' and fk[6] == 'CASCADE':
                    cascade_found = True
                    break

            if not cascade_found:
                print("✗ FAILED: CASCADE DELETE constraint not found")
                print(f"  Foreign keys: {fk_list}")
                return False

            print("✓ PASSED: CASCADE DELETE foreign key exists")
            print(f"  Table: document_sources")
            print(f"  Column: source_doc_id")
            print(f"  Action: ON DELETE CASCADE")
            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_cascade_delete_basic(self):
        """Test basic CASCADE DELETE: delete document → entries auto-deleted."""
        print("\nTest 2: Basic CASCADE DELETE")
        print("-" * 70)

        test_doc_id = f"test_doc_{int(time.time())}"
        test_entry_id = f"test_entry_{int(time.time())}"

        self.test_doc_ids.append(test_doc_id)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None
            cursor = conn.cursor()

            # Create test document
            cursor.execute("""
                INSERT INTO document_sources
                (doc_id, file_path, file_name, file_type, file_size, file_hash, ingestion_status, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_doc_id,
                "/test/cascade_delete_test.txt",
                "cascade_delete_test.txt",
                "text/plain",
                100,
                "test_hash_123",
                "completed",
                time.time()
            ))

            # Create knowledge entry linked to document
            now = time.time()
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at, source_doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_entry_id,
                'concept',
                json.dumps({'concept': 'Cascade Test', 'definition': 'Testing CASCADE DELETE'}),
                'testing',
                json.dumps(['cascade', 'test']),
                'HIGH',
                'test_agent',
                now,
                now,
                test_doc_id  # Link to document
            ))

            # Verify entry exists
            cursor.execute("SELECT knowledge_id FROM knowledge_entries WHERE knowledge_id = ?", (test_entry_id,))
            if not cursor.fetchone():
                print("✗ FAILED: Entry not created")
                conn.close()
                return False

            # Delete the document
            cursor.execute("DELETE FROM document_sources WHERE doc_id = ?", (test_doc_id,))

            # Verify entry was CASCADE deleted
            cursor.execute("SELECT knowledge_id FROM knowledge_entries WHERE knowledge_id = ?", (test_entry_id,))
            entry_after = cursor.fetchone()

            conn.close()

            if entry_after is not None:
                print("✗ FAILED: Entry still exists after document deletion")
                print(f"  Entry ID: {entry_after[0]}")
                self.test_entry_ids.append(test_entry_id)  # Need manual cleanup
                return False

            print("✓ PASSED: Entry automatically deleted with document")
            print(f"  Document ID: {test_doc_id}")
            print(f"  Entry ID: {test_entry_id}")
            print(f"  Verified CASCADE DELETE worked")

            # Remove from cleanup list since already deleted
            self.test_doc_ids.remove(test_doc_id)

            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_cascade_delete_multiple_entries(self):
        """Test CASCADE DELETE with multiple entries per document."""
        print("\nTest 3: CASCADE DELETE Multiple Entries")
        print("-" * 70)

        test_doc_id = f"test_doc_multi_{int(time.time())}"
        test_entry_ids = [
            f"test_entry_multi_1_{int(time.time())}",
            f"test_entry_multi_2_{int(time.time())}",
            f"test_entry_multi_3_{int(time.time())}"
        ]

        self.test_doc_ids.append(test_doc_id)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None
            cursor = conn.cursor()

            # Create test document
            cursor.execute("""
                INSERT INTO document_sources
                (doc_id, file_path, file_name, file_type, file_size, file_hash, ingestion_status, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_doc_id,
                "/test/multi_entries.txt",
                "multi_entries.txt",
                "text/plain",
                200,
                "test_hash_multi",
                "completed",
                time.time()
            ))

            # Create multiple knowledge entries linked to document
            now = time.time()
            for idx, entry_id in enumerate(test_entry_ids):
                cursor.execute("""
                    INSERT INTO knowledge_entries
                    (knowledge_id, knowledge_type, content_json, domain, tags_json,
                     confidence_level, source_agent, created_at, updated_at, source_doc_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    'concept',
                    json.dumps({
                        'concept': f'Concept {idx + 1}',
                        'definition': f'Definition for concept {idx + 1}'
                    }),
                    'testing',
                    json.dumps(['multi', 'test']),
                    'HIGH',
                    'test_agent',
                    now,
                    now,
                    test_doc_id
                ))

            # Verify all entries exist
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE source_doc_id = ?
            """, (test_doc_id,))
            count_before = cursor.fetchone()[0]

            if count_before != len(test_entry_ids):
                print(f"✗ FAILED: Expected {len(test_entry_ids)} entries, found {count_before}")
                conn.close()
                return False

            # Delete the document
            cursor.execute("DELETE FROM document_sources WHERE doc_id = ?", (test_doc_id,))

            # Verify ALL entries were CASCADE deleted
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries
                WHERE source_doc_id = ?
            """, (test_doc_id,))
            count_after = cursor.fetchone()[0]

            conn.close()

            if count_after != 0:
                print(f"✗ FAILED: {count_after} entries still exist after document deletion")
                self.test_entry_ids.extend(test_entry_ids)  # Need manual cleanup
                return False

            print("✓ PASSED: All entries automatically deleted with document")
            print(f"  Document ID: {test_doc_id}")
            print(f"  Entries before: {count_before}")
            print(f"  Entries after: {count_after}")

            # Remove from cleanup list since already deleted
            self.test_doc_ids.remove(test_doc_id)

            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_cascade_delete_with_tags(self):
        """Test that CASCADE DELETE also cleans up tags table."""
        print("\nTest 4: CASCADE DELETE Cleans Up Tags")
        print("-" * 70)

        test_doc_id = f"test_doc_tags_{int(time.time())}"
        test_entry_id = f"test_entry_tags_{int(time.time())}"

        self.test_doc_ids.append(test_doc_id)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None
            cursor = conn.cursor()

            # Create test document
            cursor.execute("""
                INSERT INTO document_sources
                (doc_id, file_path, file_name, file_type, file_size, file_hash, ingestion_status, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_doc_id,
                "/test/tags_test.txt",
                "tags_test.txt",
                "text/plain",
                150,
                "test_hash_tags",
                "completed",
                time.time()
            ))

            # Create knowledge entry with tags
            now = time.time()
            tags = ['cascade', 'delete', 'tags', 'test']
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at, source_doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_entry_id,
                'concept',
                json.dumps({'concept': 'Tags Test', 'definition': 'Testing tag cleanup'}),
                'testing',
                json.dumps(tags),
                'HIGH',
                'test_agent',
                now,
                now,
                test_doc_id
            ))

            # Add tags to knowledge_tags table
            for tag in tags:
                cursor.execute("""
                    INSERT OR IGNORE INTO knowledge_tags (knowledge_id, tag)
                    VALUES (?, ?)
                """, (test_entry_id, tag))

            # Verify tags exist
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_tags
                WHERE knowledge_id = ?
            """, (test_entry_id,))
            tags_before = cursor.fetchone()[0]

            if tags_before == 0:
                print("⚠  WARNING: Tags not populated (knowledge_tags might not be used)")
                # Continue test anyway

            # Delete the document
            cursor.execute("DELETE FROM document_sources WHERE doc_id = ?", (test_doc_id,))

            # Verify entry was deleted
            cursor.execute("SELECT knowledge_id FROM knowledge_entries WHERE knowledge_id = ?", (test_entry_id,))
            entry_after = cursor.fetchone()

            # Verify tags were also deleted
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_tags
                WHERE knowledge_id = ?
            """, (test_entry_id,))
            tags_after = cursor.fetchone()[0]

            conn.close()

            if entry_after is not None:
                print("✗ FAILED: Entry still exists after document deletion")
                self.test_entry_ids.append(test_entry_id)
                return False

            if tags_after != 0:
                print(f"✗ FAILED: {tags_after} tags still exist after entry deletion")
                return False

            print("✓ PASSED: Entry and tags automatically deleted")
            print(f"  Document ID: {test_doc_id}")
            print(f"  Entry ID: {test_entry_id}")
            print(f"  Tags before: {tags_before}")
            print(f"  Tags after: {tags_after}")

            # Remove from cleanup list since already deleted
            self.test_doc_ids.remove(test_doc_id)

            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def test_no_cascade_without_fk(self):
        """Test that entries WITHOUT source_doc_id are not affected."""
        print("\nTest 5: No CASCADE for Entries Without source_doc_id")
        print("-" * 70)

        test_entry_id = f"test_entry_no_fk_{int(time.time())}"

        self.test_entry_ids.append(test_entry_id)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.isolation_level = None
            cursor = conn.cursor()

            # Create knowledge entry WITHOUT source_doc_id
            now = time.time()
            cursor.execute("""
                INSERT INTO knowledge_entries
                (knowledge_id, knowledge_type, content_json, domain, tags_json,
                 confidence_level, source_agent, created_at, updated_at, source_doc_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_entry_id,
                'concept',
                json.dumps({'concept': 'No FK Test', 'definition': 'Entry without document link'}),
                'testing',
                json.dumps(['no_fk', 'test']),
                'HIGH',
                'test_agent',
                now,
                now,
                None  # No source_doc_id
            ))

            # Verify entry exists
            cursor.execute("SELECT knowledge_id FROM knowledge_entries WHERE knowledge_id = ?", (test_entry_id,))
            entry_before = cursor.fetchone()

            if not entry_before:
                print("✗ FAILED: Entry not created")
                conn.close()
                return False

            # Try to delete a non-existent document (should not affect our entry)
            cursor.execute("DELETE FROM document_sources WHERE doc_id = ?", ("non_existent_doc",))

            # Verify entry still exists
            cursor.execute("SELECT knowledge_id FROM knowledge_entries WHERE knowledge_id = ?", (test_entry_id,))
            entry_after = cursor.fetchone()

            conn.close()

            if not entry_after:
                print("✗ FAILED: Entry was deleted unexpectedly")
                self.test_entry_ids.remove(test_entry_id)  # Already deleted
                return False

            print("✓ PASSED: Entry without source_doc_id unaffected")
            print(f"  Entry ID: {test_entry_id}")
            print(f"  source_doc_id: NULL")
            print(f"  Verified entry persists independent of documents")

            return True

        except Exception as e:
            print(f"✗ FAILED: {e}")
            return False

    def run_all_tests(self):
        """Run all tests and report results."""
        self.setup()

        tests = [
            self.test_foreign_key_exists,
            self.test_cascade_delete_basic,
            self.test_cascade_delete_multiple_entries,
            self.test_cascade_delete_with_tags,
            self.test_no_cascade_without_fk,
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

    tester = TestCascadeDelete(db_path)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)
