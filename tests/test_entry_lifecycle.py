"""
Test Entry Lifecycle Management

Tests for Phase 3: edit, delete, merge, re-process operations.
"""

import os
import tempfile
from pathlib import Path
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel
from src.knowledge.knowledge_daemon import KnowledgeDaemon, DaemonConfig

def test_update_knowledge_entry():
    """Test updating a knowledge entry."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KnowledgeStore(db_path)

        # Create entry
        entry_id = store.store_knowledge(
            knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
            content={'concept': 'Python', 'definition': 'A programming language'},
            confidence_level=ConfidenceLevel.MEDIUM,
            source_agent='test',
            domain='python',
            tags=['language']
        )

        print(f"✓ Created entry: {entry_id}")

        # Update entry
        updates = {
            'content': {'concept': 'Python', 'definition': 'A high-level programming language'},
            'confidence_level': ConfidenceLevel.HIGH,
            'tags': ['language', 'programming']
        }

        success = store.update_knowledge_entry(entry_id, updates)
        assert success, "Update should succeed"
        print(f"✓ Updated entry successfully")

        # Verify update
        entry = store.get_entry_by_id(entry_id)
        assert entry is not None, "Entry should exist"
        assert entry.content['definition'] == 'A high-level programming language', "Definition should be updated"
        assert entry.confidence_level == ConfidenceLevel.HIGH, "Confidence should be updated"
        assert 'programming' in entry.tags, "New tag should be present"

        print(f"✓ Verified update: confidence={entry.confidence_level.value}, tags={entry.tags}")
        print("✅ test_update_knowledge_entry PASSED")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_merge_knowledge_entries():
    """Test merging multiple entries."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KnowledgeStore(db_path)

        # Create entries
        entry1_id = store.store_knowledge(
            knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
            content={'concept': 'Python', 'definition': 'A language'},
            confidence_level=ConfidenceLevel.MEDIUM,
            source_agent='test',
            domain='python',
            tags=['language']
        )

        entry2_id = store.store_knowledge(
            knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
            content={'concept': 'Python', 'definition': 'Used for scripting', 'version': '3.10'},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent='test',
            domain='python',
            tags=['scripting']
        )

        print(f"✓ Created two entries: {entry1_id}, {entry2_id}")

        # Merge entries
        success = store.merge_knowledge_entries(entry1_id, [entry2_id], merge_strategy="combine_content")
        assert success, "Merge should succeed"
        print(f"✓ Merged entries successfully")

        # Verify merge
        entry = store.get_entry_by_id(entry1_id)
        assert entry is not None, "Primary entry should exist"
        assert 'version' in entry.content, "Combined content from entry2 should be present"
        assert entry.confidence_level == ConfidenceLevel.HIGH, "Should take highest confidence"
        assert 'scripting' in entry.tags, "Tags should be combined"

        print(f"✓ Verified merge: confidence={entry.confidence_level.value}, tags={entry.tags}, content_keys={list(entry.content.keys())}")

        # Entry2 should be deleted
        entry2 = store.get_entry_by_id(entry2_id)
        assert entry2 is None, "Secondary entry should be deleted"
        print(f"✓ Verified secondary entry deleted")

        print("✅ test_merge_knowledge_entries PASSED")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_get_entry_by_id():
    """Test retrieving entry by ID."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        store = KnowledgeStore(db_path)

        # Create entry
        entry_id = store.store_knowledge(
            knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
            content={'concept': 'FastAPI', 'definition': 'A web framework'},
            confidence_level=ConfidenceLevel.HIGH,
            source_agent='test',
            domain='web',
            tags=['framework', 'api']
        )

        print(f"✓ Created entry: {entry_id}")

        # Retrieve entry
        entry = store.get_entry_by_id(entry_id)
        assert entry is not None, "Entry should be retrieved"
        assert entry.knowledge_id == entry_id, "Entry ID should match"
        assert entry.content['concept'] == 'FastAPI', "Content should match"
        assert entry.domain == 'web', "Domain should match"
        assert len(entry.tags) == 2, "Tags should match"

        print(f"✓ Retrieved entry: concept={entry.content['concept']}, domain={entry.domain}")

        # Test non-existent entry
        non_existent = store.get_entry_by_id('non-existent-id')
        assert non_existent is None, "Non-existent entry should return None"
        print(f"✓ Non-existent entry returns None")

        print("✅ test_get_entry_by_id PASSED")

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

def test_document_reprocessing():
    """Test document re-processing with hash detection."""
    # Create temporary files and database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        test_file = os.path.join(tmpdir, 'test.txt')

        # Write initial content
        with open(test_file, 'w') as f:
            f.write("Initial content for testing")

        # Create store and daemon
        store = KnowledgeStore(db_path)
        config = DaemonConfig(
            watch_directories=[tmpdir],
            processing_interval=1,
            enable_file_watching=False
        )
        daemon = KnowledgeDaemon(config, store)

        print(f"✓ Created test file: {test_file}")

        # First processing (new document)
        result1 = daemon.reprocess_document(test_file, force=False)
        assert result1['status'] == 'queued', "New document should be queued"
        assert result1['reason'] == 'new_document', "Should be recognized as new"
        print(f"✓ First process: {result1}")

        # Simulate document in database by manually adding it
        import hashlib
        import sqlite3
        import time
        with open(test_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO document_sources
            (doc_id, file_name, file_path, file_type, file_hash, ingestion_status, ingestion_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('test-doc-1', 'test.txt', test_file, 'txt', file_hash, 'completed', time.time()))
        conn.commit()
        conn.close()

        # Second processing (unchanged)
        result2 = daemon.reprocess_document(test_file, force=False)
        assert result2['status'] == 'skipped', "Unchanged document should be skipped"
        assert result2['reason'] == 'no_changes', "Should detect no changes"
        print(f"✓ Second process (unchanged): {result2}")

        # Modify file
        with open(test_file, 'w') as f:
            f.write("Modified content for testing")

        # Third processing (changed)
        result3 = daemon.reprocess_document(test_file, force=False)
        assert result3['status'] == 'queued', "Changed document should be queued"
        assert result3['reason'] == 'file_changed', "Should detect file change"
        assert result3['old_hash'] != result3['new_hash'], "Hashes should differ"
        print(f"✓ Third process (changed): {result3}")

        # Fourth processing (force)
        result4 = daemon.reprocess_document(test_file, force=True)
        assert result4['status'] == 'queued', "Forced reprocess should be queued"
        assert result4['reason'] == 'forced', "Should indicate forced"
        print(f"✓ Fourth process (forced): {result4}")

        print("✅ test_document_reprocessing PASSED")

def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("ENTRY LIFECYCLE MANAGEMENT TESTS")
    print("=" * 70)
    print()

    tests = [
        test_get_entry_by_id,
        test_update_knowledge_entry,
        test_merge_knowledge_entries,
        test_document_reprocessing
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            print("-" * 70)
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0

if __name__ == '__main__':
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
