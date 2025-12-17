"""
Test Entry Lifecycle Management

Tests for Phase 3: edit, delete, merge, re-process operations.
"""

import os
import tempfile
from pathlib import Path
from src.memory.knowledge_store import KnowledgeStore, KnowledgeType, ConfidenceLevel

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

def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("ENTRY LIFECYCLE MANAGEMENT TESTS")
    print("=" * 70)
    print()

    tests = [
        test_get_entry_by_id,
        test_update_knowledge_entry,
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
