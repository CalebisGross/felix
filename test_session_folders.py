#!/usr/bin/env python3
"""
Test script for SessionManager folder functionality.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli_chat.session_manager import SessionManager, Session, Folder

def test_folder_functionality():
    """Test all folder-related functionality."""

    # Use a test database
    test_db = "test_session_folders.db"

    # Remove test db if it exists
    if os.path.exists(test_db):
        os.remove(test_db)

    print("Initializing SessionManager...")
    manager = SessionManager(db_path=test_db)

    # Test 1: Create folders
    print("\n1. Testing folder creation...")
    folder1_id = manager.create_folder("Work Projects")
    folder2_id = manager.create_folder("Personal")
    folder3_id = manager.create_folder("Sub-folder", parent_id=folder1_id)
    print(f"   Created folders: {folder1_id}, {folder2_id}, {folder3_id}")

    # Test 2: Get all folders
    print("\n2. Testing get_folders()...")
    folders = manager.get_folders()
    print(f"   Found {len(folders)} folders")
    for folder in folders:
        print(f"   - {folder.name} (ID: {folder.folder_id}, Parent: {folder.parent_folder_id})")

    # Test 3: Get single folder
    print("\n3. Testing get_folder()...")
    folder = manager.get_folder(folder1_id)
    if folder:
        print(f"   Retrieved: {folder.name}")

    # Test 4: Create sessions with folder assignment
    print("\n4. Testing session creation with folders...")
    session1_id = manager.create_session(title="Session in Work")
    manager.move_session_to_folder(session1_id, folder1_id)

    session2_id = manager.create_session(title="Session in Personal")
    manager.move_session_to_folder(session2_id, folder2_id)

    session3_id = manager.create_session(title="Session in Root")
    print(f"   Created 3 sessions")

    # Test 5: Pin a session
    print("\n5. Testing session pinning...")
    manager.set_session_pinned(session1_id, True)
    print(f"   Pinned session: {session1_id}")

    # Test 6: Get sessions in folder
    print("\n6. Testing get_sessions_in_folder()...")
    work_sessions = manager.get_sessions_in_folder(folder1_id)
    print(f"   Work folder has {len(work_sessions)} sessions")
    for session in work_sessions:
        print(f"   - {session.title} (Pinned: {session.pinned})")

    root_sessions = manager.get_sessions_in_folder(None)
    print(f"   Root has {len(root_sessions)} sessions")

    # Test 7: Update session mode and knowledge settings
    print("\n7. Testing session mode and knowledge settings...")
    manager.update_session_mode(session1_id, "advanced")
    manager.update_session_knowledge_enabled(session1_id, False)

    updated_session = manager.get_session(session1_id)
    print(f"   Session mode: {updated_session.mode}")
    print(f"   Knowledge enabled: {updated_session.knowledge_enabled}")

    # Test 8: Update folder
    print("\n8. Testing folder update...")
    manager.update_folder(folder1_id, name="Work Projects (Updated)")
    updated_folder = manager.get_folder(folder1_id)
    print(f"   Updated folder name: {updated_folder.name}")

    # Test 9: Delete folder (sessions should move to root)
    print("\n9. Testing folder deletion...")
    manager.delete_folder(folder2_id)
    remaining_folders = manager.get_folders()
    print(f"   Remaining folders: {len(remaining_folders)}")

    # Check if session moved to root
    session2 = manager.get_session(session2_id)
    print(f"   Session2 folder_id after deletion: {session2.folder_id}")

    # Test 10: Verify migration works
    print("\n10. Testing database migration (existing fields)...")
    session = manager.get_session(session1_id)
    print(f"   Session has all new fields:")
    print(f"   - folder_id: {session.folder_id}")
    print(f"   - pinned: {session.pinned}")
    print(f"   - position: {session.position}")
    print(f"   - mode: {session.mode}")
    print(f"   - knowledge_enabled: {session.knowledge_enabled}")

    print("\n✓ All tests completed successfully!")

    # Cleanup
    os.remove(test_db)
    print(f"\nCleaned up test database: {test_db}")

if __name__ == "__main__":
    test_folder_functionality()
