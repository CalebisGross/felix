# Felix Knowledge Base Control System - Implementation Roadmap

## ✅ COMPLETED PHASES

### Phase 1: Emergency Cleanup Tools (COMPLETE)

**Problem Solved**: 7,851 unwanted files queued from `.venv/`, `site-packages/`, etc. No way to bulk delete or preview deletions.

**Files Modified/Created**:
- `src/memory/knowledge_store.py` - Added 6 bulk delete methods (lines 982-1421)
- `src/knowledge/knowledge_cleanup.py` - New cleanup utility module
- `src/gui/knowledge_brain.py` - Added "Cleanup" tab (lines 71-74, 1329-1786)

**Features Implemented**:
1. **Bulk Delete Methods** (`KnowledgeStore`):
   - `delete_knowledge(id)` - Delete single entry
   - `preview_delete_by_pattern(pattern)` - Preview before deleting
   - `delete_documents_by_pattern(pattern, cascade)` - Delete by path pattern
   - `delete_entries_by_source_pattern(pattern)` - Delete entries only
   - `delete_orphaned_entries()` - Clean entries with no source
   - `delete_failed_documents(max_age_days)` - Remove old failed docs

2. **Cleanup Manager** (`KnowledgeCleanupManager`):
   - `cleanup_virtual_environments()` - Remove .venv, site-packages
   - `cleanup_pending_documents()` - Clean pending queue
   - `get_cleanup_recommendations()` - AI-powered cleanup suggestions
   - Convenience functions: `quick_cleanup_venv()`, `get_cleanup_report()`

3. **GUI Cleanup Tab**:
   - Recommendations section (urgent/suggested actions)
   - Quick cleanup buttons (venv, pending, orphaned, failed)
   - Custom pattern cleanup with preview
   - Results display and database statistics
   - One-click operations with confirmation dialogs

**Usage**:
```python
from src.knowledge.knowledge_cleanup import quick_cleanup_venv, get_cleanup_report

# Get recommendations
report = get_cleanup_report()
print(f"Urgent: {len(report['urgent'])} actions")

# Clean .venv files (dry run first)
preview = quick_cleanup_venv(dry_run=True)
print(f"Would delete: {preview['total_docs_deleted']} docs")

# Execute cleanup
result = quick_cleanup_venv(dry_run=False)
print(f"Deleted: {result['total_docs_deleted']} docs, {result['total_entries_deleted']} entries")
```

---

### Phase 2: Directory Management Enhancement (COMPLETE)

**Problem Solved**: No way to prevent future accidents. No tracking of which directories created which entries. No directory-level operations.

**Files Modified/Created**:
- `src/knowledge/knowledge_daemon.py` - Added exclusion system (lines 54-85, 242-261, 420-422, 534-537, 696-747)
- `src/migration/add_watch_directories_table.py` - New migration script
- `src/memory/knowledge_store.py` - Added watch directory methods (lines 1425-1602)
- `src/knowledge/directory_index.py` - New directory index system
- `src/gui/knowledge_brain.py` - Added directory stats to Overview (lines 622-646)

**Features Implemented**:

#### 2.1: Path Exclusion System
- **DaemonConfig** now includes `exclusion_patterns` field with 24 default patterns
- Automatically excludes: `.venv/`, `venv/`, `node_modules/`, `.git/`, `__pycache__/`, `dist/`, `build/`, `site-packages/`, `.tox/`, `.nox/`, `htmlcov/`, `.coverage`, `.env`, `.vscode/`, `.idea/`
- Three enforcement points:
  - Batch processing loop filters excluded paths
  - Manual directory processing tracks excluded count
  - File watcher respects exclusions in real-time
- `_should_exclude_path()` method uses fnmatch for pattern matching

#### 2.2: Watch Directories Table
- **Migration**: `add_watch_directories_table.py` (run with: `python3 src/migration/add_watch_directories_table.py`)
- **Schema**: `watch_id`, `directory_path`, `added_at`, `enabled`, `last_scan`, `document_count`, `entry_count`, `notes`
- **Indexes**: on `directory_path` and `enabled` for fast queries
- **Auto-population**: Migrates existing directories from `document_sources`
- **Results**: Found 1 existing directory with 158 documents, 2,131 entries

**KnowledgeStore Methods**:
- `add_watch_directory(path, notes)` - Register directory
- `remove_watch_directory(path)` - Unregister directory
- `update_watch_directory_stats(path)` - Refresh counts
- `get_watch_directories(enabled_only)` - List all with stats
- `toggle_watch_directory(path)` - Enable/disable

#### 2.3: Directory Index System
- **DirectoryIndex** class manages `.felix_index.json` files
- **Index structure**:
  ```json
  {
    "directory": "/absolute/path",
    "created_at": "2024-01-01T00:00:00",
    "last_updated": "2024-01-01T12:00:00",
    "documents": {
      "file.txt": {
        "doc_id": "abc123",
        "file_hash": "def456",
        "processed_at": "2024-01-01T10:00:00",
        "status": "completed",
        "entry_count": 15,
        "entry_ids": ["entry1", "entry2", ...]
      }
    },
    "statistics": {
      "total_documents": 10,
      "completed": 8,
      "failed": 1,
      "pending": 1,
      "total_entries": 150
    }
  }
  ```
- **Automatic Updates**: `_update_directory_index()` called after document processing
- **Methods**: `create()`, `load()`, `save()`, `add_document()`, `remove_document()`, `get_statistics()`, `rebuild_from_database()`, `cleanup_missing_files()`

#### 2.4: GUI Directory Management
- **Overview Tab** now shows:
  - Total watched directories
  - Enabled vs disabled count
  - Top 3 contributors by entry count
  - Directory path, document count, entry count
- **Manage Directories Dialog** (existing, enhanced):
  - Add/remove directories
  - Persistent configuration
  - Visual feedback

**Usage**:
```python
from src.memory.knowledge_store import KnowledgeStore

ks = KnowledgeStore()

# Add directory
ks.add_watch_directory("/home/user/projects", notes="My projects")

# Get all directories with stats
dirs = ks.get_watch_directories()
for d in dirs:
    print(f"{d['directory_path']}: {d['document_count']} docs, {d['entry_count']} entries")

# Update statistics
ks.update_watch_directory_stats("/home/user/projects")

# Temporarily disable
ks.toggle_watch_directory("/home/user/projects")
```

**Directory Index Usage**:
```python
from src.knowledge.directory_index import DirectoryIndex

# Get index for directory
index = DirectoryIndex("/home/user/projects")

# Get statistics
stats = index.get_statistics()
print(f"Completed: {stats['completed']}, Failed: {stats['failed']}")

# Get all documents
docs = index.get_all_documents()
for doc in docs:
    print(f"{doc['file_name']}: {doc['entry_count']} entries")

# Rebuild from database
index.rebuild_from_database(knowledge_store)
```

---

### Phase 3: Entry Lifecycle Management (COMPLETE)

**Problem Solved**: No way to edit, merge, or re-process knowledge entries. Concepts tab was read-only with limited interaction.

**Files Modified/Created**:

- `src/memory/knowledge_store.py` - Added 3 CRUD methods (lines 890-1077)
- `src/knowledge/knowledge_daemon.py` - Added reprocess_document() (lines 867-965)
- `src/gui/knowledge_brain.py` - Redesigned Concepts tab + 8 new methods (lines 224-1234)
- `tests/test_entry_lifecycle.py` - New test suite (260 lines)
- `docs/KNOWLEDGE_BRAIN_GUI_GUIDE.md` - New user documentation (500+ lines)

**Features Implemented**:

#### 3.1: Core KnowledgeStore Methods

- **`get_entry_by_id(knowledge_id)`** - Retrieve single entry for editing
- **`update_knowledge_entry(knowledge_id, updates)`** - Update content, confidence, domain, tags
  - Supports partial updates (only specified fields)
  - Automatically invalidates embeddings on content changes
  - Updates normalized tags table
  - Transaction-safe with rollback on error
- **`merge_knowledge_entries(primary_id, secondary_ids, strategy)`** - Merge multiple entries
  - Strategies: `keep_primary`, `combine_content`, `highest_confidence`
  - Combines tags and related entries from all sources
  - Takes highest confidence level
  - Deletes secondary entries after merge
  - Preserves all valuable data

#### 3.2: Document Re-processing

- **`KnowledgeDaemon.reprocess_document(file_path, force)`**:
  - MD5 hash-based change detection
  - Automatically skips unchanged files
  - Force flag to re-process without changes
  - Safely deletes old entries before reprocessing
  - Queues documents for autonomous processing
  - Returns detailed status (queued, skipped, error)

#### 3.3: GUI Concepts Tab Redesign

- **Converted from Text widget to TreeView** (lines 224-271):
  - Columns: Concept, Domain, Confidence, Definition
  - Multi-select support (Ctrl+Click, Shift+Click)
  - Keyboard shortcuts (Double-click to edit)
  - Stores knowledge_id in TreeView tags
  - Shows 100 most recent entries
  - Truncates long definitions for readability

- **Action Buttons**:
  - Edit Selected - Opens edit dialog
  - Delete Selected - Bulk delete with confirmation
  - Merge Selected - Interactive merge wizard
  - Export Selection - Save to JSON file

#### 3.4: Entry Management UI

- **Edit Dialog** (`_edit_concept_entry()`):
  - Editable fields: concept name, definition (scrollable), domain, confidence, tags
  - Dropdown selectors for domain and confidence
  - Save button with validation
  - Activity log integration
  - Automatic refresh after save

- **Delete Operations** (`_delete_selected_concepts()`):
  - Bulk delete support
  - Confirmation dialog with count
  - "Cannot be undone" warning
  - Cleans up related entry references

- **Merge Wizard** (`_merge_selected_concepts()`):
  - Select 2+ entries
  - Choose primary entry to keep
  - Automatic content combination
  - Shows entry names for clarity
  - Deletes secondary entries after merge

- **Export Functionality** (`_export_selected_concepts()`):
  - Export to JSON format
  - Includes all fields and metadata
  - File save dialog
  - Perfect for backups

#### 3.5: Context Menu & Related Features

- **Right-click Context Menu** (`_show_concept_context_menu()`):
  - Edit - Opens edit dialog
  - Delete - Deletes selected entry
  - View Related - Shows related concepts in dialog
  - View Source Document - Displays source file info

- **Document Re-processing UI** (Documents tab):
  - "Re-process Selected" button
  - Bulk re-processing support
  - Status display (queued, skipped, errors)
  - Confirmation dialog with warning

#### 3.6: Testing & Documentation

- **Test Suite** (`tests/test_entry_lifecycle.py`):
  - `test_get_entry_by_id()` - Entry retrieval
  - `test_update_knowledge_entry()` - Update operations
  - `test_merge_knowledge_entries()` - Merge strategies
  - `test_document_reprocessing()` - Hash detection

- **User Guide** (`docs/KNOWLEDGE_BRAIN_GUI_GUIDE.md`):
  - Complete GUI documentation
  - Best practices for entry management
  - Troubleshooting guide
  - Keyboard shortcuts
  - API integration examples

**Usage**:

```python
from src.memory.knowledge_store import KnowledgeStore, ConfidenceLevel

store = KnowledgeStore()

# Get entry by ID
entry = store.get_entry_by_id('entry-id-123')

# Update entry
updates = {
    'content': {'concept': 'Python', 'definition': 'Updated definition'},
    'confidence_level': ConfidenceLevel.HIGH,
    'tags': ['language', 'programming', 'scripting']
}
success = store.update_knowledge_entry('entry-id-123', updates)

# Merge entries
success = store.merge_knowledge_entries(
    primary_id='entry-1',
    secondary_ids=['entry-2', 'entry-3'],
    merge_strategy='combine_content'
)

# Re-process document
from src.knowledge.knowledge_daemon import KnowledgeDaemon
result = daemon.reprocess_document('/path/to/file.txt', force=False)
```

**GUI Usage**:

```bash
# Start Felix GUI
python -m src.gui

# Navigate to Knowledge Brain > Concepts tab
# - Double-click to edit
# - Select multiple entries and click "Merge Selected"
# - Right-click for context menu
# - Export to JSON for backups
```

---

### Phase 4: Data Integrity & Safety (COMPLETE)

**Problem Solved**: Database lacked referential integrity constraints, FTS5 index fell out of sync, and multi-operation transactions were unsafe. Manual cleanup required for orphaned entries.

**Files Modified/Created**:

- `src/migration/add_fts5_triggers.py` - New FTS5 auto-sync triggers migration
- `src/migration/add_cascade_delete.py` - New CASCADE DELETE migration
- `src/memory/knowledge_store.py` - Added transaction() context manager (lines 265-311), refactored 4 methods
- `tests/test_fts5_triggers.py` - New test suite (5 comprehensive tests)
- `tests/test_cascade_delete.py` - New test suite (5 comprehensive tests)

**Features Implemented**:

#### 4.1: FTS5 Auto-Sync Triggers

- **Migration**: `add_fts5_triggers.py` - Creates database triggers for automatic FTS5 synchronization
- **Triggers Created**:
  - `knowledge_entries_ai` - AFTER INSERT: Populates FTS5 with new entries
  - `knowledge_entries_au` - AFTER UPDATE: Syncs FTS5 on content changes
  - `knowledge_entries_ad` - AFTER DELETE: Removes FTS5 entries
- **Content Extraction**: Uses JSON extraction logic for concept, definition, and summary fields
- **Migration Results**: Fixed sync issue - rebuilt 4,686 FTS entries
- **Benefits**: FTS5 always in sync, no manual REBUILD needed, search always accurate

#### 4.2: Transaction Context Manager

- **New Method**: `KnowledgeStore.transaction()` context manager (lines 265-311)
- **Features**:
  - Automatic BEGIN TRANSACTION
  - PRAGMA foreign_keys = ON for CASCADE DELETE support
  - Auto-commit on success, auto-rollback on failure
  - Proper connection cleanup in finally block
  - Detailed error logging
- **Refactored Methods** (4 methods now use safe transactions):
  1. `delete_documents_by_pattern()` - Line 1360
  2. `delete_entries_by_source_pattern()` - Line 1450
  3. `delete_orphaned_entries()` - Line 1533
  4. `delete_failed_documents()` - Line 1601
- **Benefits**: Atomic operations, automatic rollback on errors, prevents partial updates

#### 4.3: CASCADE DELETE Foreign Key

- **Migration**: `add_cascade_delete.py` - Recreates knowledge_entries table with CASCADE DELETE
- **Challenge Solved**: SQLite doesn't support ALTER TABLE for foreign keys
- **Migration Process**:
  1. Created new table with `FOREIGN KEY(source_doc_id) REFERENCES document_sources(doc_id) ON DELETE CASCADE`
  2. Copied all 4,686 knowledge entries
  3. Dropped old table
  4. Renamed new table
  5. Recreated 6 indexes
- **Verification**:
  - ✓ CASCADE DELETE constraint active
  - ✓ All indexes present
  - ✓ Data integrity verified (4,686 entries)
- **Benefits**: Orphaned entries automatically deleted when documents removed, no manual cleanup needed

#### 4.4: Comprehensive Testing

- **FTS5 Triggers Tests** (5 tests, all passed):
  - Trigger definitions correct
  - Sync consistency verified
  - INSERT trigger works
  - UPDATE trigger works
  - DELETE trigger works

- **CASCADE DELETE Tests** (5 tests, all passed):
  - Foreign key constraint exists
  - Basic CASCADE DELETE (1 document → 1 entry deleted)
  - Multiple entries CASCADE deleted with document
  - Tags table cleaned up automatically
  - Entries without source_doc_id unaffected

**Usage**:

```python
from src.memory.knowledge_store import KnowledgeStore

store = KnowledgeStore()

# Use transaction context manager for safe multi-operation transactions
with store.transaction() as conn:
    # Multiple operations - all or nothing
    conn.execute("DELETE FROM document_sources WHERE doc_id = ?", (doc_id,))
    # Associated entries automatically deleted via CASCADE DELETE
    conn.execute("UPDATE knowledge_entries SET domain = ?", ("updated",))
    # Changes committed automatically if no exceptions

# FTS5 automatically stays in sync via triggers
# No manual rebuild needed!
```

**Running Migrations**:

```bash
# Backup database first
cp felix_knowledge.db felix_knowledge.db.backup_$(date +%Y%m%d)

# Run FTS5 triggers migration
python3 src/migration/add_fts5_triggers.py felix_knowledge.db

# Run CASCADE DELETE migration
python3 src/migration/add_cascade_delete.py felix_knowledge.db

# Verify migrations
python3 tests/test_fts5_triggers.py felix_knowledge.db
python3 tests/test_cascade_delete.py felix_knowledge.db
```

**Database State After Phase 4**:

- **Entries**: 4,686 knowledge entries
- **Documents**: 158 documents
- **FTS5 Index**: 4,686 entries (fully synchronized via triggers)
- **Foreign Keys**: ENABLED with CASCADE DELETE active
- **Indexes**: All 6 indexes recreated and verified
- **Orphaned Entries**: 0 (CASCADE DELETE prevents orphans)

**Note on Component 4.4 (Separate Relationships Table)**:

- Status: Deferred (optional enhancement)
- Reason: High complexity, breaking API changes, medium priority
- Can be implemented in future sprint if queryable relationships needed

---

### Phase 5: Advanced Features (COMPLETE)

**Problem Solved**: Knowledge base lacked advanced querying, analytics visibility, quality assurance tools, and production-ready backup/restore capabilities.

**Status**: ✅ COMPLETE

**Files Modified/Created**:
- `src/memory/audit_log.py` - New audit logging system (470 lines)
- `src/migration/add_audit_log_table.py` - Audit log table migration (268 lines)
- `src/memory/knowledge_store.py` - Added @audit_logged decorators (8 methods) + advanced search & analytics (488 lines)
- `src/knowledge/backup_manager_extended.py` - Extended backup manager with JSON export/import (512 lines)
- `src/knowledge/knowledge_daemon.py` - Added scheduled backup thread (Mode D, 79 lines)
- `src/knowledge/quality_checker.py` - Complete quality checking system (686 lines)
- `src/gui/knowledge_brain.py` - Added Audit tab (290 lines) + Analytics tab (421 lines)

**Features Implemented**:

**Component Breakdown (Originally Planned)**:

#### 5.1: Audit Log
- **File**: `src/memory/audit_log.py` (new)
- **Schema**:
  ```sql
  CREATE TABLE knowledge_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,  -- 'insert', 'update', 'delete'
    table_name TEXT NOT NULL,
    record_id TEXT,
    user TEXT,
    timestamp REAL NOT NULL,
    details_json TEXT,
    ip_address TEXT
  );
  ```
- **Integration**: Add logging calls to all CRUD operations
- **GUI**: New "Audit" tab showing recent changes with filtering

#### 5.2: Backup & Restore
- **File**: `src/knowledge/backup_manager.py` (new)
- **Features**:
  - Full database export to JSON (with compression)
  - Selective export (by domain, directory, date range)
  - Import with conflict resolution (skip, replace, merge)
  - Scheduled backups (daily/weekly via daemon)
  - Verify backup integrity
- **GUI**: "Backup" tab with one-click backup/restore

#### 5.3: Advanced Search & Analytics
- **File**: `src/gui/knowledge_brain.py` (new "Analytics" tab)
- **Features**:
  - Multi-field search (content, domain, tags, confidence, date)
  - Saved search queries
  - Statistics dashboard:
    - Entries per domain over time (chart)
    - Entries per directory (chart)
    - Growth trends
    - Quality metrics (avg confidence by domain)
  - Duplicate detection tool
  - Quality reports:
    - Low confidence entries needing review
    - Unvalidated entries
    - Entries with no relationships
    - Orphaned concepts

#### 5.4: Knowledge Quality Tools (✅ IMPLEMENTED)

- **File**: `src/knowledge/quality_checker.py` (686 lines)
- **Implemented Features**:
  - `QualityChecker` class with comprehensive quality analysis
  - **Duplicate Detection**: 3 methods (embedding similarity, text similarity, concept matching)
    - Embedding-based: Cosine similarity on vector embeddings (threshold: 0.90)
    - Text-based: Edit distance ratio with domain filtering
    - Concept-based: Exact concept name matching across domains
  - **Contradiction Finding**: Detects same concepts with conflicting definitions
  - **Quality Scoring**: Composite scoring (confidence 35%, relationships 25%, validation 25%, success rate 15%)
  - **Merge Suggestions**: Intelligent ranking based on confidence levels and access patterns
  - **Dataclasses**: `DuplicateCandidate`, `ContradictionCandidate`, `QualityScore`
- **GUI Integration** (421 lines in Analytics tab):
  - Analytics dashboard with statistics visualization
  - Quality report generation with recommendations
  - Duplicate candidates TreeView with merge functionality
  - Side-by-side comparison dialogs
  - Bulk merge operations with confirmation

**Usage**:

```python
from src.knowledge.quality_checker import QualityChecker
from src.memory.knowledge_store import KnowledgeStore

ks = KnowledgeStore()
checker = QualityChecker(ks)

# Find duplicates
duplicates = checker.find_duplicates(similarity_threshold=0.90, method="auto")
print(f"Found {len(duplicates)} duplicate pairs")

# Calculate quality score
score = checker.calculate_quality_score("entry-id-123")
print(f"Overall quality: {score.overall_score:.2f}")
print(f"Issues: {score.issues}")

# Get merge suggestions
suggestions = checker.get_merge_suggestions(duplicates)
for suggestion in suggestions:
    print(f"Merge {suggestion['secondary_id']} into {suggestion['primary_id']}")
    print(f"Reason: {suggestion['reason']}")
```

---

## 📊 IMPLEMENTATION SUMMARY

### Completed (ALL PHASES 1-5)

**Phase 1-4:**

- ✅ Emergency cleanup tools (bulk delete, pattern-based cleanup)
- ✅ Path exclusion system (prevents future accidents)
- ✅ Watch directories table (tracks contributions)
- ✅ Directory index files (enables directory-level operations)
- ✅ GUI enhancements (Cleanup tab, directory stats)
- ✅ Entry lifecycle management (edit, delete, merge, re-process)
- ✅ Concepts tab redesign (TreeView, bulk operations)
- ✅ Context menu and keyboard shortcuts
- ✅ Document re-processing with hash detection
- ✅ FTS5 auto-sync triggers (INSERT, UPDATE, DELETE)
- ✅ CASCADE DELETE foreign key (automatic orphan cleanup)
- ✅ Transaction context manager (safe multi-operation transactions)
- ✅ Comprehensive test suites (10 tests, all passing)

**Phase 5 (NEW):**

- ✅ Complete audit logging system with @audit_logged decorator
- ✅ Audit log migration and database table (9 columns, 4 indexes)
- ✅ Audit tab in GUI with filtering, export, and history viewing
- ✅ Extended backup manager with JSON export/import
- ✅ Conflict resolution strategies (skip, replace, merge)
- ✅ Scheduled backup thread in KnowledgeDaemon (Mode D)
- ✅ Advanced search with multi-field filtering and AND/OR logic
- ✅ Analytics data generation (growth trends, quality metrics, domain stats)
- ✅ Quality report generation (6 categories of issues)
- ✅ Quality checker with duplicate detection (3 algorithms)
- ✅ Analytics tab in GUI with visualizations and quality tools
- ✅ Duplicate merge workflow with side-by-side comparison

### In Progress

- None

### System Status

- **Phase 5**: 3-4 days (Medium Priority)

---

## 🎯 RECOMMENDED NEXT STEPS

1. **System Status** (✅ Production Ready):
   - All critical phases (1-4) complete
   - Database integrity ensured with CASCADE DELETE
   - FTS5 search automatically synchronized
   - Safe transactions with automatic rollback

2. **Optional Enhancements** (next 1-2 months):
   - **Phase 5** (advanced features) - audit log, backup/restore, analytics
   - **Separate Relationships Table** - if queryable typed relationships needed

3. **Immediate Actions** (if needed):
   - Run cleanup to remove unwanted files: `felix chat -p "cleanup venv"` or use GUI Cleanup tab
   - Verify exclusion system is working with new scans

4. **Testing Strategy**:
   - Each phase should include:
     - Unit tests for new methods
     - Integration tests with GUI
     - Migration rollback verification
     - Performance testing with 10k+ entries

---

## 🔧 MAINTENANCE NOTES

### Database Migrations
All migrations are in `src/migration/` with format: `<verb>_<description>.py`

**Existing Migrations**:
- `add_knowledge_brain.py` - Initial knowledge brain tables
- `add_watch_directories_table.py` - Watch directories tracking
- `add_fts5_triggers.py` - FTS5 auto-sync triggers (Phase 4)
- `add_cascade_delete.py` - CASCADE DELETE foreign key (Phase 4)

**Run Migration**:
```bash
python3 src/migration/<migration_file>.py felix_knowledge.db
```

**Verify Migration**:
```bash
sqlite3 felix_knowledge.db ".schema <table_name>"
```

### Configuration Files
- `config/llm.yaml` - LLM provider settings (unchanged)
- `config/trust_rules.yaml` - Trust system rules (unchanged)
- `config/prompts.yaml` - Agent prompts (unchanged)
- Watch directories stored in daemon config (saved by GUI)

### GUI Tabs
1. **Overview** - Status, daemon control, directory stats
2. **Documents** - Browse ingested sources
3. **Concepts** - Explore knowledge entries
4. **Activity** - Processing log
5. **Relationships** - Knowledge graph
6. **Cleanup** - Bulk cleanup operations *(NEW)*

### Key Classes & Methods

**Cleanup**:
- `KnowledgeStore.delete_documents_by_pattern(pattern, cascade)`
- `KnowledgeStore.delete_orphaned_entries()`
- `KnowledgeCleanupManager.cleanup_virtual_environments()`
- `KnowledgeCleanupManager.get_cleanup_recommendations()`

**Directory Management**:
- `KnowledgeStore.add_watch_directory(path, notes)`
- `KnowledgeStore.get_watch_directories(enabled_only)`
- `KnowledgeStore.update_watch_directory_stats(path)`
- `DirectoryIndex.add_document(...)` - Auto-called during processing

**Exclusions**:

- `DaemonConfig.exclusion_patterns` - List of patterns to exclude
- `KnowledgeDaemon._should_exclude_path(path)` - Check if excluded

**Entry Lifecycle** (Phase 3):

- `KnowledgeStore.get_entry_by_id(knowledge_id)` - Retrieve single entry
- `KnowledgeStore.update_knowledge_entry(knowledge_id, updates)` - Update entry fields
- `KnowledgeStore.merge_knowledge_entries(primary_id, secondary_ids, strategy)` - Merge multiple entries
- `KnowledgeDaemon.reprocess_document(file_path, force)` - Re-process documents with hash detection

**Data Integrity** (Phase 4):

- `KnowledgeStore.transaction()` - Context manager for safe transactions with auto-rollback
- CASCADE DELETE constraint - Automatic orphan cleanup when documents deleted
- FTS5 triggers - Automatic search index synchronization (INSERT, UPDATE, DELETE)

---

## 📝 CHANGE LOG

### 2025-01-18 - Phase 4 Complete

- Implemented FTS5 auto-sync triggers for automatic search index synchronization
- Added CASCADE DELETE foreign key constraint for automatic orphan cleanup
- Created transaction() context manager for safe multi-operation transactions
- Refactored 4 KnowledgeStore methods to use transaction context manager
- Fixed FTS5 sync issue: rebuilt 4,686 entries
- Created comprehensive test suites (tests/test_fts5_triggers.py, tests/test_cascade_delete.py)
- All 10 tests passing: 5 FTS5 tests + 5 CASCADE DELETE tests
- Database now production-ready with referential integrity

### 2024-01-18 - Phase 3 Complete

- Implemented entry lifecycle management (edit, delete, merge, re-process)
- Redesigned Concepts tab with TreeView interface
- Added context menu and bulk operations
- Created comprehensive test suite (tests/test_entry_lifecycle.py)
- Added user documentation (docs/KNOWLEDGE_BRAIN_GUI_GUIDE.md)

### 2024-11-18 - Phase 1 & 2 Complete
- Added comprehensive cleanup tools (Phase 1)
- Implemented directory management enhancement (Phase 2)
- Created roadmap document
- Updated documentation

### Previous Changes
- See git history for full change log
- Major milestone: Knowledge Brain system integrated (Oct 2024)
