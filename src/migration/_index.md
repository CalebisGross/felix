# Migration Module

## Purpose
Versioned database schema evolution providing safe, reversible migrations with automatic backups for all Felix databases.

## Key Files

### [base_migration.py](base_migration.py)
Abstract migration framework.
- **`Migration`**: Abstract base class for all migrations
- **Required Methods**:
  - `version()`: Returns migration version number
  - `description()`: Returns human-readable description
  - `up()`: Applies schema changes (forward migration)
  - `down()`: Reverts schema changes (backward migration)

### [version_manager.py](version_manager.py)
Migration version tracking and orchestration.
- **`MigrationManager`**: Manages migration execution order and version tracking
- **Features**:
  - Dependency resolution
  - Version tracking per database
  - Migration status (pending/applied/failed)
  - Rollback support
  - Transaction safety

### [backup_manager.py](backup_manager.py)
Automatic database backups before migrations.
- **`BackupManager`**: Creates timestamped backups before schema changes
- **Features**:
  - Pre-migration snapshots
  - Timestamp-based naming
  - Compression support
  - Retention policy
  - Restore functionality

## Migration Categories

### Task Migrations

#### [task_migrations.py](task_migrations.py)
Task memory schema evolution.
- **`TaskMigration001`**: Initial task memory schema
- **`TaskMigration002`**: Add task pattern tracking
- **`TaskMigration003`**: Add outcome metrics
- **`TaskMigration004`**: Add task categorization

#### [add_learning_tables.py](add_learning_tables.py)
Learning system table creation.
- Adds `workflow_patterns`, `threshold_records`, `calibration_data`, `recommendations` tables

#### [fix_learning_schema_v2.py](fix_learning_schema_v2.py)
Learning schema corrections.
- Fixes column types and constraints in learning tables

### Knowledge Migrations

#### [add_knowledge_brain.py](add_knowledge_brain.py)
Knowledge Brain system initialization.
- **`KnowledgeBrainMigration001`**: Creates `document_sources`, `knowledge_relationships`, `knowledge_fts`, `knowledge_usage` tables
- Enables FTS5 virtual table for full-text search

#### [knowledge_migrations.py](knowledge_migrations.py)
Knowledge store schema evolution.
- **`KnowledgeMigration001`**: Initial knowledge entries schema
- **`KnowledgeMigration002`**: Add embedding and source tracking columns

#### [add_knowledge_validation.py](add_knowledge_validation.py)
Knowledge validation system.
- Adds `knowledge_validation` table for truth assessment tracking

#### [fix_fts_triggers.py](fix_fts_triggers.py)
FTS5 synchronization fixes.
- Creates/fixes triggers to keep `knowledge_fts` in sync with `knowledge_entries`
- Handles insert, update, delete operations

### System Migrations

#### [create_system_actions.py](create_system_actions.py)
System command execution tracking.
- **`SystemActionsMigration001`**: Creates `command_history` table
- **`SystemActionsMigration002`**: Adds trust management columns

#### [create_feedback_system.py](create_feedback_system.py)
User feedback collection tables.
- **`FeedbackSystemMigration001`**: Creates `user_feedback`, `feedback_patterns` tables

#### [workflow_migrations.py](workflow_migrations.py)
Workflow history schema evolution.
- **`WorkflowMigration001`**: Initial workflow history table
- **`WorkflowMigration002`**: Add performance metrics and status tracking

#### [create_agent_performance.py](create_agent_performance.py)
Agent performance tracking.
- **`AgentPerformanceMigration001`**: Creates `agent_performance` table with metrics

### Additional Migrations

#### [add_audit_log_table.py](add_audit_log_table.py)
Knowledge audit log table creation.
- **Purpose**: Creates comprehensive audit trail for all CRUD operations on knowledge entries
- **Table**: `knowledge_audit_log` with timestamp, operation type, knowledge_id, user_agent, old/new values, transaction_id
- **Features**: Before/after state capture, transaction-level grouping, indexed for fast querying
- **Integration**: Used by `AuditLogger` in `src/memory/audit_log.py`

#### [add_cascade_delete.py](add_cascade_delete.py)
CASCADE DELETE foreign key constraint migration.
- **Purpose**: Adds CASCADE DELETE to `knowledge_entries.source_doc_id` foreign key
- **Effect**: Automatically deletes associated knowledge entries when a document is deleted
- **Challenge**: SQLite requires table recreation for foreign key changes
- **Process**: Create new table → copy data → drop old → rename → recreate indexes

#### [add_fts5_triggers.py](add_fts5_triggers.py)
FTS5 auto-synchronization triggers.
- **Purpose**: Keeps `knowledge_fts` virtual table automatically synchronized with `knowledge_entries`
- **Triggers**: `knowledge_entries_ai` (INSERT), `knowledge_entries_au` (UPDATE), `knowledge_entries_ad` (DELETE)
- **Background**: Replaces removed triggers that had schema mismatch issues

#### [add_watch_directories_table.py](add_watch_directories_table.py)
Watch directories tracking table.
- **Purpose**: Tracks watched directories with metadata and statistics for Knowledge Brain
- **Table**: `watch_directories` with path, enabled status, scan timestamps, document/entry counts
- **Usage**: Enables directory-level operations and visibility in Knowledge Brain GUI

## Key Concepts

### Migration Versioning

Version format: `{category}{number}`
- Task: `task001`, `task002`, ...
- Knowledge: `knowledge001`, `knowledge002`, ...
- System: `system001`, `system002`, ...

Stored in `schema_version` table per database:
```sql
CREATE TABLE schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
)
```

### Migration Workflow

```
1. Check current version
   ↓
2. Identify pending migrations
   ↓
3. Create backup
   ↓
4. Begin transaction
   ↓
5. Apply migration.up()
   ↓
6. Update schema_version
   ↓
7. Commit transaction
   ↓
8. Verify schema
```

### Rollback Process

```
1. Identify migration to rollback
   ↓
2. Create backup (current state)
   ↓
3. Begin transaction
   ↓
4. Apply migration.down()
   ↓
5. Remove from schema_version
   ↓
6. Commit transaction
   ↓
7. Verify schema
```

### Transaction Safety

All migrations run within transactions:
- **ACID guarantees**: Atomicity, Consistency, Isolation, Durability
- **All-or-nothing**: Migration fully applies or fully reverts
- **No partial state**: Database never left in inconsistent state

### Backup Strategy

**Before every migration**:
1. Create timestamped backup: `{db_name}.{timestamp}.bak`
2. Verify backup integrity
3. Proceed with migration
4. Keep backups for 30 days (configurable)

**Backup naming**: `felix_knowledge.2025-01-15_14-30-45.bak`

### FTS5 Special Handling

Full-text search requires special migration care:
- **Virtual table creation**: `CREATE VIRTUAL TABLE ... USING fts5(...)`
- **Trigger synchronization**: Auto-update FTS5 on data changes
- **Initial population**: Bulk insert existing data
- **Rebuild support**: Optimize and rebuild FTS5 index

## Creating New Migrations

### Step 1: Create Migration File
```python
from src.migration.base_migration import Migration

class MyMigration001(Migration):
    def version(self) -> str:
        return "my_feature001"

    def description(self) -> str:
        return "Add my_table for feature X"

    def up(self, conn):
        conn.execute("""
            CREATE TABLE my_table (
                id INTEGER PRIMARY KEY,
                data TEXT
            )
        """)

    def down(self, conn):
        conn.execute("DROP TABLE my_table")
```

### Step 2: Register with MigrationManager
```python
manager = MigrationManager(db_path="felix_memory.db")
manager.register(MyMigration001())
manager.apply_pending()
```

### Step 3: Test Rollback
```python
manager.rollback(to_version="previous_version")
```

## Database Files

Migrations manage these databases:
- `felix_knowledge.db` - Knowledge store
- `felix_memory.db` - Task memory
- `felix_task_memory.db` - Additional task data
- `felix_workflow_history.db` - Workflow tracking
- `felix_command_history.db` - Command execution

## Related Modules
- [memory/](../memory/) - Databases being migrated
- [knowledge/](../knowledge/) - Knowledge Brain schema
- [learning/](../learning/) - Learning system schema
- [execution/](../execution/) - Command history schema
- [feedback/](../feedback/) - Feedback system schema
