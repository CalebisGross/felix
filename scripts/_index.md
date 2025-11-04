# Scripts Directory

## Purpose
Utility scripts for database migration, maintenance, setup automation, and development tools.

## Key Files

### [migrate_databases.py](migrate_databases.py)
Database migration and schema evolution script.
- **Purpose**: Safely migrate Felix databases to new schema versions
- **What it does**:
  - Backs up existing databases before migration
  - Applies versioned schema changes
  - Validates migration success
  - Rolls back on failure
- **Databases migrated**:
  - `felix_knowledge.db` - Knowledge entries and relationships
  - `felix_memory.db` - Task memory and patterns
  - `felix_task_memory.db` - Extended task storage
  - `felix_workflow_history.db` - Workflow execution history
- **Usage**:
  ```bash
  # Run all pending migrations
  python scripts/migrate_databases.py

  # Show migration status
  python scripts/migrate_databases.py --status

  # Rollback last migration
  python scripts/migrate_databases.py --rollback
  ```
- **Prerequisites**: All Felix dependencies installed
- **Safety**: Always backs up databases to `*.db.backup` before changes

## Common Utility Scripts (Coming Soon)

### setup.py
Automated project setup and dependency installation.
```bash
python scripts/setup.py
# - Creates virtual environment
# - Installs dependencies
# - Initializes databases
# - Validates configuration
# - Checks LLM provider connectivity
```

### health_check.py
Comprehensive system health check.
```bash
python scripts/health_check.py
# - LLM provider connection
# - Database integrity
# - Memory usage
# - Configuration validity
# - Knowledge Brain status
```

### backup_databases.py
Backup all Felix databases.
```bash
python scripts/backup_databases.py --output backups/
# - Creates timestamped backup
# - Compresses databases
# - Verifies backup integrity
```

### clean_old_data.py
Clean up old workflow history and temporary data.
```bash
python scripts/clean_old_data.py --days 30
# - Remove workflows older than N days
# - Clean temporary files
# - Optimize database indexes
```

### export_knowledge.py
Export knowledge base to portable formats.
```bash
python scripts/export_knowledge.py --format json
# - Export all knowledge entries
# - Include relationships
# - Preserve metadata
# - Output: knowledge_export.json
```

## Migration System

### Migration Files
Located in `src/migration/migrations/`:
```
migrations/
├── 001_initial_schema.py
├── 002_add_knowledge_relationships.py
├── 003_add_meta_learning.py
└── 004_add_workflow_history.py
```

### Migration Structure
```python
class Migration_003_add_meta_learning:
    version = 3
    description = "Add meta-learning tracking"

    def up(self, db_connection):
        # Apply schema changes
        db_connection.execute("""
            CREATE TABLE knowledge_usage (
                id INTEGER PRIMARY KEY,
                knowledge_id INTEGER,
                workflow_id TEXT,
                usefulness REAL
            )
        """)

    def down(self, db_connection):
        # Rollback changes
        db_connection.execute("DROP TABLE knowledge_usage")
```

### Migration Workflow
1. **Check status**: `migrate_databases.py --status`
2. **Backup**: Automatic before migration
3. **Apply**: Run pending migrations sequentially
4. **Validate**: Check schema matches expected state
5. **Rollback**: If validation fails, restore backup

## Script Development Guidelines

### Creating New Scripts

1. **Add shebang and encoding**:
   ```python
   #!/usr/bin/env python3
   # -*- coding: utf-8 -*-
   ```

2. **Import from project root**:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent))
   from src.memory.knowledge_store import KnowledgeStore
   ```

3. **Add argparse for CLI**:
   ```python
   import argparse
   parser = argparse.ArgumentParser(description="Script purpose")
   parser.add_argument("--option", help="Option description")
   args = parser.parse_args()
   ```

4. **Include error handling**:
   ```python
   try:
       # Main logic
   except Exception as e:
       print(f"Error: {e}", file=sys.stderr)
       sys.exit(1)
   ```

5. **Make executable**:
   ```bash
   chmod +x scripts/new_script.py
   ```

### Script Checklist
- [ ] Clear purpose documented at top
- [ ] Accepts command-line arguments
- [ ] Provides `--help` text
- [ ] Error handling for common failures
- [ ] Logging for debugging
- [ ] Dry-run mode for destructive operations
- [ ] Exit codes (0 = success, non-zero = failure)
- [ ] Works from any directory

## Related Modules
- [src/migration/](../src/migration/) - Migration system implementation
- [src/memory/](../src/memory/) - Database schemas and operations
- [src/knowledge/](../src/knowledge/) - Knowledge store schema
