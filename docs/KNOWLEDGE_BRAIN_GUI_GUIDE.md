# Felix Knowledge Brain - GUI User Guide

## Overview

The Felix Knowledge Brain GUI provides a comprehensive interface for managing your knowledge base. This guide covers all features available through the GUI, including the new Entry Lifecycle Management features introduced in Phase 3.

## Accessing the Knowledge Brain GUI

```bash
# Start the Felix GUI
python -m src.gui

# Navigate to the Knowledge Brain tab
```

## GUI Tabs

The Knowledge Brain interface consists of 6 tabs:

1. **Overview** - System status and daemon control
2. **Documents** - Browse and manage ingested documents
3. **Concepts** - Explore and edit knowledge entries
4. **Activity** - Real-time processing log
5. **Relationships** - Knowledge graph visualization
6. **Cleanup** - Bulk cleanup operations

---

## Overview Tab

### Daemon Control

**Start/Stop the Knowledge Daemon:**
- Click "Start Daemon" to begin autonomous document processing
- Click "Stop Daemon" to pause processing
- Status indicator shows current daemon state

### Statistics Dashboard

View real-time statistics:
- Total documents processed
- Total knowledge entries
- Processing status breakdown
- Watched directories and their contributions

---

## Documents Tab

### Browsing Documents

**Filter documents by status:**
- All documents
- Completed
- Processing
- Pending
- Failed

**View document details:**
- Double-click any document to see full details
- File name, type, status, chunk count, concept count

### Re-processing Documents (NEW in Phase 3)

**When to re-process:**
- File content has changed
- Processing failed initially
- Want to re-analyze with updated comprehension engine

**How to re-process:**
1. Select one or more documents in the list
2. Click "🔄 Re-process Selected" button
3. Confirm the operation
4. System will:
   - Delete existing knowledge entries
   - Reset document status to "pending"
   - Queue document for re-processing
   - Detect file changes via MD5 hash

**Re-processing Options:**
- **Automatic change detection**: Only re-processes if file hash changed
- **Force re-process**: Re-processes even if file unchanged
- **Bulk re-processing**: Select multiple documents at once

**Status Messages:**
- **Queued**: Document added to processing queue
- **Skipped**: No changes detected (hash match)
- **Error**: Re-processing failed (check error message)

---

## Concepts Tab (Enhanced in Phase 3)

The Concepts tab is your primary interface for viewing and editing knowledge entries. It has been completely redesigned with a TreeView interface for better interaction.

### Viewing Concepts

**TreeView Columns:**
- **Concept**: Name of the concept or knowledge entry
- **Domain**: Category (python, web, ai, database, etc.)
- **Confidence**: Confidence level (low, medium, high, verified)
- **Definition**: Summary or definition (truncated to 100 chars)

**Search and Filter:**
- Use the search box to find specific concepts
- Filter by domain using the dropdown
- Click "Show All" to reset filters

**Navigation:**
- Scroll through entries
- Click to select single entry
- Ctrl+Click or Shift+Click for multiple selection
- Double-click to edit entry

### Editing Entries (NEW)

**Edit a single entry:**

1. **Select entry** - Double-click or click "Edit Selected" button
2. **Edit dialog opens** with editable fields:
   - **Concept Name**: Main identifier for the knowledge entry
   - **Definition**: Detailed description or explanation
   - **Domain**: Category dropdown (python, web, ai, database, general, system)
   - **Confidence**: Confidence level dropdown (low, medium, high, verified)
   - **Tags**: Comma-separated tags for organization
3. **Save changes** - Click "Save" button
4. **Results**:
   - Entry updated in database
   - TreeView refreshed automatically
   - Activity log updated
   - Embeddings invalidated (will be regenerated)

**Keyboard shortcut:**
- Double-click entry in TreeView to edit

### Deleting Entries (NEW)

**Delete selected entries:**

1. **Select one or more entries** in the TreeView
2. **Click "Delete Selected"** button
3. **Confirm deletion** - Review count and confirm
4. **System performs:**
   - Deletes knowledge entries from database
   - Removes from normalized tags table
   - Updates FTS5 search index
   - Cleans up related entry references in other entries
   - Updates activity log

**Safety features:**
- Confirmation dialog with entry count
- "This action cannot be undone" warning
- Preserves document sources (only deletes knowledge entries)
- Automatic cleanup of dangling references

### Merging Entries (NEW)

**When to merge:**
- Duplicate concepts with different names
- Complementary information across multiple entries
- Consolidating knowledge from multiple sources

**How to merge:**

1. **Select 2 or more entries** in the TreeView
2. **Click "Merge Selected"** button
3. **Select primary entry** to keep (in dialog)
4. **Choose merge strategy**:
   - **keep_primary** (default): Keep primary entry content only
   - **combine_content**: Merge content from all entries
   - **highest_confidence**: Use entry with highest confidence
5. **Confirm merge**
6. **System performs:**
   - Merges content based on strategy
   - Combines all tags (no duplicates)
   - Combines related entries lists
   - Takes highest confidence level
   - Updates primary entry
   - Deletes secondary entries
   - Updates activity log

**Merge strategies explained:**

- **keep_primary**: Primary entry remains unchanged, only tags and related entries are merged
- **combine_content**: All content fields from secondary entries are added to primary (new keys only)
- **highest_confidence**: Automatically selects the entry with highest confidence as primary

**Example merge:**
```
Entry 1: {"concept": "Python", "definition": "A language"}
Entry 2: {"concept": "Python", "version": "3.10", "definition": "For scripting"}

After merge (combine_content):
{
  "concept": "Python",
  "definition": "A language",
  "version": "3.10"  // Added from Entry 2
}
```

### Exporting Entries (NEW)

**Export selected entries to JSON:**

1. **Select one or more entries** in the TreeView
2. **Click "Export Selection"** button
3. **Choose file location** - Save as JSON file
4. **System exports:**
   - Full entry data (all fields)
   - Content, metadata, relationships
   - Timestamps and statistics
   - Tags and confidence levels

**Export format:**
```json
[
  {
    "knowledge_id": "abc123",
    "knowledge_type": "domain_expertise",
    "content": {
      "concept": "Python",
      "definition": "A programming language"
    },
    "confidence_level": "high",
    "domain": "python",
    "tags": ["language", "programming"],
    "created_at": 1699564800.0,
    "updated_at": 1699568400.0,
    "related_entries": ["def456", "ghi789"]
  }
]
```

**Use cases:**
- Backup important knowledge entries
- Share knowledge with other Felix instances
- Review entry structure for debugging
- Export for external analysis

### Context Menu (NEW)

**Right-click any entry** to access quick actions:

**Actions:**
- **Edit**: Open edit dialog for selected entry
- **Delete**: Delete selected entry
- **View Related**: Show related concepts in popup dialog
- **View Source Document**: Display source document information

**View Related Concepts:**
- Shows concepts related to selected entry
- Displays relationship type (explicit, similarity, co-occurrence)
- Shows relationship strength (0.0-1.0)
- Maximum depth: 2 levels

**View Source Document:**
- Displays source file information
- File name, path, type, status
- Chunk index for pinpointing location
- Useful for understanding entry origin

---

## Activity Tab

**Real-time processing log:**
- Timestamped entries
- Processing events
- Success/failure notifications
- Error messages

**Actions:**
- **Refresh**: Update activity log
- **Clear Log**: Remove all entries

---

## Relationships Tab

**Knowledge graph exploration:**
- View relationships between concepts
- Filter by domain
- Explore concept networks
- Search for specific relationships

---

## Cleanup Tab

**Bulk cleanup operations:**
- Remove virtual environment files
- Clean pending documents
- Delete orphaned entries
- Remove failed documents
- Custom pattern-based cleanup

See [KNOWLEDGE_CONTROL_ROADMAP.md](../KNOWLEDGE_CONTROL_ROADMAP.md) for details on cleanup features.

---

## Best Practices

### Entry Management

**1. Regular Review:**
- Periodically review low-confidence entries
- Merge duplicate concepts
- Update definitions as knowledge evolves

**2. Tagging Strategy:**
- Use consistent tag names
- Keep tags short and descriptive
- Tag by topic, technology, and use case
- Example: `["api", "rest", "web", "python"]`

**3. Confidence Levels:**
- **Low**: Uncertain or incomplete information
- **Medium**: Generally accurate but needs validation
- **High**: Verified and reliable
- **Verified**: Manually reviewed and confirmed

**4. Domain Organization:**
- Assign appropriate domain to each entry
- Use custom domains for specialized topics
- Keep domain names lowercase
- Common domains: python, web, ai, database, general, system

### Document Re-processing

**When to re-process:**
- ✅ File content updated
- ✅ Processing failed with errors
- ✅ Comprehension engine improved
- ✅ Want to re-analyze with new settings

**When NOT to re-process:**
- ❌ File unchanged (will be skipped anyway)
- ❌ Document is being processed (wait for completion)
- ❌ Just to "refresh" without reason (wastes resources)

### Merge vs. Delete

**Choose Merge when:**
- Entries contain complementary information
- Same concept, different perspectives
- Want to preserve all data

**Choose Delete when:**
- Entry is completely wrong
- Duplicate with no additional value
- Outdated or irrelevant information

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Edit entry | Double-click |
| Context menu | Right-click |
| Multi-select | Ctrl+Click |
| Range select | Shift+Click |
| Select all | Ctrl+A (in TreeView) |

---

## Troubleshooting

### Entry won't edit
- **Issue**: Edit dialog doesn't open
- **Solution**: Check that knowledge_store is initialized
- **Solution**: Verify entry exists with Get Entry By ID

### Merge fails
- **Issue**: "Merge failed" error
- **Solution**: Ensure all selected entries exist
- **Solution**: Check that entries have valid content structure

### Re-process skipped
- **Issue**: Documents always skipped
- **Solution**: File may not have changed (hash matches)
- **Solution**: Use force=True option (not exposed in GUI)
- **Solution**: Delete document from database and re-ingest

### Export empty
- **Issue**: Exported JSON is empty
- **Solution**: Ensure entries are selected before exporting
- **Solution**: Check that selected entries have valid IDs

---

## API Integration

For programmatic access to these features, use the Knowledge Store API:

```python
from src.memory.knowledge_store import KnowledgeStore

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
```

See [KNOWLEDGE_BRAIN_API.md](KNOWLEDGE_BRAIN_API.md) for complete API documentation.

---

## Testing

Test entry lifecycle features:

```bash
# Run entry lifecycle tests
python tests/test_entry_lifecycle.py

# Tests cover:
# - Get entry by ID
# - Update entry fields
# - Merge multiple entries
# - Document re-processing with hash detection
```

---

## Future Enhancements

Planned features (Phase 4 & 5):
- Data integrity improvements (CASCADE DELETE, FTS5 triggers)
- Audit log for change tracking
- Backup and restore functionality
- Advanced analytics dashboard
- Quality checking tools
- Duplicate detection and merge suggestions

See [KNOWLEDGE_CONTROL_ROADMAP.md](../KNOWLEDGE_CONTROL_ROADMAP.md) for complete roadmap.
