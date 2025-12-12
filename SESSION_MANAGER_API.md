# SessionManager API Reference

Complete API reference for the extended SessionManager with folder support.

## Data Classes

### Message
```python
Message(
    role: str,                      # 'user', 'assistant', 'system'
    content: str,                   # Message content
    workflow_id: Optional[str] = None,
    message_id: Optional[int] = None,
    timestamp: Optional[datetime] = None
)
```

### Folder
```python
@dataclass
Folder:
    folder_id: str                  # Unique folder identifier
    name: str                       # Folder name
    parent_folder_id: Optional[str] = None  # Parent folder (for nesting)
    position: int = 0               # Custom sort position
    created_at: Optional[datetime] = None
```

### Session
```python
Session(
    session_id: str,                # Unique session identifier
    created_at: datetime,           # Creation timestamp
    last_active: datetime,          # Last activity timestamp
    message_count: int,             # Number of messages
    title: Optional[str] = None,    # Session title
    tags: Optional[List[str]] = None,  # Session tags
    folder_id: Optional[str] = None,   # Containing folder
    pinned: bool = False,           # Is session pinned?
    position: int = 0,              # Custom sort position
    mode: str = 'simple',           # Chat mode
    knowledge_enabled: bool = True  # Knowledge system enabled?
)
```

## SessionManager Methods

### Initialization
```python
SessionManager(db_path: str = "felix_cli_sessions.db")
```

### Session Management

#### `create_session(title: Optional[str] = None) -> str`
Create a new chat session.
- **Returns**: Session ID
- **Note**: New columns get default values automatically

#### `get_session(session_id: str) -> Optional[Session]`
Retrieve a single session by ID.
- **Returns**: Session object or None if not found

#### `list_sessions(limit: int = 20) -> List[Session]`
List recent sessions, most recent first.
- **Returns**: List of sessions

#### `get_last_session() -> Optional[str]`
Get the most recently active session ID.
- **Returns**: Session ID or None

#### `delete_session(session_id: str)`
Delete a session and all its messages.

#### `update_session_activity(session_id: str)`
Update the last_active timestamp for a session.

#### `set_title(session_id: str, title: str)`
Set the title for a session.

#### `generate_auto_title(session_id: str, max_length: int = 50) -> Optional[str]`
Generate an automatic title from the first user message.
- **Returns**: Generated title or None

### Message Management

#### `add_message(session_id: str, role: str, content: str, workflow_id: Optional[str] = None) -> Message`
Add a message to a session.
- **Returns**: Message object with ID

#### `get_messages(session_id: str, limit: Optional[int] = None, offset: int = 0) -> List[Message]`
Retrieve messages for a session.
- **Returns**: List of messages in chronological order

#### `get_recent_context(session_id: str, message_count: int = 10) -> List[Message]`
Get the most recent N messages for context window.
- **Returns**: List of recent messages

#### `search_messages(session_id: str, query: str) -> List[Message]`
Search messages by content.
- **Returns**: List of matching messages

#### `get_last_workflow_id(session_id: str) -> Optional[str]`
Get the most recent workflow ID from the session.
- **Returns**: Workflow ID or None

### Tag Management

#### `add_tags(session_id: str, tags: List[str])`
Add tags to a session (duplicates ignored).

#### `remove_tags(session_id: str, tags: List[str])`
Remove tags from a session.

### Search & Filter

#### `search_sessions(query: str, limit: int = 20) -> List[Session]`
Search sessions by keyword in title or messages.
- **Returns**: List of matching sessions

#### `filter_sessions_by_tags(tags: List[str], match_all: bool = False, limit: int = 20) -> List[Session]`
Filter sessions by tags.
- **match_all**: If True, session must have ALL tags. If False, ANY tag matches.
- **Returns**: List of matching sessions

#### `get_sessions_today() -> List[Session]`
Get all sessions active today.
- **Returns**: List of sessions

### Import/Export

#### `export_session(session_id: str) -> Dict`
Export a session and all its messages to a dictionary.
- **Returns**: Dictionary with session and messages

#### `import_session(data: Dict) -> Optional[str]`
Import a session from exported data.
- **Returns**: New session ID or None

### Folder Management (NEW)

#### `create_folder(name: str, parent_id: Optional[str] = None) -> str`
Create a new folder for organizing sessions.
- **name**: Folder name
- **parent_id**: Optional parent folder ID for nested folders
- **Returns**: The new folder ID

#### `get_folders() -> List[Folder]`
Get all folders ordered by position.
- **Returns**: List of all folders

#### `get_folder(folder_id: str) -> Optional[Folder]`
Get a single folder by ID.
- **Returns**: Folder object or None if not found

#### `update_folder(folder_id: str, name: Optional[str] = None, parent_id: Optional[str] = None)`
Update folder properties.
- **name**: New folder name (optional)
- **parent_id**: New parent folder ID (optional)

#### `delete_folder(folder_id: str)`
Delete a folder. Sessions in the folder are moved to root.
- **Note**: No sessions are lost; they're moved to folder_id = NULL

### Session Organization (NEW)

#### `move_session_to_folder(session_id: str, folder_id: Optional[str])`
Move a session to a folder (or root if folder_id is None).
- **folder_id**: Target folder ID (None for root)

#### `set_session_pinned(session_id: str, pinned: bool)`
Pin or unpin a session.
- **pinned**: True to pin, False to unpin

#### `get_sessions_in_folder(folder_id: Optional[str]) -> List[Session]`
Get all sessions in a specific folder.
- **folder_id**: The folder ID (None for root/unorganized sessions)
- **Returns**: List of sessions ordered by: pinned DESC, position ASC, last_active DESC

### Session Settings (NEW)

#### `update_session_mode(session_id: str, mode: str)`
Update the chat mode for a session.
- **mode**: Chat mode (e.g., 'simple', 'advanced', 'workflow')

#### `update_session_knowledge_enabled(session_id: str, enabled: bool)`
Enable or disable knowledge system for a session.
- **enabled**: True to enable, False to disable

## Database Tables

### sessions
- `session_id` TEXT PRIMARY KEY
- `created_at` TIMESTAMP NOT NULL
- `last_active` TIMESTAMP NOT NULL
- `message_count` INTEGER DEFAULT 0
- `title` TEXT
- `tags` TEXT DEFAULT '[]'
- `folder_id` TEXT (NEW)
- `pinned` INTEGER DEFAULT 0 (NEW)
- `position` INTEGER DEFAULT 0 (NEW)
- `mode` TEXT DEFAULT 'simple' (NEW)
- `knowledge_enabled` INTEGER DEFAULT 1 (NEW)

### folders (NEW)
- `folder_id` TEXT PRIMARY KEY
- `name` TEXT NOT NULL
- `parent_folder_id` TEXT
- `position` INTEGER DEFAULT 0
- `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP

### messages
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `session_id` TEXT NOT NULL
- `role` TEXT NOT NULL
- `content` TEXT NOT NULL
- `workflow_id` TEXT
- `timestamp` TIMESTAMP NOT NULL

## Indexes

- `idx_messages_session` - ON messages(session_id, timestamp)
- `idx_messages_workflow` - ON messages(workflow_id)
- `idx_sessions_title` - ON sessions(title)
- `idx_sessions_folder` - ON sessions(folder_id) (NEW)
- `idx_sessions_pinned` - ON sessions(pinned, position) (NEW)

## Example Usage Patterns

### Creating a hierarchical folder structure
```python
manager = SessionManager()

# Create top-level folders
work_id = manager.create_folder("Work")
personal_id = manager.create_folder("Personal")

# Create subfolders
ai_id = manager.create_folder("AI Projects", parent_id=work_id)
dev_id = manager.create_folder("Development", parent_id=work_id)

# Browse hierarchy
all_folders = manager.get_folders()
for folder in all_folders:
    indent = "  " * (0 if not folder.parent_folder_id else 1)
    print(f"{indent}{folder.name}")
```

### Organizing sessions
```python
# Create and organize
session_id = manager.create_session(title="AI Research Notes")
manager.move_session_to_folder(session_id, ai_id)
manager.set_session_pinned(session_id, True)

# Configure
manager.update_session_mode(session_id, "advanced")
manager.update_session_knowledge_enabled(session_id, True)

# Browse by folder
ai_sessions = manager.get_sessions_in_folder(ai_id)
for session in ai_sessions:
    pin_marker = "📌" if session.pinned else "  "
    print(f"{pin_marker} {session.title} [{session.mode}]")
```

### Working with unorganized sessions
```python
# Get sessions not in any folder
root_sessions = manager.get_sessions_in_folder(None)

# Organize them
for session in root_sessions:
    if "work" in session.title.lower():
        manager.move_session_to_folder(session.session_id, work_id)
    elif "personal" in session.title.lower():
        manager.move_session_to_folder(session.session_id, personal_id)
```

### Cleaning up folders
```python
# Delete folder (sessions move to root)
manager.delete_folder(old_folder_id)

# Reorganize orphaned sessions
orphaned = manager.get_sessions_in_folder(None)
for session in orphaned:
    # Re-organize as needed
    pass
```
