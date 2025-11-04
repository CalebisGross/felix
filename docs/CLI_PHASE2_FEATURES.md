# Felix CLI Phase 2 Advanced Features

## Overview

Phase 2 enhances Felix CLI with advanced session management, beautiful terminal formatting, custom commands, and productivity features inspired by modern CLI tools. These features build upon Phase 1's foundation to create a professional, fully-featured command-line experience.

---

## Table of Contents

1. [Session Management](#1-session-management)
2. [Rich Terminal Output](#2-rich-terminal-output)
3. [Custom Commands](#3-custom-commands)
4. [Command Auto-completion](#4-command-auto-completion)
5. [Installation](#installation)
6. [Complete Examples](#complete-examples)
7. [Troubleshooting](#troubleshooting)

---

## 1. Session Management

### 1.1 Session Metadata

Sessions now support titles and tags for better organization:

```bash
# Rename a session
felix sessions rename abc123 --title "API Design Project"

# Add tags to categorize sessions
felix sessions tag abc123 --tags work api-design urgent

# Remove tags
felix sessions untag abc123 --tags urgent
```

**Auto-title Generation:**

When you start a new session, the first message you send can automatically become the session title:

```python
# In code (automatically triggered in chat.py)
session_manager.generate_auto_title(session_id, max_length=50)
```

### 1.2 Advanced Session Queries

**List Recent Sessions:**
```bash
felix sessions recent                  # Last 20 sessions
felix sessions recent --limit 10       # Last 10 sessions
```

Output shows session ID, title, tags, and last active time:
```
abc123: 2025-11-04 10:30 - API Design Project [work, api-design]
def456: 2025-11-04 09:15 - Debug Authentication [work, debugging]
```

**Today's Sessions:**
```bash
felix sessions today                   # All sessions from today
```

**Search Sessions:**
```bash
# Search by keyword in title or messages
felix sessions search "authentication"

# Results show matching sessions
Sessions matching 'authentication':
==========================================
def456: 2025-11-04 09:15 - Debug Authentication [work]
ghi789: 2025-11-03 14:22 - OAuth Implementation [work, auth]
```

### 1.3 Session Import/Export

**Export Session:**
```bash
# Export to JSON file
felix sessions export abc123 -o my-session.json
felix sessions export abc123              # Default: session_abc123.json
```

**Exported JSON structure:**
```json
{
  "session": {
    "session_id": "abc123",
    "title": "API Design Project",
    "tags": ["work", "api-design"],
    "message_count": 15,
    "created_at": "2025-11-04T10:00:00",
    "last_active": "2025-11-04T10:30:00"
  },
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "Design a REST API for user management",
      "workflow_id": "wf_001",
      "timestamp": "2025-11-04T10:00:00"
    },
    ...
  ],
  "exported_at": "2025-11-04T11:00:00"
}
```

**Import Session:**
```bash
# Import from JSON file
felix sessions import my-session.json

# Output:
✓ Session imported with ID: jkl012
```

**Use Cases:**
- **Backup**: Export important sessions before deletion
- **Sharing**: Share conversation history with team members
- **Migration**: Move sessions between machines
- **Archival**: Store sessions for future reference

### 1.4 Complete Session Commands

| Command | Description | Example |
|---------|-------------|---------|
| `list` | List all sessions | `felix sessions list` |
| `recent` | List recent sessions | `felix sessions recent --limit 10` |
| `today` | List today's sessions | `felix sessions today` |
| `show` | Show session details | `felix sessions show abc123` |
| `search` | Search by keyword | `felix sessions search "keyword"` |
| `rename` | Set session title | `felix sessions rename abc123 --title "New Title"` |
| `tag` | Add tags | `felix sessions tag abc123 --tags work urgent` |
| `untag` | Remove tags | `felix sessions untag abc123 --tags urgent` |
| `export` | Export to JSON | `felix sessions export abc123 -o file.json` |
| `import` | Import from JSON | `felix sessions import file.json` |
| `delete` | Delete session | `felix sessions delete abc123` |

---

## 2. Rich Terminal Output

### 2.1 Enhanced Formatting with Rich

When `rich` library is installed, Felix CLI displays output with:
- Beautiful markdown rendering
- Syntax-highlighted code blocks
- Formatted tables
- Progress indicators
- Styled panels

**Installation:**
```bash
pip install -r requirements-cli-enhanced.txt
```

**Automatic Detection:**

Felix automatically detects if `rich` is available and uses it. If not, it falls back to basic formatting. No configuration needed!

### 2.2 Markdown Rendering

**Before (basic mode):**
```
# Heading 1
## Heading 2
**bold text**
```

**After (rich mode):**

<img src="markdown-rendering-example.png" alt="Rich markdown rendering with styled headers, bold text, and code blocks" />

### 2.3 Beautiful Tables

**Session List (rich mode):**

```
┌──────────┬────────────────────┬──────────────┬──────────┬─────────────────┐
│ Session  │ Title              │ Tags         │ Messages │ Last Active     │
│ ID       │                    │              │          │                 │
├──────────┼────────────────────┼──────────────┼──────────┼─────────────────┤
│ abc123   │ API Design Project │ work, api    │ 15       │ 2025-11-04      │
│          │                    │              │          │ 10:30           │
│ def456   │ Debug Auth         │ work, debug  │ 8        │ 2025-11-04      │
│          │                    │              │          │ 09:15           │
└──────────┴────────────────────┴──────────────┴──────────┴─────────────────┘
```

### 2.4 Progress Indicators

**Workflow Execution:**

When running workflows with rich formatting:
```
⋯ Executing workflow: Design a REST API for user management...
⠋ Processing...
```

The spinner animates while Felix works, providing visual feedback.

### 2.5 Styled Panels

**Workflow Results:**

Results are displayed in styled panels:
```
╭─ Workflow Result ─────────────────────────────────────────╮
│                                                            │
│  [Generated API design with endpoints, authentication,    │
│   and data models...]                                      │
│                                                            │
╰────────────────────────────────────────────────────────────╯

Confidence: 0.85
Agents:     4
Time:       12.3s
```

### 2.6 Code Syntax Highlighting

**Code Blocks (rich mode):**

```python
# Automatically syntax-highlighted
def hello_world():
    print("Hello, Felix!")
```

Displays with:
- Line numbers
- Syntax colors
- Proper indentation
- Monokai theme

---

## 3. Custom Commands

### 3.1 Overview

Create your own slash commands that execute predefined templates with argument substitution.

**Command Directory:**
- `.felix/commands/` (project-specific)
- `~/.felix/commands/` (user-wide)

### 3.2 Simple Custom Command

**File: `.felix/commands/hello.md`**

```markdown
Hello, {arg0}! How can I help you with {arg1} today?
```

**Usage:**
```bash
felix> /hello Alice programming
# Executes: "Hello, Alice! How can I help you with programming today?"
```

### 3.3 Commands with Frontmatter

**File: `.felix/commands/review-code.md`**

```markdown
---
description: "Review code for quality and best practices"
usage: "/review-code <file_path>"
args: ["file_path"]
aliases: ["review", "code-review"]
---

Please review the code in {arg0} for:
1. Code quality and readability
2. Performance optimizations
3. Security vulnerabilities
4. Best practices
5. Documentation completeness

Provide specific recommendations for improvement.
```

**Usage:**
```bash
felix> /review-code src/api/main.py
felix> /review src/api/main.py              # Using alias
felix> /code-review src/api/main.py         # Using alias
```

### 3.4 Argument Substitution

**Placeholders:**
- `{arg0}`, `{arg1}`, ... - Positional arguments (0-indexed)
- `{args}` - All positional arguments joined with spaces
- `{key_name}` - Named keyword arguments

**File: `.felix/commands/compare.md`**

```markdown
---
description: "Compare two items"
usage: "/compare <item1> <item2>"
args: ["item1", "item2"]
---

Compare {arg0} and {arg1} in terms of:
- Features
- Performance
- Cost
- Pros and Cons

Provide a detailed comparison table.
```

**Usage:**
```bash
felix> /compare Python JavaScript
# Executes: "Compare Python and JavaScript in terms of: ..."
```

### 3.5 Complex Example

**File: `.felix/commands/design-api.md`**

```markdown
---
description: "Design a REST API with specific requirements"
usage: "/design-api <resource> [--auth=<type>] [--db=<database>]"
args: ["resource"]
---

Design a RESTful API for managing {arg0} with the following requirements:

1. **Authentication:** {auth}
2. **Database:** {db}
3. **Resource:** {arg0}

Include:
- Endpoint definitions (GET, POST, PUT, DELETE)
- Request/response schemas
- Authentication flow
- Error handling
- Rate limiting strategy
- Database schema
- API documentation

Provide production-ready code examples.
```

**Usage:**
```bash
felix> /design-api users auth=JWT db=PostgreSQL
```

### 3.6 Creating Commands Programmatically

```python
from src.cli_chat import CustomCommandLoader

# Initialize loader
loader = CustomCommandLoader()

# Create a new command
file_path = loader.create_command_file(
    name="debug-error",
    template="Analyze this error and suggest fixes:\n\n{args}",
    description="Debug and explain error messages",
    usage="/debug-error <error_message>",
    args=["error_message"]
)

print(f"Created command at: {file_path}")

# Reload to pick up new command
loader.reload()
```

### 3.7 Command Discovery

Felix automatically discovers custom commands on startup. To reload without restarting:

```bash
# In Python
chat.command_handler.custom_commands.reload()
```

---

## 4. Command Auto-completion

### 4.1 Overview

Press `Tab` to auto-complete:
- Built-in commands (`/workflow`, `/history`, `/knowledge`, etc.)
- Custom commands from `.felix/commands/`
- File paths (for `@file` prefix)
- Command arguments

**Requires:** `prompt_toolkit` installed (included in `requirements-cli-enhanced.txt`)

### 4.2 Command Completion

```bash
felix> /wor<Tab>
# Completes to: /workflow

felix> /workflow<Tab>
# Shows: run, continue, show

felix> /rev<Tab>
# Completes to: /review-code (custom command)
```

### 4.3 File Path Completion

```bash
felix> @src/<Tab>
# Shows: api/, cli/, gui/, workflows/, ...

felix> @src/api/<Tab>
# Shows: main.py, routers/, websockets/, ...

felix> @README<Tab>
# Completes to: @README.md
```

### 4.4 Argument Completion

```bash
felix> /workflow run "my task" --<Tab>
# Shows: --max-steps, --web-search, --verbose
```

### 4.5 Completion Display

When you press `Tab`, you see:
- **Command name**: The completion text
- **Description**: What the command does (shown to the right)

```
felix> /wor<Tab>

/workflow    Execute Felix workflows with multi-agent collaboration
```

---

## Installation

### Basic Installation

```bash
# Felix works without enhanced dependencies
python -m src.cli chat
```

### Enhanced Installation (Recommended)

```bash
# Install all Phase 2 enhancements
pip install -r requirements-cli-enhanced.txt

# Includes:
# - prompt_toolkit>=3.0.43  (keyboard shortcuts, history, auto-completion)
# - rich>=13.7.0            (beautiful formatting, tables, markdown)
# - orjson>=3.9.10          (fast JSON for session export/import)
# - pygments>=2.17.2        (syntax highlighting)
```

### Verify Installation

```bash
# Check if enhanced features are available
python -c "import prompt_toolkit, rich; print('✓ Enhanced mode available')"
```

If the above command succeeds, you'll have:
- ✓ Command history and auto-suggestions
- ✓ Keyboard shortcuts (Ctrl+R, Ctrl+L, etc.)
- ✓ Tab auto-completion
- ✓ Rich formatting and markdown
- ✓ Progress indicators

---

## Complete Examples

### Example 1: Project Workflow with Session Management

```bash
# Start a new session with web search
felix chat --web-search

# First message auto-generates title
felix> Design a scalable microservices architecture

# Work on the project...
felix> /workflow run Explain service discovery patterns

# Exit and tag the session
felix> /exit

# Later, find and continue the session
felix sessions search "microservices"
# Output: abc123: 2025-11-04 10:30 - Design a scalable microservices...

felix chat -c  # or: felix chat --resume abc123

# Add tags for organization
felix sessions tag abc123 --tags architecture microservices work

# Export for documentation
felix sessions export abc123 -o architecture-discussion.json
```

### Example 2: Custom Command Workflow

**Step 1: Create custom command**

Create `.felix/commands/test-review.md`:
```markdown
---
description: "Review test coverage and suggest improvements"
usage: "/test-review <test_dir>"
args: ["test_dir"]
aliases: ["tests", "coverage"]
---

Analyze the test files in {arg0}:
1. Calculate test coverage gaps
2. Identify untested edge cases
3. Suggest additional test cases
4. Review test quality and maintainability

Focus on critical paths and error handling.
```

**Step 2: Use the command**

```bash
felix chat

# Tab completion shows your custom command
felix> /test<Tab>
# Shows: /test-review, /tests (alias)

# Use the command
felix> /test-review tests/unit/

# Felix executes the template as a workflow
```

### Example 3: Session Organization

```bash
# Create sessions for different projects
felix chat
felix> Design user authentication API
felix> /exit

felix chat
felix> Debug performance issues in database queries
felix> /exit

felix chat
felix> Research machine learning deployment strategies
felix> /exit

# Organize with tags
felix sessions today                          # See today's sessions
felix sessions tag <id1> --tags work api auth
felix sessions tag <id2> --tags work database performance
felix sessions tag <id3> --tags research ml deployment

# Later, find sessions by tag or keyword
felix sessions search "authentication"
felix sessions search "database"
felix sessions search "machine learning"

# View sessions with rich formatting (if rich installed)
felix sessions list
# Beautiful table with IDs, titles, tags, and timestamps
```

### Example 4: Advanced Custom Commands

**Multi-step analysis command:**

`.felix/commands/analyze-architecture.md`:
```markdown
---
description: "Comprehensive architecture analysis"
usage: "/analyze-architecture <system>"
args: ["system"]
---

Perform a comprehensive analysis of {arg0}:

## 1. System Components
Identify all major components and their responsibilities.

## 2. Data Flow
Map how data flows through the system.

## 3. Scalability Analysis
- Current bottlenecks
- Scaling strategies
- Load distribution

## 4. Security Review
- Authentication/authorization
- Data encryption
- Vulnerability assessment

## 5. Recommendations
Provide actionable recommendations for:
- Performance improvements
- Security hardening
- Maintainability
- Cost optimization

Include diagrams where helpful.
```

Usage:
```bash
felix> /analyze-architecture "e-commerce platform"
# Generates comprehensive analysis
```

---

## Troubleshooting

### Issue: Custom commands not loading

**Symptom:** `/my-command` shows "Unknown command"

**Solutions:**
1. Check command file location:
   ```bash
   ls .felix/commands/
   ls ~/.felix/commands/
   ```

2. Verify file extension:
   ```bash
   # Should be .md or .txt
   mv my-command my-command.md
   ```

3. Reload commands:
   ```python
   # In Python
   chat.command_handler.custom_commands.reload()
   ```

4. Check frontmatter syntax:
   ```markdown
   ---
   description: "Valid YAML"
   ---

   Command template here
   ```

### Issue: Rich formatting not working

**Symptom:** Output looks plain without colors/tables

**Solutions:**
1. Install rich:
   ```bash
   pip install rich>=13.7.0
   ```

2. Check terminal support:
   ```bash
   # Test rich in terminal
   python -c "from rich.console import Console; Console().print('[bold red]Test[/bold red]')"
   ```

3. Verify installation:
   ```python
   python -c "import rich; print(rich.__version__)"
   ```

### Issue: Auto-completion not working

**Symptom:** Tab key doesn't show completions

**Solutions:**
1. Install prompt_toolkit:
   ```bash
   pip install prompt_toolkit>=3.0.43
   ```

2. Check terminal compatibility:
   ```bash
   echo $TERM  # Should be xterm-256color or similar
   ```

3. Try in different terminal (some terminals don't support features)

4. Verify prompt_toolkit:
   ```python
   python -c "import prompt_toolkit; print('OK')"
   ```

### Issue: Session export/import fails

**Symptom:** "Failed to export/import session"

**Solutions:**
1. Check write permissions:
   ```bash
   touch session-test.json  # Should succeed
   ```

2. Verify JSON syntax (for import):
   ```bash
   python -c "import json; json.load(open('session.json'))"
   ```

3. Check session exists:
   ```bash
   felix sessions list | grep abc123
   ```

### Issue: Session search returns no results

**Symptom:** Search finds nothing despite relevant sessions existing

**Solutions:**
1. Try broader search terms
2. Check session has messages:
   ```bash
   felix sessions show abc123
   ```
3. Search is case-insensitive, try variations

---

## Performance Notes

### Session Operations
- **List/Search**: O(N) where N is number of sessions
- **Export**: O(M) where M is number of messages
- **Import**: O(M) for message insertion

### Rich Formatting
- Minimal overhead (~5-10ms per render)
- Automatically disabled if terminal doesn't support it
- Falls back to basic formatting

### Custom Commands
- Commands loaded once on startup
- Reload with `.reload()` method
- No performance impact on built-in commands

### Auto-completion
- Completions generated on-demand
- File path completion limited to current directory
- Fast even with 100+ custom commands

---

## Summary

Phase 2 brings professional-grade features to Felix CLI:

✅ **Session Management**
- Titles and tags for organization
- Search by keyword
- Import/export for backup and sharing
- Advanced filtering (recent, today, by tags)

✅ **Rich Terminal Output**
- Beautiful markdown rendering
- Syntax-highlighted code
- Formatted tables
- Progress indicators
- Graceful fallback to basic mode

✅ **Custom Commands**
- User-defined slash commands
- YAML frontmatter for metadata
- Argument substitution
- Command aliases
- Auto-discovery from .felix/commands/

✅ **Auto-completion**
- Tab completion for commands
- File path completion
- Argument completion
- Command descriptions in completion menu

✅ **Graceful Degradation**
- Works without enhanced dependencies
- Automatic feature detection
- Fallback to basic mode

These features make Felix CLI a powerful, user-friendly tool for interactive AI workflows while maintaining backward compatibility with Phase 1.

---

## What's Next?

Potential future enhancements:
- **Checkpoint/Rewind**: Undo workflows with Esc+Esc
- **Session Folders**: Organize sessions into projects
- **Command Marketplace**: Share custom commands
- **Streaming Output**: Real-time token display
- **Vim Mode**: Optional vim key bindings
- **Session Analytics**: Track usage patterns

Enjoy the enhanced Felix CLI! 🚀
