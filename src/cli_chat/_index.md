# Conversational CLI Module

## Purpose
Interactive chat interface for Felix with session management, rich formatting, custom commands, tool integrations, and proper multi-agent architecture integration via `CLIWorkflowOrchestrator`.

## Key Files

### [chat.py](chat.py)
Main interactive chat interface with prompt_toolkit integration.
- **`FelixChat`**: Interactive chat loop with keyboard shortcuts and auto-completion
- **Prompt toolkit**: Enhanced input with history search (Ctrl+R), multi-line editing, auto-suggest
- **Command dispatch**: Routes messages to natural language handler or special command processor
- **Rich output**: Beautiful terminal formatting with markdown rendering
- **Session modes**: New session, continue last, resume specific session
- **Print mode**: Non-interactive for scripting (`-p` flag or stdin)
- **Key bindings**: Ctrl+L (clear), Ctrl+R (history search), Ctrl+D (exit), Ctrl+C (interrupt)

### [cli_workflow_orchestrator.py](cli_workflow_orchestrator.py)
Thin wrapper connecting CLI to Felix's multi-agent workflow system.
- **`CLIWorkflowOrchestrator`**: Bridges CLI sessions with `run_felix_workflow()`
- **Session-workflow linking**: Links messages to workflows via `workflow_id`
- **Conversation continuity**: Uses `parent_workflow_id` for context threading
- **Progress tracking**: Terminal-friendly progress updates during execution
- **Proper integration**: Ensures CLI uses CollaborativeContextBuilder, ConceptRegistry, knowledge recording, synthesis feedback
- **CRITICAL**: Must be used instead of direct `run_felix_workflow()` calls to maintain architectural correctness

### [session_manager.py](session_manager.py)
SQLite-based session persistence and conversation threading.
- **`SessionManager`**: Manages chat sessions in `felix_cli_sessions.db`
- **`Session`**: Session metadata (ID, title, tags, created_at, last_active, message_count)
- **`Message`**: Individual messages with role (user/assistant/system), content, workflow_id, timestamp
- **Operations**: Create, load, save, list, search, delete sessions
- **Export/import**: JSON format for backup and transfer
- **Conversation threading**: Links messages via workflow IDs for context continuity

### [formatters.py](formatters.py)
Rich terminal output formatting with graceful fallback.
- **`RichOutputFormatter`**: Beautiful markdown rendering with syntax highlighting (uses `rich` library if available)
- **`PlainOutputFormatter`**: Plain text fallback for environments without `rich`
- **`get_formatter()`**: Auto-detects capabilities and returns appropriate formatter
- **Markdown support**: Headers, code blocks, lists, tables, emphasis
- **Syntax highlighting**: Language-specific code formatting
- **Theme support**: Configurable color schemes

### [command_handler.py](command_handler.py)
Command routing and special command processing.
- **`CommandHandler`**: Routes commands to appropriate handlers
- **Built-in commands**: `/help`, `/session`, `/history`, `/clear`, `/export`, `/status`, `/knowledge`
- **Custom commands**: Loads user-defined commands from `.felix/commands/`
- **Special prefixes**: `!command` (shell), `@file` (file reference), `#note` (annotation)
- **Command completion**: Tab completion for command names and arguments

### [custom_commands.py](custom_commands.py)
Custom command loader and executor.
- **`CustomCommandLoader`**: Loads YAML commands from `.felix/commands/` directory
- **YAML frontmatter**: Commands defined with metadata (name, description, usage)
- **Template substitution**: Dynamic variable replacement in command content
- **Hot reload**: Watches for command file changes and reloads automatically
- **Namespace support**: Organize commands in subdirectories

### [completers.py](completers.py)
Tab completion for commands, file paths, and arguments.
- **`FelixCompleter`**: Main completer combining all completion sources
- **`CommandCompleter`**: Completes command names (e.g., `/help`, `/session`)
- **`FilePathCompleter`**: File and directory path completion
- **`ArgumentCompleter`**: Context-aware argument completion for commands
- **prompt_toolkit integration**: Seamless tab completion in interactive mode

### [tools/](tools/)
Tool integrations for accessing Felix subsystems from CLI (workflows, history, knowledge, agents, system info, documents).

### [prompts/](prompts/)
Chat-specific prompt templates (currently empty - uses system defaults from `src/prompts/`).

## Key Concepts

### Architecture Integration
**CRITICAL**: The CLI **must** use `CLIWorkflowOrchestrator` (not direct `run_felix_workflow()` calls) to properly integrate with Felix's multi-agent architecture:
- ✅ **CollaborativeContextBuilder**: Enriches context, applies token budgets, filters by relevance
- ✅ **ConceptRegistry**: Maintains terminology consistency (session-scoped, not workflow-scoped)
- ✅ **Knowledge Recording**: Records which knowledge is helpful for meta-learning
- ✅ **Synthesis Feedback**: Broadcasts feedback for agent self-improvement
- ✅ **Workflow-Session Continuity**: Links messages to workflows for conversation threading

### Session Management
Sessions persist across CLI invocations:
- **New session**: `felix chat` creates fresh session
- **Continue**: `felix chat -c` resumes last active session
- **Resume specific**: `felix chat --resume abc123` continues specific session
- **Session metadata**: Auto-generated titles, user tags, search by content

### Command Types
1. **Built-in commands**: `/help`, `/session`, `/history`, `/clear`, `/export`, `/status`
2. **Custom commands**: User-defined in `.felix/commands/` with YAML frontmatter
3. **Special prefixes**: `!ls` (shell), `@file.txt` (file ref), `#note text` (annotation)
4. **Natural language**: Default for messages without special syntax

### Rich Terminal Experience
- **Markdown rendering**: Headers, lists, code blocks, tables
- **Syntax highlighting**: Language-specific code formatting (Python, JS, SQL, etc.)
- **Auto-completion**: Tab completion for commands and file paths
- **History search**: Ctrl+R to search previous inputs
- **Keyboard shortcuts**: Ctrl+L clear, Ctrl+C interrupt, Ctrl+D exit
- **Graceful fallback**: Plain text mode if `rich` or `prompt_toolkit` unavailable

### Print Mode for Scripting
```bash
# Single query
felix chat -p "What is quantum computing?"

# Piped input
echo "Explain helical geometry" | felix chat

# Output to file
felix chat -p "Generate API docs" > docs.md
```

### Conversation Threading
Messages link to workflows which link to parent workflows:
```
Session abc123
├─ Message 1: "Explain quantum"
│  └─ Workflow wf_001 (no parent)
├─ Message 2: "Give examples"
│  └─ Workflow wf_002 (parent: wf_001)  # Has context from wf_001
└─ Message 3: "Compare approaches"
   └─ Workflow wf_003 (parent: wf_002)  # Has context from wf_001 and wf_002
```

### Tool System
CLI tools provide structured access to Felix subsystems:
- **WorkflowTool**: Execute and monitor workflows
- **HistoryTool**: Query workflow history
- **KnowledgeTool**: Search knowledge base
- **AgentTool**: Interact with agents
- **SystemTool**: Check system status
- **DocumentTool**: Ingest documents to knowledge brain

## Usage Examples

### Interactive Mode
```bash
# Start new session
felix chat

> Hello! I'm Felix. How can I help you today?

User: Explain quantum computing
Assistant: [Rich formatted response with markdown]

User: /session save "Quantum Discussion"
> Session saved with title: Quantum Discussion

User: /history
> [Shows conversation history]

User: /clear
> [Clears screen, continues session]
```

### Print Mode (Scripting)
```bash
# One-off query
felix chat -p "Generate Python code for binary search"

# Pipe input
cat requirements.txt | felix chat -p "Analyze dependencies"

# Redirect output
felix chat -p "Write API documentation" > api_docs.md
```

### Custom Commands
Create `.felix/commands/review-code.md`:
```markdown
---
name: review-code
description: Review code for quality and security
usage: /review-code <file>
---

Review the following code for:
- Security vulnerabilities
- Performance issues
- Code quality
- Best practices

File: {file}
```

Usage: `/review-code src/api/main.py`

## Related Modules
- [workflows/](../workflows/) - Workflow execution engine (via `run_felix_workflow()`)
- [agents/](../agents/) - Multi-agent system for task processing
- [knowledge/](../knowledge/) - Knowledge Brain for semantic search
- [memory/](../memory/) - Session and workflow history persistence
- [llm/](../llm/) - LLM provider integration
