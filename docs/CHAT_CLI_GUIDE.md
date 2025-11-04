# Felix Conversational CLI Guide

## Overview

The Felix Conversational CLI provides an interactive, ChatGPT-like interface for working with the Felix multi-agent system. Unlike the traditional `felix run` command that executes a single workflow and exits, the chat interface maintains session state, supports follow-up questions, and enables natural language interactions.

## Quick Start

### Basic Usage

```bash
# Start a new chat session
python -m src.cli chat

# Start with knowledge brain enabled
python -m src.cli chat --knowledge-brain

# Resume a previous session
python -m src.cli chat --resume abc123

# Disable natural language mode (explicit commands only)
python -m src.cli chat --no-nl
```

### Your First Chat Session

```bash
$ python -m src.cli chat
🚀 Felix Conversational CLI
============================================================
Initializing Felix system...
✓ Felix system initialized

════════════════════════════════════════════════════════════
  Felix Conversational CLI
════════════════════════════════════════════════════════════

Session: abc12345

Type your question or use commands:
  /help - Show available commands
  /exit or /quit - Exit chat

felix> What does the system know about machine learning?
[Knowledge search results shown...]

felix> Run an analysis of recent AI trends
[Felix spawns agents and executes workflow...]

felix> Now focus specifically on transformer architectures
[Felix continues with context from previous workflow...]

felix> /exit
Goodbye!
```

## Features

### 1. Session Management

Every chat creates a persistent session that stores:
- All messages (user and assistant)
- Workflow IDs and results
- Session metadata (created date, last active, message count)

**Managing Sessions:**

```bash
# List all sessions
python -m src.cli sessions list

# Show session details
python -m src.cli sessions show abc12345

# Delete a session
python -m src.cli sessions delete abc12345
python -m src.cli sessions delete abc12345 --force  # Skip confirmation
```

### 2. Natural Language Processing

When natural language mode is enabled (default), you can ask questions naturally:

```
felix> What does the system know about quantum computing?
felix> Search for information about neural networks
felix> Show me recent workflows
felix> What's the system status?
```

The system will automatically detect your intent and route to the appropriate tool.

### 3. Explicit Commands

You can also use explicit slash commands for precise control:

#### Workflow Commands

```bash
/workflow run <task>              # Execute a workflow
/workflow show <workflow_id>      # Show workflow details
```

**Examples:**
```
felix> /workflow run Explain quantum computing
felix> /workflow run Design a REST API --max-steps 15
felix> /workflow show 123
```

#### History Commands

```bash
/history list [--limit N]         # List recent workflows
/history search <query>            # Search workflows by task
/history show <workflow_id>        # Show workflow details
/history thread <workflow_id>      # Show conversation thread
```

**Examples:**
```
felix> /history list --limit 10
felix> /history search "machine learning"
felix> /history show 123
```

#### Knowledge Commands

```bash
/knowledge search <query>          # Search knowledge base
/knowledge concepts [domain]       # List concepts
/knowledge domains                 # List all domains
/knowledge graph <concept>         # Show concept relationships
```

**Examples:**
```
felix> /knowledge search quantum computing
felix> /knowledge concepts ai
felix> /knowledge domains
felix> /knowledge graph "neural network" --limit 5
```

#### Agent Commands

```bash
/agent list <workflow_id>          # List agents in workflow
/agent show <workflow_id> <agent>  # Show agent details
/agent contributions <workflow_id> # Show agent contributions
```

**Examples:**
```
felix> /agent list 123
felix> /agent show 123 research
felix> /agent contributions 123
```

#### System Commands

```bash
/system status                     # Overall system status
/system providers                  # LLM provider status
/system databases                  # Database status
/system knowledge                  # Knowledge statistics
/system config                     # System configuration
```

**Examples:**
```
felix> /system status
felix> /system providers
felix> /system knowledge
```

#### Document Commands

```bash
/document ingest <file_path>       # Ingest a document
/document list                     # List all documents
/document show <doc_id>            # Show document details
/document delete <doc_id>          # Delete a document
```

**Examples:**
```
felix> /document ingest ./paper.pdf
felix> /document list
felix> /document show doc_123
```

### 4. Iterative Refinement

One of the most powerful features is the ability to refine and build on previous results:

```
felix> Design a REST API for a todo application
[Felix generates initial design...]

felix> Now add authentication using JWT
[Felix continues, building on previous workflow...]

felix> Add rate limiting and explain the security considerations
[Felix further refines with full context...]
```

Each workflow is linked to its parent, creating a conversation thread that maintains context.

## Command Line Options

### Chat Command Options

```bash
python -m src.cli chat [options]

Options:
  --resume RESUME          Resume an existing session by ID
  --no-nl                  Disable natural language mode
  --knowledge-brain        Enable knowledge brain
  --web-search             Enable web search
  --verbose, -v            Verbose output with stack traces
```

### Sessions Command Options

```bash
python -m src.cli sessions <action> [session_id] [options]

Actions:
  list                     List all sessions
  show <session_id>        Show session details
  delete <session_id>      Delete a session

Options:
  --limit N                Limit number of items (default: 20)
  --force, -f              Force delete without confirmation
```

## Architecture

### Components

The conversational CLI consists of several modular components:

1. **SessionManager** (`src/cli_chat/session_manager.py`)
   - Manages session persistence with SQLite
   - Stores messages, metadata, and workflow links
   - Supports session resume and history

2. **CommandHandler** (`src/cli_chat/command_handler.py`)
   - Parses explicit commands and natural language
   - Routes to appropriate tools
   - Falls back to keyword matching when LLM unavailable

3. **Tool System** (`src/cli_chat/tools/`)
   - Modular tool architecture
   - Each tool handles a specific domain (workflows, knowledge, agents, etc.)
   - Tools are registered in a ToolRegistry

4. **OutputFormatter** (`src/cli_chat/formatters.py`)
   - Terminal formatting with ANSI colors
   - Markdown rendering
   - Structured output for results, tables, and progress

5. **Main Chat Loop** (`src/cli_chat.py`)
   - Interactive REPL
   - Message history integration
   - Session state management

### Tool Architecture

Each tool inherits from `BaseTool` and implements:

```python
class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "mytool"

    @property
    def description(self) -> str:
        return "Description of what the tool does"

    @property
    def usage(self) -> str:
        return "/mytool [command] [args]"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        # Tool implementation
        pass
```

Tools have access to Felix context (felix_system, knowledge_store, llm_adapter, etc.) via `self.felix_context`.

## Database Schema

The conversational CLI uses its own database (`felix_cli_sessions.db`) with the following schema:

### sessions table

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    message_count INTEGER DEFAULT 0
);
```

### messages table

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    workflow_id TEXT,              -- Links to workflows
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

## Integration with Existing Felix Features

### Workflow History

Chat sessions integrate with Felix's workflow history system:
- Each workflow executed in chat is saved to `felix_workflow_history.db`
- Workflows are linked using `parent_workflow_id` for conversation threading
- You can view workflow details from chat or via the GUI

### Knowledge Brain

When enabled with `--knowledge-brain`:
- `/knowledge` commands access the full knowledge base
- `/document` commands ingest documents into the brain
- Natural language queries can trigger semantic search
- Workflows can automatically augment with relevant knowledge

### Agent Introspection

The `/agent` commands provide visibility into Felix's multi-agent system:
- See which agents were spawned for a workflow
- View agent confidence scores and positions on the helix
- Inspect individual agent contributions to synthesis
- Understand why agents were spawned (via metadata)

## Comparison: `felix run` vs `felix chat`

### `felix run` (Traditional CLI)

**Use cases:**
- One-off workflow execution
- CI/CD integration
- Scripting and automation
- Batch processing

**Characteristics:**
- Fire-and-forget: start system, run workflow, stop system
- No session state
- No follow-up questions
- Output to file or stdout
- Fast for single tasks

**Example:**
```bash
felix run "Explain quantum computing" --output result.md
```

### `felix chat` (Conversational CLI)

**Use cases:**
- Exploratory workflows
- Iterative refinement
- Knowledge base exploration
- Agent debugging
- Interactive development

**Characteristics:**
- Persistent session state
- Follow-up questions and refinement
- Natural language support
- Multi-turn conversations
- Context carryover between workflows

**Example:**
```
felix> Explain quantum computing
felix> Now compare it to classical computing
felix> Show me what the system knows about quantum entanglement
```

## Tips and Best Practices

### 1. Use Sessions for Related Work

Create a new session for each major topic or project:

```bash
# Session for API design work
felix chat  # Session 1: Design todo API, refine endpoints, add auth

# Session for research
felix chat  # Session 2: Research ML trends, explore transformers, analyze papers
```

### 2. Leverage Natural Language

Natural language mode is smart about detecting intent:

```
felix> What workflows did I run yesterday?
# Automatically routes to /history

felix> Search for information about neural networks
# Automatically routes to /knowledge search

felix> Check if LM Studio is connected
# Automatically routes to /system providers
```

### 3. Combine Commands with Natural Language

You can mix and match:

```
felix> /workflow run Design a GraphQL API
[Result shown...]

felix> Now add subscriptions for real-time updates
[Natural language continues the workflow...]

felix> /history show 123
[Explicit command shows the workflow details...]
```

### 4. Use Verbose Mode for Debugging

```bash
felix chat --verbose
```

This shows:
- Detailed error messages with stack traces
- Intent detection decisions
- LLM provider fallback chains
- Tool execution details

### 5. Resume Sessions for Long-Running Projects

```bash
# End of day 1
felix> /exit

# Start of day 2
felix chat --resume abc12345

# Session context is preserved!
felix> Continue where we left off
```

## Troubleshooting

### Chat won't start

**Issue:** "Failed to start Felix system"

**Solution:**
- Check that at least one LLM provider is configured in `config/llm.yaml`
- Test connection: `python -m src.cli test-connection`
- Check if LM Studio is running (if using local LLM)

### Natural language not working

**Issue:** Commands are interpreted incorrectly

**Solutions:**
1. Use explicit commands: `/workflow run <task>` instead of natural language
2. Disable NL mode: `felix chat --no-nl`
3. Check LLM connection: `/system providers`
4. Use more specific keywords that match tool descriptions

### Session not found

**Issue:** "Session not found: abc123"

**Solution:**
- List sessions: `python -m src.cli sessions list`
- The session may have been deleted
- Check the session ID (case-sensitive)

### Workflow not continuing context

**Issue:** Follow-up questions start fresh workflows

**Explanation:**
- Context is maintained via `parent_workflow_id`
- Natural language queries that don't mention "continue" may start fresh
- Use explicit language: "continue the previous analysis" or "build on that"

## Future Enhancements

Potential features for future versions:

1. **Streaming Output**: Real-time token streaming during workflow execution
2. **Multi-modal Input**: Image upload and analysis
3. **Workflow Templates**: Save and reuse common workflow patterns
4. **Collaboration**: Share sessions between users
5. **Export**: Export entire session as markdown report
6. **Search**: Full-text search across all session messages
7. **Favorites**: Bookmark workflows for quick access
8. **Aliases**: Create custom command shortcuts

## Conclusion

The Felix Conversational CLI bridges the gap between traditional command-line tools and modern conversational interfaces. It provides:

- ✅ Interactive, multi-turn conversations
- ✅ Session persistence and resume
- ✅ Natural language understanding
- ✅ Full access to Felix's multi-agent system
- ✅ Knowledge base exploration
- ✅ Agent introspection and debugging

For automation and scripting, continue using `felix run`. For exploration, development, and interactive work, use `felix chat`.

---

**Getting Started:**

```bash
# Start your first session
python -m src.cli chat

# Try it out
felix> /help
felix> What can you do?
felix> /workflow run Explain the helix geometry in Felix
felix> /exit
```

Welcome to conversational AI development with Felix!
