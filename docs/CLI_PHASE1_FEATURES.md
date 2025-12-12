# Felix CLI Phase 1 Enhanced Features

## Overview

Phase 1 of the Felix CLI enhancement brings several Claude Code-inspired features that dramatically improve the user experience. These features make Felix's CLI more powerful, composable, and user-friendly while maintaining backward compatibility.

---

## 1. Piped Input & Print Mode

### Description
Execute single queries without entering interactive mode. Perfect for scripting, automation, and command composition.

### Usage

**Print Mode with -p flag:**
```bash
# Single query execution
felix chat -p "What is helical geometry in Felix?"

# Output goes to stdout (can be piped)
felix chat -p "Explain quantum computing" | less
felix chat -p "Generate Python code" > output.py
```

**Piped Input (automatic print mode):**
```bash
# Pipe content to Felix
echo "Analyze this text" | felix chat

# Read file and process
cat document.txt | felix chat

# Chain with other commands
grep "TODO" src/*.py | felix chat -p "Summarize these todos"
```

**Command Composition:**
```bash
# Multi-stage processing
felix chat -p "List top 10 ML frameworks" | \
    felix chat -p "Pick the best for beginners and explain why"

# Use in scripts
RESULT=$(felix chat -p "Calculate fibonacci(10)")
echo "Result: $RESULT"
```

### Benefits
- **Scriptable**: Integrate Felix into bash scripts and CI/CD pipelines
- **Composable**: Chain multiple Felix queries
- **Fast**: No REPL overhead for one-off queries
- **Unix-friendly**: Works with standard input/output streams

---

## 2. Session Continuation

### Description
Quickly resume your last active session without remembering session IDs.

### Usage

```bash
# Continue last session
felix chat -c
felix chat --continue

# You'll see:
# "Continuing last session: abc12345"
```

### Comparison

**Before:**
```bash
# Had to manually get session ID
felix sessions list
# Find ID, copy it...
felix chat --resume abc12345
```

**After:**
```bash
# One command
felix chat -c
```

### Benefits
- **Convenience**: No need to remember or lookup session IDs
- **Workflow continuity**: Easily pick up where you left off
- **Muscle memory**: Simple `-c` flag, just like many other tools

---

## 3. Enhanced Input with prompt_toolkit

### Description
Rich terminal input handling with command history, auto-suggestions, and keyboard shortcuts.

### Features

**Command History:**
- Persistent history stored in `~/.felix_history`
- Navigate with ↑/↓ arrow keys
- Reverse search with Ctrl+R
- Auto-suggestions from history as you type

**Keyboard Shortcuts:**
- `Ctrl+R`: Reverse search command history
- `Ctrl+L`: Clear screen (preserves history)
- `Ctrl+D`: Exit chat gracefully
- `Ctrl+C`: Cancel current input (doesn't exit)
- `Esc+Esc`: Rewind/undo (placeholder for future checkpoint system)
- `↑/↓`: Navigate history
- `Home/End`: Jump to start/end of line
- `Ctrl+A/E`: Alternative home/end (Emacs-style)

**Multiline Support:**
- Prepared for multiline input (future enhancement)
- Will support `Shift+Enter` for multiline mode

### Installation

```bash
# Enhanced mode (recommended)
pip install -r requirements-cli-enhanced.txt

# Or install manually
pip install prompt_toolkit rich orjson pygments
```

### Graceful Degradation

If `prompt_toolkit` is not installed, Felix falls back to basic `input()` mode:
- Still works, just without enhanced features
- Chat will show a message explaining how to upgrade:
  ```
  Basic input mode. Install prompt_toolkit for enhanced features:
    pip install -r requirements-cli-enhanced.txt
  ```

### Benefits
- **Productivity**: Find and reuse previous commands quickly
- **Discoverability**: Auto-suggestions help you remember what you did
- **Familiar**: Standard keyboard shortcuts that work like bash/zsh
- **Optional**: Works without enhanced libraries, just with reduced UX

---

## 4. Special Input Prefixes

### Description
Quick shortcuts for common operations: execute commands, read files, save notes.

### 4.1 Bash Command Execution (`!command`)

Execute bash commands directly from Felix and capture their output.

**Usage:**
```
felix> !ls -la
Executing: ls -la
total 248
drwxrwxr-x 12 user user  4096 Nov  4 10:30 .
drwxrwxr-x  5 user user  4096 Oct 30 21:00 ..
...

Command executed. You can now ask about the output.
```

**How It Works:**
1. Type `!command`
2. Command executes in shell
3. Output is captured and displayed
4. Felix converts it to a workflow: "Analyze the command output above"
5. You can ask follow-up questions about the output

**Examples:**
```
felix> !git status
# Shows git status, then you can ask:
# "What files should I commit?"

felix> !pytest tests/
# Runs tests, then:
# "Why did test_auth fail?"

felix> !df -h
# Check disk space
# "Which directory is using the most space?"
```

**Safety:**
- 30-second timeout
- Captures both stdout and stderr
- Shows exit code
- Runs in current working directory

### 4.2 File Reading (`@filepath`)

Read a file and include its content in the conversation context.

**Usage:**
```
felix> @src/main.py
Read file: src/main.py (1234 chars)
File: src/main.py
```
[File content preview shown...]
```

# Felix then helps you analyze the file
```

**How It Works:**
1. Type `@filepath`
2. File is read
3. Content preview is shown
4. Felix converts to workflow: "Analyze the file content from {filepath}"
5. You can ask questions about the code

**Examples:**
```
felix> @README.md
# "Improve this README"

felix> @config/settings.yaml
# "What's wrong with this configuration?"

felix> @logs/error.log
# "What caused these errors?"
```

**Features:**
- Handles any text file
- Shows file size
- Previews first 500 chars
- Full content available to Felix

### 4.3 Quick Notes (`#note`)

Save quick memory notes for future reference (coming soon).

**Usage:**
```
felix> #remember to add authentication to the API
Memory note feature coming soon
Note: remember to add authentication to the API
```

**Future Plans:**
- Save to project `FELIX.md` or `~/.felix/memory.md`
- Searchable notes
- Associate notes with sessions
- Export notes to markdown

### Benefits
- **Fast**: No need for separate terminal tabs
- **Contextual**: Output stays in conversation
- **Integrated**: Results flow directly to Felix
- **Discoverable**: Easy to remember (`!` for commands, `@` for files)

---

## 5. Updated CLI Interface

### New Command Structure

```bash
# Interactive modes
felix chat                    # New session
felix chat -c                 # Continue last session
felix chat --resume ID        # Resume specific session

# Print mode (non-interactive)
felix chat -p "query"         # Single query
echo "input" | felix chat     # Piped input

# Options work in both modes
felix chat -c --knowledge-brain --web-search
felix chat -p "query" --verbose
```

### Enhanced Help

```bash
felix --help
# Shows comprehensive examples including:
# - One-off workflows
# - Interactive chat modes
# - Print mode usage
# - Piped input examples
# - Session management

felix chat --help
# Shows all chat-specific options:
# - query argument for print mode
# - -c/--continue for last session
# - -p/--print for non-interactive
# - All other flags
```

---

## Complete Example Workflow

Here's a real-world example using multiple Phase 1 features:

```bash
# 1. Check system status with print mode
felix chat -p "Check if LLM providers are connected"

# 2. Read a file and analyze it
felix chat -c  # Start/continue session
felix> @src/api/main.py
felix> What security issues do you see?

# 3. Run tests and analyze failures
felix> !pytest tests/ -v
felix> Which test failed and why?

# 4. Get git status and decide what to commit
felix> !git status
felix> What should I commit first?

# 5. Generate code, pipe to file
felix> /workflow run Generate FastAPI endpoint for user auth
# Copy the code snippet from output, or:
felix> /exit
felix chat -p "Generate only the FastAPI code" > api_endpoint.py

# 6. Chain queries
cat requirements.txt | felix chat -p "Are these versions compatible?" | \
    felix chat -p "Suggest updated versions"
```

---

## Keyboard Shortcuts Reference

When using enhanced mode (prompt_toolkit installed):

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+R` | Search | Reverse search through command history |
| `Ctrl+L` | Clear | Clear screen, keep history |
| `Ctrl+D` | Exit | Exit chat (like EOF) |
| `Ctrl+C` | Cancel | Cancel current input, don't exit |
| `Esc Esc` | Rewind | (Future) Undo last workflow |
| `↑` | Previous | Navigate to previous command |
| `↓` | Next | Navigate to next command |
| `Home` | Start | Jump to start of line |
| `End` | End | Jump to end of line |
| `Ctrl+A` | Start | Alternative to Home |
| `Ctrl+E` | End | Alternative to End |
| `Ctrl+K` | Kill | Delete from cursor to end |
| `Ctrl+U` | Kill Line | Delete entire line |
| `Tab` | Complete | Auto-complete (future) |

---

## Special Prefixes Reference

| Prefix | Purpose | Example | Result |
|--------|---------|---------|--------|
| `!` | Execute bash | `!ls -la` | Runs command, shows output |
| `@` | Read file | `@config.yaml` | Reads file, adds to context |
| `#` | Save note | `#fix bug #123` | (Future) Saves note |

---

## Installation & Setup

### Minimal (Basic Mode)
```bash
# Felix CLI works out of the box with basic features
python -m src.cli chat
```

### Enhanced (Recommended)
```bash
# Install enhanced dependencies
pip install -r requirements-cli-enhanced.txt

# Now you have:
# ✓ Command history
# ✓ Auto-suggestions
# ✓ Keyboard shortcuts
# ✓ Improved formatting
# ✓ Faster JSON operations
```

### Dependencies

**requirements-cli-enhanced.txt:**
- `prompt_toolkit>=3.0.43` - Rich input handling
- `rich>=13.7.0` - Beautiful formatting
- `orjson>=3.9.10` - Fast JSON
- `pygments>=2.17.2` - Syntax highlighting

---

## Migration Guide

### From Old CLI

**Before (Felix 0.8):**
```bash
# Only option: full REPL
felix run "Query here"  # New system for each query

# Or manual session management
felix sessions list
felix chat --resume abc123
```

**After (Felix 0.9 with Phase 1):**
```bash
# Print mode for quick queries
felix chat -p "Query here"  # Fast, scriptable

# Easy session continuation
felix chat -c  # Resume last session

# Special prefixes
felix chat
felix> !git diff
felix> @src/main.py
```

### Backward Compatibility

All old commands still work:
```bash
felix run "task"              # ✓ Still works
felix chat                    # ✓ Enhanced but same behavior
felix chat --resume abc123    # ✓ Still works
felix sessions list           # ✓ Still works
```

New features are **additive only**:
- `-c` flag: New shortcut for `--resume`
- `-p` flag: New non-interactive mode
- Query argument: New for print mode
- Special prefixes: New shortcuts

---

## Performance & Limitations

### Performance

**Print Mode:**
- Faster than interactive REPL (no session overhead)
- Suitable for scripting and automation
- Same workflow execution time as `felix run`

**Command History:**
- ~1ms overhead for history lookup
- No performance impact on workflow execution
- History file size typically <1MB

**Special Prefixes:**
- `!command`: 30s timeout, subprocess overhead
- `@file`: Limited by file read speed, typically <100ms
- `#note`: Negligible overhead

### Limitations

**Print Mode:**
- No multi-turn conversations (by design)
- Each invocation creates new session
- Not suitable for iterative refinement

**Special Prefixes:**
- `!`: 30s timeout limit
- `@`: Text files only (binary files not supported)
- `#`: Not yet implemented (placeholder)

**Keyboard Shortcuts:**
- Requires `prompt_toolkit` library
- Some terminals may not support all shortcuts
- Windows support varies by terminal

---

## Troubleshooting

### Prompt_toolkit not found

**Symptom:**
```
Basic input mode. Install prompt_toolkit for enhanced features
```

**Solution:**
```bash
pip install -r requirements-cli-enhanced.txt
```

### Command history not working

**Symptom:** Arrow keys don't navigate history

**Solution:**
1. Check if `~/.felix_history` exists and is writable
2. Ensure `prompt_toolkit` is installed
3. Try: `chmod 644 ~/.felix_history`

### Piped input not detected

**Symptom:** Print mode doesn't activate with piped input

**Solution:**
1. Ensure you're not in a pseudo-TTY
2. Try explicit `-p` flag: `echo "query" | felix chat -p`
3. Check that stdin contains data

### Special prefixes not working

**Symptom:** `!command` or `@file` shows error

**Solution:**
- `!`: Check command syntax, ensure 30s is enough
- `@`: Verify file path, check read permissions
- Both: Make sure you're in interactive mode (not print mode)

---

## Coming Soon (Phase 2+)

Features planned for future phases:

- **Checkpoint/Rewind System**: Undo workflows with Esc+Esc
- **Custom Commands**: `.felix/commands/` directory
- **Session Titles & Tags**: Better session organization
- **Streaming Output**: Real-time token display
- **Multiline Input**: Shift+Enter for multiline mode
- **Command Auto-completion**: Tab completion
- **Vim Mode**: Optional vim key bindings
- **Background Tasks**: Ctrl+B for background execution

---

## Feedback & Issues

Found a bug or have a suggestion?

1. Check existing issues: [GitHub Issues](https://github.com/your-repo/felix/issues)
2. Create new issue with:
   - Feature name (e.g., "Print Mode Bug")
   - Steps to reproduce
   - Expected vs actual behavior
   - Felix version: `felix --version`
   - Python version: `python --version`

---

## Summary

Phase 1 brings Felix CLI up to Claude Code's usability standards with:

✅ **Piped Input & Print Mode** - Scriptable, composable Felix
✅ **Session Continuation** - Quick `-c` to resume
✅ **Enhanced Input** - Command history, shortcuts, auto-suggest
✅ **Special Prefixes** - `!command`, `@file`, `#note` shortcuts
✅ **Better Help** - Comprehensive examples and documentation
✅ **Backward Compatible** - All old commands still work
✅ **Graceful Degradation** - Works without optional dependencies

These features make Felix's CLI:
- **Faster** for one-off queries (print mode)
- **More convenient** for regular use (session continuation)
- **More productive** with history and shortcuts
- **More integrated** with bash commands and files
- **More professional** with polished UX

Welcome to the enhanced Felix CLI experience! 🚀
