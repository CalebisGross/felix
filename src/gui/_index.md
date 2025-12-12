# GUI Module

## Purpose
Tkinter-based graphical interface providing system control, workflow execution, memory management, agent interaction, command approval, terminal monitoring, and Knowledge Brain oversight with dark/light theme support.

## Key Files

### [main.py](main.py)
Application entry point and tab management.
- **`MainApp`**: Main tk.Tk application with tab navigation and theme management
- **`run_gui()`**: Entry function for launching GUI

### [felix_system.py](felix_system.py)
Unified system manager coordinating all Felix components.
- **`FelixSystem`**: Central coordinator managing agents, workflows, memory, knowledge brain
- **`FelixConfig`**: Configuration loader and validator
- **System lifecycle**: Startup, shutdown, health monitoring

### [dashboard.py](dashboard.py)
System control and monitoring dashboard.
- **`DashboardFrame`**: Start/stop Felix system, view logs, check LM Studio connection
- **Features**: Real-time log streaming, system status indicators

### [workflows.py](workflows.py)
Workflow execution interface.
- **`WorkflowsFrame`**: Run tasks through linear pipeline with web search integration
- **Features**: Task input, real-time execution tracking, formatted markdown export, synthesis display

### [memory.py](memory.py)
Memory browsing and editing interface.
- **`MemoryFrame`**: Browse/edit task memory and knowledge store entries
- **Features**: Search, filter, edit confidence scores, delete entries

### [agents.py](agents.py)
Agent spawning and interaction.
- **`AgentsFrame`**: Spawn agents manually, send messages, view agent states
- **Features**: Agent list view, direct communication, status monitoring

### [approvals.py](approvals.py)
System command approval workflow.
- **`ApprovalsFrame`**: View pending commands, approve/deny with trust level selection
- **Decision Types**: APPROVE_ONCE, APPROVE_ALWAYS, DENY_ONCE, DENY_ALWAYS, REQUEST_MORE_INFO
- **Features**: Pending queue, approval history, trust management

### [terminal.py](terminal.py)
Real-time command execution monitoring.
- **`TerminalFrame`**: Monitor active command execution with streaming output
- **Features**: Real-time output display, command history browser, auto-refresh

### [learning.py](learning.py)
Learning system configuration.
- **`LearningFrame`**: Configure feedback collection and learning parameters
- **Features**: Enable/disable learning, adjust thresholds, view learning statistics

### [knowledge_brain.py](knowledge_brain.py)
Knowledge Brain monitoring and control (5 sub-tabs).
- **`KnowledgeBrainFrame`**: Main frame with tab navigation

**Sub-tabs**:
1. **Overview**: Daemon control (start/stop/refresh), status display, ingestion statistics
2. **Documents**: Browse ingested documents, filter by status (pending/processing/completed/failed)
3. **Concepts**: Search knowledge by domain, view related concepts, explore concept details
4. **Activity**: Real-time processing log with auto-refresh and log level filtering
5. **Relationships**: Explore knowledge graph, search by concept, view relationship networks

### [settings.py](settings.py)
System configuration interface.
- **`SettingsFrame`**: Edit YAML configuration, toggle Knowledge Brain, save changes
- **Features**: Syntax-highlighted YAML editor, validation, hot reload

### [prompts.py](prompts.py)
Agent prompt management.
- **`PromptsFrame`**: View, edit, and test agent prompt templates
- **Features**: Template library, variable substitution preview, A/B testing support

### [workflow_history_frame.py](workflow_history_frame.py)
Workflow execution history browser.
- **`WorkflowHistoryFrame`**: Search and filter past workflow executions
- **Features**: Date range filtering, status filtering, performance metrics, detail view

### [themes.py](themes.py)
Dark/light theme management.
- **`ThemeManager`**: Manages theme switching with persistent preference
- **Themes**: Dark mode (default), Light mode
- **Features**: Consistent styling across all frames, custom color schemes

### [logging_handler.py](logging_handler.py)
GUI-integrated logging handler.
- **`GUILoggingHandler`**: Routes Python logging to GUI text widgets
- **Features**: Thread-safe logging, log level coloring, auto-scroll

### [utils.py](utils.py)
GUI utility functions and helpers.
- **`ThreadManager`**: Manages background threads for async operations
- **`DBHelper`**: Database connection helpers for GUI queries
- **Other utilities**: Validation, formatting, UI helpers

## Key Concepts

### Tab Architecture
8 main tabs + Knowledge Brain sub-tabs:
1. Dashboard - System control
2. Workflows - Task execution
3. Memory - Data browsing
4. Agents - Agent management
5. Approvals - Command approval
6. Terminal - Command monitoring
7. Learning - Learning config
8. Knowledge Brain (5 sub-tabs) - Document learning

### Theme System
- **Persistent preference**: Theme choice saved across sessions
- **Consistent styling**: All frames inherit theme colors
- **Real-time switching**: Toggle without restart

### Real-Time Updates
- **Log streaming**: Live log display in Dashboard
- **Command output**: Streaming terminal output in Terminal tab
- **Activity monitoring**: Auto-refreshing Knowledge Brain activity log
- **Status indicators**: Live system health in Dashboard

### Approval Workflow
Five-stage decision process:
1. **APPROVE_ONCE**: Allow single execution
2. **APPROVE_ALWAYS**: Trust command permanently
3. **DENY_ONCE**: Reject single execution
4. **DENY_ALWAYS**: Block command permanently
5. **REQUEST_MORE_INFO**: Ask agent for clarification

### Markdown Export
- **Synthesis results**: Save formatted workflow outputs
- **Agent metrics**: Include performance statistics
- **Professional formatting**: Consistent markdown styling

### Knowledge Brain Control
- **Daemon management**: Start/stop background processor
- **Document browser**: Track ingestion pipeline
- **Concept explorer**: Navigate knowledge graph interactively
- **Relationship viewer**: Visualize concept connections

## LM Studio Requirement
GUI requires LM Studio running on port 1234 before starting Felix system. Connection tested via Dashboard tab.

## Related Modules
- [felix_system.py](felix_system.py) - Central coordinator used by all GUI components
- [workflows/](../workflows/) - Workflow execution backend
- [memory/](../memory/) - Data persistence for Memory tab
- [agents/](../agents/) - Agent system for Agents tab
- [execution/](../execution/) - Command execution for Approvals/Terminal
- [knowledge/](../knowledge/) - Knowledge Brain backend
- [learning/](../learning/) - Learning system configuration
