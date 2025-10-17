# Felix GUI Documentation

## Overview

The Felix GUI is a lightweight Tkinter-based interface for the Felix project, providing an intuitive way to interact with the system's core components. It features four main tabs: Dashboard, Workflows, Memory, and Agents, allowing users to control the Felix system, run workflows, manage memory and knowledge, and interact with agents.

## Prerequisites

- Python 3.x (Tkinter is included in standard Python installations)
- Felix project setup with required modules in `src/`
- Database files (`felix_memory.db` and `felix_knowledge.db`) in the project root
- LM Studio: Run LM Studio separately and load/start a model on default port 1234 before starting GUI
- Optional: LLM client setup if using LLM-based agents

## Installation

No additional dependencies are required beyond the Felix project itself. Ensure that the `src/` directory is importable from the project root.

## Usage

Run the GUI from the project root using the following command:

```bash
python -m src.gui.main
```

### Dashboard Tab

The Dashboard tab provides control over the Felix system initialization and real-time logging.

- **Start Felix**: Initializes the central post and verifies LM Studio connection (default 127.0.0.1:1234). Click the "Start Felix" button to begin the system. Verify logs for 'Connected to LM Studio' before using other features.
- **Stop Felix**: Shuts down the running system. Use the "Stop Felix" button to stop all components.
- **LM Studio Config**: Configure host and port for LM Studio connection (defaults to 127.0.0.1:1234).
- **Logs**: Real-time display of system logs, including startup messages, connection status, errors, and operational feedback.

### Workflows Tab

This tab allows users to input tasks and execute them through the linear pipeline.

- **Task Input**: Enter a task description in the text field.
- **Run Workflow**: Click the "Run Workflow" button to start the workflow. The button will be disabled during execution.
- **Progress**: An indeterminate progress bar shows that the workflow is running.
- **Output**: Displays logs and results from the workflow execution, including any errors.

### Memory Tab

The Memory tab has two sub-tabs for managing task memory and knowledge stores. Initial load on tab open; if empty, run workflows to populate DBs.

#### Memory Sub-tab (felix_memory.db)

- **List**: Displays task patterns with IDs and keyword snippets.
- **Search**: Enter a query to filter patterns by keywords.
- **Details**: View full content of selected entries.
- **Edit/Update**: Modify the content of selected entries and save changes.
- **Delete**: Remove selected entries after confirmation.

#### Knowledge Sub-tab (felix_knowledge.db)

- **List**: Displays knowledge entries with IDs and content snippets.
- **Search**: Enter keywords to filter knowledge entries.
- **Details**: View full content of selected entries.
- **Edit/Update**: Modify the content of selected entries and save changes.
- **Delete**: Remove selected entries after confirmation.

### Agents Tab

This tab enables spawning and interacting with agents.

- **Agent Type**: Select from "LLM" or "Specialized" agent types.
- **Spawn**: Click "Spawn" to create a new agent instance.
- **Active Agents**: List of spawned agents with their IDs and types.
- **Message**: Enter a message to send to the selected agent.
- **Send**: Transmit the message to the selected agent.
- **Monitor**: Displays agent details, logs, and responses for the selected agent.

## Threading and Error Handling

- All operations that may block the UI (system start/stop, workflow execution, database operations, agent spawning/interaction) are run in separate threads to keep the interface responsive.
- Errors are displayed in message boxes and logged to both the GUI and a file (`felix_gui.log`).
- Graceful fallbacks are implemented for missing optional components (e.g., LLM client, memory modules).

## Troubleshooting

- **Import Errors**: If Felix modules fail to import, check that you're running from the project root and that all required files are present in `src/`.
- **Database Issues**: Ensure `felix_memory.db` and `felix_knowledge.db` exist in the project root. The GUI will fall back to direct SQL operations if memory modules are unavailable.
- **Memory Empty**: If Memory tab shows nothing, check DB files exist and have data (e.g., via sqlite3 CLI: sqlite3 felix_memory.db 'SELECT * FROM tasks;').
- **LM Studio Not Connecting**: If LM Studio not connecting, verify server running and port (default 1234) matches GUI config. Test with curl http://127.0.0.1:1234/v1/models. Ensure firewall allows connection.
- **LLM Client**: If LLM agents fail, verify the LLM client configuration in `src/llm/` and that LM Studio connection was successful during startup.
- **System Not Starting**: If system fails to start, check that LM Studio is running with a model loaded. The system will not start without successful LM Studio connection verification.
- **Logs**: Check `felix_gui.log` for detailed error information.
- **Threading Issues**: If the GUI becomes unresponsive, restart the application.

## Future Enhancements

As Felix evolves, the GUI may be extended to include:
- Additional agent types and configurations
- Advanced workflow visualization
- Real-time system monitoring charts
- Export/import functionality for memory and knowledge
- Plugin system for custom tabs and features