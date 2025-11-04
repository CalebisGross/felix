# CLI Tools Module

## Purpose
Tool integrations providing structured access to Felix subsystems from the conversational CLI (workflows, history, knowledge, agents, system status, documents).

## Key Files

### [base_tool.py](base_tool.py)
Base classes for CLI tool system.
- **`BaseTool`**: Abstract base class defining tool interface
- **`ToolResult`**: Standardized result structure with status, data, and error info
- **`ToolRegistry`**: Registry for discovering and managing available tools
- **Tool interface**: `name()`, `description()`, `execute()` methods
- **Error handling**: Consistent error result format across all tools

### [workflow_tool.py](workflow_tool.py)
Tool for executing and monitoring workflows.
- **`WorkflowTool`**: Create and track workflow execution from CLI
- **Operations**:
  - Execute new workflow with task description
  - Check workflow status and progress
  - Retrieve workflow results
  - List active and completed workflows
- **Integration**: Uses `CLIWorkflowOrchestrator` for proper architecture integration
- **Progress updates**: Real-time status updates during execution

### [history_tool.py](history_tool.py)
Tool for querying workflow history.
- **`HistoryTool`**: Access historical workflow data
- **Query operations**:
  - List workflows by date range
  - Filter by status (completed, failed)
  - Search by task content
  - View detailed workflow records
- **Analytics**: Aggregate statistics (success rate, avg confidence, token usage)
- **Data source**: Queries `felix_workflow_history.db`

### [knowledge_tool.py](knowledge_tool.py)
Tool for semantic search and knowledge management.
- **`KnowledgeTool`**: Interface to Knowledge Brain
- **Operations**:
  - Semantic search for concepts
  - Browse knowledge by domain
  - View concept relationships
  - Check knowledge statistics
- **Requires**: Knowledge Brain enabled (`enable_knowledge_brain: true`)
- **Retrieval**: Uses `KnowledgeRetriever` with meta-learning boost

### [agent_tool.py](agent_tool.py)
Tool for agent interaction and inspection.
- **`AgentTool`**: Query and interact with active agents
- **Capabilities**:
  - List active agents with status
  - Query agent details (role, confidence, position)
  - Check team composition
  - View agent performance metrics
- **Registry access**: Queries `AgentRegistry` via `CentralPost`

### [system_tool.py](system_tool.py)
Tool for system status and configuration.
- **`SystemTool`**: Check Felix system health
- **Status checks**:
  - LLM provider connection
  - Database health
  - Knowledge Brain status
  - Memory usage
  - Configuration summary
- **Diagnostics**: Identify configuration issues

### [document_tool.py](document_tool.py)
Tool for document ingestion to Knowledge Brain.
- **`DocumentTool`**: Ingest documents from CLI
- **Operations**:
  - Ingest single document
  - Batch document processing
  - List ingested documents
  - Check document processing status
- **Formats**: PDF, TXT, MD, Python, JS, Java, C++
- **Processing**: Triggers agentic comprehension and knowledge graph building

## Key Concepts

### Tool Architecture
All tools implement the `BaseTool` interface:
```python
class MyTool(BaseTool):
    def name(self) -> str:
        return "my-tool"

    def description(self) -> str:
        return "Tool description"

    def execute(self, *args, **kwargs) -> ToolResult:
        # Tool logic
        return ToolResult(success=True, data=result)
```

### Tool Discovery
Tools are auto-discovered via `ToolRegistry`:
- Tools register themselves on import
- CLI queries registry for available tools
- Dynamic tool loading enables extensibility

### Consistent Results
All tools return `ToolResult` objects:
```python
ToolResult(
    success=bool,       # True if operation succeeded
    data=Any,           # Result data (dict, list, str, etc.)
    error=str,          # Error message if failed
    metadata=dict       # Optional metadata (timing, counts, etc.)
)
```

### Error Handling
Tools handle errors gracefully:
- Catch exceptions and wrap in `ToolResult`
- Provide user-friendly error messages
- Include diagnostics in metadata
- Never crash the CLI

### Integration with Chat
Tools are invoked from chat interface:
```
User: @workflow execute "Analyze quantum computing"
Assistant: [WorkflowTool executes and returns formatted result]

User: @history list --limit 10
Assistant: [HistoryTool queries and displays recent workflows]

User: @knowledge search "machine learning"
Assistant: [KnowledgeTool performs semantic search and shows results]
```

### Felix System Access
Tools access Felix subsystems through dependency injection:
- **WorkflowTool**: Gets `felix_system` and `session_manager` via constructor
- **HistoryTool**: Accesses `workflow_history` database directly
- **KnowledgeTool**: Uses `knowledge_store` and `knowledge_retriever`
- **AgentTool**: Queries `central_post.agent_registry`

### Extensibility
New tools can be added by:
1. Create class extending `BaseTool`
2. Implement required methods
3. Register in `ToolRegistry`
4. Import in `__init__.py`
5. Tool automatically available in CLI

## Usage Examples

### Workflow Tool
```bash
User: @workflow execute "Explain quantum entanglement"
> Workflow wf_abc123 created
> Spawning agents...
> Research agent spawned (confidence: 0.65)
> Analysis agent spawned (confidence: 0.82)
> Synthesis complete (confidence: 0.87)
> [Formatted result displayed]
```

### History Tool
```bash
User: @history list --limit 5 --status completed
> Recent Workflows:
> 1. wf_abc123 - "Explain quantum..." - 0.87 confidence - 45s
> 2. wf_def456 - "Design REST API..." - 0.91 confidence - 32s
> ...
```

### Knowledge Tool
```bash
User: @knowledge search "neural networks"
> Found 15 relevant concepts:
> 1. Neural Network Architecture (relevance: 0.92)
> 2. Backpropagation Algorithm (relevance: 0.88)
> ...
```

### Agent Tool
```bash
User: @agent list
> Active Agents (3):
> - research_001: ResearchAgent (confidence: 0.75, position: 0.2)
> - analysis_001: AnalysisAgent (confidence: 0.82, position: 0.4)
> - critic_001: CriticAgent (confidence: 0.69, position: 0.5)
```

## Related Modules
- [chat.py](../chat.py) - Main CLI that invokes tools
- [command_handler.py](../command_handler.py) - Routes tool invocations
- [workflows/](../../workflows/) - Workflow execution
- [knowledge/](../../knowledge/) - Knowledge Brain
- [memory/](../../memory/) - Workflow history
- [agents/](../../agents/) - Agent system
