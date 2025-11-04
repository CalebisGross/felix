# Built-in Agent Plugins Module

## Purpose
Built-in agent plugin implementations wrapping Felix's core specialized agents (Research, Analysis, Critic) in the plugin architecture for seamless integration with AgentFactory.

## Key Files

### [research_plugin.py](research_plugin.py)
Research agent plugin for broad information gathering and exploration.
- **`ResearchAgentPlugin`**: Plugin wrapper for `ResearchAgent`
- **Spawn range**: 0.0-0.3 (exploration phase, top of helix)
- **Capabilities**: Web search, information gathering, perspective generation, source discovery
- **Default tokens**: 800
- **Priority**: 10 (high priority for research tasks)
- **Task support**: All tasks, especially those requiring "research", "find", "explore", "investigate"
- **Complexity-aware spawning**: Earlier spawn for complex tasks (0.0-0.20), later for simple (0.05-0.25)

### [analysis_plugin.py](analysis_plugin.py)
Analysis agent plugin for detailed processing and synthesis.
- **`AnalysisAgentPlugin`**: Plugin wrapper for `AnalysisAgent`
- **Spawn range**: 0.2-0.6 (mid-phase, transitioning from exploration to synthesis)
- **Capabilities**: Data analysis, pattern recognition, synthesis, reasoning, deep processing
- **Default tokens**: 1000
- **Priority**: 8 (medium-high priority)
- **Task support**: Tasks requiring "analyze", "evaluate", "compare", "synthesize", "process"
- **Complexity-aware spawning**: Broader range for complex tasks (0.15-0.65), narrower for simple (0.25-0.55)

### [critic_plugin.py](critic_plugin.py)
Critic agent plugin for validation, quality assurance, and reasoning evaluation.
- **`CriticAgentPlugin`**: Plugin wrapper for `CriticAgent`
- **Spawn range**: 0.4-0.7 (validation phase, middle to lower helix)
- **Capabilities**: Quality validation, bias detection, logical evaluation, error checking, reasoning assessment
- **Default tokens**: 600
- **Priority**: 7 (medium priority)
- **Task support**: All tasks benefit from validation; especially "verify", "check", "validate", "critique"
- **Complexity-aware spawning**: Earlier spawn for complex tasks (0.3-0.7), later for simple (0.45-0.75)

### [\_\_init\_\_.py](__init__.py)
Module initialization and plugin registration.
- Imports all built-in plugins
- Auto-registers plugins with `AgentPluginRegistry`
- Provides `__all__` export list for clean imports

## Key Concepts

### Plugin Architecture
Each built-in agent is wrapped as a `SpecializedAgentPlugin`:
```python
class ResearchAgentPlugin(SpecializedAgentPlugin):
    def get_metadata(self) -> AgentMetadata:
        # Return agent capabilities and requirements

    def create_agent(self, ...) -> LLMAgent:
        # Instantiate the actual agent

    def supports_task(self, task_description: str, ...) -> bool:
        # Determine if agent is suitable for task
```

### Agent Metadata
Each plugin provides detailed metadata:
- **agent_type**: Type identifier (e.g., "research", "analysis", "critic")
- **display_name**: Human-readable name
- **description**: Purpose and specialization
- **spawn_range**: Normalized time range (0.0-1.0) for spawning
- **capabilities**: List of capabilities (e.g., "web_search", "reasoning")
- **tags**: Categorization tags
- **default_tokens**: Default token budget
- **version**: Plugin version
- **priority**: Spawning priority (higher = more important)

### Spawn Ranges
Agents spawn at different points along the helix:
1. **Research (0.0-0.3)**: Early exploration phase, wide helix radius, high temperature
2. **Analysis (0.2-0.6)**: Mid-phase processing, transitioning radius, moderate temperature
3. **Critic (0.4-0.7)**: Validation phase, narrowing radius, lower temperature

Ranges overlap to ensure smooth transitions and collaborative work.

### Complexity-Aware Spawning
Plugins adjust spawn ranges based on task complexity:
- **Simple tasks**: Narrower ranges, later spawns (fewer agents needed)
- **Medium tasks**: Standard ranges
- **Complex tasks**: Broader ranges, earlier spawns (more agents, longer processing)

### Task Support Detection
Each plugin analyzes task descriptions for relevant keywords:
- **Research**: "research", "find", "explore", "latest", "current"
- **Analysis**: "analyze", "compare", "evaluate", "synthesize"
- **Critic**: "verify", "check", "validate", "critique", "review"

### Legacy Compatibility
Plugins wrap existing `specialized_agents.py` implementations:
- No changes to core agent logic
- Plugin layer adds metadata and factory integration
- Maintains backward compatibility with existing workflows

### Integration with AgentFactory
`AgentFactory` uses plugins to spawn agents dynamically:
1. Query registry for suitable agent types
2. Check `supports_task()` and spawn range
3. Call `create_agent()` to instantiate
4. Register agent with `CentralPost`

### Extensibility
New agent types can be added as plugins without modifying core:
1. Create agent class (extend `LLMAgent`)
2. Create plugin wrapper (extend `SpecializedAgentPlugin`)
3. Register in `__init__.py` or custom directory
4. Auto-discovered by `AgentPluginRegistry`

## Built-in vs Custom Plugins

### Built-in Plugins (this directory)
- Core agent types shipped with Felix
- Research, Analysis, Critic
- Registered by default
- Located in `src/agents/builtin/`

### Custom Plugins (external)
- User-defined agent types
- Located in configured plugin directories (e.g., `plugins/agents/`)
- Discovered via `AgentPluginRegistry.load_from_directory()`
- See [examples/custom_agents/](../../../examples/custom_agents/) for examples

## Related Modules
- [specialized_agents.py](../specialized_agents.py) - Actual agent implementations being wrapped
- [base_specialized_agent.py](../base_specialized_agent.py) - Plugin interface definition
- [agent_plugin_registry.py](../agent_plugin_registry.py) - Plugin discovery and management
- [communication/central_post.py](../../communication/central_post.py) - AgentFactory integration
- [examples/custom_agents/](../../../examples/custom_agents/) - Custom plugin examples
