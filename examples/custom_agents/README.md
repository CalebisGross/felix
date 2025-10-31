# Custom Agent Plugins

This directory contains example custom agent plugins for the Felix Framework.

## What are Agent Plugins?

Agent plugins allow you to extend Felix with custom specialized agents without modifying the core codebase. Plugins implement the `SpecializedAgentPlugin` interface and are automatically discovered by Felix's agent system.

## Creating a Custom Agent

### Step 1: Define Your Agent Class

Create a new LLMAgent subclass with custom prompting and behavior:

```python
from src.agents.llm_agent import LLMAgent, LLMTask

class MyCustomAgent(LLMAgent):
    def __init__(self, agent_id, spawn_time, helix, llm_client, **kwargs):
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="my_custom",
            **kwargs
        )

    def create_position_aware_prompt(self, task, current_time):
        # Create custom prompts for your agent
        prompt = "You are a specialized agent for..."
        return prompt, self.max_tokens or 800
```

### Step 2: Create a Plugin Wrapper

Wrap your agent in a plugin class:

```python
from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)

class MyCustomAgentPlugin(SpecializedAgentPlugin):
    def get_metadata(self):
        return AgentMetadata(
            agent_type="my_custom",
            display_name="My Custom Agent",
            description="Does custom things",
            spawn_range=(0.2, 0.6),
            capabilities=["custom_capability"],
            tags=["custom"],
            default_tokens=800
        )

    def create_agent(self, agent_id, spawn_time, helix, llm_client,
                    token_budget_manager=None, **kwargs):
        return MyCustomAgent(
            agent_id, spawn_time, helix, llm_client,
            token_budget_manager=token_budget_manager,
            **kwargs
        )
```

### Step 3: Register Your Plugin

Load your plugin directory when initializing Felix:

```python
from src.agents.agent_plugin_registry import AgentPluginRegistry

registry = AgentPluginRegistry()
registry.discover_builtin_plugins()
registry.add_plugin_directory("./examples/custom_agents")

# Your custom agent is now available!
agent = registry.create_agent(
    agent_type="my_custom",
    agent_id="custom_001",
    spawn_time=0.3,
    helix=helix,
    llm_client=client
)
```

## Example: CodeReviewAgent

See [code_review_agent.py](code_review_agent.py) for a complete example of a custom agent that specializes in code review.

**Features:**
- Custom prompting for code analysis
- Bug detection and style checking
- Security vulnerability scanning
- Configurable review styles (quick, thorough, security-focused)

**Usage:**

```python
from src.agents.agent_plugin_registry import get_global_registry

registry = get_global_registry()
registry.add_plugin_directory("./examples/custom_agents")

# Create code review agent
agent = registry.create_agent(
    agent_type="code_review",
    agent_id="reviewer_001",
    spawn_time=0.4,
    helix=helix,
    llm_client=client,
    review_style="security-focused"
)
```

## Plugin API Reference

### AgentMetadata Fields

- **agent_type** (required): Unique identifier (e.g., "code_review")
- **display_name** (required): Human-readable name
- **description** (required): Short description
- **spawn_range**: (min, max) normalized time when agent spawns (0.0-1.0)
- **capabilities**: List of capabilities (e.g., ["code_analysis"])
- **tags**: Classification tags (e.g., ["engineering"])
- **default_tokens**: Default max tokens for completions
- **version**: Plugin version string
- **author**: Plugin author (optional)
- **priority**: Spawn priority (higher = earlier, default: 0)

### Plugin Methods

#### `get_metadata()` (required)

Returns `AgentMetadata` describing your agent.

#### `create_agent(...)` (required)

Creates an instance of your agent. Must return an `LLMAgent` subclass.

**Parameters:**
- `agent_id`: Unique identifier
- `spawn_time`: Normalized spawn time (0.0-1.0)
- `helix`: HelixGeometry instance
- `llm_client`: LLM client for completions
- `token_budget_manager`: Optional token budget manager
- `**kwargs`: Custom parameters

#### `supports_task(task_description, task_metadata)` (optional)

Returns `True` if your agent should be considered for the task.

Default: Returns `True` for all tasks.

**Example:**
```python
def supports_task(self, task_description, task_metadata):
    # Only spawn for code-related tasks
    return 'code' in task_description.lower()
```

#### `get_spawn_ranges_by_complexity()` (optional)

Returns spawn ranges for different task complexities.

Default: Uses spawn_range from metadata for all complexities.

**Example:**
```python
def get_spawn_ranges_by_complexity(self):
    return {
        "simple": (0.4, 0.7),
        "medium": (0.3, 0.6),
        "complex": (0.2, 0.5)  # Spawn earlier for complex
    }
```

## Best Practices

1. **Focused Agents**: Create agents with specific, well-defined roles
2. **Clear Naming**: Use descriptive agent_type names (e.g., "security_auditor")
3. **Appropriate Spawn Ranges**:
   - Early (0.0-0.3): Exploration, research
   - Middle (0.3-0.7): Analysis, processing
   - Late (0.7-1.0): Review, validation (synthesis is CentralPost)
4. **Task Filtering**: Implement `supports_task()` to avoid spawning for irrelevant tasks
5. **Documentation**: Document capabilities and usage in plugin docstrings

## Plugin Lifecycle

1. **Discovery**: Plugin files are scanned for `SpecializedAgentPlugin` subclasses
2. **Registration**: `get_metadata()` is called to register the agent type
3. **Task Matching**: `supports_task()` filters agents for tasks
4. **Spawning**: `create_agent()` instantiates agents based on complexity and spawn ranges
5. **Execution**: Agents process tasks through their custom prompts and logic

## Troubleshooting

**Plugin Not Loading**

- Check that your plugin inherits from `SpecializedAgentPlugin`
- Ensure `get_metadata()` and `create_agent()` are implemented
- Verify the plugin file is in the registered directory
- Check logs for validation errors

**Agent Not Spawning**

- Verify `supports_task()` returns True for your task
- Check spawn_range matches task complexity
- Ensure priority isn't too low compared to other agents

**Import Errors**

- Make sure Felix is in your PYTHONPATH
- Use absolute imports: `from src.agents.llm_agent import LLMAgent`
- Don't import from `__main__` or use relative imports

## More Examples

Want to contribute an example? Consider these agent ideas:

- **DataAnalysisAgent**: Specialized in statistical analysis and data interpretation
- **DocumentationAgent**: Generates and reviews documentation
- **SecurityAuditorAgent**: Deep security vulnerability scanning
- **PerformanceOptimizerAgent**: Identifies performance bottlenecks
- **APIDesignAgent**: Evaluates and designs REST/GraphQL APIs
- **TestGeneratorAgent**: Creates unit and integration tests

Submit a PR with your custom agent!
