# Felix Agent Plugin API

## Overview

Felix's agent plugin system enables you to extend the framework with custom specialized agents without modifying the core codebase. Plugins implement a simple interface and are automatically discovered by Felix's agent system.

## Quick Start

### 1. Create Your Agent Class

```python
from src.agents.llm_agent import LLMAgent, LLMTask

class MyCustomAgent(LLMAgent):
    def __init__(self, agent_id, spawn_time, helix, llm_client,
                 custom_param="default", **kwargs):
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="my_custom",
            **kwargs
        )
        self.custom_param = custom_param

    def create_position_aware_prompt(self, task: LLMTask, current_time: float):
        """Create custom prompt based on helix position."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        prompt = f"""You are a specialized custom agent.
Current Position: Depth {depth_ratio:.2f}/1.0

Your role: {self.custom_param}

Task: {task.description}
Context: {task.context}
"""
        return prompt, self.max_tokens or 800
```

### 2. Create Plugin Wrapper

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
            default_tokens=800,
            priority=5
        )

    def create_agent(self, agent_id, spawn_time, helix, llm_client,
                    token_budget_manager=None, **kwargs):
        return MyCustomAgent(
            agent_id, spawn_time, helix, llm_client,
            custom_param=kwargs.get('custom_param', 'default'),
            token_budget_manager=token_budget_manager,
            **kwargs
        )
```

### 3. Register and Use

```python
from src.agents.agent_plugin_registry import get_global_registry

# Load your plugin
registry = get_global_registry()
registry.add_plugin_directory("./my_plugins")

# Create agent instances
agent = registry.create_agent(
    agent_type="my_custom",
    agent_id="custom_001",
    spawn_time=0.3,
    helix=helix,
    llm_client=client,
    custom_param="special_mode"
)
```

## Plugin Interface

### Required: `get_metadata()`

Returns metadata describing your agent's capabilities and configuration.

```python
def get_metadata(self) -> AgentMetadata:
    return AgentMetadata(
        agent_type="unique_identifier",      # Required: Unique string ID
        display_name="Human Readable Name",  # Required: Display name
        description="What this agent does",  # Required: Short description
        spawn_range=(0.0, 1.0),             # Optional: Default spawn time range
        capabilities=["cap1", "cap2"],       # Optional: List of capabilities
        tags=["tag1", "tag2"],              # Optional: Classification tags
        default_tokens=800,                  # Optional: Default token limit
        version="1.0.0",                    # Optional: Version string
        author="Your Name",                  # Optional: Author info
        priority=0                           # Optional: Spawn priority
    )
```

### Required: `create_agent()`

Instantiates your agent with the provided parameters.

```python
def create_agent(self,
                agent_id: str,
                spawn_time: float,
                helix: HelixGeometry,
                llm_client: LMStudioClient,
                token_budget_manager: Optional[TokenBudgetManager] = None,
                **kwargs) -> LLMAgent:
    return MyCustomAgent(
        agent_id=agent_id,
        spawn_time=spawn_time,
        helix=helix,
        llm_client=llm_client,
        token_budget_manager=token_budget_manager,
        my_custom_param=kwargs.get('my_custom_param', 'default')
    )
```

### Optional: `supports_task()`

Filters tasks where your agent should be considered for spawning.

```python
def supports_task(self, task_description: str, task_metadata: dict) -> bool:
    """Return True if agent should be spawned for this task."""
    # Example: Only spawn for code-related tasks
    code_keywords = ['code', 'function', 'class', 'bug']
    return any(kw in task_description.lower() for kw in code_keywords)
```

### Optional: `get_spawn_ranges_by_complexity()`

Customizes spawn timing based on task complexity.

```python
def get_spawn_ranges_by_complexity(self) -> dict:
    return {
        "simple": (0.4, 0.7),    # Later spawn for simple tasks
        "medium": (0.3, 0.6),    # Standard range
        "complex": (0.2, 0.5)    # Earlier spawn for complex tasks
    }
```

## AgentMetadata Reference

### agent_type
- **Type:** `str`
- **Required:** Yes
- **Description:** Unique identifier for the agent type
- **Example:** `"code_review"`, `"security_auditor"`

### display_name
- **Type:** `str`
- **Required:** Yes
- **Description:** Human-readable name shown in GUI/CLI
- **Example:** `"Code Review Agent"`

### description
- **Type:** `str`
- **Required:** Yes
- **Description:** Short description of agent's purpose
- **Example:** `"Specialized in code quality and bug detection"`

### spawn_range
- **Type:** `Tuple[float, float]`
- **Default:** `(0.0, 1.0)`
- **Description:** Normalized spawn time range on helix (0.0-1.0)
- **Guidelines:**
  - **Early (0.0-0.3):** Exploration, research, information gathering
  - **Middle (0.3-0.7):** Analysis, processing, pattern identification
  - **Late (0.7-1.0):** Review, validation, quality assurance
  - Note: Synthesis (0.7-1.0) is handled by CentralPost, not agents

### capabilities
- **Type:** `List[str]`
- **Default:** `[]`
- **Description:** List of agent capabilities
- **Examples:**
  - `"web_search"` - Can perform web searches
  - `"code_analysis"` - Can analyze code
  - `"security_audit"` - Can audit security
  - `"pattern_identification"` - Can identify patterns

### tags
- **Type:** `List[str]`
- **Default:** `[]`
- **Description:** Classification tags for filtering and organization
- **Common Tags:**
  - `"exploration"`, `"research"` - Early phase agents
  - `"analysis"`, `"processing"` - Middle phase agents
  - `"review"`, `"quality"`, `"validation"` - Late phase agents
  - `"engineering"`, `"creative"`, `"critical"` - Domain tags

### default_tokens
- **Type:** `int`
- **Default:** `800`
- **Description:** Default maximum tokens for agent completions
- **Guidelines:**
  - `200-500` - Simple, focused responses
  - `800-1200` - Standard agent responses
  - `1500-2000` - Complex, detailed analysis

### version
- **Type:** `str`
- **Default:** `"1.0.0"`
- **Description:** Plugin version (semantic versioning)

### author
- **Type:** `Optional[str]`
- **Default:** `None`
- **Description:** Plugin author or organization

### priority
- **Type:** `int`
- **Default:** `0`
- **Description:** Spawn priority (higher = earlier spawning)
- **Examples:**
  - `10` - High priority (critical agents)
  - `5-7` - Medium-high priority
  - `0` - Normal priority
  - `-5` - Low priority (spawn only if needed)

## Helical Position System

Agents in Felix move along a 3D helix from exploration (top) to synthesis (bottom). Your agent's behavior should adapt based on its position:

### Getting Position Information

```python
def create_position_aware_prompt(self, task, current_time):
    position_info = self.get_position_info(current_time)

    depth_ratio = position_info.get("depth_ratio", 0.0)  # 0.0 = top, 1.0 = bottom
    radius = position_info.get("radius", 0.0)            # Current radius
    x, y, z = position_info.get("x"), position_info.get("y"), position_info.get("z")

    # Adapt behavior based on position
    if depth_ratio < 0.3:
        # Exploration phase: broad, creative
        temperature_hint = "high creativity"
    elif depth_ratio < 0.7:
        # Analysis phase: balanced
        temperature_hint = "balanced"
    else:
        # Late phase: focused, precise
        temperature_hint = "high precision"
```

### Spawn Timing Guidelines

| Phase | Depth Ratio | Agent Types | Behavior |
|-------|-------------|-------------|----------|
| **Exploration** | 0.0 - 0.3 | Research, Information Gathering | High creativity, broad exploration |
| **Analysis** | 0.3 - 0.7 | Analysis, Processing, Custom Logic | Balanced creativity + logic |
| **Review** | 0.7 - 1.0 | Critic, Quality Assurance, Validation | Low temperature, high precision |

Note: Synthesis (final output generation) is performed by CentralPost at the end of the helix, not by specialized agents.

## Advanced Features

### Shared Context

Agents can access information from other agents via `self.shared_context`:

```python
def create_position_aware_prompt(self, task, current_time):
    prompt = "You are a custom agent.\n\n"

    # Access shared context from other agents
    if self.shared_context:
        prompt += "Information from other agents:\n"
        for key, value in self.shared_context.items():
            if "research" in key.lower():
                prompt += f"- {key}: {value}\n"

    return prompt, self.max_tokens
```

### Knowledge Entries

Tasks may include relevant knowledge from Felix's knowledge base:

```python
def create_position_aware_prompt(self, task, current_time):
    prompt = "Your role...\n\n"

    # Include knowledge entries
    if task.knowledge_entries:
        prompt += "Relevant Knowledge:\n"
        for entry in task.knowledge_entries:
            content = entry.content if hasattr(entry, 'content') else str(entry)
            confidence = entry.confidence_level if hasattr(entry, 'confidence_level') else "unknown"
            prompt += f"- [{confidence}] {content}\n"

    return prompt, self.max_tokens
```

### Token Budget Management

Integrate with Felix's token budget system:

```python
def create_position_aware_prompt(self, task, current_time):
    position_info = self.get_position_info(current_time)
    depth_ratio = position_info.get("depth_ratio", 0.0)

    # Get token allocation
    stage_token_budget = self.max_tokens or 800

    if self.token_budget_manager:
        token_allocation = self.token_budget_manager.calculate_stage_allocation(
            self.agent_id, depth_ratio, self.processing_stage + 1
        )
        stage_token_budget = token_allocation.stage_budget

        # Use style guidance
        prompt = f"""Your role...

Token Budget Guidance:
{token_allocation.style_guidance}
"""

    return prompt, stage_token_budget
```

## Best Practices

### 1. Focused Specialization
Create agents with clear, specific roles rather than general-purpose agents.

```python
# Good: Focused agent
class SecurityAuditorAgent(LLMAgent):
    """Specialized in security vulnerability detection."""

# Bad: Too general
class GeneralPurposeAgent(LLMAgent):
    """Does everything."""
```

### 2. Appropriate Spawn Ranges
Match spawn timing to your agent's role in the workflow.

```python
# Research agents: Early exploration
spawn_range=(0.0, 0.3)

# Analysis agents: Middle processing
spawn_range=(0.3, 0.7)

# Review agents: Late validation
spawn_range=(0.7, 0.9)  # Leave 0.9-1.0 for CentralPost synthesis
```

### 3. Task Filtering
Implement `supports_task()` to avoid unnecessary spawning.

```python
def supports_task(self, task_description, task_metadata):
    # Only spawn for security-related tasks
    security_keywords = ['security', 'vulnerability', 'CVE', 'exploit']
    return any(kw in task_description.lower() for kw in security_keywords)
```

### 4. Position-Aware Behavior
Adapt agent behavior based on helical position.

```python
def create_position_aware_prompt(self, task, current_time):
    depth_ratio = self.get_position_info(current_time).get("depth_ratio", 0.0)

    if depth_ratio < 0.3:
        focus = "broad exploration of potential vulnerabilities"
    elif depth_ratio < 0.7:
        focus = "detailed analysis of identified issues"
    else:
        focus = "verification and prioritization of findings"

    prompt = f"Your focus at this stage: {focus}"
    return prompt, self.max_tokens
```

### 5. Clear Capabilities
Document capabilities to enable intelligent agent selection.

```python
capabilities=[
    "vulnerability_scanning",
    "CVE_lookup",
    "dependency_audit",
    "code_injection_detection"
]
```

## Examples

See [examples/custom_agents/](../examples/custom_agents/) for complete examples:

- **code_review_agent.py** - Code review with configurable styles
- **README.md** - Detailed usage guide

## Testing Your Plugin

### Unit Tests

```python
import pytest
from src.agents.agent_plugin_registry import AgentPluginRegistry

def test_my_plugin():
    registry = AgentPluginRegistry()
    registry.add_plugin_directory("./my_plugins")

    # Test plugin loaded
    assert "my_custom" in registry.list_agent_types()

    # Test metadata
    metadata = registry.get_metadata("my_custom")
    assert metadata.display_name == "My Custom Agent"

    # Test agent creation
    agent = registry.create_agent(
        agent_type="my_custom",
        agent_id="test_001",
        spawn_time=0.5,
        helix=mock_helix,
        llm_client=mock_client
    )
    assert agent.agent_type == "my_custom"
```

### Integration Testing

```python
from src.communication.central_post import AgentFactory

def test_plugin_integration():
    factory = AgentFactory(
        helix=helix,
        llm_client=client,
        plugin_directories=["./my_plugins"]
    )

    # Test plugin appears in available types
    assert "my_custom" in factory.list_available_agent_types()

    # Test agent creation via factory
    agent = factory.create_agent_by_type(
        agent_type="my_custom",
        complexity="medium",
        my_custom_param="test"
    )
    assert agent.agent_type == "my_custom"
```

## Troubleshooting

### Plugin Not Loading

**Symptom:** Plugin doesn't appear in `list_agent_types()`

**Solutions:**
1. Check that your class inherits from `SpecializedAgentPlugin`
2. Verify `get_metadata()` and `create_agent()` are implemented
3. Ensure plugin file doesn't start with `_` (skipped by scanner)
4. Check logs for validation errors

### Agent Not Spawning

**Symptom:** Agent type is registered but never spawns

**Solutions:**
1. Check `supports_task()` returns `True` for your test task
2. Verify `spawn_range` is appropriate for task complexity
3. Ensure `priority` isn't too low compared to other agents
4. Check dynamic spawning is enabled in AgentFactory

### Import Errors

**Symptom:** `ModuleNotFoundError` or `ImportError`

**Solutions:**
1. Ensure Felix root is in `PYTHONPATH`
2. Use absolute imports: `from src.agents.llm_agent import LLMAgent`
3. Don't use relative imports (`from .module import ...`)
4. Run from project root: `cd /path/to/felix && python your_script.py`

## API Stability

The plugin API is considered **stable** as of Felix 0.9.0:

- **Stable:** `SpecializedAgentPlugin`, `AgentMetadata`, core registry methods
- **Beta:** Hot-reloading (`reload_external_plugins()`)
- **Experimental:** None currently

Breaking changes to stable APIs will be announced with at least one minor version notice.

## Contributing Plugins

Want to share your plugin with the Felix community?

1. Create plugin in `examples/custom_agents/`
2. Add documentation and tests
3. Submit PR to Felix repository
4. Include example usage and test cases

Great plugin ideas:
- **DataAnalysisAgent** - Statistical analysis and data interpretation
- **APIDesignAgent** - REST/GraphQL API design evaluation
- **TestGeneratorAgent** - Automatic test generation
- **PerformanceOptimizerAgent** - Performance bottleneck analysis
- **DocumentationAgent** - Documentation generation and review

## Support

- **Documentation:** [https://github.com/yourusername/felix/docs](docs)
- **Examples:** [examples/custom_agents/](../examples/custom_agents/)
- **Issues:** [GitHub Issues](https://github.com/yourusername/felix/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/felix/discussions)
