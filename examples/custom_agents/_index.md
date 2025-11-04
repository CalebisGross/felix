# Custom Agent Examples

## Purpose
Examples demonstrating custom agent plugin development for extending Felix with domain-specific agent types.

## Key Files

### [frontend_agent.py](frontend_agent.py)
Frontend development specialist agent plugin.
- Specializes in React, Vue, Angular, UI/UX, accessibility
- Spawn range: 0.1-0.4 (early-mid phase for interface planning)
- Capabilities: component design, state management, responsive design
- **Use case**: Web application frontend tasks

### [backend_agent.py](backend_agent.py)
Backend development specialist agent plugin.
- Specializes in APIs, databases, authentication, scalability
- Spawn range: 0.15-0.5 (mid-phase for architecture design)
- Capabilities: REST/GraphQL APIs, SQL/NoSQL, microservices
- **Use case**: Server-side and API development tasks

### [qa_agent.py](qa_agent.py)
Quality assurance specialist agent plugin.
- Specializes in testing, test case generation, bug finding
- Spawn range: 0.5-0.8 (late phase for validation)
- Capabilities: unit/integration/e2e testing, coverage analysis
- **Use case**: Code quality and testing tasks

### [code_review_agent.py](code_review_agent.py)
Code review specialist agent plugin.
- Specializes in code quality, security, best practices
- Spawn range: 0.4-0.7 (validation phase)
- Capabilities: static analysis, security auditing, refactoring suggestions
- **Use case**: Code review and quality assurance

### [demo_engineering_agents.py](demo_engineering_agents.py)
Demonstration script showing custom agent system in action.
- Registers all custom engineering agents
- Runs sample tasks with specialized agents
- Shows agent selection based on task type
- **Run**: `python examples/custom_agents/demo_engineering_agents.py`

### [README.md](README.md)
Complete guide to building custom agent plugins.
- Plugin architecture overview
- Step-by-step plugin creation
- Metadata specification
- Task support logic
- Registration and discovery

## Creating Custom Agents

### 1. Define Agent Class
```python
from src.agents.llm_agent import LLMAgent

class DataScienceAgent(LLMAgent):
    """Agent specialized in data analysis and ML."""

    def __init__(self, agent_id, spawn_time, helix, llm_client, **kwargs):
        super().__init__(agent_id, spawn_time, helix, llm_client, **kwargs)
        self.specialty = "data_science"

    def process_task(self, task_input):
        # Add domain-specific logic
        prompt = f"As a data science expert: {task_input}"
        return self.llm_client.complete(prompt)
```

### 2. Create Plugin Wrapper
```python
from src.agents.base_specialized_agent import SpecializedAgentPlugin, AgentMetadata

class DataSciencePlugin(SpecializedAgentPlugin):
    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            agent_type="data_science",
            display_name="Data Science Agent",
            description="Specialized in data analysis, ML, and statistics",
            spawn_range=(0.2, 0.6),
            capabilities=["data_analysis", "machine_learning", "statistics"],
            default_tokens=1000,
            priority=8
        )

    def create_agent(self, agent_id, spawn_time, helix, llm_client, **kwargs):
        return DataScienceAgent(agent_id, spawn_time, helix, llm_client, **kwargs)

    def supports_task(self, task_description: str, task_metadata: dict) -> bool:
        keywords = ["data", "analysis", "ml", "statistics", "model"]
        return any(kw in task_description.lower() for kw in keywords)
```

### 3. Register Plugin
```python
from src.agents.agent_plugin_registry import AgentPluginRegistry

registry = AgentPluginRegistry()
registry.register_plugin(DataSciencePlugin())
```

## Plugin Development Checklist

- [ ] Agent class extends `LLMAgent`
- [ ] Plugin class extends `SpecializedAgentPlugin`
- [ ] `get_metadata()` returns complete `AgentMetadata`
- [ ] `create_agent()` instantiates agent correctly
- [ ] `supports_task()` checks task relevance
- [ ] Spawn range appropriate for agent role
- [ ] Capabilities list accurate
- [ ] Default token budget reasonable
- [ ] Plugin registered with registry
- [ ] Tested with sample tasks

## Related Modules
- [src/agents/base_specialized_agent.py](../../src/agents/base_specialized_agent.py) - Plugin API
- [src/agents/agent_plugin_registry.py](../../src/agents/agent_plugin_registry.py) - Registry
- [src/agents/builtin/](../../src/agents/builtin/) - Built-in plugin examples
