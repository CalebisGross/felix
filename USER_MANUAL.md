# Felix Framework User Manual

## Introduction

The Felix Framework is a Python-based multi-agent AI system designed to leverage helical geometry for agent progression and adaptation. It enables dynamic, scalable AI interactions by modeling agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks. The framework integrates local LLM clients, persistent memory systems, and modular pipelines to support complex, adaptive workflows in multi-agent environments.

At its core, Felix employs a hub-spoke communication model combined with helical progression to facilitate agent spawning, role specialization, and task execution. This architecture promotes resilience and efficiency in handling diverse AI challenges, from prompt optimization to knowledge compression. By incorporating token budgeting and context-aware memory, the system ensures sustainable performance across varying computational constraints.

Felix is structured around key modules that handle agents, communication, core geometry, LLM integration, memory management, and pipeline processing. This modular design allows for flexible deployment and extension, making it suitable for applications requiring autonomous agent coordination and adaptive learning.

Key features include:
- Helical geometry-driven agent adaptation
- Dynamic agent spawning based on confidence and content analysis
- Hub-spoke communication for efficient message routing
- Persistent memory systems with compression
- LLM integration with token budgeting
- Pipeline support for linear and chunked processing

Use cases include autonomous drone swarms for environmental monitoring, personalized AI assistants in education, and scalable chatbots for customer service. The framework supports hypothesis validation for H1 (helical progression enhances adaptation), H2 (multi-agent communication optimizes resource allocation), and H3 (memory compression reduces latency).

## Prerequisites and Setup

### System Requirements
- Python 3.8 or higher
- At least 16GB RAM recommended for optimal performance
- SQLite3 for persistent memory storage
- Optional: LM Studio server for local LLM inference

### Installation Steps
1. Ensure Python 3.8+ is installed on your system.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or .venv\Scripts\activate on Windows
   ```
   **Warning:** Always activate the virtual environment before running Felix to avoid dependency conflicts.

3. Install required dependencies:
   ```bash
   pip install openai httpx numpy scipy
   ```
   For full functionality, also install additional packages as needed (e.g., `sqlite3` is typically included with Python).

4. (Optional) Set up LM Studio server for local LLM integration. Download and run LM Studio, then configure the server endpoint in your configuration.

5. Initialize databases: Felix automatically creates `felix_memory.db` and `felix_knowledge.db` on first run for KnowledgeStore and TaskMemory persistence.

## Configuration

Felix uses YAML configuration files for parameterization. Load configuration using `yaml.safe_load()` as shown in [exp/example_workflow.py](exp/example_workflow.py). Parameters are grouped by module and balance trade-offs across hypotheses H1 (adaptation), H2 (communication efficiency), and H3 (memory latency).

### Key Parameter Groups

| Module | Parameter | Optimal Value | Description |
|--------|-----------|---------------|-------------|
| HelixGeometry | top_radius | 3.0 | Radius at top for broad exploration (H1) |
| HelixGeometry | bottom_radius | 0.5 | Radius at bottom for focused precision (H3) |
| HelixGeometry | height | 8.0 | Total progression depth |
| HelixGeometry | turns | 2 | Number of helical spirals for complexity |
| Agent Spawning | confidence_threshold | 0.80 | Trigger for dynamic spawning |
| Agent Spawning | max_agents | 10 | Maximum team size |
| Agent Spawning | spawn_time_ranges | Research: [0.0, 0.25], Analysis: [0.2, 0.6], Synthesis: [0.6, 0.9], Critic: [0.4, 0.7] | Normalized time ranges for agent types |
| LLM | temperature_range | Research: [0.5, 1.0], Analysis: [0.3, 0.7], Synthesis: [0.2, 0.4], Critic: [0.2, 0.5] | Creativity vs focus gradient |
| TokenBudget | base_budget | 2048 | Base tokens per agent |
| TokenBudget | strict_mode | true | Enforce tight budgets |
| Memory | compression_ratio | 0.3 | Context reduction ratio |
| Memory | relevance_threshold | 0.4 | Filter low-relevance content |
| Pipeline | chunk_size | 512 | Token-based chunking |

### Sample Configuration File

Create `felix_config.yaml` in your project root:

```yaml
helix:
  top_radius: 3.0
  bottom_radius: 0.5
  height: 8.0
  turns: 2

spawning:
  confidence_threshold: 0.80
  max_agents: 10
  spawn_ranges:
    research: [0.0, 0.25]
    analysis: [0.2, 0.6]
    synthesis: [0.6, 0.9]
    critic: [0.4, 0.7]

llm:
  temperature_ranges:
    research: [0.5, 1.0]
    analysis: [0.3, 0.7]
    synthesis: [0.2, 0.4]
    critic: [0.2, 0.5]
  token_budget:
    base_budget: 2048
    strict_mode: true

memory:
  compression_ratio: 0.3
  relevance_threshold: 0.4

pipeline:
  chunk_size: 512
```

Load this in your scripts:
```python
import yaml
with open('felix_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

## Running the Framework

### Basic Example Workflow
1. Activate the virtual environment: `source .venv/bin/activate`
2. Run the example script: `python exp/example_workflow.py`
   This demonstrates a complete workflow with mock LLM responses, showing agent spawning, task processing, and memory storage.

### Custom Workflows
For custom implementations, import Felix components and instantiate core objects:

```python
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent

# Initialize components
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
central_post = CentralPost(max_agents=10, enable_metrics=True, enable_memory=True)
agent_factory = AgentFactory(helix, llm_client)

# Create and register agents
research_agent = agent_factory.create_research_agent(domain="technical")
analysis_agent = agent_factory.create_analysis_agent()
synthesis_agent = agent_factory.create_synthesis_agent()

for agent in [research_agent, analysis_agent, synthesis_agent]:
    central_post.register_agent(agent)

# Process tasks (see [exp/workflow_steps.md](exp/workflow_steps.md) for detailed patterns)
```

Monitor progress via CentralPost metrics:
```python
metrics = central_post.get_performance_summary()
print(f"Messages processed: {metrics['total_messages_processed']}")
```

## Core Operations

### Agent Spawning and Lifecycle
Agents spawn dynamically based on confidence thresholds and content analysis. Static creation uses factory methods:

```python
agent = LLMAgent(helix, llm_client, agent_type="research")
agent.spawn(current_time, task)
```

Dynamic spawning monitors team performance and spawns additional agents when confidence drops below 0.75 or issues are detected.

### Task Assignment and Processing
Tasks are assigned via messages through CentralPost:

```python
from src.communication.central_post import Message, MessageType
message = Message("agent_id", MessageType.TASK_ASSIGNMENT, {"task": "research query"}, time.time())
central_post.queue_message(message)
```

Agents process tasks with position-aware LLM prompting:
```python
result = agent.process_task_with_llm(task, current_time)
```

### LLM Prompting and Token Management
Prompts adapt based on helical position, with temperature decreasing from 1.0 (top) to 0.2 (bottom). Token budgets are allocated per agent type and stage.

### Memory Queries and Storage
Store results in KnowledgeStore:
```python
central_post.store_agent_result_as_knowledge(agent_id, content, confidence, domain)
```

Query knowledge:
```python
from src.memory.knowledge_store import KnowledgeQuery
query = knowledge_store.retrieve_knowledge(KnowledgeQuery(domains=["domain"], limit=5))
```

### Pipeline Execution
Use LinearPipeline for sequential processing or chunking for large outputs:
```python
from src.pipeline.linear_pipeline import LinearPipeline
pipeline = LinearPipeline()
result = pipeline.process(task)
```

## Monitoring and Metrics

Access H1-H2-H3 metrics from CentralPost:
```python
summary = central_post.get_memory_summary()
metrics = central_post.get_performance_summary()
```

H1 metrics track adaptation efficiency (helical vs linear progression). H2 measures communication overhead and resource allocation. H3 monitors memory compression latency reduction.

Enable logging by setting log levels in your script. For visualization, export metrics to CSV and use plotting libraries like matplotlib.

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure .venv is activated and all dependencies are installed. Check Python path includes project root.
- **LLM Connection Fails**: Use mock client fallback as in [exp/example_workflow.py](exp/example_workflow.py). Verify LM Studio server is running and endpoint is correct.
- **Spawn Timing Delays**: Check confidence thresholds and spawn ranges. Ensure helix parameters allow proper progression.
- **Memory DB Locks**: Close database connections properly. Use `central_post.cleanup()` on shutdown.

### Solutions
- Restart with clean environment if imports fail.
- Test with mock LLM first, then switch to real client.
- Adjust `max_agents` downward if resource constrained.
- Monitor disk space for database growth.

## Advanced Usage

### Custom Agents
Extend base `Agent` class for specialized roles:
```python
from src.agents.agent import Agent
class CustomAgent(Agent):
    def process_task(self, task):
        # Custom logic
        pass
```

### External LLM Integration
Implement custom LLM clients by extending `LMStudioClient` or using `MultiServerClient` for distributed setups.

### Scaling to 133 Agents
Increase `max_agents` in configuration, but monitor H2 communication overhead. Use mesh extensions for very large teams.

### Hypothesis Validation
Run experiments comparing helical vs linear pipelines. Track H1 adaptation metrics, H2 resource usage, and H3 latency improvements.

## References

- [index.md](index.md) - Framework overview and project structure
- [exp/workflow_steps.md](exp/workflow_steps.md) - Detailed workflow breakdown
- [exp/example_workflow.py](exp/example_workflow.py) - Runnable demonstration script
- [exp/optimal_parameters.md](exp/optimal_parameters.md) - Parameter tuning guide
- [exp/component_interactions.md](exp/component_interactions.md) - Architecture and data flows
- [src/core/helix_geometry.py](src/core/helix_geometry.py) - Helical geometry implementation
- [src/communication/central_post.py](src/communication/central_post.py) - Hub-spoke communication
- [src/agents/](src/agents/) - Agent management modules
- [src/memory/](src/memory/) - Memory systems
- [src/llm/](src/llm/) - LLM integration
- [src/pipeline/](src/pipeline/) - Processing pipelines