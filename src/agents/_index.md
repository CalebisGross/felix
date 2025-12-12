# Agents Module

## Purpose
Core agent system implementation providing base agent classes, LLM-enabled agents, specialized role-based agents, and dynamic confidence-based agent spawning capabilities.

## Key Files

### [agent.py](agent.py)
Base agent class and lifecycle management.
- **`Agent`**: Abstract base class for all agents with state management
- **`AgentState`**: Enum defining agent lifecycle states (IDLE, WORKING, COMPLETED, ERROR)

### [llm_agent.py](llm_agent.py)
LLM-enabled agents with position-aware prompting along the helix.
- **`LLMAgent`**: Extends Agent with LLM capabilities and helix position awareness
- **`LLMTask`**: Task structure for LLM operations
- **`LLMResult`**: Result wrapper for LLM responses

### [specialized_agents.py](specialized_agents.py)
Role-specific agent implementations for different workflow phases.
- **`ResearchAgent`**: Early-phase exploration agent (spawn time: 0.0-0.25)
- **`AnalysisAgent`**: Mid-phase processing agent (spawn time: 0.2-0.6)
- **`CriticAgent`**: Continuous validation agent (spawn time: 0.4-0.7)

### [dynamic_spawning.py](dynamic_spawning.py)
Confidence-based adaptive agent team management.
- **`ConfidenceMonitor`**: Tracks team confidence levels and triggers spawning
- **`ContentAnalyzer`**: Analyzes output quality and completeness
- **`DynamicSpawning`**: Main orchestrator for spawning decisions (threshold: 0.80)
- **`TeamSizeOptimizer`**: Determines optimal team size based on task complexity

### [base_specialized_agent.py](base_specialized_agent.py)
Plugin framework for extensible agent types.
- **`SpecializedAgentPlugin`**: Abstract base for agent plugins
- **`AgentMetadata`**: Agent capability and requirement metadata

### [agent_plugin_registry.py](agent_plugin_registry.py)
Registry for managing and discovering agent plugins.
- **`AgentPluginRegistry`**: Central registry for plugin management

### [system_agent.py](system_agent.py)
Agent for executing system commands with approval workflows.
- **`SystemAgent`**: Executes shell commands with trust management

### [standalone_agent.py](standalone_agent.py)
Independent agent operation outside main workflow.
- **`StandaloneAgent`**: Self-contained agent for isolated tasks (e.g., Knowledge Brain comprehension agents)

### [prompt_optimization.py](prompt_optimization.py)
Utilities for optimizing agent prompts based on performance.
- Functions for prompt refinement and A/B testing

### [builtin/](builtin/)
Directory containing built-in agent plugin implementations.

## Key Concepts

### Agent Lifecycle
Agents transition through states: IDLE → WORKING → COMPLETED (or ERROR)

### Helix Position Awareness
LLMAgents adapt their behavior (temperature, token budget, role) based on their position along the helix:
- Top (wide exploration): High temperature, broad investigation
- Bottom (narrow synthesis): Low temperature, focused output

### Dynamic Spawning
Agents spawn automatically when team confidence falls below threshold (0.80), with spawn timing determined by normalized time and role requirements.

### Plugin Architecture
New agent types can be added through the plugin system without modifying core code.

## Related Modules
- [communication/](../communication/) - Message passing and coordination via CentralPost (includes message_types.py defining message protocols)
- [core/](../core/) - HelixGeometry for agent positioning
- [llm/](../llm/) - LLM client integration for agent reasoning
- [workflows/](../workflows/) - High-level orchestration of agent teams
