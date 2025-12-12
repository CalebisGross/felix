# Communication Module

## Purpose
Hub-spoke communication system providing O(N) message routing between agents (vs O(N²) mesh topology), with specialized coordinators for synthesis, web search, system commands, and streaming outputs.

## Key Files

### [central_post.py](central_post.py)
Core hub for agent communication and coordination.
- **`CentralPost`**: Main hub coordinator managing message routing, agent registry, synthesis, and specialized coordinators
- **`AgentRegistry`**: Phase-based agent tracking (exploration/analysis/synthesis) with team state queries
- **`AgentFactory`**: Creates agents with helix positioning and proper registration

### [message_types.py](message_types.py)
Message protocol definitions.
- **`Message`**: Standard message structure for inter-agent communication
- **`MessageType`**: Enum defining message categories (REQUEST, RESPONSE, BROADCAST, etc.)

### [spoke.py](spoke.py)
Communication endpoints for agents.
- **`Spoke`**: Agent-side communication interface to CentralPost
- **`DeliveryConfirmation`**: Acknowledgment structure for message delivery

### [synthesis_engine.py](synthesis_engine.py)
Combines agent outputs into coherent results.
- **`SynthesisEngine`**: Aggregates and synthesizes multiple agent responses with adaptive temperature (0.2-0.4) and token allocation (1500-3000) based on consensus

### [web_search_coordinator.py](web_search_coordinator.py)
Manages internet search requests from agents.
- **`WebSearchCoordinator`**: Routes search requests to WebSearchClient with caching and domain filtering

### [system_command_manager.py](system_command_manager.py)
Handles system command execution with approval workflows.
- **`SystemCommandManager`**: Manages shell command execution through approval pipeline and trust assessment

### [streaming_coordinator.py](streaming_coordinator.py)
Coordinates streaming outputs from agents and LLMs.
- **`StreamingCoordinator`**: Manages real-time token streaming with callbacks and time-batched delivery

### [memory_facade.py](memory_facade.py)
Provides memory access interface for agents.
- **`MemoryFacade`**: Simplified interface to KnowledgeStore and TaskMemory for agent queries

### [performance_monitor.py](performance_monitor.py)
Tracks communication and agent performance metrics.
- **`PerformanceMonitor`**: Collects metrics on message latency, agent performance, and system throughput

### [mesh.py](mesh.py)
Legacy mesh communication implementation for comparison.
- **`MeshCommunication`**: Direct peer-to-peer agent communication (O(N²) complexity)
- **`MeshConnection`**: Point-to-point connection management

## Key Concepts

### Hub-Spoke Architecture
- **O(N) Complexity**: Single hub routes all messages vs every agent connecting to every other
- **Scalability**: Supports 133 agents efficiently with message queuing
- **Central Synthesis**: Hub performs final synthesis rather than delegating to synthesis agent
- **Coordinator Pattern**: CentralPost delegates specialized concerns to focused subsystems

### Coordinator Architecture
CentralPost was refactored from a monolithic hub into a delegation-based architecture:
- **CentralPost**: Orchestrates coordinators, maintains agent registry, handles message routing
- **6 Specialized Coordinators**: Each handles a specific domain (synthesis, search, commands, streaming, memory, metrics)
- **Separation of Concerns**: Each coordinator has a single, well-defined responsibility
- **Maintainability**: Easier to test, modify, and extend individual coordinators
- **Message Types**: Extracted to separate module to avoid circular dependencies

### Agent Awareness
Agents can query team state through CentralPost:
- Discover peer agents by role or phase
- Check team confidence levels
- Coordinate collaborative tasks

### Adaptive Synthesis
CentralPost adjusts synthesis parameters based on:
- Agent consensus (higher consensus → lower temperature)
- Output complexity (more diverse outputs → more tokens)
- Team size and phase distribution

### Specialized Coordinators
CentralPost delegates specific concerns to specialized coordinators:
- SynthesisEngine: Result aggregation
- WebSearchCoordinator: Internet research
- SystemCommandManager: Command execution
- StreamingCoordinator: Real-time output delivery

## Related Modules
- [agents/](../agents/) - Agent implementations using hub-spoke communication
- [memory/](../memory/) - Memory systems accessed via MemoryFacade
- [llm/](../llm/) - WebSearchClient and LLM integration
- [execution/](../execution/) - System command execution backend
- [workflows/](../workflows/) - High-level orchestration using CentralPost
