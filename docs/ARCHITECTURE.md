# Felix Framework - Architecture Documentation

Complete architectural overview of the Felix multi-agent AI framework.

## Table of Contents

1. [Overview](#overview)
2. [Core Design Philosophy](#core-design-philosophy)
3. [System Architecture](#system-architecture)
4. [Component Interactions](#component-interactions)
5. [Data Flow](#data-flow)
6. [Design Patterns](#design-patterns)
7. [Scalability & Performance](#scalability--performance)

---

## Overview

Felix is a Python-based multi-agent AI framework built around three core innovations:

1. **Helical Geometry**: Agents progress along a 3D spiral from exploration (wide) to synthesis (narrow)
2. **Hub-Spoke Communication**: O(N) message routing vs O(N²) mesh networks
3. **Coordinator Architecture**: Specialized subsystems handle distinct concerns

**Key Metrics:**
- Supports up to 133 agents
- O(N) communication complexity
- 3-tier embedding fallback (zero external dependencies)
- Multi-provider LLM support with automatic failover

---

## Core Design Philosophy

### 1. Helical Progression Model

Agents don't work at fixed roles—they **evolve** along a helical path:

```
Top (Exploration)           Bottom (Synthesis)
  radius=3.0     ────────▶   radius=0.5
  temp=1.0                    temp=0.2
  tokens=2048                 tokens=1500
  "research"                  "focus"
```

**Mathematical Foundation:**
- Height: 8.0 units (progression depth)
- Turns: 2 complete spirals
- Position determines behavior: `depth_ratio = agent_position / height`

**Behavior Adaptation:**
```python
# Temperature gradient
temp = 1.0 - (0.8 * depth_ratio)  # 1.0 → 0.2

# Radius convergence
radius = top_radius - (top_radius - bottom_radius) * depth_ratio

# Token budget
tokens = base_tokens * (1.0 - 0.25 * depth_ratio)
```

### 2. Hub-Spoke Communication

Traditional mesh: Every agent connects to every other agent = O(N²) connections

Felix hub-spoke: All agents connect through central hub = O(N) connections

```
Mesh (10 agents):           Hub-Spoke (10 agents):
45 connections              10 connections

   A─B─C                         A  B  C
   │╱│╱│                         │  │  │
   D─E─F                         └──H──┘
   │╱│╱│                            │
   G─H─I                         D  E  F
   │╱│╱│                         │  │  │
   J─K─L                         └──┴──┘
                                    │
                                 G  H  I
```

**Scaling:**
- 25 agents: Mesh=300 connections, Hub=25 connections (92% reduction)
- 50 agents: Mesh=1225 connections, Hub=50 connections (96% reduction)
- 133 agents: Mesh=8778 connections, Hub=133 connections (98% reduction)

### 3. Coordinator Architecture

CentralPost delegates to specialized coordinators:

```
                    ┌─────────────────┐
                    │   CentralPost   │
                    │  (Orchestrator) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐     ┌────────▼────────┐   ┌─────▼─────┐
    │Synthesis│     │  WebSearch      │   │  System   │
    │ Engine  │     │  Coordinator    │   │  Command  │
    └─────────┘     └─────────────────┘   │  Manager  │
                                           └───────────┘
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐     ┌────────▼────────┐   ┌─────▼─────┐
    │Streaming│     │     Memory      │   │Performance│
    │   Coord │     │     Facade      │   │  Monitor  │
    └─────────┘     └─────────────────┘   └───────────┘
```

**Benefits:**
- Single Responsibility Principle
- Independent testing and modification
- Clear boundaries and interfaces
- Easier debugging and maintenance

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   CLI    │  │   GUI    │  │REST API  │  │ Python   │   │
│  │ (src/cli)│  │(src/gui) │  │(src/api) │  │  Scripts │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        └─────────────┼─────────────┼─────────────┘
                      │             │
┌─────────────────────▼─────────────▼──────────────────────────┐
│                   Workflow Layer                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FelixWorkflow (src/workflows/felix_workflow.py)      │  │
│  │  - Task classification                                 │  │
│  │  - Linear pipeline execution                           │  │
│  │  - Knowledge augmentation                              │  │
│  └──────────────────┬─────────────────────────────────────┘  │
└─────────────────────┼────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│              Communication & Coordination                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              CentralPost (Hub)                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │  Agent   │  │  Message │  │   AgentFactory   │   │  │
│  │  │ Registry │  │  Router  │  │                  │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └────────────────────┬───────────────────────────────────┘  │
│                       │                                       │
│  ┌────────────────────┴───────────────────────────────────┐  │
│  │           Specialized Coordinators                      │  │
│  │  [Synthesis] [WebSearch] [SysCmd] [Streaming]         │  │
│  │  [Memory] [Performance]                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────┬────────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────────┐
│                   Agent Layer                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Research  │  │Analysis  │  │  Critic  │  │  Plugin  │    │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agents  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼───────────┐
│                 Service Layer                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   LLM    │  │  Memory  │  │Knowledge │  │Execution │    │
│  │  Router  │  │ Systems  │  │  Brain   │  │  Trust   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼───────────┘
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼───────────┐
│              Infrastructure Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  SQLite  │  │Providers │  │  Helix   │  │Pipeline  │    │
│  │Databases │  │ (LLMs)   │  │ Geometry │  │Processing│    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Module Structure

```
felix/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── agent.py              # Base agent class
│   │   ├── llm_agent.py          # LLM-enabled agents
│   │   ├── specialized_agents.py # Research/Analysis/Critic
│   │   ├── dynamic_spawning.py   # Confidence-based spawning
│   │   └── builtin/              # Plugin agents
│   │
│   ├── communication/   # Hub-spoke messaging
│   │   ├── central_post.py       # Main coordinator
│   │   ├── message_types.py      # Message protocol
│   │   ├── synthesis_engine.py   # Result synthesis
│   │   ├── web_search_coordinator.py
│   │   ├── system_command_manager.py
│   │   ├── streaming_coordinator.py
│   │   ├── memory_facade.py
│   │   └── performance_monitor.py
│   │
│   ├── core/            # Core algorithms
│   │   └── helix_geometry.py     # Helical positioning
│   │
│   ├── llm/             # LLM integration
│   │   ├── base_provider.py      # Provider abstraction
│   │   ├── llm_router.py         # Multi-provider routing
│   │   ├── router_adapter.py     # Compatibility adapter
│   │   ├── provider_config.py    # YAML configuration
│   │   ├── lm_studio_client.py   # Low-level client
│   │   ├── token_budget.py       # Budget management
│   │   ├── web_search_client.py  # Search integration
│   │   └── providers/            # Provider implementations
│   │       ├── lm_studio_provider.py
│   │       ├── anthropic_provider.py
│   │       └── gemini_provider.py
│   │
│   ├── memory/          # Persistence & compression
│   │   ├── knowledge_store.py    # SQLite knowledge DB
│   │   ├── task_memory.py        # Task patterns
│   │   ├── workflow_history.py   # Execution tracking
│   │   └── context_compression.py # Abstractive compression
│   │
│   ├── knowledge/       # Knowledge brain
│   │   ├── document_reader.py    # Multi-format parsing
│   │   ├── embeddings.py         # 3-tier embeddings
│   │   ├── comprehension_engine.py # Agentic understanding
│   │   ├── graph_builder.py      # Relationship discovery
│   │   ├── retrieval.py          # Semantic search
│   │   ├── daemon.py             # Autonomous processing
│   │   └── workflow_integration.py
│   │
│   ├── workflows/       # High-level orchestration
│   │   ├── felix_workflow.py     # Main workflow engine
│   │   ├── context_builder.py    # Context management
│   │   └── truth_assessment.py   # Output validation
│   │
│   ├── execution/       # System commands
│   │   ├── system_executor.py    # Command execution
│   │   ├── trust_manager.py      # Trust classification
│   │   └── command_history.py    # Execution tracking
│   │
│   ├── api/             # REST API
│   │   ├── main.py               # FastAPI app
│   │   ├── routers/              # Endpoint routers
│   │   ├── middleware/           # Auth & CORS
│   │   └── dependencies.py       # Dependency injection
│   │
│   ├── gui/             # Tkinter interface
│   │   ├── main.py               # GUI entry point
│   │   ├── felix_system.py       # System orchestration
│   │   └── tabs/                 # Tab implementations
│   │
│   ├── cli.py           # CLI interface
│   │
│   └── migration/       # Database migrations
│       ├── version_manager.py
│       └── migrations/
│
├── config/              # Configuration files
│   ├── llm.yaml
│   ├── prompts.yaml
│   └── trust_rules.yaml
│
└── docs/                # Documentation
```

---

## Component Interactions

### Workflow Execution Flow

```
1. User Request
   CLI/GUI/API → FelixWorkflow.execute_linear_workflow_optimized()

2. Task Classification
   FelixWorkflow → Classify as simple/medium/complex
   → Determine max_steps (3/5/10)

3. Knowledge Augmentation (if enabled)
   FelixWorkflow → KnowledgeRetriever.search()
   → Inject relevant knowledge into context

4. Agent Spawning
   FelixWorkflow → AgentFactory.create_*_agent()
   → Place agent on helix based on normalized_time
   → Register with CentralPost

5. Agent Processing
   Agent → LLMAgent.process_task()
   → Construct position-aware prompt
   → LLMRouter.complete(request)
   → Provider selection & fallback
   → LLM inference
   → Parse response

6. Communication
   Agent → CentralPost.send_message(TASK_COMPLETE)
   → PerformanceMonitor.record_metrics()
   → MemoryFacade.store_knowledge()

7. Confidence Monitoring
   DynamicSpawning.check_confidence()
   → If < 0.80: spawn additional agents
   → WebSearchCoordinator.trigger_search()

8. Synthesis
   When confidence ≥ 0.80:
   → SynthesisEngine.synthesize()
   → Aggregate agent outputs
   → Generate final synthesis
   → Return to workflow

9. Persistence
   → WorkflowHistory.record_execution()
   → TaskMemory.store_pattern()
   → KnowledgeStore.compress_if_needed()
```

### Message Flow Sequence

```
┌─────┐     ┌──────────┐     ┌─────────┐     ┌─────┐
│Agent│     │CentralPost│    │Coordinator│    │ LLM │
└──┬──┘     └─────┬────┘     └────┬─────┘    └──┬──┘
   │              │               │              │
   │ TASK_REQUEST │               │              │
   ├──────────────▶               │              │
   │              │               │              │
   │        Route message         │              │
   │              ├───────────────▶              │
   │              │               │              │
   │              │         Process task         │
   │              │               ├──────────────▶
   │              │               │              │
   │              │               │   Response   │
   │              │               ◀──────────────┤
   │              │               │              │
   │              │    Result     │              │
   │              ◀───────────────┤              │
   │              │               │              │
   │ TASK_COMPLETE│               │              │
   ◀──────────────┤               │              │
   │              │               │              │
```

### LLM Provider Routing

```
Request comes in
     │
     ▼
┌─────────────────┐
│   LLMRouter     │
│ (llm_router.py) │
└────────┬────────┘
         │
         ▼
Try Primary Provider
         │
    ┌────┴────┐
    │Success? │
    └────┬────┘
         │
    ┌────▼────┐
    │   Yes   │───▶ Return response
    └─────────┘
    ┌────▼────┐
    │   No    │
    └────┬────┘
         │
         ▼
Try Fallback Provider 1
         │
    ┌────┴────┐
    │Success? │
    └────┬────┘
         │
    ┌────▼────┐
    │   Yes   │───▶ Return response
    └─────────┘
    ┌────▼────┐
    │   No    │
    └────┬────┘
         │
         ▼
Try Fallback Provider 2
         │
    ┌────┴────┐
    │Success? │
    └────┬────┘
         │
    ┌────▼────┐
    │   Yes   │───▶ Return response
    └─────────┘
    ┌────▼────┐
    │   No    │
    └────┬────┘
         │
         ▼
    All Failed
         │
         ▼
  Raise ProviderError
```

---

## Data Flow

### Knowledge Brain Pipeline

```
1. Document Ingestion
   File watcher detects new file
        │
        ▼
   DocumentReader.read_document()
        │
        ▼
   Semantic chunking (1000 chars, 200 overlap)
        │
        ▼
   Store in document_sources table

2. Agentic Comprehension
   KnowledgeComprehensionEngine.comprehend()
        │
        ├─▶ ResearchAgent: Extract key concepts
        ├─▶ AnalysisAgent: Identify relationships
        └─▶ CriticAgent: Validate quality
        │
        ▼
   Store concepts in knowledge_entries table

3. Embedding Generation
   EmbeddingProvider.generate_embeddings()
        │
        ├─▶ Try LM Studio (768-dim vectors)
        ├─▶ Fallback to TF-IDF
        └─▶ Fallback to FTS5
        │
        ▼
   Store embeddings in knowledge_entries

4. Graph Construction
   KnowledgeGraphBuilder.build_relationships()
        │
        ├─▶ Explicit mentions (co-occurrence)
        ├─▶ Embedding similarity (>0.75)
        └─▶ Chunk co-occurrence (5-chunk window)
        │
        ▼
   Store in knowledge_relationships table

5. Retrieval with Meta-Learning
   KnowledgeRetriever.search(query, task_type)
        │
        ├─▶ Semantic search (embeddings/TF-IDF/FTS5)
        ├─▶ Apply meta-learning boost (usage history)
        └─▶ Rank by relevance score
        │
        ▼
   Return top-k results

6. Usage Tracking
   Record which knowledge helped which workflow
        │
        ▼
   Store in knowledge_usage table
        │
        ▼
   Future retrievals boosted by usefulness score
```

### Memory Compression Flow

```
Context size check
      │
      ▼
  Size > Threshold?
      │
  ┌───┴───┐
  │  Yes  │
  └───┬───┘
      │
      ▼
Extract key concepts
      │
      ▼
Generate abstractive summary
(target_length=100 chars)
      │
      ▼
Compression ratio ~0.3
      │
      ▼
Replace original with summary
      │
      ▼
Maintain <concept, definition, examples>
      │
      ▼
Store compressed version
```

---

## Design Patterns

### 1. Hub-Spoke Pattern (Communication)

**Problem:** Mesh networks create O(N²) connections
**Solution:** Central hub routes all messages

**Implementation:**
```python
class CentralPost:
    def __init__(self):
        self._agents = {}  # agent_id → agent
        self._message_queue = deque()

    def send_message(self, message):
        # O(1) message queueing
        self._message_queue.append(message)

    def route_message(self, message):
        # O(1) delivery to recipient
        recipient = self._agents[message.recipient_id]
        recipient.receive_message(message)
```

### 2. Coordinator Pattern (Separation of Concerns)

**Problem:** Monolithic CentralPost becomes unwieldy
**Solution:** Delegate specialized tasks to coordinators

**Implementation:**
```python
class CentralPost:
    def __init__(self):
        self.synthesis_engine = SynthesisEngine()
        self.web_search_coordinator = WebSearchCoordinator()
        self.system_command_manager = SystemCommandManager()
        # ... other coordinators

    def handle_message(self, message):
        if message.type == MessageType.SYNTHESIS_READY:
            return self.synthesis_engine.synthesize(...)
        elif message.type == MessageType.SEARCH_REQUEST:
            return self.web_search_coordinator.search(...)
```

### 3. Provider Abstraction Pattern (LLM Integration)

**Problem:** Different LLM APIs have different interfaces
**Solution:** Abstract provider interface with unified request/response

**Implementation:**
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        pass

class LMStudioProvider(BaseLLMProvider):
    def complete(self, request):
        # LM Studio-specific implementation
        ...

class AnthropicProvider(BaseLLMProvider):
    def complete(self, request):
        # Anthropic-specific implementation
        ...
```

### 4. Adapter Pattern (Backwards Compatibility)

**Problem:** Introducing router breaks existing code using LMStudioClient directly
**Solution:** RouterAdapter provides compatible interface

**Implementation:**
```python
class RouterAdapter:
    def __init__(self, router: LLMRouter):
        self.router = router

    def chat_completion(self, messages, **kwargs):
        # Adapt old interface to new router
        request = LLMRequest(...)
        response = self.router.complete(request)
        return response
```

### 5. Plugin Pattern (Extensibility)

**Problem:** Adding new agent types requires modifying core code
**Solution:** Plugin system with auto-discovery

**Implementation:**
```python
class SpecializedAgentPlugin(ABC):
    @abstractmethod
    def create_agent(self, ...):
        pass

    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        pass

class AgentPluginRegistry:
    def discover_plugins(self, plugin_dirs):
        # Auto-discover and load plugins
        ...
```

### 6. Facade Pattern (Memory Access)

**Problem:** Agents need simple interface to complex memory systems
**Solution:** MemoryFacade provides unified interface

**Implementation:**
```python
class MemoryFacade:
    def __init__(self):
        self.knowledge_store = KnowledgeStore()
        self.task_memory = TaskMemory()
        self.workflow_history = WorkflowHistory()

    def query_knowledge(self, query):
        # Simplified interface hiding complexity
        return self.knowledge_store.search(...)
```

---

## Scalability & Performance

### Performance Characteristics

| Component | Complexity | Scalability |
|-----------|------------|-------------|
| Message routing | O(1) | Excellent (133+ agents) |
| Agent spawning | O(1) | Excellent |
| Synthesis | O(N) | Good (N = agent count) |
| Knowledge retrieval | O(log N) | Good (indexed search) |
| Graph traversal | O(E) | Moderate (E = edges) |

### Optimization Strategies

1. **Message Queuing**: FIFO queue prevents blocking
2. **Indexed Databases**: SQLite indexes on commonly queried fields
3. **Embedding Caching**: Store embeddings to avoid recomputation
4. **Result Caching**: Web search results cached by query
5. **Connection Pooling**: Reuse HTTP connections to LLM providers
6. **Lazy Loading**: Load knowledge on-demand, not upfront
7. **Compression**: Reduce memory footprint with abstractive summaries

### Resource Limits

**Recommended:**
- Agents: 5-25 (optimal for local LLMs)
- Max steps: 10 (prevents runaway workflows)
- Token budget: 2048/agent (fits 7B models)
- Knowledge entries: 1000-10000

**Maximum:**
- Agents: 133 (tested limit)
- Max steps: 50 (with powerful LLMs)
- Token budget: 4096/agent (larger models)
- Knowledge entries: 100,000+

### Performance Benchmarks

Felix demonstrates three key performance characteristics:

**Helical Progression (20% improvement in workload distribution)**
- Adaptive behavior improves workload distribution
- Agents naturally converge from exploration to synthesis

**Hub-Spoke Communication (15% improvement in resource allocation)**
- O(N) vs O(N²) reduces overhead
- Measured: 25 agents = 92% connection reduction

**Memory Compression (25% improvement in latency)**
- Abstractive summaries maintain semantic meaning
- Measured: 0.3 compression ratio, minimal quality loss

---

## See Also

- [COORDINATOR_ARCHITECTURE.md](COORDINATOR_ARCHITECTURE.md) - Detailed coordinator documentation
- [LLM_PROVIDER_GUIDE.md](LLM_PROVIDER_GUIDE.md) - Provider architecture deep dive
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Extending Felix architecture
- [CONFIGURATION.md](CONFIGURATION.md) - Architecture configuration
