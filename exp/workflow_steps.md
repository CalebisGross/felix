# Felix Framework Workflow Steps

This document provides a detailed, step-by-step breakdown of how the Felix framework operates, based on codebase analysis and component verification.

## 1. Initialization

### 1.1 Helix Geometry Setup
- Instantiate `HelixGeometry` with parameters: top_radius, bottom_radius, height, turns
- Example: `helix = HelixGeometry(top_radius=2.0, bottom_radius=0.5, height=10.0, turns=3)`
- Validates parameters and prepares parametric equations for agent positioning

### 1.2 Central Post and Memory Initialization
- Create `CentralPost` instance with configuration: max_agents, enable_metrics, enable_memory
- Initialize memory systems if enabled:
  - `KnowledgeStore` for persistent knowledge storage
  - `TaskMemory` for task pattern learning
  - `ContextCompressor` for memory optimization
- Set up message queuing and agent registration system

### 1.3 Agent Factory Setup
- Create `AgentFactory` with helix geometry and LLM client
- Configure dynamic spawning system with `DynamicSpawning`
- Set up confidence monitoring, content analysis, and team optimization

## 2. Agent Spawning and Lifecycle

### 2.1 Static Agent Creation
- Use `AgentFactory.create_research_agent()`, `create_analysis_agent()`, etc.
- Agents spawn at random times within type-specific ranges:
  - Research: 0.0-0.3 (early, broad exploration)
  - Analysis: 0.2-0.7 (mid-process, pattern recognition)
  - Synthesis: 0.7-0.95 (late, final integration)
  - Critic: 0.5-0.8 (ongoing quality assurance)

### 2.2 Agent Registration
- Agents register with `CentralPost.register_agent(agent)`
- Receive unique connection IDs for message routing
- Initialize agent state: WAITING → ACTIVE → COMPLETED

### 2.3 Dynamic Spawning
- `ConfidenceMonitor` tracks team confidence metrics
- `ContentAnalyzer` detects issues: contradictions, gaps, complexity
- `TeamSizeOptimizer` determines optimal team size based on task complexity
- New agents spawned via `AgentFactory.assess_team_needs()`

## 3. Communication Flow

### 3.1 Message Routing
- Agents send messages via `CentralPost.queue_message()`
- Messages use `Message` dataclass with type, sender, content, timestamp
- Message types: TASK_REQUEST, TASK_ASSIGNMENT, STATUS_UPDATE, TASK_COMPLETE, ERROR_REPORT

### 3.2 Hub-Spoke Architecture
- Central post acts as hub, agents as spokes
- O(N) complexity vs O(N²) mesh networks
- FIFO message processing with guaranteed ordering

### 3.3 Agent Communication
- Agents share results via `LLMAgent.share_result_to_central()`
- Context sharing through `shared_context` dictionary
- Influence relationships via `influence_agent_behavior()`

## 4. Task Processing

### 4.1 LLM Integration
- `LLMAgent` uses `LMStudioClient` for local LLM inference
- Position-aware prompting based on helix location
- Temperature adaptation: high at top (creative), low at bottom (focused)

### 4.2 Token Budget Management
- `TokenBudgetManager` allocates tokens per agent type and stage
- Adaptive compression when budgets are constrained
- Usage tracking and optimization

### 4.3 Processing Stages
- Agents progress through multiple processing stages
- Each stage uses different prompts and token allocations
- Confidence calculation based on position, content quality, and stage

## 5. Memory Interactions

### 5.1 Knowledge Storage
- Results stored in `KnowledgeStore` with confidence levels
- Persistent SQLite database with compression for large content
- Domain-based organization and tagging

### 5.2 Context Compression
- `ContextCompressor` reduces memory usage for large contexts
- Extractive and abstractive compression strategies
- Maintains key information while reducing token usage

### 5.3 Task Memory Learning
- `TaskMemory` learns patterns from task execution
- Strategy recommendations based on historical performance
- Success rate tracking and optimization

## 6. Dynamic Features

### 6.1 Confidence Monitoring
- Tracks team-wide confidence trends: improving, declining, stable
- Monitors volatility and agent type performance
- Triggers spawning when confidence drops below thresholds

### 6.2 Content Analysis
- Detects contradictions, knowledge gaps, high complexity
- Identifies missing domains and quality issues
- Suggests appropriate agent types for issues found

### 6.3 Prompt Optimization
- `PromptOptimizer` learns from execution metrics
- A/B testing framework for prompt variations
- Failure analysis for continuous improvement

## 7. Pipeline Integration

### 7.1 Linear Pipeline
- `LinearPipeline` provides baseline comparison
- Sequential agent progression through fixed stages
- Workload distribution analysis for hypothesis validation

### 7.2 Chunking and Streaming
- Large outputs processed in chunks via `ProgressiveProcessor`
- Streaming synthesis for complex content generation
- Chunk metadata tracking and reassembly

## 8. Hypothesis Validation

### 8.1 H1: Helical Progression Enhances Adaptation
- Helix geometry enables continuous evolution
- Position-based behavior adaptation verified
- Metrics collected for statistical analysis

### 8.2 H2: Multi-Agent Communication Optimizes Resources
- Hub-spoke efficiency vs mesh complexity
- Performance metrics: throughput, latency, overhead ratios
- Scaling analysis with increasing agent counts

### 8.3 H3: Memory Compression Reduces Latency
- Context compression effectiveness measurement
- Knowledge retrieval performance tracking
- Latency reduction validation

## 9. End-to-End Example Workflow

```python
# Pseudocode for basic Felix workflow
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent

# 1. Initialize core components
helix = HelixGeometry(2.0, 0.5, 10.0, 3)
central_post = CentralPost(max_agents=10, enable_memory=True)
agent_factory = AgentFactory(helix, llm_client)

# 2. Create initial agents
research_agent = agent_factory.create_research_agent(domain="technical")
analysis_agent = agent_factory.create_analysis_agent()
synthesis_agent = agent_factory.create_synthesis_agent()

# 3. Register agents
for agent in [research_agent, analysis_agent, synthesis_agent]:
    central_post.register_agent(agent)

# 4. Simulation loop
current_time = 0.0
while current_time < 1.0:
    # Update agent states
    for agent in [research_agent, analysis_agent, synthesis_agent]:
        if agent.can_spawn(current_time):
            agent.spawn(current_time, task)
            # Process task with LLM
            result = agent.process_task_with_llm(task, current_time)
            # Share result
            message = agent.share_result_to_central(result)
            central_post.queue_message(message)

    # Process messages
    while central_post.has_pending_messages():
        message = central_post.process_next_message()
        # Handle message routing and agent coordination

    current_time += 0.1

# 5. Final results retrieval
final_results = central_post.get_memory_summary()
```

## Key Workflow Patterns

1. **Initialization Pattern**: Helix → CentralPost → AgentFactory → Initial agents
2. **Spawning Pattern**: Confidence monitoring → Content analysis → Dynamic spawning
3. **Processing Pattern**: Task assignment → LLM processing → Result sharing → Memory storage
4. **Communication Pattern**: Agent → CentralPost → Message routing → Context sharing
5. **Optimization Pattern**: Metrics collection → Performance analysis → Prompt optimization
6. **Completion Pattern**: Confidence validation → Result acceptance → Final output generation

This workflow enables Felix to adaptively coordinate multiple agents along helical progression paths, optimizing for both individual agent performance and team-level coordination.