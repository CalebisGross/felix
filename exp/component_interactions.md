# Felix Framework Component Interactions

This document visualizes how Felix framework components interact, showing data flow patterns and architectural relationships.

## Core Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Felix Multi-Agent System                      │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ HelixGeometry│───▶│  LLMAgent   │    │  AgentFactory│         │
│  │             │    │             │    │             │         │
│  │ - Parametric │    │ - Position  │    │ - Dynamic   │         │
│  │   equations │    │   awareness │    │   spawning  │         │
│  │ - Agent pos │    │ - LLM calls  │    │ - Confidence│         │
│  │ - Path calc │    │ - Token budg │    │   monitoring│         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                     │                     │          │
│         └─────────────────────┼─────────────────────┘          │
│                               │                                │
│                       ┌───────▼───────┐                        │
│                       │ CentralPost   │                        │
│                       │               │                        │
│                       │ - Hub-spoke   │                        │
│                       │   messaging   │                        │
│                       │ - Agent reg   │                        │
│                       │ - Message Q   │                        │
│                       └───────┬───────┘                        │
│                               │                                │
│         ┌─────────────────────┼─────────────────────┐         │
│         │                     │                     │         │
│  ┌──────▼──────┐    ┌─────────▼─────────┐    ┌─────▼─────┐   │
│  │ Knowledge   │    │   TaskMemory     │    │ Context    │   │
│  │   Store     │    │                  │    │ Compressor │   │
│  │             │    │ - Pattern learn  │    │            │   │
│  │ - Persistent│    │ - Success rates │    │ - Token     │   │
│  │   storage   │    │ - Strategy rec  │    │   reduction │   │
│  └─────────────┘    └──────────────────┘    └────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Dynamic Features                           │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │Confidence   │  │ Content     │  │ Prompt      │     │   │
│  │  │ Monitor     │  │ Analyzer    │  │ Optimizer   │     │   │
│  │  │             │  │             │  │             │     │   │
│  │  │ - Team perf │  │ - Issue det │  │ - A/B test  │     │   │
│  │  │ - Trend anal│  │ - Gap ident │  │ - Learning  │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Component Relationships

### 1. Agent Lifecycle Flow

```
Agent Creation → Registration → Task Assignment → Processing → Result Sharing → Completion

     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
     │ AgentFactory│────▶│CentralPost  │────▶│  LLMAgent   │
     │             │     │             │     │             │
     │ - Spawn     │     │ - Register  │     │ - Process   │
     │ - Configure │     │ - Route     │     │ - Position  │
     └─────────────┘     └─────────────┘     └─────────────┘
                                                       │
                                                       ▼
     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
     │  Message    │────▶│ Knowledge   │────▶│ TaskMemory  │
     │  Sharing    │     │   Store     │     │             │
     │             │     │             │     │ - Patterns  │
     │ - Results   │     │ - Persistent│     │ - Learning │
     └─────────────┘     └─────────────┘     └─────────────┘
```

### 2. Communication Patterns

#### Hub-Spoke Message Flow
```
Agents (Spokes) ↔ CentralPost (Hub) ↔ Memory Systems

Research Agent ──┐
                 │
Analysis Agent ──┼──▶ CentralPost ──▶ Knowledge Store
                 │                   ├─▶ Task Memory
Synthesis Agent ─┘                   └─▶ Context Compressor
```

#### Message Types and Routing
```
MessageType.TASK_REQUEST     → CentralPost → Task Assignment
MessageType.STATUS_UPDATE    → CentralPost → Confidence Monitor
MessageType.TASK_COMPLETE    → CentralPost → Memory Storage
MessageType.ERROR_REPORT     → CentralPost → Error Handling
```

### 3. Position-Aware Processing

#### Helix Geometry Integration
```
HelixGeometry.get_position(t) ──▶ LLMAgent.get_adaptive_temperature(t)
                                ├─▶ LLMAgent.create_position_aware_prompt()
                                └─▶ Agent positioning for collaboration
```

#### Agent Behavior Adaptation
```
Position (t=0.0 to 1.0) → Temperature Range → Creativity Level
     │
     ├── t < 0.3: High temperature (0.7-0.9) - Exploration
     ├── t < 0.7: Medium temperature (0.4-0.7) - Analysis
     └── t > 0.7: Low temperature (0.1-0.5) - Synthesis
```

### 4. Memory System Interactions

#### Knowledge Storage Flow
```
LLMAgent Result → CentralPost.store_agent_result_as_knowledge()
                   ↓
KnowledgeStore.store_knowledge() → SQLite Database
                   ↓
Persistent Storage with Confidence Levels and Domains
```

#### Context Management
```
Large Context → ContextCompressor.compress_context()
                ↓
Token Reduction → Improved LLM Efficiency
                ↓
Better Performance within Budget Constraints
```

### 5. Dynamic Spawning System

#### Confidence-Driven Spawning
```
Agent Messages → ConfidenceMonitor.record_confidence()
                ↓
Trend Analysis → Spawning Decision
                ↓
ContentAnalyzer.analyze_content() → Issue Detection
                ↓
TeamSizeOptimizer → Optimal Team Size
                ↓
AgentFactory.create_*_agent() → New Agent Spawn
```

#### Spawning Triggers
```
Low Confidence (< 0.7)     → Spawn Critic Agent
Content Contradictions     → Spawn Analysis Agent
Knowledge Gaps             → Spawn Research Agent
High Complexity            → Spawn Specialized Agents
Resource Constraints       → Adjust Token Budgets
```

### 6. Pipeline Integration

#### Linear Pipeline Comparison
```
Felix Agents (Helical) vs Linear Pipeline Stages

Felix (Dynamic):
Agent₁(t=0.1) ──▶ Agent₂(t=0.4) ──▶ Agent₃(t=0.8)
     │                │                │
     └─ Collaboration ┼─ Influence ───┼─ Final Output
                      │                │
               Confidence Monitoring  │
                               Dynamic Spawning

Linear Pipeline (Static):
Stage₁ ──▶ Stage₂ ──▶ Stage₃ ──▶ Stage₄ ──▶ Output
   │         │         │         │         │
   └─ Fixed ─┼─ Sequential ────┼─ Capacity ─┘
             │                 │
      No Dynamic Spawning      │
                        No Cross-Stage Communication
```

### 7. Optimization Systems

#### Prompt Optimization Loop
```
Prompt Execution → PromptMetricsTracker.record_metrics()
                   ↓
FailureAnalyzer.analyze_failure() → Pattern Detection
                   ↓
PromptTester.create_test() → A/B Testing
                   ↓
Improved Prompts → Better Performance
```

#### Token Budget Management
```
TokenBudgetManager.calculate_stage_allocation()
                   ↓
LLMAgent.create_position_aware_prompt() → Token Guidance
                   ↓
Efficient LLM Usage within Constraints
```

## Data Flow Tables

### Agent-to-Agent Communication

| Source Agent | Target Agent | Communication Type | Purpose |
|-------------|-------------|-------------------|---------|
| Research | Analysis | Shared Context | Provide raw data |
| Analysis | Synthesis | Structured Insights | Feed final synthesis |
| Critic | All | Quality Feedback | Improve output quality |
| Synthesis | CentralPost | Final Results | Complete workflow |

### Memory System Usage

| Component | Memory Type | Operation | Purpose |
|-----------|------------|-----------|---------|
| LLMAgent | KnowledgeStore | store_knowledge() | Persist results |
| CentralPost | TaskMemory | recommend_strategy() | Learning from patterns |
| LLMAgent | ContextCompressor | compress_context() | Token efficiency |
| ConfidenceMonitor | TaskMemory | Pattern analysis | Performance tracking |

### Dynamic Feature Triggers

| Trigger Condition | Component | Action | Result |
|------------------|-----------|--------|---------|
| Confidence < 0.7 | ConfidenceMonitor | Spawn critic agent | Quality improvement |
| Contradictions detected | ContentAnalyzer | Spawn analysis agent | Issue resolution |
| Knowledge gaps found | ContentAnalyzer | Spawn research agent | Gap filling |
| High complexity | TeamSizeOptimizer | Expand team | Better coverage |
| Token budget low | TokenBudgetManager | Compression mode | Resource efficiency |

## Key Interaction Patterns

1. **Initialization Pattern**: HelixGeometry → CentralPost → AgentFactory → Initial Agents
2. **Processing Pattern**: Task → LLMAgent → LLM Call → Result → CentralPost → Memory
3. **Communication Pattern**: Agent → Message → CentralPost → Routing → Target Agent
4. **Optimization Pattern**: Metrics → Analysis → Testing → Improved Prompts
5. **Spawning Pattern**: Monitoring → Analysis → Decision → New Agent Creation
6. **Completion Pattern**: Results → Validation → Storage → Final Output

This architecture enables Felix to coordinate multiple agents effectively along helical progression paths while maintaining efficient communication, memory management, and adaptive behavior.