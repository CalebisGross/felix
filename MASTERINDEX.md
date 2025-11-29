# Felix Framework - Master System Index

> **Version**: 1.0
> **Generated**: 2025-11-25
> **Framework**: Felix Multi-Agent AI Framework with Helical Geometry

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Agent System](#2-agent-system-srcagents)
3. [Communication Hub](#3-communication-hub-srccommunication)
4. [Workflow System](#4-workflow-system-srcworkflows)
5. [Memory Systems](#5-memory-systems-srcmemory)
6. [Knowledge Brain System](#6-knowledge-brain-system-srcknowledge)
7. [LLM Integration](#7-llm-integration-srcllm)
8. [GUI System](#8-gui-system-srcgui)
9. [REST API](#9-rest-api-srcapi)
10. [CLI Systems](#10-cli-systems)
11. [Configuration](#11-configuration-config)
12. [Utilities & Support Systems](#12-utilities--support-systems)
13. [Database Schemas](#13-database-schemas)
14. [Critical Thresholds & Constants](#14-critical-thresholds--constants)
15. [Integration Points](#15-integration-points)

---

## 1. Project Overview

Felix is a Python multi-agent AI framework that uses **helical geometry** for adaptive agent progression. Agents traverse a 3D helix (spiral path) from wide exploration at the top to narrow synthesis at the bottom, with behavior adapting based on position.

### Core Architecture Principles

- **Helical Progression**: Agents move along a helix with parameters:
  - Top radius: 3.0 (wide exploration)
  - Bottom radius: 0.5 (narrow synthesis)
  - Height: 8.0 (progression depth)
  - Turns: 2 (spiral complexity)

- **Hub-Spoke Communication**: O(N) CentralPost coordinator vs O(N²) mesh
- **Confidence-Based Synthesis**: Synthesis triggers at ≥0.80 team confidence
- **Zero External Dependencies**: Knowledge Brain operates with tiered fallbacks

### Directory Structure

```
felix/
├── src/
│   ├── agents/           # Agent system (base, specialized, plugins)
│   ├── communication/    # CentralPost, coordinators, message types
│   ├── workflows/        # Workflow execution, context building
│   ├── memory/           # Knowledge, task memory, history
│   ├── knowledge/        # Knowledge Brain (ingestion, embeddings, retrieval)
│   ├── llm/              # LLM providers, router, streaming
│   ├── gui/              # Tkinter GUI (9 tabs)
│   ├── api/              # FastAPI REST API
│   ├── cli.py            # Command-line interface
│   ├── cli_chat/         # Conversational CLI
│   ├── prompts/          # Prompt management
│   ├── execution/        # Trust, approval, command execution
│   ├── pipeline/         # Output chunking
│   ├── migration/        # Database migrations
│   └── utils/            # Utilities
├── config/               # YAML configuration files
├── tests/                # Test suite
└── docs/                 # Documentation
```

---

## 2. Agent System (`src/agents/`)

### 2.1 Base Agent (`agent.py`)

**Class: `Agent`** - Basic lifecycle management for autonomous agents

| Method | Description |
|--------|-------------|
| `__init__(agent_id, spawn_time, helix, velocity)` | Initialize with spawn timing |
| `can_spawn(current_time)` | Check if agent can activate |
| `spawn(current_time, task)` | Begin processing |
| `update_position(current_time)` | Move along helix with adaptive velocity |
| `get_position(current_time)` | Get 3D coordinates (x, y, z) on helix |
| `get_position_info(current_time)` | Get detailed position (radius, depth_ratio, progress) |
| `record_confidence(confidence)` | Track confidence for adaptive progression |
| `pause_for_duration(duration, current_time)` | Pause agent progression |
| `set_velocity_multiplier(velocity)` | Adjust speed (0.1-3.0 range) |
| `_adapt_velocity_from_confidence()` | Speed up/slow down based on confidence trend |
| `get_progression_info()` | Get velocity, acceleration, confidence history |

**Enum: `AgentState`**
- `WAITING` → `SPAWNING` → `ACTIVE` → `COMPLETED` | `FAILED`

**Helper Functions:**
- `generate_spawn_times(count, seed)` - Create random spawn times [0.0, 1.0]
- `create_agents_from_spawn_times(spawn_times, helix)` - Batch agent creation
- `create_openscad_agents(helix, number_of_nodes, random_seed)` - OpenSCAD-compatible (133 nodes, seed 42069)

---

### 2.2 LLM Agent (`llm_agent.py`)

**Class: `LLMAgent(Agent)`** - LLM-powered agent with language model capabilities

#### Core Methods

| Method | Description |
|--------|-------------|
| `get_adaptive_temperature(current_time)` | Temperature gradient: 1.0 (exploration) → 0.2 (synthesis) |
| `calculate_confidence(current_time, content, stage, task)` | Complex confidence scoring with bonuses |
| `_analyze_content_quality(content)` | Score based on length, structure, depth (0.0-1.0) |
| `_calculate_consistency_bonus()` | Score based on confidence variance |
| `_calculate_collaborative_bonus(task)` | Rewards leveraging context (0.0-1.0) |
| `_learn_from_token_usage(allocated, used)` | Adaptive token budget adjustment |
| `get_token_efficiency()` | Recent efficiency ratio (used/allocated) |
| `create_position_aware_prompt(task, current_time)` | Generate (system_prompt, token_budget) |
| `process_task_with_llm(task, current_time, central_post, enable_streaming)` | Execute LLM task |
| `share_result_to_central(result)` | Send result to CentralPost |
| `receive_shared_context(message)` | Receive messages from other agents |
| `process_synthesis_feedback(feedback_message)` | Learn from synthesis integration |
| `influence_agent_behavior(other_agent, influence_type, strength)` | Affect other agents |
| `assess_collaboration_opportunities(available_agents, current_time)` | Find collaboration opportunities |
| `request_action(command, context)` | Request system command execution |

#### Helical Checkpoint System

```python
HELICAL_CHECKPOINTS = [0.0, 0.3, 0.5, 0.7, 0.9]
```

| Method | Description |
|--------|-------------|
| `should_process_at_checkpoint(current_time)` | Check if crossed new checkpoint |
| `get_current_checkpoint()` | Get current checkpoint value |
| `mark_checkpoint_processed()` | Record checkpoint processing |

#### Data Classes

**`LLMTask`**
- `task_id`, `description`, `context`, `metadata`
- `context_history`, `knowledge_entries`, `tool_instructions`

**`LLMResult`**
- `agent_id`, `task_id`, `content`, `position_info`
- `llm_response`, `processing_time`, `timestamp`, `confidence`
- `processing_stage`, `system_prompt`, `user_prompt`
- `temperature_used`, `token_budget_allocated`
- `collaborative_context_count`, `is_chunked`, `chunk_results`

#### Agent Type Parameters

| Type | Temperature Range | Max Tokens | Confidence Range |
|------|-------------------|------------|------------------|
| Research | 0.4-0.9 | 16,000 | 0.3-0.6 |
| Analysis | 0.2-0.7 | 16,000 | 0.4-0.8 |
| Synthesis | 0.1-0.5 | 20,000 | 0.6-0.95 |
| Critic | 0.1-0.6 | 16,000 | 0.5-0.8 |

---

### 2.3 Specialized Agents (`specialized_agents.py`)

#### ResearchAgent(LLMAgent)
- **Purpose**: Broad information gathering and exploration
- **Domain**: general/technical/creative
- **Special Features**:
  - "Direct answer mode" for simple queries (≥0.85 trust score)
  - Direct mode: 200 tokens, 0.2 temperature, 1-2 sentence max
  - Execution directive patterns: `WEB_SEARCH_NEEDED`, `SYSTEM_ACTION_NEEDED`

#### AnalysisAgent(LLMAgent)
- **Purpose**: Processing and organizing information
- **Type**: general/technical/critical
- **Tracking**: `identified_patterns`, `key_insights`

#### CriticAgent(LLMAgent)
- **Purpose**: Quality assurance and review
- **Focus**: accuracy/completeness/style/logic
- **Special Methods**:

| Method | Description |
|--------|-------------|
| `evaluate_reasoning_process(agent_output, agent_metadata)` | Evaluate HOW agents reasoned |

**Returns**: `reasoning_quality_score`, `logical_coherence`, `evidence_quality`, `methodology_appropriateness`, `improvement_recommendations`, `re_evaluation_needed`

#### Team Creation Functions

```python
create_specialized_team(helix, llm_client, task_complexity, ...)
```

| Complexity | Team Composition |
|------------|------------------|
| simple | 1 research + 1 analysis |
| medium | 2 research + 2 analysis + 1 critic |
| complex | 3 research + 3 analysis + 2 critics |

---

### 2.4 Dynamic Spawning (`dynamic_spawning.py`)

**Enum: `ConfidenceTrend`**
- `IMPROVING`, `DECLINING`, `STABLE`, `VOLATILE`

**Enum: `ContentIssue`**
- `CONTRADICTION`, `KNOWLEDGE_GAP`, `HIGH_COMPLEXITY`, `LOW_QUALITY`, `MISSING_DOMAIN`, `INSUFFICIENT_ANALYSIS`

**Data Classes:**

| Class | Fields |
|-------|--------|
| `ConfidenceMetrics` | current_average, trend, volatility, time_window_minutes, agent_type_breakdown, position_breakdown |
| `ContentAnalysis` | detected_issues, complexity_score, contradiction_count, gap_domains, quality_score, suggested_agent_types |
| `SpawningDecision` | should_spawn, agent_type, spawn_parameters, priority_score, reasoning |

**Class: `ConfidenceMonitor`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.8 | Minimum for synthesis |
| `volatility_threshold` | 0.15 | Trigger spawning above |
| `time_window_minutes` | 5.0 | Rolling window |

---

### 2.5 Plugin System

#### Base Plugin Interface (`base_specialized_agent.py`)

**Data Class: `AgentMetadata`**
```python
agent_type: str           # Unique identifier
display_name: str         # Human-readable name
description: str          # Purpose description
spawn_range: Tuple[float, float]  # Default (0.0, 1.0)
capabilities: List[str]   # e.g., "web_search", "code_analysis"
tags: List[str]           # Classification tags
default_tokens: int       # Default 800
version: str              # "1.0.0"
priority: int             # Higher = spawn earlier
```

**Abstract Class: `SpecializedAgentPlugin`**

| Method | Description |
|--------|-------------|
| `get_metadata()` → AgentMetadata | Describe agent capabilities |
| `create_agent(...)` → LLMAgent | Instantiate agent |
| `supports_task(task_description, task_metadata)` → bool | Task filtering (default: True) |
| `get_spawn_ranges_by_complexity()` → Dict | Custom spawn ranges |

#### Plugin Registry (`agent_plugin_registry.py`)

**Class: `AgentPluginRegistry`**

| Method | Description |
|--------|-------------|
| `discover_builtin_plugins()` | Load from `src/agents/builtin/` |
| `add_plugin_directory(path)` | Add external plugin source |
| `discover_plugins_in_directory(directory)` | Scan for plugins |
| `list_agent_types()` | Get all registered types |
| `get_agent_metadata(agent_type)` | Get agent metadata |
| `create_agent(agent_type, ...)` | Instantiate agent |
| `get_agents_for_task(task_description, complexity)` | Match agents to tasks |
| `reload_external_plugins()` | Hot-reload custom agents |

#### Built-in Plugins (`src/agents/builtin/`)
- `research_plugin.py` - ResearchAgent wrapper
- `analysis_plugin.py` - AnalysisAgent wrapper
- `critic_plugin.py` - CriticAgent wrapper

---

## 3. Communication Hub (`src/communication/`)

### 3.1 Message Types (`message_types.py`)

**Enum: `MessageType`** (18+ types)

| Category | Types |
|----------|-------|
| Basic | `TASK_REQUEST`, `TASK_ASSIGNMENT`, `STATUS_UPDATE`, `TASK_COMPLETE`, `ERROR_REPORT` |
| Phase-aware | `PHASE_ANNOUNCE`, `CONVERGENCE_SIGNAL`, `COLLABORATION_REQUEST`, `SYNTHESIS_READY`, `AGENT_QUERY`, `AGENT_DISCOVERY` |
| System Actions | `SYSTEM_ACTION_REQUEST`, `SYSTEM_ACTION_RESULT`, `SYSTEM_ACTION_APPROVAL_NEEDED`, `SYSTEM_ACTION_DENIED`, `SYSTEM_ACTION_START`, `SYSTEM_ACTION_OUTPUT`, `SYSTEM_ACTION_COMPLETE` |
| Feedback | `SYNTHESIS_FEEDBACK`, `CONTRIBUTION_EVALUATION`, `IMPROVEMENT_REQUEST` |

**Data Class: `Message`**
```python
sender_id: str
message_type: MessageType
content: Dict[str, Any]
timestamp: float
message_id: str  # UUID, auto-generated
```

---

### 3.2 Central Post (`central_post.py`)

**Class: `AgentRegistry`** - Phase-based agent tracking

| Phase | Position Range | Description |
|-------|----------------|-------------|
| exploration | 0.0-0.3 | Early information gathering |
| analysis | 0.3-0.7 | Processing and organizing |
| synthesis | 0.7-1.0 | Final integration |

| Method | Description |
|--------|-------------|
| `register_agent(agent_id, metadata)` | Register with initial metadata |
| `update_agent_position(agent_id, position_info)` | Track helix position |
| `update_agent_performance(agent_id, metrics)` | Record performance |
| `record_collaboration(agent_id, influenced_agent_id)` | Track relationships |
| `get_agents_in_phase(phase)` | Get agents by phase |
| `get_nearest_agents(agent_id, distance_threshold)` | Proximity query |
| `get_collaboration_graph()` | Get collaboration network |
| `calculate_team_statistics()` | Aggregate team metrics |

**Class: `CentralPost`** - Main coordination hub (O(N) hub-spoke)

| Method | Description |
|--------|-------------|
| `register_agent(agent_id, metadata)` | Add agent to registry |
| `process_agent_message(message)` | Route and process communications |
| `queue_message(message)` | Add to FIFO queue |
| `broadcast_to_agents(message_type, content)` | Send to all agents |
| `send_message_to_agent(agent_id, message)` | Targeted message |
| `receive_partial_thought(...)` | Handle streaming |
| `finalize_streaming_thought(...)` | Complete streaming |
| `handle_system_action_request(...)` | Process command requests |
| `broadcast_synthesis_feedback()` | Share synthesis results |
| `perform_synthesis(current_time)` | Execute final synthesis (≥0.80 confidence) |
| `get_team_statistics()` | Get team metrics |
| `get_agent_awareness_info(agent_id, query_type)` | Agent discovery |

**Subcomponents:**
- `synthesis_engine`: SynthesisEngine
- `web_search_coordinator`: WebSearchCoordinator
- `system_command_manager`: SystemCommandManager
- `memory_facade`: MemoryFacade
- `streaming_coordinator`: StreamingCoordinator
- `performance_monitor`: PerformanceMonitor
- `agent_registry`: AgentRegistry

---

### 3.3 Synthesis Engine (`synthesis_engine.py`)

**Class: `SynthesisEngine`** - Final output synthesis

| Method | Description |
|--------|-------------|
| `classify_task_complexity(task, messages)` | Returns: `SIMPLE_FACTUAL` \| `MEDIUM` \| `COMPLEX` |
| `classify_tool_requirements(task)` | Returns: `{file_operations, web_search, system_commands}` |
| `synthesize(agent_outputs, prompt, current_time)` | Generate final synthesis |
| `calculate_adaptive_parameters(confidence, complexity)` | Returns: (temperature, tokens) |
| `build_synthesis_prompt(task, outputs, context)` | Construct synthesis prompt |
| `extract_validation_flags(content)` | Extract validation dict |

**Configuration Files:**
- `config/task_complexity_patterns.yaml`
- `config/tool_requirements_patterns.yaml`

**Synthesis Parameters:**
- Temperature: Adaptive based on confidence and complexity
- Token budget: 1500-3000 tokens

---

### 3.4 Web Search Coordinator (`web_search_coordinator.py`)

**Class: `WebSearchCoordinator`**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.7 | Trigger search below this |
| `search_cooldown` | 10.0 | Seconds between searches |
| `min_samples` | 3 | Samples before triggering |

| Method | Description |
|--------|-------------|
| `set_task_context(task, workflow_id)` | Set current task |
| `update_confidence(confidence)` | Record confidence sample |
| `should_trigger_search()` | Check thresholds |
| `formulate_query(task, messages)` | Generate search query |
| `perform_search(query)` | Execute search |
| `extract_information(results, llm)` | Extract data from results |
| `store_search_results(results, store)` | Persist for retrieval |

**Features:**
- Confidence-based automatic triggering
- Query deduplication
- Domain filtering
- Result quality control

---

### 3.5 System Command Manager (`system_command_manager.py`)

**Class: `SystemCommandManager`**

| Method | Description |
|--------|-------------|
| `request_system_action(agent_id, command, context, workflow_id, cwd)` | Request execution |
| `classify_command_trust(command)` | Returns: `TrustLevel` (BLOCKED/SAFE/REVIEW) |
| `execute_command(command, context, agent_id)` | Execute with safety |
| `handle_approval_workflow(action_id, approval_id)` | Wait for user approval |
| `broadcast_command_events(action_id, result)` | Stream output to Terminal |
| `_dedup_workflow_commands(workflow_id, command_hash)` | Prevent duplicates |

**Trust Levels:**
| Level | Description | Example Commands |
|-------|-------------|------------------|
| BLOCKED | Never execute | `rm -rf`, `sudo`, `dd` |
| SAFE | Auto-execute | `ls`, `pwd`, `cat`, `git status` |
| REVIEW | Require approval | `pip install`, `git commit`, `mkdir` |

**Approval Decisions:**
- Auto-approve
- Deny
- Approve once
- Approve workflow (session-scoped)
- Approve all (same command type)

---

### 3.6 Memory Facade (`memory_facade.py`)

**Class: `MemoryFacade`** - Unified interface to memory systems

| Method | Description |
|--------|-------------|
| `store_agent_result_as_knowledge(...)` | Store as knowledge entry |
| `retrieve_knowledge(query)` | Query knowledge store |
| `get_task_strategy_recommendation(task, complexity)` | Strategy suggestions |
| `compress_context(text, strategy)` | Apply compression |
| `get_memory_status()` | Entry counts, storage size |

**Confidence Mapping:**
- HIGH: ≥0.8
- MEDIUM: ≥0.6
- LOW: <0.6

---

### 3.7 Streaming Coordinator (`streaming_coordinator.py`)

**Class: `StreamingCoordinator`** - Real-time streaming of agent thoughts

| Method | Description |
|--------|-------------|
| `receive_partial_thought(agent_id, partial, accumulated, progress, metadata)` | Time-batched chunks |
| `finalize_streaming(agent_id)` | Mark complete |
| `register_streaming_callback(callback)` | Subscribe to events |
| `get_partial_thought(agent_id)` | Get current partial |
| `get_streaming_status(agent_id)` | Get status dict |

**Metadata Tracked:**
- `agent_type`, `checkpoint`, `tokens_so_far`, `position_info`

---

### 3.8 Performance Monitor (`performance_monitor.py`)

**Class: `PerformanceMonitor`**

| Method | Description |
|--------|-------------|
| `increment_message_count(count)` | Track processed messages |
| `record_processing_time(time)` | Log latency |
| `record_overhead_ratio(ratio)` | Track O(N) efficiency |
| `record_scaling_metric(agent_count, time)` | Scaling analysis |
| `get_performance_summary()` | Get metrics dict |
| `get_throughput()` | Messages per second |
| `get_average_overhead()` | Communication efficiency |

---

## 4. Workflow System (`src/workflows/`)

### 4.1 Felix Workflow (`felix_workflow.py`)

**Main Function:**
```python
run_felix_workflow(
    felix_system,
    task_input,
    progress_callback=None,
    max_steps_override=None,
    parent_workflow_id=None
) → Dict
```

**Workflow Phases:**
1. Task analysis and complexity classification
2. Team formation (adaptive agent spawning)
3. Agent processing with helical progression
4. Continuous confidence monitoring
5. Web search coordination (confidence-triggered)
6. Dynamic agent spawning (confidence-based)
7. Synthesis when confidence ≥ 0.80
8. Knowledge persistence (file discoveries, patterns)

**Helper Functions:**
- `_map_synthesis_complexity_to_task_complexity(complexity)` → TaskComplexity
- `_store_file_discoveries(result, store, agent_id, task_id)` → count

---

### 4.2 Context Builder (`context_builder.py`)

**Data Class: `EnrichedContext`**
```python
task_description: str
context_history: List[Dict[str, Any]]
original_context_size: int
compressed_context_size: int
compression_ratio: float
knowledge_entries: List[Any]
message_count: int
tool_instructions: str           # Conditional injection
tool_instruction_ids: List[str]  # For meta-learning
context_inventory: str           # Resource checklist
existing_concepts: str           # Terminology consistency
version: int                     # Race-free synchronization
```

**Class: `CollaborativeContextBuilder`**

| Method | Description |
|--------|-------------|
| `build_agent_context(agent_type, current_time, depth_ratio)` | Build EnrichedContext |
| `build_context_inventory()` | Explicit resource checklist |
| `_get_knowledge_limit_for_complexity(complexity)` | Get knowledge entry limit |
| `_get_known_file_locations(task)` | Meta-learning file discovery |
| `_compress_context_if_needed(history)` | Apply compression |
| `integrate_concept_registry()` | Add existing concepts |

**Knowledge Limits by Complexity:**
| Complexity | Max Entries |
|------------|-------------|
| SIMPLE_FACTUAL | 8 |
| MEDIUM | 15 |
| COMPLEX | 25 |

---

### 4.3 Concept Registry (`concept_registry.py`)

**Data Class: `ConceptDefinition`**
```python
name: str
definition: str
source_agent: str
confidence: float
timestamp: float
related_concepts: List[str]
usage_count: int
```

**Data Class: `ConceptConflict`**
```python
concept_name: str
definition1: ConceptDefinition
definition2: ConceptDefinition
conflict_type: str  # 'duplicate', 'contradictory', 'overlapping'
severity: float     # 0.0-1.0
```

**Class: `ConceptRegistry`** - Workflow-scoped terminology consistency

| Method | Description |
|--------|-------------|
| `register_concept(name, definition, source_agent, confidence)` | Register concept |
| `get_concept(name)` | Retrieve concept |
| `query_related_concepts(name)` | Get related concepts |
| `detect_conflicts()` | Find conflicting definitions |
| `export_to_markdown()` | Export for analysis |

---

### 4.4 Context Relevance Evaluator (`context_relevance.py`)

**Data Class: `RelevanceScore`**
```python
score: float          # 0.0-1.0
reason: str
keywords_matched: List[str]
category: str         # 'highly_relevant', 'somewhat_relevant', 'irrelevant'
```

**Class: `ContextRelevanceEvaluator`**

| Method | Description |
|--------|-------------|
| `evaluate_relevance(fact, task_context)` | Score relevance |
| `detect_context_shift(previous, current)` | Detect topic change |

**Relevance Categories:**
| Threshold | Category |
|-----------|----------|
| ≥0.7 | highly_relevant |
| ≥0.4 | somewhat_relevant |
| <0.4 | irrelevant |

---

## 5. Memory Systems (`src/memory/`)

### 5.1 Knowledge Store (`knowledge_store.py`)

**Enums:**

| Enum | Values |
|------|--------|
| `KnowledgeType` | TASK_RESULT, AGENT_INSIGHT, PATTERN_RECOGNITION, FAILURE_ANALYSIS, OPTIMIZATION_DATA, DOMAIN_EXPERTISE, TOOL_INSTRUCTION, FILE_LOCATION |
| `ConfidenceLevel` | LOW, MEDIUM, HIGH, VERIFIED |

**Data Classes:**
- `KnowledgeEntry` - Single knowledge item
- `KnowledgeQuery` - Query structure with filters

**Class: `KnowledgeStore`**

| Method | Description |
|--------|-------------|
| `store_knowledge()` | Insert/replace with validation & deduplication |
| `retrieve_knowledge()` | Query with type/domain/confidence/time filters |
| `_apply_meta_learning_boost()` | Re-rank by historical usefulness |
| `record_knowledge_usage()` | Track helpful knowledge (0.0-1.0 score) |
| `update_knowledge_entry()` | Update existing entry |
| `merge_knowledge_entries()` | Combine multiple entries |
| `delete_knowledge()` | Remove with relationship cleanup |
| `get_knowledge_summary()` | Statistics by type/domain/confidence |
| `cleanup_old_entries()` | Delete old/low-performing entries |
| `delete_documents_by_pattern()` | Pattern-based deletion with cascade |
| `delete_orphaned_entries()` | Remove entries missing source documents |
| `advanced_search()` | Multi-field search (content, domain, tags, AND/OR) |
| `get_analytics_data()` | Comprehensive analytics |
| `generate_quality_report()` | Identify data quality issues |
| `add_watch_directory()` / `remove_watch_directory()` | Directory management |
| `transaction()` | Context manager for safe transactions |

**Meta-Learning:**
- Requires ≥2 historical samples for reliable boost
- Boost factor: 0.7-1.0 multiplier

---

### 5.2 Task Memory (`task_memory.py`)

**Enums:**

| Enum | Values |
|------|--------|
| `TaskOutcome` | SUCCESS, PARTIAL_SUCCESS, FAILURE, TIMEOUT, ERROR |
| `TaskComplexity` | SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX |

**Data Classes:**
- `TaskPattern` - Pattern metadata from execution history
- `TaskExecution` - Record of single task run
- `TaskMemoryQuery` - Query structure

**Class: `TaskMemory`**

| Method | Description |
|--------|-------------|
| `record_task_execution()` | Log task run, auto-update patterns |
| `get_patterns()` | Query by type/complexity/keywords/success_rate |
| `recommend_strategy()` | Suggest optimal strategies/agents/duration |
| `get_memory_summary()` | Statistics (patterns, executions, outcomes) |
| `cleanup_old_patterns()` | Remove unused patterns |

**Pattern Matching:**
- Keyword extraction: 3+ character words, filtered by stopwords
- Match threshold: 50% keyword overlap

---

### 5.3 Workflow History (`workflow_history.py`)

**Data Class: `WorkflowOutput`**

**Class: `WorkflowHistory`**

| Method | Description |
|--------|-------------|
| `save_workflow_output()` | Store with optional conversation threading |
| `get_workflow_outputs()` | Retrieve with status filtering/pagination |
| `get_workflow_by_id()` | Fetch single workflow |
| `search_workflows()` | Keyword search |
| `get_conversation_thread()` | Get all workflows in thread |
| `get_parent_workflow()` | Get parent workflow |
| `delete_workflow()` | Remove workflow |

---

### 5.4 Context Compression (`context_compression.py`)

**Enums:**

| Enum | Values |
|------|--------|
| `CompressionStrategy` | EXTRACTIVE_SUMMARY, ABSTRACTIVE_SUMMARY, KEYWORD_EXTRACTION, HIERARCHICAL_SUMMARY, RELEVANCE_FILTERING, PROGRESSIVE_REFINEMENT |
| `CompressionLevel` | LIGHT (80%), MODERATE (60%), HEAVY (40%), EXTREME (20%) |

**Class: `ContextCompressor`**

| Method | Description |
|--------|-------------|
| `compress_context()` | Apply compression strategy |
| `_extractive_summary()` | Keep top 1/3 sentences by importance |
| `_abstractive_summary()` | Create summaries |
| `_keyword_extraction()` | Keep high keyword-density sentences |
| `_hierarchical_summary()` | 3-level structure (core, supporting, auxiliary) |
| `_relevance_filtering()` | Filter by relevance threshold |
| `_progressive_refinement()` | Multi-pass compression |
| `get_compression_stats()` | Configuration info |

---

### 5.5 Audit Log (`audit_log.py`)

**Class: `AuditLogger`**

| Method | Description |
|--------|-------------|
| `log_operation()` | Record operation with before/after state |
| `get_audit_history()` | Query with filters |
| `get_entry_history()` | Full history for single entry |
| `get_recent_changes()` | Recent changes in last N hours |
| `get_statistics()` | Count by operation/user_agent |
| `cleanup_old_logs()` | Remove old logs |
| `export_to_csv()` | Export audit trail |

**Decorator: `@audit_logged(operation, user_agent)`**

---

### 5.6 Agent Performance Tracker (`agent_performance_tracker.py`)

**Class: `AgentPerformanceTracker`**

| Method | Description |
|--------|-------------|
| `record_agent_checkpoint()` | Record performance at checkpoint |
| `get_agent_performance_history()` | Performance timeline |
| `get_workflow_agent_summary()` | Aggregate workflow metrics |
| `get_agent_type_statistics()` | Stats for agent type |
| `get_phase_transition_analysis()` | Phase transition patterns |
| `cleanup_old_records()` | Remove old records |

---

## 6. Knowledge Brain System (`src/knowledge/`)

### 6.1 Document Reader (`document_ingest.py`)

**Enums:**

| Enum | Values |
|------|--------|
| `DocumentType` | PDF, TEXT, MARKDOWN, PYTHON, JAVASCRIPT, JAVA, CPP, C, UNKNOWN |
| `ChunkingStrategy` | FIXED_SIZE, PARAGRAPH, SECTION, SEMANTIC |

**Data Classes:**
- `DocumentMetadata` - File metadata (type, size, hash, pages, title, author)
- `DocumentChunk` - Single chunk with position/page/section metadata
- `IngestionResult` - Ingestion outcome with chunks and metadata

**Class: `DocumentReader`**

| Method | Description |
|--------|-------------|
| `ingest_document()` | Read → extract metadata → chunk → return result |
| `_extract_metadata()` | Get file hash, size, dates |
| `_read_pdf()` | PyPDF2-based PDF reading |
| `_read_text()` | Text/Markdown with encoding detection |
| `_read_code()` | Code file reading (preserves structure) |
| `_chunk_content()` | Apply chunking strategy |

**Default Parameters:**
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Strategy: SEMANTIC

---

### 6.2 Embedding Provider (`embeddings.py`)

**Enum: `EmbeddingTier`**
| Tier | Quality | Requirement |
|------|---------|-------------|
| LM_STUDIO | Best | LM Studio running |
| TFIDF | Good | numpy |
| FTS5 | Always works | SQLite |

**Classes:**
- `LMStudioEmbedder` - 768-dim via `/v1/embeddings`
- `TFIDFEmbedder` - Pure Python, 768-dim configurable
- `FTS5Embedder` - SQLite FTS5 BM25

**Class: `EmbeddingProvider`** - Unified interface with automatic fallback

| Method | Description |
|--------|-------------|
| `embed()` | Generate embedding (auto-selects tier) |
| `embed_batch()` | Batch embeddings |
| `get_tier_info()` | Current active tier |
| `similarity()` | Cosine distance |
| `serialize_embedding()` / `deserialize_embedding()` | Storage |

---

### 6.3 Comprehension Engine (`comprehension.py`)

**Data Classes:**
- `ConceptExtraction` - Extracted concept with definition, examples, confidence
- `EntityExtraction` - Extracted entity with type, description, mentions
- `ComprehensionResult` - Summary, concepts, entities, key_points, quality_score

**Class: `KnowledgeComprehensionEngine`** - Three-stage agentic pipeline

| Stage | Agent | Temperature | Max Tokens |
|-------|-------|-------------|------------|
| 1 | Research | 0.7 | 1500 |
| 2 | Analysis | 0.5 | 1500 |
| 3 | Critic | 0.3 | 1000 |

| Method | Description |
|--------|-------------|
| `comprehend_chunk()` | Run full 3-stage pipeline |
| `_research_chunk()` | Read and summarize |
| `_analyze_chunk()` | Extract concepts/entities |
| `_critique_chunk()` | Validate quality |

**Quality Threshold:** 0.6 (min_quality_threshold)
**Max Retries:** 2

---

### 6.4 Knowledge Graph Builder (`graph_builder.py`)

**Data Classes:**
- `ConceptNode` - knowledge_id, concept_name, domain, related_ids, embedding
- `RelationshipEdge` - source_id, target_id, relationship_type, strength (0.0-1.0), basis

**Relationship Discovery Strategies:**

| Strategy | Description | Threshold |
|----------|-------------|-----------|
| Explicit | Concepts list each other | N/A |
| Similarity | Cosine similarity | ≥0.75 |
| Co-occurrence | Same/nearby documents | 5 chunks |
| Entity Linking | Same entity in different contexts | N/A |

**Class: `KnowledgeGraphBuilder`**

| Method | Description |
|--------|-------------|
| `build_graph_for_document()` | Build graph for single document |
| `build_global_graph()` | Build across all documents |
| `_discover_explicit_relationships()` | From concept definitions |
| `_discover_similarity_relationships()` | Embedding-based |
| `_discover_cooccurrence_relationships()` | Document/section proximity |
| `link_entities_across_documents()` | Cross-document linking |
| `merge_duplicate_concepts()` | Identify and merge duplicates |

---

### 6.5 Knowledge Retriever (`retrieval.py`)

**Data Classes:**
- `SearchResult` - knowledge_id, content, domain, confidence_level, relevance_score, reasoning
- `RetrievalContext` - Formatted context for agents

**Retrieval Strategies:**
1. **Embedding-based** - Semantic similarity (LM Studio or TF-IDF)
2. **FTS5** - Full-text BM25 ranked search
3. **Hybrid** - Combines multiple strategies

**Class: `KnowledgeRetriever`**

| Method | Description |
|--------|-------------|
| `search()` | Main search with task_type/complexity for meta-learning |
| `_embedding_search()` | Embedding-based semantic search |
| `_fts5_search()` | FTS5 keyword search |
| `_apply_meta_learning_boost()` | Re-rank by historical usefulness |
| `record_retrieval_usage()` | Log which knowledge was used |

---

### 6.6 Knowledge Daemon (`knowledge_daemon.py`)

**Processing Modes:**
| Mode | Description | Trigger |
|------|-------------|---------|
| A | Initial Batch | Existing documents in watch directories |
| B | Continuous Refinement | Periodic (default: 1 hour) |
| C | File System Watching | New documents (watchdog) |

**Data Class: `DaemonConfig`**
```python
watch_directories: List[str]
enable_batch_processing: bool = True
enable_refinement: bool = True
enable_file_watching: bool = True
enable_scheduled_backup: bool = False
refinement_interval: int = 3600      # 1 hour
backup_interval: int = 86400         # 24 hours
processing_threads: int = 2
max_memory_mb: int = 512
chunk_size: int = 1000
chunk_overlap: int = 200
exclusion_patterns: List[str]        # .venv, node_modules, .git, etc.
```

**Class: `KnowledgeDaemon`**

| Method | Description |
|--------|-------------|
| `start()` | Start daemon (all 3 modes) |
| `stop()` | Graceful shutdown |
| `get_status()` | Current status (DaemonStatus) |
| `_batch_processor()` | Process documents in queue |
| `_refiner()` | Periodically re-analyze knowledge |
| `_file_watcher()` | Monitor directories (watchdog) |
| `_backup_processor()` | Optional automatic backups |

---

### 6.7 Additional Knowledge Utilities

| File | Purpose |
|------|---------|
| `backup_manager_extended.py` | Extended backup with compression |
| `quality_checker.py` | Validates knowledge entry quality |
| `knowledge_cleanup.py` | Clean up orphaned/failed/old entries |
| `directory_index.py` | Index watch directories |

---

## 7. LLM Integration (`src/llm/`)

### 7.1 Base Provider (`base_provider.py`)

**Enum: `ProviderType`**
- `LM_STUDIO`, `ANTHROPIC`, `GEMINI`, `OPENAI`, `AZURE`

**Data Classes:**
- `LLMRequest` - system_prompt, user_prompt, temperature, max_tokens, stream, agent_id, model
- `LLMResponse` - content, tokens_used, prompt_tokens, completion_tokens, response_time, model, provider, finish_reason

**Abstract Class: `BaseLLMProvider`**

| Abstract Method | Description |
|-----------------|-------------|
| `complete()` | Non-streaming completion |
| `complete_streaming()` | Streaming with callback |
| `health_check()` | Provider availability |
| `get_provider_name()` | Provider identifier |
| `get_available_models()` | List models |

**Exceptions:**
- `ProviderError` (base)
- `ProviderConnectionError`
- `ProviderAuthenticationError`
- `ProviderRateLimitError`
- `ProviderModelError`

---

### 7.2 LLM Router (`llm_router.py`)

**Class: `LLMRouter`** - Route requests with automatic fallback

| Method | Description |
|--------|-------------|
| `complete()` | Route with fallback |
| `complete_streaming()` | Streaming with fallback |
| `health_check()` | Check all providers |
| `get_statistics()` | Get tracking stats |

**Statistics Tracked:**
- `request_count` - Total requests
- `primary_success_count` - Primary provider successes
- `fallback_success_count` - Fallback successes
- `total_failure_count` - All providers failed

---

### 7.3 LM Studio Client (`lm_studio_client.py`)

**Enum: `RequestPriority`**
- `LOW`, `NORMAL`, `HIGH`, `URGENT`

**Class: `LMStudioClient`**

| Method | Description |
|--------|-------------|
| `complete()` | Synchronous completion |
| `complete_streaming()` | Stream with token-aware control |
| `stream_with_callback()` | Stream with callback (100ms batches) |
| `health_check()` | Test connection |
| `get_available_models()` | List loaded models |
| `estimate_tokens()` | Rough token estimation |

**Class: `TokenAwareStreamController`**
- Monitors tokens during streaming
- Soft limit (85% budget): Injects conclusion signal
- Hard limit (100% budget): Stops stream immediately
- Token estimation: ~1.33 tokens per word

---

### 7.4 Token Budget Manager (`token_budget.py`)

**Data Class: `TokenAllocation`**
- stage_budget, remaining_budget, total_budget, compression_ratio, style_guidance, depth_ratio

**Class: `TokenBudgetManager`**

| Parameter | Default |
|-----------|---------|
| `base_budget` | 20000 |
| `min_budget` | 2000 |
| `max_budget` | 45000 |
| `strict_mode` | Boolean |

**Agent Type Budgets (strict_mode=True):**
| Type | Budget |
|------|--------|
| research | 16000 |
| analysis | 12000 |
| synthesis | 10000 |
| critic | 8000 |

---

### 7.5 Web Search Client (`web_search_client.py`)

**Enum: `SearchProvider`**
- `DUCKDUCKGO`, `SEARXNG`

**Data Class: `SearchResult`**
- title, url, snippet, source, timestamp, relevance_score

**Class: `WebSearchClient`**

| Method | Description |
|--------|-------------|
| `search()` | Execute search query |
| `search_cached()` | Get cached result |
| `clear_cache()` | Clear task cache |
| `get_statistics()` | Query stats |

**Default Blocked Domains:** wikipedia.org, reddit.com

---

### 7.6 Provider Implementations (`src/llm/providers/`)

| Provider | File | Description |
|----------|------|-------------|
| LMStudioProvider | `lm_studio_provider.py` | Local LLM via LM Studio (port 1234) |
| AnthropicProvider | `anthropic_provider.py` | Claude models via API |
| GeminiProvider | `gemini_provider.py` | Google Gemini models |

---

### 7.7 Provider Config (`provider_config.py`)

**Configuration (`config/llm.yaml`):**
```yaml
primary:
  type: lm_studio|anthropic|gemini
  base_url: http://localhost:1234/v1
  api_key: ${ANTHROPIC_API_KEY}
  model: model-name
  timeout: 120

fallbacks:
  - type: anthropic
    api_key: ${API_KEY}
    model: claude-opus

router:
  retry_on_rate_limit: false
  verbose_logging: false
```

---

## 8. GUI System (`src/gui/`)

### 8.1 Main Application (`main.py`)

**Class: `MainApp`**

| Method | Description |
|--------|-------------|
| `__init__()` | Initialize 9 tabs, config, theme |
| `_load_config()` | Load `felix_gui_config.json` |
| `start_system()` | Start via FelixSystem |
| `stop_system()` | Graceful shutdown |
| `_poll_results()` | Thread-safe queue polling |
| `_validate_system_health()` | Check LM Studio connectivity |
| `_on_theme_changed()` | Theme propagation |

**Tabs:**
1. Dashboard
2. Workflows
3. Memory
4. Agents
5. Approvals
6. Terminal
7. Prompts
8. Learning
9. Knowledge Brain
10. Settings

---

### 8.2 Felix System (`felix_system.py`)

**Data Class: `FelixConfig`** (30+ parameters)
- Helix geometry (top_radius, bottom_radius, height, turns)
- LM Studio connection (host, port, model)
- Token budgets and compression settings
- Web search configuration
- Knowledge brain settings

**Class: `FelixSystem`**

| Method | Description |
|--------|-------------|
| `start()` | Initialize LLM client, agents, CentralPost, knowledge brain |
| `stop()` | Cleanup and shutdown |
| `get_system_status()` | Returns system state dict |

---

### 8.3 GUI Frames

| Frame | File | Purpose |
|-------|------|---------|
| DashboardFrame | `dashboard.py` | Start/Stop Felix, live logs |
| WorkflowsFrame | `workflows.py` | Run tasks, save results, feedback |
| MemoryFrame | `memory.py` | Browse/edit memory stores |
| AgentsFrame | `agents.py` | Spawn agents, monitor state |
| ApprovalsFrame | `approvals.py` | Approval workflow, history |
| TerminalFrame | `terminal.py` | Active commands, history |
| PromptsTab | `prompts.py` | Edit prompts, metrics |
| LearningFrame | `learning.py` | Statistics, patterns, calibration |
| KnowledgeBrainFrame | `knowledge_brain.py` | 4 sub-tabs for knowledge management |
| SettingsFrame | `settings.py` | 40+ configuration options |

---

### 8.4 Knowledge Brain Frame (`knowledge_brain.py`)

**4 Consolidated Tabs:**
1. **Control & Processing** - Daemon status, start/stop, statistics
2. **Knowledge Base** - Documents list, concepts explorer
3. **Relationships** - Graph visualization, concept connections
4. **Maintenance** - Cleanup tools, audit trail, analytics

---

## 9. REST API (`src/api/`)

### 9.1 Main Application (`main.py`)

**FastAPI Application:**
- Lifespan management with startup/shutdown
- CORS middleware (configurable via `FELIX_CORS_ORIGINS`)
- Exception handlers

**Root Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and status |
| GET | `/health` | Health check |

**System Endpoints:**
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/system/start` | Yes | Start Felix |
| POST | `/api/v1/system/stop` | Yes | Stop Felix |
| GET | `/api/v1/system/status` | No | Get status |

---

### 9.2 Pydantic Models (`models.py`)

| Model | Description |
|-------|-------------|
| `SystemStatus` | System state response |
| `SystemConfig` | Configuration schema |
| `WorkflowRequest` | Task input with max_steps |
| `WorkflowStatus` | Enum: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED |
| `AgentInfo` | Agent metadata |
| `SynthesisResult` | Synthesis output |
| `WorkflowResponse` | Complete workflow result |

---

### 9.3 API Routers

| Router | Endpoints | Description |
|--------|-----------|-------------|
| `workflows.py` | 6 | Create, list, get, cancel workflows |
| `agents.py` | 5 | List, get, create agents, list plugins |
| `knowledge.py` | 12+ | Documents, concepts, graph, search |
| `task_memory.py` | 4 | Query, store, retrieve tasks |
| `workflow_history.py` | 4 | List, get workflow history |
| `knowledge_memory.py` | 4 | Query, add, retrieve knowledge |
| `compression.py` | 3 | Compress text, list strategies |

---

### 9.4 Dependencies (`dependencies.py`)

| Dependency | Description |
|------------|-------------|
| `get_api_key_from_env()` | Read FELIX_API_KEY |
| `verify_api_key()` | Bearer token validation |
| `optional_api_key()` | Optional authentication |
| `initialize_felix()` | Create FelixSystem |
| `shutdown_felix()` | Cleanup |
| `get_authenticated_felix()` | Protected endpoint dependency |
| `get_felix()` | Get current instance |

---

### 9.5 WebSocket Support

| Module | Purpose |
|--------|---------|
| `connection_manager.py` | Manage WebSocket connections |
| `workflow_stream.py` | Real-time workflow events |

---

## 10. CLI Systems

### 10.1 Main CLI (`cli.py`)

| Command | Description |
|---------|-------------|
| `cmd_run()` | Execute workflow |
| `cmd_status()` | Check system status |
| `cmd_test_connection()` | Test LLM connectivity |
| `cmd_gui()` | Launch GUI |
| `cmd_init()` | Initialize databases |

**Run Options:**
- `--output` / `-o` - Save results to file
- `--max-steps` - Maximum workflow steps
- `--web-search` - Enable web search
- `--verbose` / `-v` - Verbose output
- `--config` - LLM config file

---

### 10.2 Conversational CLI (`src/cli_chat/`)

#### FelixChat (`chat.py`)

**Features:**
- Session management (`-c` to continue, `--resume UUID`)
- Print mode for scripting (`-p "question"`)
- Keyboard shortcuts (Ctrl+R history, Ctrl+L clear)
- Tab completion for commands/files
- Auto-suggestion from history

#### Components

| Component | File | Purpose |
|-----------|------|---------|
| CLIWorkflowOrchestrator | `cli_workflow_orchestrator.py` | Bridges CLI and multi-agent system |
| SessionManager | `session_manager.py` | SQLite persistence |
| OutputFormatter | `formatters.py` | ANSI color support |
| RichFormatter | `formatters.py` | Rich library support |
| FelixCompleter | `completers.py` | Tab completion |
| CommandHandler | `command_handler.py` | Process user input |
| CustomCommandLoader | `custom_commands.py` | Load `.felix/commands/*.md` |

#### Special Prefixes

| Prefix | Description |
|--------|-------------|
| `!command` | Execute Felix command |
| `@file` | Load file contents |
| `#note` | Save note to session |

#### CLI Tools (`cli_chat/tools/`)

| Tool | Purpose |
|------|---------|
| `workflow_tool.py` | Workflow execution, history |
| `knowledge_tool.py` | Knowledge search and retrieval |
| `agent_tool.py` | Agent spawning and monitoring |
| `document_tool.py` | Document ingestion |
| `system_tool.py` | System status and configuration |
| `history_tool.py` | Session history search |

---

## 11. Configuration (`config/`)

### 11.1 Configuration Files

| File | Purpose |
|------|---------|
| `llm.yaml` | LLM provider configuration |
| `prompts.yaml` | Agent prompt templates (v1.1) |
| `trust_rules.yaml` | Command trust levels |
| `task_complexity_patterns.yaml` | Task classification |
| `tool_requirements_patterns.yaml` | Tool classification |

---

### 11.2 LLM Configuration (`llm.yaml`)

```yaml
primary:
  type: lm_studio|anthropic|gemini
  base_url: http://localhost:1234/v1
  api_key: ${ANTHROPIC_API_KEY}
  model: model-name
  timeout: 120

fallbacks:
  - type: anthropic
    api_key: ${API_KEY}
    model: claude-opus

router:
  retry_on_rate_limit: false
  verbose_logging: false

cost_tracking:
  daily_limit: 10.0
  monthly_limit: 100.0
```

---

### 11.3 Trust Rules (`trust_rules.yaml`)

| Level | Example Commands |
|-------|------------------|
| SAFE | `ls`, `pwd`, `cat`, `git status`, `pip list` |
| REVIEW | `pip install`, `git commit`, `mkdir`, `chmod` |
| BLOCKED | `sudo`, `rm -rf`, `dd`, `mkfs` |

---

### 11.4 Prompt Templates (`prompts.yaml`)

**Structure:**
- System Actions section
- Research Agent prompts (6 variants by position/mode)
- Analysis Agent prompts (4 variants)
- Critic Agent prompts (2 variants)
- Variables documentation

---

## 12. Utilities & Support Systems

### 12.1 Prompts System (`src/prompts/`)

**Class: `PromptManager`** (`prompt_manager.py`)

**Lookup Priority:** Cache → Database → YAML → Fallback

| Method | Description |
|--------|-------------|
| `get_prompt()` | Retrieve with fallback chain |
| `save_custom_prompt()` | Save to database |
| `get_performance()` | Get performance metrics |
| `reset_to_defaults()` | Clear custom overrides |

---

### 12.2 Execution System (`src/execution/`)

#### Trust Manager (`trust_manager.py`)

| Method | Description |
|--------|-------------|
| `classify_command()` | Determine trust level |
| `create_approval_request()` | Queue approval |
| `get_pending_approvals()` | List pending |
| `approve_command()` | Record decision |
| `cleanup_expired()` | Remove expired requests |

#### Approval Manager (`approval_manager.py`)

**Enum: `ApprovalDecision`**
- APPROVE_ONCE
- APPROVE_ALWAYS_EXACT
- APPROVE_ALWAYS_COMMAND
- APPROVE_ALWAYS_PATH_PATTERN
- DENY

**Enum: `ApprovalStatus`**
- PENDING, APPROVED, DENIED, EXPIRED, AUTO_APPROVED

#### System Executor (`system_executor.py`)

**Enum: `ErrorCategory`**
- TIMEOUT, PERMISSION, NOT_FOUND, SYNTAX_ERROR, RUNTIME_ERROR, RESOURCE_LIMIT, NETWORK_ERROR

| Method | Description |
|--------|-------------|
| `execute()` | Run command with timeout |
| `execute_with_stream()` | Stream output progressively |
| `detect_venv()` | Detect active virtual environment |
| `categorize_error()` | Classify error type |

#### Command History (`command_history.py`)

| Method | Description |
|--------|-------------|
| `record()` | Store execution record |
| `query()` | Search history |
| `get_latest()` | Retrieve recent executions |
| `cleanup()` | Remove old records |

---

### 12.3 Pipeline (`src/pipeline/`)

**File: `chunking.py`**

**Data Class: `ChunkedResult`**
- chunk_id, task_id, agent_id, content_chunk, chunk_index, is_final, timestamp

**Class: `ProgressiveProcessor`**
- Manages incremental generation
- `get_next_chunk()` - Retrieve chunk by index

**Class: `ContentSummarizer`**
- Smart truncation with fallback

---

### 12.4 Migration System (`src/migration/`)

| Migration | Purpose |
|-----------|---------|
| `add_audit_log_table.py` | Audit trail schema |
| `add_cascade_delete.py` | Cascading deletes |
| `add_fts5_triggers.py` | Full-text search triggers |
| `add_knowledge_brain.py` | Knowledge brain tables |
| `add_watch_directories_table.py` | File monitoring config |
| `add_learning_tables.py` | Learning system schema |
| `add_knowledge_validation.py` | Validation rules |
| `create_feedback_system.py` | Feedback integration |
| `create_system_actions.py` | System command tracking |
| `create_agent_performance.py` | Agent metrics |
| `version_manager.py` | Migration version tracking |
| `backup_manager.py` | Database backup/restore |

---

### 12.5 Utilities (`src/utils/`)

**File: `markdown_formatter.py`**

| Function | Description |
|----------|-------------|
| `format_synthesis_markdown()` | Basic workflow result formatting |
| `format_synthesis_markdown_detailed()` | Comprehensive report with metrics |

---

## 13. Database Schemas

### 13.1 felix_knowledge.db

```sql
-- Main knowledge entries
knowledge_entries (
    knowledge_id TEXT PRIMARY KEY,
    knowledge_type TEXT,
    content_json TEXT,
    confidence_level TEXT,
    source_agent TEXT,
    domain TEXT,
    tags_json TEXT,
    created_at REAL,
    updated_at REAL,
    access_count INTEGER,
    success_rate REAL,
    related_entries_json TEXT,
    validation_score REAL,
    validation_flags TEXT,
    validation_status TEXT,
    validated_at REAL,
    embedding BLOB,          -- Knowledge Brain
    source_doc_id TEXT,      -- Knowledge Brain
    chunk_index INTEGER      -- Knowledge Brain
)

-- Normalized tags
knowledge_tags (
    knowledge_id TEXT,
    tag TEXT,
    PRIMARY KEY (knowledge_id, tag)
)

-- Full-text search (FTS5)
knowledge_fts (
    knowledge_id,
    content,
    domain,
    tags
) -- tokenize='porter unicode61'

-- Meta-learning
knowledge_usage (
    workflow_id TEXT,
    knowledge_id TEXT,
    task_type TEXT,
    task_complexity TEXT,
    useful_score REAL,
    retrieval_method TEXT,
    recorded_at REAL
)

-- Watch directories
watch_directories (
    watch_id TEXT PRIMARY KEY,
    directory_path TEXT,
    added_at REAL,
    enabled INTEGER,
    last_scan REAL,
    document_count INTEGER,
    entry_count INTEGER,
    notes TEXT
)

-- Document sources
document_sources (
    doc_id TEXT PRIMARY KEY,
    file_path TEXT,
    file_hash TEXT,
    status TEXT,
    ingested_at REAL,
    updated_at REAL,
    chunk_count INTEGER,
    metadata_json TEXT
)

-- Knowledge relationships
knowledge_relationships (
    source_id TEXT,
    target_id TEXT,
    relationship_type TEXT,
    strength REAL,
    basis TEXT,
    created_at REAL
)

-- Audit log
knowledge_audit_log (
    audit_id INTEGER PRIMARY KEY,
    timestamp REAL,
    operation TEXT,
    knowledge_id TEXT,
    user_agent TEXT,
    old_values_json TEXT,
    new_values_json TEXT,
    details TEXT,
    transaction_id TEXT
)
```

---

### 13.2 felix_memory.db / felix_task_memory.db

```sql
-- Task patterns
task_patterns (
    pattern_id TEXT PRIMARY KEY,
    task_type TEXT,
    complexity TEXT,
    keywords_json TEXT,
    typical_duration REAL,
    success_rate REAL,
    failure_modes_json TEXT,
    optimal_strategies_json TEXT,
    required_agents_json TEXT,
    context_requirements_json TEXT,
    created_at REAL,
    updated_at REAL,
    usage_count INTEGER
)

-- Task executions
task_executions (
    execution_id TEXT PRIMARY KEY,
    task_description TEXT,
    task_type TEXT,
    complexity TEXT,
    outcome TEXT,
    duration REAL,
    agents_used_json TEXT,
    strategies_used_json TEXT,
    context_size INTEGER,
    error_messages_json TEXT,
    success_metrics_json TEXT,
    patterns_matched_json TEXT,
    created_at REAL
)
```

---

### 13.3 felix_workflow_history.db

```sql
workflow_outputs (
    workflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_input TEXT,
    status TEXT,
    created_at TEXT,
    completed_at TEXT,
    final_synthesis TEXT,
    confidence REAL,
    agents_count INTEGER,
    tokens_used INTEGER,
    max_tokens INTEGER,
    processing_time REAL,
    temperature REAL,
    metadata TEXT,               -- Full JSON result
    parent_workflow_id INTEGER,  -- Conversation threading
    conversation_thread_id TEXT
)
```

---

### 13.4 felix_cli_sessions.db

```sql
sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    last_active TEXT,
    title TEXT,
    tags TEXT
)

messages (
    message_id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    workflow_id INTEGER,
    timestamp TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
```

---

### 13.5 felix_agent_performance.db

```sql
agent_performance (
    id INTEGER PRIMARY KEY,
    agent_id TEXT,
    workflow_id INTEGER,
    agent_type TEXT,
    spawn_time REAL,
    checkpoint REAL,
    confidence REAL,
    tokens_used INTEGER,
    processing_time REAL,
    depth_ratio REAL,
    phase TEXT,
    position_x REAL,
    position_y REAL,
    position_z REAL,
    content_preview TEXT,
    timestamp REAL
)
```

---

## 14. Critical Thresholds & Constants

### 14.1 Confidence Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Synthesis trigger | 0.80 | Minimum team confidence for synthesis |
| Research agent max | 0.6 | Gather info, don't decide |
| Analysis agent max | 0.8 | Process, prepare for synthesis |
| Synthesis agent max | 0.95 | Create final output |
| Critic agent max | 0.8 | Provide feedback |
| Web search trigger | 0.7 | Search below this confidence |
| Volatility threshold | 0.15 | Trigger spawning if above |

---

### 14.2 Agent Spawn Timing (Normalized 0.0-1.0)

| Agent Type | Range | Description |
|------------|-------|-------------|
| Research | 0.0-0.3 | Early exploration |
| Analysis | 0.2-0.6 | Mid-phase |
| Critic | 0.4-0.7 | Continuous validation |

---

### 14.3 Helix Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Top radius | 3.0 | Wide exploration |
| Bottom radius | 0.5 | Narrow synthesis |
| Height | 8.0 | Progression depth |
| Turns | 2 | Spiral complexity |
| Checkpoints | [0.0, 0.3, 0.5, 0.7, 0.9] | Processing points |

---

### 14.4 Token Management

| Parameter | Value |
|-----------|-------|
| Base budget | 2048 tokens/agent |
| Research max | 16,000 tokens |
| Analysis max | 16,000 tokens |
| Synthesis max | 20,000 tokens |
| Critic max | 16,000 tokens |
| Synthesis output | 1500-3000 tokens |
| Direct answer mode | 200 tokens |
| Soft stream limit | 85% of budget |
| Hard stream limit | 100% of budget |

---

### 14.5 Knowledge Brain Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chunk size | 1000 | Characters per chunk |
| Chunk overlap | 200 | Characters between chunks |
| Similarity threshold | 0.75 | Min cosine for relationship |
| Co-occurrence window | 5 | Chunks distance for link |
| Meta-learning min samples | 2 | For reliable boost |
| Meta-learning boost | 0.7-1.0 | Multiplier range |
| Quality threshold | 0.6 | Min for comprehension |
| Refinement interval | 3600 | Seconds (1 hour) |

---

### 14.6 Communication

| Parameter | Value |
|-----------|-------|
| Model | O(N) hub-spoke |
| Max distance | Helix top radius |
| Message timeout | 10 seconds |
| Search cooldown | 10 seconds |
| Search min samples | 3 |

---

## 15. Integration Points

### 15.1 Agent System → Communication

```
LLMAgent.share_result_to_central() → Message → CentralPost
LLMAgent.process_synthesis_feedback() ← SYNTHESIS_FEEDBACK
LLMAgent.request_action() → SystemCommandManager
LLMAgent.influence_agent_behavior() → Other agents
```

### 15.2 Communication → Agent System

```
CentralPost.broadcast_to_agents() → All agents
CentralPost.AgentRegistry → Agent management
SystemCommandManager → Execute agent requests
WebSearchCoordinator → Feed knowledge to agents
```

### 15.3 Workflows → Communication

```
FelixWorkflow → CentralPost coordination
CollaborativeContextBuilder ← CentralPost messages
ConceptRegistry → Terminology consistency
ContextRelevanceEvaluator → Filter knowledge
```

### 15.4 Communication → Workflows

```
SynthesisEngine → Task complexity (affects knowledge limits)
SynthesisEngine → Tool requirements (conditional injection)
MemoryFacade → Knowledge persistence
PerformanceMonitor → Track efficiency
```

### 15.5 Knowledge System → All

```
KnowledgeStore → Persist agent results
Meta-learning → Boost relevance
File discovery → Improve future searches
Tool instructions → Conditional agent injection
```

### 15.6 LLM → Memory

```
LLMRouter → Unified provider access
Responses → WorkflowHistory
TokenBudgetManager → Position-based limits
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `FELIX_API_KEY` | API authentication |
| `FELIX_LM_HOST` | LM Studio host (default: 127.0.0.1) |
| `FELIX_LM_PORT` | LM Studio port (default: 1234) |
| `FELIX_MAX_AGENTS` | Maximum concurrent agents |
| `FELIX_ENABLE_KNOWLEDGE_BRAIN` | Enable knowledge system |
| `FELIX_CORS_ORIGINS` | Allowed CORS origins |
| `ANTHROPIC_API_KEY` | Claude API key |
| `GEMINI_API_KEY` | Google Gemini key |

---

## File Index by Category

### Core Processing (13 files)
- `src/agents/agent.py`
- `src/agents/llm_agent.py`
- `src/agents/specialized_agents.py`
- `src/agents/dynamic_spawning.py`
- `src/agents/base_specialized_agent.py`
- `src/agents/agent_plugin_registry.py`
- `src/agents/builtin/research_plugin.py`
- `src/agents/builtin/analysis_plugin.py`
- `src/agents/builtin/critic_plugin.py`
- `src/communication/central_post.py`
- `src/communication/message_types.py`
- `src/communication/synthesis_engine.py`
- `src/workflows/felix_workflow.py`

### Memory & Knowledge (12 files)
- `src/memory/knowledge_store.py`
- `src/memory/task_memory.py`
- `src/memory/workflow_history.py`
- `src/memory/context_compression.py`
- `src/memory/audit_log.py`
- `src/knowledge/document_ingest.py`
- `src/knowledge/embeddings.py`
- `src/knowledge/comprehension.py`
- `src/knowledge/graph_builder.py`
- `src/knowledge/retrieval.py`
- `src/knowledge/knowledge_daemon.py`
- `src/knowledge/quality_checker.py`

### LLM Integration (8 files)
- `src/llm/base_provider.py`
- `src/llm/llm_router.py`
- `src/llm/lm_studio_client.py`
- `src/llm/token_budget.py`
- `src/llm/web_search_client.py`
- `src/llm/provider_config.py`
- `src/llm/providers/lm_studio_provider.py`
- `src/llm/providers/anthropic_provider.py`

### Interfaces (15 files)
- `src/gui/main.py`
- `src/gui/felix_system.py`
- `src/gui/dashboard.py`
- `src/gui/workflows.py`
- `src/gui/memory.py`
- `src/gui/agents.py`
- `src/gui/approvals.py`
- `src/gui/terminal.py`
- `src/gui/knowledge_brain.py`
- `src/gui/settings.py`
- `src/api/main.py`
- `src/api/models.py`
- `src/api/dependencies.py`
- `src/cli.py`
- `src/cli_chat/chat.py`

### Configuration (5 files)
- `config/llm.yaml`
- `config/prompts.yaml`
- `config/trust_rules.yaml`
- `config/task_complexity_patterns.yaml`
- `config/tool_requirements_patterns.yaml`

---

*End of Master Index*
