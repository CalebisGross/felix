# Workflows Module

## Purpose
High-level task orchestration coordinating agent teams, web search, context building, and truth assessment for complex multi-agent workflows.

## Key Files

### [felix_workflow.py](felix_workflow.py)
Main workflow orchestration integrating all Felix components.
- **`run_felix_workflow()`**: Primary workflow function coordinating agent spawning, communication, and synthesis
- **Integration**: Combines CentralPost, AgentFactory, DynamicSpawning, KnowledgeStore, web search
- **Process Flow**:
  1. Initialize CentralPost and components
  2. Spawn initial agent team based on task
  3. Distribute task via hub-spoke communication
  4. Monitor confidence for dynamic spawning
  5. Synthesize results when threshold reached (≥0.80)
  6. Store results in workflow history

### [context_builder.py](context_builder.py)
Collaborative context management for multi-agent coordination.
- **`CollaborativeContextBuilder`**: Builds shared context from multiple agent inputs
- **`EnrichedContext`**: Enhanced context structure with metadata and cross-references
- **Features**: Context merging, deduplication, relevance scoring, source tracking

### [truth_assessment.py](truth_assessment.py)
Validation framework for workflow outputs and knowledge entries.
- **`QueryType`**: Enum for query classification (FACTUAL, OPINION, MIXED, HYPOTHETICAL)
- **`assess_answer_confidence()`**: Evaluates response confidence based on multiple signals
- **`validate_knowledge_entry()`**: Validates knowledge quality before storage
- **Confidence Signals**:
  - Cross-reference validation
  - Source credibility assessment
  - Internal consistency checking
  - Temporal validity verification

### [conversation_loader.py](conversation_loader.py)
Conversation continuity for multi-turn workflows.
- **`ConversationContextLoader`**: Loads previous conversation context for continuity
- **Features**: Context compression, relevance filtering, turn tracking

### [concept_registry.py](concept_registry.py)
Workflow-scoped concept tracking for terminology consistency (referenced in CLAUDE.md).
- **`ConceptRegistry`**: Tracks concept definitions across agents to prevent conflicting interpretations
- **`ConceptDefinition`**: Data class for concept metadata (name, definition, source_agent, confidence)
- **`ConceptConflict`**: Detected conflict between concept definitions
- **Features**:
  - Check if concepts already defined before registering new ones
  - Detect duplicate, contradictory, or overlapping definitions
  - Query related concepts and relationships
  - Export to `analysis/improvement_registry.md` for review

### [context_relevance.py](context_relevance.py)
Contextual relevance evaluation for filtering irrelevant facts (referenced in CLAUDE.md).
- **`ContextRelevanceEvaluator`**: Distinguishes factual accuracy from contextual relevance
- **`RelevanceScore`**: Score and explanation for context relevance (0.0-1.0)
- **Purpose**: Prevents agents from providing accurate but irrelevant facts
- **Integration**: Used in `ContextBuilder.build_agent_context()` for filtering knowledge

### [failure_recovery.py](failure_recovery.py)
Adaptive failure recovery system (Phase 3.2).
- **`FailureRecoveryManager`**: Manages failure detection and recovery strategies
- **`FailureType`**: Enum for failure types (AGENT_ERROR, COMMAND_ERROR, TIMEOUT, LOW_CONFIDENCE, INSUFFICIENT_DATA)
- **`FailureRecord`**: Record of failure context and recovery attempts
- **Strategies**: Agent respawning with adjusted parameters, command retry with alternatives, pattern-based learning

### [task_completion_detector.py](task_completion_detector.py)
Task completion analysis system (Phase 2.3).
- **`TaskCompletionDetector`**: Distinguishes "task solved" from "ran out of time"
- **`CompletionStatus`**: Enum (COMPLETE, INCOMPLETE, UNCLEAR, ERROR)
- **Heuristics**: Question answering coverage, request fulfillment, confidence alignment, content sufficiency
- **Pattern matching**: Detects incomplete answer phrases vs completion indicators

## Key Concepts

### Workflow Lifecycle
1. **Initialization**: Create CentralPost, AgentFactory, coordinators
2. **Agent Spawning**: Initial team + dynamic spawning based on confidence
3. **Task Distribution**: Hub broadcasts task to all agents
4. **Agent Execution**: Parallel agent processing with message passing
5. **Confidence Monitoring**: Continuous assessment of team progress
6. **Dynamic Adjustment**: Spawn additional agents if confidence < 0.80
7. **Synthesis**: Hub combines agent outputs when confident
8. **Persistence**: Store workflow history and knowledge

### Integration Points
- **CentralPost**: Hub-spoke communication coordinator
- **AgentFactory**: Creates agents with helix positioning
- **DynamicSpawning**: Confidence-based team scaling
- **KnowledgeStore**: Persists agent insights
- **WebSearchClient**: Internet research for agents
- **WorkflowHistory**: Execution tracking and analysis

### Context Building
- **Collaborative**: Multiple agents contribute to shared context
- **Enriched**: Metadata, cross-references, source tracking
- **Deduplicated**: Remove redundant information
- **Scored**: Relevance scoring for prioritization

### Truth Assessment
Multi-signal confidence evaluation:
- **Cross-reference**: Multiple sources agree
- **Source credibility**: Trusted sources preferred
- **Internal consistency**: No contradictions
- **Temporal validity**: Information up-to-date

### Workflow History Tracking
Stored per execution:
- Task description and synthesis output
- Agent count and roles
- Token consumption
- Processing time
- Confidence score
- Status (SUCCESS, PARTIAL, FAILED)

### Knowledge Auto-Injection
When Knowledge Brain enabled (`knowledge_auto_augment: true`):
- Retrieves relevant knowledge for task domain
- Injects into agent context
- Tracks usage for meta-learning
- Boosts future relevance

## Configuration

```yaml
helix:
  top_radius: 3.0
  bottom_radius: 0.5
  height: 8.0
  turns: 2

spawning:
  confidence_threshold: 0.80
  max_agents: 10

knowledge_brain:
  knowledge_auto_augment: true  # Auto-inject knowledge
```

## Usage Example

```python
from src.workflows.felix_workflow import run_felix_workflow
from src.llm.lm_studio_client import LMStudioClient

llm_client = LMStudioClient()
result = run_felix_workflow(
    task="Analyze quantum computing trends",
    llm_client=llm_client,
    enable_web_search=True
)

print(f"Synthesis: {result['synthesis']}")
print(f"Confidence: {result['confidence']}")
print(f"Agents used: {result['agent_count']}")
```

## Related Modules
- [communication/](../communication/) - CentralPost coordination
- [agents/](../agents/) - Agent implementations and spawning
- [memory/](../memory/) - WorkflowHistory and KnowledgeStore persistence
- [knowledge/](../knowledge/) - WorkflowIntegration for context augmentation
- [llm/](../llm/) - LLM client and web search integration
- [gui/](../gui/) - Workflows tab for GUI execution
