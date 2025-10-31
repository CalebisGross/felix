# Feedback Module

## Purpose
User feedback collection and integration system enabling continuous improvement through structured feedback on workflows, knowledge quality, and system performance.

## Key Files

### [feedback_manager.py](feedback_manager.py)
Centralized feedback collection and storage.
- **`FeedbackManager`**: Manages feedback lifecycle from collection to storage
- **`FeedbackType`**: Enum for feedback categories (WORKFLOW_RATING, SYNTHESIS_QUALITY, AGENT_PERFORMANCE, KNOWLEDGE_QUALITY, SYSTEM_BEHAVIOR)
- **`ReasonCategory`**: Enum for negative feedback reasons (INACCURATE, INCOMPLETE, IRRELEVANT, TOO_SLOW, ERROR)
- **`PatternType`**: Enum for identified feedback patterns (RECURRING_ISSUE, CONSISTENT_PRAISE, DOMAIN_SPECIFIC, AGENT_SPECIFIC)

**Data Structures**:

#### `WorkflowRating`
- **rating**: 1-5 scale
- **comment**: Optional text feedback
- **workflow_id**: Reference to workflow
- **timestamp**: When feedback provided

#### `DetailedFeedback`
- **type**: FeedbackType enum
- **rating**: Numeric score
- **reason**: ReasonCategory for negative feedback
- **detailed_comment**: Rich text feedback
- **context**: Additional metadata
- **actionable**: Boolean flag for actionable items

#### `KnowledgeFeedback`
- **knowledge_entry_id**: Reference to knowledge entry
- **was_helpful**: Boolean usefulness flag
- **accuracy_rating**: 1-5 scale
- **relevance_rating**: 1-5 scale
- **suggestions**: Improvement suggestions

### [feedback_integrator.py](feedback_integrator.py)
Processes feedback and applies improvements.
- **`FeedbackIntegrator`**: Integrates feedback into learning systems

**Integration Points**:
1. **PatternLearner**: Adjusts workflow patterns based on ratings
2. **ConfidenceCalibrator**: Calibrates based on accuracy feedback
3. **ThresholdLearner**: Optimizes thresholds from speed feedback
4. **KnowledgeStore**: Boosts/demotes knowledge based on usefulness
5. **PromptManager**: Refines prompts based on quality feedback

## Key Concepts

### Feedback Types

#### 1. Workflow Rating
Overall satisfaction with workflow execution:
- Rating: 1 (poor) to 5 (excellent)
- Tracks synthesis quality, speed, relevance
- Used for pattern learning

#### 2. Synthesis Quality
Specific feedback on synthesis output:
- Accuracy: Factual correctness
- Completeness: Coverage of topic
- Clarity: Presentation quality
- Relevance: Alignment with task

#### 3. Agent Performance
Per-agent feedback:
- Contribution quality
- Response time
- Role alignment
- Team coordination

#### 4. Knowledge Quality
Feedback on retrieved knowledge:
- Was it helpful? (binary)
- Accuracy rating (1-5)
- Relevance rating (1-5)
- Suggestions for improvement

#### 5. System Behavior
General system feedback:
- Performance issues
- UI/UX concerns
- Feature requests
- Bug reports

### Feedback Collection Flow

```
User completes workflow
       ↓
GUI presents feedback form
       ↓
User provides rating/comments
       ↓
FeedbackManager stores feedback
       ↓
FeedbackIntegrator processes
       ↓
Updates learning systems
       ↓
Improvements applied to future workflows
```

### Pattern Recognition

FeedbackManager identifies patterns:
- **Recurring issues**: Same problem across multiple workflows
- **Consistent praise**: Successful patterns to reinforce
- **Domain-specific**: Feedback clusters by domain
- **Agent-specific**: Performance issues with specific agents

### Actionable Feedback Flagging

Feedback marked actionable when:
- Specific improvement suggested
- Issue clearly described
- Context provided for reproduction
- Tied to measurable metric

### Meta-Learning Integration

Feedback drives meta-learning:
1. **Knowledge usage tracking**: Boost helpful knowledge
2. **Workflow optimization**: Reinforce successful patterns
3. **Threshold tuning**: Adjust based on speed/quality tradeoffs
4. **Confidence calibration**: Align predictions with outcomes
5. **Prompt refinement**: Improve based on output quality

## Database Schema

**Table**: `user_feedback` (in `felix_memory.db`)
```sql
CREATE TABLE user_feedback (
    feedback_id TEXT PRIMARY KEY,
    workflow_id TEXT,
    feedback_type TEXT,
    rating INTEGER,
    reason TEXT,
    comment TEXT,
    context TEXT,  -- JSON metadata
    actionable BOOLEAN,
    created_at TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
)
```

**Table**: `feedback_patterns`
```sql
CREATE TABLE feedback_patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT,
    description TEXT,
    frequency INTEGER,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    action_taken TEXT
)
```

## Usage Example

```python
from src.feedback.feedback_manager import FeedbackManager, WorkflowRating
from src.feedback.feedback_integrator import FeedbackIntegrator

# Initialize manager
fm = FeedbackManager(db_path="felix_memory.db")

# Collect workflow feedback
feedback = WorkflowRating(
    rating=4,
    comment="Good synthesis but a bit slow",
    workflow_id="workflow_001"
)
fm.store_workflow_feedback(feedback)

# Process and integrate feedback
integrator = FeedbackIntegrator(fm)
integrator.process_pending_feedback()

# Identified patterns
patterns = fm.identify_patterns()
for pattern in patterns:
    print(f"{pattern.type}: {pattern.description}")
```

### GUI Integration

The [Learning tab](../gui/learning.py) provides:
- Feedback form after workflow completion
- Historical feedback browser
- Pattern visualization
- Actionable items queue
- Integration status dashboard

## Configuration

```yaml
feedback:
  enable_feedback_collection: true
  require_feedback: false              # Optional vs required
  min_rating_for_pattern: 4            # Threshold for "success" pattern
  max_rating_for_issue: 2              # Threshold for "issue" pattern
  pattern_frequency_threshold: 3       # Occurrences to identify pattern
  auto_integrate: true                 # Automatically apply improvements
```

## Feedback-Driven Improvements

### Example 1: Knowledge Boost
User marks knowledge as helpful → Increase retrieval ranking

### Example 2: Prompt Refinement
Multiple complaints about incomplete synthesis → Adjust synthesis prompt for more detail

### Example 3: Threshold Optimization
Feedback: "Too slow" → Decrease spawning threshold to reduce agent count

### Example 4: Agent Calibration
Feedback: "Inaccurate" with high agent confidence → Calibrate agent confidence downward

## Related Modules
- [learning/](../learning/) - PatternLearner and ConfidenceCalibrator use feedback
- [memory/](../memory/) - KnowledgeStore updated based on knowledge feedback
- [prompts/](../prompts/) - PromptManager refines templates from quality feedback
- [workflows/](../workflows/) - Workflow patterns adjusted based on ratings
- [gui/](../gui/) - Learning tab provides feedback interface
