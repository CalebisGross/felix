# Learning Module

## Purpose
Adaptive learning systems providing pattern recognition, workflow recommendations, dynamic threshold optimization, and confidence calibration for continuous framework improvement.

## Key Files

### [pattern_learner.py](pattern_learner.py)
Workflow pattern tracking and recognition.
- **`PatternLearner`**: Identifies recurring workflow patterns and tracks their success rates
- **`WorkflowRecommendation`**: Recommendation structure with confidence and reasoning
- **Features**:
  - Pattern extraction from workflow history
  - Success rate calculation
  - Similar workflow identification
  - Optimization suggestions based on historical performance

### [recommendation_engine.py](recommendation_engine.py)
Multi-strategy recommendation system.
- **`RecommendationEngine`**: Generates recommendations from multiple sources
- **`UnifiedRecommendation`**: Combined recommendation structure with metadata
- **Recommendation Sources**:
  - Pattern-based: Historical workflow patterns
  - Performance-based: Agent and synthesis performance
  - Context-based: Task characteristics and requirements
  - Meta-learning: Knowledge usage patterns

### [threshold_learner.py](threshold_learner.py)
Dynamic threshold optimization for spawning and confidence.
- **`ThresholdLearner`**: Learns optimal thresholds from execution history
- **`ThresholdRecord`**: Threshold value tracking with outcome correlation
- **Optimization Targets**:
  - Agent spawning confidence threshold (default: 0.80)
  - Dynamic team size adjustments
  - Synthesis trigger points
  - Quality vs speed tradeoffs

### [confidence_calibrator.py](confidence_calibrator.py)
Confidence score calibration and adjustment.
- **`ConfidenceCalibrator`**: Calibrates agent confidence scores against actual outcomes
- **`CalibrationRecord`**: Tracks predicted vs actual confidence alignment
- **Features**:
  - Over-confidence detection and correction
  - Under-confidence identification
  - Agent-specific calibration curves
  - Continuous recalibration based on new data

### [db_utils.py](db_utils.py)
Database utilities for learning system persistence.
- **`TransactionContext`**: Context manager for safe database transactions
- **`retry_on_locked`**: Decorator for handling database lock contention
- **Functions**: `safe_execute()`, `safe_commit()` for robust database operations

## Key Concepts

### Pattern Recognition
Identifies patterns in:
- Task types and their success strategies
- Agent team compositions
- Token budget allocations
- Web search integration effectiveness
- Synthesis approaches

### Recommendation Strategies

#### 1. Pattern-Based
- Matches current task to historical patterns
- Suggests proven agent configurations
- Recommends successful workflow structures

#### 2. Performance-Based
- Identifies high-performing agents
- Suggests optimal team sizes
- Recommends token allocation strategies

#### 3. Context-Based
- Analyzes task characteristics
- Suggests appropriate specializations
- Recommends search strategies

#### 4. Meta-Learning
- Tracks knowledge usage effectiveness
- Identifies useful concept clusters
- Optimizes retrieval strategies

### Threshold Optimization

Dynamic adjustment of:
- **Spawning threshold**: When to spawn additional agents (default: 0.80)
- **Team size**: Optimal agent count for task complexity
- **Synthesis trigger**: When team is ready to synthesize
- **Quality gates**: Minimum acceptable confidence

Learning process:
1. Track threshold value and outcomes
2. Correlate thresholds with success/failure
3. Identify optimal ranges
4. Gradually adjust defaults
5. Continuously recalibrate

### Confidence Calibration

Addresses:
- **Over-confidence**: Agents reporting higher confidence than warranted
- **Under-confidence**: Agents underestimating their accuracy
- **Agent-specific biases**: Individual agent calibration curves
- **Task-dependent calibration**: Different calibration per task type

Calibration process:
1. Record predicted confidence
2. Measure actual outcome quality
3. Calculate calibration error
4. Build agent-specific curves
5. Apply corrections to future predictions

### Database Transaction Safety
- **TransactionContext**: Ensures ACID properties
- **Retry logic**: Handles SQLite locking gracefully
- **Connection pooling**: Efficient database access
- **Safe operations**: Prevents corruption from concurrent access

## Database Schema

Learning tables in `felix_memory.db` and `felix_task_memory.db`:
- `workflow_patterns`: Pattern definitions and success rates
- `threshold_records`: Historical threshold values and outcomes
- `calibration_data`: Confidence predictions vs actuals
- `recommendations`: Generated recommendations and feedback

## Configuration

```yaml
learning:
  enable_pattern_learning: true
  enable_threshold_optimization: true
  enable_confidence_calibration: true
  min_samples_for_learning: 10        # Minimum workflows before learning activates
  calibration_window: 100             # Samples for calibration window
  threshold_adjustment_rate: 0.05     # Learning rate for threshold updates
```

## Usage Example

```python
from src.learning.pattern_learner import PatternLearner
from src.learning.recommendation_engine import RecommendationEngine

# Learn patterns
learner = PatternLearner(db_path="felix_memory.db")
patterns = learner.identify_patterns()

# Get recommendations
engine = RecommendationEngine(learner)
recommendations = engine.get_recommendations(
    task="Analyze financial trends",
    context={"domain": "finance", "complexity": "high"}
)

for rec in recommendations:
    print(f"{rec.type}: {rec.suggestion} (confidence: {rec.confidence})")
```

## Related Modules
- [memory/](../memory/) - WorkflowHistory and TaskMemory provide learning data
- [workflows/](../workflows/) - Recommendations applied to workflow execution
- [agents/](../agents/) - Agent performance tracked for calibration
- [communication/](../communication/) - Synthesis confidence used for threshold learning
- [feedback/](../feedback/) - User feedback integrated into learning
