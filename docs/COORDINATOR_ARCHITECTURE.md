# Coordinator Architecture Documentation

Deep dive into Felix's coordinator-based architecture for the communication hub.

## Table of Contents

1. [Overview](#overview)
2. [Architecture Rationale](#architecture-rationale)
3. [Coordinator Catalog](#coordinator-catalog)
4. [Integration Patterns](#integration-patterns)
5. [Adding New Coordinators](#adding-new-coordinators)

---

## Overview

Felix's CentralPost uses a **coordinator pattern** where specialized subsystems handle distinct responsibilities. This replaced a monolithic design where CentralPost handled everything directly.

### Before Refactor (Monolithic)

```python
class CentralPost:
    def __init__(self):
        # Everything in one class
        self._agents = {}
        self._message_queue = deque()
        self._synthesis_state = {}
        self._search_cache = {}
        self._command_executor = SystemExecutor()
        self._streaming_callbacks = {}
        self._memory_connections = {}
        self._metrics = {}

    def handle_message(self, message):
        # 500+ line method handling all message types
        if message.type == "synthesis":
            # synthesis logic here
        elif message.type == "search":
            # search logic here
        elif message.type == "command":
            # command logic here
        # ... many more branches
```

**Problems:**
- Single class with 2000+ lines
- Difficult to test individual features
- Changes in one area risk breaking others
- Hard to understand and maintain
- Tight coupling between unrelated features

### After Refactor (Coordinator Pattern)

```python
class CentralPost:
    def __init__(self, helix):
        # Core responsibilities
        self._agents = {}
        self._message_queue = deque()
        self.agent_registry = AgentRegistry()

        # Delegate to coordinators
        self.synthesis_engine = SynthesisEngine(helix)
        self.web_search_coordinator = WebSearchCoordinator()
        self.system_command_manager = SystemCommandManager()
        self.streaming_coordinator = StreamingCoordinator()
        self.memory_facade = MemoryFacade()
        self.performance_monitor = PerformanceMonitor()

    def handle_message(self, message):
        # Delegate to appropriate coordinator
        if message.type == "synthesis":
            return self.synthesis_engine.synthesize(...)
        elif message.type == "search":
            return self.web_search_coordinator.search(...)
        elif message.type == "command":
            return self.system_command_manager.execute(...)
```

**Benefits:**
- Each coordinator ~200-400 lines (manageable)
- Independent testing of coordinators
- Changes isolated to relevant coordinator
- Clear responsibilities and boundaries
- Easy to add new coordinators

---

## Architecture Rationale

### Why Coordinators?

The coordinator pattern provides:

1. **Separation of Concerns**: Each coordinator has one job
2. **Single Responsibility Principle**: Coordinators don't overlap
3. **Testability**: Mock coordinators easily for testing
4. **Extensibility**: Add coordinators without modifying CentralPost
5. **Maintainability**: Smaller, focused classes are easier to understand

### Design Decisions

**Q: Why not just separate modules?**
A: Coordinators need shared state (agent registry, message queue) that CentralPost manages. They're tightly coupled to the hub's lifecycle.

**Q: Why not microservices?**
A: Felix runs as a single process. Microservices would add network overhead and complexity without benefits for this use case.

**Q: Why not event bus?**
A: Direct delegation is simpler and faster. Event bus adds indirection that isn't needed for this architecture.

---

## Coordinator Catalog

### 1. SynthesisEngine

**File**: [src/communication/synthesis_engine.py](../src/communication/synthesis_engine.py)

**Purpose**: Aggregates agent outputs into coherent final synthesis.

**Key Responsibilities:**
- Collect outputs from all agents
- Determine consensus level
- Calculate adaptive synthesis parameters
- Generate final synthesis via LLM
- Return synthesis with confidence score

**Adaptive Parameters:**

| Consensus Level | Temperature | Max Tokens | Strategy |
|----------------|-------------|------------|----------|
| High (>0.8) | 0.2 | 1500 | Focused consolidation |
| Medium (0.5-0.8) | 0.3 | 2000 | Balanced synthesis |
| Low (<0.5) | 0.4 | 3000 | Exploratory synthesis |

**Algorithm:**
```python
def calculate_synthesis_params(agent_outputs, team_size):
    # Calculate consensus
    consensus = calculate_agreement(agent_outputs)

    # Adaptive temperature
    if consensus > 0.8:
        temperature = 0.2  # High agreement = focused
    elif consensus > 0.5:
        temperature = 0.3  # Medium agreement = balanced
    else:
        temperature = 0.4  # Low agreement = exploratory

    # Adaptive tokens
    diversity = calculate_output_diversity(agent_outputs)
    base_tokens = 1500
    bonus_tokens = int(diversity * 1500)
    max_tokens = min(base_tokens + bonus_tokens, 3000)

    return temperature, max_tokens
```

**Usage:**
```python
synthesis_result = central_post.synthesis_engine.synthesize(
    agent_outputs=outputs,
    task_description=task,
    team_size=len(agents)
)
# Returns: {synthesis_content, confidence, token_count}
```

**Testing:**
```python
def test_synthesis_high_consensus():
    engine = SynthesisEngine(helix)
    outputs = [
        {"content": "Python is great for ML", "confidence": 0.9},
        {"content": "Python excels at ML", "confidence": 0.85},
    ]
    result = engine.synthesize(outputs, "Evaluate Python for ML", 2)
    assert result["temperature"] == 0.2  # High consensus
```

---

### 2. WebSearchCoordinator

**File**: [src/communication/web_search_coordinator.py](../src/communication/web_search_coordinator.py)

**Purpose**: Manages web search requests with confidence-based triggering.

**Key Responsibilities:**
- Monitor team confidence levels
- Trigger searches when confidence drops
- Route searches to WebSearchClient
- Cache and distribute results
- Filter by domain/relevance

**Confidence Thresholds:**
- **>0.80**: No search needed (team confident)
- **0.60-0.80**: Optional search (moderate confidence)
- **<0.60**: Trigger search (low confidence)

**Flow:**
```
1. Agent requests search OR confidence drops
2. Check cache for similar query
3. If cached: return cached results
4. If not: WebSearchClient.search()
5. Cache results (TTL: 1 hour)
6. Broadcast to requesting agents
7. Record search in history
```

**Usage:**
```python
results = central_post.web_search_coordinator.search(
    query="latest AI trends",
    agent_id="agent_123",
    max_results=10
)
# Returns: [SearchResult(title, url, snippet, domain), ...]
```

**Caching Strategy:**
```python
def get_cache_key(query: str) -> str:
    # Normalize query for cache matching
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()

def should_trigger_search(confidence: float) -> bool:
    return confidence < 0.60
```

---

### 3. SystemCommandManager

**File**: [src/communication/system_command_manager.py](../src/communication/system_command_manager.py)

**Purpose**: Handles system command execution with trust and approval workflows.

**Key Responsibilities:**
- Parse SYSTEM_ACTION_NEEDED requests
- Classify commands by trust level
- Manage approval workflows
- Execute safe commands immediately
- Queue commands needing approval
- Track command history

**Trust Classification:**

| Trust Level | Examples | Action |
|-------------|----------|--------|
| SAFE | ls, pwd, date, pip list | Auto-execute |
| REVIEW | mkdir, pip install, file writes | Require approval |
| BLOCKED | rm -rf, credential access | Never execute |

**Command Flow:**
```
1. Agent sends SYSTEM_ACTION_REQUEST
2. Extract command from message
3. TrustManager.classify_command()
4. If SAFE: Execute immediately
5. If REVIEW: Queue for approval
6. If BLOCKED: Send denial message
7. Return result to agent
8. Record in command history
```

**Approval Workflow:**
```python
class SystemCommandManager:
    def handle_command_request(self, message):
        command = self.parse_command(message.content)
        trust_level = self.trust_manager.classify(command)

        if trust_level == TrustLevel.SAFE:
            return self.execute_immediately(command)
        elif trust_level == TrustLevel.REVIEW:
            return self.queue_for_approval(command, message.sender_id)
        else:  # BLOCKED
            return self.send_denial(message.sender_id, command)

    def queue_for_approval(self, command, agent_id):
        approval_id = str(uuid.uuid4())
        self.pending_approvals[approval_id] = {
            "command": command,
            "agent_id": agent_id,
            "timestamp": time.time()
        }
        # Pause workflow
        self.pause_event.clear()
        # GUI will handle approval dialog
        return {"status": "pending", "approval_id": approval_id}
```

**Usage:**
```python
result = central_post.system_command_manager.execute_command(
    command="mkdir new_directory",
    agent_id="agent_123",
    context="Creating output directory"
)
# Returns: CommandResult(stdout, stderr, exit_code, status)
```

---

### 4. StreamingCoordinator

**File**: [src/communication/streaming_coordinator.py](../src/communication/streaming_coordinator.py)

**Purpose**: Manages real-time token streaming with callbacks.

**Key Responsibilities:**
- Register streaming callbacks
- Batch tokens for efficient delivery
- Handle time-based batching (0.1s intervals)
- Distribute tokens to subscribers
- Clean up completed streams

**Batching Strategy:**
```python
class StreamingCoordinator:
    def __init__(self):
        self.batch_interval = 0.1  # seconds
        self.token_buffers = {}
        self.callbacks = {}

    def add_token(self, stream_id, token):
        if stream_id not in self.token_buffers:
            self.token_buffers[stream_id] = []
            self.start_batch_timer(stream_id)

        self.token_buffers[stream_id].append(token)

    def flush_batch(self, stream_id):
        tokens = self.token_buffers.get(stream_id, [])
        if tokens and stream_id in self.callbacks:
            batch = "".join(tokens)
            self.callbacks[stream_id](batch)
            self.token_buffers[stream_id] = []
```

**Usage:**
```python
# Register callback
stream_id = central_post.streaming_coordinator.register_callback(
    agent_id="agent_123",
    callback=lambda text: print(f"Received: {text}")
)

# Tokens are automatically batched and delivered
# No need to manually flush
```

**Time-Batched Delivery:**
```
Token stream:   A B C D E F G H I J
Time:          |--0.1s--|--0.1s--|--0.1s--|
Batches:        ABC      DEF      GHI     J
Callbacks:     └─called └─called └─called └─called
```

---

### 5. MemoryFacade

**File**: [src/communication/memory_facade.py](../src/communication/memory_facade.py)

**Purpose**: Provides simplified memory access interface for agents.

**Key Responsibilities:**
- Abstract complex memory operations
- Provide unified query interface
- Handle memory system initialization
- Cache frequently accessed data
- Manage memory compression

**Unified Interface:**
```python
class MemoryFacade:
    def __init__(self):
        self.knowledge_store = KnowledgeStore()
        self.task_memory = TaskMemory()
        self.workflow_history = WorkflowHistory()

    def query_knowledge(self, query, domain=None, min_confidence=None):
        """Simple knowledge query hiding complexity."""
        return self.knowledge_store.search(
            query=query,
            domain=domain,
            min_confidence=min_confidence or ConfidenceLevel.MEDIUM
        )

    def store_insight(self, content, domain, confidence, agent_id):
        """Simple storage hiding compression logic."""
        entry = KnowledgeEntry(...)
        self.knowledge_store.store(entry)

        # Auto-compress if needed
        if self.knowledge_store.size() > THRESHOLD:
            self.knowledge_store.compress()

    def recall_similar_tasks(self, task_description):
        """Find similar past tasks."""
        return self.task_memory.find_similar(task_description)
```

**Usage:**
```python
# Simple interface for agents
results = central_post.memory_facade.query_knowledge(
    query="machine learning algorithms",
    domain="technical"
)

# Complex memory operations hidden
central_post.memory_facade.store_insight(
    content={"concept": "Neural Networks", "definition": "..."},
    domain="AI",
    confidence=0.85,
    agent_id="agent_123"
)
```

---

### 6. PerformanceMonitor

**File**: [src/communication/performance_monitor.py](../src/communication/performance_monitor.py)

**Purpose**: Tracks communication and agent performance metrics.

**Key Responsibilities:**
- Record message latency
- Track agent performance
- Monitor throughput
- Calculate overhead ratios
- Generate performance reports

**Metrics Collected:**

| Metric | Description | Use Case |
|--------|-------------|----------|
| Message latency | Time from send to receive | Identify bottlenecks |
| Agent throughput | Tasks completed per second | Capacity planning |
| Synthesis time | Time to generate synthesis | Optimize synthesis |
| Memory operations | Read/write times | Optimize queries |
| LLM call duration | Time per LLM request | Provider comparison |

**Recording:**
```python
class PerformanceMonitor:
    def record_message(self, message, latency_ms):
        self.message_metrics.append({
            "message_id": message.message_id,
            "type": message.message_type,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })

    def record_agent_performance(self, agent_id, task_time, success):
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = []

        self.agent_metrics[agent_id].append({
            "task_time": task_time,
            "success": success,
            "timestamp": time.time()
        })
```

**Usage:**
```python
# Record metrics during operations
with central_post.performance_monitor.measure("synthesis"):
    result = synthesis_engine.synthesize(...)

# Generate report
report = central_post.performance_monitor.generate_report()
print(f"Avg message latency: {report['avg_latency_ms']:.2f}ms")
print(f"Total throughput: {report['messages_per_second']:.2f}/s")
```

---

## Integration Patterns

### Pattern 1: Request-Response

Used for synchronous operations where immediate response needed.

```python
# In CentralPost
def handle_synthesis_request(self, agent_outputs):
    result = self.synthesis_engine.synthesize(
        agent_outputs=agent_outputs,
        task_description=self.current_task
    )
    return result
```

### Pattern 2: Fire-and-Forget

Used for asynchronous operations with no immediate response.

```python
# In CentralPost
def handle_metrics_update(self, message):
    self.performance_monitor.record_message(message, latency)
    # No return value needed
```

### Pattern 3: Event-Driven

Used for coordinators that need to react to state changes.

```python
# In CentralPost
def on_confidence_change(self, new_confidence):
    if self.web_search_coordinator.should_trigger(new_confidence):
        self.web_search_coordinator.trigger_search(
            query=self.generate_search_query(),
            confidence=new_confidence
        )
```

### Pattern 4: Cascading Delegation

Used when coordinators need to interact with each other.

```python
# SystemCommandManager uses MemoryFacade
class SystemCommandManager:
    def __init__(self, memory_facade):
        self.memory_facade = memory_facade

    def execute_command(self, command):
        result = self.executor.execute(command)
        # Store in history via MemoryFacade
        self.memory_facade.store_command_result(result)
        return result
```

---

## Adding New Coordinators

### Step 1: Define Responsibilities

Clearly define what the coordinator will handle:
- What problem does it solve?
- What's its single responsibility?
- Does it overlap with existing coordinators?

**Example**: Add RateLimiter coordinator
- Problem: Too many LLM requests overwhelm providers
- Responsibility: Enforce rate limits per provider
- No overlap: PerformanceMonitor tracks metrics but doesn't limit

### Step 2: Create Coordinator Class

```python
# src/communication/rate_limiter_coordinator.py

import time
from collections import deque
from typing import Dict

class RateLimiterCoordinator:
    """
    Enforces rate limits for LLM providers.

    Tracks request timestamps and blocks when limits exceeded.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, deque] = {}

    def can_send_request(self, provider_name: str) -> bool:
        """Check if request allowed under rate limit."""
        if provider_name not in self.request_history:
            self.request_history[provider_name] = deque()

        history = self.request_history[provider_name]
        now = time.time()
        cutoff = now - 60  # 1 minute ago

        # Remove old requests
        while history and history[0] < cutoff:
            history.popleft()

        # Check limit
        return len(history) < self.requests_per_minute

    def record_request(self, provider_name: str):
        """Record a request timestamp."""
        if provider_name not in self.request_history:
            self.request_history[provider_name] = deque()

        self.request_history[provider_name].append(time.time())

    def wait_if_needed(self, provider_name: str):
        """Block until request allowed."""
        while not self.can_send_request(provider_name):
            time.sleep(0.1)
```

### Step 3: Integrate with CentralPost

```python
# src/communication/central_post.py

class CentralPost:
    def __init__(self, helix):
        # ... existing coordinators ...

        # Add new coordinator
        self.rate_limiter = RateLimiterCoordinator(requests_per_minute=60)

    def send_llm_request(self, provider, request):
        # Use new coordinator
        self.rate_limiter.wait_if_needed(provider.name)
        response = provider.complete(request)
        self.rate_limiter.record_request(provider.name)
        return response
```

### Step 4: Add Tests

```python
# tests/unit/communication/test_rate_limiter_coordinator.py

import time
import pytest
from src.communication.rate_limiter_coordinator import RateLimiterCoordinator

def test_allows_requests_under_limit():
    limiter = RateLimiterCoordinator(requests_per_minute=5)

    # Should allow 5 requests
    for _ in range(5):
        assert limiter.can_send_request("test_provider")
        limiter.record_request("test_provider")

    # 6th should be blocked
    assert not limiter.can_send_request("test_provider")

def test_releases_after_time_window():
    limiter = RateLimiterCoordinator(requests_per_minute=2)

    # Fill limit
    limiter.record_request("test_provider")
    limiter.record_request("test_provider")
    assert not limiter.can_send_request("test_provider")

    # Wait for window to expire
    time.sleep(61)

    # Should allow again
    assert limiter.can_send_request("test_provider")
```

### Step 5: Document

Add to [src/communication/_index.md](../src/communication/_index.md):

```markdown
### [rate_limiter_coordinator.py](rate_limiter_coordinator.py)
Enforces rate limits for LLM providers.
- **`RateLimiterCoordinator`**: Tracks request timestamps and blocks when limits exceeded
```

---

## Best Practices

### 1. Single Responsibility
Each coordinator should do **one thing well**.

❌ **Bad**: `SearchAndCacheCoordinator` (two responsibilities)
✅ **Good**: `WebSearchCoordinator` (search) + `CacheManager` (caching)

### 2. No Coordinator-to-Coordinator Direct Calls
Coordinators should not call each other directly. Go through CentralPost.

❌ **Bad**:
```python
class SynthesisEngine:
    def synthesize(self, ...):
        # Direct call to another coordinator
        self.web_search_coordinator.search(...)  # BAD!
```

✅ **Good**:
```python
class SynthesisEngine:
    def synthesize(self, ...):
        # Return signal, let CentralPost coordinate
        return {"needs_search": True, "query": "..."}

class CentralPost:
    def handle_synthesis(self, ...):
        result = self.synthesis_engine.synthesize(...)
        if result.get("needs_search"):
            self.web_search_coordinator.search(result["query"])
```

### 3. Dependency Injection
Pass dependencies to coordinators, don't create them internally.

❌ **Bad**:
```python
class SystemCommandManager:
    def __init__(self):
        self.executor = SystemExecutor()  # Hard-coded dependency
```

✅ **Good**:
```python
class SystemCommandManager:
    def __init__(self, executor: SystemExecutor):
        self.executor = executor  # Injected dependency
```

### 4. Clear Interfaces
Define what coordinators expose to CentralPost.

```python
class ICoordinator(ABC):
    """Base interface for all coordinators."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize coordinator resources."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up coordinator resources."""
        pass
```

### 5. Error Handling
Coordinators should handle errors internally and return status.

```python
class WebSearchCoordinator:
    def search(self, query):
        try:
            results = self.client.search(query)
            return {"status": "success", "results": results}
        except SearchError as e:
            logger.error(f"Search failed: {e}")
            return {"status": "error", "message": str(e)}
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Extending Felix
- [src/communication/_index.md](../src/communication/_index.md) - Communication module overview
