# Felix Developer Guide

Complete guide for developing and extending the Felix framework.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Code Organization](#code-organization)
4. [Extension Points](#extension-points)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Best Practices](#best-practices)

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool
- (Optional) LM Studio for local LLM testing

### Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd felix

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Initialize databases
python -m src.cli init

# Verify setup
python -m src.cli test-connection
```

### IDE Configuration

**VS Code** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true
}
```

**PyCharm**:
1. File → Settings → Project → Python Interpreter
2. Select `.venv/bin/python`
3. Enable pytest as test runner
4. Configure Black as formatter

---

## Project Structure

### High-Level Layout

```
felix/
├── src/                    # Main source code
│   ├── agents/            # Agent implementations
│   ├── api/               # REST API
│   ├── cli.py             # CLI interface
│   ├── communication/     # Hub-spoke messaging
│   ├── core/              # Core algorithms (helix)
│   ├── execution/         # System command execution
│   ├── feedback/          # Learning and feedback
│   ├── gui/               # Tkinter GUI
│   ├── knowledge/         # Knowledge brain
│   ├── learning/          # Learning systems
│   ├── llm/               # LLM integration
│   ├── memory/            # Persistence
│   ├── migration/         # Database migrations
│   ├── pipeline/          # Processing pipelines
│   ├── prompts/           # Prompt management
│   ├── utils/             # Utilities
│   └── workflows/         # High-level orchestration
│
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py       # Pytest configuration
│
├── config/               # Configuration files
│   ├── llm.yaml         # LLM providers
│   ├── prompts.yaml     # Prompt templates
│   └── trust_rules.yaml # Command trust rules
│
├── docs/                # Documentation
└── examples/            # Example code
```

### Module Dependencies

```
┌─────────────┐
│  Workflows  │ ← High-level orchestration
└──────┬──────┘
       │
┌──────▼──────┐
│Communication│ ← Hub-spoke messaging
└──────┬──────┘
       │
   ┌───┴────┬─────────┬─────────┐
   │        │         │         │
┌──▼───┐ ┌─▼──┐  ┌───▼───┐ ┌───▼────┐
│Agents│ │LLM │  │Memory │ │Execution│
└──────┘ └────┘  └───────┘ └─────────┘
   │        │         │
   └────────┼─────────┘
            │
      ┌─────▼─────┐
      │   Core    │ ← Foundation (helix, etc.)
      └───────────┘
```

**Import Rules:**
- Higher layers can import lower layers
- Lower layers cannot import higher layers
- Avoid circular imports (use TYPE_CHECKING)

---

## Code Organization

### Naming Conventions

**Files:**
- Module files: `snake_case.py`
- Test files: `test_<module>.py`
- Configuration: `<name>.yaml`

**Classes:**
- Classes: `PascalCase`
- Abstract classes: `BaseSomething` or `ISomething`
- Exceptions: `SomethingError`

**Functions/Methods:**
- Functions: `snake_case()`
- Private methods: `_private_method()`
- Special methods: `__special__()`

**Variables:**
- Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_private_var`

### Code Style

Felix follows **PEP 8** with some modifications:

```python
# Maximum line length: 100 characters (not 79)
# Use Black formatter for consistency

# Example module structure:
"""
Module docstring explaining purpose.

Detailed description if needed.
"""

import stdlib  # Standard library first
import third_party  # Third-party second

from src.module import something  # Local imports last

# Constants
DEFAULT_TIMEOUT = 60
MAX_AGENTS = 133

# Classes
class MyClass:
    """Class docstring."""

    def __init__(self, param: str):
        """Initialize with parameters."""
        self.param = param

    def public_method(self) -> str:
        """Public method docstring."""
        return self._private_method()

    def _private_method(self) -> str:
        """Private method for internal use."""
        return self.param

# Functions
def my_function(arg: str) -> int:
    """
    Function docstring.

    Args:
        arg: Argument description

    Returns:
        Return value description

    Raises:
        ValueError: When something is wrong
    """
    if not arg:
        raise ValueError("arg cannot be empty")
    return len(arg)
```

### Type Hints

Use type hints for better IDE support and documentation:

```python
from typing import List, Dict, Optional, Tuple, Callable

def process_agents(
    agents: List[Agent],
    config: Dict[str, any],
    callback: Optional[Callable[[str], None]] = None
) -> Tuple[bool, str]:
    """Process agents with optional callback."""
    for agent in agents:
        result = agent.process()
        if callback:
            callback(f"Processed {agent.agent_id}")
    return True, "Success"
```

---

## Extension Points

### 1. Adding a New Agent Type

**Step 1: Create Agent Class**

```python
# src/agents/my_custom_agent.py

from src.agents.llm_agent import LLMAgent
from typing import Dict, Any

class MyCustomAgent(LLMAgent):
    """
    Custom agent for specialized tasks.

    This agent does X, Y, and Z.
    """

    def __init__(self, agent_id: str, helix, llm_client, position):
        super().__init__(
            agent_id=agent_id,
            helix=helix,
            llm_client=llm_client,
            position=position
        )
        self.agent_type = "custom"

    def get_system_prompt(self) -> str:
        """Return custom system prompt."""
        return """You are a custom agent specialized in...

        Your responsibilities:
        - Task 1
        - Task 2
        """

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with custom logic."""
        # Custom preprocessing
        enhanced_task = self._preprocess(task)

        # Call parent LLM processing
        result = super().process_task(enhanced_task)

        # Custom postprocessing
        final_result = self._postprocess(result)

        return final_result

    def _preprocess(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Custom preprocessing logic."""
        # Add custom fields, validate, etc.
        return task

    def _postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Custom postprocessing logic."""
        # Parse, validate, enhance results
        return result
```

**Step 2: Create Plugin Wrapper**

```python
# src/agents/builtin/my_custom_plugin.py

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.my_custom_agent import MyCustomAgent

class MyCustomAgentPlugin(SpecializedAgentPlugin):
    """Plugin wrapper for MyCustomAgent."""

    def get_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="custom",
            version="1.0.0",
            description="Custom agent for specialized tasks",
            capabilities=["task1", "task2", "task3"],
            spawn_time_range=(0.3, 0.7),  # When agent should spawn
            requirements={}
        )

    def create_agent(self, agent_id, helix, llm_client, position):
        return MyCustomAgent(agent_id, helix, llm_client, position)
```

**Step 3: Register Plugin**

```python
# Automatic registration via AgentPluginRegistry
# Just place file in src/agents/builtin/ or custom plugin directory
```

**Step 4: Use in Workflow**

```python
# Agent factory will automatically discover and use plugin
factory = AgentFactory(central_post, helix, llm_client)

# Create custom agent
agent = factory.create_agent(
    agent_type="custom",  # Matches plugin metadata name
    agent_id="custom_001"
)
```

### 2. Adding a New Coordinator

See [COORDINATOR_ARCHITECTURE.md](COORDINATOR_ARCHITECTURE.md) for detailed guide.

**Quick Example:**

```python
# src/communication/cache_coordinator.py

class CacheCoordinator:
    """Manages caching for agent responses."""

    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_size = max_cache_size

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """Cache value with size limit."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

Add to CentralPost:

```python
# src/communication/central_post.py

class CentralPost:
    def __init__(self, helix):
        # ... existing coordinators ...
        self.cache_coordinator = CacheCoordinator()
```

### 3. Adding a New LLM Provider

See [LLM_PROVIDER_GUIDE.md](LLM_PROVIDER_GUIDE.md) for complete tutorial.

**Quick Steps:**
1. Implement `BaseLLMProvider`
2. Add to `ProviderType` enum
3. Update `create_provider()` in config loader
4. Add tests
5. Document in llm/_index.md

### 4. Adding a New Workflow

```python
# src/workflows/my_workflow.py

from src.workflows.felix_workflow import FelixWorkflow
from typing import Dict, Any

def execute_my_workflow(
    task_input: str,
    felix_system,
    custom_param: str
) -> Dict[str, Any]:
    """
    Execute custom workflow.

    Args:
        task_input: Task description
        felix_system: Felix system instance
        custom_param: Custom parameter

    Returns:
        Workflow results
    """
    # Step 1: Classify task
    complexity = classify_task(task_input)

    # Step 2: Spawn appropriate agents
    agents = spawn_agents_for_complexity(complexity, felix_system)

    # Step 3: Process in custom way
    for agent in agents:
        result = agent.process_task({
            "task": task_input,
            "custom": custom_param
        })
        # Do something with result

    # Step 4: Synthesize
    synthesis = felix_system.central_post.synthesize_agent_outputs(
        agent_outputs=[...],
        task_description=task_input
    )

    return {
        "synthesis": synthesis,
        "agents_used": len(agents),
        "custom_results": [...]
    }
```

### 5. Adding Custom Knowledge Source

```python
# src/knowledge/custom_source.py

from src.knowledge.document_reader import DocumentReader

class CustomSourceReader(DocumentReader):
    """Read from custom knowledge source."""

    def supports_file(self, file_path: str) -> bool:
        """Check if file type is supported."""
        return file_path.endswith('.custom')

    def read_document(self, file_path: str) -> str:
        """Read and parse custom format."""
        with open(file_path, 'r') as f:
            content = f.read()

        # Custom parsing logic
        parsed = self._parse_custom_format(content)
        return parsed

    def _parse_custom_format(self, content: str) -> str:
        """Parse custom format into plain text."""
        # Implementation here
        return content
```

Register with knowledge brain:

```python
# src/knowledge/comprehension_engine.py

# Add custom reader
from src.knowledge.custom_source import CustomSourceReader

class KnowledgeComprehensionEngine:
    def __init__(self):
        self.document_reader = DocumentReader()
        self.custom_reader = CustomSourceReader()

    def process_document(self, file_path: str):
        if self.custom_reader.supports_file(file_path):
            content = self.custom_reader.read_document(file_path)
        else:
            content = self.document_reader.read_document(file_path)
        # ... continue processing
```

---

## Testing

### Test Organization

```
tests/
├── unit/                  # Unit tests (isolated)
│   ├── agents/
│   │   ├── test_agent.py
│   │   └── test_llm_agent.py
│   ├── communication/
│   │   ├── test_central_post.py
│   │   └── test_synthesis_engine.py
│   └── llm/
│       └── test_llm_router.py
│
├── integration/          # Integration tests
│   ├── test_workflow_execution.py
│   └── test_agent_communication.py
│
└── conftest.py          # Shared fixtures
```

### Writing Unit Tests

```python
# tests/unit/agents/test_my_agent.py

import pytest
from unittest.mock import Mock, patch
from src.agents.my_custom_agent import MyCustomAgent
from src.core.helix_geometry import HelixGeometry

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock()
    client.chat_completion.return_value = {
        "content": "Test response",
        "tokens_used": 50
    }
    return client

@pytest.fixture
def helix():
    """Create helix geometry for testing."""
    return HelixGeometry(
        top_radius=3.0,
        bottom_radius=0.5,
        height=8.0,
        turns=2
    )

def test_agent_creation(helix, mock_llm_client):
    """Test agent can be created."""
    agent = MyCustomAgent(
        agent_id="test_001",
        helix=helix,
        llm_client=mock_llm_client,
        position=0.5
    )

    assert agent.agent_id == "test_001"
    assert agent.agent_type == "custom"

def test_process_task(helix, mock_llm_client):
    """Test task processing."""
    agent = MyCustomAgent("test_001", helix, mock_llm_client, 0.5)

    task = {"description": "Test task"}
    result = agent.process_task(task)

    assert "output" in result
    assert result["confidence"] > 0

def test_process_task_error_handling(helix, mock_llm_client):
    """Test error handling in processing."""
    mock_llm_client.chat_completion.side_effect = Exception("LLM error")

    agent = MyCustomAgent("test_001", helix, mock_llm_client, 0.5)

    with pytest.raises(Exception):
        agent.process_task({"description": "Test"})
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/agents/test_my_agent.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run integration tests only
pytest tests/integration/

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_agent"
```

### Test Best Practices

1. **Use Fixtures**: Share setup code via pytest fixtures
2. **Mock External Dependencies**: Don't call real APIs in tests
3. **Test Edge Cases**: Empty inputs, None values, errors
4. **One Assert Per Test**: Keep tests focused
5. **Clear Test Names**: Describe what is being tested
6. **Test Isolation**: Tests should not depend on each other

---

## Debugging

### Logging

Felix uses Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Detailed debug information")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error occurred")
    logger.exception("Exception with traceback")
```

**Enable Debug Logging:**

```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or configure per-module
logging.getLogger('src.agents').setLevel(logging.DEBUG)
```

### Common Debugging Scenarios

**1. Agent Not Spawning:**

```python
# Check spawn time ranges
print(f"Agent spawn range: {agent_plugin.get_metadata().spawn_time_range}")
print(f"Current normalized time: {workflow.normalized_time}")

# Verify agent type registered
from src.agents.agent_plugin_registry import AgentPluginRegistry
registry = AgentPluginRegistry()
print(f"Registered agents: {registry.list_plugins()}")
```

**2. LLM Provider Failing:**

```python
# Test provider directly
from src.llm.router_adapter import create_router_adapter

adapter = create_router_adapter()

# Test connection
if adapter.test_connection():
    print("✓ Provider connected")
else:
    print("✗ Provider failed")

# Check statistics
stats = adapter.router.get_statistics()
print(f"Success rate: {stats['overall_success_rate']:.1%}")
```

**3. Memory Issues:**

```python
# Check database size
import os
dbs = ["felix_knowledge.db", "felix_memory.db"]
for db in dbs:
    if os.path.exists(db):
        size_mb = os.path.getsize(db) / (1024 * 1024)
        print(f"{db}: {size_mb:.1f} MB")

# Check knowledge entries
from src.memory.knowledge_store import KnowledgeStore
store = KnowledgeStore()
entries = store.get_all_entries()
print(f"Total entries: {len(entries)}")
```

### Using Python Debugger (pdb)

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use Python 3.7+ breakpoint()
breakpoint()

# Common pdb commands:
# n - next line
# s - step into
# c - continue
# p variable - print variable
# l - list code
# q - quit
```

### VS Code Debugging

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Felix CLI",
            "type": "python",
            "request": "launch",
            "module": "src.cli",
            "args": ["run", "Test task"],
            "console": "integratedTerminal"
        },
        {
            "name": "Felix GUI",
            "type": "python",
            "request": "launch",
            "module": "src.gui",
            "console": "integratedTerminal"
        },
        {
            "name": "Pytest",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

---

## Best Practices

### 1. Error Handling

**Always handle errors gracefully:**

```python
# ❌ Bad: Let errors crash
def process_data(data):
    return data['key']  # KeyError if key missing

# ✅ Good: Handle specific errors
def process_data(data):
    try:
        return data['key']
    except KeyError:
        logger.warning(f"Missing key in data: {data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 2. Resource Cleanup

**Use context managers:**

```python
# ❌ Bad: Manual cleanup
file = open('data.txt')
data = file.read()
file.close()

# ✅ Good: Automatic cleanup
with open('data.txt') as file:
    data = file.read()
```

### 3. Configuration

**Use configuration files, not hardcoded values:**

```python
# ❌ Bad: Hardcoded
timeout = 60
model = "gpt-4"

# ✅ Good: From config
from src.llm.provider_config import load_config
config = load_config()
timeout = config.get('timeout', 60)
model = config.get('model', 'default-model')
```

### 4. Documentation

**Document complex logic:**

```python
def calculate_confidence(outputs):
    """
    Calculate team confidence from agent outputs.

    Uses weighted average based on individual confidences
    and output agreement. Higher agreement increases confidence.

    Algorithm:
    1. Calculate base confidence (average of individual scores)
    2. Calculate agreement factor (similarity between outputs)
    3. Apply agreement boost (up to 20%)

    Args:
        outputs: List of agent outputs with confidence scores

    Returns:
        float: Team confidence score (0.0-1.0)
    """
    # Implementation...
```

### 5. Code Review Checklist

Before submitting:
- [ ] Tests added and passing
- [ ] Type hints on public functions
- [ ] Docstrings on public classes/functions
- [ ] No hardcoded values (use config)
- [ ] Error handling for edge cases
- [ ] Logging at appropriate levels
- [ ] No debugging print statements
- [ ] Code formatted with Black
- [ ] No linter warnings

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [COORDINATOR_ARCHITECTURE.md](COORDINATOR_ARCHITECTURE.md) - Coordinator pattern
- [LLM_PROVIDER_GUIDE.md](LLM_PROVIDER_GUIDE.md) - Provider development
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [API_DEVELOPMENT.md](API_DEVELOPMENT.md) - API extension
