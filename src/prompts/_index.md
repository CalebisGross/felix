# Prompts Module

## Purpose
Agent prompt template management providing centralized storage, retrieval, versioning, and optimization of prompts for specialized agents.

## Key Files

### [prompt_manager.py](prompt_manager.py)
Centralized prompt storage and retrieval system.
- **`PromptManager`**: Manages prompt templates, variables, and versioning

## Key Concepts

### Prompt Templates

Templates use variable substitution for dynamic content:

```python
template = """
You are a {role} agent analyzing {domain} content.
Your position on the helix: {helix_position}
Temperature: {temperature}
Task: {task_description}
"""
```

### Template Variables

Common variables:
- **`{role}`**: Agent specialization (Research, Analysis, Critic)
- **`{domain}`**: Task domain (finance, science, general, etc.)
- **`{helix_position}`**: Normalized time (0.0-1.0) for position awareness
- **`{temperature}`**: Current temperature based on position
- **`{task_description}`**: Specific task to perform
- **`{context}`**: Additional context from knowledge or previous agents
- **`{token_budget}`**: Available tokens for response

### Prompt Storage

Prompts stored in:
1. **Database**: `felix_memory.db` → `agent_prompts` table
2. **File system**: `prompts/` directory with version control

**Database Schema**:
```sql
CREATE TABLE agent_prompts (
    prompt_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    template TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON with description, author, tags
)
```

### Prompt Versioning

Version format: `{role}_v{number}`
- Example: `research_v1`, `research_v2`, `analysis_v1`

Benefits:
- Track prompt evolution
- A/B testing different versions
- Rollback to previous versions
- Performance comparison

### Prompt Optimization

Methods tracked by PromptManager:
1. **Performance metrics**: Track success rate per prompt version
2. **A/B testing**: Compare multiple prompt versions simultaneously
3. **Learning integration**: Adjust prompts based on feedback
4. **Context sensitivity**: Optimize prompts for specific domains

### Role-Specific Prompts

#### Research Agent
- Emphasizes exploration and information gathering
- Encourages wide-ranging investigation
- Asks clarifying questions
- Example: "Investigate {topic} broadly. Consider multiple perspectives..."

#### Analysis Agent
- Focuses on synthesis and pattern recognition
- Analyzes relationships and implications
- Example: "Analyze the following information. Identify key patterns..."

#### Critic Agent
- Validates claims and checks consistency
- Questions assumptions
- Example: "Review the following analysis critically. Identify gaps..."

### Dynamic Prompt Adjustment

Prompts adapt based on:
- **Helix position**: More exploratory at top, more focused at bottom
- **Token budget**: Adjust expected response length
- **Previous results**: Incorporate learnings from earlier agents
- **Domain**: Specialized vocabulary and approaches per domain

## Usage Example

```python
from src.prompts.prompt_manager import PromptManager

# Initialize manager
pm = PromptManager(db_path="felix_memory.db")

# Get prompt template
template = pm.get_prompt(role="research", version=2)

# Substitute variables
prompt = template.format(
    domain="quantum computing",
    helix_position=0.25,
    temperature=0.8,
    task_description="Analyze recent advances in quantum error correction",
    token_budget=2048
)

# Use with LLM agent
result = llm_agent.execute(prompt)
```

### Prompt Management via GUI

The [Prompts tab](../gui/prompts.py) in the GUI provides:
- Template browser and editor
- Variable substitution preview
- Version management
- A/B test configuration
- Performance metrics display

## Configuration

```yaml
prompts:
  storage_path: "prompts/"
  enable_versioning: true
  default_version: "latest"
  enable_ab_testing: false
```

## Related Modules
- [agents/](../agents/) - Specialized agents use role-specific prompts
- [llm/](../llm/) - Prompts passed to LLM clients
- [learning/](../learning/) - Prompt optimization based on learning
- [gui/](../gui/) - Prompts tab for management interface
