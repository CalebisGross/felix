# System Prompts Directory

## Purpose
Storage for system-wide prompt templates and agent instructions (currently stored in database `felix_prompts.db`).

## Key Files

### [felix_prompts.db](felix_prompts.db)
SQLite database containing prompt templates.
- **Prompt storage**: All agent prompts stored in database
- **Version tracking**: Prompt versioning for A/B testing
- **Template management**: Create, read, update, delete operations
- **Variable substitution**: Dynamic content insertion
- **Accessed via**: `PromptManager` class in `src/prompts/prompt_manager.py`

**Note**: Future versions may migrate to YAML files like `config/prompts.yaml` for easier version control.

## Prompt System Architecture

### Prompt Manager
Central system for prompt loading and management:
```python
from src.prompts.prompt_manager import PromptManager

manager = PromptManager()

# Load agent prompt
research_prompt = manager.get_prompt("research_agent")

# With variable substitution
prompt = manager.get_prompt(
    "research_agent",
    variables={
        "task": "Explain quantum computing",
        "position": 0.15,
        "domain": "physics"
    }
)
```

### Prompt Types

#### 1. Agent Role Prompts
Define agent behavior and specialization:
- **Research Agent**: Broad information gathering instructions
- **Analysis Agent**: Synthesis and processing guidelines
- **Critic Agent**: Validation and quality assurance criteria

#### 2. Position-Aware Prompts
Adapt based on helix position:
- **Top of helix** (0.0-0.3): High creativity, broad exploration
- **Mid helix** (0.3-0.6): Balanced analysis, pattern recognition
- **Bottom of helix** (0.6-1.0): Focused synthesis, low temperature

#### 3. System Prompts
General instructions for all agents:
- **Core guidelines**: Accuracy, relevance, conciseness
- **Output format**: Structured responses, markdown
- **Safety guidelines**: Avoid harmful content, bias awareness

#### 4. Task-Specific Prompts
Specialized prompts for specific tasks:
- **Code review**: Focus on security, quality, best practices
- **Research**: Emphasize sources, citations, perspectives
- **Analysis**: Highlight patterns, insights, conclusions

### Prompt Variables

Templates support dynamic variable insertion:
- **{task}**: Current task description
- **{position}**: Agent's normalized time/helix position
- **{domain}**: Task domain (general, technical, creative, etc.)
- **{context}**: Additional context from knowledge or previous agents
- **{role}**: Agent type (research, analysis, critic)
- **{temperature}**: Current temperature setting
- **{confidence_threshold}**: Required confidence level

### Example Prompt Template

```yaml
research_agent:
  template: |
    You are a Research Agent in the Felix multi-agent system.

    **Your Position**: {position} (0.0 = exploration, 1.0 = synthesis)
    **Current Temperature**: {temperature}
    **Your Role**: Broad information gathering and exploration

    **Task**: {task}

    **Instructions**:
    1. Gather comprehensive information from multiple perspectives
    2. Identify key concepts, entities, and relationships
    3. Consider diverse viewpoints and sources
    4. Assess information reliability and relevance
    5. Provide your findings with confidence score (0.0-1.0)

    **Output Format**:
    - Main findings (clear, structured)
    - Key concepts identified
    - Sources or reasoning
    - Confidence score
    - Suggestions for further investigation

    Remember: At position {position}, your focus is on {phase_focus}.
```

### Prompt Optimization

Prompts are optimized through:
1. **A/B testing**: Compare prompt versions
2. **Performance tracking**: Monitor agent output quality
3. **User feedback**: Incorporate user ratings
4. **Iterative refinement**: Update based on results

### Future Migration

**Planned**: Migrate from SQLite to YAML files:
- Easier version control with Git
- Simple editing without database tools
- Transparent prompt history
- Better collaboration on prompt improvements

Migration path:
1. Export existing prompts from `felix_prompts.db`
2. Create `prompts/*.yaml` files (one per agent type)
3. Update `PromptManager` to load from YAML
4. Keep database as fallback/cache

## Prompt Guidelines

### Effective Prompts:
- **Clear role definition**: Specify agent's purpose
- **Structured instructions**: Numbered steps or bullet points
- **Output format**: Define expected response structure
- **Context awareness**: Reference position on helix
- **Confidence scoring**: Request self-assessment

### Avoid:
- Overly long prompts (> 500 words)
- Ambiguous instructions
- Conflicting guidelines
- Hardcoded values (use variables)
- Bias or leading questions

## Related Modules
- [src/prompts/](../src/prompts/) - Prompt management system
- [src/agents/llm_agent.py](../src/agents/llm_agent.py) - Uses prompts for LLM calls
- [config/prompts.yaml](../config/prompts.yaml) - Prompt configuration (alternative storage)
- [src/agents/specialized_agents.py](../src/agents/specialized_agents.py) - Agent implementations using prompts
