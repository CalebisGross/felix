# Configuration Files

## Purpose
YAML configuration files for LLM providers, prompts, system behavior, and trust management.

## Key Files

### [llm.yaml](llm.yaml)
Multi-provider LLM configuration with routing and fallback.
- **Provider definitions**: LM Studio, Anthropic, Gemini configurations
- **Priority ordering**: Primary, fallback, tertiary providers
- **Connection settings**: Base URLs, API keys, timeouts
- **Model selection**: Specify models per provider (e.g., claude-3-sonnet, gemini-1.5-pro)
- **Routing strategy**: Automatic failover, load balancing
- **Example configuration**:
  ```yaml
  providers:
    - type: lm_studio
      base_url: http://localhost:1234/v1
      priority: 1
      timeout: 120

    - type: anthropic
      model: claude-3-sonnet-20240229
      priority: 2
      api_key_env: ANTHROPIC_API_KEY

    - type: gemini
      model: gemini-1.5-pro
      priority: 3
      api_key_env: GOOGLE_API_KEY
  ```

### [llm.yaml.example](llm.yaml.example)
Example LLM configuration template.
- Copy to `llm.yaml` and customize
- Shows all available provider options
- Documents configuration parameters
- Safe to commit (no secrets)

### [prompts.yaml](prompts.yaml)
Agent prompt templates and system messages.
- **Agent prompts**: Research, Analysis, Critic agent instructions
- **Position-aware prompts**: Template variables for helix position
- **System prompts**: General instructions and guidelines
- **Prompt versioning**: Track prompt changes and A/B testing
- **Variable substitution**: Dynamic insertion of context, position, domain
- **Example structure**:
  ```yaml
  agents:
    research:
      template: |
        You are a Research Agent at position {position} on the helix.
        Your role is to gather comprehensive information about: {task}

    analysis:
      template: |
        You are an Analysis Agent at position {position}.
        Synthesize the following information: {context}
  ```

### [trust_rules.yaml](trust_rules.yaml)
Trust levels and approval rules for system command execution.

### [tool_requirements_patterns.yaml](tool_requirements_patterns.yaml)
Pattern definitions for classifying which tools tasks require.
- **Purpose**: Used by `SynthesisEngine.classify_tool_requirements()` for conditional tool memory
- **Categories**: `file_operations`, `web_search`, `system_commands`
- **Pattern syntax**: Case-insensitive Python regex
- **File operations patterns**: File reading, writing, discovery, path detection
- **Web search patterns**: Current/real-time information, explicit search requests
- **System command patterns**: Package management, process control, system operations
- **Integration**: Determines which tool instructions agents receive (reduces token usage by 40-60%)

### [task_complexity_patterns.yaml](task_complexity_patterns.yaml)
Pattern definitions for classifying task complexity levels.
- **Purpose**: Used by `SynthesisEngine.classify_task_complexity()` for prompt optimization
- **Complexity levels**: `simple_factual`, `medium`, `complex` (default)
- **Pattern syntax**: Case-insensitive Python regex, checked in priority order
- **Simple factual**: Time/date queries, file content display, greetings
- **Medium**: Explanations, comparisons, how-to questions, file analysis
- **Complex**: Default for unmatched patterns (research, multi-step tasks)
- **Integration**: Controls verbosity constraints, stage skipping, and token allocation in `PromptPipeline`

### [trust_rules.yaml](trust_rules.yaml) (continued)
- **Trust levels**: Safe, monitored, restricted, dangerous
- **Command patterns**: Regex patterns for command classification
- **Approval requirements**: Auto-execute, user approval, admin approval
- **Sandboxing rules**: Container isolation, resource limits
- **Example structure**:
  ```yaml
  trust_levels:
    safe:
      auto_execute: true
      patterns:
        - ^ls\s
        - ^cat\s
        - ^echo\s

    dangerous:
      auto_execute: false
      require_approval: true
      patterns:
        - ^rm\s+-rf
        - ^dd\s+if=
        - ^mkfs\s
  ```

## Configuration Loading

Felix loads configuration at startup:
```python
from src.llm.provider_config import ProviderConfig

# Load LLM provider config
config = ProviderConfig.from_yaml("config/llm.yaml")

# Access providers
primary_provider = config.get_primary_provider()
all_providers = config.get_all_providers()
```

## Environment Variables

Configuration files support environment variable substitution:
```yaml
# In config file
providers:
  - type: anthropic
    api_key_env: ANTHROPIC_API_KEY  # References ${ANTHROPIC_API_KEY}

# Set in shell
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Configuration Precedence

Configuration values resolved in order:
1. **Environment variables**: Override file settings
2. **Configuration file**: Main source of truth
3. **Default values**: Fallback for missing values

## Security Best Practices

### DO:
- Use `api_key_env` to reference environment variables
- Keep secrets out of config files
- Use `.gitignore` to exclude `llm.yaml` (commit `llm.yaml.example` instead)
- Rotate API keys regularly
- Use least-privilege provider accounts

### DON'T:
- Hard-code API keys in config files
- Commit files with secrets
- Share API keys between environments
- Use admin/root provider accounts

## Configuration Validation

Felix validates configuration on load:
- Provider types must be recognized
- Required fields must be present
- API key environment variables must exist (when referenced)
- URLs must be valid format
- Numeric values within acceptable ranges

Invalid configuration causes startup failure with clear error messages.

## Hot Reload

Configuration changes require restart:
```bash
# 1. Edit configuration
vim config/llm.yaml

# 2. Restart Felix
# GUI: Stop and restart via Dashboard tab
# API: Restart uvicorn server
# CLI: Restart command
```

## Related Modules
- [src/llm/provider_config.py](../src/llm/provider_config.py) - Configuration loader
- [src/llm/llm_router.py](../src/llm/llm_router.py) - Uses provider configuration
- [src/prompts/prompt_manager.py](../src/prompts/prompt_manager.py) - Loads prompt templates
- [src/execution/trust_manager.py](../src/execution/trust_manager.py) - Uses trust rules
