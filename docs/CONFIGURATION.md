# Felix Configuration Reference

Complete reference for configuring the Felix framework.

## Table of Contents

1. [Configuration Files](#configuration-files)
2. [LLM Configuration](#llm-configuration)
3. [Prompt Configuration](#prompt-configuration)
4. [Trust Rules Configuration](#trust-rules-configuration)
5. [Environment Variables](#environment-variables)
6. [Tuning Guidelines](#tuning-guidelines)

---

## Configuration Files

Felix uses YAML configuration files located in the `config/` directory:

| File | Purpose | Required |
|------|---------|----------|
| `llm.yaml` | LLM provider settings | Yes |
| `prompts.yaml` | Agent prompt templates | Optional |
| `trust_rules.yaml` | System command trust rules | Optional |

### Configuration Loading

```python
# Configurations loaded automatically by Felix
# Default location: config/

# Override location with environment variable
export FELIX_CONFIG_DIR="/path/to/config"

# Or specify in code
from src.llm.provider_config import load_config
config = load_config("custom_path/llm.yaml")
```

---

## LLM Configuration

### File Location

`config/llm.yaml`

### Basic Structure

```yaml
# Primary provider (required)
primary:
  type: "provider_type"
  # Provider-specific settings...

# Fallback providers (optional)
fallbacks:
  - type: "provider_type"
    # Provider-specific settings...

# Router settings (optional)
router:
  retry_on_rate_limit: false
  verbose_logging: false
```

### Provider Types

#### LM Studio (Local)

```yaml
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"
  timeout: 60

  # Optional settings
  max_retries: 3
  backoff_factor: 1.5
```

**Parameters:**
- `type` (required): Must be "lm_studio"
- `base_url` (required): LM Studio server URL
- `model` (optional): Model name (default: "local-model")
- `timeout` (optional): Request timeout in seconds (default: 60)
- `max_retries` (optional): Retry attempts on failure (default: 3)
- `backoff_factor` (optional): Exponential backoff multiplier (default: 1.5)

**Requirements:**
- LM Studio running on specified port
- Model loaded in LM Studio

#### Anthropic Claude

```yaml
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  timeout: 120

  # Optional settings
  max_tokens_default: 4096
  api_version: "2023-06-01"
```

**Parameters:**
- `type` (required): Must be "anthropic"
- `api_key` (required): Anthropic API key (use ${VAR} for environment variables)
- `model` (required): Model name
  - `claude-3-5-sonnet-20241022` (recommended)
  - `claude-3-opus-20240229`
  - `claude-3-sonnet-20240229`
  - `claude-3-haiku-20240307`
- `timeout` (optional): Request timeout in seconds (default: 120)
- `max_tokens_default` (optional): Default max tokens (default: 4096)
- `api_version` (optional): API version (default: "2023-06-01")

**Cost Estimates (per 1M tokens):**
- Claude 3.5 Sonnet: $3 input / $15 output
- Claude 3 Opus: $15 input / $75 output
- Claude 3 Haiku: $0.25 input / $1.25 output

#### Google Gemini

```yaml
primary:
  type: "gemini"
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-1.5-pro"
  timeout: 120

  # Optional settings
  safety_settings:
    harassment: "BLOCK_NONE"
    hate_speech: "BLOCK_NONE"
    sexually_explicit: "BLOCK_NONE"
    dangerous_content: "BLOCK_NONE"
```

**Parameters:**
- `type` (required): Must be "gemini"
- `api_key` (required): Google API key
- `model` (required): Model name
  - `gemini-1.5-pro` (1M context, $3.50/$10.50 per 1M tokens)
  - `gemini-1.5-flash` (1M context, $0.35/$1.05 per 1M tokens)
  - `gemini-1.0-pro` (32K context, $0.50/$1.50 per 1M tokens)
- `timeout` (optional): Request timeout in seconds (default: 120)
- `safety_settings` (optional): Content safety filters

**Cost Estimates (per 1M tokens):**
- Gemini 1.5 Pro: $3.50 input / $10.50 output
- Gemini 1.5 Flash: $0.35 input / $1.05 output (fastest, cheapest)

### Multi-Provider Configuration

**Scenario 1: Cloud Primary with Local Fallback**

```yaml
# Production quality with local backup
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  timeout: 120

fallbacks:
  - type: "lm_studio"
    base_url: "http://localhost:1234/v1"
    model: "mistral-7b"
    timeout: 30

router:
  retry_on_rate_limit: false
  verbose_logging: true
```

**Scenario 2: Cost-Optimized**

```yaml
# Cheap and fast primary, quality fallback
primary:
  type: "gemini"
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-1.5-flash"  # Fastest, cheapest
  timeout: 60

fallbacks:
  - type: "gemini"
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-1.5-pro"  # Better quality if flash fails
    timeout: 120

  - type: "lm_studio"
    base_url: "http://localhost:1234/v1"
    timeout: 30
```

**Scenario 3: High Availability**

```yaml
# Multiple providers for redundancy
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"

fallbacks:
  - type: "gemini"
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-1.5-pro"

  - type: "lm_studio"
    base_url: "http://localhost:1234/v1"

  - type: "lm_studio"
    base_url: "http://backup-server:1234/v1"  # Backup LM Studio

router:
  retry_on_rate_limit: true
  verbose_logging: false
  health_check_interval: 300
```

**Scenario 4: Development**

```yaml
# Local only for development
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "mistral-7b-instruct"
  timeout: 30

# No fallbacks needed for dev
```

### Router Settings

```yaml
router:
  # Retry fallbacks on rate limit errors (default: false)
  retry_on_rate_limit: false

  # Enable detailed logging (default: false)
  verbose_logging: true

  # Health check interval in seconds (default: 300)
  health_check_interval: 300

  # Maximum retry attempts per provider (default: 3)
  max_retries_per_provider: 3

  # Timeout for health checks in seconds (default: 10)
  health_check_timeout: 10
```

### Token Budget Configuration

Embedded in `llm.yaml` but affects all providers:

```yaml
token_budget:
  # Base token budget per agent (default: 2048)
  base_budget: 2048

  # Enforce strict limits (default: true)
  # When true, requests exceeding budget are truncated
  strict_mode: true

  # Budget scaling factor by agent position (default: 1.0)
  # Agents at bottom of helix get more tokens
  position_scaling: 1.0

  # Minimum tokens guaranteed per agent (default: 512)
  min_budget: 512

  # Maximum tokens per agent (default: 4096)
  max_budget: 4096
```

---

## Prompt Configuration

### File Location

`config/prompts.yaml`

### Structure

```yaml
# Agent system prompts
agents:
  research:
    system_prompt: |
      You are a research agent specialized in gathering information.

      Your responsibilities:
      - Conduct thorough research
      - Cite sources
      - Identify knowledge gaps

  analysis:
    system_prompt: |
      You are an analysis agent specialized in processing data.

      Your responsibilities:
      - Analyze patterns
      - Draw insights
      - Make recommendations

  critic:
    system_prompt: |
      You are a critic agent specialized in validation.

      Your responsibilities:
      - Identify flaws
      - Verify accuracy
      - Suggest improvements

# Workflow prompts
workflows:
  synthesis:
    prompt_template: |
      Synthesize the following agent outputs into a coherent response.

      Agent outputs:
      {outputs}

      Task: {task}

      Provide a comprehensive synthesis that:
      1. Integrates all key points
      2. Resolves contradictions
      3. Answers the original task

# Knowledge brain prompts
knowledge:
  comprehension:
    prompt_template: |
      Extract key concepts from the following document chunk.

      Document: {document_name}
      Chunk: {chunk_text}

      Extract:
      - Main concepts (3-5)
      - Definitions
      - Relationships
      - Examples
```

### Using Custom Prompts

```python
from src.prompts.prompt_manager import PromptManager

# Load custom prompts
manager = PromptManager("config/prompts.yaml")

# Get prompt
research_prompt = manager.get_agent_prompt("research")

# Get with template variables
synthesis_prompt = manager.get_workflow_prompt(
    "synthesis",
    outputs=agent_outputs,
    task=task_description
)
```

### Prompt Best Practices

1. **Be Specific**: Clear, detailed instructions
2. **Set Expectations**: Define output format
3. **Provide Examples**: Show desired format
4. **Set Constraints**: Specify length, style
5. **Test Iteratively**: Refine based on results

**Example Good Prompt:**

```yaml
agents:
  research:
    system_prompt: |
      You are a research agent for the Felix AI framework.

      RESPONSIBILITIES:
      1. Gather relevant information on the given topic
      2. Verify facts from multiple sources
      3. Identify knowledge gaps that need further research

      OUTPUT FORMAT:
      - Start with a brief summary (2-3 sentences)
      - List key findings as bullet points
      - Note any uncertainties or conflicts
      - Suggest follow-up research questions

      EXAMPLE:
      Summary: Python is a high-level programming language known for readability.

      Key findings:
      - Created by Guido van Rossum in 1991
      - Popular for data science, web development, automation
      - Dynamic typing, interpreted execution

      Uncertainties:
      - Performance vs compiled languages needs benchmarking

      Follow-up:
      - What are the main performance bottlenecks?
      - How does GIL affect multi-threading?

      CONSTRAINTS:
      - Maximum 500 words
      - Use clear, concise language
      - Cite sources when available
```

---

## Trust Rules Configuration

### File Location

`config/trust_rules.yaml`

### Structure

```yaml
# Trust classification rules for system commands
trust_rules:
  # SAFE: Auto-execute without approval
  safe_commands:
    - "ls"
    - "pwd"
    - "date"
    - "whoami"
    - "echo"
    - "cat"  # Read-only
    - "head"
    - "tail"
    - "grep"
    - "find"
    - "pip list"
    - "pip show"
    - "python --version"
    - "git status"
    - "git log"
    - "git diff"

  safe_patterns:
    - "^ls\\s"
    - "^pwd$"
    - "^echo\\s"

  # REVIEW: Require user approval
  review_commands:
    - "mkdir"
    - "touch"
    - "cp"
    - "mv"
    - "pip install"
    - "git add"
    - "git commit"
    - "npm install"

  review_patterns:
    - "^mkdir\\s"
    - "^pip install\\s"
    - ">.+"  # Redirects to files

  # BLOCKED: Never execute
  blocked_commands:
    - "rm -rf"
    - "sudo"
    - "chmod 777"
    - "wget"
    - "curl"  # Can download/execute
    - "eval"
    - "exec"
    - "dd"  # Dangerous disk operations

  blocked_patterns:
    - "rm.*-rf"
    - "sudo"
    - ".*password.*"
    - ".*credential.*"
    - ".*token.*"
    - ">/dev/"  # Write to devices

# Command-specific settings
command_settings:
  # Maximum command length
  max_length: 500

  # Allowed working directories (optional)
  allowed_directories:
    - "/home/user/project"
    - "/tmp"

  # Blocked directories
  blocked_directories:
    - "/etc"
    - "/root"
    - "/sys"
    - "/.ssh"

  # Timeout for command execution (seconds)
  execution_timeout: 30

  # Environment variables to pass
  allowed_env_vars:
    - "PATH"
    - "HOME"
    - "USER"

  # Environment variables to block
  blocked_env_vars:
    - "AWS_SECRET_ACCESS_KEY"
    - "ANTHROPIC_API_KEY"
```

### Custom Trust Rules

```yaml
# Add organization-specific rules
custom_rules:
  # Allow specific deployment commands
  safe_commands:
    - "kubectl get pods"
    - "kubectl describe"
    - "docker ps"

  # Review deployment changes
  review_commands:
    - "kubectl apply"
    - "docker run"

  # Block production changes
  blocked_patterns:
    - ".*--prod.*"
    - ".*production.*"
```

---

## Environment Variables

### Core Variables

| Variable | Purpose | Example | Required |
|----------|---------|---------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-api03-...` | If using Anthropic |
| `GOOGLE_API_KEY` | Google Gemini key | `AIzaSy...` | If using Gemini |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` | If using OpenAI |
| `FELIX_CONFIG_DIR` | Config directory | `/path/to/config` | No |
| `FELIX_API_KEY` | REST API auth key | `your-secret-key` | For API auth |

### Setting Variables

**Linux/Mac:**

```bash
# Temporary (current session)
export ANTHROPIC_API_KEY="sk-ant-..."

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc

# Using .env file
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...
EOF

# Load .env
source .env
```

**Windows:**

```cmd
# Temporary (current session)
set ANTHROPIC_API_KEY=sk-ant-...

# Permanent (system settings)
setx ANTHROPIC_API_KEY "sk-ant-..."
```

### Security Best Practices

1. **Never Commit API Keys**: Add `.env` to `.gitignore`
2. **Use Environment Variables**: Don't hardcode in config files
3. **Rotate Keys Regularly**: Change keys every 90 days
4. **Limit Key Permissions**: Use least-privilege API keys
5. **Monitor Usage**: Track API key usage for anomalies

---

## Tuning Guidelines

### For Local LLMs (LM Studio)

**Hardware Optimization:**

```yaml
# For 8GB VRAM (RTX 3070 level)
token_budget:
  base_budget: 1024  # Smaller context
  max_budget: 2048

# For 12GB VRAM (RTX 3080 level)
token_budget:
  base_budget: 2048  # Standard
  max_budget: 4096

# For 24GB+ VRAM (RTX 4090 level)
token_budget:
  base_budget: 4096  # Large context
  max_budget: 8192
```

**Model Selection:**

| Model Size | VRAM | Speed | Quality | Use Case |
|------------|------|-------|---------|----------|
| 7B (4-bit) | 4-6 GB | Fast | Good | Development |
| 7B (8-bit) | 7-8 GB | Medium | Better | Production |
| 13B (4-bit) | 8-10 GB | Medium | Better | Production |
| 34B (4-bit) | 20-24 GB | Slow | Best | High quality |

### For Cloud APIs

**Cost Optimization:**

```yaml
# Use cheaper models for simple tasks
primary:
  type: "gemini"
  model: "gemini-1.5-flash"  # $0.35/$1.05 per 1M tokens

# Reserve expensive models for complex tasks
fallbacks:
  - type: "anthropic"
    model: "claude-3-5-sonnet-20241022"  # $3/$15 per 1M tokens
```

**Latency Optimization:**

```yaml
# Shorter timeouts for faster failure
primary:
  timeout: 30  # Fail fast

fallbacks:
  - timeout: 60  # Allow more time for fallback
```

### Agent Team Size

```yaml
spawning:
  # For simple tasks
  max_agents: 3
  confidence_threshold: 0.75

  # For complex tasks
  max_agents: 10
  confidence_threshold: 0.80

  # For very complex tasks
  max_agents: 25
  confidence_threshold: 0.85
```

**Guidelines:**
- **3-5 agents**: Simple questions, fact lookup
- **5-10 agents**: Analysis, multi-step reasoning
- **10-25 agents**: Complex research, comprehensive analysis
- **25+ agents**: Experimental (requires powerful hardware)

### Workflow Steps

```yaml
workflow:
  # Simple tasks
  max_steps_simple: 3
  simple_threshold: 0.8

  # Medium tasks
  max_steps_medium: 5
  medium_threshold: 0.6

  # Complex tasks
  max_steps_complex: 10
```

### Memory Configuration

```yaml
memory:
  # Context compression threshold (characters)
  compression_threshold: 10000

  # Compression ratio (0.0-1.0)
  compression_ratio: 0.3

  # Maximum knowledge entries
  max_knowledge_entries: 10000

  # Knowledge cache size (MB)
  knowledge_cache_mb: 512
```

### Performance vs Quality Trade-offs

| Setting | Fast/Cheap | Balanced | Quality/Expensive |
|---------|------------|----------|-------------------|
| Provider | LM Studio | Gemini Flash | Claude Sonnet |
| Token Budget | 1024 | 2048 | 4096 |
| Max Agents | 5 | 10 | 25 |
| Max Steps | 3 | 5 | 10 |
| Timeout | 30s | 60s | 120s |

---

## Configuration Examples

### Development Configuration

```yaml
# config/llm.yaml (development)
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "mistral-7b-instruct"
  timeout: 30

token_budget:
  base_budget: 1024
  strict_mode: true

spawning:
  max_agents: 5
  confidence_threshold: 0.75

workflow:
  max_steps_simple: 2
  max_steps_medium: 3
  max_steps_complex: 5
```

### Production Configuration

```yaml
# config/llm.yaml (production)
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  timeout: 120

fallbacks:
  - type: "gemini"
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-1.5-pro"
    timeout: 120

  - type: "lm_studio"
    base_url: "http://backup-server:1234/v1"
    timeout: 60

router:
  retry_on_rate_limit: true
  verbose_logging: false
  health_check_interval: 300

token_budget:
  base_budget: 2048
  max_budget: 4096
  strict_mode: false

spawning:
  max_agents: 10
  confidence_threshold: 0.80

workflow:
  max_steps_simple: 3
  max_steps_medium: 5
  max_steps_complex: 10
```

---

## Troubleshooting

### Configuration Not Loading

**Check file exists:**
```bash
ls -la config/llm.yaml
```

**Validate YAML syntax:**
```bash
python -c "import yaml; yaml.safe_load(open('config/llm.yaml'))"
```

**Check permissions:**
```bash
chmod 644 config/llm.yaml
```

### Environment Variables Not Working

**Verify variable is set:**
```bash
echo $ANTHROPIC_API_KEY
```

**Check syntax in YAML:**
```yaml
# Correct
api_key: "${ANTHROPIC_API_KEY}"

# Wrong - missing quotes
api_key: ${ANTHROPIC_API_KEY}
```

### Provider Connection Fails

**Test provider directly:**
```bash
# LM Studio
curl http://localhost:1234/v1/models

# Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

---

## See Also

- [LLM_PROVIDER_GUIDE.md](LLM_PROVIDER_GUIDE.md) - Provider details
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Development guide
- [CLI_GUIDE.md](CLI_GUIDE.md) - CLI configuration
- [QUICKSTART.md](../QUICKSTART.md) - Quick setup
