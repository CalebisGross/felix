# Basic Examples

## Purpose
Simple, minimal examples for learning Felix fundamentals: basic workflows, multi-provider configuration, and core concepts.

## Key Files

### [hello_felix.py](hello_felix.py)
**Simplest possible Felix workflow** (< 50 lines)
- **Purpose**: Minimal working example showing Felix architecture
- **What it demonstrates**:
  - Felix system initialization
  - Single workflow execution
  - Agent spawning and synthesis
  - Result retrieval
- **Prerequisites**: LM Studio running OR cloud provider API key
- **Run**: `python examples/basic/hello_felix.py`
- **Expected output**: Multi-agent analysis of a simple task with synthesis result

### [multi_provider.py](multi_provider.py)
**Multi-LLM provider configuration with automatic fallback**
- **Purpose**: Shows how to configure multiple LLM providers with failover
- **What it demonstrates**:
  - Provider configuration (Anthropic, Gemini, LM Studio)
  - Automatic failover on provider failure
  - Cost comparison between providers
  - Health checking
  - Load balancing
- **Prerequisites**: At least one provider configured (ideally multiple for demonstration)
- **Environment variables**:
  ```bash
  export ANTHROPIC_API_KEY="your-key"
  export GOOGLE_API_KEY="your-key"
  ```
- **Run**: `python examples/basic/multi_provider.py`
- **Expected output**: Same task executed with different providers, showing latency and cost differences

### [knowledge_brain_demo.py](knowledge_brain_demo.py)
**Basic Knowledge Brain usage**
- **Purpose**: Introduction to document ingestion and semantic retrieval
- **What it demonstrates**:
  - Document ingestion from files
  - Agentic comprehension
  - Knowledge graph construction
  - Semantic search
- **Prerequisites**: Knowledge Brain enabled in config
- **Run**: `python examples/basic/knowledge_brain_demo.py`
- **Expected output**: Documents processed, concepts extracted, semantic queries answered

## Learning Goals

### Example 1: hello_felix.py
**Learn the basics:**
- How to initialize Felix system
- How workflows execute
- How agents spawn dynamically
- How synthesis combines agent outputs
- How to retrieve results

**Key concepts covered:**
- Helix geometry positioning
- Agent lifecycle (spawn → work → synthesis)
- Hub-spoke communication via CentralPost
- Confidence-based dynamic spawning

### Example 2: multi_provider.py
**Learn provider management:**
- Configure multiple LLM providers
- Set up automatic failover
- Compare costs across providers
- Monitor provider health
- Balance load across instances

**Key concepts covered:**
- Provider abstraction via `BaseLLMProvider`
- Routing strategy with `LLMRouter`
- Health checking and failover
- Cost optimization strategies

### Example 3: knowledge_brain_demo.py
**Learn knowledge management:**
- Ingest documents into knowledge base
- Extract concepts via agentic comprehension
- Build knowledge graphs automatically
- Perform semantic search
- Retrieve relevant context for workflows

**Key concepts covered:**
- Document reading and chunking
- Agentic comprehension (Research → Analysis → Critic)
- Knowledge graph construction
- Semantic retrieval with embeddings
- Meta-learning boost

## Quick Start

### Option 1: Local LLM (LM Studio)
```bash
# 1. Start LM Studio with a model loaded on port 1234
# 2. Run example
python examples/basic/hello_felix.py
```

### Option 2: Cloud Provider
```bash
# 1. Set API key
export ANTHROPIC_API_KEY="your-anthropic-key"
# OR
export GOOGLE_API_KEY="your-google-key"

# 2. Configure provider in config/llm.yaml
# 3. Run example
python examples/basic/hello_felix.py
```

### Option 3: Multiple Providers
```bash
# 1. Set API keys
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# 2. Configure all providers in config/llm.yaml
# 3. Run multi-provider example
python examples/basic/multi_provider.py
```

## Code Walkthrough

### hello_felix.py Structure
```python
# 1. Initialize Felix system
felix = FelixSystem(config)

# 2. Create task
task = "Explain quantum computing"

# 3. Run workflow
result = run_felix_workflow(felix, task)

# 4. Display results
print(result.synthesis.content)
```

### multi_provider.py Structure
```python
# 1. Configure multiple providers
config = {
    'providers': [
        {'type': 'anthropic', 'priority': 1},
        {'type': 'gemini', 'priority': 2},
        {'type': 'lm_studio', 'priority': 3}
    ]
}

# 2. Initialize router
router = LLMRouter(config)

# 3. Execute same task with each provider
for provider in providers:
    result = execute_with_provider(task, provider)
    # Compare latency, cost, quality

# 4. Show cost comparison
display_cost_analysis(results)
```

## Expected Output

### hello_felix.py
```
=== Felix Workflow Execution ===
Task: Explain quantum computing

Spawning agents...
- Research agent spawned (confidence: 0.65)
- Analysis agent spawned (confidence: 0.72)
- Critic agent validated (confidence: 0.84)

Synthesis complete (confidence: 0.87)

=== Result ===
Quantum computing is a revolutionary computing paradigm that...
[Full synthesis output]

Agents: 3 | Tokens: 2,847 | Time: 12.3s
```

### multi_provider.py
```
=== Multi-Provider Comparison ===

Provider: Anthropic (Claude 3 Sonnet)
- Latency: 2.1s
- Tokens: 847
- Cost: $0.004
- Quality: High

Provider: Gemini (Gemini 1.5 Pro)
- Latency: 1.8s
- Tokens: 823
- Cost: $0.001
- Quality: High

Provider: LM Studio (Local)
- Latency: 3.2s
- Tokens: 891
- Cost: $0.000
- Quality: Medium-High

Recommendation: Use Gemini for production (best cost/performance)
```

## Troubleshooting

**"ConnectionRefusedError"**
- LM Studio not running or wrong port
- Start LM Studio and load a model
- Check `config/llm.yaml` has correct base_url

**"AuthenticationError"**
- Missing or invalid API key
- Set environment variable: `export ANTHROPIC_API_KEY="your-key"`
- Check API key is valid on provider dashboard

**"No agents spawned"**
- Confidence threshold too high
- Task too simple (use more complex task)
- LLM not generating proper confidence scores

## Next Steps

After completing basic examples:
1. Try [advanced/knowledge_brain_demo.py](../advanced/) for full knowledge system
2. Explore [api_examples/](../api_examples/) for REST API usage
3. Study [custom_agents/](../custom_agents/) to build your own agents

## Related Documentation
- [Felix Workflow Architecture](../../docs/WORKFLOW_ARCHITECTURE.md)
- [Multi-Provider Setup](../../docs/MULTI_PROVIDER_GUIDE.md)
- [Configuration Reference](../../docs/CONFIGURATION.md)
