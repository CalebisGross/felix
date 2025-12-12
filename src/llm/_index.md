# LLM Module

## Purpose
Language model integration providing multi-provider routing, local LLM support via LM Studio, adaptive token management, web search capabilities, and load balancing across multiple servers.

## Key Files

### [lm_studio_client.py](lm_studio_client.py)
Local LLM integration via LM Studio with streaming support.
- **`LMStudioClient`**: Main client for LM Studio API (default port 1234) with incremental token streaming
- **`RequestPriority`**: Enum for request prioritization (LOW, NORMAL, HIGH, CRITICAL)
- **`AsyncRequest`**: Async request wrapper with priority and timeout
- **`LLMResponse`**: Response structure with content, tokens, latency
- **`StreamingChunk`**: Individual token chunks for streaming
- **`TokenAwareStreamController`**: Manages time-batched token delivery with callbacks

### [base_provider.py](base_provider.py)
Abstract provider interface for multi-LLM support.
- **`BaseLLMProvider`**: Abstract base class for all LLM providers
- **`ProviderType`**: Enum for provider types (LM_STUDIO, ANTHROPIC, GEMINI, OPENAI)
- **`LLMRequest`**: Unified request structure across providers
- **`LLMResponse`**: Unified response structure
- **Error Types**: `ProviderError`, `RateLimitError`, `TokenLimitError`, `AuthenticationError`

### [llm_router.py](llm_router.py)
Multi-provider routing and load balancing.
- **`LLMRouter`**: Routes requests to appropriate providers with fallback and load balancing
- **Features**: Provider health checking, automatic failover, request distribution

### [provider_config.py](provider_config.py)
Configuration loading and router factory.
- **`ProviderConfigLoader`**: Loads provider configs from YAML
- **`get_llm_router()`**: Factory function returning configured LLMRouter

### [token_budget.py](token_budget.py)
Adaptive token allocation for agents.
- **`TokenBudgetManager`**: Manages per-agent token budgets with adaptive allocation
- **`TokenAllocation`**: Token budget structure with tracking
- **Base Budget**: 2048 tokens (configurable for model capacity)

### [web_search_client.py](web_search_client.py)
Internet search integration for agent research.
- **`WebSearchClient`**: Unified interface for DuckDuckGo and SearxNG
- **`SearchProvider`**: Enum for search engines (DUCKDUCKGO, SEARXNG)
- **`SearchResult`**: Search result structure with title, URL, snippet, domain
- **Features**: Result caching, domain filtering, relevance scoring

### [router_adapter.py](router_adapter.py)
Adapter for provider abstraction.
- **`RouterAdapter`**: Adapts LLMRouter to provider interface for seamless integration

### [multi_server_client.py](multi_server_client.py)
Load balancing across multiple LM Studio instances.
- **`LMStudioClientPool`**: Manages multiple LM Studio servers with round-robin or load-based distribution
- **Features**: Health monitoring, automatic failover, connection pooling

### [providers/](providers/) Subdirectory
Provider-specific implementations.

#### [lm_studio_provider.py](providers/lm_studio_provider.py)
LM Studio provider implementation.
- **`LMStudioProvider`**: BaseLLMProvider implementation for LM Studio

#### [anthropic_provider.py](providers/anthropic_provider.py)
Anthropic Claude API provider.
- **`AnthropicProvider`**: BaseLLMProvider implementation for Anthropic API
- **Features**: Claude model support, streaming, message formatting

#### [gemini_provider.py](providers/gemini_provider.py)
Google Gemini API provider.
- **`GeminiProvider`**: BaseLLMProvider implementation for Gemini API
- **Features**: Gemini Pro/Ultra support, safety settings, token counting

## Key Concepts

### Provider Abstraction Architecture
Felix uses a layered architecture for LLM integration:
1. **Router Layer** (`LLMRouter`): Handles provider selection, load balancing, and failover
2. **Provider Layer** (`BaseLLMProvider`): Abstract interface defining standard operations
3. **Implementation Layer** (providers/*.py): Concrete provider implementations (LM Studio, Anthropic, Gemini)
4. **Adapter Layer** (`RouterAdapter`): Backwards compatibility bridge for existing code
5. **Client Layer** (`LMStudioClient`): Low-level networking and protocol handling

**Benefits:**
- Add new providers by implementing `BaseLLMProvider`
- Automatic fallback ensures high availability
- Unified `LLMRequest`/`LLMResponse` structures simplify code
- Router tracks statistics (success rates, latency, token usage)
- Configuration-driven provider selection (no code changes needed)

### Temperature Gradient
Agents adapt temperature based on helix position:
- **Top (exploration)**: 1.0 - Wide, creative exploration
- **Bottom (synthesis)**: 0.2 - Focused, deterministic output
- **Gradient**: Linear interpolation between positions

### Streaming Architecture
- **Time-batched delivery**: Tokens delivered in 0.1s batches
- **Callback support**: Custom handlers for real-time updates
- **Incremental processing**: Agents can start processing before completion

### Token Budget Management
- **Base budget**: 2048 tokens per agent
- **Strict mode**: Enforces limits for local LLMs
- **Adaptive allocation**: Adjusts based on task complexity and position

### Multi-Provider Support
- **Unified interface**: Same code works across providers
- **Automatic fallback**: Fails over to backup providers
- **Load balancing**: Distributes requests across instances
- **Health monitoring**: Tracks provider availability

### Web Search Integration
- **Multiple engines**: DuckDuckGo (primary), SearxNG (fallback)
- **Result caching**: Reduces redundant searches
- **Domain filtering**: Include/exclude specific domains
- **Relevance scoring**: Ranks results by query match

### Multi-Server Pool
- **Round-robin**: Distributes load evenly
- **Health-based**: Routes to healthiest servers
- **Connection pooling**: Reuses connections for efficiency
- **Automatic failover**: Removes unhealthy servers from rotation

## Configuration

```yaml
llm:
  token_budget:
    base_budget: 2048
    strict_mode: true

  providers:
    - type: lm_studio
      endpoint: http://localhost:1234
      priority: 1
    - type: anthropic
      api_key: ${ANTHROPIC_API_KEY}
      priority: 2
```

## Related Modules
- [agents/](../agents/) - LLMAgent uses LLM clients for reasoning
- [communication/](../communication/) - WebSearchCoordinator uses WebSearchClient
- [knowledge/](../knowledge/) - LMStudioEmbedder uses embedding endpoints
- [core/](../core/) - HelixGeometry determines temperature gradient
- [workflows/](../workflows/) - Workflows configure and route LLM requests
