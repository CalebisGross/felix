# LLM Providers Module

## Purpose
Provider implementations for multi-LLM support, wrapping LM Studio, Anthropic Claude, and Google Gemini APIs in a unified interface with automatic failover and load balancing.

## Key Files

### [lm_studio_provider.py](lm_studio_provider.py)
Local LLM provider via LM Studio server.
- **`LMStudioProvider`**: Wraps `LMStudioClient` with `BaseLLMProvider` interface
- **Default endpoint**: `http://localhost:1234/v1`
- **Model**: Uses whatever model is loaded in LM Studio (model parameter is informational)
- **Timeout**: 120s default (configurable)
- **Streaming**: Full support via `complete_streaming()`
- **Air-gapped support**: Works in isolated/classified environments
- **Best for**: Local inference, privacy-sensitive deployments, offline operation

### [anthropic_provider.py](anthropic_provider.py)
Anthropic Claude API provider.
- **`AnthropicProvider`**: Integration with Claude API (claude-3-opus, claude-3-sonnet, claude-3-haiku)
- **Authentication**: Requires `ANTHROPIC_API_KEY` environment variable
- **Models**: Supports Claude 3 family (Opus, Sonnet, Haiku)
- **Streaming**: Full support with token-by-token delivery
- **Features**: System prompts, JSON mode, function calling
- **Rate limiting**: Respects Anthropic API limits
- **Best for**: High-quality reasoning, long context, production deployments

### [gemini_provider.py](gemini_provider.py)
Google Gemini API provider.
- **`GeminiProvider`**: Integration with Gemini API (gemini-pro, gemini-1.5-pro)
- **Authentication**: Requires `GOOGLE_API_KEY` environment variable
- **Models**: Supports Gemini Pro and Gemini 1.5 Pro
- **Streaming**: Full support via `generate_content_stream()`
- **Features**: Multi-modal (text, vision), long context, code execution
- **Safety settings**: Configurable content filtering
- **Best for**: Multi-modal tasks, cost-effective inference, Google ecosystem

### [\_\_init\_\_.py](__init__.py)
Module initialization and provider exports.
- Imports all provider classes
- Provides clean `__all__` export list
- Enables `from src.llm.providers import LMStudioProvider, ...`

## Key Concepts

### Unified Provider Interface
All providers implement `BaseLLMProvider`:
```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion"""

    @abstractmethod
    def complete_streaming(self, request: LLMRequest, callback: Callable):
        """Generate streaming completion"""

    @abstractmethod
    def check_health(self) -> bool:
        """Check provider availability"""
```

### Provider Types
Enum defining supported providers:
- **LM_STUDIO**: Local inference via LM Studio
- **ANTHROPIC**: Claude API (cloud)
- **GEMINI**: Gemini API (cloud)
- **OPENAI**: OpenAI API (future support)

### Request/Response Structures
Standardized data structures for all providers:
```python
LLMRequest(
    agent_id="research_001",
    system_prompt="You are a research agent...",
    user_prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=1000
)

LLMResponse(
    content="Quantum computing is...",
    tokens_used=847,
    latency_ms=2341,
    provider_type=ProviderType.LM_STUDIO,
    model="llama-3-8b"
)
```

### Routing Strategy
`LLMRouter` distributes requests across providers:
1. **Primary routing**: Send to configured primary provider
2. **Health checking**: Skip unhealthy providers
3. **Automatic failover**: Retry with next available provider on failure
4. **Load balancing**: Distribute across multiple instances (optional)
5. **Statistics tracking**: Monitor success rates and latency per provider

### Provider Selection
Configured via `config/llm.yaml`:
```yaml
providers:
  - type: lm_studio
    base_url: http://localhost:1234/v1
    priority: 1  # Primary provider
    timeout: 120

  - type: anthropic
    model: claude-3-sonnet-20240229
    priority: 2  # Fallback
    api_key_env: ANTHROPIC_API_KEY

  - type: gemini
    model: gemini-1.5-pro
    priority: 3  # Second fallback
    api_key_env: GOOGLE_API_KEY
```

### Streaming Support
All providers support streaming for real-time token delivery:
```python
def token_callback(token: str, done: bool):
    print(token, end='', flush=True)
    if done:
        print()  # Newline at end

provider.complete_streaming(request, token_callback)
```

### Error Handling
Providers raise specific exceptions:
- **`ProviderConnectionError`**: Cannot connect to provider
- **`ProviderAuthError`**: Authentication failure (missing/invalid API key)
- **`ProviderRateLimitError`**: Rate limit exceeded
- **`ProviderError`**: Generic provider error

### Health Checking
Each provider implements health checks:
- **LM Studio**: HTTP GET to `/health` or test completion
- **Anthropic**: Test API call with minimal tokens
- **Gemini**: Test API call with minimal tokens
- Unhealthy providers skipped by router

### Provider Statistics
Router tracks metrics per provider:
- **Total requests**: Count of all requests sent
- **Successful requests**: Count of successful completions
- **Failed requests**: Count of errors
- **Average latency**: Mean response time
- **Last health check**: Timestamp of last health check

### Air-Gapped Operation
LM Studio provider enables **completely offline** operation:
- No external API calls required
- No internet connectivity needed
- Suitable for classified/isolated networks
- Felix is the **only multi-agent framework** that works air-gapped

### Cost Optimization
Multi-provider setup enables cost control:
1. Use local LM Studio for development/testing (free)
2. Use Gemini for production (cost-effective)
3. Use Claude for complex reasoning (premium)
4. Router handles automatic selection

## Configuration Example

### Development (Local Only)
```yaml
providers:
  - type: lm_studio
    base_url: http://localhost:1234/v1
    priority: 1
```

### Production (Cloud with Fallback)
```yaml
providers:
  - type: gemini
    model: gemini-1.5-pro
    priority: 1
    api_key_env: GOOGLE_API_KEY

  - type: anthropic
    model: claude-3-sonnet-20240229
    priority: 2
    api_key_env: ANTHROPIC_API_KEY
```

### Air-Gapped (Local Only)
```yaml
providers:
  - type: lm_studio
    base_url: http://127.0.0.1:1234/v1
    priority: 1
    timeout: 180
```

## Related Modules
- [base_provider.py](../base_provider.py) - `BaseLLMProvider` interface and data structures
- [llm_router.py](../llm_router.py) - Intelligent routing and failover
- [router_adapter.py](../router_adapter.py) - Backward compatibility adapter
- [lm_studio_client.py](../lm_studio_client.py) - Low-level LM Studio client (wrapped by provider)
- [config/llm.yaml](../../../config/llm.yaml) - Provider configuration
