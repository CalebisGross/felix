# LLM Provider Guide

Complete guide to Felix's multi-provider LLM architecture and creating custom providers.

## Table of Contents

1. [Overview](#overview)
2. [Provider Architecture](#provider-architecture)
3. [Using Existing Providers](#using-existing-providers)
4. [Creating Custom Providers](#creating-custom-providers)
5. [Router Configuration](#router-configuration)
6. [Best Practices](#best-practices)

---

## Overview

Felix uses a **provider abstraction layer** that allows seamless integration with multiple LLM backends. This enables:

- **Automatic Fallback**: If primary provider fails, fallbacks are tried
- **Load Balancing**: Distribute requests across providers
- **Unified Interface**: Same code works with any provider
- **Easy Extension**: Add new providers by implementing one interface
- **Configuration-Driven**: Switch providers via YAML config

### Supported Providers

| Provider | Type | Status | Use Case |
|----------|------|--------|----------|
| LM Studio | Local | ✅ Production | Free, private, fast |
| Anthropic Claude | Cloud API | ✅ Production | High quality, reasoning |
| Google Gemini | Cloud API | ✅ Production | Large context, vision |
| OpenAI | Cloud API | 🚧 Planned | GPT models |
| Azure OpenAI | Cloud API | 🚧 Planned | Enterprise |

---

## Provider Architecture

### Layered Design

```
┌─────────────────────────────────────────┐
│           Application Code               │
│    (Agents, Workflows, etc.)            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│          RouterAdapter                   │
│   (Backwards compatibility layer)       │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           LLMRouter                      │
│  - Provider selection                    │
│  - Fallback logic                        │
│  - Statistics tracking                   │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
┌───────▼───┐ ┌───▼────┐ ┌───▼────────┐
│LMStudio   │ │Anthropic│ │  Gemini    │
│Provider   │ │Provider │ │  Provider  │
└───────────┘ └─────────┘ └────────────┘
    │             │            │
┌───▼───┐    ┌───▼────┐   ┌───▼────┐
│LMStudio│   │Anthropic│  │ Google │
│ Server │   │   API   │  │   API  │
└────────┘   └─────────┘  └────────┘
```

### Key Components

**1. BaseLLMProvider (Abstract Base Class)**
- Defines standard interface all providers must implement
- Enforces unified request/response structures
- Located: `src/llm/base_provider.py`

**2. LLMRouter (Request Router)**
- Routes requests to appropriate provider
- Handles fallback on failure
- Tracks success rates and latency
- Located: `src/llm/llm_router.py`

**3. Provider Implementations**
- Concrete implementations of BaseLLMProvider
- Provider-specific API calls and formatting
- Located: `src/llm/providers/*.py`

**4. RouterAdapter (Compatibility Layer)**
- Adapts new router to old LMStudioClient interface
- Allows gradual migration of existing code
- Located: `src/llm/router_adapter.py`

---

## Using Existing Providers

### Quick Start with LM Studio (Local)

**1. Install LM Studio**
```bash
# Download from https://lmstudio.ai/
# Load a model (e.g., Mistral 7B)
# Start server on port 1234
```

**2. Configure Felix**
```yaml
# config/llm.yaml
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"
  timeout: 60
```

**3. Use in Code**
```python
from src.llm.router_adapter import create_router_adapter

# Creates router with config/llm.yaml
adapter = create_router_adapter()

# Make request
response = adapter.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response["content"])
```

### Using Anthropic Claude

**1. Get API Key**
```bash
# Get key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-key-here"
```

**2. Configure Felix**
```yaml
# config/llm.yaml
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"  # Reads from environment
  model: "claude-3-5-sonnet-20241022"
  timeout: 120

fallbacks:
  - type: "lm_studio"  # Fallback to local if API fails
    base_url: "http://localhost:1234/v1"
```

**3. Use in Code**
```python
# Same code works with different provider!
adapter = create_router_adapter()
response = adapter.chat_completion(...)  # Uses Claude
```

### Using Google Gemini

**1. Get API Key**
```bash
# Get key from https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your-key-here"
```

**2. Configure Felix**
```yaml
# config/llm.yaml
primary:
  type: "gemini"
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-1.5-pro"
  timeout: 120
```

### Multi-Provider with Fallback

Configure multiple providers for high availability:

```yaml
# config/llm.yaml
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
    base_url: "http://localhost:1234/v1"
    model: "local-model"
    timeout: 60
```

**Behavior:**
1. Try Anthropic Claude first
2. If Claude fails: try Gemini
3. If Gemini fails: try LM Studio (local)
4. If all fail: raise ProviderError

---

## Creating Custom Providers

### Step 1: Understand the Interface

All providers must implement `BaseLLMProvider`:

```python
from abc import ABC, abstractmethod
from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError
)

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Generate completion for the given request.

        Args:
            request: LLMRequest with prompt, temperature, max_tokens, etc.

        Returns:
            LLMResponse with content, tokens used, response time

        Raises:
            ProviderConnectionError: Cannot reach provider
            ProviderAuthenticationError: Invalid credentials
            ProviderRateLimitError: Rate limit exceeded
            ProviderError: Other provider errors
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if provider is reachable and authenticated."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return human-readable provider name."""
        pass

    def supports_streaming(self) -> bool:
        """Whether provider supports token streaming."""
        return False

    def get_model_name(self) -> str:
        """Return the model name being used."""
        return "unknown"
```

### Step 2: Create Provider Class

Example: Adding OpenAI provider

```python
# src/llm/providers/openai_provider.py

import time
import logging
import requests
from typing import Optional

from src.llm.base_provider import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderConnectionError,
    ProviderAuthenticationError,
    ProviderRateLimitError
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider implementation.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 120):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url: API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        start_time = time.time()

        # Build request payload
        messages = []
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        messages.append({
            "role": "user",
            "content": request.user_prompt
        })

        payload = {
            "model": request.model or self.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Make API request
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )

            # Handle errors
            if response.status_code == 401:
                raise ProviderAuthenticationError(
                    "Invalid OpenAI API key"
                )
            elif response.status_code == 429:
                raise ProviderRateLimitError(
                    "OpenAI rate limit exceeded"
                )
            elif response.status_code != 200:
                raise ProviderError(
                    f"OpenAI API error: {response.status_code} - {response.text}"
                )

            # Parse response
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            prompt_tokens = data["usage"]["prompt_tokens"]
            completion_tokens = data["usage"]["completion_tokens"]

            response_time = time.time() - start_time

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                response_time=response_time,
                provider_name=self.get_provider_name(),
                model_name=self.get_model_name()
            )

        except requests.exceptions.ConnectionError as e:
            raise ProviderConnectionError(
                f"Cannot connect to OpenAI API: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise ProviderError(f"OpenAI request timeout: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI error: {e}")

    def test_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            # Try a minimal request
            request = LLMRequest(
                system_prompt="",
                user_prompt="Hi",
                temperature=0.7,
                max_tokens=5
            )
            self.complete(request)
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "openai"

    def get_model_name(self) -> str:
        """Return model name."""
        return self.model

    def supports_streaming(self) -> bool:
        """OpenAI supports streaming."""
        return True
```

### Step 3: Register Provider Type

Add to `ProviderType` enum:

```python
# src/llm/base_provider.py

class ProviderType(Enum):
    """Supported LLM provider types."""
    LM_STUDIO = "lm_studio"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"  # Add this
```

### Step 4: Add to Provider Config Loader

Update config loader to handle new provider:

```python
# src/llm/provider_config.py

def create_provider(provider_config: Dict[str, Any]) -> BaseLLMProvider:
    """Create provider instance from config."""
    provider_type = provider_config.get("type")

    if provider_type == "lm_studio":
        return LMStudioProvider(...)
    elif provider_type == "anthropic":
        return AnthropicProvider(...)
    elif provider_type == "gemini":
        return GeminiProvider(...)
    elif provider_type == "openai":  # Add this
        return OpenAIProvider(
            api_key=provider_config["api_key"],
            model=provider_config.get("model", "gpt-4"),
            base_url=provider_config.get("base_url", "https://api.openai.com/v1"),
            timeout=provider_config.get("timeout", 120)
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
```

### Step 5: Test Provider

Create comprehensive tests:

```python
# tests/unit/llm/providers/test_openai_provider.py

import pytest
from unittest.mock import Mock, patch
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.base_provider import (
    LLMRequest,
    ProviderAuthenticationError,
    ProviderRateLimitError
)


def test_successful_completion():
    """Test successful completion request."""
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4"
    )

    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "total_tokens": 50,
                "prompt_tokens": 10,
                "completion_tokens": 40
            }
        }
        mock_post.return_value = mock_response

        # Make request
        request = LLMRequest(
            system_prompt="You are helpful",
            user_prompt="Hello",
            temperature=0.7
        )
        response = provider.complete(request)

        # Verify
        assert response.content == "Test response"
        assert response.tokens_used == 50
        assert response.provider_name == "openai"


def test_authentication_error():
    """Test handling of auth errors."""
    provider = OpenAIProvider(api_key="invalid-key")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        request = LLMRequest(
            system_prompt="",
            user_prompt="Test",
            temperature=0.7
        )

        with pytest.raises(ProviderAuthenticationError):
            provider.complete(request)


def test_rate_limit_error():
    """Test handling of rate limits."""
    provider = OpenAIProvider(api_key="test-key")

    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response

        request = LLMRequest(
            system_prompt="",
            user_prompt="Test",
            temperature=0.7
        )

        with pytest.raises(ProviderRateLimitError):
            provider.complete(request)
```

### Step 6: Document Provider

Add to [src/llm/_index.md](../src/llm/_index.md):

```markdown
#### [openai_provider.py](providers/openai_provider.py)
OpenAI API provider.
- **`OpenAIProvider`**: BaseLLMProvider implementation for OpenAI API
- **Features**: GPT-4/3.5-turbo support, streaming, function calling
```

### Step 7: Configure and Use

Add to `config/llm.yaml`:

```yaml
primary:
  type: "openai"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4"
  timeout: 120
```

Use like any other provider:

```python
adapter = create_router_adapter()
response = adapter.chat_completion(...)  # Uses OpenAI
```

---

## Router Configuration

### Basic Configuration

```yaml
# config/llm.yaml

# Required: Primary provider
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"
  timeout: 60

# Optional: Fallback providers (tried in order)
fallbacks:
  - type: "anthropic"
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"

# Optional: Router settings
router:
  retry_on_rate_limit: false  # Don't retry fallbacks on rate limit
  verbose_logging: true       # Enable detailed logs
```

### Advanced Configuration

```yaml
# Multiple providers with priorities
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
  timeout: 120
  # Provider-specific settings
  max_retries: 3
  backoff_factor: 2

fallbacks:
  # High-quality fallback
  - type: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    timeout: 120

  # Cost-effective fallback
  - type: "gemini"
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-1.5-flash"  # Faster, cheaper
    timeout: 60

  # Local fallback (always available)
  - type: "lm_studio"
    base_url: "http://localhost:1234/v1"
    model: "mistral-7b"
    timeout: 30

router:
  retry_on_rate_limit: true
  verbose_logging: false
  health_check_interval: 300  # Check provider health every 5 min
```

### Environment Variables

Use environment variables for secrets:

```bash
# .env
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

Config references them:

```yaml
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"  # Reads from environment
```

---

## Best Practices

### 1. Always Provide Fallbacks

**Why**: Cloud APIs can be unreliable (rate limits, outages, network issues)

```yaml
# ❌ Bad: No fallback
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"

# ✅ Good: Local fallback always available
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"

fallbacks:
  - type: "lm_studio"
    base_url: "http://localhost:1234/v1"
```

### 2. Test Connections at Startup

```python
from src.llm.router_adapter import create_router_adapter

# Create router
adapter = create_router_adapter()

# Test all providers
if not adapter.test_connection():
    print("⚠️  Warning: Primary provider unavailable")
    print("Fallbacks will be used")
```

### 3. Monitor Provider Statistics

```python
router = adapter.router
stats = router.get_statistics()

print(f"Total requests: {stats['total_requests']}")
print(f"Primary success rate: {stats['primary_success_rate']:.1%}")
print(f"Fallback usage: {stats['fallback_success_count']}")
```

### 4. Handle Provider Errors Gracefully

```python
from src.llm.base_provider import ProviderError

try:
    response = adapter.chat_completion(...)
except ProviderError as e:
    logger.error(f"All providers failed: {e}")
    # Fallback to cached response or default
    response = get_cached_response()
```

### 5. Use Appropriate Models

Match model to task:

| Task | Recommended Model | Why |
|------|------------------|-----|
| Simple classification | gemini-1.5-flash | Fast, cheap |
| Code generation | claude-3-5-sonnet | Good at code |
| Long documents | gemini-1.5-pro | 1M token context |
| Local/private | mistral-7b (LM Studio) | No data leaves machine |

### 6. Implement Timeouts

Always set reasonable timeouts:

```yaml
primary:
  type: "anthropic"
  timeout: 120  # 2 minutes max

fallbacks:
  - type: "lm_studio"
    timeout: 30  # Local should be faster
```

### 7. Use Streaming for Long Responses

Enable streaming for better UX:

```python
response = adapter.chat_completion(
    messages=[...],
    stream=True,  # Enable streaming
    stream_callback=lambda chunk: print(chunk, end='', flush=True)
)
```

### 8. Cache Responses

Cache similar requests to reduce costs:

```python
import hashlib
import json

def get_cache_key(messages, temperature):
    key_data = json.dumps({
        "messages": messages,
        "temperature": temperature
    }, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()

# Check cache before request
cache_key = get_cache_key(messages, 0.7)
if cache_key in response_cache:
    return response_cache[cache_key]

# Make request and cache
response = adapter.chat_completion(...)
response_cache[cache_key] = response
```

---

## Troubleshooting

### Provider Connection Fails

**Symptom**: `ProviderConnectionError`

**Solutions**:
1. Check provider is running (for LM Studio)
2. Verify API key is set (for cloud providers)
3. Test network connectivity
4. Check firewall/proxy settings

```bash
# Test LM Studio
curl http://localhost:1234/v1/models

# Test Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

### Rate Limit Errors

**Symptom**: `ProviderRateLimitError`

**Solutions**:
1. Enable fallback providers
2. Implement request rate limiting
3. Upgrade API tier
4. Use local provider for development

### Slow Responses

**Symptom**: Requests timeout or take very long

**Solutions**:
1. Reduce `max_tokens`
2. Use faster model (e.g., gemini-flash vs gemini-pro)
3. Check network latency
4. Use local provider for development

### High Costs

**Symptom**: API bills are high

**Solutions**:
1. Use cheaper models (gemini-flash, gpt-3.5-turbo)
2. Reduce max_tokens
3. Implement response caching
4. Use local provider for development/testing

---

## See Also

- [src/llm/_index.md](../src/llm/_index.md) - LLM module overview
- [CONFIGURATION.md](CONFIGURATION.md) - Complete configuration reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Extension guide
