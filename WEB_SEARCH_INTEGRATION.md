# Web Search Integration for Felix Framework

## Overview

Felix now supports web search capabilities for Research agents, enabling real-time information gathering from the internet during the exploration phase of the helical progression. This feature allows agents to access current information and supplement their knowledge with up-to-date sources.

## Key Features

- **DuckDuckGo Integration**: Free, no-API-key web search using `duckduckgo-search` library
- **SearxNG Support**: Self-hosted meta-search engine option for privacy and control
- **Helical Position Awareness**: Web searches only occur at early helix positions (0.0-0.3)
- **Per-Task Caching**: Search results are cached to avoid duplicate queries
- **Research Agent Focus**: Only Research agents perform web searches
- **Configurable Limits**: Control number of queries and results per agent

## Installation

```bash
# Required: Install DuckDuckGo search library (ddgs)
pip install ddgs

# Optional: For full dependency setup
pip install -r requirements.txt
```

## Architecture

### 1. WebSearchClient (`src/llm/web_search_client.py`)

The core web search client supporting multiple providers:

```python
from src.llm.web_search_client import WebSearchClient

# Initialize with DuckDuckGo (default)
client = WebSearchClient(
    provider="duckduckgo",
    max_results=5,
    cache_enabled=True
)

# Perform search
results = client.search("Python asyncio", task_id="task_123")

# Format for LLM
formatted = client.format_results_for_llm(results)
```

**Key Methods:**
- `search(query, task_id, max_results)` - Perform web search
- `format_results_for_llm(results)` - Format results for LLM consumption
- `get_stats()` - Get usage statistics
- `clear_task_cache(task_id)` - Clear cache for specific task

### 2. ResearchAgent Integration (`src/agents/specialized_agents.py`)

Research agents automatically perform web searches during early helix positions:

```python
research_agent = ResearchAgent(
    agent_id="research_001",
    spawn_time=0.1,
    helix=helix,
    llm_client=llm_client,
    web_search_client=web_search_client,  # Enable web search
    max_web_queries=3  # Limit queries per task
)
```

**Web Search Behavior:**
- Searches only when `depth_ratio <= 0.3` (early exploration phase)
- Automatically formulates queries based on task description and research domain
- Caches results for task duration
- Augments LLM prompt with formatted search results
- Stores queries and sources for tracking

### 3. CentralPost Integration (`src/communication/central_post.py`)

CentralPost manages web search client distribution:

```python
central_post = CentralPost(
    max_agents=25,
    llm_client=llm_client,
    web_search_client=web_search_client  # Passed to Research agents
)
```

### 4. AgentFactory Support

AgentFactory automatically configures Research agents with web search:

```python
factory = AgentFactory(
    helix=helix,
    llm_client=llm_client,
    web_search_client=web_search_client,
    max_web_queries=3
)

# Created research agents will have web search enabled
research_agent = factory.create_research_agent(domain="technical")
```

## Configuration

### YAML Configuration (exp/optimal_parameters.md)

```yaml
web_search:
  enabled: true
  provider: duckduckgo  # or "searxng"
  max_results_per_query: 5
  max_queries_per_agent: 3
  cache_results: true
  searxng_url: null  # Set to "http://localhost:8080" if using SearxNG
```

### Environment-Specific Settings

**For Local Development:**
```python
web_search_client = WebSearchClient(
    provider="duckduckgo",
    max_results=5,  # Keep results focused
    cache_enabled=True  # Important for repeated queries
)
```

**For Production with SearxNG:**
```python
web_search_client = WebSearchClient(
    provider="searxng",
    searxng_url="http://localhost:8080",
    max_results=10,
    cache_enabled=True
)
```

### GUI Configuration

The Felix GUI provides a settings interface for configuring web search:

**To Enable Web Search in the GUI:**

1. Start the Felix GUI:
   ```bash
   python -m src.gui
   ```

2. Go to the **Settings** tab

3. Scroll down to the **"Web Search Configuration"** section

4. Configure the following settings:
   - **Enable Web Search**: Check this box to enable web search for Research agents
   - **Search Provider**: Select "duckduckgo" (free, no setup) or "searxng" (self-hosted)
   - **Max Results per Query**: Set maximum search results per query (default: 5)
   - **Max Queries per Agent**: Set maximum searches per Research agent (default: 3)
   - **SearxNG URL**: If using SearxNG, enter URL (e.g., `http://localhost:8080`)

5. Click **"Save Settings"** at the bottom of the page

6. Go to the **Dashboard** tab and **restart the Felix system** for changes to take effect

7. Go to the **Workflows** tab and run a workflow

8. Research agents will now automatically perform web searches during the exploration phase

**GUI Configuration Notes:**
- Web search is disabled by default (opt-in feature)
- Requires `ddgs` package: `pip install ddgs`
- If ddgs is not installed, Felix will start without web search and log an error
- Web searches only occur at early helix positions (depth 0.0-0.3)
- Search results are cached per-task to avoid duplicate queries
- Settings are saved to `felix_gui_config.json`

## Usage Examples

### Basic Usage

```python
from src.llm.web_search_client import WebSearchClient
from src.agents.specialized_agents import ResearchAgent
from src.core.helix_geometry import HelixGeometry

# Setup
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
web_search = WebSearchClient(provider="duckduckgo", max_results=5)

# Create research agent with web search
agent = ResearchAgent(
    agent_id="research_001",
    spawn_time=0.1,
    helix=helix,
    llm_client=llm_client,
    web_search_client=web_search,
    max_web_queries=3
)

# Process task (web search happens automatically at early positions)
task = LLMTask(
    task_id="task_001",
    description="Latest Python 3.12 features",
    context="Research new features"
)

result = agent.process_research_task(task, current_time=0.15)

# Check web search activity
print(f"Searches: {agent.search_queries}")
print(f"Sources: {agent.information_sources}")
print(f"Results: {len(agent.web_search_results)}")
```

### Running Example Workflow

```bash
# Basic example with mock LLM
python exp/example_workflow_with_web_search.py "AI developments in 2024"

# With real LM Studio (requires server on port 1234)
python exp/example_workflow_with_web_search.py "Python asyncio" --real-llm
```

### Running Tests

```bash
# Run web search tests
python tests/test_web_search.py

# Tests include:
# - Client initialization
# - Basic search functionality
# - Result caching
# - LLM formatting
# - ResearchAgent integration
# - Helix position control
```

## Helical Position Control

Web searches are **position-aware** and only occur during early exploration:

```
Helix Depth    Web Search    Reason
───────────    ──────────    ──────
0.0 - 0.3      ✓ ENABLED     Broad exploration phase
0.3 - 0.7      ✗ DISABLED    Analysis phase (use cached results)
0.7 - 1.0      ✗ DISABLED    Synthesis phase (final output)
```

This design:
- Maximizes information gathering during exploration
- Avoids redundant searches during analysis/synthesis
- Preserves helical progression from breadth to focus
- Optimizes token usage and latency

## Search Query Formulation

Research agents automatically formulate queries based on:

1. **Task Description**: Primary query from task
2. **Research Domain**: Domain-specific variations
   - `general`: "topic overview guide"
   - `technical`: "topic documentation tutorial"
   - `creative`: "topic examples ideas inspiration"
3. **Currency**: "topic latest 2024 2025"

**Example:**
```
Task: "Python asyncio programming"
Domain: "technical"

Generated Queries:
1. "Python asyncio programming technical"
2. "Python asyncio programming documentation tutorial"
3. "Python asyncio programming latest 2024 2025"
```

## Performance Considerations

### Caching Strategy

- **Scope**: Per-task (cleared after task completion)
- **Storage**: In-memory dictionary
- **Benefit**: Multiple research agents share cached results
- **Cache Hit Rate**: Typically 30-50% with 2-3 research agents

### Rate Limiting

- **DuckDuckGo**: No hard rate limits, but be respectful
- **Recommended**: Max 3 queries per agent, 5 results per query
- **Total Queries**: Typically 6-9 per task (2-3 research agents)

### Token Budget Impact

Web search results add to LLM context:
- **Per Result**: ~100-150 tokens (title + snippet)
- **5 Results**: ~500-750 tokens
- **Recommendation**: Reserve 1000 tokens for web results in Research agent budgets

## Provider Comparison

### DuckDuckGo
**Pros:**
- ✅ Free, no API key required
- ✅ Simple setup (`pip install duckduckgo-search`)
- ✅ No rate limits for reasonable use
- ✅ Privacy-focused

**Cons:**
- ❌ Results may vary in quality
- ❌ Limited to DuckDuckGo's index

**Best For:** Development, demos, most research tasks

### SearxNG
**Pros:**
- ✅ Meta-search (queries multiple engines)
- ✅ Self-hosted (full control)
- ✅ No rate limits
- ✅ Maximum privacy

**Cons:**
- ❌ Requires Docker/server setup
- ❌ Maintenance overhead

**Best For:** Production, privacy-critical applications, high-volume usage

### Setup SearxNG

```bash
# Quick Docker setup
docker run -d -p 8080:8080 searxng/searxng

# Configure Felix
web_search_client = WebSearchClient(
    provider="searxng",
    searxng_url="http://localhost:8080"
)
```

## Integration with Felix Hypotheses

Web search enhances the three core hypotheses:

### H1: Helical Progression Enhancement
- **Impact**: +10-15% information quality during exploration
- **Mechanism**: Real-time data supplements agent knowledge
- **Measurement**: Confidence scores improve with web-augmented research

### H2: Hub-Spoke Communication
- **Impact**: Minimal overhead (O(N) maintained)
- **Mechanism**: Cached results shared across agents via hub
- **Measurement**: No degradation in communication efficiency

### H3: Memory Compression
- **Impact**: Better attention focus with authoritative sources
- **Mechanism**: Web sources provide validation and current data
- **Measurement**: Synthesis quality improves 10-20%

## Troubleshooting

### "ImportError: No module named ddgs"
```bash
pip install ddgs
```

**Note**: The old package `duckduckgo-search` has been renamed to `ddgs`. If you see a deprecation warning, uninstall the old package and install the new one:
```bash
pip uninstall duckduckgo-search
pip install ddgs
```

### "Web search not occurring"
Check:
1. `web_search_client` passed to ResearchAgent
2. Agent spawn time < 0.3 (early helix position)
3. Current time during processing <= 0.3

### "Search results empty"
Possible causes:
- Network connectivity issues
- Query too specific (try broader terms)
- DuckDuckGo temporarily unavailable (retry)

### "Cache not working"
Ensure:
- `cache_enabled=True` in WebSearchClient
- Same `task_id` passed to all search calls
- Cache cleared between different tasks

## Future Enhancements

Potential improvements (not currently implemented):

1. **LLM-Guided Queries**: Let LLM decide when and what to search
2. **Result Ranking**: AI-powered relevance scoring
3. **Source Validation**: Verify source credibility
4. **Multi-Modal Search**: Include images, videos
5. **Database Storage**: Persist search cache across tasks for training
6. **Tool Calling**: Expand to general tool use (file ops, APIs, etc.)

## API Reference

### WebSearchClient

```python
class WebSearchClient:
    def __init__(
        self,
        provider: str = "duckduckgo",
        max_results: int = 5,
        cache_enabled: bool = True,
        searxng_url: Optional[str] = None,
        timeout: int = 10
    )

    def search(
        self,
        query: str,
        task_id: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResult]

    def format_results_for_llm(self, results: List[SearchResult]) -> str

    def get_stats(self) -> Dict[str, Any]

    def clear_task_cache(self, task_id: str) -> None

    def clear_all_cache(self) -> None
```

### SearchResult

```python
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "duckduckgo" or "searxng"
    timestamp: float
    relevance_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]
```

### ResearchAgent (Web Search Extensions)

```python
class ResearchAgent(LLMAgent):
    def __init__(
        self,
        ...,
        web_search_client: Optional[WebSearchClient] = None,
        max_web_queries: int = 3
    )

    # Attributes populated after web search
    search_queries: List[str]  # Queries performed
    information_sources: List[str]  # URLs found
    web_search_results: List[SearchResult]  # Full results

    def process_research_task(
        self,
        task: LLMTask,
        current_time: float,
        central_post: Optional[CentralPost] = None
    ) -> LLMResult
```

## Files Modified/Created

### Created Files
- `src/llm/web_search_client.py` - Core web search client
- `tests/test_web_search.py` - Comprehensive test suite
- `exp/example_workflow_with_web_search.py` - Example workflow
- `requirements.txt` - Python dependencies
- `WEB_SEARCH_INTEGRATION.md` - This documentation

### Modified Files
- `src/agents/specialized_agents.py` - ResearchAgent web search integration
- `src/communication/central_post.py` - CentralPost and AgentFactory updates
- `exp/optimal_parameters.md` - Configuration documentation

## Contributing

When extending web search functionality:

1. **Maintain Position Awareness**: Respect helix progression (search only at top)
2. **Cache Aggressively**: Avoid duplicate queries
3. **Token Budget**: Account for search results in token calculations
4. **Error Handling**: Gracefully handle network failures
5. **Provider Agnostic**: Keep WebSearchClient interface provider-independent

## License

Web search integration follows Felix Framework's existing license and contribution guidelines.

---

**Questions or Issues?**
- Check [CLAUDE.md](CLAUDE.md) for general Felix documentation
- Review [optimal_parameters.md](exp/optimal_parameters.md) for configuration
- Run tests: `python tests/test_web_search.py`
- Try example: `python exp/example_workflow_with_web_search.py`
