"""
Web Search Client for Felix
Provides web search capabilities for Research agents using DuckDuckGo or SearxNG.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime


logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """Available search providers."""
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"


@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    source: str  # Which search provider
    timestamp: float
    relevance_score: float = 1.0  # For future ranking

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'source': self.source,
            'timestamp': self.timestamp,
            'relevance_score': self.relevance_score
        }


class WebSearchClient:
    """
    Web search client supporting multiple providers.

    Features:
    - DuckDuckGo search (no API key required)
    - SearxNG search (self-hosted)
    - Per-task caching to avoid duplicate queries
    - Rate limiting and error handling
    """

    def __init__(
        self,
        provider: str = "duckduckgo",
        max_results: int = 5,
        cache_enabled: bool = True,
        searxng_url: Optional[str] = None,
        timeout: int = 10,
        blocked_domains: Optional[List[str]] = None
    ):
        """
        Initialize web search client.

        Args:
            provider: Search provider to use ("duckduckgo" or "searxng")
            max_results: Maximum number of results per query
            cache_enabled: Whether to cache results per task
            searxng_url: URL for SearxNG instance (required if provider is "searxng")
            timeout: Request timeout in seconds
            blocked_domains: List of domains to filter from results (e.g., ['wikipedia.org', 'reddit.com'])
        """
        self.provider = SearchProvider(provider)
        self.max_results = max_results
        self.cache_enabled = cache_enabled
        self.searxng_url = searxng_url
        self.timeout = timeout
        self.blocked_domains = blocked_domains or ['wikipedia.org', 'reddit.com']

        # Per-task cache: {task_id: {query: [SearchResult]}}
        self._cache: Dict[str, Dict[str, List[SearchResult]]] = {}

        # Statistics
        self._stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'blocked_results': 0
        }

        # Initialize provider
        self._init_provider()

        logger.info(f"WebSearchClient initialized with provider: {self.provider.value}")

    def _init_provider(self):
        """Initialize the selected search provider."""
        if self.provider == SearchProvider.DUCKDUCKGO:
            try:
                from ddgs import DDGS
                self._ddgs = DDGS()
                logger.info("DuckDuckGo search provider initialized")
            except ImportError:
                logger.error("ddgs library not found. Install with: pip install ddgs")
                raise
        elif self.provider == SearchProvider.SEARXNG:
            if not self.searxng_url:
                raise ValueError("searxng_url must be provided when using SearxNG provider")
            try:
                import httpx
                self._httpx_client = httpx.Client(timeout=self.timeout)
                logger.info(f"SearxNG provider initialized with URL: {self.searxng_url}")
            except ImportError:
                logger.error("httpx library not found. Install with: pip install httpx")
                raise

    def _filter_blocked_domains(
        self,
        results: List[SearchResult]
    ) -> tuple[List[SearchResult], List[SearchResult]]:
        """
        Filter search results to remove blocked domains.

        Args:
            results: List of search results to filter

        Returns:
            Tuple of (kept_results, blocked_results)
        """
        kept, blocked = [], []

        for result in results:
            # Check if any blocked domain appears in the URL
            is_blocked = any(
                blocked_domain in result.url.lower()
                for blocked_domain in self.blocked_domains
            )

            if is_blocked:
                blocked.append(result)
                logger.debug(f"Blocked result from: {result.url}")
            else:
                kept.append(result)

        return kept, blocked

    def search(
        self,
        query: str,
        task_id: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform a web search.

        Args:
            query: Search query string
            task_id: Optional task ID for caching (recommended)
            max_results: Override default max_results for this query

        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to web search")
            return []

        query = query.strip()
        max_results = max_results or self.max_results

        # Check cache if enabled and task_id provided
        if self.cache_enabled and task_id:
            cached_results = self._get_from_cache(task_id, query)
            if cached_results is not None:
                logger.info(f"Cache hit for query: '{query}' (task: {task_id})")
                self._stats['cache_hits'] += 1
                return cached_results[:max_results]
            self._stats['cache_misses'] += 1

        # Perform search
        self._stats['total_queries'] += 1
        start_time = time.time()

        try:
            if self.provider == SearchProvider.DUCKDUCKGO:
                results = self._search_duckduckgo(query, max_results)
            elif self.provider == SearchProvider.SEARXNG:
                results = self._search_searxng(query, max_results)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            elapsed = time.time() - start_time

            # Filter blocked domains
            kept_results, blocked_results = self._filter_blocked_domains(results)

            if blocked_results:
                self._stats['blocked_results'] += len(blocked_results)
                logger.info(f"Search completed in {elapsed:.2f}s: '{query}' ({len(kept_results)} results, {len(blocked_results)} blocked)")
            else:
                logger.info(f"Search completed in {elapsed:.2f}s: '{query}' ({len(kept_results)} results)")

            # Cache kept results if enabled and task_id provided
            if self.cache_enabled and task_id:
                self._add_to_cache(task_id, query, kept_results)

            return kept_results

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Search error for query '{query}': {e}")
            return []

    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        try:
            results = []
            timestamp = time.time()

            # DuckDuckGo text search
            ddg_results = self._ddgs.text(query, max_results=max_results)

            for item in ddg_results:
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('href', ''),
                    snippet=item.get('body', ''),
                    source='duckduckgo',
                    timestamp=timestamp
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    def _search_searxng(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using SearxNG.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        try:
            results = []
            timestamp = time.time()

            # SearxNG API request
            params = {
                'q': query,
                'format': 'json',
                'categories': 'general',
                'language': 'en'
            }

            response = self._httpx_client.get(
                f"{self.searxng_url}/search",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            for item in data.get('results', [])[:max_results]:
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    source='searxng',
                    timestamp=timestamp
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"SearxNG search failed: {e}")
            return []

    def fetch_page_content(self, url: str, max_length: int = 5000) -> Optional[Dict[str, str]]:
        """
        Fetch and parse webpage content for deep information extraction.

        Args:
            url: URL to fetch
            max_length: Maximum content length to extract (characters)

        Returns:
            Dict with 'title', 'content', 'url', 'fetch_time' or None if failed
        """
        try:
            import httpx
            from bs4 import BeautifulSoup

            logger.info(f"🌐 Fetching page content: {url[:60]}...")

            # Fetch with timeout and user agent
            response = httpx.get(
                url,
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; FelixAI/1.0; +https://github.com/felix-ai)'
                }
            )

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None

            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')

            # Remove script, style, nav, footer, ads, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript', 'aside', 'header']):
                element.decompose()

            # Extract text content
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace and empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines)

            # Truncate if too long
            if len(content) > max_length:
                content = content[:max_length] + "..."

            # Get page title
            title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

            # Extract domain for logging
            domain = url.split('/')[2] if '/' in url else url
            logger.info(f"✓ Fetched {len(content)} chars from {domain}")

            return {
                'title': title,
                'content': content,
                'url': url,
                'fetch_time': time.time()
            }

        except ImportError as e:
            logger.error(f"Missing dependency for page fetching: {e}")
            logger.error("Install with: pip install beautifulsoup4 lxml")
            return None
        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch page content from {url}: {e}")
            return None

    def _get_from_cache(self, task_id: str, query: str) -> Optional[List[SearchResult]]:
        """Get cached results for a task and query."""
        return self._cache.get(task_id, {}).get(query)

    def _add_to_cache(self, task_id: str, query: str, results: List[SearchResult]):
        """Add results to cache for a task."""
        if task_id not in self._cache:
            self._cache[task_id] = {}
        self._cache[task_id][query] = results

    def clear_task_cache(self, task_id: str):
        """
        Clear cache for a specific task.

        Args:
            task_id: Task ID to clear cache for
        """
        if task_id in self._cache:
            del self._cache[task_id]
            logger.debug(f"Cleared cache for task: {task_id}")

    def clear_all_cache(self):
        """Clear all cached results."""
        self._cache.clear()
        logger.debug("Cleared all search cache")

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """
        Format search results for inclusion in LLM prompt.

        Args:
            results: List of SearchResult objects

        Returns:
            Formatted string for LLM consumption
        """
        if not results:
            return "No web search results found."

        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            formatted += f"   {result.snippet}\n\n"

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get search statistics.

        Returns:
            Dictionary with usage statistics
        """
        cache_hit_rate = 0.0
        total_cache_attempts = self._stats['cache_hits'] + self._stats['cache_misses']
        if total_cache_attempts > 0:
            cache_hit_rate = self._stats['cache_hits'] / total_cache_attempts

        return {
            'provider': self.provider.value,
            'total_queries': self._stats['total_queries'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_rate': round(cache_hit_rate, 2),
            'errors': self._stats['errors'],
            'blocked_results': self._stats['blocked_results'],
            'cached_tasks': len(self._cache)
        }

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_httpx_client'):
            self._httpx_client.close()


# Convenience function for quick searches
def quick_search(query: str, provider: str = "duckduckgo", max_results: int = 5) -> List[SearchResult]:
    """
    Quick search without creating a persistent client.

    Args:
        query: Search query
        provider: Search provider ("duckduckgo" or "searxng")
        max_results: Maximum results

    Returns:
        List of SearchResult objects
    """
    client = WebSearchClient(provider=provider, max_results=max_results, cache_enabled=False)
    return client.search(query)
