"""
Tests for Web Search Integration in Felix Framework

This module tests the WebSearchClient and its integration with ResearchAgent.
"""

import sys
import os
import time
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.web_search_client import WebSearchClient, SearchResult, SearchProvider
from src.agents.specialized_agents import ResearchAgent
from src.core.helix_geometry import HelixGeometry
from src.agents.llm_agent import LLMTask, LLMResult


# Mock LLM Client for testing without actual LLM
class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0

    def complete(self, agent_id, system_prompt, user_prompt, temperature=0.7, max_tokens=500, model="mock"):
        """Mock completion that returns a simple response."""
        self.call_count += 1

        # Check if web search results are in the prompt
        has_web_results = "Web Search Results:" in system_prompt or "Web Search Results:" in user_prompt

        content = f"Mock response #{self.call_count} from {agent_id}."
        if has_web_results:
            content += " I found relevant web search results to support my research."

        return type('LLMResponse', (), {
            'content': content,
            'tokens_used': len(content.split()),
            'response_time': 0.1,
            'finish_reason': 'stop'
        })()

    def complete_streaming(self, agent_id, system_prompt, user_prompt, temperature=0.7,
                          max_tokens=None, model="mock", batch_interval=0.1, callback=None):
        """Mock streaming completion."""
        return self.complete(agent_id, system_prompt, user_prompt, temperature, max_tokens, model)


def test_web_search_client_initialization():
    """Test WebSearchClient initialization with different providers."""
    print("\n=== Test 1: WebSearchClient Initialization ===")

    # Test DuckDuckGo initialization
    try:
        client = WebSearchClient(provider="duckduckgo", max_results=5)
        print(f"✓ DuckDuckGo client initialized: {client.provider.value}")
        assert client.provider == SearchProvider.DUCKDUCKGO
        assert client.max_results == 5
        assert client.cache_enabled == True
    except ImportError as e:
        print(f"✗ DuckDuckGo initialization failed (missing library): {e}")
        print("  Run: pip install ddgs")
        return False

    # Test SearxNG initialization (will fail without URL, expected)
    try:
        client = WebSearchClient(provider="searxng", searxng_url="http://localhost:8080")
        print(f"✓ SearxNG client initialized: {client.provider.value}")
    except Exception as e:
        print(f"✓ SearxNG requires URL (expected): {type(e).__name__}")

    print("✓ Test 1 passed: WebSearchClient initialization works")
    return True


def test_web_search_basic():
    """Test basic web search functionality."""
    print("\n=== Test 2: Basic Web Search ===")

    try:
        client = WebSearchClient(provider="duckduckgo", max_results=3, cache_enabled=False)

        # Perform a simple search
        query = "Python programming language"
        print(f"Searching for: '{query}'")

        results = client.search(query)

        print(f"✓ Search returned {len(results)} results")

        if results:
            result = results[0]
            print(f"  First result:")
            print(f"    Title: {result.title[:60]}...")
            print(f"    URL: {result.url[:60]}...")
            print(f"    Snippet: {result.snippet[:80]}...")
            assert hasattr(result, 'title')
            assert hasattr(result, 'url')
            assert hasattr(result, 'snippet')
            assert result.source == 'duckduckgo'

        stats = client.get_stats()
        print(f"  Stats: {stats['total_queries']} queries, {stats['errors']} errors")

        print("✓ Test 2 passed: Basic web search works")
        return True

    except ImportError:
        print("✗ Test 2 skipped: ddgs not installed")
        return False
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False


def test_web_search_caching():
    """Test search result caching."""
    print("\n=== Test 3: Search Result Caching ===")

    try:
        client = WebSearchClient(provider="duckduckgo", max_results=3, cache_enabled=True)

        task_id = "test_task_123"
        query = "artificial intelligence"

        # First search (cache miss)
        print(f"First search for: '{query}'")
        start = time.time()
        results1 = client.search(query, task_id=task_id)
        time1 = time.time() - start
        print(f"✓ First search: {len(results1)} results in {time1:.2f}s")

        # Second search (cache hit)
        print(f"Second search for: '{query}' (should be cached)")
        start = time.time()
        results2 = client.search(query, task_id=task_id)
        time2 = time.time() - start
        print(f"✓ Second search: {len(results2)} results in {time2:.2f}s")

        assert len(results1) == len(results2), "Cached results should be identical"
        assert time2 < time1 * 0.5, f"Cached search should be faster ({time2:.2f}s vs {time1:.2f}s)"

        stats = client.get_stats()
        print(f"  Cache stats: {stats['cache_hits']} hits, {stats['cache_misses']} misses")
        assert stats['cache_hits'] >= 1, "Should have at least one cache hit"

        # Clear cache
        client.clear_task_cache(task_id)
        print("✓ Cache cleared")

        print("✓ Test 3 passed: Caching works correctly")
        return True

    except ImportError:
        print("✗ Test 3 skipped: ddgs not installed")
        return False
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False


def test_web_search_formatting():
    """Test formatting search results for LLM consumption."""
    print("\n=== Test 4: Result Formatting ===")

    try:
        client = WebSearchClient(provider="duckduckgo", max_results=2, cache_enabled=False)

        results = client.search("machine learning", task_id="test_format")

        if results:
            formatted = client.format_results_for_llm(results)
            print(f"✓ Formatted {len(results)} results")
            print(f"  Formatted text length: {len(formatted)} chars")
            assert "Web Search Results:" in formatted
            assert "1." in formatted  # Should have numbered results
            print("✓ Test 4 passed: Result formatting works")
            return True
        else:
            print("✗ Test 4 failed: No results to format")
            return False

    except ImportError:
        print("✗ Test 4 skipped: ddgs not installed")
        return False
    except Exception as e:
        print(f"✗ Test 4 failed: {e}")
        return False


def test_research_agent_with_web_search():
    """Test ResearchAgent with web search integration."""
    print("\n=== Test 5: ResearchAgent with Web Search ===")

    try:
        # Create components
        helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        llm_client = MockLLMClient()
        web_search_client = WebSearchClient(provider="duckduckgo", max_results=3, cache_enabled=True)

        # Create research agent with web search
        agent = ResearchAgent(
            agent_id="research_test_001",
            spawn_time=0.1,
            helix=helix,
            llm_client=llm_client,
            research_domain="technical",
            web_search_client=web_search_client,
            max_web_queries=2
        )

        print(f"✓ Created ResearchAgent with web search enabled")
        print(f"  Domain: {agent.research_domain}")
        print(f"  Max queries: {agent.max_web_queries}")

        # Create a task
        task = LLMTask(
            task_id="web_search_test_task",
            description="Research recent developments in Python async programming",
            context="Provide a comprehensive overview with sources"
        )

        # Process task (should trigger web search at early position)
        current_time = 0.1  # Early in helix (within 0.0-0.3 range for web search)
        print(f"✓ Processing task at time {current_time} (early helix position)")

        result = agent.process_research_task(task, current_time)

        print(f"✓ Task processed")
        print(f"  Result content length: {len(result.content)} chars")
        print(f"  Search queries performed: {len(agent.search_queries)}")
        print(f"  Information sources found: {len(agent.information_sources)}")
        print(f"  Web search results: {len(agent.web_search_results)}")

        # Verify web search was used
        assert len(agent.search_queries) > 0, "Should have performed web searches"
        assert len(agent.web_search_results) > 0, "Should have web search results"

        # Check metadata
        assert result.metadata.get('web_search_enabled') == True
        assert result.metadata.get('web_search_results_count') > 0

        print(f"  Sample query: {agent.search_queries[0]}")
        if agent.information_sources:
            print(f"  Sample source: {agent.information_sources[0][:60]}...")

        print("✓ Test 5 passed: ResearchAgent web search integration works")
        return True

    except ImportError:
        print("✗ Test 5 skipped: ddgs not installed")
        return False
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_agent_helix_position_control():
    """Test that web search only happens at early helix positions."""
    print("\n=== Test 6: Helix Position Control ===")

    try:
        helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
        llm_client = MockLLMClient()
        web_search_client = WebSearchClient(provider="duckduckgo", max_results=2)

        agent = ResearchAgent(
            agent_id="research_test_002",
            spawn_time=0.1,
            helix=helix,
            llm_client=llm_client,
            web_search_client=web_search_client,
            max_web_queries=2
        )

        task = LLMTask(
            task_id="position_test",
            description="Test helix position control",
            context="Testing"
        )

        # Spawn the agent first
        agent.spawn(agent.spawn_time, task)

        # Test at early position (should search)
        early_time = 0.15
        agent.search_queries = []  # Reset
        result1 = agent.process_research_task(task, early_time)
        early_searches = len(agent.search_queries)
        early_depth = agent.get_position_info(early_time)['depth_ratio']
        print(f"✓ Early position (t={early_time}, depth={early_depth:.2f}): {early_searches} searches performed")

        # Test at late position (should NOT search)
        late_time = 0.5
        agent.search_queries = []  # Reset
        result2 = agent.process_research_task(task, late_time)
        late_searches = len(agent.search_queries)
        late_depth = agent.get_position_info(late_time)['depth_ratio']
        print(f"✓ Late position (t={late_time}, depth={late_depth:.2f}): {late_searches} searches performed")

        assert early_searches > 0, "Should search at early position"
        assert late_searches == 0, "Should NOT search at late position"

        print("✓ Test 6 passed: Helix position control works correctly")
        return True

    except ImportError:
        print("✗ Test 6 skipped: ddgs not installed")
        return False
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        return False


def run_all_tests():
    """Run all web search tests."""
    print("=" * 70)
    print("Felix Framework - Web Search Integration Tests")
    print("=" * 70)

    tests = [
        test_web_search_client_initialization,
        test_web_search_basic,
        test_web_search_caching,
        test_web_search_formatting,
        test_research_agent_with_web_search,
        test_research_agent_helix_position_control
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL/SKIP"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed or skipped")
        print("\nNote: Install ddgs to run all tests:")
        print("  pip install ddgs")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
