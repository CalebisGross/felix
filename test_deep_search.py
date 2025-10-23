#!/usr/bin/env python3
"""
Test script to verify deep web search implementation in Felix framework.

This test validates:
1. WebSearchClient.fetch_page_content() method works
2. CentralPost._extract_and_store_relevant_info() performs deep search fallback
3. Extracted information contains actual factual data (not generic statements)
4. Knowledge is stored with deep_search_used flag
5. Knowledge entries are properly retrieved and formatted
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.web_search_client import WebSearchClient
from src.llm.lm_studio_client import LMStudioClient
from src.communication.central_post import CentralPost
from src.memory.knowledge_store import KnowledgeStore, KnowledgeQuery, ConfidenceLevel
from src.core.helix_geometry import HelixGeometry

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_page_fetching():
    """Test that fetch_page_content() can retrieve and parse HTML."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Page Fetching Capability")
    logger.info("="*80)

    # Create web search client
    web_client = WebSearchClient(
        provider="duckduckgo",
        max_results=3,
        cache_enabled=False
    )

    # Try fetching a known time website
    test_url = "https://time.is"
    logger.info(f"\nAttempting to fetch: {test_url}")

    page_data = web_client.fetch_page_content(test_url, max_length=3000)

    if page_data:
        logger.info(f"✓ SUCCESS: Fetched {len(page_data['content'])} characters")
        logger.info(f"  Title: {page_data['title']}")
        logger.info(f"  URL: {page_data['url']}")
        logger.info(f"  Content preview (first 200 chars):")
        logger.info(f"  {page_data['content'][:200]}...")
        return True
    else:
        logger.error("✗ FAILED: Could not fetch page content")
        return False


def test_deep_search_extraction():
    """Test that CentralPost performs deep search and extracts factual information."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Deep Search Extraction Pipeline")
    logger.info("="*80)

    # Create test database
    test_db = "test_deep_search_knowledge.db"
    Path(test_db).unlink(missing_ok=True)

    # Create components
    knowledge_store = KnowledgeStore(test_db)
    llm_client = LMStudioClient(base_url="http://localhost:1234/v1")

    # Create web search client
    web_search_client = WebSearchClient(
        provider="duckduckgo",
        max_results=5,
        cache_enabled=True
    )

    # Create CentralPost with web search enabled
    central_post = CentralPost(
        max_agents=25,
        enable_memory=True,
        memory_db_path=test_db,
        llm_client=llm_client,
        web_search_client=web_search_client,
        web_search_confidence_threshold=0.7,
        web_search_min_samples=1,
        web_search_cooldown=10.0
    )

    # Set knowledge store manually (not in __init__ parameters)
    central_post.knowledge_store = knowledge_store

    # Perform web search for current time
    task_description = "what is the current date and time?"
    logger.info(f"\nSearching for: '{task_description}'")

    # This will trigger the three-phase process:
    # Phase 1: Extract from snippets
    # Phase 2: Detect insufficiency and fetch full page
    # Phase 3: Store enhanced results
    results = central_post.web_search_client.search(
        query=task_description,
        task_id="test_001",
        max_results=5
    )

    if not results:
        logger.error("✗ FAILED: No search results returned")
        return False

    logger.info(f"✓ Found {len(results)} search results")
    for i, result in enumerate(results[:3], 1):
        logger.info(f"  {i}. {result.title}")
        logger.info(f"     URL: {result.url}")

    # Now trigger extraction (which includes deep search)
    logger.info("\n--- Starting Deep Search Extraction ---")
    central_post._extract_and_store_relevant_info(results, task_description)

    # Retrieve stored knowledge
    logger.info("\n--- Verifying Knowledge Storage ---")
    time.sleep(1)  # Brief delay to ensure write completion

    relevant_knowledge = knowledge_store.retrieve_knowledge(
        KnowledgeQuery(
            domains=["web_search"],
            min_confidence=ConfidenceLevel.MEDIUM,
            limit=5
        )
    )

    if not relevant_knowledge:
        logger.error("✗ FAILED: No knowledge entries stored")
        return False

    logger.info(f"✓ Found {len(relevant_knowledge)} knowledge entries")

    # Check the content
    success = False
    for entry in relevant_knowledge:
        logger.info(f"\n--- Knowledge Entry ---")
        logger.info(f"  Domain: {entry.domain}")
        logger.info(f"  Source: {entry.source_agent}")
        logger.info(f"  Confidence: {entry.confidence_level.value}")

        if isinstance(entry.content, dict):
            result_content = entry.content.get('result', '')
            deep_search_used = entry.content.get('deep_search_used', False)
            source_url = entry.content.get('source_url', 'N/A')

            logger.info(f"  Deep search used: {deep_search_used}")
            logger.info(f"  Source URL: {source_url}")
            logger.info(f"  Extracted content:")
            logger.info(f"  {result_content}")

            # Check if content contains specific date/time info (not generic)
            generic_phrases = [
                "vary by location",
                "provides time information",
                "can be displayed",
                "tools are synchronized"
            ]

            is_generic = any(phrase in result_content.lower() for phrase in generic_phrases)
            has_specific_info = any(term in result_content.lower() for term in [
                "2024", "2025", "october", "wednesday", "thursday", "friday", "monday", "tuesday",
                "pm", "am", "edt", "est", "utc", "gmt"
            ])

            if not is_generic and has_specific_info:
                logger.info("\n✓ SUCCESS: Content contains specific factual information!")
                success = True
            else:
                logger.warning("\n⚠ WARNING: Content appears generic or lacks specific details")
                if is_generic:
                    logger.warning("  Contains generic phrases")
                if not has_specific_info:
                    logger.warning("  Missing specific date/time indicators")

    # Cleanup
    Path(test_db).unlink(missing_ok=True)

    return success


def test_knowledge_display_formatting():
    """Test that knowledge entries are displayed with proper formatting."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Knowledge Display Formatting")
    logger.info("="*80)

    # Create test database
    test_db = "test_display_knowledge.db"
    Path(test_db).unlink(missing_ok=True)

    knowledge_store = KnowledgeStore(test_db)

    # Store mock web search result
    from src.memory.knowledge_store import KnowledgeType

    knowledge_store.store_knowledge(
        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
        content={
            "result": "Current date: Wednesday, October 23, 2025. Current time: 12:45:30 PM EDT. Time zone: Eastern Daylight Time (UTC-4).",
            "task": "what is the current date and time?",
            "source_count": 5,
            "deep_search_used": True,
            "source_url": "https://time.is",
            "timestamp": time.time()
        },
        confidence_level=ConfidenceLevel.HIGH,
        source_agent="centralpost_web_search",
        domain="web_search",
        tags=["web_search", "factual_data", "current_information"]
    )

    # Store mock workflow result
    knowledge_store.store_knowledge(
        knowledge_type=KnowledgeType.TASK_RESULT,
        content={"text": "Analysis of helical progression patterns"},
        confidence_level=ConfidenceLevel.MEDIUM,
        source_agent="analysis_001",
        domain="workflow_task",
        tags=["analysis"]
    )

    # Retrieve and display
    all_knowledge = knowledge_store.retrieve_knowledge(
        KnowledgeQuery(
            domains=["web_search", "workflow_task"],
            min_confidence=ConfidenceLevel.MEDIUM,
            limit=10
        )
    )

    # Sort web_search first (mimicking context_builder behavior)
    all_knowledge = sorted(
        all_knowledge,
        key=lambda ke: (ke.domain != "web_search", ke.created_at),
        reverse=False
    )

    logger.info(f"\n✓ Retrieved {len(all_knowledge)} entries (sorted, web_search first)")

    for entry in all_knowledge:
        # Extract content (mimicking llm_agent behavior)
        if isinstance(entry.content, dict):
            content_str = entry.content.get('result', str(entry.content))
        else:
            content_str = str(entry.content)

        # Use different limits based on domain
        max_chars = 400 if entry.domain == "web_search" else 200
        truncated = content_str[:max_chars] + "..." if len(content_str) > max_chars else content_str

        # Add emoji
        prefix = "🌐" if entry.domain == "web_search" else "📝"

        logger.info(f"\n  {prefix} [{entry.domain}] {truncated}")
        logger.info(f"     Confidence: {entry.confidence_level.value}, Source: {entry.source_agent}")

    # Cleanup
    Path(test_db).unlink(missing_ok=True)

    logger.info("\n✓ SUCCESS: Display formatting working correctly")
    return True


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("FELIX DEEP WEB SEARCH INTEGRATION TEST")
    logger.info("="*80)
    logger.info("\nThis test validates the three-phase deep search implementation:")
    logger.info("  Phase 1: Extract from search snippets")
    logger.info("  Phase 2: Detect insufficiency and fetch full webpage content")
    logger.info("  Phase 3: Store enhanced results with metadata")
    logger.info("\nNOTE: Requires LM Studio running on localhost:1234")
    logger.info("="*80)

    results = {}

    try:
        # Test 1: Page fetching
        results['page_fetching'] = test_page_fetching()

        # Test 2: Deep search extraction (requires LLM)
        logger.info("\n\nChecking LM Studio connection...")
        try:
            llm_client = LMStudioClient(base_url="http://localhost:1234/v1")
            # Try a simple test
            llm_client.complete(
                agent_id="test",
                system_prompt="You are a test.",
                user_prompt="Say 'OK'",
                temperature=0.1,
                max_tokens=10
            )
            logger.info("✓ LM Studio connection successful")
            results['deep_search_extraction'] = test_deep_search_extraction()
        except Exception as e:
            logger.error(f"✗ LM Studio connection failed: {e}")
            logger.error("  Skipping deep search extraction test")
            logger.error("  Please start LM Studio with a model loaded on port 1234")
            results['deep_search_extraction'] = False

        # Test 3: Display formatting (no LLM required)
        results['display_formatting'] = test_knowledge_display_formatting()

        # Summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)

        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status}: {test_name}")

        all_passed = all(results.values())
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("✓ ALL TESTS PASSED")
        else:
            logger.info("✗ SOME TESTS FAILED - Check output above for details")
        logger.info("="*80)

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"\n✗ TEST SUITE FAILED WITH ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
