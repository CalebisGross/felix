"""
Web Search Coordinator for the Felix Framework.

Handles intelligent web search triggering, query formulation, content extraction,
and knowledge storage integration for the Felix multi-agent system.

Key Features:
- Confidence-based automatic search triggering
- Hybrid query formulation (task-based + agent analysis)
- Deep search with full page content extraction
- LLM-based information extraction from search results
- Knowledge base integration for agent access
- Domain filtering and result quality control
- Search cooldown and deduplication

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import logging
import re
from typing import List, Optional, Any, Dict
from collections import deque

# Import search and knowledge components
from src.memory.knowledge_store import KnowledgeStore, KnowledgeEntry, KnowledgeType, ConfidenceLevel

# Import message types
from src.communication.message_types import Message, MessageType

# Set up logging
logger = logging.getLogger(__name__)


class WebSearchCoordinator:
    """
    Coordinates web search operations for Felix agent workflows.

    Responsibilities:
    - Monitor team confidence and trigger searches when needed
    - Formulate effective search queries from task and agent context
    - Extract and parse search results
    - Perform deep content extraction from web pages
    - Store extracted knowledge for agent retrieval
    - Manage search cooldowns and prevent redundant searches
    """

    def __init__(self,
                 web_search_client: Optional[Any],
                 knowledge_store: Optional[KnowledgeStore],
                 llm_client: Optional[Any],
                 agent_registry: Any,
                 message_queue_callback: Any,
                 confidence_threshold: float = 0.7,
                 search_cooldown: float = 10.0,
                 min_samples: int = 3):
        """
        Initialize Web Search Coordinator.

        Args:
            web_search_client: Web search backend (DuckDuckGo/SearxNG)
            knowledge_store: Knowledge base for storing extracted info
            llm_client: LLM for extracting information from search results
            agent_registry: AgentRegistry for accessing agent messages
            message_queue_callback: Callback to queue messages
            confidence_threshold: Trigger search below this confidence (default 0.7)
            search_cooldown: Minimum seconds between searches (default 10.0)
            min_samples: Minimum confidence samples before triggering (default 3)
        """
        self.web_search_client = web_search_client
        self.knowledge_store = knowledge_store
        self.llm_client = llm_client
        self.agent_registry = agent_registry
        self._queue_message = message_queue_callback

        # Confidence monitoring
        self._web_search_trigger_threshold = confidence_threshold
        self._web_search_min_samples = min_samples
        self._recent_confidences: deque = deque(maxlen=10)

        # Search rate limiting
        self._last_search_time: float = 0.0
        self._search_cooldown = search_cooldown
        self._search_count: int = 0

        # Task context
        self._current_task_description: Optional[str] = None
        self._current_workflow_id: Optional[str] = None
        self._processed_messages: List[Message] = []  # For query formulation

        logger.info("✓ WebSearchCoordinator initialized")

    def set_task_context(self, task_description: str, workflow_id: Optional[str] = None):
        """Set current task context for search query formulation."""
        self._current_task_description = task_description
        self._current_workflow_id = workflow_id

    def update_confidence(self, confidence: float):
        """Add confidence value to rolling window."""
        self._recent_confidences.append(confidence)

    def add_processed_message(self, message: Message):
        """Add message to history for query formulation context."""
        self._processed_messages.append(message)
        # Keep only recent messages to avoid memory growth
        if len(self._processed_messages) > 50:
            self._processed_messages = self._processed_messages[-50:]

    def check_confidence_and_search(self) -> None:
        """
        Check rolling average confidence and trigger web search if low.

        Called after each message is processed to monitor team consensus.
        If confidence drops below threshold and cooldown expired, performs web search.
        """
        # Need enough data points and web search client
        if not self.web_search_client or len(self._recent_confidences) < self._web_search_min_samples:
            return

        # Check if cooldown period has passed
        time_since_last_search = time.time() - self._last_search_time
        if time_since_last_search < self._search_cooldown:
            return

        # Calculate rolling average confidence
        avg_confidence = sum(self._recent_confidences) / len(self._recent_confidences)

        # Trigger search if confidence is low
        if avg_confidence < self._web_search_trigger_threshold:
            logger.info(f"Low confidence detected (avg: {avg_confidence:.2f} < {self._web_search_trigger_threshold})")
            self.perform_web_search(self._current_task_description or "information gathering")
            self._last_search_time = time.time()

    def update_confidence_threshold(self, new_threshold: float, reason: str = "") -> None:
        """
        Dynamically update the confidence threshold for synthesis/web search triggering.

        Args:
            new_threshold: New confidence threshold value (0.0-1.0)
            reason: Explanation for threshold change (for logging)
        """
        old_threshold = self._web_search_trigger_threshold
        self._web_search_trigger_threshold = max(0.0, min(1.0, new_threshold))

        if old_threshold != self._web_search_trigger_threshold:
            logger.info(f"🎯 Adaptive threshold: {old_threshold:.2f} → {self._web_search_trigger_threshold:.2f}")
            if reason:
                logger.info(f"   Reason: {reason}")

    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self._web_search_trigger_threshold

    def get_current_task_description(self) -> Optional[str]:
        """Get the current task description for context."""
        return self._current_task_description

    def check_proactive_search_needed(self, task_description: str, task_complexity: str = "COMPLEX") -> bool:
        """
        Check if task requires immediate proactive web search.

        Analyzes task patterns to detect time-sensitive queries or factual lookups
        that would benefit from immediate web search before agent processing.

        Args:
            task_description: The task description from user
            task_complexity: Task complexity classification (default "COMPLEX")

        Returns:
            True if proactive search is recommended, False otherwise

        Examples:
            >>> coordinator.check_proactive_search_needed("What time is it?", "SIMPLE_FACTUAL")
            True
            >>> coordinator.check_proactive_search_needed("Explain quantum physics", "MEDIUM")
            False
        """
        import re

        task_lower = task_description.lower()

        # Only trigger for SIMPLE_FACTUAL tasks
        if task_complexity != "SIMPLE_FACTUAL":
            return False

        # Check for web search client availability
        if not self.web_search_client:
            return False

        # Check cooldown period
        time_since_last_search = time.time() - self._last_search_time
        if time_since_last_search < self._search_cooldown:
            return False

        # Time/date query patterns (proactive search highly recommended)
        time_patterns = [
            r'\bwhat\s+time\b',
            r'\bwhat\s+date\b',
            r'\btoday\'?s?\s+date\b',
            r'\bcurrent\s+(time|date|datetime)\b',
            r'\bwhat\s+(is|are)\s+(the\s+)?current\s+(time|date)',
        ]

        for pattern in time_patterns:
            if re.search(pattern, task_lower):
                logger.info(f"🔍 Proactive search triggered for time/date query: {task_description[:50]}...")
                return True

        # Latest news/events (benefit from recent data)
        news_patterns = [
            r'\blatest\s+(news|update|information)\b',
            r'\brecent\s+(events|news|developments)\b',
            r'\bwhat\s+happened\s+(today|recently|lately)\b',
        ]

        for pattern in news_patterns:
            if re.search(pattern, task_lower):
                logger.info(f"🔍 Proactive search triggered for news query: {task_description[:50]}...")
                return True

        return False

    def perform_web_search(self, task_description: str) -> None:
        """
        Perform web search when consensus is low and store relevant info in knowledge base.

        Args:
            task_description: The current workflow task to guide search queries
        """
        # FAILSAFE: Use print() in case logger is broken
        print(f"[WEB_SEARCH_COORDINATOR] perform_web_search ENTRY - task: {task_description[:50]}...")
        print(f"[WEB_SEARCH_COORDINATOR] web_search_client: {'AVAILABLE' if self.web_search_client else 'NONE'}")

        logger.info(f"🔍 perform_web_search ENTRY - task: {task_description[:50]}...")
        logger.info(f"🔍 web_search_client status: {'AVAILABLE' if self.web_search_client else 'NONE'}")

        # Check if we already have trustable knowledge (prevent redundant searches)
        if self.knowledge_store and self._search_count >= 2:
            if self._check_existing_trustable_knowledge(task_description):
                return

        if not self.web_search_client:
            print("[WEB_SEARCH_COORDINATOR] ERROR: web_search_client is None!")
            logger.warning("⚠ perform_web_search called but web_search_client is None!")
            return

        try:
            start_time = time.time()

            # Increment search counter
            self._search_count += 1

            # Log search initiation with human-readable format
            print("[WEB_SEARCH_COORDINATOR] === WEB SEARCH TRIGGERED ===")
            logger.info("=" * 60)
            logger.info("WEB SEARCH TRIGGERED")
            logger.info("=" * 60)

            # Calculate stats
            recent_confs = list(self._recent_confidences)
            avg_conf = sum(recent_confs) / len(recent_confs) if recent_confs else 0.0
            logger.info(f"Reason: Low confidence (avg: {avg_conf:.2f}, threshold: {self._web_search_trigger_threshold:.2f})")
            logger.info(f"Task: {task_description}")
            logger.info(f"Agents analyzed: {len(recent_confs)} outputs")
            logger.info(f"Search attempt: #{self._search_count}")
            logger.info("")

            # Generate search queries (hybrid approach)
            logger.info("📝 Formulating search queries...")
            queries = self._formulate_search_queries(task_description)
            logger.info(f"📝 Generated {len(queries)} search queries")
            for idx, q in enumerate(queries, 1):
                logger.info(f"   {idx}. \"{q}\"")
            logger.info("")

            # Track all results and blocked domains
            all_results = []
            blocked_count = 0

            # Perform searches
            for i, query in enumerate(queries, 1):
                logger.info(f"🔍 Executing Query {i}/{len(queries)}: \"{query}\"")

                try:
                    results = self.web_search_client.search(
                        query=query,
                        task_id=f"websearch_{int(time.time())}"
                    )

                    logger.info(f"  📄 Received {len(results)} results from search provider")

                    # Log each result
                    for j, result in enumerate(results, 1):
                        domain = result.url.split('/')[2] if '/' in result.url else result.url
                        logger.info(f"  {j}. {domain} - {result.title[:60]}...")

                    all_results.extend(results)

                except Exception as e:
                    logger.error(f"  ✗ Search query failed with exception: {e}", exc_info=True)

                logger.info("")

            # Get blocked stats from web_search_client
            stats = self.web_search_client.get_stats()
            blocked_count = stats.get('blocked_results', 0)

            # Log statistics
            elapsed = time.time() - start_time
            logger.info("📊 Search Statistics:")
            logger.info(f"  • Total sources found: {len(all_results) + blocked_count}")
            logger.info(f"  • Blocked by filter: {blocked_count} ({', '.join(self.web_search_client.blocked_domains) if blocked_count > 0 else 'none'})")
            logger.info(f"  • Relevant sources: {len(all_results)}")
            logger.info(f"  • Search time: {elapsed:.2f}s")
            logger.info("")

            # Extract and store relevant information
            if all_results:
                logger.info(f"🔬 Calling extract_and_store_relevant_info with {len(all_results)} results...")
                self._extract_and_store_relevant_info(all_results, task_description)
                logger.info(f"✓ extract_and_store_relevant_info completed")
            else:
                logger.warning("⚠ No search results available after filtering - CANNOT EXTRACT KNOWLEDGE")
                logger.warning(f"⚠ This means agents will not have web search data!")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Web search failed with EXCEPTION: {e}", exc_info=True)
            logger.error(f"❌ Search will NOT be available to agents due to this failure")

    def _check_existing_trustable_knowledge(self, task_description: str) -> bool:
        """Check if trustable knowledge already exists for this task."""
        # Skip if knowledge store is disabled
        if not self.knowledge_store:
            return False

        try:
            from src.memory.knowledge_store import KnowledgeQuery
            from src.workflows.truth_assessment import assess_answer_confidence

            # Retrieve recent knowledge
            current_time = time.time()
            one_hour_ago = current_time - 3600

            knowledge_entries = self.knowledge_store.retrieve_knowledge(
                KnowledgeQuery(
                    domains=["web_search"],
                    min_confidence=ConfidenceLevel.HIGH,
                    time_range=(one_hour_ago, current_time),
                    limit=5
                )
            )

            if knowledge_entries:
                trustable, score, reason = assess_answer_confidence(knowledge_entries, task_description)
                if trustable:
                    logger.info(f"⏭️  Skipping web search: Trustable knowledge already exists ({reason})")
                    logger.info(f"   {len(knowledge_entries)} HIGH confidence entries available")
                    return True
        except Exception as e:
            logger.warning(f"Could not assess existing knowledge: {e}")

        return False

    def handle_web_search_request(self, message: Message) -> None:
        """
        Handle explicit web search request from an agent.

        Args:
            message: Message containing WEB_SEARCH_NEEDED request
        """
        if not self.web_search_client:
            logger.warning("Web search requested but no client available")
            return

        try:
            content = message.content.get('content', '')
            agent_id = message.sender_id

            # Extract search query from WEB_SEARCH_NEEDED: pattern
            pattern = r'WEB_SEARCH_NEEDED:\s*(.+?)(?:\n|$)'
            matches = re.findall(pattern, content, re.IGNORECASE)

            if not matches:
                logger.warning(f"Agent {agent_id} used WEB_SEARCH_NEEDED but no query found")
                return

            # Use the first query found
            query = matches[0].strip()

            logger.info("=" * 60)
            logger.info(f"AGENT-REQUESTED WEB SEARCH")
            logger.info("=" * 60)
            logger.info(f"Requesting Agent: {agent_id}")
            logger.info(f"Query: \"{query}\"")
            logger.info("")

            # Perform search
            results = self.web_search_client.search(
                query=query,
                task_id=f"agent_request_{int(time.time())}"
            )

            # Log results
            for result in results:
                logger.info(f"  ✓ {result.url.split('/')[2] if '/' in result.url else result.url} - {result.title[:60]}...")

            logger.info(f"\n📊 Found {len(results)} results for agent {agent_id}")

            # Extract and store relevant information
            if results:
                self._extract_and_store_relevant_info(results, query)
            else:
                logger.warning("⚠ No search results found")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Agent web search request failed: {e}", exc_info=True)

    def _formulate_search_queries(self, task_description: str) -> List[str]:
        """
        Formulate search queries using hybrid approach: task + agent analysis.

        Args:
            task_description: Base task description

        Returns:
            List of search query strings (2-3 queries)
        """
        queries = []

        # Base query from task
        base_query = task_description.strip()
        if base_query:
            queries.append(base_query)

        # Analyze recent agent messages for gaps/keywords
        if self._processed_messages:
            # Get last few messages
            recent_msgs = self._processed_messages[-5:] if len(self._processed_messages) >= 5 else self._processed_messages

            # Extract keywords from agent outputs (simple approach)
            keywords = []
            for msg in recent_msgs:
                if 'agent_type' in msg.content:
                    agent_type = msg.content['agent_type']
                    if agent_type == 'research':
                        keywords.append('latest')
                        keywords.append('2024 2025')
                    elif agent_type == 'analysis':
                        keywords.append('detailed')
                    elif agent_type == 'critic':
                        keywords.append('verified')

            # Add enhanced query with keywords
            if keywords and base_query:
                enhanced_query = f"{base_query} {' '.join(set(keywords[:2]))}"
                queries.append(enhanced_query)

        # Limit to 2-3 queries
        return queries[:3]

    def _extract_and_store_relevant_info(self, search_results: List, task_description: str) -> None:
        """
        Use LLM to extract relevant information with deep search fallback.

        Phase 1: Extract from search snippets
        Phase 2: If insufficient, fetch and parse actual webpage content
        Phase 3: Store enhanced results in knowledge base

        Args:
            search_results: List of SearchResult objects
            task_description: Task to determine relevance
        """
        logger.info("=" * 60)
        logger.info("🔬 extract_and_store_relevant_info ENTRY")
        logger.info("=" * 60)
        logger.info(f"  Search results count: {len(search_results)}")
        logger.info(f"  Task: {task_description[:100]}...")

        # CRITICAL VALIDATION: Check prerequisites
        if not self.llm_client:
            logger.error("❌ FATAL: llm_client is None - CANNOT EXTRACT KNOWLEDGE")
            logger.error("❌ This will cause agents to have NO web search data")
            return

        if not self.knowledge_store:
            logger.error("❌ FATAL: knowledge_store is None - CANNOT STORE KNOWLEDGE")
            logger.error("❌ This will cause agents to have NO web search data")
            return

        logger.info("✓ Prerequisites validated (llm_client and knowledge_store available)")
        logger.info("")

        try:
            # PHASE 1: Extract from snippets
            logger.info("📄 PHASE 1: Extracting from search snippets...")
            formatted_snippets = self.web_search_client.format_results_for_llm(search_results)
            logger.info(f"  Formatted snippets length: {len(formatted_snippets)} characters")

            snippet_prompt = f"""Extract key facts relevant to '{task_description}' from these search snippets.

{formatted_snippets}

IMPORTANT: If the snippets contain the actual answer (e.g., specific date, time, number), provide it as bullet points.
If snippets only mention that information exists but don't contain the actual answer, respond EXACTLY with:
"NEED_PAGE_CONTENT"

Provide bullet points or the NEED_PAGE_CONTENT signal."""

            logger.info("  🤖 Calling LLM for snippet extraction...")

            # Initial extraction from snippets
            response = self.llm_client.complete(
                agent_id="web_search_extractor",
                system_prompt="You extract facts from search snippets. Be specific about what information is actually present.",
                user_prompt=snippet_prompt,
                temperature=0.2,
                max_tokens=300
            )

            initial_extraction = response.content.strip()
            logger.info(f"  ✓ LLM snippet extraction complete: {len(initial_extraction)} characters")
            logger.info(f"  📝 Extracted content preview: {initial_extraction[:150]}...")

            # PHASE 2: Deep search if needed
            page_data = None
            needs_deep_search = "NEED_PAGE_CONTENT" in initial_extraction or len(initial_extraction) < 50

            if needs_deep_search:
                logger.info("")
                logger.info("📄 PHASE 2: Deep search needed (snippets insufficient)")
                logger.info(f"  Reason: {'NEED_PAGE_CONTENT signal detected' if 'NEED_PAGE_CONTENT' in initial_extraction else f'extraction too short ({len(initial_extraction)} chars)'}")

                # Try fetching content from top results
                for i, result in enumerate(search_results[:3], 1):  # Try top 3 results
                    logger.info(f"  🌐 Attempting to fetch full page {i}/3...")
                    logger.info(f"     URL: {result.url}")
                    page_data = self.web_search_client.fetch_page_content(result.url, max_length=3000)
                    if page_data:
                        logger.info(f"  ✓ Successfully fetched {len(page_data['content'])} chars from {page_data['url'].split('/')[2]}")
                        logger.info(f"     Title: {page_data['title']}")
                        break
                    else:
                        logger.warning(f"  ✗ Failed to fetch page {i} - trying next result")

                if page_data:
                    logger.info("")
                    logger.info("  🤖 Calling LLM for deep content extraction...")
                    # Re-extract from full page content
                    content_prompt = f"""Extract SPECIFIC facts relevant to '{task_description}' from this webpage content.

Title: {page_data['title']}
URL: {page_data['url']}

Content:
{page_data['content'][:2000]}

Provide ONLY the specific factual answer as bullet points. Be precise and extract exact values (dates, times, numbers, etc.)."""

                    enhanced_response = self.llm_client.complete(
                        agent_id="web_search_deep_extractor",
                        system_prompt="You extract specific facts from webpage content. Be precise and factual. Extract exact values.",
                        user_prompt=content_prompt,
                        temperature=0.1,  # Very low for factual extraction
                        max_tokens=500
                    )

                    extracted_info = enhanced_response.content.strip()
                    logger.info(f"  ✓ Deep extraction complete: {len(extracted_info)} chars")
                    logger.info(f"  📝 Deep extracted content: {extracted_info[:200]}...")
                else:
                    logger.warning("")
                    logger.warning("  ⚠ Deep search FAILED - could not fetch page content from ANY result")
                    logger.warning("  ⚠ Falling back to snippet extraction (may be incomplete)")
                    extracted_info = initial_extraction if "NEED_PAGE_CONTENT" not in initial_extraction else ""
            else:
                logger.info("")
                logger.info("✓ PHASE 2 skipped: Snippet extraction sufficient")
                extracted_info = initial_extraction

            # PHASE 3: Store results
            logger.info("")
            logger.info("📦 PHASE 3: Storing in knowledge base...")

            if extracted_info and "NEED_PAGE_CONTENT" not in extracted_info and len(extracted_info) > 20:
                logger.info(f"  ✓ Extracted info valid: {len(extracted_info)} chars")

                # Prepare storage payload
                storage_content = {
                    "result": extracted_info,
                    "task": task_description,
                    "source_count": len(search_results),
                    "deep_search_used": page_data is not None,
                    "source_url": page_data['url'] if page_data else search_results[0].url,
                    "timestamp": time.time()
                }

                logger.info("  📦 Storage parameters:")
                logger.info(f"     - Type: DOMAIN_EXPERTISE")
                logger.info(f"     - Confidence: HIGH")
                logger.info(f"     - Domain: web_search")
                logger.info(f"     - Source: {storage_content['source_url'].split('/')[2] if '/' in storage_content['source_url'] else storage_content['source_url']}")
                logger.info(f"     - Deep search: {storage_content['deep_search_used']}")

                # Store in knowledge base
                try:
                    self.knowledge_store.store_knowledge(
                        knowledge_type=KnowledgeType.DOMAIN_EXPERTISE,
                        content=storage_content,
                        confidence_level=ConfidenceLevel.HIGH,
                        source_agent="websearch_coordinator",
                        domain="web_search",
                        tags=["web_search", "factual_data", "current_information"]
                    )
                    logger.info("  ✓ Knowledge stored successfully in knowledge base!")
                except Exception as store_error:
                    logger.error(f"  ❌ STORAGE FAILED: {store_error}", exc_info=True)
                    logger.error("  ❌ This means agents will NOT have this knowledge!")
                    raise

                logger.info("")
                logger.info("📄 Extracted Information (now available to agents):")
                # Log bullet points
                for line in extracted_info.split('\n'):
                    if line.strip():
                        logger.info(f"  • {line.strip()}")

                logger.info("")
                if page_data:
                    logger.info(f"✅ SUCCESS: Deep search information stored (source: {page_data['url'].split('/')[2]})")
                else:
                    logger.info("✅ SUCCESS: Snippet information stored in knowledge base")
                logger.info("✅ Agents will now be able to retrieve this knowledge")
            else:
                logger.error("")
                logger.error("❌ PHASE 3 FAILED: Extraction yielded no usable information")
                logger.error(f"   - extracted_info exists: {bool(extracted_info)}")
                logger.error(f"   - extracted_info length: {len(extracted_info) if extracted_info else 0}")
                logger.error(f"   - contains NEED_PAGE_CONTENT: {'NEED_PAGE_CONTENT' in extracted_info if extracted_info else 'N/A'}")
                logger.error("❌ NO KNOWLEDGE WILL BE STORED - agents will have no web search data!")

            logger.info("=" * 60)

        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ EXTRACTION FAILED WITH EXCEPTION: {e}", exc_info=True)
            logger.error("❌ NO KNOWLEDGE STORED - agents will NOT have web search data")
            logger.error("=" * 60)
