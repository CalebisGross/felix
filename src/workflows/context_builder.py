"""
Collaborative Context Builder for Felix Framework

Builds enriched context for agents by:
- Retrieving recent messages from CentralPost
- Compressing accumulated context when needed
- Integrating relevant knowledge from memory
- Creating agent-type-specific prompts
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Use 'felix_workflows' logger to match GUI configuration and ensure visibility
logger = logging.getLogger('felix_workflows')


@dataclass
class EnrichedContext:
    """Container for enriched agent context."""
    task_description: str
    context_history: List[Dict[str, Any]]
    original_context_size: int
    compressed_context_size: int
    compression_ratio: float
    knowledge_entries: List[Any]
    message_count: int


class CollaborativeContextBuilder:
    """
    Builds enriched context for agents from accumulated messages.

    This enables true collaborative multi-agent workflows where each agent
    builds upon the work of previous agents rather than working in isolation.
    """

    def __init__(self, central_post, knowledge_store=None, context_compressor=None):
        """
        Initialize collaborative context builder.

        Args:
            central_post: CentralPost instance for message retrieval
            knowledge_store: Optional KnowledgeStore for memory integration
            context_compressor: Optional ContextCompressor for context optimization
        """
        self.central_post = central_post
        self.knowledge_store = knowledge_store
        self.context_compressor = context_compressor

        # Track context building statistics
        self.contexts_built = 0
        self.total_compression_ratio = 0.0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using 4 chars ≈ 1 token heuristic.

        This is a standard approximation for English text. More accurate
        than character count, doesn't require full tokenizer library.

        Args:
            text: Text to estimate token count for

        Returns:
            Estimated number of tokens
        """
        return len(text) // 4

    def build_agent_context(self,
                           original_task: str,
                           agent_type: str,
                           agent_id: str,
                           current_time: float,
                           max_context_tokens: int = 2000,
                           message_limit: int = 10) -> EnrichedContext:
        """
        Build enriched context for agent including previous agent outputs.

        Args:
            original_task: The original user task
            agent_type: Type of agent (research, critic, analysis, synthesis)
            agent_id: ID of the agent receiving context
            current_time: Current simulation time
            max_context_tokens: Maximum tokens for context (for compression)
            message_limit: Maximum messages to retrieve

        Returns:
            EnrichedContext with task description and accumulated context
        """
        logger.info(f"="*60)
        logger.info(f"Building collaborative context for {agent_type} agent: {agent_id}")
        logger.info(f"  Task: {original_task[:80]}..." if len(original_task) > 80 else f"  Task: {original_task}")

        # Retrieve recent messages from CentralPost (exclude self)
        from src.communication.central_post import MessageType
        recent_messages = self.central_post.get_recent_messages(
            limit=message_limit,
            since_time=None,  # Get all messages
            message_types=[MessageType.STATUS_UPDATE, MessageType.SYSTEM_ACTION_RESULT],  # Include agent outputs AND system command results
            exclude_sender=agent_id  # Don't include own previous messages
        )

        logger.info(f"  Retrieved {len(recent_messages)} messages from CentralPost")
        if recent_messages:
            logger.info(f"  Previous agents: {[msg.sender_id for msg in recent_messages]}")

        # Build context history from messages
        context_history = []
        original_size = self.estimate_tokens(original_task)

        logger.info(f"  Building context history in chronological order...")
        for i, msg in enumerate(reversed(recent_messages), 1):  # Chronological order
            msg_content = msg.content

            # Handle different message types
            if msg.message_type == MessageType.STATUS_UPDATE:
                # Agent response
                agent_response = msg_content.get("content", "")
                agent_conf = msg_content.get("confidence", 0.0)
                sender_type = msg_content.get("agent_type", "unknown")

            elif msg.message_type == MessageType.SYSTEM_ACTION_RESULT:
                # System command result
                command = msg_content.get("command", "")
                stdout = msg_content.get("stdout", "")
                stderr = msg_content.get("stderr", "")
                success = msg_content.get("success", False)
                exit_code = msg_content.get("exit_code", -1)

                # Format as readable context entry
                agent_response = f"System Command Execution:\nCommand: {command}\nSuccess: {success}\nExit Code: {exit_code}"
                if stdout:
                    agent_response += f"\nOutput: {stdout}"
                if stderr:
                    agent_response += f"\nErrors: {stderr}"

                agent_conf = 1.0  # High confidence for actual command results
                sender_type = "system_action"

            else:
                # Unknown message type, skip
                continue

            context_entry = {
                "agent_id": msg.sender_id,
                "agent_type": sender_type,
                "response": agent_response,
                "confidence": agent_conf,
                "timestamp": msg.timestamp
            }

            context_history.append(context_entry)
            original_size += self.estimate_tokens(str(agent_response))

            # Log each context entry being added
            response_preview = agent_response[:60] + "..." if len(agent_response) > 60 else agent_response
            logger.info(f"    {i}. {sender_type} ({msg.sender_id}): conf={agent_conf:.2f}, '{response_preview}'")

        # ENFORCE TOKEN BUDGET: Ensure context never exceeds agent capacity
        # Reserve space for collaborative instructions (~150 tokens after 4:1 ratio)
        available_tokens = max_context_tokens - 150  # Reserve for instructions (600 chars / 4)
        cumulative_tokens = self.estimate_tokens(original_task)

        # Build context from MOST RECENT messages first, staying within budget
        enforced_context_history = []
        for entry in reversed(context_history):  # Newest first
            entry_size = self.estimate_tokens(str(entry['response']))
            if cumulative_tokens + entry_size <= available_tokens:
                enforced_context_history.insert(0, entry)  # Maintain chronological order
                cumulative_tokens += entry_size
            else:
                # Stop adding messages when budget would be exceeded
                break

        # Report enforcement results
        if len(enforced_context_history) < len(context_history):
            logger.info(f"  Token budget enforcement: kept {len(enforced_context_history)}/{len(context_history)} messages ({cumulative_tokens}/{available_tokens} tokens)")

        context_history = enforced_context_history
        original_size = cumulative_tokens

        # Apply compression if context is too large
        compressed_size = original_size
        compression_ratio = 1.0

        if self.context_compressor and original_size > max_context_tokens:
            logger.info(f"  Context size {original_size} exceeds limit {max_context_tokens}, applying compression...")

            # Combine all context history into a single text for compression
            combined_context = "\n\n".join([
                f"[{entry.get('agent_type', 'Unknown')}]: {entry.get('response', '')}"
                for entry in context_history
            ])

            try:
                # Apply actual compression using the configured strategy
                compressed_result = self.context_compressor.compress_context(
                    combined_context,
                    max_tokens=int(max_context_tokens * 0.8)  # Leave 20% buffer for headers
                )

                # Store both compressed and original for reference
                if compressed_result and compressed_result.compressed_text:
                    # Create compressed context entry
                    compressed_entry = {
                        "agent_type": "system_compression",
                        "response": compressed_result.compressed_text,
                        "confidence": 0.9,  # High confidence for compressed content
                        "original_entries": len(context_history),
                        "compression_method": compressed_result.method.value
                    }

                    # Keep most recent 2 entries uncompressed + compressed summary
                    recent_entries = context_history[-2:] if len(context_history) > 2 else context_history
                    context_history = [compressed_entry] + recent_entries

                    compressed_size = self.estimate_tokens(compressed_result.compressed_text)
                    for entry in recent_entries:
                        compressed_size += self.estimate_tokens(str(entry['response']))

                    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                    logger.info(f"  ✓ Compressed {compressed_result.original_entries} entries to {compressed_size} tokens (ratio: {compression_ratio:.2f})")
                    logger.info(f"  Compression method: {compressed_result.method.value}")
                else:
                    # Fallback to truncation if compression fails
                    logger.warning("  Compression returned empty result, falling back to truncation")
                    context_history = context_history[-5:]
                    compressed_size = sum(self.estimate_tokens(str(entry['response'])) for entry in context_history)
                    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

            except Exception as e:
                logger.warning(f"  Compression failed: {e}, falling back to truncation")
                context_history = context_history[-5:]
                compressed_size = sum(self.estimate_tokens(str(entry['response'])) for entry in context_history)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        # Retrieve relevant knowledge (if available)
        logger.info("")
        logger.info("🔍 KNOWLEDGE RETRIEVAL")
        logger.info("="*60)

        knowledge_entries = []
        if self.knowledge_store:
            logger.info("  ✓ Knowledge store available")
            try:
                import time as time_module
                from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel
                from src.workflows.truth_assessment import detect_query_type, QueryType

                # Detect query type to determine appropriate freshness window
                current_time = time_module.time()
                query_type = detect_query_type(original_task)
                freshness_limits = {
                    QueryType.TIME: 300,           # 5 minutes for time queries
                    QueryType.DATE: 3600,          # 1 hour for date queries
                    QueryType.CURRENT_EVENT: 1800, # 30 minutes for current events
                    QueryType.GENERAL_FACT: 86400, # 24 hours for general facts
                    QueryType.ANALYSIS: 86400,     # 24 hours for analysis
                }
                max_age = freshness_limits.get(query_type, 3600)  # Default to 1 hour
                time_window_start = current_time - max_age

                logger.info(f"  📝 Query parameters:")
                logger.info(f"     - Domains: web_search, workflow_task")
                logger.info(f"     - Min confidence: MEDIUM")
                logger.info(f"     - Query type: {query_type.value} (freshness: {max_age}s / {max_age/60:.1f} min)")
                logger.info(f"     - Time range: {int(time_window_start)} to {int(current_time)}")
                logger.info(f"     - Limit: 5 entries")

                # CRITICAL: Retrieve from BOTH web_search and workflow_task domains
                # Web search results are stored in "web_search" domain (central_post.py)
                # Agent results are stored in "workflow_task" domain
                logger.info("  🔍 Calling knowledge_store.retrieve_knowledge()...")
                relevant_knowledge = self.knowledge_store.retrieve_knowledge(
                    KnowledgeQuery(
                        domains=["web_search", "workflow_task"],  # Include web search results!
                        min_confidence=ConfidenceLevel.MEDIUM,
                        time_range=(time_window_start, current_time),  # Dynamic freshness based on query type
                        limit=5  # Increased to allow more web search results
                    )
                )
                knowledge_entries = relevant_knowledge

                logger.info(f"  ✓ Retrieved {len(knowledge_entries)} entries from knowledge store")

                if len(knowledge_entries) == 0:
                    logger.warning("")
                    logger.warning("  ⚠️ NO KNOWLEDGE ENTRIES FOUND!")
                    logger.warning("  ⚠️ This means either:")
                    logger.warning("     1. Web search hasn't run yet")
                    logger.warning("     2. Web search failed to store knowledge")
                    logger.warning(f"     3. Knowledge is older than freshness window ({max_age/60:.1f} min)")
                    logger.warning("  ⚠️ Agents will NOT have web search data!")
                else:
                    # Sort web_search entries first (they're typically most relevant for current queries)
                    knowledge_entries = sorted(knowledge_entries, key=lambda ke: (ke.domain != "web_search", ke.created_at), reverse=False)

                    logger.info("")
                    logger.info(f"  📚 Knowledge Entries (sorted with web_search first):")
                    for i, ke in enumerate(knowledge_entries, 1):
                        # Extract content from dictionary if present
                        if isinstance(ke.content, dict):
                            # Prefer 'result' key which contains extracted web search info
                            content_str = ke.content.get('result', str(ke.content))
                        else:
                            content_str = str(ke.content)

                        # Use longer display for web_search domain
                        max_chars = 200 if ke.domain == "web_search" else 50
                        truncated = content_str[:max_chars] + "..." if len(content_str) > max_chars else content_str

                        # Add emoji for web_search entries
                        prefix = "🌐" if ke.domain == "web_search" else "📝"

                        logger.info(f"    {i}. {prefix} [{ke.domain}] {ke.confidence_level.value}")
                        logger.info(f"       Content: {truncated}")

                        # Log additional metadata for web_search entries
                        if ke.domain == "web_search" and isinstance(ke.content, dict):
                            if 'source_url' in ke.content:
                                logger.info(f"       Source: {ke.content['source_url']}")
                            if 'deep_search_used' in ke.content:
                                logger.info(f"       Deep search: {ke.content['deep_search_used']}")

            except Exception as e:
                logger.error(f"  ❌ Knowledge retrieval FAILED: {e}", exc_info=True)
                logger.error(f"  ❌ Agents will NOT have any knowledge entries!")
        else:
            logger.warning("  ⚠️ Knowledge store NOT AVAILABLE")
            logger.warning("  ⚠️ Agents will NOT have access to stored knowledge")

        logger.info("="*60)

        # Track statistics
        self.contexts_built += 1
        self.total_compression_ratio += compression_ratio

        enriched = EnrichedContext(
            task_description=original_task,
            context_history=context_history,
            original_context_size=original_size,
            compressed_context_size=compressed_size,
            compression_ratio=compression_ratio,
            knowledge_entries=knowledge_entries,
            message_count=len(recent_messages)
        )

        logger.info(f"  ✓ Collaborative context built successfully:")
        logger.info(f"    - Messages: {len(context_history)}")
        logger.info(f"    - Knowledge entries: {len(knowledge_entries)}")
        logger.info(f"    - Context size: {compressed_size} tokens (budget: {max_context_tokens})")
        logger.info(f"    - Compression ratio: {compression_ratio:.2f}")
        logger.info(f"="*60)

        return enriched

    def get_statistics(self) -> Dict[str, Any]:
        """Get context building statistics."""
        avg_compression = (self.total_compression_ratio / self.contexts_built
                          if self.contexts_built > 0 else 0.0)

        return {
            "contexts_built": self.contexts_built,
            "average_compression_ratio": avg_compression
        }
