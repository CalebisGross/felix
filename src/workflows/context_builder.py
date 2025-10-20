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
            message_types=[MessageType.STATUS_UPDATE],  # Agents post STATUS_UPDATE messages
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
            agent_response = msg_content.get("content", "")
            agent_conf = msg_content.get("confidence", 0.0)
            sender_type = msg_content.get("agent_type", "unknown")

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
            logger.info(f"  Context size {original_size} exceeds limit {max_context_tokens}, compressing...")
            # TODO: Implement actual compression when Phase 4 is ready
            # For now, just truncate to most recent messages
            if len(context_history) > 5:
                context_history = context_history[-5:]
                compressed_size = sum(self.estimate_tokens(str(entry['response'])) for entry in context_history)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
                logger.info(f"  Compressed to {compressed_size} tokens (ratio: {compression_ratio:.2f})")

        # Retrieve relevant knowledge (if available)
        knowledge_entries = []
        if self.knowledge_store:
            try:
                from src.memory.knowledge_store import KnowledgeQuery, ConfidenceLevel
                relevant_knowledge = self.knowledge_store.retrieve_knowledge(
                    KnowledgeQuery(
                        domains=["workflow_task"],
                        min_confidence=ConfidenceLevel.MEDIUM,  # Use enum instead of float
                        limit=3
                    )
                )
                knowledge_entries = relevant_knowledge
                logger.info(f"  Retrieved {len(knowledge_entries)} relevant knowledge entries from store")
                if knowledge_entries:
                    for ke in knowledge_entries:
                        content_str = str(ke.content)
                        logger.info(f"    - Knowledge: {content_str[:50]}... (conf: {ke.confidence_level.value})")
            except Exception as e:
                logger.warning(f"  Knowledge retrieval failed: {e}")

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
