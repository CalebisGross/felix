"""
Memory Facade for the Felix Framework.

Provides unified interface to Felix memory systems including knowledge storage,
task memory, and context compression.

Key Features:
- Knowledge base integration (store/retrieve agent results)
- Task memory strategy recommendations
- Context compression for large inputs
- Memory system status and analytics
- Confidence level mapping
- Domain-based knowledge organization

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import logging
from typing import Dict, List, Optional, Any

# Import memory system components
from src.memory.knowledge_store import (
    KnowledgeStore, KnowledgeEntry, KnowledgeType,
    ConfidenceLevel, KnowledgeQuery
)
from src.memory.task_memory import TaskMemory, TaskComplexity
from src.memory.context_compression import ContextCompressor, CompressionStrategy

# Set up logging
logger = logging.getLogger(__name__)


class MemoryFacade:
    """
    Unified interface to Felix memory systems.

    Responsibilities:
    - Store agent results as persistent knowledge
    - Retrieve relevant knowledge by domain/type/confidence
    - Get task strategy recommendations from task memory
    - Compress large context when needed
    - Provide memory system analytics and summaries
    """

    def __init__(self,
                 knowledge_store: Optional[KnowledgeStore] = None,
                 task_memory: Optional[TaskMemory] = None,
                 context_compressor: Optional[ContextCompressor] = None,
                 memory_enabled: bool = True):
        """
        Initialize Memory Facade.

        Args:
            knowledge_store: Knowledge persistence backend
            task_memory: Task pattern storage
            context_compressor: Context compression system
            memory_enabled: Enable/disable memory systems
        """
        self.knowledge_store = knowledge_store
        self.task_memory = task_memory
        self.context_compressor = context_compressor
        self._memory_enabled = memory_enabled

        if memory_enabled:
            logger.info("✓ MemoryFacade initialized (memory enabled)")
        else:
            logger.info("✓ MemoryFacade initialized (memory disabled)")

    def store_agent_result_as_knowledge(self, agent_id: str, content: str,
                                       confidence: float, domain: str = "general",
                                       tags: Optional[List[str]] = None) -> bool:
        """
        Store agent result as knowledge in the persistent knowledge base.

        Args:
            agent_id: ID of the agent producing the result
            content: Content of the result to store
            confidence: Confidence level of the result (0.0 to 1.0)
            domain: Domain/category for the knowledge
            tags: Optional tags for the knowledge entry

        Returns:
            True if knowledge was stored successfully, False otherwise
        """
        if not self._memory_enabled or not self.knowledge_store:
            return False

        try:
            # Convert confidence to ConfidenceLevel enum
            if confidence >= 0.8:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence >= 0.6:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW

            # Store in knowledge base using correct method signature
            entry_id = self.knowledge_store.store_knowledge(
                knowledge_type=KnowledgeType.TASK_RESULT,
                content={"result": content, "confidence": confidence},
                confidence_level=confidence_level,
                source_agent=agent_id,
                domain=domain,
                tags=tags
            )
            return entry_id is not None

        except Exception as e:
            logger.error(f"Failed to store knowledge from agent {agent_id}: {e}")
            return False

    def retrieve_relevant_knowledge(self, domain: Optional[str] = None,
                                   knowledge_type: Optional[KnowledgeType] = None,
                                   keywords: Optional[List[str]] = None,
                                   min_confidence: Optional[ConfidenceLevel] = None,
                                   limit: int = 10) -> List[KnowledgeEntry]:
        """
        Retrieve relevant knowledge from the knowledge base.

        Args:
            domain: Filter by domain
            knowledge_type: Filter by knowledge type
            keywords: Keywords to search for
            min_confidence: Minimum confidence level
            limit: Maximum number of entries to return

        Returns:
            List of relevant knowledge entries
        """
        if not self._memory_enabled or not self.knowledge_store:
            return []

        try:
            query = KnowledgeQuery(
                knowledge_types=[knowledge_type] if knowledge_type else None,
                domains=[domain] if domain else None,
                content_keywords=keywords,
                min_confidence=min_confidence,
                limit=limit
            )
            return self.knowledge_store.retrieve_knowledge(query)
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []

    def retrieve_knowledge_with_query(self, query: 'KnowledgeQuery') -> List[KnowledgeEntry]:
        """
        Retrieve knowledge using a full KnowledgeQuery object.

        This provides maximum flexibility for complex queries including
        time ranges, multiple domains, and custom filtering criteria.

        Args:
            query: KnowledgeQuery object with full query parameters

        Returns:
            List of matching knowledge entries

        Example:
            >>> query = KnowledgeQuery(
            ...     domains=["web_search", "workflow_task"],
            ...     min_confidence=ConfidenceLevel.MEDIUM,
            ...     time_range=(start_time, end_time),
            ...     limit=5
            ... )
            >>> entries = memory_facade.retrieve_knowledge_with_query(query)
        """
        if not self._memory_enabled or not self.knowledge_store:
            return []

        try:
            return self.knowledge_store.retrieve_knowledge(query)
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge with query: {e}")
            return []

    def get_task_strategy_recommendations(self, task_description: str,
                                        task_type: str = "general",
                                        complexity: str = "MODERATE") -> Dict[str, Any]:
        """
        Get strategy recommendations based on task memory.

        Args:
            task_description: Description of the task
            task_type: Type of task (e.g., "research", "analysis", "synthesis")
            complexity: Task complexity level ("SIMPLE", "MODERATE", "COMPLEX", "VERY_COMPLEX")

        Returns:
            Dictionary containing strategy recommendations
        """
        if not self._memory_enabled or not self.task_memory:
            return {}

        try:
            # Convert string complexity to enum
            complexity_enum = TaskComplexity.MODERATE
            if complexity.upper() == "SIMPLE":
                complexity_enum = TaskComplexity.SIMPLE
            elif complexity.upper() == "COMPLEX":
                complexity_enum = TaskComplexity.COMPLEX
            elif complexity.upper() == "VERY_COMPLEX":
                complexity_enum = TaskComplexity.VERY_COMPLEX

            return self.task_memory.recommend_strategy(
                task_description=task_description,
                task_type=task_type,
                complexity=complexity_enum
            )
        except Exception as e:
            logger.error(f"Failed to get strategy recommendations: {e}")
            return {}

    def compress_large_context(self, context: str,
                              strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE_SUMMARY,
                              target_size: Optional[int] = None):
        """
        Compress large context using the context compression system.

        Args:
            context: Content to compress
            strategy: Compression strategy to use
            target_size: Optional target size for compression

        Returns:
            CompressedContext object or None if compression failed
        """
        if not self._memory_enabled or not self.context_compressor:
            return None

        try:
            # Convert string context to dict format expected by compressor
            context_dict = {"main_content": context}
            return self.context_compressor.compress_context(
                context=context_dict,
                target_size=target_size,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Failed to compress context: {e}")
            return None

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory system status and contents.

        Returns:
            Dictionary with memory system summary including:
                - memory_enabled: Whether memory systems are active
                - knowledge_entries: Total knowledge entries count
                - knowledge_by_domain: Breakdown by domain
                - task_patterns: Number of task patterns stored
                - task_executions: Total task execution records
                - success_rate: Overall success rate across tasks
                - top_task_types: Most common task types
                - success_by_complexity: Success rates by complexity level
        """
        if not self._memory_enabled:
            return {
                "knowledge_entries": 0,
                "task_patterns": 0,
                "memory_enabled": False
            }

        try:
            summary: Dict[str, Any] = {"memory_enabled": True}

            if self.knowledge_store:
                # Get knowledge entry count using proper query
                query = KnowledgeQuery(limit=1000)
                all_knowledge = self.knowledge_store.retrieve_knowledge(query)
                summary["knowledge_entries"] = len(all_knowledge)

                # Get domain breakdown
                domains: Dict[str, int] = {}
                for entry in all_knowledge:
                    domains[entry.domain] = domains.get(entry.domain, 0) + 1
                summary["knowledge_by_domain"] = domains
            else:
                summary["knowledge_entries"] = 0
                summary["knowledge_by_domain"] = {}

            if self.task_memory:
                # Get task pattern count and summary
                memory_summary = self.task_memory.get_memory_summary()
                summary["task_patterns"] = memory_summary.get("total_patterns", 0)
                summary["task_executions"] = memory_summary.get("total_executions", 0)

                # Handle success rate calculation from outcome distribution
                outcome_dist = memory_summary.get("outcome_distribution", {})
                total_executions = sum(outcome_dist.values()) if outcome_dist else 0
                if total_executions > 0:
                    successful_outcomes = outcome_dist.get("success", 0) + outcome_dist.get("partial_success", 0)
                    summary["success_rate"] = successful_outcomes / total_executions
                else:
                    summary["success_rate"] = 0.0

                # Get top task types
                summary["top_task_types"] = memory_summary.get("top_task_types", {})
                summary["success_by_complexity"] = memory_summary.get("success_by_complexity", {})
            else:
                summary["task_patterns"] = 0
                summary["task_executions"] = 0
                summary["success_rate"] = 0.0
                summary["top_task_types"] = {}
                summary["success_by_complexity"] = {}

            return summary

        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {
                "knowledge_entries": 0,
                "task_patterns": 0,
                "memory_enabled": True,
                "error": str(e)
            }

    def is_enabled(self) -> bool:
        """Check if memory systems are enabled."""
        return self._memory_enabled

    def has_knowledge_store(self) -> bool:
        """Check if knowledge store is available."""
        return self.knowledge_store is not None

    def has_task_memory(self) -> bool:
        """Check if task memory is available."""
        return self.task_memory is not None

    def has_context_compressor(self) -> bool:
        """Check if context compressor is available."""
        return self.context_compressor is not None
