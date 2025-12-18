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

    def retrieve_tool_instructions(self, tool_requirements: Dict[str, bool]) -> tuple[str, List[str]]:
        """
        Retrieve tool instructions based on task requirements (conditional tool injection).

        This implements the "subconscious memory" pattern where agents only receive
        instructions for tools they actually need, reducing token waste.

        Args:
            tool_requirements: Dictionary with tool requirement flags:
                {'needs_file_ops': bool, 'needs_web_search': bool, 'needs_system_commands': bool}

        Returns:
            Tuple of (instructions_string, list_of_knowledge_ids):
            - instructions_string: Assembled tool instructions for required tools
            - list_of_knowledge_ids: IDs of tool instruction entries (for meta-learning)

        Example:
            >>> tool_reqs = {'needs_file_ops': True, 'needs_web_search': False, 'needs_system_commands': False}
            >>> instructions, ids = memory_facade.retrieve_tool_instructions(tool_reqs)
            >>> # Returns file operation instructions and their knowledge IDs
        """
        if not self._memory_enabled or not self.knowledge_store:
            logger.debug("Tool instruction retrieval skipped: memory disabled or no knowledge store")
            return "", []

        if not tool_requirements or not any(tool_requirements.values()):
            logger.debug("No tools required for this task - skipping tool instruction retrieval")
            return "", []

        try:
            # Map tool requirement flags to tool instruction identifiers
            tool_map = {
                'needs_file_ops': 'file_operations',
                'needs_web_search': 'web_search',
                'needs_system_commands': 'system_commands'
            }

            # Collect needed tool names
            needed_tools = [
                tool_name
                for flag, tool_name in tool_map.items()
                if tool_requirements.get(flag, False)
            ]

            if not needed_tools:
                return "", []

            logger.info(f"Retrieving tool instructions for: {', '.join(needed_tools)}")

            # Retrieve tool instructions from knowledge store (domain="tool_instructions")
            assembled_instructions = []
            knowledge_ids = []  # Track IDs for meta-learning

            for tool_name in needed_tools:
                # Query by domain and check tool_name in content dict
                # Note: We use domain filtering and then filter by tool_name in content
                query = KnowledgeQuery(
                    domains=["tool_instructions"],
                    limit=10  # Get all tool instructions, we'll filter below
                )
                all_tool_entries = self.knowledge_store.retrieve_knowledge(query)

                # Filter to find the specific tool by tool_name in content
                entries = [
                    entry for entry in all_tool_entries
                    if isinstance(entry.content, dict) and entry.content.get('tool_name') == tool_name
                ]

                if entries:
                    # Extract instruction text and ID from knowledge entry
                    for entry in entries:
                        if isinstance(entry.content, dict):
                            instruction_text = entry.content.get('instructions', str(entry.content))
                        else:
                            instruction_text = str(entry.content)

                        assembled_instructions.append(instruction_text)
                        knowledge_ids.append(entry.knowledge_id)  # Track ID for meta-learning
                        logger.debug(f"  ✓ Retrieved {tool_name} instructions ({len(instruction_text)} chars, ID: {entry.knowledge_id})")
                else:
                    logger.warning(f"  ⚠ No instructions found for tool: {tool_name}")

            # Assemble all instructions with separators
            if assembled_instructions:
                result = "\n\n".join(assembled_instructions)
                logger.info(f"✓ Assembled tool instructions: {len(result)} characters, {len(assembled_instructions)} tools")
                logger.info(f"  Tool instruction IDs: {knowledge_ids}")
                return result, knowledge_ids
            else:
                logger.warning("No tool instructions could be retrieved")
                return "", []

        except Exception as e:
            logger.error(f"Failed to retrieve tool instructions: {e}")
            return "", []

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
                # Use efficient statistics method instead of retrieving all entries
                knowledge_stats = self.knowledge_store.get_knowledge_summary()
                summary["knowledge_entries"] = knowledge_stats.get("total_entries", 0)
                summary["knowledge_by_domain"] = knowledge_stats.get("domain_distribution", {})
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

    def record_knowledge_usage(self, workflow_id: str, knowledge_ids: List[str],
                               task_type: str, useful_score: float) -> bool:
        """
        Record knowledge usage for meta-learning.

        Routes through memory facade for consistent auditing (Issue #4.5).

        Args:
            workflow_id: ID of the workflow using the knowledge
            knowledge_ids: List of knowledge entry IDs that were used
            task_type: Type of task that used the knowledge
            useful_score: Usefulness score (0.0-1.0) from synthesis confidence

        Returns:
            True if recorded successfully, False otherwise
        """
        if not self._memory_enabled or not self.knowledge_store:
            return False

        try:
            self.knowledge_store.record_knowledge_usage(
                workflow_id=workflow_id,
                knowledge_ids=knowledge_ids,
                task_type=task_type,
                useful_score=useful_score
            )
            logger.debug(f"Recorded knowledge usage for {len(knowledge_ids)} entries "
                        f"(workflow={workflow_id}, task_type={task_type})")
            return True
        except Exception as e:
            logger.warning(f"Failed to record knowledge usage: {e}")
            return False
