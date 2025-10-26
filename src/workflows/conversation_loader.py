"""
Conversation Context Loader Module

This module provides functionality to load and build context from previous workflows,
enabling conversation continuity across multiple workflow executions.
"""

import logging
from typing import Dict, Any, Optional, List
from src.memory.workflow_history import WorkflowHistory, WorkflowOutput

logger = logging.getLogger(__name__)


class ConversationContextLoader:
    """
    Loads context from previous workflows to enable conversation continuity.
    """

    def __init__(self, workflow_history: WorkflowHistory):
        """
        Initialize the conversation context loader.

        Args:
            workflow_history: WorkflowHistory instance for database access
        """
        self.workflow_history = workflow_history

    def load_parent_workflow(self, parent_workflow_id: int) -> Optional[Dict[str, Any]]:
        """
        Load complete parent workflow data.

        Args:
            parent_workflow_id: ID of the parent workflow

        Returns:
            Dictionary with parent workflow data, or None if not found
        """
        try:
            parent_output = self.workflow_history.get_workflow_by_id(parent_workflow_id)
            if not parent_output:
                logger.warning(f"Parent workflow {parent_workflow_id} not found")
                return None

            logger.info(f"Loaded parent workflow {parent_workflow_id}")
            return {
                "workflow_id": parent_output.workflow_id,
                "task_input": parent_output.task_input,
                "final_synthesis": parent_output.final_synthesis,
                "confidence": parent_output.confidence,
                "agents_count": parent_output.agents_count,
                "metadata": parent_output.metadata
            }

        except Exception as e:
            logger.error(f"Failed to load parent workflow: {e}", exc_info=True)
            return None

    def build_continuation_prompt(self,
                                   parent_result: Dict[str, Any],
                                   follow_up_question: str,
                                   max_context_length: int = 1000) -> str:
        """
        Build a prompt that includes parent context and follow-up question.

        Args:
            parent_result: Parent workflow result dictionary
            follow_up_question: User's follow-up question
            max_context_length: Maximum characters for parent synthesis context

        Returns:
            Formatted continuation prompt
        """
        parent_task = parent_result.get("task_input", "")
        parent_synthesis = parent_result.get("final_synthesis", "")

        # Truncate parent synthesis if too long
        if len(parent_synthesis) > max_context_length:
            parent_synthesis = parent_synthesis[:max_context_length] + "... [truncated]"

        continuation_prompt = f"""[CONTINUING FROM PREVIOUS WORKFLOW]

Previous Task: {parent_task}

Previous Synthesis:
{parent_synthesis}

---

Follow-up Question: {follow_up_question}

[Please build upon the previous context and address the follow-up question]"""

        logger.info(f"Built continuation prompt (parent context: {len(parent_synthesis)} chars)")
        return continuation_prompt

    def extract_key_knowledge_from_parent(self, parent_result: Dict[str, Any]) -> List[str]:
        """
        Extract key knowledge entries from parent workflow metadata.

        Args:
            parent_result: Parent workflow result dictionary

        Returns:
            List of knowledge entry IDs or summaries
        """
        try:
            metadata = parent_result.get("metadata", {})
            knowledge_entries = metadata.get("knowledge_entries", [])

            # Extract just the key info we need
            key_knowledge = []
            for entry in knowledge_entries:
                if isinstance(entry, dict):
                    summary = entry.get("content", "")
                    if summary:
                        key_knowledge.append(summary)

            logger.info(f"Extracted {len(key_knowledge)} knowledge entries from parent")
            return key_knowledge

        except Exception as e:
            logger.warning(f"Failed to extract knowledge from parent: {e}")
            return []

    def get_conversation_history(self, workflow_id: int) -> List[WorkflowOutput]:
        """
        Get full conversation thread history.

        Args:
            workflow_id: ID of any workflow in the thread

        Returns:
            List of WorkflowOutput objects in chronological order
        """
        try:
            thread = self.workflow_history.get_conversation_thread(workflow_id)
            logger.info(f"Retrieved conversation thread with {len(thread)} workflows")
            return thread

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}", exc_info=True)
            return []

    def build_multi_turn_context(self,
                                  conversation_thread: List[WorkflowOutput],
                                  max_turns: int = 3) -> str:
        """
        Build context from multiple conversation turns.

        Args:
            conversation_thread: List of workflow outputs in chronological order
            max_turns: Maximum number of previous turns to include

        Returns:
            Formatted multi-turn context string
        """
        if not conversation_thread:
            return ""

        # Take the last N turns (or all if fewer than N)
        recent_turns = conversation_thread[-max_turns:] if len(conversation_thread) > max_turns else conversation_thread

        context_parts = ["[CONVERSATION HISTORY]"]

        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"\nTurn {i}:")
            context_parts.append(f"Task: {turn.task_input}")

            # Truncate synthesis for readability
            synthesis = turn.final_synthesis
            if len(synthesis) > 500:
                synthesis = synthesis[:500] + "... [truncated]"

            context_parts.append(f"Result: {synthesis}")
            context_parts.append(f"Confidence: {turn.confidence:.2f}")
            context_parts.append("---")

        full_context = "\n".join(context_parts)
        logger.info(f"Built multi-turn context from {len(recent_turns)} turns")
        return full_context

    def should_compress_parent_context(self,
                                       parent_synthesis: str,
                                       threshold: int = 2000) -> bool:
        """
        Determine if parent context should be compressed.

        Args:
            parent_synthesis: Parent synthesis text
            threshold: Character threshold for compression

        Returns:
            True if compression recommended
        """
        return len(parent_synthesis) > threshold
