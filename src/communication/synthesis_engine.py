"""
Synthesis Engine for the Felix Framework.

Handles final output synthesis from agent communications, implementing
the convergence point where all helical agent trajectories meet.

Key Features:
- Adaptive synthesis parameters based on agent confidence
- Task complexity-aware synthesis (SIMPLE/MEDIUM/COMPLEX)
- Integration of agent outputs AND system command results
- Temperature and token budget optimization
- Helical-aware synthesis prompting
- CentralPost represents the central axis convergence point

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import logging
from typing import Dict, List, Any, Optional

# Import message types
from src.communication.message_types import Message, MessageType

# Import validation functions
from src.workflows.truth_assessment import calculate_validation_score, get_validation_flags

# Set up logging
logger = logging.getLogger(__name__)


class SynthesisEngine:
    """
    Manages synthesis of final outputs from agent communications.

    Responsibilities:
    - Calculate adaptive synthesis parameters
    - Build synthesis prompts with helical context
    - Integrate agent outputs and system results
    - Generate final synthesized output via LLM
    """

    def __init__(self,
                 llm_client: Any,
                 get_recent_messages_callback: Any):
        """
        Initialize Synthesis Engine.

        Args:
            llm_client: LLM client for synthesis generation
            get_recent_messages_callback: Callback to get recent messages from CentralPost
        """
        self.llm_client = llm_client
        self._get_recent_messages = get_recent_messages_callback

        logger.info("✓ SynthesisEngine initialized")

    def classify_task_complexity(self, task_description: str) -> str:
        """
        Classify task complexity to optimize synthesis strategy.

        Task complexity affects:
        - Synthesis temperature and token budgets
        - Agent spawning strategies
        - Workflow execution parameters

        Args:
            task_description: The task description from user

        Returns:
            Task complexity: "SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX"

        Examples:
            >>> engine.classify_task_complexity("What time is it?")
            'SIMPLE_FACTUAL'
            >>> engine.classify_task_complexity("Explain quantum computing")
            'MEDIUM'
            >>> engine.classify_task_complexity("Design a microservices architecture")
            'COMPLEX'
        """
        import re

        task_lower = task_description.lower()

        # Simple factual patterns that can be answered quickly with web search
        simple_patterns = [
            r'\b(what|when|who|where)\s+(is|are|was|were)\s+(the\s+)?current',
            r'\bwhat\s+time\b',
            r'\bwhat\s+date\b',
            r'\btoday\'?s?\s+(date|time)',
            r'\bcurrent\s+(time|date|datetime)',
            r'\bwho\s+(won|is|was)\b',
            r'\bwhen\s+(did|is|was)\b',
            r'\bhow\s+many\b.*\b(now|current|today)',
            r'\blatest\s+(news|update)\b',
            # Greetings and trivial social queries
            r'^\s*(hello|hi|hey|greetings?|howdy|yo)\s*[!.?]*\s*$',
            r'^\s*good\s+(morning|afternoon|evening|night|day)\s*[!.?]*\s*$',
        ]

        # Check for simple factual patterns
        for pattern in simple_patterns:
            if re.search(pattern, task_lower):
                return "SIMPLE_FACTUAL"

        # Medium complexity: specific questions but may need analysis
        medium_patterns = [
            r'\bexplain\b',
            r'\bcompare\b',
            r'\bwhat\s+are\s+the\s+(benefits|advantages|disadvantages)',
            r'\bhow\s+does\b',
            r'\bhow\s+to\b',
            r'\blist\b',
            r'\bsummarize\b',
        ]

        for pattern in medium_patterns:
            if re.search(pattern, task_lower):
                return "MEDIUM"

        # Default to complex for open-ended, analytical tasks
        return "COMPLEX"

    def synthesize_agent_outputs(self, task_description: str, max_messages: int = 20,
                                 task_complexity: str = "COMPLEX") -> Dict[str, Any]:
        """
        Synthesize final output from all agent communications.

        This is the core synthesis capability of CentralPost, replacing the need for
        synthesis agents. CentralPost represents the central axis of the helix where
        all agent trajectories converge.

        Args:
            task_description: Original task description
            max_messages: Maximum number of agent messages to include in synthesis
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Dict containing:
                - synthesis_content: Final synthesized output text
                - confidence: Synthesis confidence score (0.0-1.0)
                - temperature: Temperature used for synthesis
                - tokens_used: Number of tokens used
                - max_tokens: Token budget allocated
                - agents_synthesized: Number of agent outputs included
                - timestamp: Synthesis timestamp

        Raises:
            RuntimeError: If no LLM client available for synthesis
        """
        if not self.llm_client:
            raise RuntimeError("Synthesis requires LLM client")

        logger.info("=" * 60)
        logger.info("SYNTHESIS ENGINE STARTING")
        logger.info("=" * 60)

        # Gather recent agent messages AND system action results
        messages = self._get_recent_messages(
            limit=max_messages,
            message_types=[MessageType.STATUS_UPDATE, MessageType.SYSTEM_ACTION_RESULT]
        )

        if not messages:
            logger.warning("No agent messages available for synthesis")
            return {
                "synthesis_content": "No agent outputs available for synthesis.",
                "confidence": 0.0,
                "temperature": 0.0,
                "tokens_used": 0,
                "max_tokens": 0,
                "agents_synthesized": 0,
                "timestamp": time.time(),
                "error": "no_messages"
            }

        # Validate messages and calculate dynamic confidence
        confidences = []
        validation_scores = []
        critic_count = 0
        flagged_count = 0

        for msg in messages:
            if msg.message_type == MessageType.STATUS_UPDATE:
                conf = msg.content.get('confidence', 0.0)
                agent_type = msg.content.get('agent_type', 'unknown')

                if conf > 0:
                    confidences.append(conf)

                # Calculate validation score for this message
                try:
                    content_dict = {"result": msg.content.get('content', '')}
                    score = calculate_validation_score(
                        content=content_dict,
                        source_agent=msg.sender_id,
                        domain=msg.content.get('domain', 'workflow_task'),
                        confidence_level=msg.content.get('confidence_level', 'MEDIUM')
                    )

                    flags = get_validation_flags(
                        content=content_dict,
                        source_agent=msg.sender_id,
                        domain=msg.content.get('domain', 'workflow_task')
                    )

                    validation_scores.append(score)

                    if agent_type == 'critic':
                        critic_count += 1
                    if len(flags) > 0:
                        flagged_count += 1

                except Exception as e:
                    logger.warning(f"Validation failed for message from {msg.sender_id}: {e}")
                    # Fail-open: use neutral validation score
                    validation_scores.append(0.7)

        avg_agent_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        avg_validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else 0.5

        # Calculate dynamic synthesis confidence (weighted: 60% validation, 40% agent confidence)
        synthesis_confidence = avg_validation_score * 0.6 + avg_agent_confidence * 0.4

        # Boost for critic validation
        if critic_count >= 1:
            synthesis_confidence = min(1.0, synthesis_confidence * 1.1)

        # Penalty for validation issues
        if flagged_count > len(messages) * 0.3:  # More than 30% flagged
            synthesis_confidence *= 0.8

        # Calculate adaptive synthesis parameters
        temperature = self.calculate_synthesis_temperature(avg_agent_confidence)
        max_tokens = self.calculate_synthesis_tokens(len(messages), task_complexity)

        logger.info(f"Synthesis Parameters:")
        logger.info(f"  Task complexity: {task_complexity}")
        logger.info(f"  Agent messages: {len(messages)}")
        logger.info(f"  Average agent confidence: {avg_agent_confidence:.2f}")
        logger.info(f"  Average validation score: {avg_validation_score:.2f}")
        logger.info(f"  Dynamic synthesis confidence: {synthesis_confidence:.2f}")
        logger.info(f"  Critics present: {critic_count}")
        logger.info(f"  Flagged messages: {flagged_count}")
        logger.info(f"  Adaptive temperature: {temperature}")
        logger.info(f"  Adaptive token budget: {max_tokens}")

        # Build synthesis prompt
        user_prompt = self.build_synthesis_prompt(task_description, messages, task_complexity)

        # Truth-seeking system prompt
        system_prompt = """You are the Central Post of the Felix helical multi-agent system - a truth-seeking synthesis engine.

Felix agents operate along a helical geometry:
- Top of helix: Broad exploration (research agents)
- Middle spiral: Focused analysis (analysis agents)
- Bottom convergence: Critical validation (critic agents)

Your role is NOT to simply concatenate or summarize agent outputs. Your role is to REASON about them, VALIDATE them, and SYNTHESIZE TRUTH.

Your synthesis must:

1. **Validate Facts**: Cross-check claims between agents. If agents disagree, identify the disagreement and reason about which source is more authoritative or recent.

2. **Prioritize Critic Feedback**: CRITIC agents provide quality control. Their concerns must be explicitly addressed, not ignored. If critics identified issues, explain how you resolved them or acknowledge limitations.

3. **Acknowledge Contradictions**: If agent outputs contradict each other, DO NOT paper over the conflict. Explicitly note the contradiction and either:
   - Resolve it with reasoning (e.g., "Source A is more recent/authoritative")
   - Acknowledge uncertainty (e.g., "Conflicting information exists, confidence is low")

4. **Express Appropriate Uncertainty**: If validation is weak, sources are questionable, or agents disagree, EXPRESS THIS. Use phrases like:
   - "Based on available information, but with low confidence..."
   - "Sources disagree on this point..."
   - "This claim could not be verified..."

5. **Identify Gaps**: If important information is missing or agents didn't address key aspects of the query, say so.

6. **Reason Transparently**: Don't just state conclusions. Briefly explain WHY you believe certain facts over others (recency, authority, consensus, verification).

7. **Preserve Helical Insights**: Still integrate exploration (research), analysis, and validation (critic) phases - but critically, not blindly.

Your output should reflect JUSTIFIED confidence, not reflexive confidence. If the agent outputs are low-quality, your synthesis confidence should reflect that reality."""

        # Call LLM for synthesis
        start_time = time.time()
        try:
            llm_response = self.llm_client.complete(
                agent_id="synthesis_engine",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            synthesis_time = time.time() - start_time

            logger.info(f"✓ Synthesis complete in {synthesis_time:.2f}s")
            logger.info(f"  Tokens used: {llm_response.tokens_used} / {max_tokens}")
            logger.info(f"  Content length: {len(llm_response.content)} chars")
            logger.info("=" * 60)

            return {
                "synthesis_content": llm_response.content,
                "confidence": synthesis_confidence,  # Dynamic confidence based on validation
                "temperature": temperature,
                "tokens_used": llm_response.tokens_used,
                "max_tokens": max_tokens,
                "agents_synthesized": len(messages),
                "avg_agent_confidence": avg_agent_confidence,
                "avg_validation_score": avg_validation_score,
                "critic_count": critic_count,
                "flagged_count": flagged_count,
                "synthesis_time": synthesis_time,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"✗ Synthesis failed: {e}")
            raise

    def calculate_synthesis_temperature(self, avg_confidence: float) -> float:
        """
        Calculate adaptive temperature for synthesis based on agent confidence consensus.

        High confidence → focused synthesis (0.2)
        Medium confidence → balanced synthesis (0.3)
        Low confidence → creative integration (0.4)

        Args:
            avg_confidence: Average confidence from agent outputs (0.0-1.0)

        Returns:
            Temperature value (0.2-0.4)
        """
        if avg_confidence >= 0.9:
            return 0.2  # High confidence → very focused
        elif avg_confidence >= 0.75:
            return 0.3  # Medium confidence → balanced
        else:
            return 0.4  # Lower confidence → more creative integration

    def calculate_synthesis_tokens(self, agent_count: int, task_complexity: str = "COMPLEX") -> int:
        """
        Calculate adaptive token budget for synthesis based on number of agents and task complexity.

        More agents → more content to synthesize → larger budget
        Simpler tasks → less synthesis needed → smaller budget

        Args:
            agent_count: Number of agent outputs to synthesize
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Token budget (200-3000)
        """
        # Simple factual queries need minimal synthesis
        if task_complexity == "SIMPLE_FACTUAL":
            return 200  # Just answer the question directly

        # Medium complexity gets moderate token budget
        if task_complexity == "MEDIUM":
            return 800 if agent_count < 5 else 1200

        # Complex tasks get full token budget based on team size
        if agent_count >= 10:
            return 3000  # Many agents → comprehensive synthesis
        elif agent_count >= 5:
            return 2000  # Medium team → balanced synthesis
        else:
            return 1500  # Small team → focused synthesis

    def build_synthesis_prompt(self, task_description: str, messages: List[Message],
                                task_complexity: str = "COMPLEX") -> str:
        """
        Build synthesis prompt from task description and agent messages.

        Args:
            task_description: Original task description
            messages: List of agent messages to synthesize
            task_complexity: Task complexity ("SIMPLE_FACTUAL", "MEDIUM", or "COMPLEX")

        Returns:
            Formatted synthesis prompt
        """
        prompt_parts = [
            f"Original Task: {task_description}",
            "",
            "Agent Communications to Synthesize:",
            ""
        ]

        # Add each agent output and system action result with metadata
        for i, msg in enumerate(messages, 1):
            if msg.message_type == MessageType.STATUS_UPDATE:
                agent_type = msg.content.get('agent_type', 'unknown')
                content = msg.content.get('content', '')
                confidence = msg.content.get('confidence', 0.0)

                prompt_parts.append(
                    f"{i}. {agent_type.upper()} Agent (confidence: {confidence:.2f}):"
                )
                prompt_parts.append(content)
                prompt_parts.append("")

            elif msg.message_type == MessageType.SYSTEM_ACTION_RESULT:
                command = msg.content.get('command', '')
                stdout = msg.content.get('stdout', '')
                stderr = msg.content.get('stderr', '')
                success = msg.content.get('success', False)
                exit_code = msg.content.get('exit_code', -1)

                prompt_parts.append(f"{i}. SYSTEM COMMAND EXECUTION:")
                prompt_parts.append(f"   Command: {command}")
                prompt_parts.append(f"   Success: {success}")
                prompt_parts.append(f"   Exit Code: {exit_code}")
                if stdout:
                    prompt_parts.append(f"   Output: {stdout}")
                if stderr:
                    prompt_parts.append(f"   Errors: {stderr}")
                prompt_parts.append("")

        prompt_parts.append("---")
        prompt_parts.append("")

        # Add task-complexity-specific synthesis instructions
        if task_complexity == "SIMPLE_FACTUAL":
            prompt_parts.append("🎯 SIMPLE FACTUAL QUERY DETECTED")
            prompt_parts.append("")
            prompt_parts.append("This is a straightforward factual question. Your synthesis should:")
            prompt_parts.append("- Provide a DIRECT, CONCISE answer in 1-3 sentences")
            prompt_parts.append("- State the key fact or information clearly")
            prompt_parts.append("- NO philosophical analysis, NO elaborate discussion")
            prompt_parts.append("- NO exploration of implications or deeper meanings")
            prompt_parts.append("- Just answer the question directly")
            prompt_parts.append("")
            prompt_parts.append("Example format: \"The current date and time is [answer]. (Source: [if applicable])\"")
        elif task_complexity == "MEDIUM":
            prompt_parts.append("Create a focused synthesis (3-5 paragraphs) that directly addresses the question.")
            prompt_parts.append("Balance completeness with conciseness.")
        else:
            prompt_parts.append("Create a comprehensive final synthesis that integrates all agent findings above.")

        return "\n".join(prompt_parts)
