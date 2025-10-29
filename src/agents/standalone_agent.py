"""
Standalone Agent for Non-Workflow Tasks

Provides simplified agent interface for tasks that don't fit the helix progression
model, such as Knowledge Brain document comprehension.

Unlike LLMAgent (designed for workflow tasks with helix positioning), StandaloneAgent
provides a simple process_task(context) interface without spawn times or helix geometry.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StandaloneAgent:
    """
    Simplified agent for non-workflow tasks.

    Designed for autonomous systems like Knowledge Brain that need agent-like
    behavior (research, analysis, criticism) but don't operate within the
    Felix helix progression model.

    Key differences from LLMAgent:
    - No helix geometry or spawn time concepts
    - Simple process_task(context) interface
    - Mode-based behavior (research/analysis/critic)
    - Independent of workflow machinery

    Usage:
        research_agent = StandaloneAgent(
            agent_id="research_comprehension",
            mode="research",
            llm_client=llm_client,
            temperature=0.7,
            max_tokens=1500
        )

        response = research_agent.process_task(
            "Summarize this document: [content]"
        )
    """

    def __init__(self,
                 agent_id: str,
                 mode: str,
                 llm_client,
                 temperature: float = 0.7,
                 max_tokens: int = 1500):
        """
        Initialize standalone agent.

        Args:
            agent_id: Unique identifier for this agent
            mode: Agent mode - "research", "analysis", or "critic"
            llm_client: LLM client for generating responses
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.agent_id = agent_id
        self.mode = mode
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Validate mode
        valid_modes = ["research", "analysis", "critic"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        logger.debug(f"StandaloneAgent initialized: {agent_id} (mode={mode})")

    def process_task(self, context: str) -> str:
        """
        Process a task with the LLM.

        Args:
            context: Task description/prompt

        Returns:
            LLM response content as string

        Raises:
            Exception: If LLM call fails
        """
        system_prompt = self._build_system_prompt()

        try:
            response = self.llm_client.complete(
                agent_id=self.agent_id,
                system_prompt=system_prompt,
                user_prompt=context,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.content

        except Exception as e:
            logger.error(f"StandaloneAgent {self.agent_id} failed: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """
        Build system prompt based on agent mode.

        Returns:
            System prompt string
        """
        if self.mode == "research":
            return """You are a research agent that reads and comprehends documents.

Your role:
- Read document excerpts carefully
- Identify the main topics and themes
- Extract key information and facts
- Provide clear, structured summaries
- Note important details that might be referenced later

Be thorough but concise. Focus on understanding content deeply."""

        elif self.mode == "analysis":
            return """You are an analysis agent that extracts structured knowledge from text.

Your role:
- Identify key concepts and their definitions
- Extract entities (people, places, organizations, technical terms)
- Note relationships between concepts
- Categorize information by type
- Structure findings clearly

Output should be systematic and well-organized. Focus on extracting actionable knowledge."""

        elif self.mode == "critic":
            return """You are a critic agent that validates knowledge extraction quality.

Your role:
- Evaluate completeness of extracted information
- Check for logical consistency
- Identify missing or ambiguous details
- Assess confidence in extracted knowledge
- Provide quality scores and feedback

Be objective and constructive. Focus on ensuring high-quality knowledge."""

        else:
            # Fallback (should never reach here due to __init__ validation)
            return "You are an AI assistant that helps process tasks."

    def __repr__(self) -> str:
        return f"StandaloneAgent(id={self.agent_id}, mode={self.mode})"
