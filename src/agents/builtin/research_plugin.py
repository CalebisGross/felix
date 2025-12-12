"""
Research agent plugin for the Felix Framework.

This plugin provides the ResearchAgent, specialized in broad information
gathering and exploration tasks. Research agents spawn early in the workflow
(exploration phase) and focus on collecting diverse perspectives and sources.
"""

from typing import Optional, TYPE_CHECKING

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.specialized_agents import ResearchAgent

if TYPE_CHECKING:
    from src.core.helix_geometry import HelixGeometry
    from src.llm.lm_studio_client import LMStudioClient
    from src.agents.llm_agent import LLMAgent
    from src.llm.token_budget import TokenBudgetManager


class ResearchAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for Felix's built-in Research Agent.

    Research agents are specialized in:
    - Broad information gathering
    - Exploring multiple perspectives
    - Web search integration
    - Early-phase exploration

    They spawn in the exploration phase (0.0-0.3) when agents are at the
    top of the helix with maximum radius and creativity.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the research agent."""
        return AgentMetadata(
            agent_type="research",
            display_name="Research Agent",
            description="Specialized in broad information gathering and exploration",
            spawn_range=(0.0, 0.3),  # Exploration phase
            capabilities=[
                "web_search",
                "information_gathering",
                "perspective_generation",
                "source_discovery"
            ],
            tags=["exploration", "research", "information"],
            default_tokens=800,
            version="1.0.0",
            author="Felix Framework",
            priority=10  # High priority for research tasks
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: "HelixGeometry",
                    llm_client: "LMStudioClient",
                    token_budget_manager: Optional["TokenBudgetManager"] = None,
                    **kwargs) -> "LLMAgent":
        """
        Create a ResearchAgent instance.

        Additional kwargs supported:
            - research_domain: Domain focus (default: "general")
            - max_tokens: Maximum tokens (default: from metadata)
            - web_search_client: Web search client (legacy, handled by CentralPost)
            - max_web_queries: Max web queries (legacy, handled by CentralPost)
            - prompt_manager: Optional prompt manager
        """
        return ResearchAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            research_domain=kwargs.get('research_domain', 'general'),
            token_budget_manager=token_budget_manager,
            max_tokens=kwargs.get('max_tokens', self.get_metadata().default_tokens),
            web_search_client=kwargs.get('web_search_client'),  # Legacy
            max_web_queries=kwargs.get('max_web_queries', 3),  # Legacy
            prompt_manager=kwargs.get('prompt_manager')
        )

    def supports_task(self, task_description: str, task_metadata: dict) -> bool:
        """
        Research agents support most tasks, especially those requiring:
        - Information gathering
        - Web search
        - Multiple perspectives
        """
        # Keywords that indicate research needs
        research_keywords = [
            'research', 'find', 'search', 'gather', 'collect',
            'explore', 'investigate', 'discover', 'what is', 'who is',
            'latest', 'current', 'recent', 'information about'
        ]

        task_lower = task_description.lower()
        has_research_keywords = any(kw in task_lower for kw in research_keywords)

        # Research agents are generally useful for all complexities
        return True  # Support all tasks by default

    def get_spawn_ranges_by_complexity(self) -> dict:
        """
        Get spawn ranges based on task complexity.

        For complex tasks, research agents spawn earlier to gather
        comprehensive information before other agents begin processing.
        """
        return {
            "simple": (0.05, 0.25),   # Later spawn for simple tasks
            "medium": (0.02, 0.25),   # Standard spawn range
            "complex": (0.0, 0.20)    # Earlier spawn for complex tasks
        }
