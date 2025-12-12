"""
Critic agent plugin for the Felix Framework.

This plugin provides the CriticAgent, specialized in quality assurance
and review. Critic agents spawn later in the workflow to evaluate work
from other agents and ensure quality standards.
"""

from typing import Optional, TYPE_CHECKING

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.specialized_agents import CriticAgent

if TYPE_CHECKING:
    from src.core.helix_geometry import HelixGeometry
    from src.llm.lm_studio_client import LMStudioClient
    from src.agents.llm_agent import LLMAgent
    from src.llm.token_budget import TokenBudgetManager


class CriticAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for Felix's built-in Critic Agent.

    Critic agents are specialized in:
    - Quality assurance and review
    - Identifying gaps and errors
    - Suggesting improvements
    - Ensuring completeness and accuracy

    They spawn in the later phases (0.4-0.8) to review work from
    research and analysis agents before final synthesis.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the critic agent."""
        return AgentMetadata(
            agent_type="critic",
            display_name="Critic Agent",
            description="Specialized in quality assurance and review",
            spawn_range=(0.4, 0.8),  # Late phase for review
            capabilities=[
                "quality_assurance",
                "error_detection",
                "gap_identification",
                "improvement_suggestions",
                "accuracy_verification",
                "completeness_check"
            ],
            tags=["review", "quality", "validation"],
            default_tokens=800,
            version="1.0.0",
            author="Felix Framework",
            priority=6  # Medium priority (spawn after research/analysis)
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: "HelixGeometry",
                    llm_client: "LMStudioClient",
                    token_budget_manager: Optional["TokenBudgetManager"] = None,
                    **kwargs) -> "LLMAgent":
        """
        Create a CriticAgent instance.

        Additional kwargs supported:
            - review_focus: Review focus (default: "general")
            - max_tokens: Maximum tokens (default: from metadata)
            - prompt_manager: Optional prompt manager
        """
        return CriticAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            review_focus=kwargs.get('review_focus', 'general'),
            token_budget_manager=token_budget_manager,
            max_tokens=kwargs.get('max_tokens', self.get_metadata().default_tokens),
            prompt_manager=kwargs.get('prompt_manager')
        )

    def supports_task(self, task_description: str, task_metadata: dict) -> bool:
        """
        Critic agents support tasks requiring:
        - Quality review
        - Error detection
        - Validation
        """
        # Keywords that indicate critic needs
        critic_keywords = [
            'review', 'critique', 'validate', 'verify', 'check',
            'quality', 'error', 'bug', 'issue', 'problem',
            'improve', 'correction', 'accuracy', 'completeness'
        ]

        task_lower = task_description.lower()
        has_critic_keywords = any(kw in task_lower for kw in critic_keywords)

        # Critic agents are most useful for medium and complex tasks
        complexity = task_metadata.get('complexity', 'medium')
        is_suitable_complexity = complexity in ['medium', 'complex']

        return has_critic_keywords or is_suitable_complexity

    def get_spawn_ranges_by_complexity(self) -> dict:
        """
        Get spawn ranges based on task complexity.

        Critic agents spawn after research and analysis to review their work.
        """
        return {
            "simple": (0.5, 0.8),     # May not spawn for simple tasks
            "medium": (0.45, 0.75),   # Standard spawn range
            "complex": (0.4, 0.70)    # Earlier spawn for complex tasks
        }
