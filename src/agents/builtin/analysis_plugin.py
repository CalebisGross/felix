"""
Analysis agent plugin for the Felix Framework.

This plugin provides the AnalysisAgent, specialized in processing and
organizing information gathered by research agents. Analysis agents spawn
in the middle phase and focus on pattern identification and insight extraction.
"""

from typing import Optional, TYPE_CHECKING

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.specialized_agents import AnalysisAgent

if TYPE_CHECKING:
    from src.core.helix_geometry import HelixGeometry
    from src.llm.lm_studio_client import LMStudioClient
    from src.agents.llm_agent import LLMAgent
    from src.llm.token_budget import TokenBudgetManager


class AnalysisAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for Felix's built-in Analysis Agent.

    Analysis agents are specialized in:
    - Processing information from research agents
    - Identifying patterns and themes
    - Organizing findings into structured insights
    - Looking for contradictions and gaps

    They spawn in the analysis phase (0.2-0.7) when agents are converging
    toward synthesis, balancing creativity with logical processing.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the analysis agent."""
        return AgentMetadata(
            agent_type="analysis",
            display_name="Analysis Agent",
            description="Specialized in processing and organizing information",
            spawn_range=(0.2, 0.7),  # Analysis phase
            capabilities=[
                "pattern_identification",
                "information_organization",
                "insight_extraction",
                "gap_detection",
                "contradiction_analysis"
            ],
            tags=["analysis", "processing", "organization"],
            default_tokens=800,
            version="1.0.0",
            author="Felix Framework",
            priority=8  # Medium-high priority
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: "HelixGeometry",
                    llm_client: "LMStudioClient",
                    token_budget_manager: Optional["TokenBudgetManager"] = None,
                    **kwargs) -> "LLMAgent":
        """
        Create an AnalysisAgent instance.

        Additional kwargs supported:
            - analysis_type: Analysis specialization (default: "general")
            - max_tokens: Maximum tokens (default: from metadata)
            - prompt_manager: Optional prompt manager
        """
        return AnalysisAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            analysis_type=kwargs.get('analysis_type', 'general'),
            token_budget_manager=token_budget_manager,
            max_tokens=kwargs.get('max_tokens', self.get_metadata().default_tokens),
            prompt_manager=kwargs.get('prompt_manager')
        )

    def supports_task(self, task_description: str, task_metadata: dict) -> bool:
        """
        Analysis agents support tasks requiring:
        - Pattern identification
        - Organization
        - Processing of information
        """
        # Keywords that indicate analysis needs
        analysis_keywords = [
            'analyze', 'analysis', 'compare', 'contrast', 'organize',
            'pattern', 'identify', 'process', 'structure', 'categorize',
            'evaluate', 'assess', 'examine', 'relationships'
        ]

        task_lower = task_description.lower()
        has_analysis_keywords = any(kw in task_lower for kw in analysis_keywords)

        # Analysis agents are useful for medium and complex tasks
        complexity = task_metadata.get('complexity', 'medium')
        return complexity in ['medium', 'complex'] or has_analysis_keywords

    def get_spawn_ranges_by_complexity(self) -> dict:
        """
        Get spawn ranges based on task complexity.

        Analysis agents spawn after research agents to process gathered information.
        """
        return {
            "simple": (0.3, 0.7),     # Later spawn for simple tasks
            "medium": (0.25, 0.65),   # Standard spawn range
            "complex": (0.2, 0.60)    # Slightly earlier for complex tasks
        }
