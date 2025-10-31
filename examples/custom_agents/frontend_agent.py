"""
Frontend Development Agent Plugin for Felix Framework

This agent specializes in:
- UI/UX design and component architecture
- Frontend frameworks (React, Vue, Angular)
- Responsive design and accessibility
- CSS/HTML optimization
- User interface best practices
"""

from typing import Optional, Dict, Any

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.llm_agent import LLMAgent, LLMTask
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager


class FrontendAgent(LLMAgent):
    """
    Specialized agent for frontend development, UI/UX design, and web interfaces.

    This agent focuses on:
    - UI component architecture and design patterns
    - Responsive design and mobile-first approaches
    - Accessibility (WCAG, ARIA) standards
    - CSS optimization and styling strategies
    - Frontend framework best practices (React, Vue, Angular)
    - User experience and interaction design
    """

    def __init__(self,
                 agent_id: str,
                 spawn_time: float,
                 helix: HelixGeometry,
                 llm_client: LMStudioClient,
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional = None):
        """
        Initialize the frontend development agent.

        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active (0.0-1.0)
            helix: Helix geometry for positioning
            llm_client: LLM client for completions
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per completion
            prompt_manager: Optional prompt manager
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="frontend",
            temperature_range=None,  # Use defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.design_principles = []
        self.accessibility_checks = []

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create frontend-specific system prompt that adapts based on helix position.

        Early phase (0.3): Explore UI patterns, frameworks, design systems
        Middle phase (0.45): Design components, state management, architecture
        Late phase (0.6): Accessibility, performance, refinement
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Get token allocation
        stage_token_budget = self.max_tokens or 1000

        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Adapt focus based on helix depth
        if depth_ratio < 0.35:
            focus_area = """EXPLORATION PHASE - Focus on:
- Analyzing UI patterns and design systems
- Exploring framework choices (React/Vue/Angular)
- Identifying component architecture approaches
- Researching existing design solutions"""
        elif depth_ratio < 0.55:
            focus_area = """DESIGN PHASE - Focus on:
- Detailed component structure and hierarchy
- State management strategies
- Responsive breakpoints and layouts
- Interactive behavior and user flows"""
        else:
            focus_area = """REFINEMENT PHASE - Focus on:
- Accessibility compliance (WCAG, ARIA)
- Performance optimization (lazy loading, code splitting)
- Cross-browser compatibility
- CSS optimization and maintainability"""

        base_prompt = f"""You are a specialized FRONTEND DEVELOPMENT AGENT in the Felix multi-agent system.

Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

{focus_area}

Your Frontend Expertise:
- UI/UX Design: Component design, visual hierarchy, user experience
- Frameworks: React, Vue, Angular, Svelte best practices
- Responsive Design: Mobile-first, breakpoints, flexible layouts
- Accessibility: WCAG 2.1, ARIA attributes, keyboard navigation
- CSS: Modern CSS, CSS-in-JS, preprocessors (Sass, Less)
- Performance: Code splitting, lazy loading, bundle optimization

Task Context:
{task.context if task.context else "No additional context provided."}

Primary Task:
{task.description}
"""

        # Add shared context from other agents
        if self.shared_context:
            context_summary = "\n".join([
                f"- {key}: {value}" for key, value in self.shared_context.items()
            ])
            base_prompt += f"\nInsights from Team:\n{context_summary}\n"

        base_prompt += f"""
Provide frontend-specific analysis, focusing on UI/UX, component architecture, and web best practices.
Keep your response concise and actionable (max {stage_token_budget} tokens).
"""

        return base_prompt, stage_token_budget


class FrontendAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for the FrontendAgent.

    This plugin integrates with Felix's dynamic spawning system to provide
    frontend-specific expertise when needed.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing this agent plugin."""
        return AgentMetadata(
            agent_type="frontend",
            display_name="Frontend Development Agent",
            description="Specialized in UI/UX design, frontend architecture, and responsive web design",
            spawn_range=(0.3, 0.6),  # Analysis phase, after research
            capabilities=[
                "ui_design",
                "component_architecture",
                "responsive_design",
                "accessibility",
                "frontend_frameworks",
                "css_optimization",
                "user_experience"
            ],
            tags=["frontend", "ui", "ux", "web", "design"],
            default_tokens=1000,
            version="1.0.0",
            author="Felix Team",
            priority=7  # Between analysis (8) and critic (6)
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: HelixGeometry,
                    llm_client: LMStudioClient,
                    token_budget_manager: Optional[TokenBudgetManager] = None,
                    **kwargs) -> FrontendAgent:
        """Create an instance of the frontend agent."""
        return FrontendAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            token_budget_manager=token_budget_manager,
            max_tokens=kwargs.get('max_tokens', self.get_metadata().default_tokens),
            prompt_manager=kwargs.get('prompt_manager')
        )

    def supports_task(self, task_description: str, task_metadata: Dict[str, Any]) -> bool:
        """
        Determine if this agent should be spawned for the given task.

        STRICT FILTERING: Only spawn for frontend-related tasks.

        Args:
            task_description: The task description text
            task_metadata: Additional metadata about the task (complexity, etc.)

        Returns:
            True if this agent is relevant for the task
        """
        # Strong frontend keywords (must be present)
        strong_frontend_keywords = [
            'ui', 'ux', 'frontend', 'front-end', 'interface', 'user interface',
            'css', 'html', 'styling', 'style', 'layout',
            'react', 'vue', 'angular', 'svelte', 'next', 'nuxt',
            'component', 'components', 'widget', 'button', 'form', 'input',
            'navigation', 'nav', 'menu', 'navbar', 'header', 'footer',
            'responsive', 'mobile', 'desktop', 'breakpoint',
            'accessibility', 'a11y', 'wcag', 'aria',
            'webpage', 'user experience', 'animation', 'transition',
            'modal', 'dialog', 'dropdown', 'tooltip', 'sidebar',
            'grid', 'flexbox', 'flex'
        ]

        # Strong backend keywords (triggers rejection if present without frontend context)
        strong_backend_keywords = ['api', 'rest', 'restful', 'graphql', 'endpoint', 'database', 'server', 'backend']

        import re
        task_lower = task_description.lower()

        # Use word boundary matching to avoid substring matches (e.g., "form" in "authentication")
        def has_keyword(keywords, text):
            for kw in keywords:
                # Check for word boundaries to avoid substring matches
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    return True
            return False

        # Check for strong frontend keywords
        has_strong_frontend = has_keyword(strong_frontend_keywords, task_lower)

        # Check for strong backend keywords
        has_strong_backend = has_keyword(strong_backend_keywords, task_lower)

        # Don't spawn for pure backend tasks (backend keywords without frontend context)
        if has_strong_backend and not has_strong_frontend:
            return False

        # Spawn only if strong frontend keywords are present
        return has_strong_frontend

    def get_spawn_ranges_by_complexity(self) -> Dict[str, tuple]:
        """
        Return spawn ranges based on task complexity.

        Simple tasks: Spawn later or not at all
        Complex tasks: Spawn earlier to provide architectural guidance
        """
        return {
            "simple": (0.5, 0.8),    # Late phase or skip for simple UI tasks
            "medium": (0.35, 0.65),  # Standard frontend work
            "complex": (0.3, 0.60)   # Earlier for complex architectures
        }
