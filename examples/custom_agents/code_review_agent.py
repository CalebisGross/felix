"""
Example custom agent plugin: CodeReviewAgent

This file demonstrates how to create a custom agent plugin for Felix.
Place this file in a directory and register it with Felix using:

    registry = AgentPluginRegistry()
    registry.add_plugin_directory("./examples/custom_agents")

Your custom agent will then be available for spawning alongside built-in agents.
"""

from typing import Optional

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata
)
from src.agents.llm_agent import LLMAgent, LLMTask
from src.core.helix_geometry import HelixGeometry
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager


class CodeReviewAgent(LLMAgent):
    """
    Custom agent specialized in code review.

    This agent focuses on:
    - Identifying bugs and potential issues
    - Checking code style and best practices
    - Suggesting improvements and refactorings
    - Security vulnerability detection
    """

    def __init__(self,
                 agent_id: str,
                 spawn_time: float,
                 helix: HelixGeometry,
                 llm_client: LMStudioClient,
                 review_style: str = "thorough",
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 max_tokens: Optional[int] = None,
                 prompt_manager: Optional = None):
        """
        Initialize the code review agent.

        Args:
            agent_id: Unique identifier
            spawn_time: When agent becomes active (0.0-1.0)
            helix: Helix geometry for positioning
            llm_client: LLM client for completions
            review_style: Review style ("quick", "thorough", "security-focused")
            token_budget_manager: Optional token budget manager
            max_tokens: Maximum tokens per completion
            prompt_manager: Optional prompt manager
        """
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="code_review",
            temperature_range=None,  # Use defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.review_style = review_style
        self.issues_found = []
        self.suggestions = []

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """Create code review-specific system prompt."""
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Get token allocation
        stage_token_budget = self.max_tokens or 1000

        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        base_prompt = f"""You are a specialized CODE REVIEW AGENT in the Felix multi-agent system.

Review Style: {self.review_style}
Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

Your Code Review Approach:
- Identify bugs, errors, and potential issues
- Check code style and best practices
- Suggest improvements and refactorings
- Look for security vulnerabilities
- Evaluate code maintainability

Review Focus Based on Style:
"""

        if self.review_style == "quick":
            base_prompt += """
- QUICK SCAN: Focus on obvious bugs and critical issues
- Flag syntax errors and common mistakes
- Provide concise feedback
"""
        elif self.review_style == "security-focused":
            base_prompt += """
- SECURITY ANALYSIS: Identify vulnerabilities
- Check for SQL injection, XSS, insecure dependencies
- Validate input handling and authentication
- Review cryptographic implementations
"""
        else:  # thorough
            base_prompt += """
- COMPREHENSIVE REVIEW: Deep analysis of code quality
- Evaluate architecture and design patterns
- Check error handling and edge cases
- Review documentation and naming conventions
- Suggest performance optimizations
"""

        # Add shared context from other agents
        if self.shared_context:
            base_prompt += "\n\nContext from Other Agents:\n"
            for key, value in self.shared_context.items():
                if len(str(value)) > 200:
                    value = str(value)[:200] + "..."
                base_prompt += f"- {key}: {value}\n"

        base_prompt += f"""
Task Context: {task.context}

Provide specific, actionable feedback with:
1. Issue severity (LOW, MEDIUM, HIGH, CRITICAL)
2. Line numbers or code snippets
3. Explanation of the issue
4. Suggested fix or improvement
"""

        return base_prompt, stage_token_budget


class CodeReviewAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for CodeReviewAgent.

    This plugin registers the CodeReviewAgent with Felix's agent system,
    making it available for spawning alongside built-in agents.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing the code review agent."""
        return AgentMetadata(
            agent_type="code_review",
            display_name="Code Review Agent",
            description="Specialized in reviewing code for bugs, style, and security",
            spawn_range=(0.3, 0.7),  # Analysis phase
            capabilities=[
                "code_analysis",
                "bug_detection",
                "style_checking",
                "security_analysis",
                "refactoring_suggestions"
            ],
            tags=["engineering", "quality", "security"],
            default_tokens=1000,
            version="1.0.0",
            author="Example Custom Plugin",
            priority=7  # Medium-high priority for code tasks
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: HelixGeometry,
                    llm_client: LMStudioClient,
                    token_budget_manager: Optional[TokenBudgetManager] = None,
                    **kwargs) -> LLMAgent:
        """
        Create a CodeReviewAgent instance.

        Additional kwargs:
            - review_style: "quick", "thorough", or "security-focused"
            - max_tokens: Maximum tokens
            - prompt_manager: Optional prompt manager
        """
        return CodeReviewAgent(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            review_style=kwargs.get('review_style', 'thorough'),
            token_budget_manager=token_budget_manager,
            max_tokens=kwargs.get('max_tokens', self.get_metadata().default_tokens),
            prompt_manager=kwargs.get('prompt_manager')
        )

    def supports_task(self, task_description: str, task_metadata: dict) -> bool:
        """
        CodeReviewAgent supports code-related tasks.

        Looks for keywords like: code, function, class, bug, review, etc.
        """
        code_keywords = [
            'code', 'function', 'class', 'method', 'bug', 'review',
            'refactor', 'optimize', 'python', 'javascript', 'java',
            'programming', 'software', 'implementation'
        ]

        task_lower = task_description.lower()
        return any(kw in task_lower for kw in code_keywords)

    def get_spawn_ranges_by_complexity(self) -> dict:
        """
        Code review agents spawn in analysis phase.

        For complex code tasks, spawn earlier to provide continuous review.
        """
        return {
            "simple": (0.4, 0.7),
            "medium": (0.3, 0.65),
            "complex": (0.25, 0.60)
        }
