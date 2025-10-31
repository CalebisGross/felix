"""
QA & Testing Agent Plugin for Felix Framework

This agent specializes in:
- Test strategy and planning
- Quality assurance and validation
- Test case generation
- Coverage analysis and edge case identification
- Acceptance criteria definition
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


class QAAgent(LLMAgent):
    """
    Specialized agent for quality assurance, testing, and validation.

    This agent focuses on:
    - Test strategy development (unit, integration, e2e)
    - Test case generation and edge case identification
    - Coverage analysis and gap detection
    - Acceptance criteria definition
    - Quality metrics and validation planning
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
        Initialize the QA and testing agent.

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
            agent_type="qa",
            temperature_range=None,  # Use defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.test_cases = []
        self.edge_cases = []
        self.quality_metrics = {}

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create QA-specific system prompt that adapts based on helix position.

        Early phase (0.5): Review design for testability, identify test needs
        Middle phase (0.65): Generate test cases, edge cases, coverage analysis
        Late phase (0.8): Final quality metrics, acceptance criteria, validation plans
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # Get token allocation
        stage_token_budget = self.max_tokens or 800

        if self.token_budget_manager:
            token_allocation = self.token_budget_manager.calculate_stage_allocation(
                self.agent_id, depth_ratio, self.processing_stage + 1
            )
            stage_token_budget = token_allocation.stage_budget

        # Adapt focus based on helix depth
        if depth_ratio < 0.60:
            focus_area = """TESTABILITY REVIEW PHASE - Focus on:
- Analyzing proposed designs for testability
- Identifying areas that need test coverage
- Suggesting test-friendly architecture patterns
- Planning test strategy (unit, integration, e2e)"""
        elif depth_ratio < 0.75:
            focus_area = """TEST GENERATION PHASE - Focus on:
- Creating comprehensive test cases
- Identifying edge cases and boundary conditions
- Coverage gap analysis
- Error scenario identification"""
        else:
            focus_area = """VALIDATION PHASE - Focus on:
- Final acceptance criteria definition
- Quality metrics and KPIs
- Test execution planning
- Regression test identification"""

        base_prompt = f"""You are a specialized QA & TESTING AGENT in the Felix multi-agent system.

Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

{focus_area}

Your QA Expertise:
- Test Strategy: Unit testing, integration testing, E2E testing, acceptance testing
- Test Design: Test cases, test scenarios, edge cases, boundary conditions
- Coverage Analysis: Code coverage, branch coverage, path coverage
- Quality Metrics: Defect density, test effectiveness, cyclomatic complexity
- Testing Tools: Jest, Mocha, Pytest, Selenium, Cypress, JUnit
- Methodologies: TDD, BDD, exploratory testing, regression testing

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
Provide QA-specific analysis, focusing on test strategy, test cases, and quality validation.
Keep your response concise and actionable (max {stage_token_budget} tokens).
"""

        return base_prompt, stage_token_budget


class QAAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for the QAAgent.

    This plugin integrates with Felix's dynamic spawning system to provide
    QA and testing expertise when needed.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing this agent plugin."""
        return AgentMetadata(
            agent_type="qa",
            display_name="QA & Testing Agent",
            description="Specialized in test strategy, quality assurance, and validation planning",
            spawn_range=(0.5, 0.8),  # Late phase, after implementation discussion
            capabilities=[
                "test_strategy",
                "quality_assurance",
                "test_generation",
                "coverage_analysis",
                "edge_case_identification",
                "validation_planning",
                "acceptance_criteria"
            ],
            tags=["qa", "testing", "quality", "validation"],
            default_tokens=800,
            version="1.0.0",
            author="Felix Team",
            priority=5  # Lower than critic (6), spawns after main agents
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: HelixGeometry,
                    llm_client: LMStudioClient,
                    token_budget_manager: Optional[TokenBudgetManager] = None,
                    **kwargs) -> QAAgent:
        """Create an instance of the QA agent."""
        return QAAgent(
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

        STRICT FILTERING: Spawn for testing-related tasks OR medium/complex development tasks.

        Args:
            task_description: The task description text
            task_metadata: Additional metadata about the task (complexity, etc.)

        Returns:
            True if this agent is relevant for the task
        """
        # Strong QA keywords (testing-focused)
        strong_qa_keywords = [
            'test', 'testing', 'tests',
            'qa', 'quality assurance', 'quality control',
            'validation', 'validate', 'verify', 'verification',
            'coverage', 'test coverage', 'code coverage',
            'unit test', 'integration test', 'e2e', 'end-to-end',
            'test case', 'test cases', 'test plan', 'test strategy',
            'acceptance criteria', 'acceptance test',
            'regression', 'regression test',
            'automated test', 'test automation',
            'edge case', 'edge cases', 'boundary condition',
            'mock', 'mocking', 'stub', 'fixture', 'test suite'
        ]

        # Weak QA keywords (only trigger for non-simple complexity)
        weak_qa_keywords = ['bug', 'bugs', 'issue', 'issues', 'defect', 'defects']

        # Development keywords (triggers auto-spawn for medium/complex)
        development_keywords = [
            'develop', 'build', 'create', 'implement',
            'api', 'endpoint', 'service', 'application', 'app', 'system'
        ]

        import re
        task_lower = task_description.lower()
        complexity = task_metadata.get('complexity', 'medium')

        # Use word boundary matching to avoid substring matches
        def has_keyword(keywords, text):
            for kw in keywords:
                # Check for word boundaries to avoid substring matches
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    return True
            return False

        # Check for strong QA keywords
        has_strong_qa = has_keyword(strong_qa_keywords, task_lower)

        # Check for weak QA keywords (only count for medium/complex)
        has_weak_qa = has_keyword(weak_qa_keywords, task_lower)

        # Check if it's a development task
        is_development = has_keyword(development_keywords, task_lower)

        # Spawn if:
        # 1. Task has strong QA keywords (explicit testing), OR
        # 2. Task has weak QA keywords AND complexity is not simple, OR
        # 3. Task is a development task AND complexity is medium/complex
        if has_strong_qa:
            return True

        if has_weak_qa and complexity != 'simple':
            return True

        if is_development and complexity in ['medium', 'complex']:
            return True

        return False

    def get_spawn_ranges_by_complexity(self) -> Dict[str, tuple]:
        """
        Return spawn ranges based on task complexity.

        Simple tasks: Spawn very late or not at all
        Complex tasks: Spawn earlier to influence testability
        """
        return {
            "simple": (0.7, 0.9),    # Very late or skip for simple tasks
            "medium": (0.55, 0.80),  # Standard QA work
            "complex": (0.5, 0.75)   # Earlier for comprehensive test suites
        }
