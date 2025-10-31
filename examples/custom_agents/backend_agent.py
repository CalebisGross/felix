"""
Backend Development Agent Plugin for Felix Framework

This agent specializes in:
- API design and implementation (REST, GraphQL)
- Database architecture and data modeling
- Server-side logic and microservices
- Authentication and authorization
- Performance optimization and scalability
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


class BackendAgent(LLMAgent):
    """
    Specialized agent for backend development, API design, and server architecture.

    This agent focuses on:
    - REST API and GraphQL endpoint design
    - Database schema and data model optimization
    - Authentication and authorization strategies
    - Microservices architecture and service communication
    - Server-side business logic
    - Performance optimization and caching strategies
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
        Initialize the backend development agent.

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
            agent_type="backend",
            temperature_range=None,  # Use defaults
            max_tokens=max_tokens,
            token_budget_manager=token_budget_manager,
            prompt_manager=prompt_manager
        )

        self.architecture_patterns = []
        self.security_considerations = []

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create backend-specific system prompt that adapts based on helix position.

        Early phase (0.3): Explore architecture patterns, tech choices, scalability
        Middle phase (0.45): Design APIs, data models, service boundaries
        Late phase (0.6): Performance optimization, security hardening, deployment
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
- Analyzing architecture patterns (monolith vs microservices)
- Exploring technology stack choices
- Identifying scalability requirements
- Researching existing API patterns and standards"""
        elif depth_ratio < 0.55:
            focus_area = """DESIGN PHASE - Focus on:
- Detailed API endpoint design (REST/GraphQL)
- Database schema and relationship modeling
- Authentication/authorization strategies
- Service boundaries and communication protocols"""
        else:
            focus_area = """OPTIMIZATION PHASE - Focus on:
- Performance optimization (caching, indexing, query optimization)
- Security hardening (input validation, encryption, rate limiting)
- Deployment strategies (CI/CD, containerization)
- Monitoring and observability implementation"""

        base_prompt = f"""You are a specialized BACKEND DEVELOPMENT AGENT in the Felix multi-agent system.

Current Position: Depth {depth_ratio:.2f}/1.0 on the helix

{focus_area}

Your Backend Expertise:
- API Design: RESTful APIs, GraphQL, WebSockets, API versioning
- Databases: SQL (PostgreSQL, MySQL), NoSQL (MongoDB, Redis), ORM patterns
- Authentication: JWT, OAuth 2.0, session management, API keys
- Architecture: Microservices, monoliths, event-driven, serverless
- Performance: Caching strategies, query optimization, load balancing
- Security: Input validation, SQL injection prevention, encryption, CORS

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
Provide backend-specific analysis, focusing on API design, data modeling, and scalable architecture.
Keep your response concise and actionable (max {stage_token_budget} tokens).
"""

        return base_prompt, stage_token_budget


class BackendAgentPlugin(SpecializedAgentPlugin):
    """
    Plugin wrapper for the BackendAgent.

    This plugin integrates with Felix's dynamic spawning system to provide
    backend-specific expertise when needed.
    """

    def get_metadata(self) -> AgentMetadata:
        """Return metadata describing this agent plugin."""
        return AgentMetadata(
            agent_type="backend",
            display_name="Backend Development Agent",
            description="Specialized in API design, database architecture, and server-side logic",
            spawn_range=(0.3, 0.6),  # Parallel with frontend, analysis phase
            capabilities=[
                "api_design",
                "database_design",
                "server_architecture",
                "authentication",
                "authorization",
                "performance_optimization",
                "microservices",
                "data_modeling"
            ],
            tags=["backend", "api", "database", "server", "architecture"],
            default_tokens=1000,
            version="1.0.0",
            author="Felix Team",
            priority=7  # Equal with frontend
        )

    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: HelixGeometry,
                    llm_client: LMStudioClient,
                    token_budget_manager: Optional[TokenBudgetManager] = None,
                    **kwargs) -> BackendAgent:
        """Create an instance of the backend agent."""
        return BackendAgent(
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

        STRICT FILTERING: Only spawn for backend-related tasks.

        Args:
            task_description: The task description text
            task_metadata: Additional metadata about the task (complexity, etc.)

        Returns:
            True if this agent is relevant for the task
        """
        # Strong backend keywords (must be present)
        strong_backend_keywords = [
            'api', 'rest', 'restful', 'graphql', 'endpoint', 'endpoints',
            'backend', 'back-end', 'server', 'server-side',
            'database', 'db', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'data model', 'schema', 'table', 'query', 'queries',
            'microservice', 'microservices',
            'authentication', 'auth', 'authorization', 'jwt', 'oauth',
            'middleware', 'route', 'routes', 'routing', 'controller', 'controllers',
            'crud',
            'migration', 'migrations', 'orm', 'sequelize', 'prisma',
            'websocket', 'socket',
            'serverless', 'lambda'
        ]

        # Strong frontend keywords (triggers rejection if present without backend context)
        strong_frontend_keywords = ['css', 'styling', 'style', 'layout', 'flexbox', 'grid', 'html', 'responsive']

        import re
        task_lower = task_description.lower()

        # Use word boundary matching to avoid substring matches
        def has_keyword(keywords, text):
            for kw in keywords:
                # Check for word boundaries to avoid substring matches
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    return True
            return False

        # Check for strong backend keywords
        has_strong_backend = has_keyword(strong_backend_keywords, task_lower)

        # Check for strong frontend keywords
        has_strong_frontend = has_keyword(strong_frontend_keywords, task_lower)

        # Don't spawn for pure frontend tasks (frontend keywords without backend context)
        if has_strong_frontend and not has_strong_backend:
            return False

        # Spawn only if strong backend keywords are present
        return has_strong_backend

    def get_spawn_ranges_by_complexity(self) -> Dict[str, tuple]:
        """
        Return spawn ranges based on task complexity.

        Simple tasks: Spawn later for basic endpoints
        Complex tasks: Spawn earlier for distributed systems
        """
        return {
            "simple": (0.5, 0.8),    # Late phase for simple endpoints
            "medium": (0.35, 0.65),  # Standard backend work
            "complex": (0.3, 0.60)   # Earlier for distributed systems/microservices
        }
