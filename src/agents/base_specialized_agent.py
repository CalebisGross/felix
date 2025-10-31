"""
Base interface for specialized agent plugins in the Felix Framework.

This module defines the plugin API that all specialized agents must implement
to be compatible with Felix's agent system. Agent plugins can be built-in or
loaded from external directories.

Plugin API Overview:
1. Inherit from SpecializedAgentPlugin
2. Implement get_metadata() to describe capabilities
3. Implement create_agent() to instantiate your agent
4. Optional: Override supports_task() for intelligent agent selection

Example Custom Agent Plugin:
    ```python
    from src.agents.base_specialized_agent import SpecializedAgentPlugin, AgentMetadata
    from src.agents.llm_agent import LLMAgent

    class CodeReviewAgent(SpecializedAgentPlugin):
        def get_metadata(self) -> AgentMetadata:
            return AgentMetadata(
                agent_type="code_review",
                display_name="Code Review Agent",
                description="Specialized in reviewing code for bugs and quality",
                spawn_range=(0.3, 0.7),  # Analysis phase
                capabilities=["code_analysis", "bug_detection", "style_review"],
                tags=["engineering", "quality"]
            )

        def create_agent(self, agent_id: str, spawn_time: float,
                        helix, llm_client, **kwargs) -> LLMAgent:
            return CodeReviewAgent(agent_id, spawn_time, helix, llm_client, **kwargs)
    ```
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.helix_geometry import HelixGeometry
    from src.llm.lm_studio_client import LMStudioClient
    from src.agents.llm_agent import LLMAgent
    from src.llm.token_budget import TokenBudgetManager


@dataclass
class AgentMetadata:
    """
    Metadata describing an agent plugin's capabilities and characteristics.

    This metadata is used by the system to:
    - Display available agent types in GUI/CLI
    - Determine which agents to spawn for a given task
    - Configure spawn timing and parameters
    - Enable agent discovery and filtering

    Attributes:
        agent_type: Unique identifier (e.g., "research", "analysis", "code_review")
        display_name: Human-readable name (e.g., "Research Agent")
        description: Short description of agent's purpose
        spawn_range: Default normalized spawn time range (0.0-1.0)
        capabilities: List of capabilities (e.g., ["web_search", "code_analysis"])
        tags: Classification tags (e.g., ["engineering", "creative", "critical"])
        default_tokens: Default max tokens for this agent type
        version: Plugin version string
        author: Plugin author (optional)
        priority: Spawn priority (higher = spawn earlier for same complexity)
    """
    agent_type: str
    display_name: str
    description: str
    spawn_range: Tuple[float, float] = (0.0, 1.0)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    default_tokens: int = 800
    version: str = "1.0.0"
    author: Optional[str] = None
    priority: int = 0  # Higher priority agents spawn first


class SpecializedAgentPlugin(ABC):
    """
    Abstract base class for all specialized agent plugins.

    All custom agents must inherit from this class and implement the required methods.
    The plugin system uses these methods to:
    1. Discover available agent types
    2. Create agent instances dynamically
    3. Match agents to tasks intelligently

    Lifecycle:
        1. Plugin discovery: System scans for SpecializedAgentPlugin subclasses
        2. Metadata extraction: get_metadata() called to register agent type
        3. Task matching: supports_task() called to filter compatible agents
        4. Agent creation: create_agent() called to instantiate the agent
    """

    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """
        Return metadata describing this agent plugin.

        This method is called during plugin discovery to register the agent type
        with the system. Metadata determines how and when the agent is spawned.

        Returns:
            AgentMetadata describing the agent's capabilities and configuration

        Example:
            ```python
            def get_metadata(self) -> AgentMetadata:
                return AgentMetadata(
                    agent_type="research",
                    display_name="Research Agent",
                    description="Broad information gathering and exploration",
                    spawn_range=(0.0, 0.3),  # Early exploration phase
                    capabilities=["web_search", "document_analysis"],
                    tags=["exploration", "information_gathering"],
                    default_tokens=800
                )
            ```
        """
        pass

    @abstractmethod
    def create_agent(self,
                    agent_id: str,
                    spawn_time: float,
                    helix: "HelixGeometry",
                    llm_client: "LMStudioClient",
                    token_budget_manager: Optional["TokenBudgetManager"] = None,
                    **kwargs) -> "LLMAgent":
        """
        Create an instance of this specialized agent.

        This method is called by AgentFactory when the system needs to spawn
        a new agent of this type. The implementation should instantiate your
        agent class with the provided parameters.

        Args:
            agent_id: Unique identifier for the agent instance
            spawn_time: Normalized spawn time (0.0-1.0) on the helix
            helix: HelixGeometry instance for position calculations
            llm_client: LLM client for agent-LLM communication
            token_budget_manager: Optional token budget manager
            **kwargs: Additional parameters (domain, focus, etc.)

        Returns:
            Instance of your specialized agent (must inherit from LLMAgent)

        Example:
            ```python
            def create_agent(self, agent_id, spawn_time, helix, llm_client,
                           token_budget_manager=None, **kwargs):
                return MyCustomAgent(
                    agent_id=agent_id,
                    spawn_time=spawn_time,
                    helix=helix,
                    llm_client=llm_client,
                    token_budget_manager=token_budget_manager,
                    custom_param=kwargs.get('custom_param', 'default')
                )
            ```
        """
        pass

    def supports_task(self, task_description: str, task_metadata: Dict[str, Any]) -> bool:
        """
        Determine if this agent type is suitable for the given task.

        This method enables intelligent agent selection based on task characteristics.
        The default implementation returns True (agent supports all tasks).
        Override this method to implement custom task filtering logic.

        Args:
            task_description: Human-readable task description
            task_metadata: Structured task metadata (complexity, domain, etc.)

        Returns:
            True if this agent type should be considered for the task

        Example:
            ```python
            def supports_task(self, task_description, task_metadata):
                # Code review agent only for code-related tasks
                keywords = ['code', 'bug', 'function', 'class', 'review']
                return any(kw in task_description.lower() for kw in keywords)
            ```
        """
        return True  # Default: support all tasks

    def get_spawn_ranges_by_complexity(self) -> Dict[str, Tuple[float, float]]:
        """
        Get spawn time ranges for different task complexities.

        Override this method to customize spawn timing based on task complexity.
        The default implementation uses the same range from metadata for all complexities.

        Returns:
            Dictionary mapping complexity level to (min, max) spawn time range

        Example:
            ```python
            def get_spawn_ranges_by_complexity(self):
                return {
                    "simple": (0.1, 0.3),
                    "medium": (0.05, 0.25),
                    "complex": (0.0, 0.2)  # Spawn earlier for complex tasks
                }
            ```
        """
        metadata = self.get_metadata()
        return {
            "simple": metadata.spawn_range,
            "medium": metadata.spawn_range,
            "complex": metadata.spawn_range
        }


class AgentPluginError(Exception):
    """Base exception for agent plugin errors."""
    pass


class AgentPluginLoadError(AgentPluginError):
    """Raised when a plugin fails to load."""
    pass


class AgentPluginValidationError(AgentPluginError):
    """Raised when a plugin fails validation."""
    pass
