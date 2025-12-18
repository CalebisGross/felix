"""
Agent plugin registry for the Felix Framework.

This module provides plugin discovery, loading, and management functionality
for specialized agents. It enables Felix to dynamically discover agent plugins
from both built-in and external sources.

Features:
- Automatic plugin discovery in specified directories
- Hot-reloading of external plugins
- Plugin validation and error handling
- Task-based agent filtering and selection
- Priority-based agent spawning

Usage:
    ```python
    # Initialize registry with builtin plugins
    registry = AgentPluginRegistry()
    registry.discover_builtin_plugins()

    # Add external plugin directory
    registry.add_plugin_directory("./custom_agents")

    # Get all available agent types
    agent_types = registry.list_agent_types()

    # Create agent instance
    agent = registry.create_agent(
        agent_type="research",
        agent_id="research_001",
        spawn_time=0.1,
        helix=helix,
        llm_client=client
    )

    # Filter agents by task
    suitable_agents = registry.get_agents_for_task(
        task_description="Analyze Python code for bugs",
        task_complexity="medium"
    )
    ```
"""

import os
import sys
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import asdict, dataclass

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata,
    AgentPluginError,
    AgentPluginLoadError,
    AgentPluginValidationError
)

if TYPE_CHECKING:
    from src.core.helix_geometry import HelixGeometry
    from src.llm.lm_studio_client import LMStudioClient
    from src.agents.llm_agent import LLMAgent
    from src.llm.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)


@dataclass
class PluginLoadResult:
    """
    Result of a plugin loading operation.

    Provides detailed feedback about what succeeded and what failed,
    allowing callers to distinguish "no plugins" from "all failed".
    """
    loaded: int
    failed: int
    errors: List[str]

    @property
    def success(self) -> bool:
        """Returns True if at least one plugin loaded successfully."""
        return self.loaded > 0

    @property
    def all_failed(self) -> bool:
        """Returns True if there were plugins to load but all failed."""
        return self.failed > 0 and self.loaded == 0


class AgentPluginRegistry:
    """
    Registry for discovering, loading, and managing agent plugins.

    The registry maintains a catalog of available agent types and provides
    methods to create agent instances dynamically. Plugins can be built-in
    (shipped with Felix) or external (loaded from custom directories).

    Architecture:
        - Builtin plugins: Located in src/agents/builtin/
        - External plugins: Loaded from user-specified directories
        - Plugin classes: Must inherit from SpecializedAgentPlugin
        - Auto-discovery: Scans directories for valid plugin classes
    """

    def __init__(self):
        """Initialize the agent plugin registry."""
        # Registry: agent_type -> plugin instance
        self._plugins: Dict[str, SpecializedAgentPlugin] = {}

        # Metadata cache: agent_type -> AgentMetadata
        self._metadata_cache: Dict[str, AgentMetadata] = {}

        # Plugin source tracking: agent_type -> source path
        self._plugin_sources: Dict[str, str] = {}

        # External plugin directories
        self._external_dirs: List[Path] = []

        # Statistics
        self._stats = {
            "plugins_loaded": 0,
            "plugins_failed": 0,
            "builtin_count": 0,
            "external_count": 0
        }

    def discover_builtin_plugins(self) -> int:
        """
        Discover and load built-in agent plugins from src/agents/builtin/.

        This method scans the builtin directory for Python modules containing
        SpecializedAgentPlugin subclasses and registers them.

        Returns:
            Number of builtin plugins successfully loaded

        Raises:
            AgentPluginLoadError: If builtin directory is missing or inaccessible
        """
        builtin_dir = Path(__file__).parent / "builtin"

        if not builtin_dir.exists():
            logger.warning(f"Builtin plugins directory not found: {builtin_dir}")
            return 0

        logger.info(f"Discovering builtin plugins in {builtin_dir}")
        result = self._scan_directory(builtin_dir, is_builtin=True)
        self._stats["builtin_count"] = result.loaded

        if result.all_failed:
            logger.error(f"All builtin plugins failed to load: {result.errors}")
        elif result.failed > 0:
            logger.warning(f"Some builtin plugins failed: {result.failed} failed, {result.loaded} loaded")

        logger.info(f"Loaded {result.loaded} builtin agent plugins")
        return result.loaded

    def add_plugin_directory(self, directory: str) -> int:
        """
        Add an external directory for plugin discovery.

        External plugins enable users to create custom agent types without
        modifying Felix's core code. Plugins are loaded from Python modules
        in the specified directory.

        Args:
            directory: Path to directory containing plugin modules

        Returns:
            Number of plugins successfully loaded from the directory

        Raises:
            AgentPluginLoadError: If directory doesn't exist or is inaccessible

        Example:
            ```python
            registry = AgentPluginRegistry()
            count = registry.add_plugin_directory("./my_custom_agents")
            print(f"Loaded {count} custom plugins")
            ```
        """
        plugin_dir = Path(directory).resolve()

        if not plugin_dir.exists():
            raise AgentPluginLoadError(f"Plugin directory not found: {plugin_dir}")

        if not plugin_dir.is_dir():
            raise AgentPluginLoadError(f"Not a directory: {plugin_dir}")

        self._external_dirs.append(plugin_dir)
        logger.info(f"Scanning external plugin directory: {plugin_dir}")

        result = self._scan_directory(plugin_dir, is_builtin=False)
        self._stats["external_count"] += result.loaded

        if result.all_failed:
            logger.error(f"All plugins failed to load from {plugin_dir}: {result.errors}")
        elif result.failed > 0:
            logger.warning(f"Some plugins failed in {plugin_dir}: {result.failed} failed, {result.loaded} loaded")

        logger.info(f"Loaded {result.loaded} external plugins from {plugin_dir}")
        return result.loaded

    def _scan_directory(self, directory: Path, is_builtin: bool = False) -> PluginLoadResult:
        """
        Scan a directory for plugin modules and load them.

        Args:
            directory: Directory path to scan
            is_builtin: Whether this is the builtin plugins directory

        Returns:
            PluginLoadResult with loaded count, failed count, and error messages
        """
        loaded_count = 0
        failed_count = 0
        errors: List[str] = []

        # Find all .py files in directory
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip __init__.py and private modules

            try:
                plugins = self._load_plugins_from_file(py_file, is_builtin)
                loaded_count += len(plugins)
            except Exception as e:
                error_msg = f"Failed to load plugins from {py_file}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed_count += 1
                self._stats["plugins_failed"] += 1

        return PluginLoadResult(loaded=loaded_count, failed=failed_count, errors=errors)

    def _load_plugins_from_file(self, file_path: Path, is_builtin: bool) -> List[str]:
        """
        Load plugin classes from a Python file.

        Args:
            file_path: Path to Python module file
            is_builtin: Whether this is a builtin plugin

        Returns:
            List of agent_types successfully loaded from the file
        """
        loaded_types = []

        try:
            # Import the module
            module_name = file_path.stem
            if is_builtin:
                # Use package import for builtin plugins
                module_name = f"src.agents.builtin.{module_name}"
                module = importlib.import_module(module_name)
            else:
                # Use file-based import for external plugins
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    raise AgentPluginLoadError(f"Could not load module from {file_path}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

            # Find all SpecializedAgentPlugin subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # Check if it's a class that inherits from SpecializedAgentPlugin
                if (isinstance(attr, type) and
                    issubclass(attr, SpecializedAgentPlugin) and
                    attr is not SpecializedAgentPlugin):

                    try:
                        # Instantiate the plugin
                        plugin = attr()

                        # Validate and register
                        self._register_plugin(plugin, str(file_path))
                        loaded_types.append(plugin.get_metadata().agent_type)

                    except Exception as e:
                        logger.error(f"Failed to instantiate plugin {attr_name}: {e}")
                        self._stats["plugins_failed"] += 1

        except Exception as e:
            logger.error(f"Error loading module {file_path}: {e}")
            raise AgentPluginLoadError(f"Failed to load {file_path}: {e}")

        return loaded_types

    def _register_plugin(self, plugin: SpecializedAgentPlugin, source: str) -> None:
        """
        Register a plugin instance with the registry.

        Args:
            plugin: Plugin instance to register
            source: Source file path

        Raises:
            AgentPluginValidationError: If plugin validation fails
        """
        try:
            # Get and validate metadata
            metadata = plugin.get_metadata()
            self._validate_metadata(metadata)

            agent_type = metadata.agent_type

            # Check for duplicates
            if agent_type in self._plugins:
                logger.warning(
                    f"Plugin '{agent_type}' already registered from "
                    f"{self._plugin_sources[agent_type]}, replacing with {source}"
                )

            # Register the plugin
            self._plugins[agent_type] = plugin
            self._metadata_cache[agent_type] = metadata
            self._plugin_sources[agent_type] = source
            self._stats["plugins_loaded"] += 1

            logger.info(
                f"Registered plugin: {agent_type} "
                f"({metadata.display_name}) from {source}"
            )

        except Exception as e:
            raise AgentPluginValidationError(f"Plugin validation failed: {e}")

    def _validate_metadata(self, metadata: AgentMetadata) -> None:
        """
        Validate plugin metadata.

        Args:
            metadata: Metadata to validate

        Raises:
            AgentPluginValidationError: If validation fails
        """
        if not metadata.agent_type:
            raise AgentPluginValidationError("agent_type cannot be empty")

        if not metadata.display_name:
            raise AgentPluginValidationError("display_name cannot be empty")

        if not metadata.description:
            raise AgentPluginValidationError("description cannot be empty")

        min_spawn, max_spawn = metadata.spawn_range
        if not (0.0 <= min_spawn <= max_spawn <= 1.0):
            raise AgentPluginValidationError(
                f"Invalid spawn_range: {metadata.spawn_range}. "
                "Must be (min, max) where 0.0 <= min <= max <= 1.0"
            )

        if metadata.default_tokens <= 0:
            raise AgentPluginValidationError(
                f"default_tokens must be positive, got {metadata.default_tokens}"
            )

    def create_agent(self,
                    agent_type: str,
                    agent_id: str,
                    spawn_time: float,
                    helix: "HelixGeometry",
                    llm_client: "LMStudioClient",
                    token_budget_manager: Optional["TokenBudgetManager"] = None,
                    **kwargs) -> "LLMAgent":
        """
        Create an agent instance of the specified type.

        Args:
            agent_type: Type of agent to create (e.g., "research", "analysis")
            agent_id: Unique identifier for the agent instance
            spawn_time: Normalized spawn time (0.0-1.0)
            helix: HelixGeometry instance
            llm_client: LLM client for the agent
            token_budget_manager: Optional token budget manager
            **kwargs: Additional parameters passed to the agent plugin

        Returns:
            Instance of the requested agent type

        Raises:
            AgentPluginError: If agent_type is not registered or creation fails

        Example:
            ```python
            agent = registry.create_agent(
                agent_type="research",
                agent_id="research_001",
                spawn_time=0.1,
                helix=helix,
                llm_client=client,
                research_domain="technical"
            )
            ```
        """
        if agent_type not in self._plugins:
            raise AgentPluginError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {', '.join(self.list_agent_types())}"
            )

        plugin = self._plugins[agent_type]

        try:
            agent = plugin.create_agent(
                agent_id=agent_id,
                spawn_time=spawn_time,
                helix=helix,
                llm_client=llm_client,
                token_budget_manager=token_budget_manager,
                **kwargs
            )

            logger.debug(
                f"Created agent {agent_id} of type {agent_type} "
                f"with spawn_time {spawn_time:.3f}"
            )

            return agent

        except Exception as e:
            raise AgentPluginError(f"Failed to create {agent_type} agent: {e}")

    def list_agent_types(self) -> List[str]:
        """
        Get list of all registered agent types.

        Returns:
            List of agent type identifiers (e.g., ["research", "analysis", "critic"])
        """
        return sorted(self._plugins.keys())

    def get_metadata(self, agent_type: str) -> Optional[AgentMetadata]:
        """
        Get metadata for a specific agent type.

        Args:
            agent_type: Type identifier

        Returns:
            AgentMetadata if registered, None otherwise
        """
        return self._metadata_cache.get(agent_type)

    def get_all_metadata(self) -> Dict[str, AgentMetadata]:
        """
        Get metadata for all registered agent types.

        Returns:
            Dictionary mapping agent_type to AgentMetadata
        """
        return dict(self._metadata_cache)

    def get_agents_for_task(self,
                           task_description: str,
                           task_complexity: str = "medium",
                           task_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get list of agent types suitable for a given task.

        This method filters agents based on:
        1. Plugin's supports_task() method
        2. Task complexity and agent spawn ranges
        3. Agent priority

        Args:
            task_description: Human-readable task description
            task_complexity: Task complexity ("simple", "medium", "complex")
            task_metadata: Optional structured task metadata

        Returns:
            List of agent types sorted by priority (highest first)

        Example:
            ```python
            agents = registry.get_agents_for_task(
                task_description="Review Python code for bugs",
                task_complexity="medium"
            )
            # Returns: ["code_review", "critic", "analysis"]
            ```
        """
        if task_metadata is None:
            task_metadata = {"complexity": task_complexity}
        else:
            task_metadata["complexity"] = task_complexity

        suitable_agents = []

        for agent_type, plugin in self._plugins.items():
            try:
                # Check if plugin supports this task
                if plugin.supports_task(task_description, task_metadata):
                    metadata = self._metadata_cache[agent_type]
                    suitable_agents.append((agent_type, metadata.priority))
            except Exception as e:
                logger.error(f"Error checking task support for {agent_type}: {e}")

        # Sort by priority (descending)
        suitable_agents.sort(key=lambda x: x[1], reverse=True)

        return [agent_type for agent_type, _ in suitable_agents]

    def get_spawn_range(self, agent_type: str, complexity: str = "medium") -> Tuple[float, float]:
        """
        Get spawn time range for an agent type and task complexity.

        Args:
            agent_type: Type of agent
            complexity: Task complexity ("simple", "medium", "complex")

        Returns:
            (min_spawn, max_spawn) tuple

        Raises:
            AgentPluginError: If agent_type is not registered
        """
        if agent_type not in self._plugins:
            raise AgentPluginError(f"Unknown agent type: {agent_type}")

        plugin = self._plugins[agent_type]
        ranges = plugin.get_spawn_ranges_by_complexity()

        return ranges.get(complexity, self._metadata_cache[agent_type].spawn_range)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with plugin loading statistics
        """
        return {
            **self._stats,
            "total_registered": len(self._plugins),
            "agent_types": self.list_agent_types()
        }

    def reload_external_plugins(self) -> PluginLoadResult:
        """
        Reload all plugins from external directories.

        This enables hot-reloading of custom plugins without restarting Felix.
        Builtin plugins are not reloaded. Stats are properly reset before reloading.

        Returns:
            PluginLoadResult with loaded count, failed count, and any errors
        """
        # Count external plugins being removed
        external_types = [
            agent_type for agent_type, source in self._plugin_sources.items()
            if not source.startswith("builtin")
        ]
        removed_count = len(external_types)

        # Clear external plugins
        for agent_type in external_types:
            del self._plugins[agent_type]
            del self._metadata_cache[agent_type]
            del self._plugin_sources[agent_type]

        # Reset external stats before reloading (Issue 5.5 fix)
        self._stats["external_count"] = 0
        self._stats["plugins_loaded"] -= removed_count
        # Note: We don't reset plugins_failed as that's cumulative across all loads

        # Rescan external directories
        total_loaded = 0
        total_failed = 0
        all_errors: List[str] = []

        for directory in self._external_dirs:
            result = self._scan_directory(directory, is_builtin=False)
            total_loaded += result.loaded
            total_failed += result.failed
            all_errors.extend(result.errors)
            self._stats["external_count"] += result.loaded

        combined_result = PluginLoadResult(
            loaded=total_loaded,
            failed=total_failed,
            errors=all_errors
        )

        if combined_result.all_failed:
            logger.error(f"All external plugins failed to reload: {all_errors}")
        elif combined_result.failed > 0:
            logger.warning(
                f"Some external plugins failed to reload: "
                f"{total_failed} failed, {total_loaded} loaded"
            )

        logger.info(f"Reloaded {total_loaded} external plugins")
        return combined_result


# Global registry instance (singleton pattern)
_global_registry: Optional[AgentPluginRegistry] = None


def get_global_registry() -> AgentPluginRegistry:
    """
    Get the global agent plugin registry instance.

    Returns:
        Global AgentPluginRegistry singleton
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentPluginRegistry()
        # Auto-discover builtin plugins on first access
        _global_registry.discover_builtin_plugins()

    return _global_registry
