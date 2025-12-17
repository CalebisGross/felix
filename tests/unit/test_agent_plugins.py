"""
Unit tests for agent plugin system.

Tests the plugin architecture including:
- Plugin discovery and loading
- Agent creation via registry
- Task-based agent filtering
- Builtin and custom plugins
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.agents.base_specialized_agent import (
    SpecializedAgentPlugin,
    AgentMetadata,
    AgentPluginError,
    AgentPluginLoadError,
    AgentPluginValidationError
)
from src.agents.agent_plugin_registry import AgentPluginRegistry
from src.agents.llm_agent import LLMAgent


@pytest.mark.unit
@pytest.mark.plugins
class TestAgentMetadata:
    """Tests for AgentMetadata dataclass."""

    def test_agent_metadata_creation(self):
        """Test creating AgentMetadata with required fields."""
        metadata = AgentMetadata(
            agent_type="test_agent",
            display_name="Test Agent",
            description="Test agent for unit tests"
        )

        assert metadata.agent_type == "test_agent"
        assert metadata.display_name == "Test Agent"
        assert metadata.description == "Test agent for unit tests"
        assert metadata.spawn_range == (0.0, 1.0)  # Default
        assert metadata.default_tokens == 800  # Default
        assert metadata.priority == 0  # Default

    def test_agent_metadata_with_custom_values(self):
        """Test AgentMetadata with all custom values."""
        metadata = AgentMetadata(
            agent_type="custom",
            display_name="Custom Agent",
            description="Custom test agent",
            spawn_range=(0.2, 0.7),
            capabilities=["test_capability"],
            tags=["test"],
            default_tokens=1000,
            version="2.0.0",
            author="Test Author",
            priority=5
        )

        assert metadata.spawn_range == (0.2, 0.7)
        assert metadata.capabilities == ["test_capability"]
        assert metadata.tags == ["test"]
        assert metadata.default_tokens == 1000
        assert metadata.version == "2.0.0"
        assert metadata.author == "Test Author"
        assert metadata.priority == 5


@pytest.mark.unit
@pytest.mark.plugins
class TestPluginRegistry:
    """Tests for AgentPluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return AgentPluginRegistry()

    def test_registry_initialization(self, registry):
        """Test registry initializes empty."""
        assert len(registry.list_agent_types()) == 0
        stats = registry.get_statistics()
        assert stats["total_registered"] == 0

    def test_discover_builtin_plugins(self, registry):
        """Test discovery of builtin plugins."""
        count = registry.discover_builtin_plugins()

        # Should discover research, analysis, and critic plugins
        assert count >= 3
        assert "research" in registry.list_agent_types()
        assert "analysis" in registry.list_agent_types()
        assert "critic" in registry.list_agent_types()

    def test_get_metadata(self, registry):
        """Test getting metadata for registered plugins."""
        registry.discover_builtin_plugins()

        research_meta = registry.get_metadata("research")
        assert research_meta is not None
        assert research_meta.agent_type == "research"
        assert research_meta.display_name == "Research Agent"

    def test_get_metadata_nonexistent(self, registry):
        """Test getting metadata for non-existent plugin."""
        metadata = registry.get_metadata("nonexistent")
        assert metadata is None

    def test_create_agent_research(self, registry, mock_helix, mock_llm_provider):
        """Test creating a research agent via registry."""
        registry.discover_builtin_plugins()

        # Convert mock_llm_provider (BaseLLMProvider) to mock LMStudioClient
        mock_client = Mock()
        mock_client.complete = Mock()

        agent = registry.create_agent(
            agent_type="research",
            agent_id="test_research_001",
            spawn_time=0.1,
            helix=mock_helix,
            llm_client=mock_client
        )

        assert agent is not None
        assert agent.agent_id == "test_research_001"
        assert agent.spawn_time == 0.1
        assert agent.agent_type == "research"

    def test_create_agent_unknown_type(self, registry, mock_helix, mock_llm_provider):
        """Test creating agent with unknown type raises error."""
        registry.discover_builtin_plugins()

        mock_client = Mock()

        with pytest.raises(AgentPluginError, match="Unknown agent type"):
            registry.create_agent(
                agent_type="nonexistent",
                agent_id="test_001",
                spawn_time=0.5,
                helix=mock_helix,
                llm_client=mock_client
            )

    def test_get_agents_for_task(self, registry):
        """Test filtering agents by task characteristics."""
        registry.discover_builtin_plugins()

        # Research-related task should include research agent
        agents = registry.get_agents_for_task(
            task_description="Research quantum computing",
            task_complexity="medium"
        )

        assert "research" in agents

    def test_get_spawn_range(self, registry):
        """Test getting spawn range for agent type and complexity."""
        registry.discover_builtin_plugins()

        # Get spawn range for research agent on complex task
        spawn_range = registry.get_spawn_range("research", "complex")

        assert isinstance(spawn_range, tuple)
        assert len(spawn_range) == 2
        assert 0.0 <= spawn_range[0] <= spawn_range[1] <= 1.0

    def test_statistics_tracking(self, registry):
        """Test registry statistics tracking."""
        count = registry.discover_builtin_plugins()

        stats = registry.get_statistics()

        assert stats["total_registered"] == count
        assert stats["builtin_count"] == count
        assert stats["external_count"] == 0
        assert len(stats["agent_types"]) == count


@pytest.mark.unit
@pytest.mark.plugins
class TestCustomPluginLoading:
    """Tests for loading custom plugins from external directories."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary directory for test plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return AgentPluginRegistry()

    def test_load_external_plugin(self, registry, temp_plugin_dir):
        """Test loading a custom plugin from external directory."""
        # Create a simple test plugin file
        plugin_code = '''
from src.agents.base_specialized_agent import SpecializedAgentPlugin, AgentMetadata
from src.agents.llm_agent import LLMAgent

class TestAgent(LLMAgent):
    def __init__(self, agent_id, spawn_time, helix, llm_client, **kwargs):
        super().__init__(agent_id, spawn_time, helix, llm_client, "test", **kwargs)

    def create_position_aware_prompt(self, task, current_time):
        return "Test prompt", 100

class TestAgentPlugin(SpecializedAgentPlugin):
    def get_metadata(self):
        return AgentMetadata(
            agent_type="test_custom",
            display_name="Test Custom Agent",
            description="Test plugin"
        )

    def create_agent(self, agent_id, spawn_time, helix, llm_client,
                    token_budget_manager=None, **kwargs):
        return TestAgent(agent_id, spawn_time, helix, llm_client, **kwargs)
'''

        # Write plugin file
        plugin_file = temp_plugin_dir / "test_plugin.py"
        plugin_file.write_text(plugin_code)

        # Load plugins
        count = registry.add_plugin_directory(str(temp_plugin_dir))

        assert count == 1
        assert "test_custom" in registry.list_agent_types()

    def test_load_from_nonexistent_directory(self, registry):
        """Test loading from non-existent directory raises error."""
        with pytest.raises(AgentPluginLoadError):
            registry.add_plugin_directory("/nonexistent/path")

    def test_skip_private_modules(self, registry, temp_plugin_dir):
        """Test that modules starting with _ are skipped."""
        # Create plugin with underscore prefix
        plugin_file = temp_plugin_dir / "_private_plugin.py"
        plugin_file.write_text("# This should be skipped")

        count = registry.add_plugin_directory(str(temp_plugin_dir))

        assert count == 0  # Should skip _private_plugin.py


@pytest.mark.unit
@pytest.mark.plugins
class TestPluginValidation:
    """Tests for plugin validation."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary directory for test plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def registry(self):
        """Create fresh registry."""
        return AgentPluginRegistry()

    def test_invalid_spawn_range(self, registry, temp_plugin_dir):
        """Test that invalid spawn range is rejected."""
        plugin_code = '''
from src.agents.base_specialized_agent import SpecializedAgentPlugin, AgentMetadata

class InvalidPlugin(SpecializedAgentPlugin):
    def get_metadata(self):
        return AgentMetadata(
            agent_type="invalid",
            display_name="Invalid",
            description="Invalid spawn range",
            spawn_range=(1.5, 2.0)  # Invalid: > 1.0
        )

    def create_agent(self, *args, **kwargs):
        pass
'''

        plugin_file = Path(temp_plugin_dir) / "invalid_plugin.py"
        plugin_file.write_text(plugin_code)

        # Should fail to load due to validation error
        count = registry.add_plugin_directory(str(temp_plugin_dir))

        assert count == 0  # Plugin rejected
        assert "invalid" not in registry.list_agent_types()


@pytest.mark.unit
@pytest.mark.plugins
class TestAgentFactoryIntegration:
    """Tests for AgentFactory integration with plugin registry."""

    def test_factory_uses_registry(self, mock_helix, mock_llm_provider):
        """Test that AgentFactory uses plugin registry."""
        from src.communication.central_post import AgentFactory

        # Convert mock provider to client
        mock_client = Mock()

        factory = AgentFactory(
            helix=mock_helix,
            llm_client=mock_client
        )

        # Should have access to registry
        assert factory.agent_registry is not None

        # Should list builtin agents
        agent_types = factory.list_available_agent_types()
        assert "research" in agent_types
        assert "analysis" in agent_types
        assert "critic" in agent_types

    def test_factory_create_agent_by_type(self, mock_helix, mock_llm_provider):
        """Test creating agent via factory's create_agent_by_type."""
        from src.communication.central_post import AgentFactory

        mock_client = Mock()

        factory = AgentFactory(
            helix=mock_helix,
            llm_client=mock_client
        )

        # Create research agent
        agent = factory.create_agent_by_type(
            agent_type="research",
            complexity="medium",
            research_domain="technical"
        )

        assert agent is not None
        assert agent.agent_type == "research"

    def test_factory_backward_compatibility(self, mock_helix, mock_llm_provider):
        """Test that old factory methods still work."""
        from src.communication.central_post import AgentFactory

        mock_client = Mock()

        factory = AgentFactory(
            helix=mock_helix,
            llm_client=mock_client
        )

        # Old methods should still work
        research = factory.create_research_agent(domain="general")
        assert research.agent_type == "research"

        analysis = factory.create_analysis_agent(analysis_type="general")
        assert analysis.agent_type == "analysis"

        critic = factory.create_critic_agent(review_focus="general")
        assert critic.agent_type == "critic"


@pytest.mark.unit
@pytest.mark.plugins
class TestBuiltinPlugins:
    """Tests for builtin agent plugins."""

    @pytest.fixture
    def registry(self):
        """Create registry with builtin plugins."""
        reg = AgentPluginRegistry()
        reg.discover_builtin_plugins()
        return reg

    def test_research_plugin_metadata(self, registry):
        """Test ResearchAgentPlugin metadata."""
        metadata = registry.get_metadata("research")

        assert metadata.agent_type == "research"
        assert metadata.display_name == "Research Agent"
        assert "web_search" in metadata.capabilities
        assert "exploration" in metadata.tags

    def test_analysis_plugin_metadata(self, registry):
        """Test AnalysisAgentPlugin metadata."""
        metadata = registry.get_metadata("analysis")

        assert metadata.agent_type == "analysis"
        assert metadata.display_name == "Analysis Agent"
        assert "pattern_identification" in metadata.capabilities
        assert "analysis" in metadata.tags

    def test_critic_plugin_metadata(self, registry):
        """Test CriticAgentPlugin metadata."""
        metadata = registry.get_metadata("critic")

        assert metadata.agent_type == "critic"
        assert metadata.display_name == "Critic Agent"
        assert "quality_assurance" in metadata.capabilities
        assert "review" in metadata.tags

    def test_spawn_ranges_by_complexity(self, registry):
        """Test that builtin plugins have appropriate spawn ranges."""
        research_ranges = registry._plugins["research"].get_spawn_ranges_by_complexity()

        # Research should spawn earlier for complex tasks
        assert research_ranges["complex"][0] <= research_ranges["simple"][0]
