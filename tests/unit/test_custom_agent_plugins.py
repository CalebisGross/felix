"""
Tests for custom agent plugins (Frontend, Backend, QA).

Tests verify:
1. Plugin metadata and configuration
2. Task filtering (supports_task) correctness
3. Complexity-based spawn ranges
4. Integration with AgentFactory
5. Scenario-based spawning behavior
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from examples.custom_agents.frontend_agent import FrontendAgentPlugin
from examples.custom_agents.backend_agent import BackendAgentPlugin
from examples.custom_agents.qa_agent import QAAgentPlugin


@pytest.mark.unit
@pytest.mark.plugins
class TestFrontendAgentPlugin:
    """Tests for Frontend Agent Plugin."""

    def test_frontend_metadata(self):
        """Test frontend agent metadata configuration."""
        plugin = FrontendAgentPlugin()
        metadata = plugin.get_metadata()

        assert metadata.agent_type == "frontend"
        assert metadata.display_name == "Frontend Development Agent"
        assert metadata.spawn_range == (0.3, 0.6)
        assert metadata.priority == 7
        assert "ui_design" in metadata.capabilities
        assert "frontend" in metadata.tags

    def test_frontend_supports_ui_task(self):
        """Frontend agent should spawn for UI/UX tasks."""
        plugin = FrontendAgentPlugin()

        tasks_should_support = [
            "Design a responsive navigation menu",
            "Fix CSS alignment bug on homepage",
            "Build React component for user profile",
            "Create accessible form with ARIA labels",
            "Implement mobile-first layout"
        ]

        for task in tasks_should_support:
            assert plugin.supports_task(task, {}), f"Should support: {task}"

    def test_frontend_rejects_backend_task(self):
        """Frontend agent should NOT spawn for pure backend tasks."""
        plugin = FrontendAgentPlugin()

        tasks_should_reject = [
            "Design REST API for user management",
            "Create database schema for orders",
            "Build authentication server with JWT"
        ]

        for task in tasks_should_reject:
            assert not plugin.supports_task(task, {}), f"Should reject: {task}"

    def test_frontend_complexity_spawn_ranges(self):
        """Frontend agent should have different spawn ranges by complexity."""
        plugin = FrontendAgentPlugin()
        ranges = plugin.get_spawn_ranges_by_complexity()

        assert ranges["simple"] == (0.5, 0.8)  # Later for simple
        assert ranges["medium"] == (0.35, 0.65)  # Standard
        assert ranges["complex"] == (0.3, 0.60)  # Earlier for complex


@pytest.mark.unit
@pytest.mark.plugins
class TestBackendAgentPlugin:
    """Tests for Backend Agent Plugin."""

    def test_backend_metadata(self):
        """Test backend agent metadata configuration."""
        plugin = BackendAgentPlugin()
        metadata = plugin.get_metadata()

        assert metadata.agent_type == "backend"
        assert metadata.display_name == "Backend Development Agent"
        assert metadata.spawn_range == (0.3, 0.6)
        assert metadata.priority == 7
        assert "api_design" in metadata.capabilities
        assert "backend" in metadata.tags

    def test_backend_supports_api_task(self):
        """Backend agent should spawn for API/database tasks."""
        plugin = BackendAgentPlugin()

        tasks_should_support = [
            "Design REST API for user authentication",
            "Create database schema for e-commerce",
            "Build GraphQL endpoint for products",
            "Implement JWT authentication middleware",
            "Design microservices architecture"
        ]

        for task in tasks_should_support:
            assert plugin.supports_task(task, {}), f"Should support: {task}"

    def test_backend_rejects_frontend_task(self):
        """Backend agent should NOT spawn for pure frontend tasks."""
        plugin = BackendAgentPlugin()

        tasks_should_reject = [
            "Fix CSS styling on button",
            "Create responsive layout with flexbox"
        ]

        for task in tasks_should_reject:
            assert not plugin.supports_task(task, {}), f"Should reject: {task}"

    def test_backend_complexity_spawn_ranges(self):
        """Backend agent should have different spawn ranges by complexity."""
        plugin = BackendAgentPlugin()
        ranges = plugin.get_spawn_ranges_by_complexity()

        assert ranges["simple"] == (0.5, 0.8)
        assert ranges["medium"] == (0.35, 0.65)
        assert ranges["complex"] == (0.3, 0.60)


@pytest.mark.unit
@pytest.mark.plugins
class TestQAAgentPlugin:
    """Tests for QA Agent Plugin."""

    def test_qa_metadata(self):
        """Test QA agent metadata configuration."""
        plugin = QAAgentPlugin()
        metadata = plugin.get_metadata()

        assert metadata.agent_type == "qa"
        assert metadata.display_name == "QA & Testing Agent"
        assert metadata.spawn_range == (0.5, 0.8)  # Late phase
        assert metadata.priority == 5  # Lower than critic
        assert "test_strategy" in metadata.capabilities
        assert "qa" in metadata.tags

    def test_qa_supports_testing_task(self):
        """QA agent should spawn for testing/QA tasks."""
        plugin = QAAgentPlugin()

        tasks_should_support = [
            "Create test strategy for payment system",
            "Write unit tests for user service",
            "Design E2E test suite for checkout flow",
            "Identify edge cases for validation logic"
        ]

        for task in tasks_should_support:
            assert plugin.supports_task(task, {}), f"Should support: {task}"

    def test_qa_supports_development_tasks(self):
        """QA agent should spawn for medium/complex development tasks."""
        plugin = QAAgentPlugin()

        # Should spawn for development tasks with medium/complex complexity
        assert plugin.supports_task("Build todo app", {"complexity": "medium"})
        assert plugin.supports_task("Design payment system", {"complexity": "complex"})

        # Should NOT spawn for simple development tasks
        assert not plugin.supports_task("Build todo app", {"complexity": "simple"})

    def test_qa_complexity_spawn_ranges(self):
        """QA agent should have different spawn ranges by complexity."""
        plugin = QAAgentPlugin()
        ranges = plugin.get_spawn_ranges_by_complexity()

        assert ranges["simple"] == (0.7, 0.9)  # Very late or not at all
        assert ranges["medium"] == (0.55, 0.80)  # Standard
        assert ranges["complex"] == (0.5, 0.75)  # Earlier for complex


@pytest.mark.unit
@pytest.mark.plugins
class TestScenarioBasedSpawning:
    """Test real-world scenarios with multiple plugins."""

    def test_scenario_rest_api_design(self):
        """Scenario: Design REST API - should spawn backend, not frontend."""
        task = "Design REST API for user management with authentication"
        metadata = {"complexity": "complex"}

        frontend = FrontendAgentPlugin()
        backend = BackendAgentPlugin()
        qa = QAAgentPlugin()

        assert not frontend.supports_task(task, metadata)
        assert backend.supports_task(task, metadata)
        assert qa.supports_task(task, metadata)  # Development task

    def test_scenario_fullstack_app(self):
        """Scenario: Build full-stack app - should spawn all agents."""
        task = "Build todo app with React frontend and Node.js backend"
        metadata = {"complexity": "complex"}

        frontend = FrontendAgentPlugin()
        backend = BackendAgentPlugin()
        qa = QAAgentPlugin()

        assert frontend.supports_task(task, metadata)  # Has "frontend" and "React"
        assert backend.supports_task(task, metadata)  # Has "backend" and "Node"
        assert qa.supports_task(task, metadata)  # Development task + complex

    def test_scenario_css_bug(self):
        """Scenario: Fix CSS bug - should spawn frontend only."""
        task = "Fix CSS alignment bug on homepage navigation"
        metadata = {"complexity": "simple"}

        frontend = FrontendAgentPlugin()
        backend = BackendAgentPlugin()
        qa = QAAgentPlugin()

        assert frontend.supports_task(task, metadata)  # Has "CSS"
        assert not backend.supports_task(task, metadata)  # No backend keywords
        assert not qa.supports_task(task, metadata)  # Simple complexity

    def test_scenario_non_technical(self):
        """Scenario: Non-technical task - no custom agents should spawn."""
        task = "Explain the history of quantum computing"
        metadata = {"complexity": "medium"}

        frontend = FrontendAgentPlugin()
        backend = BackendAgentPlugin()
        qa = QAAgentPlugin()

        assert not frontend.supports_task(task, metadata)
        assert not backend.supports_task(task, metadata)
        assert not qa.supports_task(task, metadata)

    def test_scenario_testing_focus(self):
        """Scenario: Testing focus - QA should spawn, others conditional."""
        task = "Create comprehensive test suite for payment processing API"
        metadata = {"complexity": "complex"}

        frontend = FrontendAgentPlugin()
        backend = BackendAgentPlugin()
        qa = QAAgentPlugin()

        assert not frontend.supports_task(task, metadata)  # No UI keywords
        assert backend.supports_task(task, metadata)  # Has "API"
        assert qa.supports_task(task, metadata)  # Has "test"


@pytest.mark.unit
@pytest.mark.plugins
class TestPluginCreation:
    """Test that plugins can create agent instances."""

    def test_frontend_creates_agent(self, mock_helix, mock_llm_provider):
        """Frontend plugin should create agent instance."""
        plugin = FrontendAgentPlugin()

        agent = plugin.create_agent(
            agent_id="frontend-1",
            spawn_time=0.4,
            helix=mock_helix,
            llm_client=mock_llm_provider
        )

        assert agent is not None
        assert agent.agent_id == "frontend-1"
        assert agent.agent_type == "frontend"

    def test_backend_creates_agent(self, mock_helix, mock_llm_provider):
        """Backend plugin should create agent instance."""
        plugin = BackendAgentPlugin()

        agent = plugin.create_agent(
            agent_id="backend-1",
            spawn_time=0.4,
            helix=mock_helix,
            llm_client=mock_llm_provider
        )

        assert agent is not None
        assert agent.agent_id == "backend-1"
        assert agent.agent_type == "backend"

    def test_qa_creates_agent(self, mock_helix, mock_llm_provider):
        """QA plugin should create agent instance."""
        plugin = QAAgentPlugin()

        agent = plugin.create_agent(
            agent_id="qa-1",
            spawn_time=0.6,
            helix=mock_helix,
            llm_client=mock_llm_provider
        )

        assert agent is not None
        assert agent.agent_id == "qa-1"
        assert agent.agent_type == "qa"
