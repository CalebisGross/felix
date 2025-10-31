"""
Demo script showing Felix's agent plugin system.

This script demonstrates:
1. Loading builtin agent plugins
2. Loading custom agent plugins from external directory
3. Listing available agent types
4. Creating agents using the plugin system
5. Using AgentFactory with plugin registry

Usage:
    python examples/plugin_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.helix_geometry import HelixGeometry
from src.agents.agent_plugin_registry import AgentPluginRegistry, get_global_registry
from src.communication.central_post import AgentFactory
from src.llm.router_adapter import create_router_adapter


def demo_basic_registry():
    """Demonstrate basic registry usage."""
    print("=" * 60)
    print("1. Basic Registry Usage")
    print("=" * 60)

    # Get global registry (auto-loads builtin plugins)
    registry = get_global_registry()

    # List available agent types
    print("\nAvailable builtin agent types:")
    for agent_type in registry.list_agent_types():
        metadata = registry.get_metadata(agent_type)
        print(f"  - {agent_type}: {metadata.display_name}")
        print(f"    {metadata.description}")
        print(f"    Spawn range: {metadata.spawn_range}")
        print(f"    Capabilities: {', '.join(metadata.capabilities)}")
        print()


def demo_custom_plugins():
    """Demonstrate loading custom plugins."""
    print("=" * 60)
    print("2. Custom Plugin Loading")
    print("=" * 60)

    registry = AgentPluginRegistry()
    registry.discover_builtin_plugins()

    # Load custom plugins from examples/custom_agents
    custom_dir = project_root / "examples" / "custom_agents"
    if custom_dir.exists():
        try:
            count = registry.add_plugin_directory(str(custom_dir))
            print(f"\nLoaded {count} custom plugin(s) from {custom_dir}")

            print("\nAll available agents (builtin + custom):")
            for agent_type in registry.list_agent_types():
                metadata = registry.get_metadata(agent_type)
                source = "builtin" if agent_type in ["research", "analysis", "critic"] else "custom"
                print(f"  - [{source}] {agent_type}: {metadata.display_name}")

        except Exception as e:
            print(f"\nNote: Could not load custom plugins: {e}")
            print("This is normal if custom_agents directory doesn't have valid plugins yet.")
    else:
        print(f"\nNote: Custom agents directory not found at {custom_dir}")


def demo_agent_creation():
    """Demonstrate creating agents via registry."""
    print("\n" + "=" * 60)
    print("3. Creating Agents via Registry")
    print("=" * 60)

    # Setup
    helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)

    # Create LLM client (will use router adapter)
    try:
        llm_client = create_router_adapter()
        print("\nLLM client initialized successfully")
    except Exception as e:
        print(f"\nNote: Could not initialize LLM client: {e}")
        print("Using mock client for demo purposes")
        from unittest.mock import Mock
        llm_client = Mock()

    # Get registry
    registry = get_global_registry()

    # Create research agent
    print("\nCreating research agent...")
    try:
        research = registry.create_agent(
            agent_type="research",
            agent_id="demo_research_001",
            spawn_time=0.1,
            helix=helix,
            llm_client=llm_client,
            research_domain="technical"
        )
        print(f"  ✓ Created: {research.agent_id}")
        print(f"    Type: {research.agent_type}")
        print(f"    Spawn time: {research.spawn_time}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Create analysis agent
    print("\nCreating analysis agent...")
    try:
        analysis = registry.create_agent(
            agent_type="analysis",
            agent_id="demo_analysis_001",
            spawn_time=0.4,
            helix=helix,
            llm_client=llm_client,
            analysis_type="technical"
        )
        print(f"  ✓ Created: {analysis.agent_id}")
        print(f"    Type: {analysis.agent_type}")
        print(f"    Spawn time: {analysis.spawn_time}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def demo_agent_factory():
    """Demonstrate AgentFactory with plugin system."""
    print("\n" + "=" * 60)
    print("4. AgentFactory with Plugin System")
    print("=" * 60)

    # Setup
    helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)

    try:
        llm_client = create_router_adapter()
    except Exception:
        from unittest.mock import Mock
        llm_client = Mock()

    # Create factory (automatically uses global registry)
    factory = AgentFactory(
        helix=helix,
        llm_client=llm_client,
        enable_dynamic_spawning=False  # Disable for demo
    )

    print("\nAgentFactory initialized with plugin registry")

    # List available agent types
    print("\nAvailable agent types via factory:")
    for agent_type in factory.list_available_agent_types():
        print(f"  - {agent_type}")

    # Create agents using new create_agent_by_type method
    print("\nCreating agents via factory...")
    try:
        # Create research agent
        research = factory.create_agent_by_type(
            agent_type="research",
            complexity="medium",
            research_domain="general"
        )
        print(f"  ✓ Research agent: {research.agent_id}")

        # Create critic agent
        critic = factory.create_agent_by_type(
            agent_type="critic",
            complexity="complex",
            review_focus="quality"
        )
        print(f"  ✓ Critic agent: {critic.agent_id}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # Get suitable agents for a task
    print("\nFinding suitable agents for task...")
    task = "Review Python code for security vulnerabilities"
    suitable = factory.get_suitable_agents_for_task(task, "complex")
    print(f"  Task: {task}")
    print(f"  Suitable agents: {', '.join(suitable)}")


def demo_task_filtering():
    """Demonstrate task-based agent filtering."""
    print("\n" + "=" * 60)
    print("5. Task-Based Agent Filtering")
    print("=" * 60)

    registry = get_global_registry()

    test_tasks = [
        ("Research quantum computing", "medium"),
        ("Analyze customer feedback data", "complex"),
        ("Review code for bugs", "medium"),
        ("Find latest AI research papers", "simple")
    ]

    for task, complexity in test_tasks:
        suitable = registry.get_agents_for_task(task, complexity)
        print(f"\nTask: '{task}' ({complexity})")
        print(f"  Suitable agents: {', '.join(suitable[:3])}")  # Show top 3


def demo_statistics():
    """Show registry statistics."""
    print("\n" + "=" * 60)
    print("6. Registry Statistics")
    print("=" * 60)

    registry = get_global_registry()

    stats = registry.get_statistics()
    print("\nRegistry Statistics:")
    print(f"  Total plugins loaded: {stats['plugins_loaded']}")
    print(f"  Builtin plugins: {stats['builtin_count']}")
    print(f"  External plugins: {stats['external_count']}")
    print(f"  Failed loads: {stats['plugins_failed']}")
    print(f"  Total registered: {stats['total_registered']}")
    print(f"  Agent types: {', '.join(stats['agent_types'])}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "Felix Agent Plugin System Demo" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        demo_basic_registry()
        demo_custom_plugins()
        demo_agent_creation()
        demo_agent_factory()
        demo_task_filtering()
        demo_statistics()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. See docs/PLUGIN_API.md for full plugin documentation")
        print("  2. See examples/custom_agents/README.md for creating custom agents")
        print("  3. Try creating your own agent plugin!")
        print()

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
