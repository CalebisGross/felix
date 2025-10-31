"""
Felix Command-Line Interface

Provides command-line access to Felix functionality without requiring the GUI.

Usage:
    felix run "Your task here"
    felix status
    felix test-connection
    felix gui
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional


def cmd_run(args):
    """Run a Felix workflow from command line."""
    from src.core.helix_geometry import HelixGeometry
    from src.communication.central_post import CentralPost, AgentFactory
    from src.llm.router_adapter import create_router_adapter
    from src.workflows.felix_workflow import execute_linear_workflow_optimized

    print(f"🚀 Felix Workflow")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Initialize components
    try:
        helix = HelixGeometry(
            top_radius=3.0,
            bottom_radius=0.5,
            height=8.0,
            turns=2
        )

        llm_client = create_router_adapter(args.config)
        central_post = CentralPost(helix)
        agent_factory = AgentFactory(central_post, helix, llm_client)

        print("✓ Felix system initialized\n")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("Make sure at least one LLM provider is configured.")
        return 1

    # Create simple system object
    class SimpleSystem:
        def __init__(self):
            self.helix = helix
            self.central_post = central_post
            self.agent_factory = agent_factory
            self.lm_client = llm_client

            class Config:
                workflow_max_steps = args.max_steps
                enable_web_search = args.web_search
                workflow_simple_threshold = 0.8
                workflow_medium_threshold = 0.6
                workflow_max_steps_simple = 3
                workflow_max_steps_medium = 5
                workflow_max_steps_complex = 10
                confidence_threshold = 0.80

            self.config = Config()
            self.task_memory = None

    system = SimpleSystem()

    # Run workflow
    print("🤖 Running workflow...\n")

    try:
        result = execute_linear_workflow_optimized(
            task_input=args.task,
            felix_system=system,
            max_steps_override=args.max_steps
        )

        # Extract results
        synthesis = result.get("centralpost_synthesis", {})
        content = synthesis.get("synthesis_content", "No result generated")
        confidence = synthesis.get("confidence", 0.0)
        agents_spawned = result.get("agents_spawned", [])

        # Display
        print("=" * 60)
        print("📝 RESULT")
        print("=" * 60)
        print()
        print(content)
        print()
        print("=" * 60)
        print(f"📊 Confidence: {confidence:.2f}")
        print(f"🤖 Agents: {len(agents_spawned)}")
        print(f"⏱️  Steps: {result.get('steps_executed', 0)}")

        # Save to file if requested
        if args.output:
            output_data = {
                "task": args.task,
                "result": content,
                "confidence": confidence,
                "agents_count": len(agents_spawned),
                "agents": agents_spawned
            }

            with open(args.output, 'w') as f:
                if args.output.endswith('.json'):
                    json.dump(output_data, f, indent=2)
                else:
                    f.write(content)

            print(f"💾 Saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1


def cmd_status(args):
    """Check Felix system status."""
    print("Felix System Status")
    print("=" * 60)

    # Check LLM providers
    print("\n🔌 LLM Providers:")
    try:
        from src.llm.router_adapter import create_router_adapter

        adapter = create_router_adapter(args.config)
        router = adapter.router

        # Test connections
        results = router.test_all_connections()
        for provider, status in results.items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {provider}")

        # Show statistics
        stats = router.get_statistics()
        if stats['total_requests'] > 0:
            print(f"\n📊 Router Statistics:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Success rate: {stats['overall_success_rate']:.1%}")

    except Exception as e:
        print(f"  ✗ Could not initialize router: {e}")

    # Check databases
    print("\n💾 Databases:")
    db_files = [
        "felix_knowledge.db",
        "felix_workflow_history.db",
        "felix_memory.db",
        "felix_task_memory.db",
        "felix_system_actions.db"
    ]

    for db_file in db_files:
        if Path(db_file).exists():
            size = Path(db_file).stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {db_file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {db_file} (not found)")

    # Check knowledge stats
    print("\n📚 Knowledge:")
    try:
        import sqlite3
        conn = sqlite3.connect("felix_knowledge.db")
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entries = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM document_sources")
        docs = cursor.fetchone()[0]

        conn.close()

        print(f"  Entries: {entries}")
        print(f"  Documents: {docs}")

    except Exception as e:
        print(f"  Could not read knowledge database")

    print("\n" + "=" * 60)
    return 0


def cmd_test_connection(args):
    """Test LLM provider connection."""
    print("Testing LLM Connection")
    print("=" * 60)

    try:
        from src.llm.router_adapter import create_router_adapter

        print("Initializing router...")
        adapter = create_router_adapter(args.config)
        router = adapter.router

        print(f"Primary provider: {router.get_primary_provider().get_provider_name()}")

        # Test primary
        print("\nTesting primary provider...")
        if adapter.test_connection():
            print("✓ Primary provider connected")
        else:
            print("✗ Primary provider failed")

        # Test all
        print("\nTesting all providers...")
        results = router.test_all_connections()
        for provider, status in results.items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {provider}")

        print("\n" + "=" * 60)
        return 0

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_gui(args):
    """Launch Felix GUI."""
    print("Launching Felix GUI...")
    try:
        from src.gui import main as gui_main
        gui_main.main()
        return 0
    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return 1


def cmd_init(args):
    """Initialize Felix databases."""
    print("Initializing Felix databases...")
    try:
        from src.migration.version_manager import VersionManager
        manager = VersionManager()
        manager.initialize_databases()
        print("✓ Databases initialized")
        return 0
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Felix Framework - Multi-Agent AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  felix run "Explain quantum computing"
  felix run "Design a REST API" --max-steps 10 --output result.md
  felix status
  felix test-connection
  felix gui
        """
    )

    parser.add_argument('--version', action='version', version='Felix 0.9.0')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run a workflow')
    run_parser.add_argument('task', help='Task description')
    run_parser.add_argument('--max-steps', type=int, default=10,
                           help='Maximum workflow steps (default: 10)')
    run_parser.add_argument('--output', '-o', help='Output file (txt, md, or json)')
    run_parser.add_argument('--config', default='config/llm.yaml',
                           help='LLM config file (default: config/llm.yaml)')
    run_parser.add_argument('--web-search', action='store_true',
                           help='Enable web search')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    run_parser.set_defaults(func=cmd_run)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--config', default='config/llm.yaml',
                              help='LLM config file')
    status_parser.set_defaults(func=cmd_status)

    # Test connection command
    test_parser = subparsers.add_parser('test-connection',
                                        help='Test LLM connection')
    test_parser.add_argument('--config', default='config/llm.yaml',
                            help='LLM config file')
    test_parser.add_argument('--verbose', '-v', action='store_true')
    test_parser.set_defaults(func=cmd_test_connection)

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch GUI')
    gui_parser.set_defaults(func=cmd_gui)

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize databases')
    init_parser.set_defaults(func=cmd_init)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
