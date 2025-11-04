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
    from src.gui.felix_system import FelixSystem, FelixConfig
    from src.workflows.felix_workflow import run_felix_workflow

    print(f"🚀 Felix Workflow")
    print(f"Task: {args.task}")
    print("=" * 60)

    # Create Felix configuration from CLI arguments
    config = FelixConfig(
        # Disable features not needed for CLI
        enable_streaming=True,  # Enable streaming for faster LLM response
        enable_spoke_topology=True,  # Enable full topology for proper agent management
        enable_memory=True,  # Enable memory for persistence
        enable_compression=True,  # Enable context compression
        enable_knowledge_brain=False,  # Disable knowledge brain for CLI speed
        knowledge_daemon_enabled=False,  # No background daemon for CLI

        # Web search configuration
        web_search_enabled=args.web_search,

        # Workflow configuration from args
        workflow_max_steps_simple=3,
        workflow_max_steps_medium=5,
        workflow_max_steps_complex=args.max_steps,
        workflow_simple_threshold=0.8,
        workflow_medium_threshold=0.6,

        # Agent and token configuration
        max_agents=25,
        base_token_budget=2500,

        # LLM configuration (will use config/llm.yaml)
        verbose_llm_logging=args.verbose
    )

    # Initialize Felix system
    felix_system = FelixSystem(config)

    try:
        print("Initializing Felix system...")
        if not felix_system.start():
            print("❌ Failed to start Felix system")
            print("Make sure at least one LLM provider is configured.")
            return 1

        print("✓ Felix system initialized\n")

        # Run workflow
        print("🤖 Running workflow...\n")

        result = run_felix_workflow(
            felix_system=felix_system,
            task_input=args.task,
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

    finally:
        # Always cleanup Felix system
        if felix_system.running:
            print("\n🛑 Shutting down Felix system...")
            felix_system.stop()


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


def cmd_chat(args):
    """Start conversational CLI interface."""
    from src.gui.felix_system import FelixSystem, FelixConfig
    from src.cli_chat import run_chat, run_single_query
    from src.cli_chat.session_manager import SessionManager

    # Check for print mode or piped input
    is_piped = not sys.stdin.isatty()
    is_print_mode = args.print_mode or is_piped

    # Get query from argument or stdin
    query = None
    if is_print_mode:
        if args.query:
            query = args.query
        elif is_piped:
            query = sys.stdin.read().strip()
        else:
            print("Error: Print mode (-p) requires a query argument or piped input")
            return 1

        if not query:
            print("Error: Empty query")
            return 1

    # Print header only in interactive mode
    if not is_print_mode:
        print("🚀 Felix Conversational CLI")
        print("=" * 60)

    # Determine session ID (only for interactive mode)
    session_id = None
    if not is_print_mode:
        session_id = args.resume

        # Handle -c / --continue flag
        if args.continue_last:
            manager = SessionManager()
            last_session = manager.get_last_session()

            if last_session:
                session_id = last_session
                print(f"Continuing last session: {session_id}")
            else:
                print("No previous sessions found. Starting new session...")

    # Create Felix configuration
    config = FelixConfig(
        enable_streaming=True,  # Enable streaming for faster LLM response
        enable_spoke_topology=True,
        enable_memory=True,
        enable_compression=True,
        enable_knowledge_brain=args.knowledge_brain,
        knowledge_daemon_enabled=False,
        web_search_enabled=args.web_search,
        auto_approve_system_actions=True,  # CLI mode: auto-approve commands without blocking
        max_agents=25,
        base_token_budget=2500,
        verbose_llm_logging=args.verbose
    )

    # Initialize Felix system
    felix_system = FelixSystem(config)

    try:
        if not is_print_mode:
            print("Initializing Felix system...")

        if not felix_system.start():
            if not is_print_mode:
                print("❌ Failed to start Felix system")
                print("Make sure at least one LLM provider is configured.")
            return 1

        if not is_print_mode:
            print("✓ Felix system initialized\n")

        # Run in appropriate mode
        if is_print_mode:
            # Print mode: single query, output result, exit
            result = run_single_query(
                query=query,
                felix_system=felix_system,
                enable_nl=not args.no_nl,
                verbose=args.verbose
            )
            print(result)
        else:
            # Interactive mode: full chat REPL
            run_chat(
                felix_system=felix_system,
                session_id=session_id,
                enable_nl=not args.no_nl,
                verbose=args.verbose
            )

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130

    except Exception as e:
        print(f"❌ Chat failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # Always cleanup Felix system
        if felix_system.running:
            print("\n🛑 Shutting down Felix system...")
            felix_system.stop()


def cmd_sessions(args):
    """Manage chat sessions."""
    from src.cli_chat.session_manager import SessionManager
    from src.cli_chat.formatters import OutputFormatter

    manager = SessionManager()
    formatter = OutputFormatter()

    if args.action == 'list':
        # List sessions
        sessions = manager.list_sessions(limit=args.limit)

        if not sessions:
            print("No chat sessions found")
            return 0

        formatter.print_session_list([s.to_dict() for s in sessions])

    elif args.action == 'recent':
        # List recent sessions (alias for list)
        sessions = manager.list_sessions(limit=args.limit)

        if not sessions:
            print("No recent chat sessions found")
            return 0

        print(f"Recent sessions (last {args.limit}):")
        print("=" * 80)
        for session in sessions:
            title_str = f" - {session.title}" if session.title else ""
            tags_str = f" {session.tags}" if session.tags else ""
            print(f"{session.session_id}: {session.last_active.strftime('%Y-%m-%d %H:%M')}{title_str}{tags_str}")

    elif args.action == 'today':
        # List today's sessions
        sessions = manager.get_sessions_today()

        if not sessions:
            print("No sessions active today")
            return 0

        print(f"Sessions active today:")
        print("=" * 80)
        for session in sessions:
            title_str = f" - {session.title}" if session.title else ""
            tags_str = f" {session.tags}" if session.tags else ""
            print(f"{session.session_id}: {session.last_active.strftime('%H:%M')}{title_str}{tags_str}")

    elif args.action == 'show':
        # Show session details
        if not args.session_id:
            print("Error: session_id required for 'show' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        # Print session info
        print(f"\nSession: {session.session_id}")
        print(f"Created: {session.created_at}")
        print(f"Last active: {session.last_active}")
        print(f"Messages: {session.message_count}")
        if session.title:
            print(f"Title: {session.title}")
        if session.tags:
            print(f"Tags: {', '.join(session.tags)}")
        print()

        # Print messages
        messages = manager.get_messages(args.session_id, limit=args.limit)
        for msg in messages:
            role_label = f"[{msg.role}]"
            print(f"\n{role_label} {msg.timestamp}")
            print(msg.content[:200] + "..." if len(msg.content) > 200 else msg.content)

    elif args.action == 'search':
        # Search sessions
        if not args.query:
            print("Error: query required for 'search' action")
            return 1

        sessions = manager.search_sessions(args.query, limit=args.limit)

        if not sessions:
            print(f"No sessions found matching '{args.query}'")
            return 0

        print(f"Sessions matching '{args.query}':")
        print("=" * 80)
        for session in sessions:
            title_str = f" - {session.title}" if session.title else ""
            tags_str = f" {session.tags}" if session.tags else ""
            print(f"{session.session_id}: {session.last_active.strftime('%Y-%m-%d %H:%M')}{title_str}{tags_str}")

    elif args.action == 'rename':
        # Rename session (set title)
        if not args.session_id:
            print("Error: session_id required for 'rename' action")
            return 1

        if not args.title:
            print("Error: title required for 'rename' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        manager.set_title(args.session_id, args.title)
        print(f"✓ Session {args.session_id} renamed to: {args.title}")

    elif args.action == 'tag':
        # Add tags to session
        if not args.session_id:
            print("Error: session_id required for 'tag' action")
            return 1

        if not args.tags:
            print("Error: at least one tag required for 'tag' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        manager.add_tags(args.session_id, args.tags)
        print(f"✓ Added tags to session {args.session_id}: {', '.join(args.tags)}")

    elif args.action == 'untag':
        # Remove tags from session
        if not args.session_id:
            print("Error: session_id required for 'untag' action")
            return 1

        if not args.tags:
            print("Error: at least one tag required for 'untag' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        manager.remove_tags(args.session_id, args.tags)
        print(f"✓ Removed tags from session {args.session_id}: {', '.join(args.tags)}")

    elif args.action == 'export':
        # Export session to JSON
        if not args.session_id:
            print("Error: session_id required for 'export' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        data = manager.export_session(args.session_id)

        output_file = args.output or f"session_{args.session_id}.json"

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Session {args.session_id} exported to: {output_file}")

    elif args.action == 'import':
        # Import session from JSON
        if not args.input_file:
            print("Error: input file required for 'import' action")
            return 1

        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)

            new_session_id = manager.import_session(data)

            if new_session_id:
                print(f"✓ Session imported with ID: {new_session_id}")
            else:
                print("❌ Failed to import session (invalid format)")
                return 1

        except FileNotFoundError:
            print(f"Error: File not found: {args.input_file}")
            return 1
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file: {args.input_file}")
            return 1

    elif args.action == 'delete':
        # Delete session
        if not args.session_id:
            print("Error: session_id required for 'delete' action")
            return 1

        session = manager.get_session(args.session_id)
        if not session:
            print(f"Session not found: {args.session_id}")
            return 1

        # Confirm deletion
        if not args.force:
            response = input(f"Delete session {args.session_id}? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled")
                return 0

        manager.delete_session(args.session_id)
        print(f"✓ Session {args.session_id} deleted")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Felix Framework - Multi-Agent AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-off workflow execution
  felix run "Explain quantum computing"
  felix run "Design a REST API" --max-steps 10 --output result.md

  # Interactive chat mode
  felix chat                                    # Start new chat session
  felix chat -c                                 # Continue last session
  felix chat --resume abc123                    # Resume specific session

  # Print mode (non-interactive, single query)
  felix chat -p "What is helical geometry?"     # Single query execution
  echo "Analyze this" | felix chat              # Piped input (auto print mode)
  felix chat "Quick question" -p                # Can also put query first

  # Session management
  felix sessions list                           # List all sessions
  felix sessions recent                         # List recent sessions
  felix sessions today                          # List today's sessions
  felix sessions show abc123                    # Show session details
  felix sessions search "keyword"               # Search sessions by keyword
  felix sessions rename abc123 --title "New"    # Rename a session
  felix sessions tag abc123 --tags work urgent  # Add tags to session
  felix sessions untag abc123 --tags urgent     # Remove tags from session
  felix sessions export abc123 -o session.json  # Export session to JSON
  felix sessions import session.json            # Import session from JSON
  felix sessions delete abc123                  # Delete a session

  # System commands
  felix status                                  # Check system status
  felix test-connection                         # Test LLM providers
  felix gui                                     # Launch GUI
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

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start conversational CLI')
    chat_parser.add_argument('query', nargs='?', help='Query for print mode (single execution)')
    chat_parser.add_argument('--resume', help='Resume an existing session by ID')
    chat_parser.add_argument('-c', '--continue', dest='continue_last', action='store_true',
                            help='Continue last active session')
    chat_parser.add_argument('-p', '--print', dest='print_mode', action='store_true',
                            help='Print mode: execute single query and exit (non-interactive)')
    chat_parser.add_argument('--no-nl', action='store_true',
                            help='Disable natural language mode (explicit commands only)')
    chat_parser.add_argument('--knowledge-brain', action='store_true',
                            help='Enable knowledge brain')
    chat_parser.add_argument('--web-search', action='store_true',
                            help='Enable web search')
    chat_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose output')
    chat_parser.set_defaults(func=cmd_chat)

    # Sessions command
    sessions_parser = subparsers.add_parser('sessions', help='Manage chat sessions')
    sessions_parser.add_argument('action',
                                choices=['list', 'recent', 'today', 'show', 'search',
                                        'rename', 'tag', 'untag', 'export', 'import', 'delete'],
                                help='Action to perform')
    sessions_parser.add_argument('session_id', nargs='?',
                                help='Session ID (required for show/delete/rename/tag/untag/export)')
    sessions_parser.add_argument('--query', help='Search query (for search action)')
    sessions_parser.add_argument('--title', help='New title (for rename action)')
    sessions_parser.add_argument('--tags', nargs='+', help='Tags (for tag/untag actions)')
    sessions_parser.add_argument('--output', '-o', help='Output file (for export action)')
    sessions_parser.add_argument('--input-file', help='Input file (for import action)')
    sessions_parser.add_argument('--limit', type=int, default=20,
                                help='Limit number of items shown')
    sessions_parser.add_argument('--force', '-f', action='store_true',
                                help='Force delete without confirmation')
    sessions_parser.set_defaults(func=cmd_sessions)

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
