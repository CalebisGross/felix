"""
Felix Conversational CLI

Interactive chat interface for Felix with session management and natural language support.
"""

import sys
from typing import Optional
from pathlib import Path

from .session_manager import SessionManager, Message
from .formatters import get_formatter
from .command_handler import CommandHandler
from .completers import create_felix_completer
from .tools import (
    ToolRegistry,
    WorkflowTool,
    HistoryTool,
    KnowledgeTool,
    AgentTool,
    SystemTool,
    DocumentTool
)

# Try to import prompt_toolkit for enhanced input handling
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.formatted_text import HTML
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


class FelixChat:
    """Main conversational CLI interface for Felix."""

    def __init__(
        self,
        felix_system=None,
        session_id: Optional[str] = None,
        enable_nl: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Felix chat interface.

        Args:
            felix_system: Initialized FelixSystem instance
            session_id: Existing session ID to resume, or None for new session
            enable_nl: Enable natural language processing
            verbose: Verbose output
        """
        self.felix_system = felix_system
        self.session_manager = SessionManager()
        self.formatter = get_formatter(use_rich=True)  # Use rich formatter if available
        self.verbose = verbose
        self.enable_nl = enable_nl

        # Resume or create session
        if session_id:
            self.session = self.session_manager.get_session(session_id)
            if not self.session:
                raise ValueError(f"Session not found: {session_id}")
            self.session_id = session_id
        else:
            self.session_id = self.session_manager.create_session()
            self.session = self.session_manager.get_session(self.session_id)

        # Build Felix context for tools
        self.felix_context = {
            'felix_system': felix_system,
            'session_id': self.session_id,
            'session_manager': self.session_manager,  # Add for orchestrator
            'verbose': verbose,
            'last_workflow_id': None,
            'formatter': self.formatter  # Add formatter for tool use
        }

        # Add Felix components to context if system is available
        if felix_system:
            self.felix_context['knowledge_store'] = getattr(felix_system, 'knowledge_store', None)
            self.felix_context['llm_adapter'] = getattr(felix_system, 'llm_client', None)
            self.felix_context['knowledge_retriever'] = getattr(felix_system, 'knowledge_retriever', None)

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._register_tools()

        # Initialize command handler
        self.command_handler = CommandHandler(
            tool_registry=self.tool_registry,
            formatter=self.formatter,
            felix_context=self.felix_context,
            enable_nl=enable_nl
        )

        self.running = False
        self.multiline_mode = False

        # Initialize prompt session if prompt_toolkit is available
        if HAS_PROMPT_TOOLKIT:
            # Create completer with tool registry and custom commands
            completer = create_felix_completer(
                tool_registry=self.tool_registry,
                custom_commands=self.command_handler.custom_commands
            )

            self.prompt_session = PromptSession(
                history=FileHistory(str(Path.home() / '.felix_history')),
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=self._create_key_bindings(),
                enable_history_search=True,
                multiline=False,  # Will be toggled by keybinding
                completer=completer,  # Add tab completion
                complete_while_typing=False,  # Only complete on Tab press
            )
        else:
            self.prompt_session = None

    def _register_tools(self):
        """Register all available tools."""
        self.tool_registry.register(WorkflowTool(self.felix_context))
        self.tool_registry.register(HistoryTool(self.felix_context))
        self.tool_registry.register(KnowledgeTool(self.felix_context))
        self.tool_registry.register(AgentTool(self.felix_context))
        self.tool_registry.register(SystemTool(self.felix_context))
        self.tool_registry.register(DocumentTool(self.felix_context))

    def _create_key_bindings(self):
        """Create keyboard shortcuts for enhanced input handling."""
        if not HAS_PROMPT_TOOLKIT:
            return None

        kb = KeyBindings()

        @kb.add('c-l')
        def clear_screen(event):
            """Ctrl+L: Clear screen but preserve history."""
            event.app.renderer.clear()

        @kb.add('escape', 'escape')
        def handle_double_escape(event):
            """Esc+Esc: Rewind/undo (future feature)."""
            # For now, just show info message
            print("\n(Rewind feature coming soon - will undo last workflow)")

        @kb.add('c-d')
        def handle_eof(event):
            """Ctrl+D: Exit chat gracefully."""
            self.running = False
            event.app.exit()

        return kb

    def _get_user_input(self) -> Optional[str]:
        """
        Get user input with enhanced features if prompt_toolkit is available.

        Returns:
            User input string, or None if EOF (Ctrl+D)
        """
        if self.prompt_session:
            # Use prompt_toolkit for rich input
            try:
                user_input = self.prompt_session.prompt(
                    HTML('<ansiCyan>felix&gt; </ansiCyan>'),
                    multiline=self.multiline_mode
                )
                return user_input.strip()
            except EOFError:
                return None
            except KeyboardInterrupt:
                print()  # New line after Ctrl+C
                return ""  # Empty string signals continue
        else:
            # Fallback to basic input
            try:
                user_input = input(self.formatter._color("felix> ", "cyan"))
                return user_input.strip()
            except EOFError:
                return None
            except KeyboardInterrupt:
                print()
                return ""

    def _handle_special_prefixes(self, user_input: str) -> str:
        """
        Handle special input prefixes: !, @, #

        Args:
            user_input: Raw user input

        Returns:
            Processed input (may be modified or expanded)
        """
        import subprocess
        import os

        # !command - Execute bash command with trust-based approval
        if user_input.startswith('!'):
            command = user_input[1:].strip()

            if not command:
                self.formatter.print_error("Empty command after !")
                return ""

            try:
                # Route through trust manager for security classification
                from src.execution.trust_manager import TrustLevel
                trust_manager = self.orchestrator.felix_system.trust_manager
                trust_level = trust_manager.classify_command(command)

                # BLOCKED commands cannot execute
                if trust_level == TrustLevel.BLOCKED:
                    self.formatter.print_error(f"❌ Command blocked for security")
                    self.formatter.print_info(f"   Command: {command}")
                    self.formatter.print_info(f"   Reason: This command is classified as dangerous")
                    return ""

                # REVIEW commands require user approval
                if trust_level == TrustLevel.REVIEW:
                    self.formatter.print_warning("\n⚠️  Command Requires Approval")
                    self.formatter.print_info(f"   Command: {command}")
                    self.formatter.print_info(f"   Risk level: {trust_level.value}")
                    print()

                    response = input("Execute this command? [y/N]: ").strip().lower()
                    if response != 'y':
                        self.formatter.print_info("Command cancelled by user")
                        return ""

                # Execute (either SAFE or user-approved REVIEW)
                self.formatter.print_info(f"Executing: {command}")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.getcwd()
                )

                output = f"Command: {command}\n"
                if result.stdout:
                    output += f"Output:\n{result.stdout}\n"
                if result.stderr:
                    output += f"Errors:\n{result.stderr}\n"
                output += f"Exit code: {result.returncode}"

                # Return the output as context for the next query
                self.formatter.print_dim("Command executed. You can now ask about the output.")
                print()
                print(output)
                print()
                return "/workflow run Analyze the command output above"

            except subprocess.TimeoutExpired:
                self.formatter.print_error(f"Command timed out after 30s")
                return ""
            except Exception as e:
                self.formatter.print_error(f"Command failed: {e}")
                return ""

        # @filepath - Read file and include in context
        elif user_input.startswith('@'):
            filepath = user_input[1:].strip()

            if not filepath:
                self.formatter.print_error("Empty file path after @")
                return ""

            try:
                with open(filepath, 'r') as f:
                    content = f.read()

                file_info = f"File: {filepath}\n```\n{content}\n```"
                self.formatter.print_info(f"Read file: {filepath} ({len(content)} chars)")
                print()
                print(file_info[:500] + "..." if len(file_info) > 500 else file_info)
                print()

                # Return a workflow command that includes the file content
                return f"/workflow run Analyze the file content from {filepath}"

            except FileNotFoundError:
                self.formatter.print_error(f"File not found: {filepath}")
                return ""
            except Exception as e:
                self.formatter.print_error(f"Failed to read file: {e}")
                return ""

        # #note - Quick memory append (future feature)
        elif user_input.startswith('#'):
            note = user_input[1:].strip()

            if not note:
                self.formatter.print_error("Empty note after #")
                return ""

            # For now, just show the note
            self.formatter.print_info("Memory note feature coming soon")
            self.formatter.print_dim(f"Note: {note}")
            print()

            # Could save to a FELIX.md file or database
            # For now, just return empty to continue
            return ""

        # No special prefix, return as-is
        return user_input

    def start(self):
        """Start the interactive chat loop."""
        self.running = True

        # Print welcome message
        self.formatter.print_welcome(self.session_id)

        # Show recent messages if resuming session
        if self.session.message_count > 0:
            self._show_recent_context()

        # Show prompt_toolkit status
        if HAS_PROMPT_TOOLKIT:
            self.formatter.print_dim("Enhanced input mode: Ctrl+R (search), Ctrl+L (clear), Ctrl+D (exit)")
        else:
            self.formatter.print_dim("Basic input mode. Install prompt_toolkit for enhanced features:")
            self.formatter.print_dim("  pip install -r requirements-cli-enhanced.txt")

        print()

        # Main loop
        try:
            while self.running:
                # Get user input
                user_input = self._get_user_input()

                # Handle EOF (Ctrl+D)
                if user_input is None:
                    break

                # Handle empty input or interrupt (Ctrl+C)
                if not user_input:
                    continue

                # Handle special prefixes (!command, @file, #note)
                processed_input = self._handle_special_prefixes(user_input)

                # If prefix handler returns empty, continue to next iteration
                if not processed_input:
                    continue

                # Use processed input for the rest
                user_input = processed_input

                # Check for exit commands
                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    self.running = False
                    continue

                # Check for help command
                if user_input.lower() in ['/help', 'help']:
                    self._show_help()
                    continue

                # Save user message
                self.session_manager.add_message(
                    session_id=self.session_id,
                    role='user',
                    content=user_input
                )

                # Handle the input
                result = self.command_handler.handle(user_input)

                # Display result
                if result.success:
                    print()
                    if result.content:
                        self.formatter.print_markdown(result.content)
                    print()

                    # Save assistant response
                    self.session_manager.add_message(
                        session_id=self.session_id,
                        role='assistant',
                        content=result.content,
                        workflow_id=result.workflow_id
                    )

                    # Update last workflow ID in context
                    if result.workflow_id:
                        self.felix_context['last_workflow_id'] = result.workflow_id

                else:
                    self.formatter.print_error(result.error or "Command failed")

        except Exception as e:
            self.formatter.print_error(f"Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

        finally:
            # Show goodbye message
            self.formatter.print_goodbye()

    def _show_recent_context(self):
        """Show recent messages from the session."""
        messages = self.session_manager.get_recent_context(
            session_id=self.session_id,
            message_count=5
        )

        if messages:
            self.formatter.print_info(f"Resuming session (last {len(messages)} messages):")
            print()

            for msg in messages:
                role_label = self.formatter._color(f"[{msg.role}]", "yellow" if msg.role == "user" else "green")
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"{role_label} {content_preview}")

            print()

    def _show_help(self):
        """Display help information."""
        help_text = self.tool_registry.get_help_text()

        self.formatter.print_header("Felix Chat - Help")
        print(help_text)

        print("\nGeneral commands:")
        print("  /help                    Show this help message")
        print("  /exit or /quit           Exit the chat")
        print()

        print("Special input prefixes:")
        print("  !<command>               Execute bash command and capture output")
        print("                           Example: !ls -la")
        print("  @<filepath>              Read file and include in context")
        print("                           Example: @src/main.py")
        print("  #<note>                  Quick memory note (coming soon)")
        print("                           Example: #remember to add tests")
        print()

        if HAS_PROMPT_TOOLKIT:
            print("Keyboard shortcuts:")
            print("  Ctrl+R                   Reverse search command history")
            print("  Ctrl+L                   Clear screen (preserves history)")
            print("  Ctrl+D                   Exit chat")
            print("  Ctrl+C                   Cancel current input")
            print("  Esc+Esc                  Rewind/undo (coming soon)")
            print("  ↑/↓                      Navigate command history")
            print()

        if self.enable_nl:
            print("Natural Language Mode: Enabled")
            print("  You can type questions naturally, and Felix will interpret them.")
            print("  Example: 'What does the system know about AI?'")
            print()
        else:
            print("Natural Language Mode: Disabled")
            print("  Use explicit /commands for all interactions.")
            print()


def run_chat(
    felix_system=None,
    session_id: Optional[str] = None,
    enable_nl: bool = True,
    verbose: bool = False
):
    """
    Run the conversational CLI.

    Args:
        felix_system: Initialized FelixSystem instance
        session_id: Session ID to resume
        enable_nl: Enable natural language processing
        verbose: Verbose output
    """
    try:
        chat = FelixChat(
            felix_system=felix_system,
            session_id=session_id,
            enable_nl=enable_nl,
            verbose=verbose
        )
        chat.start()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def run_single_query(
    query: str,
    felix_system=None,
    enable_nl: bool = True,
    verbose: bool = False
) -> str:
    """
    Run a single query in non-interactive (print) mode.

    This is useful for piped input, scripting, and one-off queries.

    Args:
        query: The query string to process
        felix_system: Initialized FelixSystem instance
        enable_nl: Enable natural language processing
        verbose: Verbose output

    Returns:
        Result string from the query

    Example:
        echo "What is quantum computing?" | felix chat -p
        felix chat -p "Explain helical geometry"
    """
    try:
        # Create temporary session (won't be saved if not needed)
        chat = FelixChat(
            felix_system=felix_system,
            session_id=None,  # New session
            enable_nl=enable_nl,
            verbose=verbose
        )

        # Process the query
        result = chat.command_handler.handle(query)

        if result.success:
            return result.content
        else:
            error_msg = result.error or "Query failed"
            if verbose:
                raise RuntimeError(error_msg)
            else:
                return f"Error: {error_msg}"

    except Exception as e:
        error_msg = f"Error: {e}"
        if verbose:
            import traceback
            error_msg += f"\n\n{traceback.format_exc()}"
        return error_msg


if __name__ == "__main__":
    # Simple test mode
    print("Felix Chat - Standalone Test Mode")
    print("(No Felix system initialized - limited functionality)")
    print()

    run_chat(felix_system=None, enable_nl=False, verbose=True)
