"""
Command Auto-completion for Felix CLI

Provides tab-completion for commands, arguments, and file paths.
"""

from typing import List, Optional, Iterable
from pathlib import Path

try:
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.document import Document
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    # Fallback classes if prompt_toolkit not installed
    HAS_PROMPT_TOOLKIT = False

    class Completion:
        def __init__(self, text, start_position=0, display=None, display_meta=None):
            self.text = text
            self.start_position = start_position
            self.display = display or text
            self.display_meta = display_meta or ''

    class Document:
        def __init__(self, text='', cursor_position=0):
            self.text = text
            self.cursor_position = cursor_position

        def get_word_before_cursor(self, WORD=False):
            text = self.text[:self.cursor_position]
            return text.split()[-1] if text.split() else ''

    class Completer:
        def get_completions(self, document, complete_event):
            return []


class FelixCompleter(Completer):
    """
    Tab completion for Felix CLI commands.

    Provides completions for:
    - Built-in commands (/workflow, /history, etc.)
    - Custom commands (.felix/commands/*)
    - Command arguments
    - File paths
    """

    def __init__(self, tool_registry=None, custom_commands=None):
        """
        Initialize completer.

        Args:
            tool_registry: ToolRegistry instance for built-in commands
            custom_commands: CustomCommandLoader instance
        """
        self.tool_registry = tool_registry
        self.custom_commands = custom_commands

        # Cache of commands
        self._command_cache = None
        self._last_refresh = 0

    def _get_commands(self) -> List[tuple]:
        """
        Get list of available commands.

        Returns:
            List of (command_name, description) tuples
        """
        commands = []

        # Built-in tools
        if self.tool_registry:
            for tool in self.tool_registry.list_tools():
                commands.append((tool.name, tool.description))

        # Custom commands
        if self.custom_commands:
            for custom_cmd in self.custom_commands.list_commands():
                commands.append((custom_cmd.name, f"[custom] {custom_cmd.description}"))

        # Common meta-commands
        commands.extend([
            ('help', 'Show available commands'),
            ('exit', 'Exit chat'),
            ('quit', 'Exit chat'),
            ('clear', 'Clear screen'),
        ])

        return commands

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """
        Get completions for current input.

        Args:
            document: Current document/input
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        # Check if we're completing a command (starts with /)
        if text.startswith('/'):
            # Remove leading slash for matching
            command_prefix = text[1:].split()[0] if text[1:].split() else text[1:]

            # Get all commands
            commands = self._get_commands()

            # Filter commands matching prefix
            for cmd_name, cmd_desc in commands:
                if cmd_name.startswith(command_prefix):
                    # Calculate start position
                    start_pos = -len(command_prefix)

                    yield Completion(
                        text=cmd_name,
                        start_position=start_pos,
                        display=f"/{cmd_name}",
                        display_meta=cmd_desc
                    )

        # Check if we're completing a file path (contains @ or file-like pattern)
        elif '@' in text or '/' in word_before_cursor or '.' in word_before_cursor:
            # File path completion
            if '@' in text:
                # Extract file path after @
                path_start = text.rfind('@') + 1
                path_prefix = text[path_start:]
            else:
                path_prefix = word_before_cursor

            # Get file completions
            for completion in self._get_file_completions(path_prefix):
                yield completion

    def _get_file_completions(self, prefix: str) -> Iterable[Completion]:
        """
        Get file path completions.

        Args:
            prefix: File path prefix to complete

        Yields:
            Completion objects for matching files
        """
        try:
            # Expand user home directory
            if prefix.startswith('~'):
                prefix = str(Path(prefix).expanduser())

            # Get directory and filename parts
            if '/' in prefix:
                dir_path = '/'.join(prefix.split('/')[:-1]) or '.'
                file_prefix = prefix.split('/')[-1]
            else:
                dir_path = '.'
                file_prefix = prefix

            # List files in directory
            base_dir = Path(dir_path)

            if not base_dir.exists():
                return

            # Get matching files
            for path in base_dir.iterdir():
                if path.name.startswith(file_prefix):
                    # Calculate completion text
                    if path.is_dir():
                        completion_text = path.name + '/'
                        display_meta = '<dir>'
                    else:
                        completion_text = path.name
                        display_meta = f'{path.stat().st_size} bytes'

                    yield Completion(
                        text=completion_text,
                        start_position=-len(file_prefix),
                        display=completion_text,
                        display_meta=display_meta
                    )

        except (OSError, PermissionError):
            # Silently ignore errors
            pass


class WorkflowArgumentCompleter(Completer):
    """
    Completer for workflow command arguments.

    Provides suggestions for common workflow arguments like --max-steps, --web-search, etc.
    """

    def __init__(self):
        """Initialize workflow argument completer."""
        self.arguments = [
            ('--max-steps', 'Maximum workflow steps'),
            ('--web-search', 'Enable web search'),
            ('--verbose', 'Verbose output'),
        ]

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """
        Get completions for workflow arguments.

        Args:
            document: Current document
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        # Only complete if we're in a /workflow command
        if not text.startswith('/workflow'):
            return

        # Complete arguments starting with --
        if word_before_cursor.startswith('--') or text.endswith(' --'):
            prefix = word_before_cursor if word_before_cursor.startswith('--') else ''

            for arg_name, arg_desc in self.arguments:
                if arg_name.startswith(prefix):
                    yield Completion(
                        text=arg_name,
                        start_position=-len(prefix),
                        display=arg_name,
                        display_meta=arg_desc
                    )


def create_felix_completer(tool_registry=None, custom_commands=None) -> Completer:
    """
    Create the main Felix completer.

    Args:
        tool_registry: ToolRegistry instance
        custom_commands: CustomCommandLoader instance

    Returns:
        Completer instance
    """
    if not HAS_PROMPT_TOOLKIT:
        # Return dummy completer if prompt_toolkit not available
        return Completer()

    # For now, just use FelixCompleter
    # In the future, could merge multiple completers
    return FelixCompleter(tool_registry, custom_commands)
