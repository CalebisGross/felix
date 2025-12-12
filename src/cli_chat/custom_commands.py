"""
Custom Command Loader for Felix CLI

Allows users to define custom commands in .felix/commands/ directory.
Supports YAML frontmatter for metadata and argument substitution.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


def parse_yaml_frontmatter(content: str) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Parse YAML frontmatter from markdown content.

    Args:
        content: File content with potential frontmatter

    Returns:
        Tuple of (metadata dict or None, content without frontmatter)
    """
    # Check for YAML frontmatter (---\n...\n---)
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        return None, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    # Parse YAML frontmatter manually (simple key: value format)
    metadata = {}
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Parse lists
            if value.startswith('[') and value.endswith(']'):
                # Simple list parsing
                value = [v.strip().strip('"').strip("'") for v in value[1:-1].split(',')]

            metadata[key] = value

    return metadata, body


def substitute_arguments(template: str, args: List[str], kwargs: Dict[str, Any]) -> str:
    """
    Substitute arguments in template string.

    Supports:
    - {arg0}, {arg1}, ... - Positional arguments
    - {args} - All positional arguments joined
    - {key_name} - Named keyword arguments

    Args:
        template: Template string with placeholders
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        String with substituted values
    """
    result = template

    # Substitute positional arguments
    for i, arg in enumerate(args):
        result = result.replace(f"{{arg{i}}}", arg)

    # Substitute all args as a single string
    result = result.replace("{args}", " ".join(args))

    # Substitute keyword arguments
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))

    return result


class CustomCommand:
    """Represents a custom user-defined command."""

    def __init__(
        self,
        name: str,
        template: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize custom command.

        Args:
            name: Command name (without leading slash)
            template: Command template with placeholders
            metadata: Optional metadata from frontmatter
        """
        self.name = name
        self.template = template
        self.metadata = metadata or {}

        # Extract metadata fields
        self.description = self.metadata.get('description', f'Custom command: {name}')
        self.usage = self.metadata.get('usage', f'/{name} [args...]')
        self.args = self.metadata.get('args', [])
        self.aliases = self.metadata.get('aliases', [])

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> str:
        """
        Execute the command by substituting arguments in template.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Expanded template string
        """
        return substitute_arguments(self.template, args, kwargs)

    def __repr__(self) -> str:
        return f"CustomCommand(name={self.name}, description={self.description})"


class CustomCommandLoader:
    """Loads and manages custom user-defined commands."""

    def __init__(self, commands_dir: Optional[str] = None):
        """
        Initialize command loader.

        Args:
            commands_dir: Directory containing custom commands
                         (defaults to ./.felix/commands/ and ~/.felix/commands/)
        """
        self.commands: Dict[str, CustomCommand] = {}

        # Default command directories
        if commands_dir:
            self.command_dirs = [Path(commands_dir)]
        else:
            self.command_dirs = [
                Path.cwd() / '.felix' / 'commands',
                Path.home() / '.felix' / 'commands'
            ]

    def load_commands(self):
        """Load all custom commands from command directories."""
        self.commands.clear()

        for cmd_dir in self.command_dirs:
            if not cmd_dir.exists():
                continue

            # Load all .md and .txt files as commands
            for file_path in cmd_dir.glob('*.md'):
                self._load_command_file(file_path)

            for file_path in cmd_dir.glob('*.txt'):
                self._load_command_file(file_path)

    def _load_command_file(self, file_path: Path):
        """
        Load a single command file.

        Args:
            file_path: Path to command file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter
            metadata, template = parse_yaml_frontmatter(content)

            # Command name is the filename without extension
            command_name = file_path.stem

            # Create command
            command = CustomCommand(
                name=command_name,
                template=template.strip(),
                metadata=metadata
            )

            # Register command
            self.commands[command_name] = command

            # Register aliases
            for alias in command.aliases:
                self.commands[alias] = command

        except Exception as e:
            print(f"Warning: Failed to load command from {file_path}: {e}")

    def get_command(self, name: str) -> Optional[CustomCommand]:
        """
        Get a command by name or alias.

        Args:
            name: Command name (without leading slash)

        Returns:
            CustomCommand instance or None
        """
        return self.commands.get(name)

    def list_commands(self) -> List[CustomCommand]:
        """
        Get list of all loaded commands.

        Returns:
            List of CustomCommand instances
        """
        # Return unique commands (excluding aliases)
        seen = set()
        unique_commands = []

        for command in self.commands.values():
            if command.name not in seen:
                seen.add(command.name)
                unique_commands.append(command)

        return unique_commands

    def reload(self):
        """Reload all commands from disk."""
        self.load_commands()

    def create_command_file(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        usage: Optional[str] = None,
        args: Optional[List[str]] = None
    ) -> Path:
        """
        Create a new command file.

        Args:
            name: Command name
            template: Command template
            description: Optional description
            usage: Optional usage string
            args: Optional list of argument names

        Returns:
            Path to created file
        """
        # Use first command directory
        cmd_dir = self.command_dirs[0]
        cmd_dir.mkdir(parents=True, exist_ok=True)

        # Build frontmatter
        frontmatter_lines = []
        if description:
            frontmatter_lines.append(f'description: "{description}"')
        if usage:
            frontmatter_lines.append(f'usage: "{usage}"')
        if args:
            args_str = ', '.join(f'"{arg}"' for arg in args)
            frontmatter_lines.append(f'args: [{args_str}]')

        # Build file content
        if frontmatter_lines:
            content = "---\n" + "\n".join(frontmatter_lines) + "\n---\n\n" + template
        else:
            content = template

        # Write file
        file_path = cmd_dir / f"{name}.md"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path


# Example command files:
"""
# Example 1: Simple command without frontmatter
# File: .felix/commands/hello.md

Hello, {arg0}! How can I help you today?

# Example 2: Command with frontmatter
# File: .felix/commands/review-code.md

---
description: "Review code for quality and best practices"
usage: "/review-code <file_path>"
args: ["file_path"]
aliases: ["review", "code-review"]
---

Please review the code in {arg0} for:
1. Code quality and readability
2. Performance optimizations
3. Security vulnerabilities
4. Best practices
5. Documentation completeness

Provide specific recommendations for improvement.

# Example 3: Command with multiple arguments
# File: .felix/commands/compare.md

---
description: "Compare two items"
usage: "/compare <item1> <item2>"
args: ["item1", "item2"]
---

Compare {arg0} and {arg1} in terms of:
- Features
- Performance
- Cost
- Pros and Cons

Provide a detailed comparison table.
"""
