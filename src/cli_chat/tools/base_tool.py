"""
Base Tool interface for Felix conversational CLI.

Defines the abstract interface that all tools must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    success: bool
    content: str
    data: Optional[Dict[str, Any]] = None
    workflow_id: Optional[str] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return self.content
        else:
            return f"Error: {self.error or self.content}"


class BaseTool(ABC):
    """Abstract base class for all CLI chat tools."""

    def __init__(self, felix_context: Optional[Dict[str, Any]] = None):
        """
        Initialize tool with Felix context.

        Args:
            felix_context: Dictionary containing Felix system components
                          (workflow, knowledge_store, llm_router, etc.)
        """
        self.felix_context = felix_context or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g., 'workflow', 'knowledge', 'agent')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the tool does."""
        pass

    @property
    @abstractmethod
    def usage(self) -> str:
        """Usage string showing how to use the tool."""
        pass

    @abstractmethod
    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments (flags, options)

        Returns:
            ToolResult with success status and content
        """
        pass

    def parse_args(self, command_parts: List[str]) -> tuple[List[str], Dict[str, Any]]:
        """
        Parse command parts into args and kwargs.

        Args:
            command_parts: List of command tokens

        Returns:
            Tuple of (positional_args, keyword_args)
        """
        args = []
        kwargs = {}
        i = 0

        while i < len(command_parts):
            part = command_parts[i]

            # Flag with value (--key value or -k value)
            if part.startswith('-'):
                key = part.lstrip('-')

                # Check if next part is a value
                if i + 1 < len(command_parts) and not command_parts[i + 1].startswith('-'):
                    kwargs[key] = command_parts[i + 1]
                    i += 2
                else:
                    # Boolean flag
                    kwargs[key] = True
                    i += 1
            else:
                # Positional argument
                args.append(part)
                i += 1

        return args, kwargs

    def format_error(self, message: str) -> ToolResult:
        """Create an error result."""
        return ToolResult(
            success=False,
            content="",
            error=message
        )

    def format_success(
        self,
        content: str,
        data: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None
    ) -> ToolResult:
        """Create a success result."""
        return ToolResult(
            success=True,
            content=content,
            data=data,
            workflow_id=workflow_id
        )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Get list of all registered tools."""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self._tools.keys())

    def get_help_text(self) -> str:
        """Generate help text for all tools."""
        lines = ["Available commands:\n"]

        for tool in sorted(self._tools.values(), key=lambda t: t.name):
            lines.append(f"  /{tool.name}")
            lines.append(f"    {tool.description}")
            lines.append(f"    Usage: {tool.usage}")
            lines.append("")

        return "\n".join(lines)
