"""
Command Handler for Felix conversational CLI.

Handles command parsing, routing, and natural language intent detection.
"""

import shlex
from typing import Dict, Any, List, Optional, Tuple
from .tools import ToolRegistry, ToolResult
from .formatters import OutputFormatter
from .custom_commands import CustomCommandLoader


class CommandHandler:
    """Handles parsing and execution of commands and natural language queries."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        formatter: OutputFormatter,
        felix_context: Optional[Dict[str, Any]] = None,
        enable_nl: bool = True,
        custom_commands_dir: Optional[str] = None
    ):
        """
        Initialize command handler.

        Args:
            tool_registry: Registry of available tools
            formatter: Output formatter
            felix_context: Felix system context (felix_system, knowledge_store, etc.)
            enable_nl: Enable natural language intent detection
            custom_commands_dir: Optional directory for custom commands
        """
        self.tool_registry = tool_registry
        self.formatter = formatter
        self.felix_context = felix_context or {}
        self.enable_nl = enable_nl

        # Initialize custom command loader
        self.custom_commands = CustomCommandLoader(custom_commands_dir)
        self.custom_commands.load_commands()

    def handle(self, user_input: str) -> ToolResult:
        """
        Handle user input - either explicit command or natural language.

        Args:
            user_input: User input string

        Returns:
            ToolResult from command execution
        """
        user_input = user_input.strip()

        if not user_input:
            return ToolResult(
                success=False,
                content="",
                error="Empty input"
            )

        # Check for explicit command
        if user_input.startswith('/'):
            return self._handle_explicit_command(user_input)

        # Otherwise, try natural language if enabled
        if self.enable_nl:
            return self._handle_natural_language(user_input)
        else:
            return ToolResult(
                success=False,
                content="",
                error="Natural language mode not enabled. Use /help for available commands."
            )

    def _handle_explicit_command(self, command_str: str) -> ToolResult:
        """Handle explicit slash command."""
        try:
            # Remove leading slash
            command_str = command_str[1:]

            # Parse command parts (handle quotes properly)
            try:
                parts = shlex.split(command_str)
            except ValueError:
                # Fallback to simple split if shlex fails
                parts = command_str.split()

            if not parts:
                return ToolResult(
                    success=False,
                    content="",
                    error="Empty command"
                )

            # First part is the command name
            command_name = parts[0]
            remaining = parts[1:]

            # Check for custom command first
            custom_command = self.custom_commands.get_command(command_name)
            if custom_command:
                # Execute custom command with argument substitution
                # Parse kwargs from remaining parts
                args = []
                kwargs = {}
                for part in remaining:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        kwargs[key] = value
                    else:
                        args.append(part)

                expanded_template = custom_command.execute(args, kwargs)

                # Execute the expanded template as a workflow
                workflow_tool = self.tool_registry.get('workflow')
                if workflow_tool:
                    return workflow_tool.execute(['run', expanded_template], {})
                else:
                    return ToolResult(
                        success=True,
                        content=expanded_template,
                        data={'custom_command': True}
                    )

            # Otherwise, check for built-in tool
            tool = self.tool_registry.get(command_name)

            if not tool:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Unknown command: /{command_name}\nUse /help for available commands."
                )

            # Parse remaining parts into args and kwargs
            args, kwargs = tool.parse_args(remaining)

            # Execute the tool
            return tool.execute(args, kwargs)

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Command execution failed: {str(e)}"
            )

    def _handle_natural_language(self, query: str) -> ToolResult:
        """Handle natural language query using LLM for intent detection."""
        try:
            # Get LLM adapter from context
            llm_adapter = self.felix_context.get('llm_adapter')

            if not llm_adapter:
                # Fallback: try to detect intent with simple keyword matching
                return self._handle_nl_fallback(query)

            # Use LLM to detect intent
            intent_prompt = self._build_intent_prompt(query)

            try:
                response = llm_adapter.generate_response(
                    prompt=intent_prompt,
                    max_tokens=200,
                    temperature=0.3
                )

                # Parse LLM response to extract tool and args
                tool_name, args, kwargs = self._parse_intent_response(response)

                if not tool_name:
                    # Couldn't detect intent, fall back to keyword matching
                    return self._handle_nl_fallback(query)

                # Get the tool
                tool = self.tool_registry.get(tool_name)
                if not tool:
                    return self._handle_nl_fallback(query)

                # Execute the tool
                return tool.execute(args, kwargs)

            except Exception as e:
                # LLM failed, fall back to keyword matching
                return self._handle_nl_fallback(query)

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to process query: {str(e)}"
            )

    def _handle_nl_fallback(self, query: str) -> ToolResult:
        """Fallback natural language handling with simple keyword matching."""
        import re
        query_lower = query.lower()

        # Check if query is likely a math problem (has numbers and operators)
        is_math_query = bool(re.search(r'[0-9+\-*/=]|plus|minus|times|divided|equals', query))

        # Keyword patterns for different tools
        # More specific knowledge keywords to avoid catching math/general queries
        if not is_math_query and any(word in query_lower for word in ['search knowledge', 'find knowledge', 'knowledge about', 'tell me about']):
            # Knowledge search
            tool = self.tool_registry.get('knowledge')
            if tool:
                return tool.execute(['search', query], {})

        elif any(word in query_lower for word in ['run', 'execute', 'do', 'create', 'build', 'design', 'analyze']):
            # Workflow execution
            tool = self.tool_registry.get('workflow')
            if tool:
                return tool.execute(['run', query], {})

        elif any(word in query_lower for word in ['history', 'previous', 'past', 'recent workflows']):
            # History
            tool = self.tool_registry.get('history')
            if tool:
                return tool.execute(['list'], {'limit': 10})

        elif any(word in query_lower for word in ['status', 'health', 'check', 'system']):
            # System status
            tool = self.tool_registry.get('system')
            if tool:
                return tool.execute(['status'], {})

        elif any(word in query_lower for word in ['document', 'ingest', 'upload']):
            # Document management (removed 'file' - too broad, causes false matches)
            tool = self.tool_registry.get('document')
            if tool:
                return tool.execute(['list'], {})

        else:
            # Default to workflow execution
            tool = self.tool_registry.get('workflow')
            if tool:
                return tool.execute(['run', query], {})

        return ToolResult(
            success=False,
            content="",
            error="Could not understand query. Try using explicit commands like /workflow run <task>"
        )

    def _build_intent_prompt(self, query: str) -> str:
        """Build prompt for LLM intent detection."""
        tool_descriptions = []
        for tool in self.tool_registry.list_tools():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        tools_text = "\n".join(tool_descriptions)

        prompt = f"""You are a command interpreter for the Felix AI system. Given a user query, determine which tool to use and what arguments to provide.

Available tools:
{tools_text}

User query: "{query}"

Respond in this exact format:
TOOL: <tool_name>
ARGS: <comma-separated positional arguments>
KWARGS: <key=value pairs separated by commas>

Examples:
Query: "Search for information about quantum computing"
TOOL: knowledge
ARGS: search, quantum computing
KWARGS:

Query: "Run an analysis of machine learning trends"
TOOL: workflow
ARGS: run, analyze machine learning trends
KWARGS:

Query: "Show me the last 5 workflows"
TOOL: history
ARGS: list
KWARGS: limit=5

Now analyze the user query and respond:"""

        return prompt

    def _parse_intent_response(self, response: str) -> Tuple[Optional[str], List[str], Dict[str, Any]]:
        """
        Parse LLM intent detection response.

        Returns:
            Tuple of (tool_name, args, kwargs)
        """
        tool_name = None
        args = []
        kwargs = {}

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()

            if line.startswith('TOOL:'):
                tool_name = line.split(':', 1)[1].strip()

            elif line.startswith('ARGS:'):
                args_str = line.split(':', 1)[1].strip()
                if args_str:
                    args = [a.strip() for a in args_str.split(',')]

            elif line.startswith('KWARGS:'):
                kwargs_str = line.split(':', 1)[1].strip()
                if kwargs_str:
                    for pair in kwargs_str.split(','):
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            kwargs[key.strip()] = value.strip()

        return tool_name, args, kwargs
