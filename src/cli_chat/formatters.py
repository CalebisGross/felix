"""
Output formatters for Felix conversational CLI.

Handles terminal formatting, markdown rendering, and progress display.
"""

import sys
from typing import Optional, List, Dict, Any
from datetime import datetime

# Try to import rich for enhanced formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.live import Live
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class OutputFormatter:
    """Handles formatted output to terminal."""

    def __init__(self, use_colors: bool = True):
        """
        Initialize formatter.

        Args:
            use_colors: Whether to use ANSI color codes
        """
        self.use_colors = use_colors and sys.stdout.isatty()

        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'gray': '\033[90m',
        }

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"

    def print_header(self, text: str):
        """Print a header."""
        print(f"\n{self._color('═' * 60, 'blue')}")
        print(self._color(f"  {text}", 'bold'))
        print(f"{self._color('═' * 60, 'blue')}\n")

    def print_subheader(self, text: str):
        """Print a subheader."""
        print(f"\n{self._color(text, 'cyan')}")
        print(self._color('─' * len(text), 'cyan'))

    def print_success(self, text: str):
        """Print success message."""
        print(f"{self._color('✓', 'green')} {text}")

    def print_error(self, text: str):
        """Print error message."""
        print(f"{self._color('✗', 'red')} {self._color(text, 'red')}")

    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{self._color('⚠', 'yellow')} {self._color(text, 'yellow')}")

    def print_info(self, text: str):
        """Print info message."""
        print(f"{self._color('ℹ', 'blue')} {text}")

    def print_workflow_result(self, content: str, confidence: float, metrics: Optional[Dict] = None):
        """Print workflow execution result."""
        self.print_subheader("Workflow Result")
        print(f"\n{content}\n")

        # Print metrics
        if metrics:
            self.print_dim(f"Confidence: {confidence:.2f}")
            self.print_dim(f"Agents: {metrics.get('agent_count', 'N/A')}")
            self.print_dim(f"Tokens: {metrics.get('tokens_used', 'N/A')}")
            self.print_dim(f"Time: {metrics.get('processing_time', 'N/A')}s")
            print()

    def print_agent_contribution(self, agent_name: str, content: str, confidence: float):
        """Print agent contribution."""
        print(f"\n{self._color(f'[{agent_name}]', 'magenta')} {self._color(f'(confidence: {confidence:.2f})', 'dim')}")
        print(content)

    def print_knowledge_entry(self, entry: Dict[str, Any]):
        """Print a knowledge entry."""
        domain = entry.get('domain', 'general')
        confidence = entry.get('confidence', 0.0)
        content = entry.get('content', '')

        print(f"\n{self._color(f'[{domain}]', 'cyan')} {self._color(f'confidence: {confidence:.2f}', 'dim')}")
        print(content)

    def print_session_list(self, sessions: List[Dict[str, Any]]):
        """Print list of sessions."""
        self.print_subheader("Chat Sessions")

        if not sessions:
            self.print_dim("No sessions found")
            return

        for session in sessions:
            session_id = session.get('session_id', 'unknown')
            created_at = session.get('created_at', '')
            last_active = session.get('last_active', '')
            message_count = session.get('message_count', 0)

            print(f"\n{self._color(session_id, 'yellow')}")
            self.print_dim(f"  Created: {created_at}")
            self.print_dim(f"  Last active: {last_active}")
            self.print_dim(f"  Messages: {message_count}")

    def print_workflow_history(self, workflows: List[Dict[str, Any]]):
        """Print workflow history."""
        self.print_subheader("Recent Workflows")

        if not workflows:
            self.print_dim("No workflows found")
            return

        for wf in workflows:
            workflow_id = wf.get('workflow_id', 'unknown')
            task = wf.get('task_description', 'No description')
            status = wf.get('status', 'unknown')
            confidence = wf.get('final_confidence', 0.0)
            created_at = wf.get('created_at', '')

            status_color = 'green' if status == 'completed' else 'yellow'

            print(f"\n{self._color(workflow_id[:8], 'blue')} {self._color(f'[{status}]', status_color)}")
            print(f"  {task[:60]}{'...' if len(task) > 60 else ''}")
            self.print_dim(f"  Confidence: {confidence:.2f} | {created_at}")

    def print_table(self, headers: List[str], rows: List[List[str]]):
        """Print a simple table."""
        if not rows:
            return

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(f"\n{self._color(header_line, 'bold')}")
        print(self._color('─' * len(header_line), 'dim'))

        # Print rows
        for row in rows:
            row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            print(row_line)

    def print_dim(self, text: str):
        """Print dimmed text."""
        print(self._color(text, 'dim'))

    def print_bold(self, text: str):
        """Print bold text."""
        print(self._color(text, 'bold'))

    def print_streaming_token(self, token: str):
        """Print a token for streaming output."""
        print(token, end='', flush=True)

    def print_progress(self, message: str):
        """Print progress message."""
        print(f"{self._color('⋯', 'blue')} {self._color(message, 'dim')}", flush=True)

    def print_markdown(self, content: str):
        """
        Print markdown content with basic formatting.

        Supports:
        - Headers (# ## ###)
        - Bold (**text**)
        - Code blocks (```)
        - Lists (- *)
        """
        lines = content.split('\n')

        for line in lines:
            # Headers
            if line.startswith('### '):
                print(self._color(line[4:], 'cyan'))
            elif line.startswith('## '):
                print(self._color(line[3:], 'cyan'))
            elif line.startswith('# '):
                print(self._color(line[2:], 'bold'))
            # Code blocks
            elif line.startswith('```'):
                print(self._color(line, 'gray'))
            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                print(f"  {self._color('•', 'blue')} {line[2:]}")
            # Bold text (simple implementation)
            elif '**' in line:
                # Simple bold replacement
                parts = line.split('**')
                result = []
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are bold
                        result.append(self._color(part, 'bold'))
                    else:
                        result.append(part)
                print(''.join(result))
            else:
                print(line)

    def print_help(self, help_text: str):
        """Print help text."""
        self.print_header("Felix Chat - Help")
        print(help_text)
        print()

    def print_welcome(self, session_id: str):
        """Print welcome message."""
        self.print_header("Felix Conversational CLI")
        print(f"Session: {self._color(session_id, 'yellow')}")
        print()
        print("Type your question or use commands:")
        print(f"  {self._color('/help', 'cyan')} - Show available commands")
        print(f"  {self._color('/exit', 'cyan')} or {self._color('/quit', 'cyan')} - Exit chat")
        print()

    def print_goodbye(self):
        """Print goodbye message."""
        print(f"\n{self._color('Goodbye!', 'green')}\n")

    def clear_line(self):
        """Clear the current line."""
        if self.use_colors:
            print('\r\033[K', end='', flush=True)


class RichOutputFormatter:
    """
    Enhanced formatter using rich library for beautiful terminal output.

    Falls back to OutputFormatter if rich is not installed.
    """

    def __init__(self):
        """Initialize the rich formatter."""
        if HAS_RICH:
            self.console = Console()
            self.use_rich = True
        else:
            # Fallback to basic formatter
            self.fallback = OutputFormatter()
            self.use_rich = False

    def print_header(self, text: str):
        """Print a header."""
        if self.use_rich:
            self.console.print(f"\n[bold blue]{'═' * 60}[/bold blue]")
            self.console.print(f"[bold]  {text}[/bold]")
            self.console.print(f"[bold blue]{'═' * 60}[/bold blue]\n")
        else:
            self.fallback.print_header(text)

    def print_subheader(self, text: str):
        """Print a subheader."""
        if self.use_rich:
            self.console.print(f"\n[cyan]{text}[/cyan]")
            self.console.print(f"[dim cyan]{'─' * len(text)}[/dim cyan]")
        else:
            self.fallback.print_subheader(text)

    def print_success(self, text: str):
        """Print success message."""
        if self.use_rich:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            self.fallback.print_success(text)

    def print_error(self, text: str):
        """Print error message."""
        if self.use_rich:
            self.console.print(f"[red]✗ {text}[/red]")
        else:
            self.fallback.print_error(text)

    def print_warning(self, text: str):
        """Print warning message."""
        if self.use_rich:
            self.console.print(f"[yellow]⚠ {text}[/yellow]")
        else:
            self.fallback.print_warning(text)

    def print_info(self, text: str):
        """Print info message."""
        if self.use_rich:
            self.console.print(f"[blue]ℹ[/blue] {text}")
        else:
            self.fallback.print_info(text)

    def print_workflow_result(self, content: str, confidence: float, metrics: Optional[Dict] = None):
        """Print workflow execution result."""
        if self.use_rich:
            self.print_subheader("Workflow Result")

            # Use Panel for result content
            self.console.print(Panel(content, border_style="green", padding=(1, 2)))

            # Print metrics as a table
            if metrics:
                table = Table(show_header=False, box=box.SIMPLE)
                table.add_row("Confidence:", f"{confidence:.2f}")
                table.add_row("Agents:", str(metrics.get('agent_count', 'N/A')))
                table.add_row("Tokens:", str(metrics.get('tokens_used', 'N/A')))
                table.add_row("Time:", f"{metrics.get('processing_time', 'N/A')}s")
                self.console.print(table)
        else:
            self.fallback.print_workflow_result(content, confidence, metrics)

    def print_agent_contribution(self, agent_name: str, content: str, confidence: float):
        """Print agent contribution."""
        if self.use_rich:
            self.console.print(f"\n[magenta][{agent_name}][/magenta] [dim](confidence: {confidence:.2f})[/dim]")
            self.console.print(content)
        else:
            self.fallback.print_agent_contribution(agent_name, content, confidence)

    def print_knowledge_entry(self, entry: Dict[str, Any]):
        """Print a knowledge entry."""
        if self.use_rich:
            domain = entry.get('domain', 'general')
            confidence = entry.get('confidence', 0.0)
            content = entry.get('content', '')

            self.console.print(f"\n[cyan][{domain}][/cyan] [dim]confidence: {confidence:.2f}[/dim]")
            self.console.print(content)
        else:
            self.fallback.print_knowledge_entry(entry)

    def print_session_list(self, sessions: List[Dict[str, Any]]):
        """Print list of sessions."""
        if self.use_rich:
            self.print_subheader("Chat Sessions")

            if not sessions:
                self.console.print("[dim]No sessions found[/dim]")
                return

            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("Session ID", style="yellow")
            table.add_column("Title", style="white")
            table.add_column("Tags", style="cyan")
            table.add_column("Messages", justify="right", style="dim")
            table.add_column("Last Active", style="dim")

            for session in sessions:
                session_id = session.get('session_id', 'unknown')
                title = session.get('title', '') or '-'
                tags = ', '.join(session.get('tags', [])) or '-'
                message_count = str(session.get('message_count', 0))
                last_active = session.get('last_active', '')

                # Format timestamp
                try:
                    dt = datetime.fromisoformat(last_active)
                    last_active = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    pass

                table.add_row(session_id, title, tags, message_count, last_active)

            self.console.print(table)
        else:
            self.fallback.print_session_list(sessions)

    def print_workflow_history(self, workflows: List[Dict[str, Any]]):
        """Print workflow history."""
        if self.use_rich:
            self.print_subheader("Recent Workflows")

            if not workflows:
                self.console.print("[dim]No workflows found[/dim]")
                return

            table = Table(show_header=True, box=box.ROUNDED)
            table.add_column("ID", style="blue")
            table.add_column("Task", style="white")
            table.add_column("Status", style="green")
            table.add_column("Confidence", justify="right")
            table.add_column("Created", style="dim")

            for wf in workflows:
                workflow_id = wf.get('workflow_id', 'unknown')[:8]
                task = wf.get('task_description', 'No description')
                task = task[:60] + '...' if len(task) > 60 else task
                status = wf.get('status', 'unknown')
                confidence = f"{wf.get('final_confidence', 0.0):.2f}"
                created_at = wf.get('created_at', '')

                status_style = 'green' if status == 'completed' else 'yellow'
                status_text = f"[{status_style}]{status}[/{status_style}]"

                table.add_row(workflow_id, task, status_text, confidence, created_at)

            self.console.print(table)
        else:
            self.fallback.print_workflow_history(workflows)

    def print_table(self, headers: List[str], rows: List[List[str]]):
        """Print a table."""
        if self.use_rich:
            table = Table(show_header=True, box=box.ROUNDED)

            for header in headers:
                table.add_column(header)

            for row in rows:
                table.add_row(*[str(cell) for cell in row])

            self.console.print(table)
        else:
            self.fallback.print_table(headers, rows)

    def print_dim(self, text: str):
        """Print dimmed text."""
        if self.use_rich:
            self.console.print(f"[dim]{text}[/dim]")
        else:
            self.fallback.print_dim(text)

    def print_bold(self, text: str):
        """Print bold text."""
        if self.use_rich:
            self.console.print(f"[bold]{text}[/bold]")
        else:
            self.fallback.print_bold(text)

    def print_streaming_token(self, token: str):
        """Print a token for streaming output."""
        if self.use_rich:
            self.console.print(token, end='')
        else:
            self.fallback.print_streaming_token(token)

    def print_progress(self, message: str):
        """Print progress message."""
        if self.use_rich:
            self.console.print(f"[blue]⋯[/blue] [dim]{message}[/dim]")
        else:
            self.fallback.print_progress(message)

    def print_markdown(self, content: str):
        """Print markdown content with rich formatting."""
        if self.use_rich:
            md = Markdown(content)
            self.console.print(md)
        else:
            self.fallback.print_markdown(content)

    def print_code(self, code: str, language: str = "python"):
        """Print syntax-highlighted code."""
        if self.use_rich:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            # Fallback: print with basic code block markers
            print(f"```{language}")
            print(code)
            print("```")

    def print_panel(self, content: str, title: Optional[str] = None, style: str = "white"):
        """Print content in a panel."""
        if self.use_rich:
            panel = Panel(content, title=title, border_style=style, padding=(1, 2))
            self.console.print(panel)
        else:
            # Fallback: print with simple box
            if title:
                print(f"\n╔═ {title} " + "═" * (56 - len(title)))
            else:
                print("\n" + "═" * 60)
            print(content)
            print("═" * 60)

    def print_help(self, help_text: str):
        """Print help text."""
        if self.use_rich:
            self.print_header("Felix Chat - Help")
            self.print_markdown(help_text)
        else:
            self.fallback.print_help(help_text)

    def print_welcome(self, session_id: str):
        """Print welcome message."""
        if self.use_rich:
            self.print_header("Felix Conversational CLI")
            self.console.print(f"Session: [yellow]{session_id}[/yellow]")
            self.console.print()
            self.console.print("Type your question or use commands:")
            self.console.print(f"  [cyan]/help[/cyan] - Show available commands")
            self.console.print(f"  [cyan]/exit[/cyan] or [cyan]/quit[/cyan] - Exit chat")
            self.console.print()
        else:
            self.fallback.print_welcome(session_id)

    def print_goodbye(self):
        """Print goodbye message."""
        if self.use_rich:
            self.console.print("\n[green]Goodbye![/green]\n")
        else:
            self.fallback.print_goodbye()

    def clear_line(self):
        """Clear the current line."""
        if self.use_rich:
            self.console.print('\r\033[K', end='')
        else:
            self.fallback.clear_line()

    def create_progress(self, description: str = "Processing..."):
        """
        Create a progress indicator context manager.

        Usage:
            with formatter.create_progress("Working...") as progress:
                # Do work
                pass
        """
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        else:
            # Fallback: return a dummy context manager
            class DummyProgress:
                def __enter__(self):
                    print(f"⋯ {description}")
                    return self
                def __exit__(self, *args):
                    pass
                def add_task(self, *args, **kwargs):
                    return 0
            return DummyProgress()


def get_formatter(use_rich: bool = True) -> OutputFormatter:
    """
    Get the appropriate formatter based on availability and preference.

    Args:
        use_rich: Prefer rich formatter if available

    Returns:
        RichOutputFormatter if rich is available and requested, else OutputFormatter
    """
    if use_rich and HAS_RICH:
        return RichOutputFormatter()
    else:
        return OutputFormatter()
