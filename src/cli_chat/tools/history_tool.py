"""
History Tool for browsing Felix workflow history in conversational CLI.
"""

from typing import Dict, Any, List, Optional
from .base_tool import BaseTool, ToolResult


class HistoryTool(BaseTool):
    """Tool for browsing and managing workflow history."""

    @property
    def name(self) -> str:
        return "history"

    @property
    def description(self) -> str:
        return "Browse and search workflow execution history"

    @property
    def usage(self) -> str:
        return "/history [list|search|show] [options]"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute history command.

        Commands:
            /history list [--limit N]           - List recent workflows
            /history search <query>              - Search workflows by task
            /history show <id>                   - Show workflow details
            /history thread <id>                 - Show conversation thread
        """
        if not args:
            # Default to list
            return self._list_workflows(kwargs)

        command = args[0]

        if command == "list":
            return self._list_workflows(kwargs)
        elif command == "search":
            return self._search_workflows(args[1:], kwargs)
        elif command == "show":
            return self._show_workflow(args[1:], kwargs)
        elif command == "thread":
            return self._show_thread(args[1:], kwargs)
        else:
            return self.format_error(f"Unknown history command: {command}")

    def _list_workflows(self, kwargs: Dict[str, Any]) -> ToolResult:
        """List recent workflows."""
        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            limit = int(kwargs.get('limit', 10))

            # Get recent workflows
            workflows = history.get_workflow_outputs(limit=limit)

            if not workflows:
                return self.format_success("No workflows found")

            # Format output
            output_lines = ["Recent Workflows:", ""]

            for wf in workflows:
                workflow_id = wf.workflow_id
                task = wf.task_input[:60] + "..." if len(wf.task_input) > 60 else wf.task_input
                status = wf.status
                confidence = wf.confidence
                created_at = wf.created_at

                output_lines.append(f"[{workflow_id}] {status}")
                output_lines.append(f"  Task: {task}")
                output_lines.append(f"  Confidence: {confidence:.2f} | {created_at}")
                output_lines.append("")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to list workflows: {str(e)}")

    def _search_workflows(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Search workflows by task description."""
        if not args:
            return self.format_error("Usage: /history search <query>")

        query = " ".join(args)

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflows = history.search_workflows(query, limit=20)

            if not workflows:
                return self.format_success(f"No workflows found matching '{query}'")

            # Format output
            output_lines = [f"Workflows matching '{query}':", ""]

            for wf in workflows:
                workflow_id = wf.workflow_id
                task = wf.task_input[:60] + "..." if len(wf.task_input) > 60 else wf.task_input
                status = wf.status
                confidence = wf.confidence

                output_lines.append(f"[{workflow_id}] {status} (confidence: {confidence:.2f})")
                output_lines.append(f"  {task}")
                output_lines.append("")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to search workflows: {str(e)}")

    def _show_workflow(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show details of a specific workflow."""
        if not args:
            # Show last workflow from context
            last_workflow_id = self.felix_context.get('last_workflow_id')
            if not last_workflow_id:
                return self.format_error("Usage: /history show <workflow_id>")
            workflow_id = last_workflow_id
        else:
            workflow_id = args[0]

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflow = history.get_workflow_output(int(workflow_id))

            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            # Format workflow details
            output_lines = [
                f"Workflow {workflow.workflow_id}",
                "=" * 60,
                "",
                f"Task: {workflow.task_input}",
                f"Status: {workflow.status}",
                f"Confidence: {workflow.confidence:.2f}",
                f"Created: {workflow.created_at}",
                f"Completed: {workflow.completed_at}",
                f"Processing Time: {workflow.processing_time:.2f}s",
                f"Agents: {workflow.agents_count}",
                f"Tokens: {workflow.tokens_used} / {workflow.max_tokens}",
                ""
            ]

            # Add parent/thread info if available
            if workflow.parent_workflow_id:
                output_lines.append(f"Parent Workflow: {workflow.parent_workflow_id}")
            if workflow.conversation_thread_id:
                output_lines.append(f"Thread ID: {workflow.conversation_thread_id}")

            output_lines.extend([
                "",
                "Result:",
                "-" * 60,
                workflow.final_synthesis or "No result generated"
            ])

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to retrieve workflow: {str(e)}")

    def _show_thread(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show conversation thread for a workflow."""
        if not args:
            return self.format_error("Usage: /history thread <workflow_id>")

        workflow_id = int(args[0])

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()

            # Get the workflow to find thread ID
            workflow = history.get_workflow_output(workflow_id)
            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            thread_id = workflow.conversation_thread_id
            if not thread_id:
                return self.format_success(f"Workflow {workflow_id} is not part of a conversation thread")

            # Get all workflows in thread
            thread_workflows = history.get_conversation_thread(thread_id)

            if not thread_workflows:
                return self.format_success("No workflows in thread")

            # Format thread
            output_lines = [
                f"Conversation Thread: {thread_id}",
                "=" * 60,
                ""
            ]

            for wf in thread_workflows:
                indent = "  " if wf.parent_workflow_id else ""
                output_lines.append(f"{indent}[{wf.workflow_id}] {wf.status}")
                output_lines.append(f"{indent}  Task: {wf.task_input[:60]}")
                output_lines.append(f"{indent}  Confidence: {wf.confidence:.2f}")
                output_lines.append("")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to retrieve thread: {str(e)}")
