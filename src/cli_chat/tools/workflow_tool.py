"""
Workflow Tool for executing Felix workflows in conversational CLI.

This tool now properly integrates with Felix's multi-agent architecture
through the CLIWorkflowOrchestrator, ensuring:
- Helical agent progression
- CentralPost communication
- Knowledge store integration with meta-learning
- Self-improvement through feedback loops
- Session-workflow continuity
"""

from typing import Dict, Any, List, Optional
from .base_tool import BaseTool, ToolResult
from ..cli_workflow_orchestrator import CLIWorkflowOrchestrator


class WorkflowTool(BaseTool):
    """Tool for executing and managing Felix workflows."""

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def description(self) -> str:
        return "Execute Felix workflows with multi-agent collaboration"

    @property
    def usage(self) -> str:
        return "/workflow run <task> [--max-steps N] [--web-search]"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute workflow command.

        Commands:
            /workflow run <task>       - Run a new workflow
            /workflow continue         - Continue from last workflow
            /workflow show <id>        - Show workflow details
        """
        if not args:
            return self.format_error("Usage: /workflow run <task>")

        command = args[0]

        if command == "run":
            return self._run_workflow(args[1:], kwargs)
        elif command == "continue":
            return self._continue_workflow(kwargs)
        elif command == "show":
            return self._show_workflow(args[1:], kwargs)
        else:
            return self.format_error(f"Unknown workflow command: {command}")

    def _run_workflow(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Run a new workflow using CLIWorkflowOrchestrator.

        This properly integrates with Felix's multi-agent architecture,
        unlike the previous direct workflow calls that bypassed the system.
        """
        if not args:
            return self.format_error("Usage: /workflow run <task>")

        # Join remaining args as task description
        task = " ".join(args)

        # Get required components from context
        felix_system = self.felix_context.get('felix_system')
        session_manager = self.felix_context.get('session_manager')
        session_id = self.felix_context.get('session_id')
        formatter = self.felix_context.get('formatter')

        if not felix_system:
            return self.format_error("Felix system not initialized")

        if not felix_system.running:
            return self.format_error("Felix system is not running")

        if not session_manager or not session_id:
            return self.format_error("Session not initialized")

        # Parse options
        max_steps = int(kwargs.get('max-steps', kwargs.get('max_steps', 10)))
        web_search = kwargs.get('web-search', False) or kwargs.get('web_search', False)

        try:
            # Initialize orchestrator (creates if not exists in context)
            orchestrator = self.felix_context.get('orchestrator')
            if not orchestrator:
                orchestrator = CLIWorkflowOrchestrator(
                    felix_system=felix_system,
                    session_manager=session_manager,
                    formatter=formatter
                )
                self.felix_context['orchestrator'] = orchestrator

            # Execute workflow through orchestrator (this properly uses Felix architecture)
            if formatter and hasattr(formatter, 'create_progress'):
                with formatter.create_progress("Running multi-agent workflow...") as progress:
                    if hasattr(progress, 'add_task'):
                        task_id = progress.add_task("Felix agents collaborating...", total=None)

                    result = orchestrator.execute_workflow(
                        session_id=session_id,
                        task_input=task,
                        max_steps=max_steps,
                        web_search=web_search
                    )
            else:
                result = orchestrator.execute_workflow(
                    session_id=session_id,
                    task_input=task,
                    max_steps=max_steps,
                    web_search=web_search
                )

            # Extract results
            synthesis = result.get("centralpost_synthesis", {})
            content = synthesis.get("synthesis_content", "No result generated")
            confidence = synthesis.get("confidence", 0.0)
            agents_spawned = result.get("agents_spawned", [])
            workflow_id = result.get("workflow_id")

            # Clear progress line
            if formatter and hasattr(formatter, 'clear_line'):
                formatter.clear_line()

            # Format output with rich formatting if available
            if formatter and hasattr(formatter, 'print_workflow_result'):
                formatter.print_workflow_result(
                    content=content,
                    confidence=confidence,
                    metrics={
                        'agent_count': len(agents_spawned),
                        'tokens_used': result.get('tokens_used', 'N/A'),
                        'processing_time': result.get('execution_time', 0)
                    }
                )

            # Format output for tool result
            output_lines = [
                content,
                "",
                f"Confidence: {confidence:.2f}",
                f"Agents: {len(agents_spawned)}",
                f"Steps: {result.get('steps_executed', 0)}"
            ]

            output = "\n".join(output_lines)

            # Update context with last workflow ID for continuity
            self.felix_context['last_workflow_id'] = workflow_id

            # Return result
            return self.format_success(
                content=output,
                data={
                    'synthesis_content': content,
                    'confidence': confidence,
                    'agents_spawned': agents_spawned,
                    'agent_count': len(agents_spawned),
                    'workflow_id': workflow_id,
                    'execution_time': result.get('execution_time', 0)
                },
                workflow_id=workflow_id
            )

        except Exception as e:
            import traceback
            error_msg = f"Workflow execution failed: {str(e)}"
            if self.felix_context.get('verbose'):
                error_msg += f"\n\n{traceback.format_exc()}"
            return self.format_error(error_msg)

    def _continue_workflow(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Continue from the last workflow."""
        # Get last workflow ID from context
        last_workflow_id = self.felix_context.get('last_workflow_id')

        if not last_workflow_id:
            return self.format_error("No previous workflow to continue from")

        # For now, just inform user - full continuation requires more context
        return self.format_error(
            "Workflow continuation not yet implemented. "
            "Use '/workflow run' with your follow-up task instead."
        )

    def _show_workflow(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show details of a specific workflow."""
        if not args:
            # Show last workflow
            workflow_id = self.felix_context.get('last_workflow_id')
            if not workflow_id:
                return self.format_error("No workflow to show. Specify a workflow ID.")
        else:
            workflow_id = args[0]

        try:
            # Query workflow history
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflow = history.get_workflow(workflow_id)

            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            # Format workflow details
            output_lines = [
                f"Workflow: {workflow_id}",
                f"Task: {workflow.get('task_description', 'N/A')}",
                f"Status: {workflow.get('status', 'unknown')}",
                f"Confidence: {workflow.get('final_confidence', 0.0):.2f}",
                f"Created: {workflow.get('created_at', 'N/A')}",
                "",
                "Result:",
                workflow.get('synthesis_output', 'No output')
            ]

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to retrieve workflow: {str(e)}")
