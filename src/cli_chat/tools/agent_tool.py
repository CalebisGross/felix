"""
Agent Tool for inspecting agent decisions and contributions in conversational CLI.
"""

from typing import Dict, Any, List, Optional
import json
from .base_tool import BaseTool, ToolResult


class AgentTool(BaseTool):
    """Tool for agent introspection and debugging."""

    @property
    def name(self) -> str:
        return "agent"

    @property
    def description(self) -> str:
        return "Inspect agent decisions, contributions, and behavior"

    @property
    def usage(self) -> str:
        return "/agent [list|show|contributions] <workflow_id>"

    def execute(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """
        Execute agent command.

        Commands:
            /agent list <workflow_id>           - List agents in a workflow
            /agent show <workflow_id> <agent>   - Show specific agent details
            /agent contributions <workflow_id>  - Show all agent contributions
        """
        if not args:
            return self.format_error("Usage: /agent list <workflow_id>")

        command = args[0]

        if command == "list":
            return self._list_agents(args[1:], kwargs)
        elif command == "show":
            return self._show_agent(args[1:], kwargs)
        elif command == "contributions":
            return self._show_contributions(args[1:], kwargs)
        else:
            # If not a command, assume it's a workflow ID for listing
            return self._list_agents(args, kwargs)

    def _list_agents(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """List all agents that participated in a workflow."""
        # Get workflow ID
        if not args:
            workflow_id = self.felix_context.get('last_workflow_id')
            if not workflow_id:
                return self.format_error("Usage: /agent list <workflow_id>")
        else:
            workflow_id = args[0]

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflow = history.get_workflow_output(int(workflow_id))

            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            # Extract agent information from metadata
            try:
                metadata = json.loads(workflow.metadata)
            except:
                metadata = {}

            agents_spawned = metadata.get('agents_spawned', [])

            if not agents_spawned:
                return self.format_success(f"No agent information available for workflow {workflow_id}")

            # Format output
            output_lines = [
                f"Agents in Workflow {workflow_id}",
                "=" * 60,
                f"Task: {workflow.task_input}",
                "",
                "Agents:"
            ]

            for i, agent in enumerate(agents_spawned, 1):
                if isinstance(agent, dict):
                    agent_type = agent.get('type', agent.get('agent_type', 'Unknown'))
                    agent_id = agent.get('agent_id', agent.get('id', 'unknown'))
                    confidence = agent.get('confidence', 'N/A')
                    position = agent.get('position', 'N/A')

                    output_lines.append(f"\n{i}. {agent_type} (ID: {agent_id})")
                    if confidence != 'N/A':
                        output_lines.append(f"   Confidence: {confidence}")
                    if position != 'N/A':
                        output_lines.append(f"   Position: {position}")
                else:
                    # Simple string representation
                    output_lines.append(f"\n{i}. {agent}")

            output_lines.append(f"\nTotal agents: {len(agents_spawned)}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to list agents: {str(e)}")

    def _show_agent(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show detailed information about a specific agent."""
        if len(args) < 2:
            return self.format_error("Usage: /agent show <workflow_id> <agent_id>")

        workflow_id = args[0]
        agent_identifier = args[1]

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflow = history.get_workflow_output(int(workflow_id))

            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            # Extract agent information from metadata
            try:
                metadata = json.loads(workflow.metadata)
            except:
                metadata = {}

            agents_spawned = metadata.get('agents_spawned', [])
            agent_details = metadata.get('agent_details', {})

            # Find the agent
            target_agent = None
            for agent in agents_spawned:
                if isinstance(agent, dict):
                    agent_id = agent.get('agent_id', agent.get('id', ''))
                    agent_type = agent.get('type', agent.get('agent_type', ''))

                    if agent_id == agent_identifier or agent_type.lower() == agent_identifier.lower():
                        target_agent = agent
                        break

            if not target_agent:
                return self.format_error(f"Agent not found: {agent_identifier}")

            # Format output
            output_lines = [
                f"Agent Details",
                "=" * 60,
                ""
            ]

            for key, value in target_agent.items():
                output_lines.append(f"{key}: {value}")

            # Add additional details if available
            agent_id = target_agent.get('agent_id', target_agent.get('id'))
            if agent_id and agent_id in agent_details:
                details = agent_details[agent_id]
                output_lines.append("\nAdditional Details:")
                for key, value in details.items():
                    if isinstance(value, (str, int, float, bool)):
                        output_lines.append(f"  {key}: {value}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to show agent: {str(e)}")

    def _show_contributions(self, args: List[str], kwargs: Dict[str, Any]) -> ToolResult:
        """Show all agent contributions to a workflow."""
        # Get workflow ID
        if not args:
            workflow_id = self.felix_context.get('last_workflow_id')
            if not workflow_id:
                return self.format_error("Usage: /agent contributions <workflow_id>")
        else:
            workflow_id = args[0]

        try:
            from src.memory.workflow_history import WorkflowHistory

            history = WorkflowHistory()
            workflow = history.get_workflow_output(int(workflow_id))

            if not workflow:
                return self.format_error(f"Workflow not found: {workflow_id}")

            # Extract agent information from metadata
            try:
                metadata = json.loads(workflow.metadata)
            except:
                metadata = {}

            # Get agent outputs/contributions
            agent_outputs = metadata.get('agent_outputs', {})
            agent_contributions = metadata.get('agent_contributions', [])
            agents_spawned = metadata.get('agents_spawned', [])

            if not agent_outputs and not agent_contributions:
                return self.format_success(
                    f"No agent contribution details available for workflow {workflow_id}\n"
                    f"Agents participated: {len(agents_spawned)}"
                )

            # Format output
            output_lines = [
                f"Agent Contributions - Workflow {workflow_id}",
                "=" * 60,
                f"Task: {workflow.task_input}",
                ""
            ]

            # Show contributions
            if agent_contributions:
                for i, contrib in enumerate(agent_contributions, 1):
                    agent_type = contrib.get('agent_type', 'Unknown')
                    agent_id = contrib.get('agent_id', 'unknown')
                    contribution = contrib.get('contribution', contrib.get('output', 'N/A'))
                    confidence = contrib.get('confidence', 'N/A')

                    output_lines.append(f"\n{i}. {agent_type} (ID: {agent_id})")
                    if confidence != 'N/A':
                        output_lines.append(f"   Confidence: {confidence}")
                    output_lines.append(f"   Contribution:")
                    # Truncate long contributions
                    if len(contribution) > 200:
                        output_lines.append(f"   {contribution[:200]}...")
                    else:
                        output_lines.append(f"   {contribution}")

            # Show outputs
            elif agent_outputs:
                for agent_id, output in agent_outputs.items():
                    output_lines.append(f"\nAgent {agent_id}:")
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if isinstance(value, str) and len(value) > 200:
                                value = value[:200] + "..."
                            output_lines.append(f"  {key}: {value}")
                    else:
                        if len(str(output)) > 200:
                            output_lines.append(f"  {str(output)[:200]}...")
                        else:
                            output_lines.append(f"  {output}")

            return self.format_success("\n".join(output_lines))

        except Exception as e:
            return self.format_error(f"Failed to show contributions: {str(e)}")
