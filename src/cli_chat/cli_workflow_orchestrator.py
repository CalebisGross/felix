"""
CLI Workflow Orchestrator for Felix

Thin wrapper around run_felix_workflow() that adds CLI-specific features:
- Session-workflow linking
- Progress updates for terminal
- Workflow ID tracking

The heavy lifting (context building, agent spawning, knowledge retrieval,
feedback broadcasting, etc.) is handled by run_felix_workflow() which already
has all the correct API usage.
"""

import time
import logging
from typing import Dict, Any, Optional, List

# Import core Felix workflow
from src.workflows.felix_workflow import run_felix_workflow

logger = logging.getLogger('felix_cli')


class CLIWorkflowOrchestrator:
    """
    Thin orchestration layer for CLI workflow execution.

    This class adds CLI-specific features on top of run_felix_workflow():
    - Links workflows to CLI sessions
    - Provides progress updates to terminal
    - Tracks workflow history for session
    - Adds session context metadata

    All the complex multi-agent coordination is handled by run_felix_workflow()
    which already uses the correct Felix architecture.
    """

    def __init__(
        self,
        felix_system,
        session_manager,
        formatter=None
    ):
        """
        Initialize CLI workflow orchestrator.

        Args:
            felix_system: Initialized FelixSystem instance
            session_manager: SessionManager for tracking conversation
            formatter: OutputFormatter for progress updates (optional)
        """
        self.felix_system = felix_system
        self.session_manager = session_manager
        self.formatter = formatter

        # Track workflow history for this session
        self.workflow_ids: List[str] = []

    def execute_workflow(
        self,
        session_id: str,
        task_input: str,
        max_steps: int = 10,
        web_search: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a workflow with CLI-specific enhancements.

        This method:
        1. Gets parent workflow ID for conversation continuity
        2. Calls run_felix_workflow() which does ALL the proper architecture work
        3. Links workflow to session for CLI history
        4. Tracks workflow IDs for this session

        Args:
            session_id: Current CLI session ID
            task_input: User query or task description
            max_steps: Maximum workflow steps
            web_search: Enable web search for this workflow

        Returns:
            Dictionary with workflow results from run_felix_workflow()
        """

        workflow_start_time = time.time()
        original_task_input = task_input  # Store for later

        try:
            # Show progress if formatter available
            if self.formatter and hasattr(self.formatter, 'print_progress'):
                self.formatter.print_progress(f"Executing Felix multi-agent workflow...")

            # Get parent workflow ID for conversation continuity
            parent_workflow_id = self._get_parent_workflow_id(session_id)

            # Execute workflow through Felix's proper architecture
            # run_felix_workflow() handles:
            # - CollaborativeContextBuilder with correct APIs
            # - Agent spawning via AgentFactory
            # - CentralPost coordination
            # - Helical progression
            # - Knowledge retrieval with meta-learning
            # - Synthesis feedback broadcasting
            # - Concept registry creation and management
            result = run_felix_workflow(
                felix_system=self.felix_system,
                task_input=task_input,
                progress_callback=self._progress_callback,
                max_steps_override=max_steps,
                parent_workflow_id=parent_workflow_id
            )

            # Extract workflow ID
            workflow_id = result.get('workflow_id', f"workflow_unknown")

            # Track this workflow in session history
            self.workflow_ids.append(workflow_id)

            # Link workflow to session (CLI-specific feature)
            self._link_workflow_to_session(
                session_id=session_id,
                workflow_id=workflow_id,
                result=result
            )

            # Add CLI-specific metadata
            result['execution_time'] = time.time() - workflow_start_time
            result['session_id'] = session_id
            result['cli_metadata'] = {
                'parent_workflow_id': parent_workflow_id,
                'workflows_in_session': len(self.workflow_ids)
            }

            logger.info(f"CLI workflow {workflow_id} completed in {result['execution_time']:.2f}s")

            return result

        except Exception as e:
            logger.error(f"CLI workflow orchestration failed: {str(e)}", exc_info=True)
            raise

    def _get_parent_workflow_id(self, session_id: str) -> Optional[str]:
        """
        Get the most recent workflow ID from session for continuity.

        Args:
            session_id: Current session ID

        Returns:
            Parent workflow ID or None
        """

        # First check our tracked workflow IDs
        if self.workflow_ids:
            return self.workflow_ids[-1]

        # Fall back to session manager
        return self.session_manager.get_last_workflow_id(session_id)

    def _link_workflow_to_session(
        self,
        session_id: str,
        workflow_id: str,
        result: Dict[str, Any]
    ):
        """
        Link workflow execution to CLI session.

        This ensures session history includes workflow metadata and
        enables proper conversation continuity.

        Args:
            session_id: Current session ID
            workflow_id: Workflow ID to link
            result: Workflow execution result
        """

        try:
            # Add workflow metadata message to session
            synthesis = result.get('centralpost_synthesis', {})

            # Store as assistant message with workflow_id reference
            self.session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=synthesis.get('synthesis_content', ''),
                workflow_id=workflow_id
            )

            logger.info(f"Linked workflow {workflow_id} to session {session_id}")

        except Exception as e:
            logger.warning(f"Failed to link workflow to session: {e}")

    def _progress_callback(self, status: str, progress: float):
        """
        Handle progress updates from workflow execution.

        Args:
            status: Status message
            progress: Progress percentage (0.0-1.0)
        """

        if self.formatter and hasattr(self.formatter, 'print_progress'):
            self.formatter.print_progress(status)

    def export_session_state(self) -> Dict[str, Any]:
        """
        Export complete session state including workflow history.

        Returns:
            Dictionary with session state
        """

        return {
            'workflow_ids': self.workflow_ids,
            'total_workflows': len(self.workflow_ids)
        }
