"""
System Command Manager for the Felix Framework.

Handles system command execution, approval workflows, trust management,
and real-time command streaming for the Felix multi-agent system.

Key Features:
- Trust-based command classification (BLOCKED/SAFE/REVIEW)
- Approval workflow management with multiple decision types
- Command deduplication within workflows
- Auto-approval rules for workflow-scoped commands
- Real-time command output streaming for GUI Terminal tab
- CommandHistory database integration for audit trail

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

# Import system execution components
from src.execution import (
    SystemExecutor, TrustManager, CommandHistory,
    TrustLevel, CommandResult
)
from src.execution.approval_manager import ApprovalManager, ApprovalDecision

# Import message types for broadcasting
from src.communication.message_types import Message, MessageType

# Set up logging
logger = logging.getLogger(__name__)


class SystemCommandManager:
    """
    Manages system command execution, approvals, and streaming for Felix agents.

    Responsibilities:
    - Command trust classification (BLOCKED/SAFE/REVIEW)
    - Immediate execution of SAFE commands
    - Approval workflow for REVIEW commands
    - Command deduplication within workflows
    - Real-time streaming of command output
    - Broadcasting of command events (start/output/complete)
    """

    def __init__(self,
                 system_executor: SystemExecutor,
                 trust_manager: TrustManager,
                 command_history: CommandHistory,
                 approval_manager: ApprovalManager,
                 agent_registry: Any,  # AgentRegistry reference
                 message_queue_callback: Callable[[Message], None],
                 config: Any = None):  # Felix configuration for auto-approval
        """
        Initialize System Command Manager.

        Args:
            system_executor: Backend for command execution
            trust_manager: Command trust classification
            command_history: Database for execution tracking
            approval_manager: Approval workflow management
            agent_registry: AgentRegistry for agent info lookup
            message_queue_callback: Callback to queue messages (central_post.queue_message)
            config: Optional FelixConfig for checking auto-approval settings
        """
        self.system_executor = system_executor
        self.trust_manager = trust_manager
        self.command_history = command_history
        self.approval_manager = approval_manager
        self.agent_registry = agent_registry
        self._queue_message = message_queue_callback
        self.config = config  # Store config for auto-approval checks

        # Action tracking
        self._action_results: Dict[str, CommandResult] = {}
        self._action_id_counter: int = 0
        self._action_approvals: Dict[str, str] = {}  # action_id -> approval_id
        self._approval_events: Dict[str, threading.Event] = {}  # approval_id -> Event

        # Command deduplication (workflow-scoped)
        self._executed_commands: Dict[str, Dict[str, CommandResult]] = {}  # workflow_id -> {hash -> result}

        # Live command output for Terminal tab streaming
        self._live_command_outputs: Dict[int, List[tuple]] = {}  # execution_id -> [(line, stream_type)]

        logger.info("✓ SystemCommandManager initialized")

    def request_system_action(self, agent_id: str, command: str,
                             context: str = "", workflow_id: Optional[str] = None) -> str:
        """
        Agent requests system action (command execution).

        Args:
            agent_id: ID of requesting agent
            command: Command to execute
            context: Context/reason for command
            workflow_id: Associated workflow ID

        Returns:
            action_id for tracking the request
        """
        logger.info(f"System action requested by {agent_id}: {command}")
        if context:
            logger.info(f"  Context: {context}")

        # Generate action ID
        self._action_id_counter += 1
        action_id = f"action_{self._action_id_counter:04d}"

        # Classify command by trust level
        trust_level = self.trust_manager.classify_command(command)

        logger.info(f"  Trust level: {trust_level.value}")
        logger.info(f"  Action ID: {action_id}")

        if trust_level == TrustLevel.BLOCKED:
            return self._handle_blocked_command(action_id, agent_id, command)

        elif trust_level == TrustLevel.SAFE:
            return self._handle_safe_command(action_id, agent_id, command, context, workflow_id, trust_level)

        else:  # TrustLevel.REVIEW
            return self._handle_review_command(action_id, agent_id, command, context, workflow_id, trust_level)

    def _handle_blocked_command(self, action_id: str, agent_id: str, command: str) -> str:
        """Handle BLOCKED commands - deny immediately."""
        logger.warning(f"✗ Command BLOCKED: {command}")

        # Create denial message
        self._broadcast_action_denial(action_id, agent_id, command, "Command is blocked by trust policy")

        # Store denial result
        result = CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr="Command blocked by trust policy",
            duration=0.0,
            success=False,
            error_category=None,
            cwd=str(self.system_executor.default_cwd),
            venv_active=False
        )
        self._action_results[action_id] = result

        return action_id

    def _handle_safe_command(self, action_id: str, agent_id: str, command: str,
                            context: str, workflow_id: Optional[str],
                            trust_level: TrustLevel) -> str:
        """Handle SAFE commands - execute immediately with streaming."""
        logger.info(f"✓ Executing SAFE command immediately (streaming)")

        # Execute with streaming and store result
        result = self._execute_command_with_streaming(
            action_id=action_id,
            command=command,
            agent_id=agent_id,
            context=context,
            workflow_id=workflow_id,
            trust_level=trust_level,
            approved_by="auto"
        )

        # Log command output for visibility
        logger.info(f"📤 Command result broadcast:")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Exit code: {result.exit_code}")
        logger.info(f"   Duration: {result.duration:.2f}s")
        if result.stdout:
            logger.info(f"   Output: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"   Errors: {result.stderr[:500]}")

        return action_id

    def _handle_review_command(self, action_id: str, agent_id: str, command: str,
                              context: str, workflow_id: Optional[str],
                              trust_level: TrustLevel) -> str:
        """Handle REVIEW commands - check deduplication, auto-approval, or request approval."""
        logger.info(f"⚠ Command requires APPROVAL")

        # 1. Check for command deduplication within workflow
        if workflow_id:
            command_hash = self.system_executor.compute_command_hash(command)

            # Check if already executed in this workflow
            if workflow_id in self._executed_commands:
                if command_hash in self._executed_commands[workflow_id]:
                    cached_result = self._executed_commands[workflow_id][command_hash]
                    logger.info(f"⚡ Command already executed in workflow (using cached result)")
                    logger.info(f"   Previous execution: success={cached_result.success}, exit_code={cached_result.exit_code}")

                    # Return cached result
                    self._action_results[action_id] = cached_result

                    # Broadcast cached result with deduplication notice
                    self._broadcast_action_result(action_id, agent_id, command, cached_result)

                    return action_id

        # 2. Check for workflow-scoped auto-approval rules
        auto_approve_rule = self.approval_manager.check_auto_approve(command, workflow_id)

        if auto_approve_rule:
            logger.info(f"⚡ Command auto-approved by workflow rule: {auto_approve_rule} (streaming)")

            # Execute with auto-approval
            result = self._execute_command_with_streaming(
                action_id=action_id,
                command=command,
                agent_id=agent_id,
                context=context,
                workflow_id=workflow_id,
                trust_level=trust_level,
                approved_by="auto_rule"
            )

            # Cache in workflow deduplication
            if workflow_id:
                command_hash = self.system_executor.compute_command_hash(command)
                if workflow_id not in self._executed_commands:
                    self._executed_commands[workflow_id] = {}
                self._executed_commands[workflow_id][command_hash] = result

            # Log output
            logger.info(f"📤 Auto-approved command result:")
            logger.info(f"   Success: {result.success}")
            logger.info(f"   Exit code: {result.exit_code}")
            if result.stdout:
                logger.info(f"   Output: {result.stdout[:500]}")

            return action_id

        # 2.5. Check for global auto-approval flag (CLI mode)
        if self._should_auto_approve():
            logger.info(f"⚡ CLI mode: auto-approving REVIEW command without blocking")

            # Execute immediately as if it were a SAFE command
            result = self._execute_command_with_streaming(
                action_id=action_id,
                command=command,
                agent_id=agent_id,
                context=context,
                workflow_id=workflow_id,
                trust_level=trust_level,
                approved_by="cli_auto_approve"
            )

            # Cache in workflow deduplication
            if workflow_id:
                command_hash = self.system_executor.compute_command_hash(command)
                if workflow_id not in self._executed_commands:
                    self._executed_commands[workflow_id] = {}
                self._executed_commands[workflow_id][command_hash] = result

            logger.info(f"📤 Auto-approved command result (CLI mode):")
            logger.info(f"   Success: {result.success}")
            logger.info(f"   Exit code: {result.exit_code}")

            return action_id

        # 3. Request user approval via ApprovalManager
        approval_id = self.approval_manager.request_approval(
            command=command,
            agent_id=agent_id,
            context=context,
            trust_level=trust_level,
            workflow_id=workflow_id
        )

        # Track action -> approval mapping
        self._action_approvals[action_id] = approval_id

        # Create threading event for workflow to wait on
        approval_event = threading.Event()
        self._approval_events[approval_id] = approval_event

        logger.info(f"  Approval ID: {approval_id}")
        logger.info(f"  Workflow ID: {workflow_id or 'None'}")

        # Broadcast approval needed message
        self._broadcast_approval_needed(action_id, approval_id, agent_id, command, context)

        return action_id

    def _should_auto_approve(self) -> bool:
        """
        Check if auto-approval is enabled via config.

        Returns:
            True if auto_approve_system_actions config flag is set (CLI mode), False otherwise
        """
        if self.config and hasattr(self.config, 'auto_approve_system_actions'):
            return self.config.auto_approve_system_actions
        return False

    def _execute_command_with_streaming(self, action_id: str, command: str, agent_id: str,
                                       context: str, workflow_id: Optional[str],
                                       trust_level: TrustLevel, approved_by: str) -> CommandResult:
        """
        Execute command with streaming output - single consolidated implementation.

        This method replaces 3 duplicate patterns that existed for SAFE/auto-approved/user-approved commands.

        Returns:
            CommandResult from execution
        """
        # Create execution placeholder to get execution_id
        command_hash = self.system_executor.compute_command_hash(command)
        agent_info = self.agent_registry.get_agent_info(agent_id)
        agent_type = agent_info.get('metadata', {}).get('agent_type') if agent_info else None

        execution_id = self.command_history.create_execution_placeholder(
            command=command,
            command_hash=command_hash,
            agent_id=agent_id,
            agent_type=agent_type,
            workflow_id=workflow_id,
            trust_level=trust_level,
            approved_by=approved_by,
            context=context
        )

        # Broadcast command start
        self._broadcast_command_start(action_id, execution_id, command, agent_id, context)

        # Define output callback for streaming
        def output_callback(line: str, stream_type: str):
            self._broadcast_command_output(action_id, execution_id, line, stream_type)

        # Execute with streaming
        result = self.system_executor.execute_command_streaming(
            command=command,
            context=context,
            output_callback=output_callback
        )

        # Update database with final result
        self.command_history.update_execution_result(execution_id, result)

        # Broadcast command complete
        self._broadcast_command_complete(action_id, execution_id, result)

        # Store result for retrieval
        self._action_results[action_id] = result

        # Broadcast result (for agent awareness)
        self._broadcast_action_result(action_id, agent_id, command, result)

        return result

    def get_action_result(self, action_id: str) -> Optional[CommandResult]:
        """
        Get result of a system action.

        Args:
            action_id: Action ID to query

        Returns:
            CommandResult if available, None otherwise
        """
        return self._action_results.get(action_id)

    def wait_for_approval(self, action_id: str, timeout: float = 300.0) -> Optional[CommandResult]:
        """
        Wait for approval to be processed and return result.

        This method blocks until the approval is processed (approved or denied)
        or the timeout is reached. Used by workflows to pause execution while
        waiting for user approval.

        Args:
            action_id: Action ID to wait for
            timeout: Maximum seconds to wait (default 300 = 5 minutes)

        Returns:
            CommandResult if approval processed, None if timeout or not found
        """
        # Check if action already has result (SAFE commands, cached, auto-approved)
        result = self._action_results.get(action_id)
        if result is not None:
            return result

        # Get approval_id for this action
        approval_id = self._action_approvals.get(action_id)
        if not approval_id:
            logger.warning(f"No approval_id found for action {action_id}")
            return None

        # Get the event for this approval
        event = self._approval_events.get(approval_id)
        if not event:
            logger.warning(f"No approval event found for {approval_id}")
            return None

        logger.info(f"⏸️  Waiting for approval: {approval_id} (timeout: {timeout}s)")

        # Wait for event to be signaled (blocks workflow thread)
        if event.wait(timeout):
            # Approval processed - get result
            result = self._action_results.get(action_id)
            if result:
                logger.info(f"✓ Approval processed, action completed: success={result.success}")
            else:
                logger.warning(f"⚠️  Approval processed but no result found for {action_id}")
            return result
        else:
            # Timeout
            logger.error(f"❌ Approval timeout after {timeout}s for {approval_id}")
            return None

    def approve_system_action(self, approval_id: str, decision: ApprovalDecision,
                             decided_by: str = "user") -> bool:
        """
        Approve a pending system action with specific decision type.

        This method integrates with ApprovalManager to handle approval decisions
        including "always approve" rules that apply for the current workflow session.

        Args:
            approval_id: Approval request ID
            decision: Type of approval decision (ApprovalDecision enum)
            decided_by: Who made the decision (default: "user")

        Returns:
            True if approved and executed successfully, False otherwise
        """
        logger.info(f"=" * 60)
        logger.info(f"Processing approval decision: {approval_id}")
        logger.info(f"  Decision: {decision.value}")
        logger.info(f"  Decided by: {decided_by}")

        # Process decision in ApprovalManager
        success = self.approval_manager.decide_approval(
            approval_id=approval_id,
            decision=decision,
            decided_by=decided_by
        )

        if not success:
            logger.error(f"Failed to process approval decision: {approval_id}")
            return False

        # Get approval request details
        approval_request = self.approval_manager.get_approval_status(approval_id)

        if not approval_request:
            logger.error(f"Approval request not found: {approval_id}")
            return False

        # Handle denial
        if decision == ApprovalDecision.DENY:
            return self._handle_approval_denial(approval_id, approval_request)

        # Execute approved command
        return self._handle_approval_execution(approval_id, approval_request, decided_by)

    def _handle_approval_denial(self, approval_id: str, approval_request: Any) -> bool:
        """Handle denial of approval request."""
        logger.info(f"✗ Command DENIED by user")

        # Find corresponding action_id
        action_id = None
        for aid, apid in self._action_approvals.items():
            if apid == approval_id:
                action_id = aid
                break

        if action_id:
            # Create denial result
            denial_result = CommandResult(
                command=approval_request.command,
                exit_code=-1,
                stdout="",
                stderr="Command denied by user",
                duration=0.0,
                success=False,
                error_category=None,
                cwd=str(self.system_executor.default_cwd),
                venv_active=False
            )

            # Store denial result
            self._action_results[action_id] = denial_result

            # Broadcast denial
            self._broadcast_action_denial(
                action_id=action_id,
                agent_id=approval_request.agent_id,
                command=approval_request.command,
                reason="Denied by user"
            )

        # Signal waiting workflow thread that approval is processed (denied)
        if approval_id in self._approval_events:
            self._approval_events[approval_id].set()
            logger.info(f"▶️  Signaled workflow to resume (command denied)")
            # Clean up event
            del self._approval_events[approval_id]

        return True

    def _handle_approval_execution(self, approval_id: str, approval_request: Any, decided_by: str) -> bool:
        """Execute approved command with streaming."""
        logger.info(f"✓ Executing approved command: {approval_request.command} (streaming)")

        # Find corresponding action_id before execution
        action_id = None
        for aid, apid in self._action_approvals.items():
            if apid == approval_id:
                action_id = aid
                break

        if not action_id:
            # Generate new action_id if mapping not found
            self._action_id_counter += 1
            action_id = f"action_{self._action_id_counter:04d}"
            logger.warning(f"No action_id mapping found, generated new: {action_id}")

        # Execute with streaming
        result = self._execute_command_with_streaming(
            action_id=action_id,
            command=approval_request.command,
            agent_id=approval_request.agent_id,
            context=approval_request.context,
            workflow_id=approval_request.workflow_id,
            trust_level=approval_request.trust_level,
            approved_by=decided_by
        )

        # Cache in workflow deduplication
        if approval_request.workflow_id:
            command_hash = self.system_executor.compute_command_hash(approval_request.command)
            if approval_request.workflow_id not in self._executed_commands:
                self._executed_commands[approval_request.workflow_id] = {}
            self._executed_commands[approval_request.workflow_id][command_hash] = result

        # Log output
        logger.info(f"📤 Approved command executed:")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Exit code: {result.exit_code}")
        logger.info(f"   Duration: {result.duration:.2f}s")
        if result.stdout:
            logger.info(f"   Output: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"   Errors: {result.stderr[:500]}")
        logger.info(f"=" * 60)

        # Signal waiting workflow thread that approval is processed
        if approval_id in self._approval_events:
            self._approval_events[approval_id].set()
            logger.info(f"▶️  Signaled workflow to resume (approval processed)")
            # Clean up event
            del self._approval_events[approval_id]

        return True

    def approve_action(self, approval_id: str, approver: str = "user") -> bool:
        """
        Approve a pending system action (legacy method).

        Args:
            approval_id: Approval request ID
            approver: Who approved (default "user")

        Returns:
            True if approved and executed successfully
        """
        logger.info(f"Approving action: {approval_id}")

        # Approve in trust manager
        success = self.trust_manager.approve_command(approval_id, approver)

        if not success:
            logger.error(f"Failed to approve: {approval_id}")
            return False

        # Get approval request details
        request = self.trust_manager.get_approval_status(approval_id)

        if not request:
            logger.error(f"Approval request not found: {approval_id}")
            return False

        # Execute the command
        logger.info(f"Executing approved command: {request.command}")

        result = self.system_executor.execute_command(
            command=request.command,
            context=request.context
        )

        # Store in database
        command_hash = self.system_executor.compute_command_hash(request.command)
        self.command_history.record_execution(
            command=request.command,
            command_hash=command_hash,
            result=result,
            agent_id=request.agent_id,
            agent_type=None,  # Will be looked up if needed
            workflow_id=None,  # Not tracked for approvals
            trust_level=request.trust_level,
            approved_by=approver,
            context=request.context
        )

        # Find corresponding action_id (search by approval_id)
        # For now, generate a new action_id
        self._action_id_counter += 1
        action_id = f"action_{self._action_id_counter:04d}"

        # Store result
        self._action_results[action_id] = result

        # Broadcast result
        self._broadcast_action_result(action_id, request.agent_id, request.command, result)

        return True

    def get_pending_actions(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of pending action approvals.

        Args:
            workflow_id: Optional workflow ID to filter approvals

        Returns:
            List of pending approval dictionaries
        """
        pending = self.approval_manager.get_pending_approvals(workflow_id=workflow_id)

        return [{
            'approval_id': req.approval_id,
            'command': req.command,
            'agent_id': req.agent_id,
            'context': req.context,
            'trust_level': req.trust_level.value,
            'risk_assessment': req.risk_assessment,
            'requested_at': req.requested_at,
            'expires_at': req.expires_at,
            'workflow_id': req.workflow_id
        } for req in pending]

    def handle_system_action_request(self, message: Message) -> None:
        """
        Handle system action request from agent.

        Args:
            message: SYSTEM_ACTION_REQUEST message
        """
        agent_id = message.sender_id
        command = message.content.get('command', '')
        context = message.content.get('context', '')
        workflow_id = message.content.get('workflow_id')

        if not command:
            logger.warning(f"Empty command in action request from {agent_id}")
            return

        # Request action (will handle classification and execution/approval)
        action_id = self.request_system_action(agent_id, command, context, workflow_id)

        logger.info(f"System action request processed: {action_id}")

    def get_live_command_output(self, execution_id: int) -> List[tuple]:
        """
        Get accumulated live output for a command execution.

        Used by Terminal tab to poll for real-time command output during execution.

        Args:
            execution_id: Database execution ID

        Returns:
            List of (output_line, stream_type) tuples, or empty list if none available
        """
        return self._live_command_outputs.get(execution_id, [])

    def get_workflow_command_status(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get command execution status for a workflow.

        Args:
            workflow_id: Optional workflow ID to filter by. If None, returns all actions.

        Returns:
            Dictionary with:
                - completed_count: Number of completed commands
                - pending_count: Number of pending approvals
                - all_succeeded: Whether all completed commands succeeded
                - results: List of CommandResult objects for completed commands
                - pending: List of pending action info dictionaries

        Example:
            >>> status = manager.get_workflow_command_status(workflow_id)
            >>> if status['all_succeeded'] and status['completed_count'] > 0:
            >>>     print("All commands completed successfully!")
        """
        # Filter completed action results by workflow_id if provided
        if workflow_id:
            results = [r for r in self._action_results.values()
                      if hasattr(r, 'workflow_id') and r.workflow_id == workflow_id]
        else:
            results = list(self._action_results.values())

        # Get pending actions filtered by workflow_id
        pending = self.get_pending_actions(workflow_id)

        return {
            'completed_count': len(results),
            'pending_count': len(pending),
            'all_succeeded': all(r.success for r in results) if results else False,
            'results': results,
            'pending': pending
        }

    def get_workflow_executed_commands(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Get all commands that have been executed in a specific workflow.

        This provides visibility into what system actions have already completed,
        allowing agents to avoid repeating commands and coordinate better.

        Args:
            workflow_id: Workflow ID to query

        Returns:
            List of dictionaries containing:
                - command: The command that was executed
                - result: CommandResult object with success status and output
                - timestamp: When the command was executed

        Example:
            >>> executed = manager.get_workflow_executed_commands(workflow_id)
            >>> for cmd in executed:
            >>>     print(f"Executed: {cmd['command']} - Success: {cmd['result'].success}")
        """
        if workflow_id not in self._executed_commands:
            return []

        executed_list = []
        for command_hash, result in self._executed_commands[workflow_id].items():
            # Extract command from result if available
            command_str = getattr(result, 'command', f"<hash:{command_hash}>")
            executed_list.append({
                'command': command_str,
                'result': result,
                'timestamp': getattr(result, 'end_time', None)
            })

        return executed_list

    # =============================================================================
    # BROADCAST METHODS - Unified pattern for message broadcasting
    # =============================================================================

    def _broadcast_action_result(self, action_id: str, agent_id: str,
                                 command: str, result: CommandResult) -> None:
        """Broadcast action result back to requesting agent."""
        result_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_RESULT,
            content={
                'action_id': action_id,
                'target_agent': agent_id,
                'command': command,
                'success': result.success,
                'exit_code': result.exit_code,
                'stdout': result.stdout[:500],  # Preview
                'stderr': result.stderr[:500],
                'duration': result.duration,
                'error_category': result.error_category.value if result.error_category else None
            },
            timestamp=time.time()
        )

        self._queue_message(result_message)

    def _broadcast_approval_needed(self, action_id: str, approval_id: str,
                                   agent_id: str, command: str, context: str) -> None:
        """Broadcast that a command needs approval."""
        approval_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_APPROVAL_NEEDED,
            content={
                'action_id': action_id,
                'approval_id': approval_id,
                'agent_id': agent_id,
                'command': command,
                'context': context
            },
            timestamp=time.time()
        )

        self._queue_message(approval_message)

    def _broadcast_action_denial(self, action_id: str, agent_id: str,
                                command: str, reason: str) -> None:
        """Broadcast that a command was denied."""
        denial_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_DENIED,
            content={
                'action_id': action_id,
                'target_agent': agent_id,
                'command': command,
                'reason': reason
            },
            timestamp=time.time()
        )

        self._queue_message(denial_message)

    def _broadcast_command_start(self, action_id: str, execution_id: int,
                                 command: str, agent_id: str, context: str = "") -> None:
        """Broadcast that command execution has started."""
        # Initialize live output buffer immediately so Terminal can poll it
        self._live_command_outputs[execution_id] = []

        start_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_START,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'command': command,
                'agent_id': agent_id,
                'context': context,
                'status': 'running'
            },
            timestamp=time.time()
        )

        self._queue_message(start_message)
        logger.debug(f"📡 Broadcast: Command started - {action_id}")

    def _broadcast_command_output(self, action_id: str, execution_id: int,
                                  output_line: str, stream_type: str) -> None:
        """Broadcast real-time command output line."""
        # Store in live output buffer for Terminal tab polling
        if execution_id not in self._live_command_outputs:
            self._live_command_outputs[execution_id] = []
        self._live_command_outputs[execution_id].append((output_line, stream_type))

        output_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_OUTPUT,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'output_line': output_line,
                'stream_type': stream_type
            },
            timestamp=time.time()
        )

        self._queue_message(output_message)

    def _broadcast_command_complete(self, action_id: str, execution_id: int,
                                    result: CommandResult) -> None:
        """Broadcast that command execution has completed."""
        # Schedule cleanup of live output buffer (5 seconds delay for Terminal tab final poll)
        if execution_id in self._live_command_outputs:
            threading.Timer(5.0, lambda: self._live_command_outputs.pop(execution_id, None)).start()

        complete_message = Message(
            sender_id="central_post",
            message_type=MessageType.SYSTEM_ACTION_COMPLETE,
            content={
                'action_id': action_id,
                'execution_id': execution_id,
                'success': result.success,
                'exit_code': result.exit_code,
                'duration': result.duration,
                'status': 'completed' if result.success else 'failed',
                'error_category': result.error_category.value if result.error_category else None
            },
            timestamp=time.time()
        )

        self._queue_message(complete_message)
        logger.debug(f"📡 Broadcast: Command completed - {action_id} (success={result.success})")
