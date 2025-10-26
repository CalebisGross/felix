"""
Approval Manager for Felix System Autonomy

Manages user approval workflow for system commands classified as REVIEW.
Supports per-workflow "always approve" rules and command deduplication.

Key Features:
- Approval request queue with expiration
- Per-workflow approval rules (session-scoped)
- Multiple approval decision types (once, always command, always path pattern)
- Thread-safe for concurrent approval requests
- Automatic cleanup of expired approvals
"""

import time
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

from src.execution.trust_manager import TrustLevel

logger = logging.getLogger(__name__)


class ApprovalDecision(Enum):
    """Types of approval decisions user can make."""
    APPROVE_ONCE = "approve_once"  # Execute this command only
    APPROVE_ALWAYS_EXACT = "approve_always_exact"  # Always approve this exact command
    APPROVE_ALWAYS_COMMAND = "approve_always_command"  # Always approve this command type (e.g., all mkdir)
    APPROVE_ALWAYS_PATH_PATTERN = "approve_always_path_pattern"  # Always approve command in path pattern
    DENY = "deny"  # Reject this command


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ApprovalRequest:
    """
    Represents a request for user approval of a system command.
    """
    approval_id: str
    command: str
    agent_id: str
    context: str
    trust_level: TrustLevel
    workflow_id: Optional[str]
    risk_assessment: str
    requested_at: float
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: Optional[float] = None
    decided_by: Optional[str] = None  # "user" or "auto_rule"
    decision: Optional[ApprovalDecision] = None
    expires_at: float = field(default_factory=lambda: time.time() + 300)  # 5 minutes


@dataclass
class WorkflowApprovalRule:
    """
    Represents an "always approve" rule for a specific workflow session.
    Rules are scoped to workflow and expire when workflow completes.
    """
    rule_id: str
    workflow_id: str
    rule_type: ApprovalDecision  # What kind of always-approve rule
    command_pattern: str  # Pattern to match (exact command, command type, or path pattern)
    created_at: float
    created_by_approval: str  # Original approval_id that created this rule

    def matches(self, command: str) -> bool:
        """Check if command matches this approval rule."""
        if self.rule_type == ApprovalDecision.APPROVE_ALWAYS_EXACT:
            return command == self.command_pattern

        elif self.rule_type == ApprovalDecision.APPROVE_ALWAYS_COMMAND:
            # Extract command name (first word)
            cmd_name = command.split()[0] if command else ""
            pattern_cmd = self.command_pattern.split()[0] if self.command_pattern else ""
            return cmd_name == pattern_cmd

        elif self.rule_type == ApprovalDecision.APPROVE_ALWAYS_PATH_PATTERN:
            # Match if command contains the path pattern
            # Pattern format: "mkdir /home/hubcaps/*" matches all mkdir in that path
            try:
                # Extract path from pattern (e.g., "mkdir -p /home/hubcaps/*" -> "/home/hubcaps/")
                pattern_parts = self.command_pattern.split()
                base_path = None
                for part in pattern_parts:
                    if part.startswith('/') and '*' in part:
                        base_path = part.replace('*', '')
                        break

                if base_path and base_path in command:
                    # Also check command type matches
                    cmd_name = command.split()[0] if command else ""
                    pattern_cmd = pattern_parts[0] if pattern_parts else ""
                    return cmd_name == pattern_cmd
            except Exception as e:
                logger.error(f"Error matching path pattern: {e}")
                return False

        return False


class ApprovalManager:
    """
    Manages approval requests and workflow-scoped approval rules.

    Features:
    - Queue pending approval requests
    - Track per-workflow "always approve" rules
    - Expire old approval requests
    - Thread-safe operations
    - Auto-approval based on workflow rules
    """

    def __init__(self, approval_timeout: float = 300.0):
        """
        Initialize approval manager.

        Args:
            approval_timeout: Time in seconds before approval requests expire (default: 5 minutes)
        """
        self.approval_timeout = approval_timeout

        # Pending and historical approval requests
        self._approvals: Dict[str, ApprovalRequest] = {}

        # Per-workflow approval rules (session-scoped)
        self._workflow_rules: Dict[str, List[WorkflowApprovalRule]] = {}

        # Counters for ID generation
        self._approval_counter = 0
        self._rule_counter = 0

        # Thread safety
        self._lock = threading.Lock()

        logger.info("ApprovalManager initialized")
        logger.info(f"  Approval timeout: {approval_timeout}s")

    def request_approval(self, command: str, agent_id: str, context: str,
                        trust_level: TrustLevel, workflow_id: Optional[str] = None) -> str:
        """
        Request user approval for a command.

        Args:
            command: Command to execute
            agent_id: Agent requesting execution
            context: Context/reason for command
            trust_level: Trust level from trust manager
            workflow_id: Associated workflow ID (for rule scoping)

        Returns:
            approval_id for tracking the request
        """
        with self._lock:
            self._approval_counter += 1
            approval_id = f"approval_{self._approval_counter:04d}"

            # Assess risk based on command and trust level
            risk_assessment = self._assess_risk(command, trust_level)

            approval = ApprovalRequest(
                approval_id=approval_id,
                command=command,
                agent_id=agent_id,
                context=context,
                trust_level=trust_level,
                workflow_id=workflow_id,
                risk_assessment=risk_assessment,
                requested_at=time.time()
            )

            self._approvals[approval_id] = approval

            logger.info(f"Approval requested: {approval_id}")
            logger.info(f"  Command: {command}")
            logger.info(f"  Agent: {agent_id}")
            logger.info(f"  Workflow: {workflow_id or 'None'}")
            logger.info(f"  Risk: {risk_assessment}")

            return approval_id

    def check_auto_approve(self, command: str, workflow_id: Optional[str]) -> Optional[str]:
        """
        Check if command can be auto-approved based on workflow rules.

        Args:
            command: Command to check
            workflow_id: Associated workflow ID

        Returns:
            Rule ID if auto-approved, None otherwise
        """
        if not workflow_id:
            return None

        with self._lock:
            rules = self._workflow_rules.get(workflow_id, [])
            for rule in rules:
                if rule.matches(command):
                    logger.info(f"Command auto-approved by workflow rule: {rule.rule_id}")
                    logger.info(f"  Rule type: {rule.rule_type.value}")
                    logger.info(f"  Pattern: {rule.command_pattern}")
                    return rule.rule_id

        return None

    def decide_approval(self, approval_id: str, decision: ApprovalDecision,
                       decided_by: str = "user") -> bool:
        """
        Make a decision on an approval request.

        Args:
            approval_id: Approval ID to decide
            decision: Decision type
            decided_by: Who made the decision (default: "user")

        Returns:
            True if decision successful, False if approval not found or expired
        """
        with self._lock:
            approval = self._approvals.get(approval_id)

            if not approval:
                logger.error(f"Approval not found: {approval_id}")
                return False

            if approval.status != ApprovalStatus.PENDING:
                logger.warning(f"Approval already decided: {approval_id} (status: {approval.status.value})")
                return False

            # Check if expired
            if time.time() > approval.expires_at:
                approval.status = ApprovalStatus.EXPIRED
                logger.warning(f"Approval expired: {approval_id}")
                return False

            # Update approval status
            approval.decided_at = time.time()
            approval.decided_by = decided_by
            approval.decision = decision

            if decision == ApprovalDecision.DENY:
                approval.status = ApprovalStatus.DENIED
                logger.info(f"Approval denied: {approval_id}")
                return True

            # Approve decisions
            approval.status = ApprovalStatus.APPROVED
            logger.info(f"Approval granted: {approval_id} (decision: {decision.value})")

            # Create workflow rule if "always approve" decision
            if decision in [ApprovalDecision.APPROVE_ALWAYS_EXACT,
                           ApprovalDecision.APPROVE_ALWAYS_COMMAND,
                           ApprovalDecision.APPROVE_ALWAYS_PATH_PATTERN]:
                self._create_workflow_rule(approval, decision)

            return True

    def _create_workflow_rule(self, approval: ApprovalRequest, rule_type: ApprovalDecision):
        """Create a workflow approval rule based on decision."""
        if not approval.workflow_id:
            logger.warning(f"Cannot create workflow rule without workflow_id: {approval.approval_id}")
            return

        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"

        # Generate pattern based on rule type
        if rule_type == ApprovalDecision.APPROVE_ALWAYS_EXACT:
            pattern = approval.command
        elif rule_type == ApprovalDecision.APPROVE_ALWAYS_COMMAND:
            # Extract command name
            cmd_name = approval.command.split()[0] if approval.command else ""
            pattern = cmd_name
        elif rule_type == ApprovalDecision.APPROVE_ALWAYS_PATH_PATTERN:
            # Extract base path and create wildcard pattern
            cmd_parts = approval.command.split()
            cmd_name = cmd_parts[0] if cmd_parts else ""
            # Find path in command
            base_path = None
            for part in cmd_parts:
                if part.startswith('/'):
                    # Extract base path (up to last directory)
                    base_path = '/'.join(part.split('/')[:-1]) + '/*'
                    break
            pattern = f"{cmd_name} {base_path}" if base_path else approval.command
        else:
            pattern = approval.command

        rule = WorkflowApprovalRule(
            rule_id=rule_id,
            workflow_id=approval.workflow_id,
            rule_type=rule_type,
            command_pattern=pattern,
            created_at=time.time(),
            created_by_approval=approval.approval_id
        )

        if approval.workflow_id not in self._workflow_rules:
            self._workflow_rules[approval.workflow_id] = []

        self._workflow_rules[approval.workflow_id].append(rule)

        logger.info(f"Created workflow rule: {rule_id}")
        logger.info(f"  Workflow: {approval.workflow_id}")
        logger.info(f"  Type: {rule_type.value}")
        logger.info(f"  Pattern: {pattern}")

    def get_pending_approvals(self, workflow_id: Optional[str] = None) -> List[ApprovalRequest]:
        """
        Get pending approval requests, optionally filtered by workflow.

        Args:
            workflow_id: Optional workflow ID to filter

        Returns:
            List of pending approval requests
        """
        with self._lock:
            pending = [
                approval for approval in self._approvals.values()
                if approval.status == ApprovalStatus.PENDING and
                (workflow_id is None or approval.workflow_id == workflow_id)
            ]
            return pending

    def get_approval_status(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Get status of an approval request."""
        with self._lock:
            return self._approvals.get(approval_id)

    def cleanup_expired(self):
        """Mark expired approval requests and remove old workflows."""
        with self._lock:
            current_time = time.time()

            # Expire old approval requests
            for approval in self._approvals.values():
                if approval.status == ApprovalStatus.PENDING and current_time > approval.expires_at:
                    approval.status = ApprovalStatus.EXPIRED
                    logger.info(f"Approval expired: {approval.approval_id}")

    def clear_workflow_rules(self, workflow_id: str):
        """
        Clear all approval rules for a workflow (called when workflow completes).

        Args:
            workflow_id: Workflow ID to clear rules for
        """
        with self._lock:
            if workflow_id in self._workflow_rules:
                rules_count = len(self._workflow_rules[workflow_id])
                del self._workflow_rules[workflow_id]
                logger.info(f"Cleared {rules_count} approval rules for workflow: {workflow_id}")

    def get_workflow_rules(self, workflow_id: str) -> List[WorkflowApprovalRule]:
        """Get all approval rules for a workflow."""
        with self._lock:
            return self._workflow_rules.get(workflow_id, []).copy()

    def get_approval_history(self, limit: int = 50) -> List[ApprovalRequest]:
        """
        Get approval history (completed approvals only).

        Args:
            limit: Maximum number of history entries to return (default: 50)

        Returns:
            List of completed approval requests, sorted by decision time (most recent first)
        """
        with self._lock:
            # Filter for completed approvals (not pending)
            completed = [
                approval for approval in self._approvals.values()
                if approval.status in [ApprovalStatus.APPROVED, ApprovalStatus.DENIED,
                                      ApprovalStatus.EXPIRED, ApprovalStatus.AUTO_APPROVED]
            ]

            # Sort by decided_at timestamp (most recent first)
            # For expired approvals, use expires_at instead
            completed.sort(
                key=lambda a: a.decided_at if a.decided_at else a.expires_at,
                reverse=True
            )

            return completed[:limit]

    def _assess_risk(self, command: str, trust_level: TrustLevel) -> str:
        """
        Assess risk level of a command.

        Args:
            command: Command to assess
            trust_level: Trust level from trust manager

        Returns:
            Human-readable risk assessment
        """
        if trust_level == TrustLevel.BLOCKED:
            return "Critical - Command is blocked"

        # Check for dangerous patterns
        dangerous_patterns = [
            (r'rm\s+-rf', "High - Recursive deletion"),
            (r'rm\s+.*/', "High - Directory deletion"),
            (r'sudo', "High - Requires elevated privileges"),
            (r'chmod.*777', "High - Overly permissive permissions"),
            (r'/etc/', "High - System configuration directory"),
            (r'/var/', "Moderate - System variable directory"),
            (r'>', "Moderate - File write/overwrite"),
            (r'>>', "Low - File append"),
            (r'mkdir', "Low - Directory creation"),
            (r'touch', "Low - File creation"),
        ]

        for pattern, risk in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return risk

        return "Low - Standard operation"

    def get_statistics(self) -> Dict[str, Any]:
        """Get approval manager statistics."""
        with self._lock:
            total = len(self._approvals)
            pending = sum(1 for a in self._approvals.values() if a.status == ApprovalStatus.PENDING)
            approved = sum(1 for a in self._approvals.values() if a.status == ApprovalStatus.APPROVED)
            denied = sum(1 for a in self._approvals.values() if a.status == ApprovalStatus.DENIED)
            expired = sum(1 for a in self._approvals.values() if a.status == ApprovalStatus.EXPIRED)
            auto_approved = sum(1 for a in self._approvals.values() if a.status == ApprovalStatus.AUTO_APPROVED)

            total_rules = sum(len(rules) for rules in self._workflow_rules.values())
            active_workflows = len(self._workflow_rules)

            return {
                "total_approvals": total,
                "pending": pending,
                "approved": approved,
                "denied": denied,
                "expired": expired,
                "auto_approved": auto_approved,
                "total_rules": total_rules,
                "active_workflows": active_workflows
            }
