"""
Trust Manager for Command Classification and Approval.

Implements a three-tier trust system:
- SAFE: Auto-execute (ls, pwd, cat, etc.)
- REVIEW: Require user approval (pip install, git commit, etc.)
- BLOCKED: Never execute (sudo rm -rf, dd, mkfs, etc.)

Trust rules are configurable via YAML.
"""

import re
import time
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for command execution."""
    SAFE = "safe"          # Auto-execute
    REVIEW = "review"      # Require approval
    BLOCKED = "blocked"    # Never execute


@dataclass
class ApprovalRequest:
    """Request for command approval."""
    approval_id: str
    command: str
    agent_id: str
    context: str
    trust_level: TrustLevel
    risk_assessment: Dict[str, Any]
    requested_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    denial_reason: Optional[str] = None


class TrustManager:
    """
    Manages command trust classification and approval workflow.

    Features:
    - Pattern-based command classification
    - Configurable trust rules (YAML)
    - Approval queue for REVIEW commands
    - Command history integration
    - Risk assessment
    """

    def __init__(self, rules_path: Optional[Path] = None):
        """
        Initialize trust manager.

        Args:
            rules_path: Path to trust rules YAML file (None = use defaults)
        """
        self.rules_path = rules_path or Path("config/trust_rules.yaml")
        self.approval_queue: Dict[str, ApprovalRequest] = {}
        self._approval_id_counter = 0

        # Load trust rules
        self.trust_rules = self._load_trust_rules()

        logger.info(f"TrustManager initialized")
        logger.info(f"  Rules file: {self.rules_path}")
        logger.info(f"  SAFE patterns: {len(self.trust_rules['safe'])}")
        logger.info(f"  REVIEW patterns: {len(self.trust_rules['review'])}")
        logger.info(f"  BLOCKED patterns: {len(self.trust_rules['blocked'])}")

    def _load_trust_rules(self) -> Dict[str, List[str]]:
        """
        Load trust rules from YAML file or use defaults.

        Returns:
            Dictionary with 'safe', 'review', 'blocked' pattern lists
        """
        # Default rules
        default_rules = {
            'safe': [
                # Information gathering (safe, read-only)
                r'^ls\s',
                r'^pwd$',
                r'^echo\s',
                r'^cat\s',
                r'^head\s',
                r'^tail\s',
                r'^grep\s',
                r'^find\s',
                r'^which\s',
                r'^whereis\s',
                r'^whoami$',
                r'^date$',
                r'^uptime$',
                r'^df\s',
                r'^du\s',
                r'^free\s',
                r'^ps\s',
                r'^top\s',
                r'^htop\s',

                # Git read-only
                r'^git\s+status',
                r'^git\s+log',
                r'^git\s+diff',
                r'^git\s+show',
                r'^git\s+branch(\s|$)',
                r'^git\s+remote\s+',

                # Python info
                r'^python\s+--version',
                r'^python3\s+--version',
                r'^pip\s+list',
                r'^pip\s+show\s',
                r'^pip\s+freeze',

                # Virtual environment (safe operations)
                r'^source\s+\.?v?env/bin/activate',
                r'^source\s+[^/]+/bin/activate',
                r'^\./?\.?v?env/Scripts/activate',
            ],

            'review': [
                # Package management
                r'^pip\s+install\s',
                r'^pip\s+uninstall\s',
                r'^npm\s+install',
                r'^npm\s+uninstall',
                r'^apt\s+install',
                r'^apt\s+remove',
                r'^brew\s+install',

                # Git write operations
                r'^git\s+add\s',
                r'^git\s+commit',
                r'^git\s+push',
                r'^git\s+pull',
                r'^git\s+merge',
                r'^git\s+rebase',
                r'^git\s+checkout\s',
                r'^git\s+clone\s',

                # File operations (write)
                r'^mv\s',
                r'^cp\s',
                r'^mkdir\s',
                r'^touch\s',
                r'^chmod\s',

                # Script execution
                r'^python\s+[^-]',  # Python script (not --version)
                r'^python3\s+[^-]',
                r'^bash\s',
                r'^sh\s',
                r'^node\s',

                # Build/test operations
                r'^make\s',
                r'^npm\s+run\s',
                r'^pytest\s',
                r'^cargo\s+build',
                r'^cargo\s+test',
            ],

            'blocked': [
                # Destructive operations
                r'sudo\s+rm\s+-rf\s+/',
                r'rm\s+-rf\s+/',
                r'^dd\s',
                r'^mkfs\s',
                r'^fdisk\s',
                r'^shutdown',
                r'^reboot',
                r'^halt',
                r'^init\s+[06]',

                # Fork bomb
                r':\(\)\s*\{\s*:\|:\&\s*\}',

                # System-wide changes requiring sudo
                r'^sudo\s+apt\s+',
                r'^sudo\s+yum\s+',
                r'^sudo\s+pacman\s+',

                # Dangerous redirects
                r'>\s*/dev/sd[a-z]',
                r'>\s*/dev/null',

                # Credential theft attempts
                r'cat\s+.*\.ssh/',
                r'cat\s+.*\.aws/',
                r'cat\s+.*password',
                r'cat\s+.*secret',
            ]
        }

        # Try to load from file
        if self.rules_path.exists():
            try:
                with open(self.rules_path, 'r') as f:
                    file_rules = yaml.safe_load(f)
                    logger.info(f"Loaded trust rules from {self.rules_path}")
                    return file_rules
            except Exception as e:
                logger.warning(f"Could not load trust rules from {self.rules_path}: {e}")
                logger.info("Using default trust rules")

        return default_rules

    def classify_command(self, command: str) -> TrustLevel:
        """
        Classify command by trust level.

        Args:
            command: Command string to classify

        Returns:
            TrustLevel enum (SAFE, REVIEW, or BLOCKED)
        """
        command = command.strip()

        # Check BLOCKED first (highest priority)
        for pattern in self.trust_rules['blocked']:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Command BLOCKED by pattern: {pattern}")
                return TrustLevel.BLOCKED

        # Check SAFE
        for pattern in self.trust_rules['safe']:
            if re.search(pattern, command, re.IGNORECASE):
                logger.info(f"Command classified as SAFE")
                return TrustLevel.SAFE

        # Check REVIEW
        for pattern in self.trust_rules['review']:
            if re.search(pattern, command, re.IGNORECASE):
                logger.info(f"Command classified as REVIEW")
                return TrustLevel.REVIEW

        # Default: REVIEW (conservative)
        logger.info(f"Command not matched, defaulting to REVIEW")
        return TrustLevel.REVIEW

    def request_approval(self,
                        command: str,
                        agent_id: str,
                        context: str = "",
                        timeout: float = 300.0) -> str:
        """
        Request approval for a command.

        Args:
            command: Command requiring approval
            agent_id: ID of requesting agent
            context: Context/reason for command
            timeout: Approval timeout in seconds (default 5 minutes)

        Returns:
            approval_id for tracking the request
        """
        # Generate approval ID
        self._approval_id_counter += 1
        approval_id = f"approval_{self._approval_id_counter:04d}"

        # Assess risk
        risk_assessment = self._assess_risk(command)

        # Create approval request
        request = ApprovalRequest(
            approval_id=approval_id,
            command=command,
            agent_id=agent_id,
            context=context,
            trust_level=self.classify_command(command),
            risk_assessment=risk_assessment,
            expires_at=time.time() + timeout if timeout > 0 else None
        )

        # Add to queue
        self.approval_queue[approval_id] = request

        logger.info(f"Approval request created: {approval_id}")
        logger.info(f"  Command: {command}")
        logger.info(f"  Agent: {agent_id}")
        logger.info(f"  Risk level: {risk_assessment.get('risk_level', 'unknown')}")

        return approval_id

    def _assess_risk(self, command: str) -> Dict[str, Any]:
        """
        Assess risk level of a command.

        Args:
            command: Command to assess

        Returns:
            Dictionary with risk assessment details
        """
        reasons = []
        risk_score = 0

        # Check for sudo
        if 'sudo' in command.lower():
            reasons.append("Uses sudo (elevated privileges)")
            risk_score += 30

        # Check for destructive patterns
        destructive_keywords = ['rm', 'delete', 'drop', 'truncate', 'format']
        for keyword in destructive_keywords:
            if keyword in command.lower():
                reasons.append(f"Contains destructive keyword: {keyword}")
                risk_score += 20

        # Check for network operations
        network_keywords = ['curl', 'wget', 'download', 'fetch', 'clone']
        for keyword in network_keywords:
            if keyword in command.lower():
                reasons.append(f"Network operation: {keyword}")
                risk_score += 10

        # Check for system modifications
        system_keywords = ['install', 'uninstall', 'update', 'upgrade']
        for keyword in system_keywords:
            if keyword in command.lower():
                reasons.append(f"System modification: {keyword}")
                risk_score += 15

        # Determine risk level
        if risk_score >= 50:
            risk_level = "high"
        elif risk_score >= 20:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'reasons': reasons
        }

    def approve_command(self, approval_id: str, approver: str = "user") -> bool:
        """
        Approve a pending command.

        Args:
            approval_id: Approval request ID
            approver: Who approved (default "user")

        Returns:
            True if approved successfully, False if not found or expired
        """
        if approval_id not in self.approval_queue:
            logger.error(f"Approval request not found: {approval_id}")
            return False

        request = self.approval_queue[approval_id]

        # Check if expired
        if request.expires_at and time.time() > request.expires_at:
            logger.warning(f"Approval request expired: {approval_id}")
            request.approved = False
            request.denial_reason = "Request expired"
            return False

        # Approve
        request.approved = True
        request.approved_by = approver
        request.approved_at = time.time()

        logger.info(f"✓ Command approved: {approval_id}")
        logger.info(f"  Approved by: {approver}")

        return True

    def deny_command(self, approval_id: str, reason: str = "User denied") -> bool:
        """
        Deny a pending command.

        Args:
            approval_id: Approval request ID
            reason: Denial reason

        Returns:
            True if denied successfully, False if not found
        """
        if approval_id not in self.approval_queue:
            logger.error(f"Approval request not found: {approval_id}")
            return False

        request = self.approval_queue[approval_id]

        # Deny
        request.approved = False
        request.denial_reason = reason
        request.approved_at = time.time()

        logger.info(f"✗ Command denied: {approval_id}")
        logger.info(f"  Reason: {reason}")

        return True

    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """
        Get list of pending approval requests.

        Returns:
            List of ApprovalRequest objects (pending only)
        """
        pending = [
            request for request in self.approval_queue.values()
            if request.approved is None
        ]

        # Remove expired requests
        now = time.time()
        pending = [
            req for req in pending
            if not req.expires_at or req.expires_at > now
        ]

        return pending

    def get_approval_status(self, approval_id: str) -> Optional[ApprovalRequest]:
        """
        Get status of an approval request.

        Args:
            approval_id: Approval request ID

        Returns:
            ApprovalRequest object, or None if not found
        """
        return self.approval_queue.get(approval_id)

    def cleanup_old_requests(self, max_age: float = 3600.0):
        """
        Remove old approval requests from queue.

        Args:
            max_age: Maximum age in seconds (default 1 hour)
        """
        now = time.time()
        to_remove = []

        for approval_id, request in self.approval_queue.items():
            age = now - request.requested_at

            # Remove if old and resolved (approved/denied)
            if age > max_age and request.approved is not None:
                to_remove.append(approval_id)

        for approval_id in to_remove:
            del self.approval_queue[approval_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old approval requests")
