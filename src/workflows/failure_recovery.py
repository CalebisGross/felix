"""
Failure Recovery System (Phase 3.2)

Implements adaptive failure recovery strategies:
- Agent failure: Spawn replacement with adjusted parameters
- Command failure: Retry with alternative approaches
- Pattern tracking: Learn from failure patterns
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger('felix_workflows')


class FailureType(Enum):
    """Types of failures that can occur."""
    AGENT_ERROR = "agent_error"           # Agent LLM call failed
    COMMAND_ERROR = "command_error"       # System command failed
    TIMEOUT = "timeout"                   # Operation timed out
    LOW_CONFIDENCE = "low_confidence"     # Agent produced low-confidence result
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough context/knowledge


@dataclass
class FailureRecord:
    """Record of a failure and its context."""
    failure_type: FailureType
    component_id: str  # Agent ID or command
    error_message: str
    timestamp: float
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None


class FailureRecoveryManager:
    """
    Manages failure detection and recovery strategies.

    Tracks failures, applies recovery strategies, and learns from patterns.
    """

    def __init__(self):
        """Initialize failure recovery manager."""
        self.failure_history: List[FailureRecord] = []
        self.recovery_stats = {
            'total_failures': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0
        }

        # Failure patterns (component_id -> count)
        self.failure_patterns: Dict[str, int] = {}

    def record_failure(self,
                      failure_type: FailureType,
                      component_id: str,
                      error_message: str,
                      context: Optional[Dict[str, Any]] = None) -> FailureRecord:
        """
        Record a failure for tracking and recovery.

        Args:
            failure_type: Type of failure
            component_id: ID of failed component (agent/command)
            error_message: Error description
            context: Additional context about the failure

        Returns:
            FailureRecord instance
        """
        record = FailureRecord(
            failure_type=failure_type,
            component_id=component_id,
            error_message=error_message,
            timestamp=time.time(),
            context=context or {}
        )

        self.failure_history.append(record)
        self.recovery_stats['total_failures'] += 1

        # Track failure patterns
        self.failure_patterns[component_id] = self.failure_patterns.get(component_id, 0) + 1

        logger.warning(f"⚠️ Failure recorded: {failure_type.value} in {component_id}")
        logger.warning(f"   Error: {error_message[:100]}")

        return record

    def attempt_recovery(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Attempt to recover from a failure using appropriate strategy.

        Args:
            record: FailureRecord to recover from

        Returns:
            Dictionary with recovery result:
                - success: bool
                - strategy: str
                - adjusted_parameters: Dict[str, Any]
                - message: str
        """
        self.recovery_stats['recovery_attempts'] += 1
        record.recovery_attempted = True

        if record.failure_type == FailureType.AGENT_ERROR:
            return self._recover_from_agent_error(record)
        elif record.failure_type == FailureType.COMMAND_ERROR:
            return self._recover_from_command_error(record)
        elif record.failure_type == FailureType.TIMEOUT:
            return self._recover_from_timeout(record)
        elif record.failure_type == FailureType.LOW_CONFIDENCE:
            return self._recover_from_low_confidence(record)
        elif record.failure_type == FailureType.INSUFFICIENT_DATA:
            return self._recover_from_insufficient_data(record)
        else:
            return {
                'success': False,
                'strategy': 'none',
                'adjusted_parameters': {},
                'message': f"No recovery strategy for {record.failure_type.value}"
            }

    def _recover_from_agent_error(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Recover from agent LLM call failure.

        Strategy: Retry with lower temperature and higher max_tokens
        """
        logger.info(f"🔄 Attempting recovery from agent error: {record.component_id}")

        # Extract agent parameters from context
        original_params = record.context.get('agent_params', {})
        current_temp = original_params.get('temperature', 0.7)
        current_tokens = original_params.get('max_tokens', 1000)

        # Adjust parameters: lower temperature, increase tokens
        adjusted_params = {
            'temperature': max(current_temp * 0.7, 0.2),  # Reduce by 30%, min 0.2
            'max_tokens': min(current_tokens * 1.5, 2000),  # Increase by 50%, max 2000
            'retry': True
        }

        logger.info(f"  Adjusted parameters:")
        logger.info(f"    Temperature: {current_temp:.2f} → {adjusted_params['temperature']:.2f}")
        logger.info(f"    Max tokens: {current_tokens} → {adjusted_params['max_tokens']:.0f}")

        record.recovery_strategy = "retry_with_adjusted_params"
        record.recovery_successful = True
        self.recovery_stats['successful_recoveries'] += 1

        return {
            'success': True,
            'strategy': 'retry_with_adjusted_params',
            'adjusted_parameters': adjusted_params,
            'message': "Retrying agent with lower temperature and more tokens"
        }

    def _recover_from_command_error(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Recover from system command failure.

        Strategy: Try alternative commands or simplified versions
        """
        logger.info(f"🔄 Attempting recovery from command error: {record.component_id}")

        command = record.context.get('command', '')
        alternatives = self._get_command_alternatives(command)

        if alternatives:
            logger.info(f"  Alternative commands available:")
            for alt in alternatives:
                logger.info(f"    - {alt}")

            record.recovery_strategy = "try_alternatives"
            record.recovery_successful = True
            self.recovery_stats['successful_recoveries'] += 1

            return {
                'success': True,
                'strategy': 'try_alternatives',
                'adjusted_parameters': {'alternatives': alternatives},
                'message': f"Trying {len(alternatives)} alternative command(s)"
            }
        else:
            logger.warning(f"  No alternatives found for command: {command}")
            record.recovery_successful = False
            self.recovery_stats['failed_recoveries'] += 1

            return {
                'success': False,
                'strategy': 'no_alternatives',
                'adjusted_parameters': {},
                'message': "No alternative commands available"
            }

    def _recover_from_timeout(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Recover from timeout.

        Strategy: Retry with extended timeout and simpler task
        """
        logger.info(f"🔄 Attempting recovery from timeout: {record.component_id}")

        current_timeout = record.context.get('timeout', 60.0)
        extended_timeout = current_timeout * 2.0

        logger.info(f"  Extending timeout: {current_timeout}s → {extended_timeout}s")

        record.recovery_strategy = "extend_timeout"
        record.recovery_successful = True
        self.recovery_stats['successful_recoveries'] += 1

        return {
            'success': True,
            'strategy': 'extend_timeout',
            'adjusted_parameters': {'timeout': extended_timeout},
            'message': f"Retrying with extended timeout ({extended_timeout}s)"
        }

    def _recover_from_low_confidence(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Recover from low confidence result.

        Strategy: Spawn additional agents for validation
        """
        logger.info(f"🔄 Attempting recovery from low confidence: {record.component_id}")

        logger.info(f"  Strategy: Spawn critic agent for validation")

        record.recovery_strategy = "spawn_critic"
        record.recovery_successful = True
        self.recovery_stats['successful_recoveries'] += 1

        return {
            'success': True,
            'strategy': 'spawn_critic',
            'adjusted_parameters': {'spawn_critic': True},
            'message': "Spawning critic agent to validate low-confidence result"
        }

    def _recover_from_insufficient_data(self, record: FailureRecord) -> Dict[str, Any]:
        """
        Recover from insufficient data.

        Strategy: Trigger web search or additional research
        """
        logger.info(f"🔄 Attempting recovery from insufficient data: {record.component_id}")

        logger.info(f"  Strategy: Trigger additional research or web search")

        record.recovery_strategy = "additional_research"
        record.recovery_successful = True
        self.recovery_stats['successful_recoveries'] += 1

        return {
            'success': True,
            'strategy': 'additional_research',
            'adjusted_parameters': {'enable_web_search': True},
            'message': "Triggering additional research with web search"
        }

    def _get_command_alternatives(self, command: str) -> List[str]:
        """
        Get alternative commands that might work if the original fails.

        Args:
            command: Original command that failed

        Returns:
            List of alternative commands to try
        """
        alternatives = []

        # find command alternatives
        if command.startswith('find '):
            if '-name' in command and '-type f' not in command:
                alternatives.append(command.replace('-name', '-type f -name'))
            if 'find .' in command and 'find ./' not in command:
                alternatives.append(command.replace('find .', 'find ./'))
            if '-name' in command:
                # Try locate as alternative
                import re
                match = re.search(r'-name\s+["\']?([^"\']+)["\']?', command)
                if match:
                    filename = match.group(1)
                    alternatives.append(f"locate {filename}")

        # cat command alternatives
        elif command.startswith('cat '):
            filepath = command[4:].strip()
            alternatives.append(f"head -n 1000 {filepath}")  # Try limited read
            alternatives.append(f"test -f {filepath} && cat {filepath}")  # Check existence first

        # head/tail command alternatives
        elif command.startswith('head ') or command.startswith('tail '):
            alternatives.append(f"cat {command.split()[-1]}")  # Try full cat instead

        # ls command alternatives
        elif command.startswith('ls '):
            path = command.split()[-1]
            alternatives.append(f"find {path} -maxdepth 1")  # Use find instead

        return alternatives

    def get_failure_summary(self) -> Dict[str, Any]:
        """
        Get summary of failures and recovery statistics.

        Returns:
            Dictionary with failure statistics
        """
        return {
            'total_failures': self.recovery_stats['total_failures'],
            'recovery_attempts': self.recovery_stats['recovery_attempts'],
            'successful_recoveries': self.recovery_stats['successful_recoveries'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'success_rate': (
                self.recovery_stats['successful_recoveries'] / self.recovery_stats['recovery_attempts']
                if self.recovery_stats['recovery_attempts'] > 0 else 0.0
            ),
            'top_failing_components': sorted(
                self.failure_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def should_abandon_recovery(self, component_id: str) -> bool:
        """
        Determine if recovery should be abandoned for a component.

        Args:
            component_id: Component that keeps failing

        Returns:
            True if component has failed too many times
        """
        failure_count = self.failure_patterns.get(component_id, 0)
        return failure_count >= 3  # Abandon after 3 failures
