"""
Failure Recovery System (Phase 3.2)

Implements adaptive failure recovery strategies:
- Agent failure: Spawn replacement with adjusted parameters
- Command failure: Retry with alternative approaches
- Pattern tracking: Learn from failure patterns
- Circuit breaker: Escalate to user on cascading failures
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import time

logger = logging.getLogger('felix_workflows')


class WorkflowAbortedException(Exception):
    """Raised when user aborts workflow after cascading failures."""
    pass


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker state."""
    failure_count: int = 0
    state: str = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    last_failure_time: float = 0.0
    failure_history: List[Dict[str, Any]] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker that escalates to user when failures cascade.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, blocking until user decides
    - HALF_OPEN: Testing if system has recovered after user override

    When the circuit opens, it pauses and asks the user what to do via
    the escalation callback.
    """

    def __init__(self,
                 failure_threshold: int = 3,
                 reset_timeout: float = 60.0,
                 escalation_callback: Optional[Callable[[Dict], str]] = None):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before auto-transitioning to HALF_OPEN
            escalation_callback: Callback to notify user when circuit opens.
                Receives dict with failure info, returns user decision string.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.escalation_callback = escalation_callback
        self._state = CircuitBreakerState()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        return self._state.state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._state.failure_count

    def record_failure(self, component_id: str, error_msg: str) -> Optional[str]:
        """
        Record a failure and potentially open the circuit.

        Args:
            component_id: ID of the component that failed
            error_msg: Error message describing the failure

        Returns:
            User's decision if circuit opened, None otherwise
        """
        self._state.failure_count += 1
        self._state.last_failure_time = time.time()
        self._state.failure_history.append({
            'component': component_id,
            'error': error_msg[:200],  # Truncate for readability
            'timestamp': self._state.last_failure_time
        })

        logger.warning(f"⚡ Circuit breaker: failure {self._state.failure_count}/{self.failure_threshold}")

        if self._state.failure_count >= self.failure_threshold:
            self._state.state = 'OPEN'
            logger.error(f"🔴 Circuit breaker OPEN - {self._state.failure_count} failures exceeded threshold")
            return self._escalate_to_user()

        return None

    def _escalate_to_user(self) -> str:
        """
        Pause and ask user what to do.

        Returns:
            User's decision string
        """
        if self.escalation_callback:
            try:
                decision = self.escalation_callback({
                    'type': 'circuit_breaker_open',
                    'failure_count': self._state.failure_count,
                    'recent_failures': self._state.failure_history[-5:],
                    'options': [
                        {'id': 'retry_with_adjustments', 'label': 'Retry with adjusted parameters'},
                        {'id': 'spawn_backup_agents', 'label': 'Spawn fresh backup agents'},
                        {'id': 'abort_workflow', 'label': 'Abort workflow'},
                        {'id': 'continue_degraded', 'label': 'Continue with degraded output'}
                    ]
                })
                logger.info(f"⚡ User decision on circuit breaker: {decision}")
                return decision
            except Exception as e:
                logger.error(f"Circuit breaker escalation failed: {e}")
                return 'abort_workflow'

        # Default if no callback
        logger.warning("⚡ No escalation callback - defaulting to abort_workflow")
        return 'abort_workflow'

    def should_allow_request(self) -> bool:
        """
        Check if requests should be allowed through.

        Returns:
            True if requests are allowed, False if circuit is open
        """
        if self._state.state == 'CLOSED':
            return True

        if self._state.state == 'OPEN':
            # Check if enough time has passed for auto-reset to HALF_OPEN
            if time.time() - self._state.last_failure_time > self.reset_timeout:
                logger.info("⚡ Circuit breaker auto-transitioning to HALF_OPEN")
                self._state.state = 'HALF_OPEN'
                return True
            return False

        # HALF_OPEN: allow one attempt
        return True

    def user_override(self, decision: str) -> None:
        """
        Process user's escalation decision.

        Args:
            decision: One of 'retry_with_adjustments', 'spawn_backup_agents',
                     'continue_degraded', or 'abort_workflow'

        Raises:
            WorkflowAbortedException: If user chose to abort
        """
        logger.info(f"⚡ Processing circuit breaker override: {decision}")

        if decision == 'retry_with_adjustments':
            self._state.state = 'HALF_OPEN'
            self._state.failure_count = 0
            logger.info("⚡ Circuit breaker: HALF_OPEN, will retry with adjustments")

        elif decision == 'spawn_backup_agents':
            self._state.state = 'HALF_OPEN'
            self._state.failure_count = 0
            logger.info("⚡ Circuit breaker: HALF_OPEN, spawning backup agents")

        elif decision == 'continue_degraded':
            self._state.state = 'CLOSED'
            logger.warning("⚡ Circuit breaker: CLOSED (degraded mode)")

        elif decision == 'abort_workflow':
            logger.error("⚡ Circuit breaker: User aborted workflow")
            raise WorkflowAbortedException(
                f"User aborted after {self._state.failure_count} cascading failures. "
                f"Recent failures: {[f['component'] for f in self._state.failure_history[-3:]]}"
            )

    def record_success(self) -> None:
        """Record a successful operation, potentially closing the circuit."""
        if self._state.state == 'HALF_OPEN':
            logger.info("⚡ Circuit breaker: Success in HALF_OPEN, closing circuit")
            self._state.state = 'CLOSED'
            self._state.failure_count = 0
            self._state.failure_history.clear()

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitBreakerState()
        logger.info("⚡ Circuit breaker reset")


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
    Integrates with CircuitBreaker to escalate cascading failures to user.
    """

    def __init__(self,
                 circuit_breaker_threshold: int = 3,
                 escalation_callback: Optional[Callable[[Dict], str]] = None):
        """
        Initialize failure recovery manager.

        Args:
            circuit_breaker_threshold: Number of failures before circuit opens
            escalation_callback: Callback for user escalation on circuit open
        """
        self.failure_history: List[FailureRecord] = []
        self.recovery_stats = {
            'total_failures': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'agents_spawned_for_recovery': 0
        }

        # Failure patterns (component_id -> count)
        self.failure_patterns: Dict[str, int] = {}

        # Circuit breaker for cascading failure protection
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            escalation_callback=escalation_callback
        )

        # Track spawned recovery agents
        self.recovery_agents: List[str] = []

    def record_failure(self,
                      failure_type: FailureType,
                      component_id: str,
                      error_message: str,
                      context: Optional[Dict[str, Any]] = None) -> FailureRecord:
        """
        Record a failure for tracking and recovery.

        Also notifies the circuit breaker, which may trigger user escalation
        if too many failures have occurred.

        Args:
            failure_type: Type of failure
            component_id: ID of failed component (agent/command)
            error_message: Error description
            context: Additional context about the failure

        Returns:
            FailureRecord instance

        Raises:
            WorkflowAbortedException: If circuit breaker escalates and user aborts
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

        # Notify circuit breaker - may escalate to user if threshold exceeded
        user_decision = self.circuit_breaker.record_failure(component_id, error_message)
        if user_decision:
            # Circuit opened and user made a decision
            self.circuit_breaker.user_override(user_decision)
            # Store decision in record context for downstream handling
            record.context['circuit_breaker_decision'] = user_decision

        return record

    def attempt_recovery(self,
                        record: FailureRecord,
                        agent_factory: Optional[Any] = None,
                        central_post: Optional[Any] = None,
                        helix: Optional[Any] = None,
                        llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Attempt to recover from a failure using appropriate strategy.

        Args:
            record: FailureRecord to recover from
            agent_factory: Optional AgentFactory for spawning recovery agents
            central_post: Optional CentralPost for agent registration
            helix: Optional Helix geometry for agent creation
            llm_client: Optional LLM client for agent creation

        Returns:
            Dictionary with recovery result:
                - success: bool
                - strategy: str
                - adjusted_parameters: Dict[str, Any]
                - message: str
                - spawned_agent: Optional[str] - ID of spawned agent if any
        """
        # Check circuit breaker before attempting recovery
        if not self.circuit_breaker.should_allow_request():
            logger.warning("🔴 Circuit breaker OPEN - recovery blocked")
            return {
                'success': False,
                'strategy': 'circuit_breaker_blocked',
                'adjusted_parameters': {},
                'message': "Circuit breaker is open - too many failures"
            }

        self.recovery_stats['recovery_attempts'] += 1
        record.recovery_attempted = True

        if record.failure_type == FailureType.AGENT_ERROR:
            return self._recover_from_agent_error(record)
        elif record.failure_type == FailureType.COMMAND_ERROR:
            return self._recover_from_command_error(record)
        elif record.failure_type == FailureType.TIMEOUT:
            return self._recover_from_timeout(record)
        elif record.failure_type == FailureType.LOW_CONFIDENCE:
            return self._recover_from_low_confidence(
                record,
                agent_factory=agent_factory,
                central_post=central_post,
                helix=helix,
                llm_client=llm_client
            )
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

    def _recover_from_low_confidence(self,
                                      record: FailureRecord,
                                      agent_factory: Optional[Any] = None,
                                      central_post: Optional[Any] = None,
                                      helix: Optional[Any] = None,
                                      llm_client: Optional[Any] = None) -> Dict[str, Any]:
        """
        Recover from low confidence result by actually spawning a critic agent.

        Strategy: Spawn a critic agent to validate the low-confidence result.
        This is a REAL recovery that spawns an actual agent, not just a flag.

        Args:
            record: FailureRecord with low confidence details
            agent_factory: AgentFactory for creating the critic agent
            central_post: CentralPost for registering the new agent
            helix: Helix geometry for agent positioning
            llm_client: LLM client for the new agent

        Returns:
            Dictionary with recovery result including spawned_agent ID if successful
        """
        logger.info(f"🔄 Attempting recovery from low confidence: {record.component_id}")

        # Check if we have the dependencies needed to spawn an agent
        if not agent_factory:
            logger.warning("  ⚠️ No agent_factory provided - cannot spawn recovery agent")
            record.recovery_successful = False
            self.recovery_stats['failed_recoveries'] += 1
            return {
                'success': False,
                'strategy': 'spawn_critic_failed',
                'adjusted_parameters': {},
                'message': "Cannot spawn critic: agent_factory not provided"
            }

        # Extract context about the low-confidence result
        low_conf_result = record.context.get('result')
        agent_type = record.context.get('agent_type', 'unknown')
        confidence = low_conf_result.confidence if low_conf_result else 0.0

        logger.info(f"  Original agent type: {agent_type}, confidence: {confidence:.2f}")
        logger.info(f"  Strategy: Spawn critic agent for validation")

        try:
            # Spawn a critic agent to validate the low-confidence result
            critic = agent_factory.create_critic_agent(
                helix=helix,
                llm_client=llm_client,
                spawn_time=time.time(),
                focus='validation'
            )

            # Register with CentralPost if available
            if central_post:
                central_post.register_agent(critic.agent_id, {
                    'type': 'critic',
                    'spawned_for_recovery': True,
                    'validating_agent': record.component_id,
                    'original_confidence': confidence
                })
                logger.info(f"  ✓ Critic {critic.agent_id} registered with CentralPost")

            # Track the recovery agent
            self.recovery_agents.append(critic.agent_id)
            self.recovery_stats['agents_spawned_for_recovery'] += 1

            record.recovery_strategy = "spawned_critic"
            record.recovery_successful = True
            self.recovery_stats['successful_recoveries'] += 1

            # Record success with circuit breaker
            self.circuit_breaker.record_success()

            logger.info(f"  ✓ Spawned critic agent {critic.agent_id} for validation")

            return {
                'success': True,
                'strategy': 'spawned_critic',
                'spawned_agent': critic.agent_id,
                'spawned_agent_instance': critic,
                'adjusted_parameters': {
                    'spawn_critic': True,
                    'critic_id': critic.agent_id,
                    'validating': record.component_id
                },
                'message': f"Spawned critic agent {critic.agent_id} to validate low-confidence result"
            }

        except Exception as e:
            logger.error(f"  ✗ Failed to spawn critic agent: {e}")
            record.recovery_successful = False
            self.recovery_stats['failed_recoveries'] += 1

            return {
                'success': False,
                'strategy': 'spawn_critic_failed',
                'adjusted_parameters': {},
                'message': f"Failed to spawn critic agent: {str(e)}"
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
