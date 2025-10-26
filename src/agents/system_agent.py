"""
SystemAgent - Specialized agent for system operations.

This agent type is optimized for system-level tasks including:
- Command execution and troubleshooting
- Package management
- Virtual environment handling
- File system operations
- Environment configuration

SystemAgent has enhanced prompts and behaviors tailored for:
- Precise, deterministic outputs
- Error analysis and resolution
- Step-by-step procedural thinking
- Safety-conscious command construction
"""

import logging
from typing import Optional, Dict, Any

from src.agents.llm_agent import LLMAgent, LLMTask
from src.core.helix_geometry import HelixGeometry
from src.llm.token_budget import TokenBudgetManager

logger = logging.getLogger(__name__)


class SystemAgent(LLMAgent):
    """
    Specialized agent for system operations and command execution.

    Optimized for:
    - Low temperature (0.1-0.4) for precise, deterministic outputs
    - Moderate token budget (1200-1500) for clear, focused responses
    - Position-aware prompts emphasizing safety and verification
    - Command construction and error resolution
    """

    def __init__(self, agent_id: str, spawn_time: float, helix: HelixGeometry,
                 llm_client, max_tokens: Optional[int] = None,
                 token_budget_manager: Optional[TokenBudgetManager] = None,
                 prompt_optimizer: Optional['PromptOptimizer'] = None,
                 prompt_manager: Optional['PromptManager'] = None):
        """
        Initialize SystemAgent with system operation defaults.

        Args:
            agent_id: Unique identifier
            spawn_time: Time when agent becomes active (0.0 to 1.0)
            helix: Helix geometry for path calculation
            llm_client: LLM client for inference
            max_tokens: Maximum tokens per generation (default: 1500)
            token_budget_manager: Optional budget manager
            prompt_optimizer: Optional prompt optimization
            prompt_manager: Optional prompt manager
        """
        # SystemAgent defaults to low temperature for precision
        super().__init__(
            agent_id=agent_id,
            spawn_time=spawn_time,
            helix=helix,
            llm_client=llm_client,
            agent_type="system",
            temperature_range=(0.1, 0.4),  # Low temp for precise outputs
            max_tokens=max_tokens or 1500,  # Moderate budget for clear responses
            token_budget_manager=token_budget_manager,
            prompt_optimizer=prompt_optimizer,
            prompt_manager=prompt_manager
        )

        # SystemAgent-specific state
        self.commands_executed = []
        self.venv_state = None
        self.last_error = None

        logger.info(f"SystemAgent {agent_id} initialized for system operations")

    def create_position_aware_prompt(self, task: LLMTask, current_time: float) -> tuple[str, int]:
        """
        Create system-focused position-aware prompt.

        Enhances base prompt with system operation guidance:
        - Safety considerations for commands
        - Step-by-step verification procedures
        - Error analysis and resolution strategies

        Args:
            task: Task to process
            current_time: Current simulation time

        Returns:
            Tuple of (enhanced system prompt, token budget)
        """
        # Get base prompt from parent class
        base_prompt, token_budget = super().create_position_aware_prompt(task, current_time)

        # Add SystemAgent-specific guidance
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        system_guidance = self._get_system_guidance(depth_ratio)

        enhanced_prompt = base_prompt + "\n\n" + system_guidance

        return enhanced_prompt, token_budget

    def _get_system_guidance(self, depth_ratio: float) -> str:
        """
        Get system-specific guidance based on helix position.

        Args:
            depth_ratio: Agent's depth ratio (0.0 = top, 1.0 = bottom)

        Returns:
            System guidance text
        """
        if depth_ratio < 0.3:
            # Early exploration: Environment assessment
            return """
=== SYSTEM AGENT GUIDANCE ===
Phase: ENVIRONMENT ASSESSMENT

Your role as a SystemAgent:
1. Assess current system state (working directory, venv status, permissions)
2. Identify what tools and packages are available
3. Detect potential issues or blockers
4. Plan sequence of safe commands

Safety priorities:
- Always check before executing potentially destructive commands
- Verify file/directory existence before operations
- Confirm virtual environment state before package operations
- Use read-only commands first to gather information

Output format: Clear, concise assessment with specific recommendations.
"""
        elif depth_ratio < 0.7:
            # Mid-phase: Command construction and execution
            return """
=== SYSTEM AGENT GUIDANCE ===
Phase: COMMAND EXECUTION

Your role as a SystemAgent:
1. Construct precise, safe commands for the identified task
2. Break complex operations into sequential safe steps
3. Include verification commands to confirm success
4. Provide clear context for why each command is needed

Command construction principles:
- Use full paths when ambiguity possible
- Include error checking (test -f, command -v, etc.)
- Activate venv explicitly if Python operations needed
- Chain commands with && for safety (stops on first failure)
- Avoid destructive operations without explicit user request

Output format: Step-by-step command sequence with explanations.
"""
        else:
            # Late phase: Verification and resolution
            return """
=== SYSTEM AGENT GUIDANCE ===
Phase: VERIFICATION & RESOLUTION

Your role as a SystemAgent:
1. Verify that executed commands achieved intended result
2. Analyze any errors or unexpected outputs
3. Propose corrections or alternative approaches
4. Provide clear success/failure assessment

Error resolution approach:
- Examine error messages for root cause
- Identify missing dependencies or permissions
- Suggest specific corrective commands
- Provide fallback strategies if primary approach fails

Output format: Clear verification results and any needed corrections.
"""

    def calculate_confidence(self, current_time: float, content: str, stage: int,
                           task: Optional[LLMTask] = None) -> float:
        """
        Calculate confidence for SystemAgent outputs.

        SystemAgent has unique confidence ranges:
        - 0.4-0.75 range (moderate, not decision-making)
        - Higher confidence for verification tasks
        - Lower confidence for exploratory commands

        Args:
            current_time: Current simulation time
            content: Generated content
            stage: Processing stage
            task: Optional task context

        Returns:
            Confidence score (0.0 to 1.0)
        """
        position_info = self.get_position_info(current_time)
        depth_ratio = position_info.get("depth_ratio", 0.0)

        # SystemAgent confidence range: 0.4-0.75
        # Lower than synthesis but higher than pure research
        base_confidence = 0.4 + (depth_ratio * 0.35)  # 0.4-0.75 range
        max_confidence = 0.75

        # Content quality bonus
        content_quality = self._analyze_content_quality(content)
        content_bonus = content_quality * 0.1

        # System-specific quality indicators
        system_quality = self._analyze_system_content_quality(content)
        system_bonus = system_quality * 0.15

        total_confidence = base_confidence + content_bonus + system_bonus

        return min(max(total_confidence, 0.0), max_confidence)

    def _analyze_system_content_quality(self, content: str) -> float:
        """
        Analyze system content quality with system-specific heuristics.

        Args:
            content: Content to analyze

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not content:
            return 0.0

        content_lower = content.lower()
        quality_score = 0.0

        # Command presence (0.3 weight)
        command_indicators = [
            'cd ' in content_lower or 'ls ' in content_lower,
            'source ' in content_lower or 'activate' in content_lower,
            'pip ' in content_lower or 'python ' in content_lower,
            'test -' in content_lower or 'echo ' in content_lower,
        ]
        command_score = sum(command_indicators) / len(command_indicators)
        quality_score += command_score * 0.3

        # Safety indicators (0.3 weight)
        safety_indicators = [
            'test -f' in content_lower or 'test -d' in content_lower,
            '&&' in content,  # Command chaining for safety
            'check' in content_lower or 'verify' in content_lower,
            'venv' in content_lower or 'virtual' in content_lower,
        ]
        safety_score = sum(safety_indicators) / len(safety_indicators)
        quality_score += safety_score * 0.3

        # Explanation quality (0.2 weight)
        explanation_indicators = [
            'because' in content_lower or 'to' in content_lower,
            'first' in content_lower or 'then' in content_lower or 'step' in content_lower,
            'ensure' in content_lower or 'confirm' in content_lower,
        ]
        explanation_score = sum(explanation_indicators) / len(explanation_indicators)
        quality_score += explanation_score * 0.2

        # Precision indicators (0.2 weight)
        precision_indicators = [
            content.count('\n') >= 3,  # Multi-line structured output
            any(char == '`' for char in content),  # Code formatting
            'command:' in content_lower or 'execute:' in content_lower,
            len(content.split()) > 30,  # Substantial explanation
        ]
        precision_score = sum(precision_indicators) / len(precision_indicators)
        quality_score += precision_score * 0.2

        return min(quality_score, 1.0)

    def record_command_execution(self, command: str, action_id: str,
                                success: Optional[bool] = None) -> None:
        """
        Record command execution for tracking.

        Args:
            command: Command that was executed
            action_id: Action ID from request_action()
            success: Whether command succeeded (None if pending)
        """
        self.commands_executed.append({
            "command": command,
            "action_id": action_id,
            "success": success,
            "timestamp": self._progress
        })

        logger.debug(f"SystemAgent {self.agent_id} recorded command: {command[:50]}")

    def get_command_history(self) -> list:
        """
        Get history of commands executed by this agent.

        Returns:
            List of command execution records
        """
        return self.commands_executed.copy()

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get SystemAgent-specific statistics.

        Returns:
            Dictionary with agent statistics
        """
        stats = super().get_agent_stats()

        # Add SystemAgent-specific stats
        stats["agent_subtype"] = "system"
        stats["commands_executed_count"] = len(self.commands_executed)
        stats["successful_commands"] = sum(
            1 for cmd in self.commands_executed if cmd.get("success") is True
        )
        stats["failed_commands"] = sum(
            1 for cmd in self.commands_executed if cmd.get("success") is False
        )
        stats["venv_state"] = self.venv_state
        stats["last_error"] = self.last_error

        return stats


def create_system_agent(agent_id: str, spawn_time: float, helix: HelixGeometry,
                       llm_client, **kwargs) -> SystemAgent:
    """
    Factory function to create a SystemAgent.

    Args:
        agent_id: Unique agent identifier
        spawn_time: When agent becomes active (0.0-1.0)
        helix: Helix geometry
        llm_client: LLM client for inference
        **kwargs: Additional arguments for SystemAgent constructor

    Returns:
        Configured SystemAgent instance
    """
    return SystemAgent(
        agent_id=agent_id,
        spawn_time=spawn_time,
        helix=helix,
        llm_client=llm_client,
        **kwargs
    )
